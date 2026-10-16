#!/usr/bin/env python3
"""MLIP-4 MTP active learning loop for amorphous carbon.

Designed to run unattended inside a tmux session.  At each iteration:
    1. Train MTP on current dataset
    2. Run ASE MD with MTP across a temperature ladder
    3. Compute extrapolation grades
    4. Select top structures with gamma_save < gamma < gamma_break
    5. Run CP2K single-point on selected structures
    6. Append converged DFT results to dataset
    7. Retrain MTP and repeat

Usage:
    python run_al_loop.py --config al_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Pin OpenMP/MKL/OpenBLAS threads to 1.  MLIP-4 training is not threaded, and
# leaving these libraries unbound can cause memory spikes and oversubscription
# on the head node where the AL loop runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).parent))

from cp2k_utils import run_cp2k_single_point, run_cp2k_single_points_batch
from mlip4_utils import run_md_exploration, select_by_grade, train_mtp
from timing import TimingTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: Path) -> list[Atoms]:
    logger.info(f"Loading dataset from {dataset_path}")
    atoms = list(read(str(dataset_path), index=":"))
    logger.info(f"  Loaded {len(atoms)} structures")
    return atoms


def save_dataset(atoms: list[Atoms], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms)


def main():
    parser = argparse.ArgumentParser(description="MLIP-4 AL loop for a-C")
    parser.add_argument("--config", type=str, default="al_config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--start-iter", type=int, default=1,
                        help="AL iteration number to resume from (1 = first AL iteration after seed)")
    parser.add_argument("--train-cores", type=int, default=1,
                        help="CPU cores for MTP training")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    work_dir = Path(config["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path(config["seed_dataset"])
    dataset = load_dataset(seed_path)

    al_cfg = config["active_learning"]
    md_cfg = config["md"]
    mtp_cfg = config["mtp"]
    cp2k_cfg = config["cp2k"]

    # Iteration directories
    al_loop_dir = work_dir / "02_al_loop"
    al_loop_dir.mkdir(parents=True, exist_ok=True)

    # Timing
    timing = TimingTracker(output_file=al_loop_dir / "timing_report.json")

    # Initial structure for MD exploration
    initial_structure = al_cfg.get("initial_structure")
    if initial_structure:
        md_init = read(initial_structure)
    else:
        md_init = dataset[0].copy()

    # Ensure PBC and cell
    if md_init.get_pbc().sum() == 0:
        md_init.set_pbc(True)
    if np.allclose(md_init.get_cell(), 0):
        # Fallback cell from seed dataset first frame
        md_init.set_cell(dataset[0].get_cell())

    max_al_iter = al_cfg["max_iterations"]
    gamma_threshold = al_cfg["gamma_threshold"]
    gamma_break = al_cfg["gamma_break"]
    max_select = al_cfg["max_select_per_iter"]

    # -------------------------------------------------------------------------
    # Optional: load existing state if resuming
    # -------------------------------------------------------------------------
    current_mtp_path: Path | None = None
    if args.start_iter >= 1:
        prev_iter = args.start_iter - 1
        prev_iter_dir = al_loop_dir / f"iter_{prev_iter:03d}"
        prev_mtp = prev_iter_dir / "model" / f"mtp_iter{prev_iter:03d}.json"
        prev_dataset = prev_iter_dir / "dataset" / f"dataset_iter{prev_iter:03d}.extxyz"
        if prev_mtp.exists() and prev_dataset.exists():
            logger.info(f"Resuming from AL iteration {args.start_iter}")
            current_mtp_path = prev_mtp
            dataset = load_dataset(prev_dataset)
        else:
            logger.warning("Resume files not found; starting from seed")
            args.start_iter = 1

    # -------------------------------------------------------------------------
    # Initial MTP training on seed dataset
    # -------------------------------------------------------------------------
    if current_mtp_path is None:
        iter_dir = al_loop_dir / "iter_000"
        current_mtp_path = iter_dir / "model" / "mtp_iter000.json"
        current_mtp_path = train_mtp(
            dataset,
            mtp_cfg,
            output_path=current_mtp_path,
            timing=timing,
            train_cores=args.train_cores,
            iteration=0,
        )
        save_dataset(dataset, iter_dir / "dataset" / "dataset_iter000.extxyz")

    # -------------------------------------------------------------------------
    # Active learning loop
    # -------------------------------------------------------------------------
    converged = False
    for al_iter in range(args.start_iter, max_al_iter + 1):
        logger.info("=" * 70)
        logger.info(f"ACTIVE LEARNING ITERATION {al_iter} / {max_al_iter}")
        logger.info("=" * 70)

        iter_dir = al_loop_dir / f"iter_{al_iter:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        md_dir = iter_dir / "md"
        dft_dir = iter_dir / "dft"
        model_dir = iter_dir / "model"
        dataset_dir = iter_dir / "dataset"

        # Step 1: MD exploration with current MTP
        timing.start("md_exploration", n_cores=1, iteration=al_iter)
        md_frames, md_grades = run_md_exploration(
            initial_atoms=md_init,
            mtp_path=current_mtp_path,
            md_config=md_cfg,
            output_dir=md_dir,
            iteration=al_iter,
        )
        timing.stop("md_exploration", iteration=al_iter,
                    n_frames=len(md_frames))

        # Step 2: Select candidates by extrapolation grade
        timing.start("selection", n_cores=1, iteration=al_iter)
        selected_frames, selected_grades, converged = select_by_grade(
            md_frames,
            md_grades,
            gamma_threshold=gamma_threshold,
            gamma_break=gamma_break,
            max_select=max_select,
        )
        timing.stop("selection", iteration=al_iter,
                    n_selected=len(selected_frames))

        if converged:
            logger.info(f"AL converged at iteration {al_iter}")
            break

        # Save selected structures before DFT
        write(str(iter_dir / "selected_structures.xyz"), selected_frames)
        with open(iter_dir / "selected_grades.json", "w") as f:
            json.dump({
                "iteration": al_iter,
                "n_selected": len(selected_frames),
                "grades": selected_grades,
            }, f, indent=2)

        # Step 3: CP2K single-point on selected structures
        # Submit in parallel batches to keep the queue busy without flooding it.
        labels = [f"iter{al_iter:03d}_cfg{i:04d}" for i in range(len(selected_frames))]
        max_parallel = cp2k_cfg.get("max_parallel_jobs", 4)
        dft_results = run_cp2k_single_points_batch(
            selected_frames,
            cp2k_cfg,
            work_dir=dft_dir,
            labels=labels,
            timing=timing,
            iteration=al_iter,
            max_parallel=max_parallel,
        )

        new_structures: list[Atoms] = []
        failed_labels: list[str] = []
        for i, (frame, result) in enumerate(zip(selected_frames, dft_results)):
            label = labels[i]
            if result["converged"] and result["energy"] is not None and result["forces"] is not None:
                atoms = frame.copy()
                atoms.set_cell(md_init.get_cell())
                atoms.set_pbc(True)
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=result["energy"],
                    forces=result["forces"],
                )
                atoms.info["iteration"] = al_iter
                atoms.info["selection_index"] = i
                atoms.info["gamma"] = selected_grades[i]
                atoms.info["source"] = "mlip4_al"
                atoms.info["dft_label"] = label
                new_structures.append(atoms)
            else:
                failed_labels.append(label)

        if failed_labels:
            logger.warning(f"CP2K failed for {len(failed_labels)} structures: {failed_labels}")

        if not new_structures:
            logger.warning("No converged DFT structures this iteration; skipping retrain")
            continue

        # Unit audit gate: parse_cp2k_energy/parse_cp2k_forces convert to
        # eV / eV/Angstrom. Raw CP2K totals for carbon are ~-154 eV/atom, so
        # values of order a few eV/atom mean the conversion was lost — refuse
        # to contaminate the training dataset in that case.
        e_per_atom = np.abs([a.get_potential_energy() / len(a) for a in new_structures])
        median_e = float(np.median(e_per_atom))
        if median_e < 20.0:
            raise RuntimeError(
                f"Unit audit failed at iteration {al_iter}: median |E/atom| = "
                f"{median_e:.3f} looks like Hartree, not eV. Dataset not updated."
            )

        # Step 4: Append to dataset and retrain
        dataset.extend(new_structures)
        prev_dataset_path = dataset_dir / f"dataset_iter{al_iter - 1:03d}.extxyz"
        save_dataset(dataset, prev_dataset_path)

        # Save the current dataset *before* training as well, so a training
        # crash (e.g. OOM) does not force us to re-run the expensive DFT step.
        current_dataset_path_pretrain = dataset_dir / f"dataset_iter{al_iter:03d}.extxyz"
        save_dataset(dataset, current_dataset_path_pretrain)

        current_mtp_path = model_dir / f"mtp_iter{al_iter:03d}.json"
        current_mtp_path = train_mtp(
            dataset,
            mtp_cfg,
            output_path=current_mtp_path,
            initial_mtp_path=None,  # retrain from scratch each iteration is more stable
            timing=timing,
            train_cores=args.train_cores,
            iteration=al_iter,
        )

        current_dataset_path = dataset_dir / f"dataset_iter{al_iter:03d}.extxyz"
        save_dataset(dataset, current_dataset_path)

        # Save state summary
        state = {
            "iteration": al_iter,
            "dataset_size": len(dataset),
            "n_new_structures": len(new_structures),
            "n_failed": len(failed_labels),
            "model_path": str(current_mtp_path),
            "dataset_path": str(current_dataset_path),
        }
        with open(iter_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        timing.save()
        logger.info(f"Iteration {al_iter} complete. Dataset size: {len(dataset)}")

    # -------------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------------
    timing.save()
    final_dataset_path = work_dir / "03_final_dataset" / "mlip4_al_dataset.extxyz"
    save_dataset(dataset, final_dataset_path)

    summary = {
        "total_iterations": max_al_iter if not converged else al_iter,
        "converged": converged,
        "final_dataset_size": len(dataset),
        "final_dataset_path": str(final_dataset_path),
        "timing_report": str(timing.output_file),
    }
    with open(work_dir / "03_final_dataset" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("ACTIVE LEARNING LOOP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final dataset: {final_dataset_path} ({len(dataset)} structures)")
    logger.info(f"Timing report: {timing.output_file}")
    logger.info(f"Total core-hours: {timing.total_core_hours():.4f}")


if __name__ == "__main__":
    main()
