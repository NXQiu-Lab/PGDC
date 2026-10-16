#!/usr/bin/env python3
"""Recover MLIP-4 AL iteration 010 from existing CP2K outputs.

The AL loop was killed by SIGKILL during the iter_010 MTP retraining.  This
script re-constructs the iter_010 dataset from the already-converged CP2K
single-point outputs, trains the missing mtp_iter010.json, and writes the
state.json so that run_al_loop.py can resume from iter_011.
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

# Pin OpenMP/MKL/OpenBLAS threads to 1 to avoid memory spikes on the head node.
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

from cp2k_utils import parse_cp2k_energy, parse_cp2k_forces, cp2k_finished_normally
from mlip4_utils import train_mtp
from timing import TimingTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_dft_result(dft_dir: Path, label: str, frame: Atoms) -> dict[str, Any] | None:
    """Parse a single CP2K output directory."""
    out_path = dft_dir / label / f"{label}.out"
    if not out_path.exists():
        logger.error(f"Missing output file: {out_path}")
        return None

    out_text = out_path.read_text(encoding="utf-8")
    converged = cp2k_finished_normally(out_text)
    if not converged:
        logger.warning(f"{label}: CP2K did not finish normally")
        return None

    energy = parse_cp2k_energy(out_text)
    forces = parse_cp2k_forces(out_text, len(frame))
    if energy is None or forces is None:
        logger.warning(f"{label}: failed to parse energy/forces")
        return None

    return {"energy": energy, "forces": forces}


def main():
    parser = argparse.ArgumentParser(description="Recover MLIP-4 AL iter_010")
    parser.add_argument("--config", type=str, default="al_config.yaml")
    parser.add_argument("--train-cores", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse DFT outputs but do not train or write datasets")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    work_dir = Path(config["work_dir"])
    al_loop_dir = work_dir / "02_al_loop"
    iter_dir = al_loop_dir / "iter_010"

    if not iter_dir.exists():
        raise RuntimeError(f"iter_010 directory not found: {iter_dir}")

    dft_dir = iter_dir / "dft"
    selected_path = iter_dir / "selected_structures.xyz"
    grades_path = iter_dir / "selected_grades.json"
    dataset_dir = iter_dir / "dataset"
    model_dir = iter_dir / "model"

    if not selected_path.exists() or not grades_path.exists():
        raise RuntimeError("Missing selected_structures.xyz or selected_grades.json")

    # Load previous dataset (iter_009)
    prev_dataset_path = al_loop_dir / "iter_009" / "dataset" / "dataset_iter009.extxyz"
    logger.info(f"Loading previous dataset: {prev_dataset_path}")
    dataset = list(read(str(prev_dataset_path), index=":"))
    logger.info(f"  Loaded {len(dataset)} structures")

    # Load selected structures and grades
    logger.info(f"Loading selected structures: {selected_path}")
    selected_frames = list(read(str(selected_path), index=":"))
    with open(grades_path, "r") as f:
        grades_data = json.load(f)
    selected_grades = grades_data["grades"]
    assert len(selected_frames) == len(selected_grades), "Mismatch between frames and grades"
    logger.info(f"  Loaded {len(selected_frames)} selected structures")

    # Parse DFT outputs
    logger.info("Parsing CP2K single-point outputs ...")
    labels = [f"iter010_cfg{i:04d}" for i in range(len(selected_frames))]
    new_structures: list[Atoms] = []
    failed_labels: list[str] = []

    for i, (frame, grade, label) in enumerate(zip(selected_frames, selected_grades, labels)):
        result = parse_dft_result(dft_dir, label, frame)
        if result is None:
            failed_labels.append(label)
            continue

        atoms = frame.copy()
        atoms.set_cell(frame.get_cell())
        atoms.set_pbc(True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=result["energy"],
            forces=result["forces"],
        )
        atoms.info["iteration"] = 10
        atoms.info["selection_index"] = i
        atoms.info["gamma"] = grade
        atoms.info["source"] = "mlip4_al"
        atoms.info["dft_label"] = label
        new_structures.append(atoms)

    logger.info(f"Parsed {len(new_structures)} / {len(selected_frames)} DFT results")
    if failed_labels:
        logger.warning(f"Failed labels ({len(failed_labels)}): {failed_labels[:10]}...")

    if not new_structures:
        raise RuntimeError("No converged DFT structures found; cannot recover")

    if args.dry_run:
        logger.info("Dry run complete; no files written")
        return

    # Append and save datasets
    dataset.extend(new_structures)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    prev_path = dataset_dir / "dataset_iter009.extxyz"
    current_path = dataset_dir / "dataset_iter010.extxyz"

    logger.info(f"Saving dataset_iter009.extxyz ({len(dataset)} structures)")
    write(str(prev_path), dataset)
    logger.info(f"Saving dataset_iter010.extxyz ({len(dataset)} structures)")
    write(str(current_path), dataset)

    # Train iter_010 MTP
    timing = TimingTracker(output_file=al_loop_dir / "timing_report.json")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mtp_iter010.json"

    logger.info("Training mtp_iter010.json ...")
    train_mtp(
        dataset,
        config["mtp"],
        output_path=model_path,
        initial_mtp_path=None,
        timing=timing,
        train_cores=args.train_cores,
        iteration=10,
    )

    # Save state.json
    state = {
        "iteration": 10,
        "dataset_size": len(dataset),
        "n_new_structures": len(new_structures),
        "n_failed": len(failed_labels),
        "model_path": str(model_path),
        "dataset_path": str(current_path),
    }
    with open(iter_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    timing.save()
    logger.info("=" * 70)
    logger.info("iter_010 recovery complete")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  State: {iter_dir / 'state.json'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
