#!/usr/bin/env python3
"""Amorphous carbon dataset builder with profile-guided active guidance.

Remote SLURM edition. The workflow is asynchronous and resumable:

  seed-submit  →  (disconnect)  →  seed-collect
  al-submit    →  (disconnect)  →  al-collect

A local JSON manifest (remote_manifest.json) persists job state.

# ============================================================================
# Complete Workflow Commands
# ============================================================================

# 1. Seed phase — generate initial amorphous carbon seeds via melt-quench
uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --remote-config scripts/amorphous_carbon/remote_config.json \
    --n-atoms 64 \
    --seed-density 2.0 \
    --seed-temps "3000" \
    --seed-runs-per-temp 4 \
    seed-submit

# ... wait for jobs to finish ...

uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --remote-config scripts/amorphous_carbon/remote_config.json \
    seed-collect

# 2. AL loop — iterative active learning (repeat submit/collect until convergence)
uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --remote-config scripts/amorphous_carbon/remote_config.json \
    --al-runs-per-iter 10 \
    --al-induction-mode annealing \
    al-submit

uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --remote-config scripts/amorphous_carbon/remote_config.json \
    al-collect

# 3. Auto mode — blocking loop that submits and collects automatically
uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --remote-config scripts/amorphous_carbon/remote_config.json \
    --al-induction-mode npt \
    --event-driven \
    auto

# 3b. Ballistic annealing mode — high-energy initial velocities for boundary expansion
uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --remote-config scripts/amorphous_carbon/remote_config.json \
    --al-induction-mode ballistic \
    --al-runs-per-iter 10 \
    al-submit

# 4. Status check
uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py status \
    --build-dir amorphous_carbon_al_build

# 5. Build final 10k dataset with density-adaptive k-center sampling
uv run python scripts/amorphous_carbon/build_amorphous_carbon_dataset_remote.py \
    --build-dir amorphous_carbon_al_build \
    build-final-dataset \
    --n-structures 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure project src and scripts dir are on path
_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "amorphous_carbon"))
sys.path.insert(0, str(_scripts_dir.parent / "src"))

from ase import Atoms
from ase.io import read, write

from amorphous_carbon_utils import (
    build_cp2k_input_carbon,
    check_carbon_candidate,
    density_adaptive_kcenter_sample,
    generate_amorphous_carbon_seed,
    generate_liquid_carbon_seed,
    get_density,
    get_sp3_fraction,
    perturb_amorphous_carbon,
    set_density,
    summarize_carbon_structure,
)
from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.profiling import DatasetProfiler
from atom_dataset_index.sampling import (
    CandidateGenerator,
    DeterministicPreFilter,
    ase_to_record,
    record_to_ase,
)
from atom_dataset_index.sampling.trajectory_miner import mine_cp2k_trajectory
from common.cp2k_utils import (
    get_db_stats,
    load_al_manifest,
    load_json,
    merge_trajectory_to_db,
    parse_cp2k_cell,
    parse_cp2k_md_cells,
    parse_cp2k_ener,
    save_al_manifest,
    save_json,
)
from common.remote_slurm_utils import RemoteConfig, RemoteSLURMClient, load_remote_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BuilderConfig:
    """Top-level configuration for the amorphous carbon dataset builder."""

    # Build directory and database
    build_dir: str = "amorphous_carbon_al_build"
    db_name: str = "dataset.db"

    # System
    n_atoms: int = 64
    seed_density_g_cm3: float = 2.0
    density_min: float = 1.2
    density_max: float = 3.5

    # Seed phase
    seed_temps: tuple[float, ...] = (3000.0,)
    seed_runs_per_temp: int = 4
    seed_steps: int = 20000  # 20 ps @ 1 fs
    seed_quench_rate_k_per_ps: float = 250.0  # 3000K -> 300K in ~11 ps

    # AL loop
    al_iterations: int = 4
    al_runs_per_iter: int = 10
    al_induction_mode: str = "annealing"  # annealing | npt | strain | ballistic | md
    # Candidate structure generation is independent from the CP2K induction mode.
    candidate_strategy: str = "perturbation"
    candidate_strategy_params: dict[str, Any] = field(default_factory=dict)
    al_steps: int = 10000  # 10 ps production AIMD
    al_prescreen_steps: int = 1000  # 1 ps short AIMD pre-screening
    al_candidate_pool_size: int = 50
    al_temperature: float = 300.0
    al_prescreen_steps: int = 1000
    event_driven_sampling: bool = False
    skip_prescreen: bool = False

    # NPT induction mode grid
    npt_pressures_gpa: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 60.0)
    npt_temperatures: tuple[float, ...] = (300.0, 1000.0, 2000.0)

    # Annealing induction mode
    annealing_temps_high: tuple[float, ...] = (2000.0, 2500.0, 3000.0)
    annealing_temps_low: tuple[float, ...] = (300.0,)

    # Strain induction mode
    strain_axes: tuple[str, ...] = ("x", "y", "z")
    strain_values: tuple[float, ...] = (-0.10, -0.05, 0.05, 0.10, 0.15)

    # Ballistic induction mode
    ballistic_velocity_temps: tuple[float, ...] = (1500.0, 2000.0, 2500.0)

    # CP2K parameters
    cp2k_cutoff: int = 500
    cp2k_rel_cutoff: int = 50
    cp2k_basis_set: str = "DZVP-MOLOPT-SR-GTH"
    cp2k_xc_functional: str = "PBE"
    cp2k_timestep_fs: float = 1.0
    cp2k_output_stride: int = 40

    # Remote
    remote_config_path: str = "scripts/amorphous_carbon/remote_config.json"
    remote_ppn: int = 64
    remote_nprocs: int = 64
    remote_queue: str = "compute"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuilderConfig":
        # Filter unknown keys to avoid dataclass errors
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# 1. Structure generation and candidate screening
# ---------------------------------------------------------------------------


def generate_seed_structures(config: BuilderConfig) -> list[Atoms]:
    """Generate initial amorphous carbon seed structures for melt-quench.

    We generate both ordered-seed perturbations and random liquid-like seeds
    to increase the chance of successful melt-quench trajectories.
    """
    seeds: list[Atoms] = []
    for temp in config.seed_temps:
        for run in range(config.seed_runs_per_temp):
            # Alternate between ordered seed and random liquid seed
            if run % 2 == 0:
                atoms = generate_amorphous_carbon_seed(
                    n_atoms=config.n_atoms,
                    target_density_g_cm3=config.seed_density_g_cm3,
                    temperature=temp,
                    seed=hash((temp, run)) % 2**32,
                )
            else:
                atoms = generate_liquid_carbon_seed(
                    n_atoms=config.n_atoms,
                    target_density_g_cm3=config.seed_density_g_cm3,
                    seed=hash((temp, run)) % 2**32,
                )
            passed, reason = check_carbon_candidate(
                atoms,
                density_min=config.density_min,
                density_max=config.density_max,
            )
            if passed:
                atoms.info["seed_temp"] = float(temp)
                atoms.info["seed_run"] = run
                atoms.info["phase"] = "seed"
                seeds.append(atoms)
            else:
                logger.warning(f"Seed candidate rejected: {reason}")
    return seeds


def generate_candidates_for_iteration(
    config: BuilderConfig,
    iteration: int,
) -> list[Atoms]:
    """Generate candidate structures for an AL iteration.

    Candidates are created by perturbing existing structures from the dataset
    rather than from scratch. This keeps the pass rate high while still
    exploring new regions of the density-sp³ phase space.
    """
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name

    candidates: list[Atoms] = []
    if not db_path.exists():
        logger.warning("No dataset found; falling back to seed structures")
        return generate_seed_structures(config)

    # Load existing dataset
    dataset = AseDbDatasetReader(str(db_path)).read()
    if len(dataset.records) == 0:
        logger.warning("Empty dataset; falling back to seed structures")
        return generate_seed_structures(config)

    # Candidate generation is independent from the CP2K induction mode.
    # The carbon-specific perturbation generator remains the default fallback.
    if config.candidate_strategy in ("", "perturbation", "random"):
        candidates = _generate_candidates_by_perturbation(config, dataset)
    else:
        try:
            generator = CandidateGenerator(dataset=dataset)
            params = dict(config.candidate_strategy_params)
            if config.candidate_strategy in ("boundary", "composite"):
                params.setdefault(
                    "temperature_k",
                    sum(config.npt_temperatures) / len(config.npt_temperatures),
                )
            records = generator.generate(
                strategy=config.candidate_strategy,
                n_samples=config.al_candidate_pool_size,
                **params,
            )
            candidates = [record_to_ase(r) for r in records]
        except Exception as exc:
            logger.warning(
                f"CandidateGenerator strategy '{config.candidate_strategy}' failed: "
                f"{exc}. Using perturbation fallback."
            )
            candidates = _generate_candidates_by_perturbation(config, dataset)

    # Pre-filter: use data-driven classification (EQ/MS/TP/TD/LU/AB) when a
    # reference dataset exists.  This replaces the simple geometric check with
    # the project's DeterministicPreFilter, which uses density/bond/CN envelopes
    # and a harmonic force estimate (when forces are unavailable) to decide which
    # candidates are AIMD-ready.  LU/AB are rejected; EQ/MS/TP/TD are kept.
    # If the prefilter rejects every candidate (common for very aggressive modes
    # like ballistic), we fall back to the geometric check so the iteration can
    # still proceed.
    pf: DeterministicPreFilter | None = None
    if db_path.exists():
        try:
            pf = DeterministicPreFilter(dataset)
            logger.info(f"Built DeterministicPreFilter from {len(dataset.records)} reference structures")
        except Exception as exc:
            logger.warning(f"Failed to build DeterministicPreFilter: {exc}. Using geometric fallback.")

    passed: list[Atoms] = []
    class_counts: dict[str, int] = {}
    for atoms in candidates:
        if pf is not None:
            try:
                record = ase_to_record(atoms, sample_id=0)
                result = pf.classify(record)
                class_counts[result.label] = class_counts.get(result.label, 0) + 1
                if result.passed:
                    atoms.info["prefilter_class"] = result.label
                    passed.append(atoms)
                    continue
                logger.debug(f"Candidate rejected by DeterministicPreFilter: {result.reason}")
            except Exception as exc:
                logger.debug(f"DeterministicPreFilter classification failed: {exc}. Falling back to geometric check.")

        ok, reason = check_carbon_candidate(
            atoms,
            density_min=config.density_min,
            density_max=config.density_max,
        )
        if ok:
            passed.append(atoms)
        else:
            logger.debug(f"Candidate rejected by geometry check: {reason}")

    logger.info(
        f"Iteration {iteration}: {len(passed)}/{len(candidates)} candidates passed screening"
    )
    if class_counts:
        logger.info(f"  PreFilter class counts: {class_counts}")
    return passed


def _reconstruct_cell(atoms: Atoms, config: BuilderConfig) -> Atoms:
    """Reconstruct a cubic cell for atoms without periodic boundary info.

    Uses the target density to compute the required volume for a cubic cell,
    then centers the atoms within it.
    """
    from ase.geometry import cellpar_to_cell

    n_atoms = len(atoms)
    mass_amu = sum(atoms.get_masses())
    target_density = (config.density_min + config.density_max) / 2.0
    # g/cm³ -> amu/Å³
    target_density_amu_per_a3 = target_density / 1.66053906660
    volume = mass_amu / target_density_amu_per_a3
    side = volume ** (1.0 / 3.0)
    cell = cellpar_to_cell([side, side, side, 90.0, 90.0, 90.0])
    atoms.set_cell(cell, scale_atoms=False)
    # Center atoms in the cell
    pos = atoms.get_positions()
    pos -= pos.mean(axis=0)
    pos += side / 2.0
    atoms.set_positions(pos)
    atoms.set_pbc(True)
    return atoms


def _generate_candidates_by_perturbation(
    config: BuilderConfig,
    dataset: Any,
) -> list[Atoms]:
    """Fallback candidate generation: perturb random frames from the dataset."""
    rng = random.Random(config.al_iterations)
    candidates: list[Atoms] = []
    records = list(dataset.records)

    for i in range(config.al_candidate_pool_size):
        record = rng.choice(records)
        atoms = record_to_ase(record)

        if atoms.cell is None or atoms.cell.rank != 3:
            atoms = _reconstruct_cell(atoms, config)

        # Strategy-aware perturbation
        if config.al_induction_mode == "npt":
            # Vary density for pressure-driven exploration
            target_density = rng.uniform(config.density_min, config.density_max)
            atoms = set_density(atoms, target_density)
            strain_range = None
        elif config.al_induction_mode == "strain":
            # Larger strain perturbations
            axis = rng.choice(config.strain_axes)
            sx = sy = sz = 0.0
            strain_val = rng.choice(config.strain_values)
            if axis == "x":
                sx = strain_val
            elif axis == "y":
                sy = strain_val
            else:
                sz = strain_val
            strain_range = (sx, sy, sz)
        elif config.al_induction_mode == "ballistic":
            # Ballistic annealing: symmetry-breaking perturbations to create
            # high-energy initial states.  Positional/strain perturbations are
            # kept moderate to avoid immediate geometric rejection; the bulk of
            # the high-energy character comes from the high-temperature initial
            # velocities injected by CP2K in the BALLISTIC motion block.
            axis = rng.choice(config.strain_axes)
            sx = sy = sz = 0.0
            strain_val = rng.choice([-0.08, -0.05, 0.05, 0.08])
            if axis == "x":
                sx = strain_val
            elif axis == "y":
                sy = strain_val
            else:
                sz = strain_val
            strain_range = (sx, sy, sz)

        if config.al_induction_mode == "ballistic":
            # Moderate positional disorder to avoid C-C overlaps while still
            # breaking symmetry (C-C covalent radius ~0.76 Å).
            position_std = rng.uniform(0.05, 0.15)
        else:
            position_std = rng.uniform(0.08, 0.18)

        perturbed = perturb_amorphous_carbon(
            atoms,
            position_std=position_std,
            strain_range=strain_range,
            seed=hash((i, config.al_induction_mode)) % 2**32,
        )
        perturbed.info["parent_id"] = getattr(record, "id", i)
        perturbed.info["candidate_source"] = f"perturbation_{config.al_induction_mode}"
        candidates.append(perturbed)

    return candidates


# ---------------------------------------------------------------------------
# 2. CP2K input generation per induction mode
# ---------------------------------------------------------------------------


def write_cp2k_inputs_for_seed(
    run_dir: Path,
    atoms: Atoms,
    config: BuilderConfig,
    high_temp: float,
) -> dict[str, Path]:
    """Write CP2K inputs for a 4-stage melt-quench seed run.

    Stages are chained via EXT_RESTART / WFN_RESTART so that each stage
    continues from the final geometry/wavefunction of the previous stage.

    Returns dict mapping phase name to input file path.
    """
    inputs: dict[str, Path] = {}
    project = f"seed_T{int(high_temp)}"
    common_kwargs = {
        "timestep": config.cp2k_timestep_fs,
        "output_stride": config.cp2k_output_stride,
        "cutoff": config.cp2k_cutoff,
        "rel_cutoff": config.cp2k_rel_cutoff,
        "basis_set": config.cp2k_basis_set,
        "xc_functional": config.cp2k_xc_functional,
    }

    # Stage 1: Heat / melt (from initial structure)
    inp_heat = build_cp2k_input_carbon(
        atoms,
        project_name=f"{project}_heat",
        run_type="ANNEALING",
        temperature=300.0,
        n_steps=5000,
        annealing_start_temp=300.0,
        annealing_stop_temp=high_temp,
        scf_guess="ATOMIC",
        **common_kwargs,
    )
    inputs["heat"] = run_dir / "heat.inp"
    inputs["heat"].write_text(inp_heat)

    # Stage 2: Hold at high T (continue from heat)
    inp_hold = build_cp2k_input_carbon(
        atoms,
        project_name=f"{project}_hold",
        run_type="MD",
        temperature=high_temp,
        n_steps=5000,
        scf_guess="RESTART",
        wfn_restart_file=f"{project}_heat-RESTART.wfn",
        ext_restart_file=f"{project}_heat-1.restart",
        **common_kwargs,
    )
    inputs["hold"] = run_dir / "hold.inp"
    inputs["hold"].write_text(inp_hold)

    # Stage 3: Quench (continue from hold)
    inp_quench = build_cp2k_input_carbon(
        atoms,
        project_name=f"{project}_quench",
        run_type="ANNEALING",
        temperature=high_temp,
        n_steps=10000,
        annealing_start_temp=high_temp,
        annealing_stop_temp=300.0,
        scf_guess="RESTART",
        wfn_restart_file=f"{project}_hold-RESTART.wfn",
        ext_restart_file=f"{project}_hold-1.restart",
        **common_kwargs,
    )
    inputs["quench"] = run_dir / "quench.inp"
    inputs["quench"].write_text(inp_quench)

    # Stage 4: Relax at 300K (continue from quench)
    inp_relax = build_cp2k_input_carbon(
        atoms,
        project_name=f"{project}_relax",
        run_type="MD",
        temperature=300.0,
        n_steps=5000,
        scf_guess="RESTART",
        wfn_restart_file=f"{project}_quench-RESTART.wfn",
        ext_restart_file=f"{project}_quench-1.restart",
        **common_kwargs,
    )
    inputs["relax"] = run_dir / "relax.inp"
    inputs["relax"].write_text(inp_relax)

    return inputs


def write_cp2k_input_for_induction(
    run_dir: Path,
    atoms: Atoms,
    config: BuilderConfig,
    induction_mode: str,
    run_id: str,
    **kwargs: Any,
) -> Path:
    """Write a single CP2K input for an AL induction run."""
    input_path = run_dir / "input.inp"

    if induction_mode == "annealing":
        t_high = kwargs.get("t_high", config.annealing_temps_high[0])
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=f"al_{run_id}_anneal",
            run_type="ANNEALING",
            temperature=t_high,
            n_steps=config.al_steps,
            annealing_start_temp=t_high,
            annealing_stop_temp=300.0,
            timestep=config.cp2k_timestep_fs,
            output_stride=config.cp2k_output_stride,
            cutoff=config.cp2k_cutoff,
            rel_cutoff=config.cp2k_rel_cutoff,
            basis_set=config.cp2k_basis_set,
            xc_functional=config.cp2k_xc_functional,
        )
    elif induction_mode == "npt":
        pressure = kwargs.get("pressure", 0.0)
        temperature = kwargs.get("temperature", config.al_temperature)
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=f"al_{run_id}_npt",
            run_type="NPT",
            ensemble="NPT_I",
            temperature=temperature,
            pressure_gpa=pressure,
            n_steps=config.al_steps,
            timestep=config.cp2k_timestep_fs,
            output_stride=config.cp2k_output_stride,
            cutoff=config.cp2k_cutoff,
            rel_cutoff=config.cp2k_rel_cutoff,
            basis_set=config.cp2k_basis_set,
            xc_functional=config.cp2k_xc_functional,
        )
    elif induction_mode == "strain":
        # Strain is encoded in the initial structure; run standard NVT
        temperature = kwargs.get("temperature", config.al_temperature)
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=f"al_{run_id}_strain",
            run_type="MD",
            temperature=temperature,
            n_steps=config.al_steps,
            timestep=config.cp2k_timestep_fs,
            output_stride=config.cp2k_output_stride,
            cutoff=config.cp2k_cutoff,
            rel_cutoff=config.cp2k_rel_cutoff,
            basis_set=config.cp2k_basis_set,
            xc_functional=config.cp2k_xc_functional,
        )
    elif induction_mode == "ballistic":
        v_temp = kwargs.get("v_temp", config.ballistic_velocity_temps[0])
        ensemble = kwargs.get("ensemble", "NVE")
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=f"al_{run_id}_ballistic",
            run_type="BALLISTIC",
            temperature=config.al_temperature,
            ballistic_velocity_temp=v_temp,
            ballistic_ensemble=ensemble,
            n_steps=config.al_steps,
            timestep=config.cp2k_timestep_fs,
            output_stride=config.cp2k_output_stride,
            cutoff=config.cp2k_cutoff,
            rel_cutoff=config.cp2k_rel_cutoff,
            basis_set=config.cp2k_basis_set,
            xc_functional=config.cp2k_xc_functional,
        )
    else:
        # Default NVT MD
        temperature = kwargs.get("temperature", config.al_temperature)
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=f"al_{run_id}_md",
            run_type="MD",
            temperature=temperature,
            n_steps=config.al_steps,
            timestep=config.cp2k_timestep_fs,
            output_stride=config.cp2k_output_stride,
            cutoff=config.cp2k_cutoff,
            rel_cutoff=config.cp2k_rel_cutoff,
            basis_set=config.cp2k_basis_set,
            xc_functional=config.cp2k_xc_functional,
        )

    input_path.write_text(inp)
    return input_path


# ---------------------------------------------------------------------------
# 3. Short AIMD pre-screening (1 ps)
# ---------------------------------------------------------------------------


def _write_prescreen_input(
    run_dir: Path,
    atoms: Atoms,
    config: BuilderConfig,
    temperature: float,
) -> Path:
    """Write a 1 ps NVT pre-screen input for a candidate."""
    input_path = run_dir / "prescreen.inp"
    inp = build_cp2k_input_carbon(
        atoms,
        project_name="prescreen",
        run_type="MD",
        temperature=temperature,
        n_steps=config.al_prescreen_steps,
        timestep=config.cp2k_timestep_fs,
        output_stride=config.cp2k_output_stride,
        cutoff=config.cp2k_cutoff,
        rel_cutoff=config.cp2k_rel_cutoff,
        basis_set=config.cp2k_basis_set,
        xc_functional=config.cp2k_xc_functional,
    )
    input_path.write_text(inp)
    return input_path


def _check_prescreen_stability(
    run_dir: Path,
    energy_drift_threshold_per_step: float = 0.05,
) -> tuple[bool, str]:
    """Check whether a 1 ps pre-screen trajectory is stable enough for production.

    Stability criteria:
      - CP2K output exists and does not contain SCF convergence errors
      - .ener file exists and potential energy does not drift too much
      - trajectory file exists and has multiple frames

    Returns:
        (is_stable, reason) tuple.
    """
    out_file = run_dir / "out"
    ener_file = run_dir / "prescreen-1.ener"
    if not ener_file.exists():
        ener_file = run_dir / "prescreen.ener"
    pos_file = run_dir / "prescreen-pos-1.xyz"

    if not out_file.exists():
        return False, "Missing CP2K output"

    out_text = out_file.read_text()
    if "SCF run NOT converged" in out_text or "ABORT" in out_text:
        return False, "SCF not converged"

    if not ener_file.exists():
        return False, "Missing energy file"

    energy_records = parse_cp2k_ener(ener_file)
    if len(energy_records) < 3:
        return False, "Too few energy records"

    energies = [r["potential_energy"] for r in energy_records]
    e_range = max(energies) - min(energies)
    n_steps = len(energy_records)
    if e_range > energy_drift_threshold_per_step * n_steps:
        return False, f"Energy drift too large: {e_range:.2f} eV over {n_steps} steps"

    if pos_file.exists():
        try:
            frames = read(str(pos_file), index=":")
            if len(frames) < 2:
                return False, "Too few trajectory frames"
        except Exception as exc:
            return False, f"Failed to read trajectory: {exc}"
    else:
        return False, "Missing trajectory file"

    return True, "Pre-screen trajectory stable"


def _submit_prescreen_jobs(
    config: BuilderConfig,
    client: RemoteSLURMClient,
    candidates: list[Atoms],
    iteration: int,
) -> list[dict[str, Any]]:
    """Submit 1 ps pre-screen AIMD jobs for all candidates. Return job metadata."""
    build_dir = Path(config.build_dir)
    prescreen_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}" / "prescreen"
    prescreen_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for idx, atoms in enumerate(candidates):
        run_dir = prescreen_dir / f"prescreen_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write(run_dir / "initial.xyz", atoms)

        input_path = _write_prescreen_input(run_dir, atoms, config, temperature=config.al_temperature)
        _write_slurm_submit_script(run_dir, input_file=input_path.name, nprocs=config.remote_nprocs)

        remote_subdir = (
            f"amorphous_carbon_al_build/phase_1_al_loop/"
            f"iter_{iteration:03d}/prescreen/prescreen_{idx:03d}"
        )
        client.upload_run(run_dir, remote_subdir)
        job_id = client.submit_existing_script(remote_subdir, script_name="submit.slurm")

        jobs.append({
            "local_dir": str(run_dir),
            "remote_subdir": remote_subdir,
            "job_id": job_id,
            "candidate_idx": idx,
        })
        print(f"  Submitted prescreen {idx:03d}: {job_id}")

    return jobs


def _collect_prescreen_jobs(
    config: BuilderConfig,
    client: RemoteSLURMClient,
    jobs: list[dict[str, Any]],
    candidates: list[Atoms],
) -> list[tuple[Atoms, dict[str, Any]]]:
    """Download pre-screen results and return stable candidates with metadata."""
    stable: list[tuple[Atoms, dict[str, Any]]] = []

    for job_info in jobs:
        local_run_dir = Path(job_info["local_dir"])
        remote_subdir = job_info["remote_subdir"]

        if client.is_job_running(job_info["job_id"]):
            print(f"  Prescreen job {job_info['job_id']} still running, skipping")
            continue

        client.download_results(
            local_run_dir,
            remote_subdir,
            patterns=["*-pos-1.xyz", "*-frc-1.xyz", "*.ener", "out", "err"],
        )

        is_stable, reason = _check_prescreen_stability(local_run_dir)
        candidate = candidates[job_info["candidate_idx"]]
        if is_stable:
            # Tag candidate with prescreen metadata
            candidate.info["prescreen_stable"] = True
            candidate.info["prescreen_dir"] = str(local_run_dir)
            stable.append((candidate, job_info))
            print(f"  Prescreen {job_info['candidate_idx']:03d} stable")
        else:
            print(f"  Prescreen {job_info['candidate_idx']:03d} unstable: {reason}")

    return stable


# ---------------------------------------------------------------------------
# 4. Remote job submission helpers (mirrors sio2 script)
# ---------------------------------------------------------------------------


def _write_slurm_submit_script(
    run_dir: Path,
    input_file: str = "input.inp",
    nprocs: int = 64,
    queue: str = "compute",
) -> Path:
    """Write a SLURM submit.slurm for a single CP2K run."""
    script_path = run_dir / "submit.slurm"
    script_content = f"""#!/bin/bash
#SBATCH --job-name=ac_{run_dir.name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --partition={queue}
#SBATCH --output=out
#SBATCH --error=err

# Handle path difference between head (/home/work) and compute nodes (/work)
SUBMIT_DIR="${{SLURM_SUBMIT_DIR:-$(pwd)}}"
HEAD_DIR="$SUBMIT_DIR"
NODE_DIR="${{SUBMIT_DIR/\\/home\\/work/\\/work}}"
if [ -d "$HEAD_DIR" ]; then
    cd "$HEAD_DIR"
elif [ -d "$NODE_DIR" ]; then
    cd "$NODE_DIR"
else
    echo "ERROR: Cannot find work directory"
    exit 1
fi

source /opt/software/cp2k/tools/toolchain/install/setup
export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH
export PMIX_MCA_psec=native

srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./{input_file}
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path


def _write_seed_chain_slurm_script(
    run_dir: Path,
    project_name: str,
    nprocs: int = 64,
    queue: str = "compute",
) -> Path:
    """Write a SLURM submit.slurm that runs heat→hold→quench→relax sequentially."""
    script_path = run_dir / "submit.slurm"
    script_content = f"""#!/bin/bash
#SBATCH --job-name=ac_seed_{run_dir.name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --partition={queue}
#SBATCH --output=out
#SBATCH --error=err

# Handle path difference between head (/home/work) and compute nodes (/work)
SUBMIT_DIR="${{SLURM_SUBMIT_DIR:-$(pwd)}}"
HEAD_DIR="$SUBMIT_DIR"
NODE_DIR="${{SUBMIT_DIR/\\/home\\/work/\\/work}}"
if [ -d "$HEAD_DIR" ]; then
    cd "$HEAD_DIR"
elif [ -d "$NODE_DIR" ]; then
    cd "$NODE_DIR"
else
    echo "ERROR: Cannot find work directory"
    exit 1
fi

source /opt/software/cp2k/tools/toolchain/install/setup
export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH
export PMIX_MCA_psec=native

echo "=== Stage 1: Heat ==="
srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./heat.inp

if [ ! -f "{project_name}_heat-1.restart" ]; then
    echo "ERROR: heat stage did not produce {project_name}_heat-1.restart"
    exit 1
fi

echo "=== Stage 2: Hold ==="
srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./hold.inp

if [ ! -f "{project_name}_hold-1.restart" ]; then
    echo "ERROR: hold stage did not produce {project_name}_hold-1.restart"
    exit 1
fi

echo "=== Stage 3: Quench ==="
srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./quench.inp

if [ ! -f "{project_name}_quench-1.restart" ]; then
    echo "ERROR: quench stage did not produce {project_name}_quench-1.restart"
    exit 1
fi

echo "=== Stage 4: Relax ==="
srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./relax.inp
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path


# ---------------------------------------------------------------------------
# 4. Seed phase workflow
# ---------------------------------------------------------------------------


def seed_submit(config: BuilderConfig, client: RemoteSLURMClient) -> dict[str, Any]:
    """Generate seed structures and submit melt-quench jobs."""
    build_dir = Path(config.build_dir)
    seed_dir = build_dir / "phase_0_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    seeds = generate_seed_structures(config)
    if not seeds:
        raise RuntimeError("No seed structures passed geometry check")

    manifest = load_al_manifest(build_dir)
    submitted: list[dict[str, Any]] = []

    for idx, atoms in enumerate(seeds):
        run_dir = seed_dir / f"seed_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write(run_dir / "initial.xyz", atoms)

        high_temp = atoms.info.get("seed_temp", 3000.0)
        inputs = write_cp2k_inputs_for_seed(run_dir, atoms, config, high_temp=high_temp)
        project_name = f"seed_T{int(high_temp)}"
        _write_seed_chain_slurm_script(
            run_dir,
            project_name=project_name,
            nprocs=config.remote_nprocs,
            queue=config.remote_queue,
        )

        # Upload and submit via SLURM client
        remote_subdir = f"amorphous_carbon_al_build/phase_0_seed/seed_{idx:03d}"
        client.upload_run(run_dir, remote_subdir)
        job_id = client.submit_existing_script(remote_subdir, script_name="submit.slurm")

        submitted.append({
            "local_dir": str(run_dir),
            "remote_subdir": remote_subdir,
            "job_id": job_id,
            "inputs": {k: v.name for k, v in inputs.items()},
            "project_name": project_name,
        })
        print(f"  Submitted seed {idx:03d}: {job_id}")

    manifest["seed_jobs"] = submitted
    manifest["seed_submitted"] = True
    save_al_manifest(build_dir, manifest)
    return {"n_submitted": len(submitted)}


def seed_collect(config: BuilderConfig, client: RemoteSLURMClient) -> dict[str, Any]:
    """Download seed results and merge trajectories into the dataset."""
    build_dir = Path(config.build_dir)
    seed_dir = build_dir / "phase_0_seed"
    db_path = build_dir / config.db_name
    manifest = load_al_manifest(build_dir)

    downloaded_count = 0
    merged_count = 0

    for job_info in manifest.get("seed_jobs", []):
        local_run_dir = Path(job_info["local_dir"])
        remote_subdir = job_info["remote_subdir"]

        # Skip if job still running
        if client.is_job_running(job_info["job_id"]):
            print(f"  Seed job {job_info['job_id']} still running, skipping")
            continue

        # Download results
        downloaded = client.download_results(
            local_run_dir,
            remote_subdir,
            patterns=["*-pos-1.xyz", "*-frc-1.xyz", "*.ener", "*-RESTART.wfn", "out", "err"],
        )
        downloaded_count += len(downloaded)

        # Merge quench and relax trajectories
        project_name = job_info.get("project_name", f"seed_T3000")
        cell = parse_cp2k_cell(local_run_dir / "out")
        if cell is None:
            logger.warning("Could not parse cell from %s/out", local_run_dir.name)
        for phase in ("quench", "relax"):
            project = f"{project_name}_{phase}"
            pos_xyz = local_run_dir / f"{project}-pos-1.xyz"
            frc_xyz = local_run_dir / f"{project}-frc-1.xyz"
            ener_file = local_run_dir / f"{project}-1.ener"
            if not ener_file.exists():
                ener_file = local_run_dir / f"{project}.ener"
            if pos_xyz.exists():
                n = merge_trajectory_to_db(
                    db_path=db_path,
                    pos_xyz_path=pos_xyz,
                    frc_xyz_path=frc_xyz if frc_xyz.exists() else None,
                    ener_path=ener_file if ener_file.exists() else None,
                    metadata={"phase": "seed", "stage": phase},
                    event_driven=config.event_driven_sampling,
                    subsample=1,
                    cell=cell,
                )
                merged_count += n
                print(f"  Merged {n} frames from {local_run_dir.name}/{phase}")

    manifest["seed_completed"] = True
    manifest["total_frames"] = get_db_stats(db_path).get("n_structures", 0)
    save_al_manifest(build_dir, manifest)

    return {
        "downloaded_files": downloaded_count,
        "merged_frames": merged_count,
        "total_frames": manifest["total_frames"],
    }


# ---------------------------------------------------------------------------
# 5. AL loop workflow
# ---------------------------------------------------------------------------


def al_submit(config: BuilderConfig, client: RemoteSLURMClient, iteration: int) -> dict[str, Any]:
    """Generate candidates, run short pre-screen AIMD, and submit production jobs."""
    build_dir = Path(config.build_dir)
    al_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}"
    al_dir.mkdir(parents=True, exist_ok=True)

    # Generate candidates
    candidates = generate_candidates_for_iteration(config, iteration)
    if not candidates:
        print(f"  No candidates for iteration {iteration}, skipping")
        return {"n_submitted": 0}

    # 1 ps short AIMD pre-screening (unless explicitly skipped)
    if not config.skip_prescreen:
        print(f"\n  Running 1 ps pre-screen for {len(candidates)} candidates...")
        prescreen_jobs = _submit_prescreen_jobs(config, client, candidates, iteration)
        print(f"  Waiting for {len(prescreen_jobs)} prescreen jobs...")
        while any(client.is_job_running(j["job_id"]) for j in prescreen_jobs):
            time.sleep(30)
        stable_candidates = _collect_prescreen_jobs(config, client, prescreen_jobs, candidates)
        print(f"  {len(stable_candidates)}/{len(candidates)} candidates passed prescreen")
        if not stable_candidates:
            print(f"  No stable candidates for iteration {iteration}, skipping")
            return {"n_submitted": 0}
        candidates_for_production = [atoms for atoms, _ in stable_candidates[: config.al_runs_per_iter]]
    else:
        print("  Pre-screening skipped (--skip-prescreen)")
        candidates_for_production = candidates[: config.al_runs_per_iter]

    manifest = load_al_manifest(build_dir)
    submitted: list[dict[str, Any]] = []

    for idx, atoms in enumerate(candidates_for_production):
        run_dir = al_dir / f"run_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write(run_dir / "initial.xyz", atoms)

        # Choose induction parameters based on mode and iteration
        kwargs = _choose_induction_parameters(config, iteration, idx)
        run_id = f"iter{iteration:03d}_run{idx:03d}"
        # Use the same suffix that write_cp2k_input_for_induction puts in the
        # CP2K PROJECT_NAME, so that al_collect can locate the output files.
        _mode_suffix = {"annealing": "anneal"}.get(config.al_induction_mode, config.al_induction_mode)
        project_name = f"al_{run_id}_{_mode_suffix}"
        input_path = write_cp2k_input_for_induction(
            run_dir, atoms, config, config.al_induction_mode, run_id, **kwargs
        )
        _write_slurm_submit_script(run_dir, input_file=input_path.name, nprocs=config.remote_nprocs)

        remote_subdir = f"amorphous_carbon_al_build/phase_1_al_loop/iter_{iteration:03d}/run_{idx:03d}"
        client.upload_run(run_dir, remote_subdir)
        job_id = client.submit_existing_script(remote_subdir, script_name="submit.slurm")

        submitted.append({
            "local_dir": str(run_dir),
            "remote_subdir": remote_subdir,
            "job_id": job_id,
            "induction_mode": config.al_induction_mode,
            "induction_params": kwargs,
            "project_name": project_name,
        })
        print(f"  Submitted iter {iteration} run {idx:03d}: {job_id}")

    # Track in manifest
    manifest.setdefault("al_iterations", []).append({
        "iteration": iteration,
        "n_submitted": len(submitted),
        "submitted": submitted,
        "collected": False,
    })
    save_al_manifest(build_dir, manifest)

    return {"n_submitted": len(submitted)}


def _choose_induction_parameters(
    config: BuilderConfig,
    iteration: int,
    run_idx: int,
) -> dict[str, Any]:
    """Choose induction parameters for a given AL run."""
    rng = random.Random(iteration * 1000 + run_idx)
    mode = config.al_induction_mode

    if mode == "annealing":
        return {
            "t_high": rng.choice(config.annealing_temps_high),
        }
    elif mode == "npt":
        return {
            "pressure": rng.choice(config.npt_pressures_gpa),
            "temperature": rng.choice(config.npt_temperatures),
        }
    elif mode == "strain":
        return {
            "temperature": rng.choice(config.npt_temperatures),
        }
    elif mode == "ballistic":
        return {
            "v_temp": rng.choice(config.ballistic_velocity_temps),
            "ensemble": rng.choice(["NVE", "NVT"]),
        }
    return {"temperature": config.al_temperature}


def al_collect(config: BuilderConfig, client: RemoteSLURMClient, iteration: int) -> dict[str, Any]:
    """Download AL iteration results and merge trajectories."""
    build_dir = Path(config.build_dir)
    al_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}"
    db_path = build_dir / config.db_name
    manifest = load_al_manifest(build_dir)

    merged_count = 0
    iter_info = None
    for it in manifest.get("al_iterations", []):
        if it.get("iteration") == iteration:
            iter_info = it
            break

    if iter_info is None:
        print(f"  No manifest entry for iteration {iteration}")
        return {"merged_frames": 0}

    for job_info in iter_info.get("submitted", []):
        local_run_dir = Path(job_info["local_dir"])
        remote_subdir = job_info["remote_subdir"]

        if client.is_job_running(job_info["job_id"]):
            print(f"  AL job {job_info['job_id']} still running, skipping")
            continue

        client.download_results(
            local_run_dir,
            remote_subdir,
            patterns=["*-pos-1.xyz", "*-frc-1.xyz", "*.ener", "*-RESTART.wfn", "out", "err"],
        )

        project_name = job_info.get("project_name", "input")
        # Normalise mode suffix: old manifests stored "annealing" but CP2K
        # output files use "anneal".  Detect and correct on the fly.
        _suffix_fix = {"annealing": "anneal"}
        _base, _, _mode = project_name.rpartition("_")
        if _mode in _suffix_fix:
            project_name = f"{_base}_{_suffix_fix[_mode]}"

        cell = parse_cp2k_cell(local_run_dir / "out")
        cell_by_step = parse_cp2k_md_cells(local_run_dir / "out")
        if cell is None:
            logger.warning("Could not parse cell from %s/out", local_run_dir.name)
        pos_xyz = local_run_dir / f"{project_name}-pos-1.xyz"
        frc_xyz = local_run_dir / f"{project_name}-frc-1.xyz"
        ener_file = local_run_dir / f"{project_name}-1.ener"
        if not ener_file.exists():
            ener_file = local_run_dir / f"{project_name}.ener"
        if pos_xyz.exists():
            n = merge_trajectory_to_db(
                db_path=db_path,
                pos_xyz_path=pos_xyz,
                frc_xyz_path=frc_xyz if frc_xyz.exists() else None,
                ener_path=ener_file if ener_file.exists() else None,
                metadata={
                    "phase": "al",
                    "iteration": iteration,
                    "induction_mode": job_info.get("induction_mode", "md"),
                    "induction_params": job_info.get("induction_params", {}),
                },
                event_driven=config.event_driven_sampling,
                subsample=1,
                cell=cell,
                cell_by_step=cell_by_step,
            )
            merged_count += n
            print(f"  Merged {n} frames from {local_run_dir.name}")
        else:
            logger.warning("No trajectory found in %s for project '%s', skipping", local_run_dir, project_name)

    iter_info["collected"] = True
    manifest["total_frames"] = get_db_stats(db_path).get("n_structures", 0)
    save_al_manifest(build_dir, manifest)

    return {"merged_frames": merged_count, "total_frames": manifest["total_frames"]}


# ---------------------------------------------------------------------------
# 6. Profile-guided strategy selection (stub for future full automation)
# ---------------------------------------------------------------------------


def run_profile_and_suggest_strategy(
    config: BuilderConfig,
) -> tuple[dict[str, float], str]:
    """Profile the current dataset and suggest the next induction mode.

    This is intentionally a lightweight helper, not a full closed-loop selector.
    The user can override the suggestion via CLI.
    """
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name
    if not db_path.exists():
        return {}, config.al_induction_mode

    profiler = DatasetProfiler()
    dataset = AseDbDatasetReader(str(db_path)).read()
    profile = profiler.profile(dataset)
    scores = profile.summary if hasattr(profile, "summary") else {}

    # Simple rule-based suggestion
    suggested_mode = config.al_induction_mode
    if isinstance(scores, dict):
        structural = scores.get("structural_diversity_score", 100.0)
        uniformity = scores.get("uniformity_score", 100.0)
        coverage = scores.get("coverage_score", 100.0)
        redundancy = scores.get("redundancy_score", 0.0)
        if structural < 70:
            suggested_mode = "strain"
        elif uniformity < 70:
            suggested_mode = "annealing"
        elif coverage < 60:
            # Ballistic annealing is used for aggressive high-energy boundary expansion
            # when coverage is still insufficient after annealing / npt / strain rounds.
            suggested_mode = "ballistic"
        elif redundancy > 50:
            suggested_mode = "annealing"  # switch mode to break local redundancy
        else:
            suggested_mode = "npt"

    return scores, suggested_mode


# ---------------------------------------------------------------------------
# 7. Collect from raw CP2K outputs (offline, no remote client needed)
# ---------------------------------------------------------------------------


def collect_from_raw(
    config: BuilderConfig,
    raw_dir: Path | None = None,
    subsample: int = 1,
    event_driven: bool = False,
    event_window: int = 5,
    sparse_stride: int = 20,
    fresh_db: bool = False,
) -> dict[str, Any]:
    """Scan local CP2K output directories and merge trajectories into dataset.db.

    Unlike ``seed-collect`` / ``al-collect``, this function does not talk to a
    remote scheduler and does not require ``al_manifest.json``.  It simply walks
    the standard directory tree under ``raw_dir`` and merges every trajectory it
    finds into ``build_dir / db_name``.

    Args:
        config: Builder configuration.
        raw_dir: Root directory to scan.  Defaults to ``config.build_dir``.
        subsample: Uniform stride for merging frames (1 = every frame).
        event_driven: If True, use TrajectoryMiner instead of uniform stride.
        event_window: Dense window around detected events (event-driven only).
        sparse_stride: Stride for non-event regions (event-driven only).
        fresh_db: If True, remove an existing dataset.db before collecting.

    Returns:
        Dict with merged frame counts per phase and total frames.
    """
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name
    raw_root = raw_dir or build_dir

    if fresh_db and db_path.exists():
        print(f"Removing existing database: {db_path}")
        db_path.unlink()

    merged_seed = 0
    merged_al = 0

    # Seed phase: phase_0_seed/seed_*/
    seed_root = raw_root / "phase_0_seed"
    if seed_root.exists():
        for run_dir in sorted(seed_root.glob("seed_*")):
            if not run_dir.is_dir():
                continue
            out_file = run_dir / "out"
            cell = parse_cp2k_cell(out_file) if out_file.exists() else None
            if cell is None:
                logger.warning("Could not parse cell from %s/out", run_dir.name)

            # Collect quench and relax stages; ignore heat/hold.
            for pos_xyz in sorted(run_dir.glob("*-pos-1.xyz")):
                stem = pos_xyz.name.replace("-pos-1.xyz", "")
                stage = stem.rsplit("_", 1)[-1]
                if stage not in ("quench", "relax"):
                    continue

                frc_xyz = run_dir / f"{stem}-frc-1.xyz"
                ener_file = run_dir / f"{stem}-1.ener"
                if not ener_file.exists():
                    ener_file = run_dir / f"{stem}.ener"

                n = merge_trajectory_to_db(
                    db_path=db_path,
                    pos_xyz_path=pos_xyz,
                    frc_xyz_path=frc_xyz if frc_xyz.exists() else None,
                    ener_path=ener_file if ener_file.exists() else None,
                    metadata={"phase": "seed", "stage": stage},
                    event_driven=event_driven,
                    subsample=subsample,
                    event_window=event_window,
                    sparse_stride=sparse_stride,
                    cell=cell,
                )
                merged_seed += n
                print(f"  Merged {n:4d} seed frames from {run_dir.name}/{stage}")

    # AL phase: phase_1_al_loop/iter_*/run_*/
    al_root = raw_root / "phase_1_al_loop"
    if al_root.exists():
        for iter_dir in sorted(al_root.glob("iter_*")):
            if not iter_dir.is_dir():
                continue
            try:
                iteration = int(iter_dir.name.split("_")[-1])
            except ValueError:
                logger.warning("Skipping unexpected iter directory: %s", iter_dir)
                continue

            for run_dir in sorted(iter_dir.glob("run_*")):
                if not run_dir.is_dir():
                    continue
                out_file = run_dir / "out"
                cell = parse_cp2k_cell(out_file) if out_file.exists() else None
                cell_by_step = parse_cp2k_md_cells(out_file) if out_file.exists() else {}
                if cell is None:
                    logger.warning("Could not parse cell from %s/out", run_dir.name)

                pos_files = sorted(run_dir.glob("*-pos-1.xyz"))
                if not pos_files:
                    logger.warning("No trajectory found in %s, skipping", run_dir)
                    continue

                for pos_xyz in pos_files:
                    stem = pos_xyz.name.replace("-pos-1.xyz", "")
                    frc_xyz = run_dir / f"{stem}-frc-1.xyz"
                    ener_file = run_dir / f"{stem}-1.ener"
                    if not ener_file.exists():
                        ener_file = run_dir / f"{stem}.ener"

                    # Infer induction mode from filename suffix.
                    # CP2K PROJECT uses "anneal", manifest convention uses "annealing".
                    mode = stem.rsplit("_", 1)[-1]
                    induction_mode = {"anneal": "annealing"}.get(mode, mode)

                    n = merge_trajectory_to_db(
                        db_path=db_path,
                        pos_xyz_path=pos_xyz,
                        frc_xyz_path=frc_xyz if frc_xyz.exists() else None,
                        ener_path=ener_file if ener_file.exists() else None,
                        metadata={
                            "phase": "al",
                            "iteration": iteration,
                            "induction_mode": induction_mode,
                        },
                        event_driven=event_driven,
                        subsample=subsample,
                        event_window=event_window,
                        sparse_stride=sparse_stride,
                        cell=cell,
                        cell_by_step=cell_by_step,
                    )
                    merged_al += n
                    print(f"  Merged {n:4d} AL frames from {iter_dir.name}/{run_dir.name} ({induction_mode})")

    total = get_db_stats(db_path).get("n_structures", 0)

    # Keep manifest in sync if it exists.
    manifest = load_al_manifest(build_dir)
    manifest["total_frames"] = total
    save_al_manifest(build_dir, manifest)

    print(f"\nRaw collection complete:")
    print(f"  Seed frames merged:  {merged_seed}")
    print(f"  AL frames merged:    {merged_al}")
    print(f"  Total frames in DB:  {total}")

    return {
        "seed_frames": merged_seed,
        "al_frames": merged_al,
        "total_frames": total,
    }


# ---------------------------------------------------------------------------
# 8. Final 10k dataset construction
# ---------------------------------------------------------------------------


def build_final_dataset(
    config: BuilderConfig,
    n_structures: int = 10000,
) -> dict[str, Any]:
    """Build the final density-adaptive k-center sampled dataset.

    Uses a carbon-specific low-dimensional fingerprint (RDF + density + sp3
    fraction + energy/force statistics) and density-stratified k-center greedy
    sampling to remove local redundancy while preserving coverage across the
    density-sp3 phase space.
    """
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name
    final_db_path = build_dir / f"dataset_final_{n_structures}.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Dataset not found: {db_path}")

    stats = get_db_stats(db_path)
    n_total = stats.get("n_structures", 0)
    if n_total < n_structures:
        print(f"WARNING: dataset has only {n_total} structures, less than requested {n_structures}")

    # Remove old final DB if exists
    if final_db_path.exists():
        final_db_path.unlink()

    result = density_adaptive_kcenter_sample(
        src_db_path=db_path,
        dst_db_path=final_db_path,
        n_select=min(n_structures, n_total),
        density_bin_edges=(config.density_min, 1.8, 2.4, 3.0, config.density_max),
        density_k=10,
    )

    print(f"Final dataset written to {final_db_path}")
    print(f"  Source frames: {result['n_source']}")
    print(f"  Selected frames: {result['n_selected']}")
    print(f"  Density bin quotas: {result['bin_quotas']}")
    print(f"  Density bin scores: {result['bin_scores']}")

    return {
        "source_db": str(db_path),
        "final_db": str(final_db_path),
        "n_requested": n_structures,
        "n_source": result["n_source"],
        "n_selected": result["n_selected"],
        "bin_quotas": result["bin_quotas"],
        "bin_scores": result["bin_scores"],
    }


# ---------------------------------------------------------------------------
# 8. Status / auto mode
# ---------------------------------------------------------------------------


def status(config: BuilderConfig) -> dict[str, Any]:
    """Print current build status."""
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name
    manifest = load_al_manifest(build_dir)

    print(f"Build directory: {build_dir}")
    print(f"Database exists: {db_path.exists()}")
    if db_path.exists():
        stats = get_db_stats(db_path)
        print(f"  Total structures: {stats.get('n_structures', 0)}")
    print(f"Seed submitted: {manifest.get('seed_submitted', False)}")
    print(f"Seed completed: {manifest.get('seed_completed', False)}")
    print(f"AL iterations: {len(manifest.get('al_iterations', []))}")
    return manifest


def auto_mode(config: BuilderConfig, client: RemoteSLURMClient) -> dict[str, Any]:
    """Blocking loop that submits and collects seed + AL iterations.

    The loop is resumable: already submitted/collected steps are skipped.
    """
    manifest = load_al_manifest(Path(config.build_dir))

    if not manifest.get("seed_completed", False):
        if not manifest.get("seed_submitted", False):
            seed_submit(config, client)
        print("Waiting for seed jobs...")
        while any(client.is_job_running(j["job_id"]) for j in manifest.get("seed_jobs", [])):
            time.sleep(60)
        seed_collect(config, client)

    existing_iters = {
        it["iteration"]: it
        for it in load_al_manifest(Path(config.build_dir)).get("al_iterations", [])
    }

    for iteration in range(1, config.al_iterations + 1):
        print(f"\n=== AL Iteration {iteration} ===")

        iter_info = existing_iters.get(iteration)
        if iter_info and iter_info.get("collected", False):
            print(f"  Iteration {iteration} already collected, skipping")
            continue

        if not iter_info:
            al_submit(config, client, iteration)
            iter_info = next(
                (it for it in load_al_manifest(Path(config.build_dir)).get("al_iterations", [])
                 if it.get("iteration") == iteration),
                None,
            )

        if iter_info:
            print("Waiting for AL jobs...")
            time.sleep(60)
            while any(
                client.is_job_running(j["job_id"])
                for j in iter_info.get("submitted", [])
            ):
                time.sleep(60)
                iter_info = next(
                    (it for it in load_al_manifest(Path(config.build_dir)).get("al_iterations", [])
                     if it.get("iteration") == iteration),
                    iter_info,
                )
            al_collect(config, client, iteration)

    return {"status": "auto_completed"}


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------


def _build_config_from_args(args: argparse.Namespace) -> BuilderConfig:
    config = BuilderConfig()

    if args.build_dir:
        config.build_dir = args.build_dir
    if args.n_atoms:
        config.n_atoms = args.n_atoms
    if args.seed_density:
        config.seed_density_g_cm3 = args.seed_density
    if args.seed_temps:
        config.seed_temps = tuple(args.seed_temps)
    if args.seed_runs_per_temp:
        config.seed_runs_per_temp = args.seed_runs_per_temp
    if args.al_iterations:
        config.al_iterations = args.al_iterations
    if args.al_runs_per_iter:
        config.al_runs_per_iter = args.al_runs_per_iter
    if args.al_induction_mode:
        config.al_induction_mode = args.al_induction_mode
    if args.candidate_strategy:
        config.candidate_strategy = args.candidate_strategy
    if args.event_driven:
        config.event_driven_sampling = True
    if args.skip_prescreen:
        config.skip_prescreen = True
    if args.remote_config:
        config.remote_config_path = args.remote_config

    # Load overrides from config file if provided
    if args.config:
        config = BuilderConfig.from_dict({**config.to_dict(), **load_json(args.config)})

    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build amorphous carbon dataset with profile-guided active guidance.")

    # General options
    parser.add_argument("--build-dir", default="amorphous_carbon_al_build",
                        help="Build working directory.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with BuilderConfig overrides.")
    parser.add_argument("--remote-config", default="scripts/amorphous_carbon/remote_config.json",
                        help="Path to remote SLURM config JSON.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without submitting jobs.")

    # System options
    parser.add_argument("--n-atoms", type=int, default=64,
                        help="Number of carbon atoms per structure.")
    parser.add_argument("--seed-density", type=float, default=2.0,
                        help="Seed structure density in g/cm³.")
    parser.add_argument("--density-min", type=float, default=1.2,
                        help="Minimum allowed density for geometry check.")
    parser.add_argument("--density-max", type=float, default=3.5,
                        help="Maximum allowed density for geometry check.")

    # Seed phase options
    parser.add_argument("--seed-temps", type=float, nargs="+", default=[3000.0],
                        help="List of high temperatures for melt-quench seed runs.")
    parser.add_argument("--seed-runs-per-temp", type=int, default=4,
                        help="Number of independent seed runs per temperature.")

    # AL loop options
    parser.add_argument("--al-iterations", type=int, default=4,
                        help="Number of AL iterations.")
    parser.add_argument("--al-runs-per-iter", type=int, default=10,
                        help="Number of production AIMD runs per AL iteration.")
    parser.add_argument("--al-induction-mode", default="annealing",
                        choices=["annealing", "npt", "strain", "ballistic", "md"],
                        help="Induction mode for AL runs.")
    parser.add_argument(
        "--candidate-strategy",
        default="perturbation",
        choices=[
            "perturbation", "random", "boundary", "composite",
            "strain_uniaxial", "strain_biaxial", "strain_shear",
            "strain_hydrostatic",
        ],
        help="Candidate structure generation strategy, independent of induction mode.",
    )
    parser.add_argument("--event-driven", action="store_true",
                        help="Use event-driven trajectory sampling.")
    parser.add_argument("--skip-prescreen", action="store_true",
                        help="Skip 1 ps short AIMD pre-screening (submit candidates directly to production).")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seed-submit", help="Generate seeds and submit melt-quench jobs.")
    subparsers.add_parser("seed-collect", help="Collect seed results and merge into DB.")
    subparsers.add_parser("al-submit", help="Submit one AL iteration (uses current iter from manifest).")
    subparsers.add_parser("al-collect", help="Collect current AL iteration results.")
    subparsers.add_parser("auto", help="Blocking auto loop: seed + all AL iterations.")
    subparsers.add_parser("status", help="Print build status.")

    profile_parser = subparsers.add_parser("profile", help="Profile dataset and suggest next mode.")

    raw_parser = subparsers.add_parser(
        "collect-from-raw",
        help="Scan local CP2K output directories and merge trajectories into dataset.db.",
    )
    raw_parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Root directory containing phase_0_seed and phase_1_al_loop. Defaults to --build-dir.",
    )
    raw_parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Uniform stride for merging frames (default: 1 = every frame).",
    )
    raw_parser.add_argument(
        "--event-driven",
        action="store_true",
        help="Use event-driven sampling instead of uniform stride.",
    )
    raw_parser.add_argument(
        "--event-window",
        type=int,
        default=5,
        help="Dense window around detected events (event-driven only).",
    )
    raw_parser.add_argument(
        "--sparse-stride",
        type=int,
        default=20,
        help="Stride for non-event regions (event-driven only).",
    )
    raw_parser.add_argument(
        "--fresh-db",
        action="store_true",
        help="Remove existing dataset.db before collecting.",
    )

    final_parser = subparsers.add_parser("build-final-dataset", help="Build final sampled dataset.")
    final_parser.add_argument("--n-structures", type=int, default=10000,
                              help="Target number of structures in final dataset.")

    args = parser.parse_args(argv)
    config = _build_config_from_args(args)

    if args.command == "status":
        status(config)
        return 0

    if args.command == "profile":
        scores, suggestion = run_profile_and_suggest_strategy(config)
        print("Profile scores:", json.dumps(scores, indent=2, default=str))
        print(f"Suggested next induction mode: {suggestion}")
        return 0

    if args.command == "collect-from-raw":
        collect_from_raw(
            config,
            raw_dir=Path(args.raw_dir) if args.raw_dir else None,
            subsample=args.subsample,
            event_driven=args.event_driven,
            event_window=args.event_window,
            sparse_stride=args.sparse_stride,
            fresh_db=args.fresh_db,
        )
        return 0

    if args.command == "build-final-dataset":
        build_final_dataset(config, n_structures=args.n_structures)
        return 0

    # Commands requiring remote client
    remote_cfg = load_remote_config(args.remote_config)
    client = RemoteSLURMClient(remote_cfg)

    try:
        if args.command == "seed-submit":
            seed_submit(config, client)
        elif args.command == "seed-collect":
            seed_collect(config, client)
        elif args.command == "al-submit":
            manifest = load_al_manifest(Path(config.build_dir))
            iteration = len(manifest.get("al_iterations", [])) + 1
            al_submit(config, client, iteration)
        elif args.command == "al-collect":
            manifest = load_al_manifest(Path(config.build_dir))
            iteration = len(manifest.get("al_iterations", []))
            if iteration == 0:
                print("No AL iterations to collect")
                return 1
            al_collect(config, client, iteration)
        elif args.command == "auto":
            auto_mode(config, client)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
