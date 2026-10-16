#!/usr/bin/env python3
"""MLIP-4 MTP helpers: training, ASE MD exploration, and EG selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs

logger = logging.getLogger(__name__)

EV_TO_HA = 1.0 / 27.211386245988
EV_ANG_TO_HA_BOHR = 1.0 / 51.42208619083232
HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = 51.42208619083232


def train_mtp(
    atoms_list: list[Atoms],
    mtp_config: dict[str, Any],
    output_path: Path,
    initial_mtp_path: Path | None = None,
    timing: Any | None = None,
    train_cores: int = 1,
    iteration: int | None = None,
) -> Path:
    """Train an MTP on a list of ASE Atoms and save to output_path.

    Args:
        atoms_list: Training structures with energy and forces attached.
        mtp_config: Configuration dict for MTP (level, rb_*, training params, weights).
        output_path: Path to write the trained MTP JSON.
        initial_mtp_path: Optional path to a previously trained MTP to fine-tune.
        timing: Optional TimingTracker for core-hour accounting.
        train_cores: Number of CPU cores used for MTP training.
        iteration: AL iteration number (for logging).

    Returns:
        Path to saved MTP JSON.
    """
    import mlip_4
    import time

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if timing is not None:
        timing.start("mtp_train", n_cores=train_cores, iteration=iteration,
                     n_train=len(atoms_list))
    t0 = time.perf_counter()

    # Build loss function
    lf = mlip_4.LossFunction()
    for at in atoms_list:
        cfg = mlip_4.Cfg.from_ase(at)
        e_ha = float(at.get_potential_energy()) * EV_TO_HA
        f_ha_bohr = at.get_forces() * EV_ANG_TO_HA_BOHR
        calc_cfg = mlip_4.CalcCfg(cfg=cfg, energy=e_ha, forces=f_ha_bohr, stress=False)
        loss_cfg = mlip_4.LossCfg(
            sim=calc_cfg,
            weight_energy=mtp_config.get("weight_energy", 1.0),
            weight_forces=mtp_config.get("weight_forces", 0.01),
            weight_stress=mtp_config.get("weight_stress", 0.0),
        )
        lf.add(loss_cfg)

    # Build radial basis
    rb = mlip_4.RadialBasisCinf(
        size=mtp_config.get("rb_size", 8),
        min_dist=mtp_config.get("rb_min_dist", 1.0),
        cutoff=mtp_config.get("rb_cutoff", 5.0),
    )

    if initial_mtp_path is not None and Path(initial_mtp_path).exists():
        logger.info(f"Loading initial MTP from {initial_mtp_path}")
        with open(initial_mtp_path, "rb") as f:
            mtp = mlip_4.MTP.from_json_bytes(f.read())
    else:
        mtp = mlip_4.MTP(
            rb,
            species=mtp_config.get("species", [6]),
            level=mtp_config.get("level", 16),
            jit=False,
        )
        mtp.params[:] = np.random.uniform(low=-1.0, high=1.0, size=len(mtp.params))

    tp = mlip_4.TrainingParams(
        max_iter_count=mtp_config.get("max_iter_count", 2000),
        linstop=mtp_config.get("linstop", 1.0e-4),
        rescaling=mtp_config.get("rescaling", True),
        pretraining=mtp_config.get("pretraining", True),
    )

    trainer = mlip_4.Trainer(loss_func=lf)
    logger.info(f"Training MTP (level={mtp_config.get('level', 16)}, "
                f"N_train={len(atoms_list)})...")
    trainer.train(mtp, training_params=tp)

    with open(output_path, "wb") as f:
        f.write(mtp.to_json_bytes())

    wall = time.perf_counter() - t0
    core_hours = wall * train_cores / 3600.0
    logger.info(f"MTP saved to {output_path} (wall={wall:.1f}s, cores={train_cores}, "
                f"core_hours={core_hours:.4f})")
    if timing is not None:
        timing.stop("mtp_train", iteration=iteration)
    return output_path


def load_mlip4_calculator(mtp_path: Path):
    """Return an ASE-compatible MLIP-4 calculator for the given MTP."""
    import mlip_4

    class MLIP4ASECalculator(Calculator):
        """ASE calculator wrapping mlip_4.CfgCalc."""

        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, potential_path: Path, **kwargs):
            super().__init__(**kwargs)
            self.potential_path = Path(potential_path)
            with open(self.potential_path, "rb") as f:
                self._mtp = mlip_4.MTP.from_json_bytes(f.read())
            self._cfg_calc = mlip_4.CfgCalc(pot=self._mtp)

        def calculate(self, atoms: Atoms | None = None, properties=None, system_changes=all_changes):
            if atoms is not None:
                self.atoms = atoms.copy()
            if self.atoms is None:
                raise RuntimeError("No atoms set")

            properties = properties or self.implemented_properties
            cfg = mlip_4.Cfg.from_ase(self.atoms)
            result = self._cfg_calc.calc(cfg, what="epfs")

            self.results = {}
            if "energy" in properties:
                # MLIP-4 returns energies in Hartree atomic units; ASE expects eV
                self.results["energy"] = float(result.get("energy", 0.0)) * HA_TO_EV
            if "forces" in properties:
                forces = result.get("forces")
                if forces is not None:
                    # MLIP-4 returns forces in Hartree/bohr; ASE expects eV/Angstrom
                    self.results["forces"] = np.array(forces, dtype=float) * HA_BOHR_TO_EV_ANG
                else:
                    self.results["forces"] = np.zeros((len(self.atoms), 3))
            if "stress" in properties:
                stress = result.get("stress")
                if stress is not None:
                    s = np.array(stress, dtype=float)
                    if s.shape == (3, 3):
                        self.results["stress"] = np.array([
                            s[0, 0], s[1, 1], s[2, 2],
                            s[1, 2], s[0, 2], s[0, 1],
                        ])
                    else:
                        self.results["stress"] = s
                else:
                    self.results["stress"] = np.zeros(6)

        def get_extrapolation_grade(self, atoms: Atoms | None = None) -> float:
            if atoms is not None:
                self.calculate(atoms)
            if self.atoms is None:
                raise RuntimeError("No atoms set")
            cfg = mlip_4.Cfg.from_ase(self.atoms)
            return float(self._cfg_calc.grade(cfg))

    return MLIP4ASECalculator(mtp_path)


def run_md_exploration(
    initial_atoms: Atoms,
    mtp_path: Path,
    md_config: dict[str, Any],
    output_dir: Path,
    iteration: int,
) -> tuple[list[Atoms], list[float]]:
    """Run ASE MD with the current MTP across a temperature ladder.

    Returns:
        (all_sampled_frames, all_extrapolation_grades)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calc = load_mlip4_calculator(mtp_path)
    temperatures = md_config.get("temperatures", [300, 500, 800, 1000, 1500, 2000, 3000])
    timestep_fs = md_config.get("timestep_fs", 1.0)
    n_steps = md_config.get("n_steps", 5000)
    sample_interval = md_config.get("sample_interval", 50)
    friction = md_config.get("friction", 0.002)

    all_frames: list[Atoms] = []
    all_grades: list[float] = []

    for temp in temperatures:
        logger.info(f"Iteration {iteration}: MD at T={temp}K ...")
        atoms = initial_atoms.copy()
        atoms.calc = calc
        atoms.set_cell(initial_atoms.get_cell())
        atoms.set_pbc(True)

        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * fs,
            temperature_K=temp,
            friction=friction / fs,
        )

        temp_frames: list[Atoms] = []

        def sample():
            nonlocal temp_frames
            temp_frames.append(atoms.copy())

        # Attach to sample every sample_interval steps
        dyn.attach(sample, interval=sample_interval)
        dyn.run(n_steps)

        # Evaluate EG for each sampled frame
        temp_grades = []
        for frame in temp_frames:
            grade = calc.get_extrapolation_grade(frame)
            temp_grades.append(grade)
            frame.info["temperature"] = temp
            frame.info["iteration"] = iteration
            frame.info["md_step"] = len(temp_grades) * sample_interval

        # Save per-temperature trajectory
        traj_path = output_dir / f"md_iter{iteration:03d}_T{temp:.0f}K.xyz"
        write(str(traj_path), temp_frames)

        all_frames.extend(temp_frames)
        all_grades.extend(temp_grades)
        logger.info(f"  T={temp}K: {len(temp_frames)} frames, "
                    f"EG min/mean/max={np.min(temp_grades):.3f}/"
                    f"{np.mean(temp_grades):.3f}/{np.max(temp_grades):.3f}")

    # Save grades summary
    summary = {
        "iteration": iteration,
        "n_frames": len(all_frames),
        "temperatures": temperatures,
        "grades": {
            "min": float(np.min(all_grades)) if all_grades else None,
            "max": float(np.max(all_grades)) if all_grades else None,
            "mean": float(np.mean(all_grades)) if all_grades else None,
            "median": float(np.median(all_grades)) if all_grades else None,
        },
    }
    with open(output_dir / f"md_iter{iteration:03d}_grades.json", "w") as f:
        json.dump(summary, f, indent=2)

    return all_frames, all_grades


def select_by_grade(
    frames: list[Atoms],
    grades: list[float],
    gamma_threshold: float,
    gamma_break: float,
    max_select: int,
) -> tuple[list[Atoms], list[float], bool]:
    """Select top-N frames by extrapolation grade.

    Frames with grade >= gamma_break are treated as model-breakdown cases and
    skipped.  Convergence is declared only when no frame exceeds
    gamma_threshold (i.e. the whole exploration is inside the training domain).

    Returns:
        (selected_frames, selected_grades, converged)
    """
    if len(frames) != len(grades):
        raise ValueError("frames and grades must have the same length")

    gamma_threshold = float(gamma_threshold)
    gamma_break = float(gamma_break)

    valid = [(f, g) for f, g in zip(frames, grades) if g < gamma_break]
    if not valid:
        max_grade = max(grades) if grades else 0.0
        logger.info(f"No frames below gamma_break={gamma_break} (max grade={max_grade:.3e}).")
        return [], [], True

    max_grade = max(g for _, g in valid)
    if max_grade <= gamma_threshold:
        logger.info(f"Max grade {max_grade:.3e} <= threshold {gamma_threshold}. Converged.")
        return [], [], True

    # Sort by grade descending and keep top max_select
    valid.sort(key=lambda x: x[1], reverse=True)
    selected = valid[:max_select]

    selected_frames = [f for f, _ in selected]
    selected_grades = [g for _, g in selected]

    logger.info(f"Selected {len(selected_frames)} frames "
                f"(threshold={gamma_threshold}, break={gamma_break}, "
                f"max_select={max_select}). Grade range: "
                f"[{selected_grades[-1]:.3e}, {selected_grades[0]:.3e}]")
    return selected_frames, selected_grades, False
