#!/usr/bin/env python3
"""Parse phase_0_seed CP2K trajectories into an MLIP-4 training dataset.

Reads positions, forces, and energies from the seed melt-quench trajectories
and writes a uniform subsample to seed.extxyz for MTP pre-training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CP2K writes energies in Hartree and force trajectories in Hartree/bohr;
# this project standardises on eV and eV/Angstrom. Convert at parse time.
HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = 51.42208619083232


def parse_cp2k_ener(ener_path: Path) -> list[dict]:
    """Parse CP2K .ener file into records (energies converted Ha -> eV)."""
    records = []
    if not ener_path.exists():
        return records
    with ener_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                records.append(
                    {
                        "step": int(parts[0]),
                        "time_fs": float(parts[1]),
                        "kinetic_energy": float(parts[2]) * HA_TO_EV,
                        "temperature": float(parts[3]),
                        "potential_energy": float(parts[4]) * HA_TO_EV,
                        "total_energy": float(parts[5]) * HA_TO_EV,
                        "conserved_quantity": float(parts[6]) * HA_TO_EV,
                        "pressure": float(parts[7]) if len(parts) >= 8 else None,
                    }
                )
            except (ValueError, IndexError):
                continue
    return records


def parse_cp2k_trajectory(
    pos_xyz_path: Path,
    frc_xyz_path: Path | None = None,
) -> list[dict]:
    """Parse CP2K position xyz and optional force xyz into frame dicts.

    Forces from ``-frc-1.xyz`` are converted from Ha/bohr to eV/Angstrom.
    """
    from ase.io import read

    if not pos_xyz_path.exists():
        return []

    pos_frames = read(str(pos_xyz_path), index=":")
    if not isinstance(pos_frames, list):
        pos_frames = [pos_frames]

    frc_frames = []
    if frc_xyz_path is not None and frc_xyz_path.exists():
        frc_frames = read(str(frc_xyz_path), index=":")
        if not isinstance(frc_frames, list):
            frc_frames = [frc_frames]

    frames = []
    for i, atoms in enumerate(pos_frames):
        frame = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions(),
            "forces": None,
        }
        if i < len(frc_frames):
            frame["forces"] = frc_frames[i].get_positions() * HA_BOHR_TO_EV_ANG
        frames.append(frame)
    return frames


def build_seed_dataset(
    seed_raw_dir: Path,
    output_path: Path,
    subsample: int = 5,
    cell_ang: float = 8.609794,
) -> dict:
    """Build a seed training set from phase_0_seed trajectories.

    Args:
        seed_raw_dir: Directory containing seed_000, seed_001, ...
        output_path: Output .extxyz file path.
        subsample: Keep every N-th frame from each trajectory segment.
        cell_ang: Cubic cell length in Angstrom.

    Returns:
        Statistics dict with frame counts and energy range.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = ["heat", "hold", "quench", "relax"]
    all_atoms: list[Atoms] = []
    stats = {"seed_runs": [], "total_frames": 0, "segment_counts": {}}

    for seed_dir in sorted(seed_raw_dir.glob("seed_*")):
        seed_name = seed_dir.name
        run_stats = {"name": seed_name, "segments": {}}
        for segment in segments:
            prefix = f"seed_T3000_{segment}"
            pos_path = seed_dir / f"{prefix}-pos-1.xyz"
            frc_path = seed_dir / f"{prefix}-frc-1.xyz"
            ener_path = seed_dir / f"{prefix}-1.ener"

            frames = parse_cp2k_trajectory(pos_path, frc_path)
            energies = parse_cp2k_ener(ener_path)

            if not frames:
                logger.warning(f"No frames for {seed_name}/{segment}")
                continue

            segment_atoms: list[Atoms] = []
            for i in range(0, len(frames), subsample):
                frame = frames[i]
                energy = energies[i]["potential_energy"] if i < len(energies) else None
                temperature = energies[i]["temperature"] if i < len(energies) else None

                atoms = Atoms(
                    symbols=frame["symbols"],
                    positions=frame["positions"],
                    cell=[cell_ang, cell_ang, cell_ang],
                    pbc=True,
                )

                calc = SinglePointCalculator(
                    atoms,
                    energy=energy,
                    forces=frame["forces"],
                )
                atoms.calc = calc
                atoms.info["seed_run"] = seed_name
                atoms.info["segment"] = segment
                atoms.info["frame_index"] = i
                atoms.info["temperature"] = temperature
                atoms.info["source"] = "seed"
                segment_atoms.append(atoms)

            all_atoms.extend(segment_atoms)
            run_stats["segments"][segment] = len(segment_atoms)
            stats["segment_counts"][f"{seed_name}/{segment}"] = len(segment_atoms)
            logger.info(
                f"{seed_name}/{segment}: {len(segment_atoms)} frames "
                f"(subsample={subsample}, original={len(frames)})"
            )

        stats["seed_runs"].append(run_stats)

    if not all_atoms:
        raise RuntimeError("No seed structures parsed.")

    # Unit audit gate: energies must be true CP2K total energies in eV
    # (order 10^2 eV/atom for carbon GTH pseudopotentials). Values of order
    # a few eV/atom indicate Hartree values that were not converted — refuse
    # to write in that case.
    e_per_atom = np.abs([a.get_potential_energy() / len(a) for a in all_atoms])
    median_e = float(np.median(e_per_atom))
    if median_e < 20.0:
        raise RuntimeError(
            f"Unit audit failed: median |E/atom| = {median_e:.3f} looks like "
            f"Hartree, not eV. Refusing to write {output_path}."
        )

    write(str(output_path), all_atoms)
    stats["total_frames"] = len(all_atoms)
    stats["output_file"] = str(output_path)

    energies = [a.get_potential_energy() for a in all_atoms]
    stats["energy_min_ev"] = float(np.min(energies))
    stats["energy_max_ev"] = float(np.max(energies))
    stats["energy_mean_ev"] = float(np.mean(energies))

    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Wrote {len(all_atoms)} seed frames to {output_path}")
    logger.info(f"Energy range: {stats['energy_min_ev']:.3f} ~ {stats['energy_max_ev']:.3f} eV")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare seed dataset from phase_0_seed")
    parser.add_argument(
        "--seed-raw-dir",
        type=str,
        default="data/mlip4_al/00_seed_raw/phase_0_seed",
        help="Directory containing seed_* subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mlip4_al/01_seed_dataset/seed.extxyz",
        help="Output extxyz file",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Keep every N-th frame (default=1 keeps all frames)",
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=8.609794,
        help="Cubic cell length in Angstrom",
    )
    args = parser.parse_args()

    stats = build_seed_dataset(
        seed_raw_dir=Path(args.seed_raw_dir),
        output_path=Path(args.output),
        subsample=args.subsample,
        cell_ang=args.cell,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
