#!/usr/bin/env python3
"""Convert the final MLIP-4 AL dataset (extxyz) to an ASE DB.

The resulting .db file can be read by the current project's AseDbDatasetReader
and profiled with DatasetProfiler.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_extxyz_to_db(
    extxyz_path: Path,
    db_path: Path,
    metadata: dict | None = None,
) -> dict:
    """Convert an extxyz file to ASE DB with energy/forces and metadata."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = list(read(str(extxyz_path), index=":"))
    metadata = metadata or {}

    added = 0
    with connect(str(db_path), append=False) as db:
        for i, atoms in enumerate(atoms_list):
            # Ensure calculator has energy/forces
            if atoms.calc is None or not hasattr(atoms.calc, "results"):
                logger.warning(f"Frame {i} has no calculator; skipping")
                continue

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

            kvp = {
                "source": atoms.info.get("source", "mlip4_al"),
                "iteration": atoms.info.get("iteration", 0),
                "frame_index": i,
            }
            if "temperature" in atoms.info:
                kvp["temperature"] = atoms.info["temperature"]
            if "gamma" in atoms.info:
                kvp["gamma"] = atoms.info["gamma"]
            if "seed_run" in atoms.info:
                kvp["seed_run"] = atoms.info["seed_run"]
            if "segment" in atoms.info:
                kvp["segment"] = atoms.info["segment"]
            if "dft_label" in atoms.info:
                kvp["dft_label"] = atoms.info["dft_label"]
            kvp.update(metadata)

            db.write(atoms, **kvp)
            added += 1

    stats = {
        "input_file": str(extxyz_path),
        "output_db": str(db_path),
        "n_structures": added,
    }
    logger.info(f"Wrote {added} structures to {db_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert MLIP-4 AL extxyz to ASE DB")
    parser.add_argument("--input", type=str,
                        default="outputs/mlip4_al/03_final_dataset/mlip4_al_dataset.extxyz",
                        help="Input extxyz file")
    parser.add_argument("--output", type=str,
                        default="outputs/mlip4_al/03_final_dataset/mlip4_al_dataset.db",
                        help="Output ASE DB file")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Optional JSON metadata to add to every row")
    args = parser.parse_args()

    metadata = {}
    if args.metadata:
        with open(args.metadata, "r") as f:
            metadata = json.load(f)

    stats = convert_extxyz_to_db(Path(args.input), Path(args.output), metadata)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
