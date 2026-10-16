#!/usr/bin/env python3
"""
Prepare Stability ablation datasets for Table 2 (R4.2).

This script constructs the three P1 stability-ablation subsets used in the
"dimension-wise ablation" section of the paper (Table 2).  It reads the full
SiO2 AL dataset, classifies every structure into EQ/MS/TP/TD/LU/AB using
DeterministicPreFilter, and writes three ASE databases:

    - dataset_eq_only.db        : EQ structures only
    - dataset_ms_only.db        : MS structures only (off-equilibrium proxy)
    - dataset_eq_ms_mixed.db    : EQ + MS mixed subset

For the current SiO2 data DeterministicPreFilter yields zero TP structures
because D_norm is compressed below the TP threshold (D_norm >= 0.3).  The
script therefore uses MS as the off-equilibrium class by default, matching the
paper's R1.2/B ablation design.  Use --use-tp to switch to TP if your data
contains TP-classified structures.

Outputs are written to outputs/ablation_stability/.
This script is CPU-only and does not require a trained MLFF.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

# Project imports
from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.sampling import DeterministicPreFilter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DB = ROOT / "sio2_al_build" / "dataset.db"
OUTPUT_DIR = ROOT / "outputs" / "ablation_stability"


def _record_to_atoms(record: object) -> Atoms:
    """Convert a StructureRecord to an ASE Atoms object with energy/forces."""
    cell = list(record.cell) if record.cell is not None else None
    atoms = Atoms(
        numbers=list(record.atomic_numbers),
        positions=list(record.positions),
        cell=cell,
        pbc=list(record.pbc),
    )
    energy = float(record.energy) if record.energy is not None else None
    forces = np.asarray(record.forces, dtype=np.float64) if record.forces else None
    if energy is not None or forces is not None:
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc
    return atoms


def _build_kwargs(record: object) -> dict[str, Any]:
    """Build ASE DB keyword arguments from record metadata."""
    kwargs: dict[str, Any] = {}
    for key, value in record.metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            kwargs[key] = value
    return kwargs


def _resolve_subsets(label_counts: Counter, use_tp: bool) -> dict[str, tuple[str, ...]]:
    """Decide which labels form the off-equilibrium subset.

    Default uses MS because SiO2 data currently yields zero TP structures
    under DeterministicPreFilter (D_norm max ~0.09, TP requires >=0.3).
    """
    if use_tp:
        if label_counts.get("TP", 0) == 0:
            raise ValueError(
                "--use-tp requested but no TP structures were classified. "
                "Run without --use-tp to use MS as the off-equilibrium class."
            )
        return {
            "EQ only": ("EQ",),
            "TP only": ("TP",),
            "EQ+TP mixed": ("EQ", "TP"),
        }

    return {
        "EQ only": ("EQ",),
        "MS only": ("MS",),
        "EQ+MS mixed": ("EQ", "MS"),
    }


def prepare_ablation_dbs(
    source_db: Path,
    output_dir: Path,
    use_tp: bool = False,
) -> dict[str, int]:
    """Classify source structures and write ablation DBs for Table 2."""
    print(f"Reading source database: {source_db}")
    dataset = AseDbDatasetReader(source_db).read(dataset_name="sio2_full")
    print(f"  Total structures: {len(dataset.records)}")

    print("\nBuilding DeterministicPreFilter envelopes ...")
    prefilter = DeterministicPreFilter(reference_dataset=dataset)

    print("Classifying structures ...")
    labels: list[str] = []
    for record in dataset.records:
        result = prefilter.classify(record)
        labels.append(result.label)

    label_counts = Counter(labels)
    print("\nClassification counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:>3}: {count:>6}")

    subsets = _resolve_subsets(label_counts, use_tp)
    off_eq_label = "TP" if use_tp else "MS"
    print(f"\nUsing '{off_eq_label}' as the off-equilibrium class.")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    for subset_name, allowed in subsets.items():
        selected = [
            (record, label)
            for record, label in zip(dataset.records, labels)
            if label in allowed
        ]

        safe_name = subset_name.lower().replace(" ", "_").replace("+", "_")
        out_path = output_dir / f"dataset_{safe_name}.db"

        if out_path.exists():
            out_path.unlink()

        with connect(out_path) as db:
            for record, label in selected:
                atoms = _record_to_atoms(record)
                kwargs = _build_kwargs(record)
                kwargs["struct_class"] = label
                db.write(atoms, **kwargs)

        stats[subset_name] = len(selected)
        print(f"  Wrote {len(selected):>6} structures to {out_path}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare EQ/MS/EQ+MS (or EQ/TP/EQ+TP) stability ablation datasets for Table 2."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DB,
        help="Source ASE database path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for ablation databases.",
    )
    parser.add_argument(
        "--use-tp",
        action="store_true",
        help="Use TP as the off-equilibrium class instead of MS. "
             "Only works if the data contains TP-classified structures.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: source database not found: {args.source}", file=sys.stderr)
        return 1

    try:
        stats = prepare_ablation_dbs(args.source, args.output_dir, use_tp=args.use_tp)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 50)
    print("Table 2 stability ablation datasets summary")
    print("=" * 50)
    for name, count in stats.items():
        print(f"  {name:<15}: {count:>6} structures")
    print(f"\nOutput directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
