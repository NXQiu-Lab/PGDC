#!/usr/bin/env python3
"""
Convert ASE databases from top-level attribute format to schnetpack-compatible
row.data format.

schnetpack's ASEAtomsData reads properties via row.data[pname] (see
schnetpack/data/atoms.py:167), but several SiO2 databases store energy/forces
as top-level row attributes with empty data dicts and empty metadata.

This script fixes both issues:
  1. Sets metadata: {"_distance_unit": "Angstrom",
                     "_property_unit_dict": {"energy": "eV",
                                             "forces": "eV/Angstrom",
                                             "stress": "eV/Angstrom/Angstrom/Angstrom"},
                     "atomrefs": {}}
  2. Moves row.energy → data["energy"] (as shape (1,) ndarray)
     and  row.forces → data["forces"] (as (N,3) ndarray)
  3. Preserves row.data["stress"] if present and converts ASE Voigt 6-vectors
     to the 3x3 tensor required by schnetpack.

Usage:
    uv run python scripts/paper/convert_db_for_schnetpack.py [DB_PATHS ...]

If no paths given, converts the 5 known SiO2 training databases.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB_PATHS = [
    ROOT / "sio2_al_build" / "dataset.db",
    ROOT / "outputs" / "ablation_sparse" / "dataset_sparse_10000_adaptive.db",
    ROOT / "outputs" / "ablation_stability" / "dataset_eq_only.db",
    ROOT / "outputs" / "ablation_stability" / "dataset_ms_only.db",
    ROOT / "outputs" / "ablation_stability" / "dataset_eq_ms_mixed.db",
]

SPK_METADATA = {
    "_distance_unit": "Angstrom",
    "_property_unit_dict": {"energy": "eV", "forces": "eV/Angstrom"},
    "atomrefs": {},
}

STRESS_UNIT = "eV/Angstrom/Angstrom/Angstrom"


def _voigt_to_full(stress: np.ndarray) -> np.ndarray:
    """Convert an ASE-style 6-component Voigt stress to a 3x3 tensor.

    ASE Voigt order: [xx, yy, zz, yz, xz, xy].
    schnetpack expects the full symmetric (3, 3) tensor.
    """
    s = np.asarray(stress, dtype=np.float64).ravel()
    if s.size == 9:
        return s.reshape(3, 3)
    if s.size != 6:
        raise ValueError(
            f"Stress must have 6 (Voigt) or 9 (full tensor) components, got {s.size}"
        )
    xx, yy, zz, yz, xz, xy = s
    return np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64
    )


def _stress_to_schnetpack(stress: np.ndarray) -> np.ndarray:
    """Return stress with schnetpack's expected shape (1, 3, 3).

    The ASEAtomsData collate function concatenates along the first axis, so
    per-structure properties must carry a leading batch dimension of 1.
    """
    full = _voigt_to_full(stress)
    if full.shape == (3, 3):
        full = full[np.newaxis, :, :]
    if full.shape != (1, 3, 3):
        raise ValueError(f"Stress shape must be (1, 3, 3) for schnetpack, got {full.shape}")
    return full


def _needs_conversion(db_path: Path) -> bool:
    """Check whether a DB has empty metadata (i.e. needs conversion)."""
    with connect(db_path) as db:
        md = db.metadata
        return "_property_unit_dict" not in md


def _read_stress(row) -> np.ndarray | None:
    """Read stress from row.data, preferring the 3x3 form."""
    data = row.data if hasattr(row, "data") and row.data else {}
    if "stress" in data:
        return _voigt_to_full(data["stress"])
    return None


def convert_db(db_path: Path, force: bool = False) -> int:
    """Convert a single DB in-place. Returns number of rows converted."""
    if not _needs_conversion(db_path) and not force:
        print(f"  SKIP  {db_path.name} (already has _property_unit_dict)")
        return 0

    with connect(db_path) as db:
        rows = []
        has_stress = False
        for row in db.select():
            atoms = row.toatoms()
            # Prefer top-level ASE columns, fall back to row.data for databases
            # that store properties exclusively in the data dict.
            energy = getattr(row, "energy", None)
            forces = getattr(row, "forces", None)
            data = row.data if hasattr(row, "data") and row.data else {}
            if energy is None and "energy" in data:
                energy = data["energy"]
            if forces is None and "forces" in data:
                forces = data["forces"]
            stress = _read_stress(row)
            if stress is not None:
                stress = _stress_to_schnetpack(stress)
                has_stress = True

            # Collect all key_value_pairs + extra columns
            kvp = dict(row.key_value_pairs) if row.key_value_pairs else {}

            # Convert energy to shape (1,) ndarray
            energy_arr = np.atleast_1d(np.asarray(energy, dtype=np.float64))
            forces_arr = np.asarray(forces, dtype=np.float64) if forces is not None else None

            rows.append((atoms, energy_arr, forces_arr, stress, kvp))

    n = len(rows)
    print(f"  READ  {db_path.name}: {n} rows")

    # Write to a temp file then replace
    with tempfile.NamedTemporaryFile(
        dir=db_path.parent, suffix=".db", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with connect(tmp_path) as new_db:
            # Set metadata
            metadata = dict(SPK_METADATA)
            if has_stress:
                metadata["_property_unit_dict"] = dict(metadata["_property_unit_dict"])
                metadata["_property_unit_dict"]["stress"] = STRESS_UNIT
            new_db.metadata = metadata

            for atoms, energy_arr, forces_arr, stress, kvp in rows:
                data = {"energy": energy_arr}
                if forces_arr is not None:
                    data["forces"] = forces_arr
                if stress is not None:
                    data["stress"] = stress
                new_db.write(atoms, data=data, key_value_pairs=kvp)

        # Verify
        with connect(tmp_path) as verify_db:
            md = verify_db.metadata
            row = next(verify_db.select())
            assert "_property_unit_dict" in md, "metadata missing _property_unit_dict"
            assert "energy" in row.data, "row.data missing energy"
            assert "forces" in row.data, "row.data missing forces"
            assert row.data["energy"].shape == (1,), (
                f"energy shape expected (1,), got {row.data['energy'].shape}"
            )
            if has_stress:
                assert "stress" in md["_property_unit_dict"], "metadata missing stress unit"
                assert "stress" in row.data, "row.data missing stress"
                assert row.data["stress"].shape == (1, 3, 3), (
                    f"stress shape expected (1, 3, 3), got {row.data['stress'].shape}"
                )

        # Replace original
        tmp_path.replace(db_path)
        print(f"  WRITE {db_path.name}: {n} rows converted")
        if has_stress:
            print(f"        stress present (1, 3, 3) tensor, unit={STRESS_UNIT}")
        return n

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ASE DBs to schnetpack-compatible format."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="DB paths to convert. If empty, converts the 5 known SiO2 databases.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if metadata already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be converted without writing.",
    )
    args = parser.parse_args()

    paths = args.paths if args.paths else DEFAULT_DB_PATHS

    total = 0
    for p in paths:
        if not p.exists():
            print(f"  ERROR {p} not found", file=sys.stderr)
            continue
        if args.dry_run:
            needs = _needs_conversion(p)
            print(f"  {'NEEDS' if needs else 'OK   '} {p}")
        else:
            total += convert_db(p, force=args.force)

    if not args.dry_run:
        print(f"\nDone. {total} rows converted across {len(paths)} databases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
