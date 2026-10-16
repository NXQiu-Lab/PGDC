#!/usr/bin/env python3
"""Fix the legacy 10000-structure schnetpack DB: stress sign + energy alignment.

Two defects existed in ``amorphous_carbon_dataset_final_10000.db``:

1. **Stress sign convention.**  The stress tensors were parsed from CP2K
   single-point outputs, which print stress with *positive = compressive* —
   the opposite of the ASE / schnetpack convention (positive = tensile,
   pressure = -trace(sigma)/3).  Verified empirically on iter_006 NPT runs:
   a 150 GPa barostat equilibrates at trace(sigma)/3 = +149 GPa in raw CP2K
   output.  All stored tensors are therefore multiplied by -1.

2. **Index-based energy/temperature misalignment.**  The DB inherits the
   legacy misalignment from ``dataset.db`` (fixed separately by
   ``fix_legacy_alignment.py``): frame ``i`` was labelled with the energy of
   MD step ``i`` instead of step ``stride*i``.  Energies/temperatures are
   re-derived from the raw trajectories via exact position matching.

Rows originating from AL iteration 2 (NPT) are a special case: their stress
single points were computed on the stored (expanded, step-0) cell while their
forces come from the compressed trajectory cell, and the single-point outputs
needed to rebuild consistent labels were cleaned up after collection.  They
are flagged with ``labels_inconsistent=True`` (stress sign is still flipped
for convention uniformity) and should be excluded from training or re-labelled
with fresh single points.

Example
-------

.. code-block:: bash

    uv run python scripts/amorphous_carbon/fix_legacy_10000_db.py \
        --build-dir amorphous_carbon_al_build --dry-run

    uv run python scripts/amorphous_carbon/fix_legacy_10000_db.py \
        --build-dir amorphous_carbon_al_build
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "amorphous_carbon"))

import numpy as np
from ase.db import connect

from fix_legacy_alignment import build_label_lookup

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build-dir", type=str, default="amorphous_carbon_al_build")
    p.add_argument("--db", type=str, default=None,
                   help="ASE DB path (default: <build-dir>/amorphous_carbon_dataset_final_10000.db)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    build_dir = Path(args.build_dir)
    db_path = Path(args.db) if args.db else build_dir / "amorphous_carbon_dataset_final_10000.db"
    print(f"DB: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")

    lut = build_label_lookup(build_dir)
    print(f"\nLookup built: {len(lut)} frames")

    n_rows = n_stress_flipped = n_energy_fixed = n_temp_fixed = 0
    n_iter2_flagged = n_unmatched = 0
    unmatched_ids: list[int] = []

    with connect(str(db_path)) as db:
        rows = list(db.select())
        n_rows = len(rows)
        print(f"Rows: {n_rows}")

        for row in rows:
            kv = row.key_value_pairs
            data_update: dict[str, np.ndarray] = {}

            stress = row.data.get("stress") if row.data else None
            if stress is not None:
                data_update["stress"] = -np.asarray(stress)
                n_stress_flipped += 1

            if kv.get("iteration") == 2:
                # Stress was computed on the expanded step-0 cell; trajectory
                # labels cannot be made consistent without fresh single points.
                n_iter2_flagged += 1
                if not args.dry_run:
                    db.update(row.id, data=data_update, labels_inconsistent=True)
                continue

            key = np.ascontiguousarray(np.asarray(row.positions)).tobytes()
            label = lut.get(key)
            if label is None:
                n_unmatched += 1
                unmatched_ids.append(row.id)
                if not args.dry_run:
                    db.update(row.id, data=data_update)
                continue

            e_old = float(np.asarray(row.data["energy"]).flat[0])
            kvp_update: dict[str, object] = {}
            if abs(e_old - label.energy) > 1e-12:
                data_update["energy"] = np.array([label.energy])
                n_energy_fixed += 1
            if label.temperature is not None and kv.get("temperature") != label.temperature:
                kvp_update["temperature"] = label.temperature
                n_temp_fixed += 1

            if not args.dry_run:
                db.update(row.id, data=data_update, **kvp_update)

    print(f"\nRows:                     {n_rows}")
    print(f"Stress sign flipped:      {n_stress_flipped}")
    print(f"Energy corrected:         {n_energy_fixed}")
    print(f"Temperature corrected:    {n_temp_fixed}")
    print(f"iter2 rows flagged:       {n_iter2_flagged}")
    print(f"Unmatched (non-iter2):    {n_unmatched}")
    if unmatched_ids:
        print(f"  unmatched row ids (first 10): {unmatched_ids[:10]}")
    if args.dry_run:
        print("Dry run: DB was not modified.")
    else:
        print(f"Updates written to {db_path}.")


if __name__ == "__main__":
    main()
