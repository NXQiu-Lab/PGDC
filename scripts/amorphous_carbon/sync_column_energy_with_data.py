#!/usr/bin/env python3
"""Sync the ASE DB energy column with ``data['energy']`` for legacy a-C databases.

Background
----------
``fix_legacy_10000_db.py`` corrected the index-based energy misalignment in
``amorphous_carbon_dataset_final_10000.db`` by rewriting ``data['energy']``
with position-matched labels re-derived from raw CP2K trajectories, but it did
not update the ASE DB ``energy`` column. Derived databases
(``amorphous_carbon_dataset_stress_merged.db`` and
``amorphous_carbon_comparison_12k.db``) inherited the stale column.

schnetpack reads ``row.data['energy']`` (correct), but analysis code and the
``energy_threshold``/matching helpers read the column (stale). This script
sets the column to the data value wherever they differ. Forces are untouched
(position/force pairing was always correct — see fix_legacy_alignment.py).

Verified: for a sample of mismatched rows, ``data['energy']`` matches the
alignment-corrected ``dataset.db`` column under exact position matching,
while the stale column value does not.

Example
-------
.. code-block:: bash

    uv run python scripts/amorphous_carbon/sync_column_energy_with_data.py \
        --db amorphous_carbon_al_build/amorphous_carbon_dataset_final_10000.db \
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.db import connect

logger = logging.getLogger("sync-column-energy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", nargs="+", action="append", required=True,
                   help="ASE DB path(s) to repair (repeatable flag).")
    p.add_argument("--dry-run", action="store_true", help="Report only; do not modify.")
    p.add_argument("--no-backup", action="store_true", help="Skip the .bak file backup.")
    return p.parse_args()


def repair_db(db_path: Path, dry_run: bool, backup: bool) -> None:
    with connect(str(db_path)) as db:
        mismatched = []
        for row in db.select():
            data = row.data
            if not data or "energy" not in data:
                continue
            data_energy = float(np.asarray(data["energy"]).flat[0])
            if abs(data_energy - row.energy) > 1e-6:
                mismatched.append((row.id, data_energy))

    print(f"{db_path}")
    print(f"  rows with stale column energy: {len(mismatched)}")
    if dry_run or not mismatched:
        print("  DRY RUN: not modified." if dry_run else "  Nothing to do.")
        return

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_name(f"{db_path.name}.bak_sync_{stamp}")
        shutil.copy2(db_path, backup_path)
        print(f"  Backup: {backup_path}")

    with connect(str(db_path)) as db:
        for row_id, data_energy in mismatched:
            atoms = db.get(row_id).toatoms()
            atoms.calc.results["energy"] = data_energy
            if "free_energy" in atoms.calc.results:
                atoms.calc.results["free_energy"] = data_energy
            db.update(row_id, atoms=atoms)
    print(f"  Synced {len(mismatched)} rows (column <- data['energy']).")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for db_str in [d for group in args.db for d in group]:
        db_path = Path(db_str)
        if not db_path.exists():
            logger.error("DB not found: %s", db_path)
            sys.exit(1)
        repair_db(db_path, args.dry_run, not args.no_backup)


if __name__ == "__main__":
    main()
