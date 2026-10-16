#!/usr/bin/env python3
"""Backfill ``row.data['energy'|'forces']`` from reserved columns.

Rows written before ``merge_trajectory_to_db`` stored data copies (or written
by paths that never did) have valid reserved columns (energy/forces) but an
empty ``row.data``. schnetpack's ``ASEAtomsData`` reads labels exclusively
from ``row.data`` and crashes with ``KeyError: 'energy'`` on such rows.

This script copies the reserved column values into ``row.data`` for rows
where the keys are missing, preserving the project conventions
(energy shape (1,), forces shape (n, 3), float64). A timestamped backup is
made before writing. Idempotent: rows that already have the keys are skipped.

Usage::

    uv run python scripts/common/backfill_row_data.py --db <path> [--db <path> ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.db import connect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", action="append", required=True, help="ASE DB path (repeatable).")
    p.add_argument("--dry-run", action="store_true", help="Report only, no writes.")
    return p.parse_args()


def backfill(db_path: Path, dry_run: bool) -> None:
    with connect(str(db_path)) as db:
        targets = [
            row.id
            for row in db.select()
            if not row.data or "energy" not in row.data or "forces" not in row.data
        ]

    print(f"{db_path}")
    print(f"  rows missing data['energy'/'forces']: {len(targets)}")
    if dry_run or not targets:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(f"{db_path.name}.bak_datafill_{stamp}")
    shutil.copy2(db_path, backup)
    print(f"  Backup: {backup}")

    with connect(str(db_path)) as db:
        for row_id in targets:
            row = db.get(id=row_id)
            data = dict(row.data) if row.data else {}
            if "energy" not in data:
                data["energy"] = np.array([row.energy], dtype=np.float64)
            if "forces" not in data:
                data["forces"] = np.asarray(row.forces, dtype=np.float64)
            db.update(row_id, data=data)
    print(f"  Backfilled {len(targets)} rows.")


def main() -> None:
    args = parse_args()
    for db in args.db:
        backfill(Path(db), args.dry_run)


if __name__ == "__main__":
    main()
