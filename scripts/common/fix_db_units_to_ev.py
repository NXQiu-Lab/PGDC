#!/usr/bin/env python3
"""Fix mislabeled units in PGDC-pipeline ASE databases: Ha -> eV, Ha/bohr -> eV/Angstrom.

Background
----------
The CP2K ingest chain (``parse_cp2k_ener`` / ``parse_cp2k_trajectory`` in
``scripts/common/cp2k_utils.py``) stores raw CP2K output values: energies in
Hartree and forces in Hartree/bohr, while the DB metadata declares eV and
eV/Angstrom. This script converts the ``energy`` and ``forces`` properties —
both the ASE DB columns and the ``row.data`` copies that schnetpack actually
reads during training — to true eV and eV/Angstrom.

``stress`` is NOT touched: it was backfilled with proper GPa -> eV/Angstrom^3
conversion (``backfill_stress_from_raw.py`` / ``compute_stress_for_db.py``) and
is already correct.

Rows that are already in eV (e.g. ``source=mlip4_al`` rows imported via the
mlip4_al_baseline pipeline, whose parser converts Ha -> eV) must be excluded
with ``--exclude-source mlip4_al``.

A metadata marker ``_unit_fix_ev`` is written after conversion; re-running the
script on an already-fixed DB is a no-op. The original DB file is backed up as
``<db>.bak_YYYYMMDD_HHMMSS`` before any write (unless ``--no-backup``).

Example
-------
.. code-block:: bash

    # dry-run inspection
    uv run python scripts/common/fix_db_units_to_ev.py \
        --db amorphous_carbon_al_build/amorphous_carbon_dataset_stress_merged.db \
        --dry-run

    # mixed-unit DB: skip rows that are already in eV
    uv run python scripts/common/fix_db_units_to_ev.py \
        --db amorphous_carbon_al_build/amorphous_carbon_comparison_12k.db \
        --exclude-source mlip4_al
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.db import connect

logger = logging.getLogger("fix-db-units")

HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = 51.42208619083232

MARKER_KEY = "_unit_fix_ev"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", nargs="+", action="append", required=True,
                   help="ASE DB path(s) to fix (repeatable flag).")
    p.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Skip rows whose 'source' key-value pair matches this value "
        "(repeatable). Use for rows that are already in eV, e.g. mlip4_al.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report only; do not modify.")
    p.add_argument("--no-backup", action="store_true", help="Skip the .bak file backup.")
    return p.parse_args()


def _row_stats(energies: list[float], max_forces: list[float]) -> str:
    if not energies:
        return "n=0"
    e = np.asarray(energies)
    f = np.asarray(max_forces)
    return (
        f"n={len(e)}  E/atom p50={np.median(e):.4f} (range {e.min():.3f}..{e.max():.3f})  "
        f"maxF p50={np.median(f):.4f} (p99 {np.percentile(f, 99):.3f})"
    )


def fix_db(db_path: Path, exclude_sources: set[str], dry_run: bool, backup: bool) -> None:
    with connect(str(db_path)) as db:
        metadata = db.metadata
        if isinstance(metadata, dict) and MARKER_KEY in metadata:
            print(f"SKIP {db_path}: marker {MARKER_KEY} present, already fixed")
            return

        selected: list = []
        skipped = 0
        for row in db.select():
            src = row.key_value_pairs.get("source")
            if src is not None and str(src) in exclude_sources:
                skipped += 1
                continue
            selected.append(row)

    before_e, before_f = [], []
    for row in selected:
        before_e.append(row.energy / row.natoms)
        before_f.append(float(np.max(np.linalg.norm(row.forces, axis=1))))
    print(f"{db_path}")
    print(f"  selected {len(selected)} rows, skipped {skipped} (exclude-source: {sorted(exclude_sources) or 'none'})")
    print(f"  BEFORE: {_row_stats(before_e, before_f)}")
    print(f"  AFTER (projected): {_row_stats([e * HA_TO_EV for e in before_e], [f * HA_BOHR_TO_EV_ANG for f in before_f])}")

    if dry_run:
        print("  DRY RUN: not modified.")
        return

    if not selected:
        print("  Nothing to do.")
        return

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_name(f"{db_path.name}.bak_{stamp}")
        shutil.copy2(db_path, backup_path)
        print(f"  Backup: {backup_path}")

    with connect(str(db_path)) as db:
        for row in selected:
            new_energy = float(row.energy) * HA_TO_EV
            new_forces = np.asarray(row.forces, dtype=np.float64) * HA_BOHR_TO_EV_ANG
            # energy/forces are reserved row fields encoded via the Atoms
            # calculator results, so they must be rewritten through atoms=.
            atoms = row.toatoms()
            atoms.calc.results["energy"] = new_energy
            atoms.calc.results["forces"] = new_forces
            if "free_energy" in atoms.calc.results:
                atoms.calc.results["free_energy"] = new_energy
            data_patch: dict = {}
            data = row.data
            if data:
                if "energy" in data:
                    data_patch["energy"] = np.asarray(data["energy"], dtype=np.float64) * HA_TO_EV
                if "forces" in data:
                    data_patch["forces"] = np.asarray(data["forces"], dtype=np.float64) * HA_BOHR_TO_EV_ANG
            if data_patch:
                db.update(row.id, atoms=atoms, data=data_patch)
            else:
                db.update(row.id, atoms=atoms)

        md = dict(db.metadata) if isinstance(db.metadata, dict) else {}
        md[MARKER_KEY] = json.dumps({
            "date": datetime.now().isoformat(timespec="seconds"),
            "converted": {"energy": "Ha -> eV", "forces": "Ha/bohr -> eV/Angstrom"},
            "exclude_source": sorted(exclude_sources),
            "n_rows": len(selected),
        })
        db.metadata = md

    print(f"  Converted {len(selected)} rows (energy x{HA_TO_EV:.6f}, forces x{HA_BOHR_TO_EV_ANG:.6f}).")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for db_str in [d for group in args.db for d in group]:
        db_path = Path(db_str)
        if not db_path.exists():
            logger.error("DB not found: %s", db_path)
            sys.exit(1)
        fix_db(db_path, set(args.exclude_source), args.dry_run, not args.no_backup)


if __name__ == "__main__":
    main()
