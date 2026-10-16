#!/usr/bin/env python3
"""Sync relabeled energy/forces/stress from the AL dataset into comparison_12k.

Rows in ``amorphous_carbon_comparison_12k.db`` carry an ``al_frame_index``
keypoint: the corresponding row in ``amorphous_carbon_al_stress_dataset.db``
has ``id = al_frame_index + 1`` (verified for all 5403 mapped rows). After
``compute_stress_for_db.py`` relabels the AL dataset (``_sp_relabel`` marker
in ``row.data``), this script copies the new labels into the comparison DB —
both the reserved columns (via the Atoms calculator results) and the
``row.data`` copies that schnetpack reads.

Only rows whose source row already carries ``_sp_relabel`` are updated, so the
script can be re-run while collection is still in progress. A timestamped
backup of the destination DB is made once per day before the first write.

Usage::

    uv run python scripts/amorphous_carbon/sync_relabeled_labels.py \
        --src amorphous_carbon_al_build/amorphous_carbon_al_stress_dataset.db \
        --dst amorphous_carbon_al_build/amorphous_carbon_comparison_12k.db \
        [--dry-run]
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
    p.add_argument("--src", required=True, help="Source DB (relabeled AL dataset).")
    p.add_argument("--dst", required=True, help="Destination DB (comparison_12k).")
    p.add_argument("--dry-run", action="store_true", help="Report only, no writes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_path = str(Path(args.src).resolve())
    dst_path = Path(args.dst).resolve()

    # Collect dst rows that can be synced (have al_frame_index and a relabeled
    # source row).
    pending: list[tuple[int, int]] = []  # (dst_row_id, src_row_id)
    already = 0
    with connect(src_path) as src, connect(str(dst_path)) as dst:
        for row in dst.select():
            idx = row.key_value_pairs.get("al_frame_index")
            if idx is None:
                continue
            src_row = src.get(id=idx + 1)
            src_data = src_row.data or {}
            if "_sp_relabel" not in src_data:
                continue
            dst_data = row.data or {}
            if dst_data.get("_sp_relabel") == src_data["_sp_relabel"]:
                already += 1
                continue
            pending.append((row.id, idx + 1))

    print(f"rows to sync: {len(pending)} (already up to date: {already})")
    if args.dry_run or not pending:
        return

    day = datetime.now().strftime("%Y%m%d")
    if not list(dst_path.parent.glob(dst_path.name + f".bak_relabel_{day}*")):
        backup = dst_path.with_name(
            f"{dst_path.name}.bak_relabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(dst_path, backup)
        print(f"Backup: {backup}")

    n_synced = 0
    with connect(src_path) as src, connect(str(dst_path)) as dst:
        for dst_id, src_id in pending:
            src_row = src.get(id=src_id)
            dst_row = dst.get(id=dst_id)

            atoms = dst_row.toatoms()
            atoms.calc.results["energy"] = float(src_row.energy)
            atoms.calc.results["forces"] = np.asarray(src_row.forces, dtype=np.float64)
            if "free_energy" in atoms.calc.results:
                atoms.calc.results["free_energy"] = float(src_row.energy)

            src_data = src_row.data or {}
            data = dict(dst_row.data) if dst_row.data else {}
            for key in ("energy", "forces", "stress"):
                if key not in src_data:
                    continue
                value = np.asarray(src_data[key], dtype=np.float64)
                old = data.get(key)
                if old is not None:
                    value = value.reshape(np.asarray(old).shape)
                data[key] = value
            data["_sp_relabel"] = src_data["_sp_relabel"]
            dst.update(dst_id, atoms=atoms, data=data)
            n_synced += 1
            if n_synced % 500 == 0:
                print(f"  synced {n_synced}/{len(pending)}")

    print(f"Synced {n_synced} rows in {dst_path}")


if __name__ == "__main__":
    main()
