#!/usr/bin/env python3
"""Fix legacy alignment errors in the amorphous carbon ASE databases.

Background
----------
Rows collected before the step-aware alignment fix in
``merge_trajectory_to_db`` (seed phase + AL iterations 1-4) were labelled by
*frame index* instead of *MD step*: frame ``i`` (MD step ``stride*i``) was
given the energy/temperature of MD step ``i``.  In addition, NPT rows from
iteration 2 stored the fixed step-0 cell instead of the per-step cell from
the CP2K ``out`` file.  Iteration 6 was collected with the corrected code and
is not affected.

This script re-derives the correct labels from the raw CP2K outputs by
matching DB rows to trajectory frames via their atomic positions (exact
float64 match) and updates:

* ``energy``     -> ``.ener`` potential energy at the frame's MD step
* ``temperature``-> ``.ener`` temperature at the frame's MD step (kvp)
* ``cell``       -> per-step cell parsed from ``out`` (falls back to the
                    fixed cell for NVT runs, where it is identical)

Forces are untouched: position/force frame pairing was always index-based on
the *trajectory* files and is correct.

Example
-------

.. code-block:: bash

    uv run python scripts/amorphous_carbon/fix_legacy_alignment.py \
        --build-dir amorphous_carbon_al_build --dry-run

    uv run python scripts/amorphous_carbon/fix_legacy_alignment.py \
        --build-dir amorphous_carbon_al_build
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))

import numpy as np
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator

from common.cp2k_utils import (
    _cellpar_to_matrix,
    parse_cp2k_cell,
    parse_cp2k_ener,
    parse_cp2k_md_cells,
    parse_cp2k_trajectory,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameLabel:
    energy: float
    temperature: float | None
    cell: np.ndarray | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build-dir", type=str, default="amorphous_carbon_al_build")
    p.add_argument("--db", type=str, default=None,
                   help="ASE DB path (default: <build-dir>/dataset.db)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without modifying the DB")
    return p.parse_args()


def build_label_lookup(build_dir: Path) -> dict[bytes, FrameLabel]:
    """Scan raw CP2K runs and map frame positions -> correct labels."""
    lut: dict[bytes, FrameLabel] = {}

    run_groups: list[tuple[Path, str, tuple[str, ...]]] = [
        # (glob root, run pattern, allowed trajectory stages or None for all)
        (build_dir / "phase_0_seed", "seed_*", ("quench", "relax")),
        (build_dir / "phase_1_al_loop" / "iter_001", "run_*", ()),
        (build_dir / "phase_1_al_loop" / "iter_002", "run_*", ()),
        (build_dir / "phase_1_al_loop" / "iter_003", "run_*", ()),
        (build_dir / "phase_1_al_loop" / "iter_004", "run_*", ()),
    ]

    for root, pattern, stages in run_groups:
        if not root.exists():
            logger.warning("Missing directory: %s", root)
            continue
        for run_dir in sorted(root.glob(pattern)):
            if not run_dir.is_dir():
                continue
            out_file = run_dir / "out"
            cells_by_step = parse_cp2k_md_cells(out_file) if out_file.exists() else {}
            fixed_cell = parse_cp2k_cell(out_file) if out_file.exists() else None

            for pos_xyz in sorted(run_dir.glob("*-pos-1.xyz")):
                stem = pos_xyz.name.replace("-pos-1.xyz", "")
                if stages and stem.rsplit("_", 1)[-1] not in stages:
                    continue
                ener_file = run_dir / f"{stem}-1.ener"
                if not ener_file.exists():
                    ener_file = run_dir / f"{stem}.ener"
                if not ener_file.exists():
                    logger.warning("No ener file for %s", pos_xyz)
                    continue
                records = parse_cp2k_ener(ener_file)
                rec_by_step = {int(r["step"]): r for r in records}

                n = 0
                for frame in parse_cp2k_trajectory(pos_xyz, None):
                    step = frame.get("step")
                    if step is None:
                        continue
                    rec = rec_by_step.get(int(step))
                    if rec is None:
                        continue
                    cellpar = cells_by_step.get(int(step), fixed_cell)
                    cell = _cellpar_to_matrix(cellpar) if cellpar is not None else None
                    key = np.ascontiguousarray(frame["positions"]).tobytes()
                    lut[key] = FrameLabel(
                        energy=float(rec["potential_energy"]),
                        temperature=(
                            float(rec["temperature"])
                            if rec.get("temperature") is not None else None
                        ),
                        cell=cell,
                    )
                    n += 1
                print(f"  {run_dir.parent.name}/{run_dir.name}/{stem}: {n} frames indexed")
    return lut


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    build_dir = Path(args.build_dir)
    db_path = Path(args.db) if args.db else build_dir / "dataset.db"
    print(f"DB: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")

    lut = build_label_lookup(build_dir)
    print(f"\nLookup built: {len(lut)} frames")

    stats = {
        "matched": 0,
        "unmatched": 0,
        "energy_changed": 0,
        "temperature_changed": 0,
        "cell_changed": 0,
    }
    unmatched_ids: list[int] = []

    with connect(str(db_path)) as db:
        # iter6 was collected with the fixed alignment; skip it.
        rows = [r for r in db.select() if r.key_value_pairs.get("iteration") != 6]
        print(f"Rows to check (all except iteration=6): {len(rows)}")

        for row in rows:
            key = np.ascontiguousarray(np.asarray(row.positions)).tobytes()
            label = lut.get(key)
            if label is None:
                stats["unmatched"] += 1
                unmatched_ids.append(row.id)
                continue
            stats["matched"] += 1

            e_old = float(row.energy)
            t_old = row.key_value_pairs.get("temperature")
            cell_old = np.asarray(row.cell)

            if abs(e_old - label.energy) > 1e-12:
                stats["energy_changed"] += 1
            if label.temperature is not None and t_old != label.temperature:
                stats["temperature_changed"] += 1
            cell_new = label.cell if label.cell is not None else cell_old
            if not np.allclose(cell_old, cell_new, atol=1e-10):
                stats["cell_changed"] += 1

            if args.dry_run:
                continue

            atoms = row.toatoms()
            atoms.set_cell(cell_new)
            atoms.set_pbc(True)
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=label.energy,
                forces=np.asarray(row.forces),
            )
            kvp_update = {}
            if label.temperature is not None:
                kvp_update["temperature"] = label.temperature
            db.update(row.id, atoms=atoms, **kvp_update)

    print(f"\nMatched:            {stats['matched']}")
    print(f"Unmatched:          {stats['unmatched']}")
    if unmatched_ids:
        print(f"  unmatched row ids (first 10): {unmatched_ids[:10]}")
    print(f"Energy corrected:     {stats['energy_changed']}")
    print(f"Temperature corrected:{stats['temperature_changed']}")
    print(f"Cell corrected:       {stats['cell_changed']}")
    if args.dry_run:
        print("Dry run: DB was not modified.")
    else:
        print(f"Updates written to {db_path}.")


if __name__ == "__main__":
    main()
