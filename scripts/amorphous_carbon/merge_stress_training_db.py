#!/usr/bin/env python3
"""Merge stress-labelled rows into a single schnetpack-ready training DB.

Sources:
  * ``dataset.db`` rows from AL iteration 6 (stress backfilled from raw CP2K
    NPT outputs by ``backfill_stress_from_raw.py``).
  * ``amorphous_carbon_dataset_final_10000.db`` rows whose stress comes from
    the single-point campaign, EXCLUDING rows flagged
    ``labels_inconsistent=True`` (iteration-2 NPT rows whose stress was
    computed on an expanded cell; see ``fix_legacy_10000_db.py``).

All rows are written in schnetpack format directly: ``energy`` (1,),
``forces`` (n, 3) and ``stress`` (1, 3, 3) in ``row.data`` plus the
``_property_unit_dict`` metadata, so the result can be passed to
``painn.py --datapath`` (with ``--stress-weight > 0``) without going through
``convert_db_for_schnetpack.py`` (which requires all-or-nothing stress).

Example
-------

.. code-block:: bash

    uv run python scripts/amorphous_carbon/merge_stress_training_db.py \
        --build-dir amorphous_carbon_al_build
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))

import numpy as np
from ase.db import connect

logger = logging.getLogger(__name__)

SPK_METADATA = {
    "_distance_unit": "Angstrom",
    "_property_unit_dict": {
        "energy": "eV",
        "forces": "eV/Angstrom",
        "stress": "eV/Angstrom/Angstrom/Angstrom",
    },
    "atomrefs": {},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build-dir", type=str, default="amorphous_carbon_al_build")
    p.add_argument("--dataset-db", type=str, default=None,
                   help="Main dataset DB (default: <build-dir>/dataset.db)")
    p.add_argument("--legacy-db", type=str, default=None,
                   help="Legacy 10000 DB (default: <build-dir>/amorphous_carbon_dataset_final_10000.db)")
    p.add_argument("--out", type=str, default=None,
                   help="Output DB (default: <build-dir>/amorphous_carbon_dataset_stress_merged.db)")
    p.add_argument("--iteration", type=int, default=6,
                   help="Dataset DB iteration with backfilled stress")
    return p.parse_args()


def _row_to_schnetpack(row) -> tuple[dict, dict]:
    """Return (data, kvp) with energy/forces/stress in schnetpack shapes."""
    data = row.data if row.data else {}
    energy = data.get("energy")
    if energy is None:
        energy = getattr(row, "energy", None)
    forces = data.get("forces")
    if forces is None:
        forces = getattr(row, "forces", None)
    stress = data.get("stress")
    if energy is None or forces is None or stress is None:
        raise ValueError("row is missing energy/forces/stress")

    energy_arr = np.atleast_1d(np.asarray(energy, dtype=np.float64))
    forces_arr = np.asarray(forces, dtype=np.float64)
    stress_arr = np.asarray(stress, dtype=np.float64).reshape(1, 3, 3)

    kvp = dict(row.key_value_pairs) if row.key_value_pairs else {}
    kvp.pop("labels_inconsistent", None)
    return (
        {"energy": energy_arr, "forces": forces_arr, "stress": stress_arr},
        kvp,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    build_dir = Path(args.build_dir)
    dataset_db = Path(args.dataset_db) if args.dataset_db else build_dir / "dataset.db"
    legacy_db = Path(args.legacy_db) if args.legacy_db else build_dir / "amorphous_carbon_dataset_final_10000.db"
    out_db = Path(args.out) if args.out else build_dir / "amorphous_carbon_dataset_stress_merged.db"

    if out_db.exists():
        print(f"Output exists, removing: {out_db}")
        out_db.unlink()

    n_iter = n_legacy = n_skipped = 0
    with connect(str(out_db)) as out:
        out.metadata = SPK_METADATA

        with connect(str(dataset_db)) as db:
            for row in db.select(phase="al", iteration=args.iteration):
                data, kvp = _row_to_schnetpack(row)
                out.write(row.toatoms(), data=data, key_value_pairs=kvp)
                n_iter += 1

        with connect(str(legacy_db)) as db:
            for row in db.select():
                if row.key_value_pairs.get("labels_inconsistent"):
                    n_skipped += 1
                    continue
                data, kvp = _row_to_schnetpack(row)
                out.write(row.toatoms(), data=data, key_value_pairs=kvp)
                n_legacy += 1

    print(f"iter{args.iteration} rows from {dataset_db.name}: {n_iter}")
    print(f"legacy rows from {legacy_db.name}:      {n_legacy}")
    print(f"legacy rows skipped (inconsistent):     {n_skipped}")
    print(f"Total: {n_iter + n_legacy} -> {out_db}")


if __name__ == "__main__":
    main()
