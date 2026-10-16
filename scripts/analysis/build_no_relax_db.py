#!/usr/bin/env python3
"""Build variant SiO2 datasets for ablation study.

Keeps:
  - phase_0_seed (all)
  - al_iter1 (all, single-phase MD)
  - al_iter2 (optional, drops relax if included)
  - al_iter3 heating/high_temp/quench/hold (drops relax)

Usage:
  uv run python scripts/analysis/build_no_relax_db.py
  uv run python scripts/analysis/build_no_relax_db.py --skip-iter2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.db import connect


def build_dataset(
    src_db_path: str | Path = "sio2_al_build/dataset.db",
    output_root: str | Path = "outputs/ablation_no_relax",
    skip_iter2: bool = False,
) -> Path:
    src_db_path = Path(src_db_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    suffix = "skip_iter2" if skip_iter2 else "no_relax"
    dst_db_path = output_root / f"dataset_{suffix}.db"

    if dst_db_path.exists():
        dst_db_path.unlink()

    kept = 0
    skipped = 0
    phase_counts: dict[str, int] = {}

    with connect(str(src_db_path)) as src, connect(str(dst_db_path), append=True) as dst:
        for row in src.select():
            run_name = row.key_value_pairs.get("run_name", "")
            phase = row.key_value_pairs.get("phase", "")

            if phase == "seed" or phase == "al_iter1":
                keep = True
            elif phase == "al_iter2":
                if skip_iter2:
                    keep = False
                else:
                    keep = not run_name.endswith("_relax")
            elif phase == "al_iter3":
                keep = not run_name.endswith("_relax")
            else:
                keep = True

            if keep:
                atoms = row.toatoms()
                kvp = dict(row.key_value_pairs)
                dst.write(atoms, **kvp)
                kept += 1
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            else:
                skipped += 1

    print(f"Built {dst_db_path}")
    print(f"  Kept:   {kept}")
    print(f"  Skipped: {skipped}")
    print(f"  By phase: {phase_counts}")
    return dst_db_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ablation datasets")
    parser.add_argument(
        "--skip-iter2",
        action="store_true",
        help="Also exclude all al_iter2 structures.",
    )
    args = parser.parse_args()
    build_dataset(skip_iter2=args.skip_iter2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
