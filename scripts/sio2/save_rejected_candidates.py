#!/usr/bin/env python3
"""
Save rejected candidate examples for cohort extraction.

Run this BEFORE prefilter_candidates() in an AL iteration to preserve
AB (Absurd) and LU (Locally-Unstable) structures that would otherwise be deleted.

用法：
    uv run python scripts/sio2/save_rejected_candidates.py \
        --build-dir sio2_al_build \
        --output-dir outputs/ne_cohort
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ase.io import read, write

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir.parent / "src"))
sys.path.insert(0, str(_scripts_dir))

from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.sampling import DeterministicPreFilter, ase_to_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Save rejected candidates for cohort.")
    parser.add_argument("--build-dir", default="sio2_al_build", help="Build directory")
    parser.add_argument("--output-dir", default="outputs/ne_cohort", help="Output directory")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    output_dir = Path(args.output_dir)
    db_path = build_dir / "dataset.db"

    rejected_dir = output_dir / "rejected_examples"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # Load reference dataset for PreFilter
    if not db_path.exists():
        print(f"ERROR: Reference DB not found: {db_path}")
        return 1

    print("Loading reference dataset...")
    dataset = AseDbDatasetReader(db_path).read()
    pf = DeterministicPreFilter(dataset)
    print(f"  Loaded {len(dataset.records)} structures")

    # Scan latest AL iteration candidates
    al_loop_dir = build_dir / "phase_1_al_loop"
    if not al_loop_dir.exists():
        print("ERROR: No AL iterations found.")
        return 1

    latest_iter = max(sorted(al_loop_dir.glob("iter_*")), key=lambda p: p.name)
    cand_dir = latest_iter / "candidates"

    if not cand_dir.exists():
        print(f"ERROR: No candidates directory: {cand_dir}")
        return 1

    print(f"\nScanning candidates in {cand_dir}...")

    saved = 0
    for poscar in sorted(cand_dir.glob("POSCAR_*")):
        try:
            atoms = read(str(poscar), format="vasp")
        except Exception as e:
            print(f"  SKIP {poscar.name}: read error ({e})")
            continue

        record = ase_to_record(atoms, sample_id=0)
        result = pf.classify(record)

        if result.label in ("AB", "LU"):
            out_name = f"{poscar.stem}_{result.label}.xyz"
            out_path = rejected_dir / out_name
            write(str(out_path), atoms)
            print(f"  SAVED {out_name}: {result.reason}")
            saved += 1

    print(f"\nDone. Saved {saved} rejected examples to {rejected_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
