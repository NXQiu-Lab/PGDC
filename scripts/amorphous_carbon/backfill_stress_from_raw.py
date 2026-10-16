#!/usr/bin/env python3
"""Backfill per-frame DFT stress tensors into an ASE DB from raw CP2K outputs.

Some AL iterations (e.g. iter_006, NPT high-density) were run with
``STRESS_TENSOR ANALYTICAL`` + ``&PRINT/&STRESS_TENSOR ON``, so every MD step's
stress tensor is available in the run's ``out`` file.  This script reconstructs
the step->stress mapping for each run, matches DB rows to (run, step) via
their stored potential energy (exact float64 match, positions as a
disambiguation fallback), and writes the stress tensor into
``row.data['stress']`` with shape (1, 3, 3).

Sign convention: CP2K prints the stress tensor with *positive = compressive*,
which is the opposite of the ASE / schnetpack convention (positive = tensile,
pressure = -trace(sigma)/3).  Verified empirically on iter_006: a run
equilibrated at a 150 GPa barostat shows trace(sigma)/3 = +149 GPa in the raw
output.  All tensors are therefore multiplied by -1 before being stored.

Example
-------

.. code-block:: bash

    uv run python scripts/amorphous_carbon/backfill_stress_from_raw.py \
        --build-dir amorphous_carbon_al_build --iteration 6 --dry-run

    uv run python scripts/amorphous_carbon/backfill_stress_from_raw.py \
        --build-dir amorphous_carbon_al_build --iteration 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))

import numpy as np
from ase.db import connect

from common.cp2k_utils import (
    parse_cp2k_ener,
    parse_cp2k_md_stress,
    parse_cp2k_trajectory,
)

logger = logging.getLogger(__name__)

# CP2K prints stress with positive = compressive; ASE/schnetpack use
# positive = tensile.  Multiply raw tensors by this factor before storage.
CP2K_TO_ASE_STRESS_SIGN = -1.0

EV_PER_ANGSTROM3_TO_GPA = 160.2176567


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build-dir", type=str, default="amorphous_carbon_al_build")
    p.add_argument("--db", type=str, default=None,
                   help="ASE DB path (default: <build-dir>/dataset.db)")
    p.add_argument("--iteration", type=int, default=6,
                   help="AL iteration whose frames should receive stress")
    p.add_argument("--dry-run", action="store_true",
                   help="Match rows and report, but do not modify the DB")
    return p.parse_args()


def load_run_targets(build_dir: Path, iteration: int) -> dict[str, dict[str, Any]]:
    """Map run directory name -> induction params from the AL manifest."""
    manifest_path = build_dir / "al_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    for it in manifest.get("al_iterations", []):
        if it.get("iteration") == iteration:
            return {
                Path(s["local_dir"]).name: s.get("induction_params", {})
                for s in it.get("submitted", [])
            }
    return {}


def build_energy_lookup(
    iter_dir: Path,
) -> tuple[dict[float, list[tuple[str, int]]], dict[str, dict[int, np.ndarray]], dict[str, float]]:
    """Parse every run in ``iter_dir``.

    Returns:
        energy_lookup: potential energy -> list of (run_name, step) candidates.
        stress_by_run: run_name -> {step: (3, 3) stress in eV/A^3, ASE conv.}.
        late_pressure_by_run: run_name -> mean hydrostatic pressure (GPa) over
            the last 25% of steps, for sanity reporting.
    """
    energy_lookup: dict[float, list[tuple[str, int]]] = {}
    stress_by_run: dict[str, dict[int, np.ndarray]] = {}
    late_pressure_by_run: dict[str, float] = {}

    for run_dir in sorted(iter_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        out_file = run_dir / "out"
        if not out_file.exists():
            logger.warning("No 'out' file in %s, skipping", run_dir)
            continue

        raw_stress = parse_cp2k_md_stress(out_file)
        if not raw_stress:
            logger.warning("No stress blocks in %s, skipping", run_dir)
            continue
        stresses = {
            step: tensor * CP2K_TO_ASE_STRESS_SIGN
            for step, tensor in raw_stress.items()
        }
        stress_by_run[run_dir.name] = stresses

        steps = sorted(stresses)
        late = steps[int(len(steps) * 0.75):]
        late_pressure_by_run[run_dir.name] = float(
            np.mean([-np.trace(stresses[k]) / 3.0 * EV_PER_ANGSTROM3_TO_GPA for k in late])
        )

        pos_files = sorted(run_dir.glob("*-pos-1.xyz"))
        if not pos_files:
            logger.warning("No trajectory in %s, skipping", run_dir)
            continue
        for pos_xyz in pos_files:
            stem = pos_xyz.name.replace("-pos-1.xyz", "")
            ener_file = run_dir / f"{stem}-1.ener"
            if not ener_file.exists():
                ener_file = run_dir / f"{stem}.ener"
            energy_records = parse_cp2k_ener(ener_file)
            energy_by_step = {int(r["step"]): r for r in energy_records}

            frames = parse_cp2k_trajectory(pos_xyz, None)
            n_mapped = 0
            for frame in frames:
                step = frame.get("step")
                if step is None:
                    continue
                record = energy_by_step.get(int(step))
                if record is None or int(step) not in stresses:
                    continue
                key = float(record["potential_energy"])
                energy_lookup.setdefault(key, []).append((run_dir.name, int(step)))
                n_mapped += 1
            print(f"  {run_dir.name}: {len(stresses)} stress blocks, "
                  f"{n_mapped}/{len(frames)} frames mapped "
                  f"(late pressure {late_pressure_by_run[run_dir.name]:+.1f} GPa)")

    return energy_lookup, stress_by_run, late_pressure_by_run


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    build_dir = Path(args.build_dir)
    db_path = Path(args.db) if args.db else build_dir / "dataset.db"
    iter_dir = build_dir / "phase_1_al_loop" / f"iter_{args.iteration:03d}"
    targets = load_run_targets(build_dir, args.iteration)

    print(f"DB:        {db_path}")
    print(f"Iteration: {args.iteration} ({iter_dir})")
    print(f"Mode:      {'DRY RUN' if args.dry_run else 'WRITE'}")

    energy_lookup, stress_by_run, late_pressure = build_energy_lookup(iter_dir)
    n_keys = len(energy_lookup)
    n_collisions = sum(1 for v in energy_lookup.values() if len(v) > 1)
    print(f"\nEnergy lookup: {n_keys} unique energies, {n_collisions} collisions")

    # Sanity report: late pressure vs barostat target.
    if targets:
        print("\nLate-trajectory hydrostatic pressure vs target:")
        for run_name in sorted(late_pressure):
            tgt = targets.get(run_name, {}).get("pressure")
            tgt_str = f"{tgt:6.1f}" if tgt is not None else "  ??? "
            print(f"  {run_name}: target {tgt_str} GPa, measured {late_pressure[run_name]:+7.1f} GPa")

    matched = 0
    unmatched: list[int] = []
    ambiguous: list[int] = []

    with connect(str(db_path)) as db:
        rows = list(db.select(phase="al", iteration=args.iteration))
        print(f"\nDB rows tagged phase=al iteration={args.iteration}: {len(rows)}")

        for row in rows:
            energy = row.get("energy")
            candidates = energy_lookup.get(float(energy)) if energy is not None else None
            if not candidates:
                unmatched.append(row.id)
                continue
            if len(candidates) > 1:
                # Disambiguate by comparing positions against the candidate frames.
                run_name, step = _disambiguate_by_positions(
                    iter_dir, candidates, np.asarray(row.positions)
                )
                if run_name is None:
                    ambiguous.append(row.id)
                    continue
            else:
                run_name, step = candidates[0]

            stress = stress_by_run[run_name][step]
            if not args.dry_run:
                db.update(row.id, data={"stress": stress[np.newaxis, :, :]})
            matched += 1

    print(f"\nMatched:   {matched}/{len(rows)}")
    if unmatched:
        print(f"Unmatched: {len(unmatched)} (row ids: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''})")
    if ambiguous:
        print(f"Ambiguous: {len(ambiguous)} (row ids: {ambiguous[:10]}{'...' if len(ambiguous) > 10 else ''})")
    if args.dry_run:
        print("Dry run: DB was not modified.")
    else:
        print(f"Stress written to {db_path} for {matched} rows.")


def _disambiguate_by_positions(
    iter_dir: Path,
    candidates: list[tuple[str, int]],
    row_positions: np.ndarray,
) -> tuple[str | None, int | None]:
    """Resolve an energy collision by comparing atomic positions."""
    for run_name, step in candidates:
        run_dir = iter_dir / run_name
        for pos_xyz in sorted(run_dir.glob("*-pos-1.xyz")):
            for frame in parse_cp2k_trajectory(pos_xyz, None):
                if frame.get("step") is None or int(frame["step"]) != step:
                    continue
                if np.allclose(frame["positions"], row_positions, atol=1e-4):
                    return run_name, step
    return None, None


if __name__ == "__main__":
    main()
