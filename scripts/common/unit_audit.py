#!/usr/bin/env python3
"""Unit-consistency audit for CP2K-derived datasets and ASE databases.

Rationale
---------
This project's CP2K ingest chain once stored raw Hartree / Hartree-per-bohr
values into ASE databases whose metadata declared eV / eV-per-Angstrom (see
``scripts/common/fix_db_units_to_ev.py``). To prevent recurrence, every collect
path calls :func:`audit_frame_values` on freshly parsed frames *before* writing
them to a database; if the audit fails, nothing is written and the collect
aborts. :func:`audit_db_units` provides the same check for an existing DB and
is exposed as a CLI for manual inspection.

Heuristic
---------
All datasets built by this project store *raw CP2K total energies* (no atomic
reference subtraction). In eV, pseudopotential total energies are of order
10^2–10^3 eV/atom (e.g. carbon GTH-PBE ≈ −154 eV/atom, SiO₂ ≈ −328 eV/atom).
Values of order −1…−20 eV/atom therefore indicate energies still in Hartree.
The force check is looser: genuine near-equilibrium datasets may legitimately
have small forces, so a suspiciously small median max|F| only downgrades the
report to a warning unless ``strict`` is enabled.

CLI
---
.. code-block:: bash

    uv run python scripts/common/unit_audit.py \
        --db amorphous_carbon_al_build/amorphous_carbon_dataset_stress_merged.db
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.db import connect

# Raw CP2K total energies in eV are never this small per atom; below this
# threshold the values are almost certainly still in Hartree.
DEFAULT_MIN_ABS_E_PER_ATOM_EV = 20.0
DEFAULT_MAX_ABS_E_PER_ATOM_EV = 5000.0
# Median max|F| below this suggests forces are still in Ha/bohr (factor ~51
# smaller than eV/Angstrom). Warning only: relaxed datasets can be quieter.
DEFAULT_MIN_MEDIAN_MAX_FORCE_EV_ANG = 0.5
DEFAULT_MAX_FORCE_EV_ANG = 1.0e5


@dataclass
class UnitAuditReport:
    ok: bool
    n_frames: int
    median_abs_e_per_atom: float | None = None
    median_max_force: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"frames={self.n_frames}  "
            f"median |E/atom|={self.median_abs_e_per_atom:.3f}  "
            f"median max|F|={self.median_max_force:.4f}"
            if self.median_abs_e_per_atom is not None and self.median_max_force is not None
            else f"frames={self.n_frames}",
        ]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        lines += [f"  ERROR:   {e}" for e in self.errors]
        lines.append("  verdict: " + ("OK" if self.ok else "FAILED"))
        return "\n".join(lines)


def audit_frame_values(
    energies: list[float],
    natoms: list[int],
    forces: list[np.ndarray],
    *,
    min_abs_e_per_atom: float = DEFAULT_MIN_ABS_E_PER_ATOM_EV,
    max_abs_e_per_atom: float = DEFAULT_MAX_ABS_E_PER_ATOM_EV,
    min_median_max_force: float = DEFAULT_MIN_MEDIAN_MAX_FORCE_EV_ANG,
    max_force: float = DEFAULT_MAX_FORCE_EV_ANG,
    strict: bool = False,
) -> UnitAuditReport:
    """Audit freshly parsed CP2K frame values (must already be in eV, eV/Å)."""
    report = UnitAuditReport(ok=True, n_frames=len(energies))
    if not energies:
        report.ok = False
        report.errors.append("no frames to audit")
        return report

    e_per_atom = np.abs(np.asarray(energies, dtype=float) / np.asarray(natoms, dtype=float))
    med_e = float(np.median(e_per_atom))
    report.median_abs_e_per_atom = med_e
    if not (min_abs_e_per_atom <= med_e <= max_abs_e_per_atom):
        report.ok = False
        report.errors.append(
            f"median |E/atom| = {med_e:.3f} outside [{min_abs_e_per_atom}, "
            f"{max_abs_e_per_atom}] eV — values are probably still in Hartree "
            f"(did a parser skip the Ha -> eV conversion?)"
        )

    max_forces = np.asarray(
        [float(np.max(np.linalg.norm(np.asarray(f, dtype=float), axis=1))) for f in forces]
    )
    med_f = float(np.median(max_forces))
    report.median_max_force = med_f
    if float(np.max(max_forces)) > max_force:
        msg = f"max|F| reaches {float(np.max(max_forces)):.1f} eV/Angstrom (limit {max_force})"
        if strict:
            report.ok = False
            report.errors.append(msg)
        else:
            report.warnings.append(msg)
    if med_f < min_median_max_force:
        msg = (
            f"median max|F| = {med_f:.4f} eV/Angstrom is suspiciously small — "
            f"forces may still be in Ha/bohr (or the frames are fully relaxed)"
        )
        if strict:
            report.ok = False
            report.errors.append(msg)
        else:
            report.warnings.append(msg)
    return report


def audit_db_units(
    db_path: str | Path,
    *,
    sample_limit: int = 1000,
    strict: bool = False,
    **kwargs,
) -> UnitAuditReport:
    """Audit the energy/forces units of an existing ASE database."""
    energies, natoms, forces = [], [], []
    with connect(str(db_path)) as db:
        for row in db.select(limit=sample_limit):
            try:
                energies.append(float(row.energy))
            except Exception:
                continue
            natoms.append(int(row.natoms))
            forces.append(np.asarray(row.forces, dtype=float))
    return audit_frame_values(energies, natoms, forces, strict=strict, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", nargs="+", action="append", required=True,
                   help="ASE DB path(s) to audit (repeatable flag).")
    p.add_argument("--sample-limit", type=int, default=1000)
    p.add_argument("--strict", action="store_true",
                   help="Treat force-magnitude warnings as failures too.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for db_str in [d for group in args.db for d in group]:
        report = audit_db_units(db_str, sample_limit=args.sample_limit, strict=args.strict)
        print(f"{db_str}")
        print(report.format())
        failed = failed or not report.ok
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
