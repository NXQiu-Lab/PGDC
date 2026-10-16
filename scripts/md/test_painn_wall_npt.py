#!/usr/bin/env python3
"""Stability test: PaiNN + short-range repulsive wall vs plain PaiNN.

Re-runs the scenarios in which the plain PaiNN model blew up at 0 GPa /
300 K (see outputs/amorphous_carbon_density_sp3_stress_merged):

  1. NVT, 300 K, 0.5 fs (plain PaiNN went NaN at ~step 700)
  2. NPT Berendsen, 0 GPa, 0.5 fs, taup=500 fs (plain PaiNN failed at
     step 717 with "SVD did not converge")

Usage:

    uv run python scripts/md/test_painn_wall_npt.py \
        --model-path "training_runs 2/amorphous_carbon_dataset_stress_merged_20260729_160842/best_inference_model" \
        --db-path amorphous_carbon_al_build/amorphous_carbon_dataset_stress_merged.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.db import connect
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.calculators.mixing import LinearCombinationCalculator

_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from painn_calculator import PaiNNCalculator
from repulsive_wall import RepulsiveWallCC
from force_cap import ForceCappedCalculator

GPA_TO_EV_PER_ANGSTROM3 = 1.0 / 160.2176567
BAR_TO_EV_PER_ANGSTROM3 = 1.0e-4 * GPA_TO_EV_PER_ANGSTROM3
INV_BAR_TO_ANGSTROM3_PER_EV = 1.0 / BAR_TO_EV_PER_ANGSTROM3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--db-path", required=True)
    p.add_argument("--db-id", type=int, default=15515,
                   help="ASE DB row id of the initial structure.")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--r-on", type=float, default=1.10,
                   help="Repulsive wall onset distance in Angstrom.")
    p.add_argument("--fmax-cap", type=float, default=50.0,
                   help="Per-atom force cap in eV/Angstrom applied to the "
                        "PaiNN forces (0 disables capping).")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--nsteps", type=int, default=5000)
    p.add_argument("--timestep", type=float, default=0.5)
    p.add_argument("--mode", choices=["nvt", "npt", "both"], default="both")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("wall-test")

    painn = PaiNNCalculator(
        model_path=args.model_path, device=args.device,
        cutoff=args.cutoff, calc_stress=True,
    )
    if args.fmax_cap > 0.0:
        painn = ForceCappedCalculator(painn, fmax_cap=args.fmax_cap)
        log.info("PaiNN forces capped at %.1f eV/A per atom", args.fmax_cap)
    wall = RepulsiveWallCC(r_on=args.r_on)
    log.info(
        "Wall: A=%.2f eV rho=%.3f A r_on=%.2f A (zero above r_on)",
        wall.A, wall.rho, wall.r_on,
    )
    combo = LinearCombinationCalculator([painn, wall], [1.0, 1.0])

    with connect(args.db_path) as db:
        atoms0 = db.get(id=args.db_id).toatoms()
    atoms0.set_pbc(True)

    def probe(atoms, tag):
        temp = atoms.get_temperature()
        max_f = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
        stress = atoms.get_stress(voigt=False)
        press = -np.trace(stress) / 3.0 * 160.2176567
        n_capped = getattr(painn, "n_capped_atoms", 0)
        log.info(
            "%s: T=%.0f K maxF=%.2f eV/A P=%.2f GPa vol=%.1f A^3 capped_atoms=%d",
            tag, temp, max_f, press, atoms.get_volume(), n_capped,
        )

    def fresh_atoms():
        a = atoms0.copy()
        a.calc = combo
        MaxwellBoltzmannDistribution(a, temperature_K=args.temperature,
                                     rng=np.random.default_rng(42))
        Stationary(a)
        return a

    if args.mode in ("nvt", "both"):
        a = fresh_atoms()
        dyn = NVTBerendsen(a, timestep=args.timestep * units.fs,
                           temperature_K=args.temperature, taut=100 * units.fs)
        dyn.attach(lambda: probe(a, "NVT+wall"), interval=250)
        try:
            dyn.run(args.nsteps)
            probe(a, "NVT+wall final")
            log.info("NVT+wall result: SURVIVED %d steps", args.nsteps)
        except Exception as exc:
            log.info("NVT+wall result: FAILED at step %d: %s",
                     dyn.get_number_of_steps(), exc)

    if args.mode in ("npt", "both"):
        a = fresh_atoms()
        dyn = NPTBerendsen(
            a,
            timestep=args.timestep * units.fs,
            temperature_K=args.temperature,
            taut=100 * units.fs,
            pressure_au=0.0,
            taup=500 * units.fs,
            compressibility_au=1.5e-6 * INV_BAR_TO_ANGSTROM3_PER_EV,
        )
        dyn.attach(lambda: probe(a, "NPT+wall"), interval=250)
        try:
            dyn.run(args.nsteps)
            probe(a, "NPT+wall final")
            log.info("NPT+wall result: SURVIVED %d steps", args.nsteps)
        except Exception as exc:
            log.info("NPT+wall result: FAILED at step %d: %s",
                     dyn.get_number_of_steps(), exc)


if __name__ == "__main__":
    main()
