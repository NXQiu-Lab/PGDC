#!/usr/bin/env python3
"""Force-capping wrapper calculator for stabilising MLIP molecular dynamics.

Motivation
----------
Finite-temperature MD with the stress-merged amorphous-carbon PaiNN model
fails at 300 K after a few hundred steps.  Frame-by-frame analysis (see
outputs/wall_debug) showed the failure chain:

1. PaiNN develops a spurious *many-body* attraction for a collapsing local
   environment; its forces reach hundreds of eV/Angstrom while the closest
   pair is still at ~1.1 Angstrom.
2. A single Verlet step with such a force moves the pair by ~0.3 Angstrom,
   jumping straight across the repulsive core region.
3. The integrator explodes (forces ~1e15 eV/A, then NaN).

The model's core repulsion below ~0.9 Angstrom is actually finite and
strongly repulsive — the integrator simply never gets a chance to feel it.
Capping the per-atom force magnitude at a value far above anything seen in
normal dynamics (max |F| < 1.2 eV/A in 300 K equilibrium MD) leaves all
physical dynamics untouched while limiting the velocity kick during a
collapse event to ~0.02 A/fs, so the pair decelerates gradually into the
repulsive core and is reflected instead of being kicked through it.

Caveats
-------
- Force capping breaks energy conservation during capped events; use with a
  thermostat (NVT/NPT), not for NVE production.
- This is an integrator safeguard, not a model fix.  Configurations that
  trigger capping are off-distribution for the model and should be treated
  as active-learning candidates for retraining.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class ForceCappedCalculator(Calculator):
    """Wrap an ASE calculator, clipping per-atom forces and stress magnitude.

    Parameters
    ----------
    calc : Calculator
        The wrapped calculator (e.g. PaiNNCalculator).
    fmax_cap : float
        Maximum allowed per-atom force magnitude in eV/Angstrom.  Forces
        with larger norm are rescaled to this norm (direction preserved).
    stress_cap : float or None
        Maximum allowed absolute value of each stress-tensor component in
        eV/Angstrom^3 (protects barostats from stress spikes).  None keeps
        the wrapped stress unmodified.
    """

    def __init__(
        self,
        calc: Calculator,
        fmax_cap: float = 50.0,
        stress_cap: float | None = 3.0,
    ) -> None:
        super().__init__()
        self.calc = calc
        self.fmax_cap = float(fmax_cap)
        self.stress_cap = None if stress_cap is None else float(stress_cap)
        self.implemented_properties = list(calc.implemented_properties)
        self.n_capped_atoms = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        # Delegate fully to the wrapped calculator, then post-process.
        self.atoms = atoms.copy() if atoms is not None else self.atoms
        self.results = {}

        energy = self.calc.get_potential_energy(self.atoms)
        self.results["energy"] = energy

        forces = self.calc.get_forces(self.atoms)
        norms = np.linalg.norm(forces, axis=1)
        over = norms > self.fmax_cap
        self.n_capped_atoms = int(over.sum())
        if self.n_capped_atoms:
            forces = forces.copy()
            forces[over] *= (self.fmax_cap / norms[over])[:, None]
        self.results["forces"] = forces

        if "stress" in self.implemented_properties:
            stress = self.calc.get_stress(self.atoms)
            if self.stress_cap is not None:
                stress = np.clip(stress, -self.stress_cap, self.stress_cap)
            self.results["stress"] = stress
