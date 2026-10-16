#!/usr/bin/env python3
"""Short-range repulsive wall calculator to stabilise PaiNN MD.

Physical motivation
-------------------
PaiNN (like any MLIP) is only trained on pair distances down to a certain
minimum (~1.0 Angstrom in the stress-merged amorphous-carbon dataset, with
99% of structures having no pair below 1.17 Angstrom).  During finite-
temperature MD, thermal fluctuations can push a pair of atoms below that
limit, where the model's forces are unconstrained and can turn spuriously
attractive, leading to catastrophic integration failure.

This module adds a physically motivated short-range repulsive pair potential
that is **exactly zero** (value and first derivative) above an onset distance
``r_on`` chosen just below the training-data lower limit, so the PaiNN
potential energy surface is untouched everywhere the model has data.  Below
``r_on`` the potential is a Born-Mayer exponential core repulsion,

    V(r) = A * exp(-r / rho) + B * r + C          (r < r_on)

which is the standard quantum-mechanical form for Pauli repulsion between
closed-shell atomic cores.  The linear/constant terms B, C enforce
V(r_on) = 0 and V'(r_on) = 0 (smooth switch-on, no force discontinuity).
The screening length rho (default 0.25 Angstrom, a typical core-repulsion
decay length for carbon) and the amplitude A are fixed by anchoring the
value of the potential to the ZBL (Ziegler-Biersack-Littmark) screened
Coulomb potential at ``r_match`` (0.8 Angstrom) — the universally used
description of nuclear repulsion in ion-collision / radiation-damage
simulations (SRIM standard).  The resulting wall is somewhat steeper than
raw ZBL between r_match and r_on, which is intentional: its job is to stop
runaway pairs decisively within a few integration steps.

Use together with a PaiNN calculator via ASE's LinearCombinationCalculator:

    from ase.calculators.mixing import LinearCombinationCalculator
    calc = LinearCombinationCalculator([painn_calc, RepulsiveWallCC()], [1.0, 1.0])
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress

# ZBL universal screening function coefficients (Ziegler, Biersack, Littmark,
# "The Stopping and Range of Ions in Solids", 1985).
_ZBL_PHI_COEFF = [0.1818, 0.5099, 0.2802, 0.02817]
_ZBL_PHI_EXP = [3.2, 0.9423, 0.4029, 0.2016]
_BOHR_ANGSTROM = 0.529177210903
_E2_EV_ANGSTROM = 14.3996454784255  # e^2 / (4 pi eps0) in eV * Angstrom


def zbl_energy(r: float, z1: float, z2: float) -> float:
    """ZBL screened-Coulomb pair energy (eV) at separation r (Angstrom)."""
    a = 0.8854 * _BOHR_ANGSTROM / (z1 ** 0.23 + z2 ** 0.23)
    x = r / a
    phi = sum(c * np.exp(-e * x) for c, e in zip(_ZBL_PHI_COEFF, _ZBL_PHI_EXP))
    return z1 * z2 * _E2_EV_ANGSTROM / r * phi


def zbl_force(r: float, z1: float, z2: float) -> float:
    """ZBL repulsive force magnitude -dV/dr (eV/Angstrom)."""
    a = 0.8854 * _BOHR_ANGSTROM / (z1 ** 0.23 + z2 ** 0.23)
    x = r / a
    phi = sum(c * np.exp(-e * x) for c, e in zip(_ZBL_PHI_COEFF, _ZBL_PHI_EXP))
    dphi = sum(-c * e / a * np.exp(-e * x) for c, e in zip(_ZBL_PHI_COEFF, _ZBL_PHI_EXP))
    pref = z1 * z2 * _E2_EV_ANGSTROM
    return -(pref * (-phi / r ** 2 + dphi / r))


def _solve_born_mayer(
    r_on: float,
    r_match: float,
    rho: float,
    z1: float,
    z2: float,
) -> tuple[float, float, float, float]:
    """Solve A, B, C of V(r) = A exp(-r/rho) + B r + C for a given rho.

    Constraints:
      V(r_on) = 0, V'(r_on) = 0            (smooth switch-on)
      V(r_match) = ZBL(r_match)            (anchor to ZBL value)

    All three conditions are linear in A once rho is fixed, so the solution
    is analytic.  (Matching the ZBL slope as well overconstrains this
    functional form and has no solution for C-C; a value-anchored wall with a
    fixed, physically typical screening length is the standard compromise.)
    """
    v_m = zbl_energy(r_match, z1, z2)
    e_on = np.exp(-r_on / rho)
    e_m = np.exp(-r_match / rho)
    denom = e_m + e_on * (r_match - r_on) / rho - e_on
    if denom <= 0.0:
        raise ValueError(
            f"Born-Mayer solve degenerate for r_on={r_on}, r_match={r_match}, rho={rho}"
        )
    A = v_m / denom
    B = A * e_on / rho
    C = -A * e_on * (1.0 + r_on / rho)
    return A, rho, B, C


class RepulsiveWallCC(Calculator):
    """C-C short-range repulsive wall (Born-Mayer anchored to ZBL).

    Parameters
    ----------
    r_on : float
        Onset distance in Angstrom.  The potential is exactly zero (with zero
        force) for all pairs at r >= r_on.  Must be below the minimum pair
        distance present in the MLIP training data so the training domain is
        untouched.
    r_match : float
        Distance where the Born-Mayer form is anchored to the ZBL potential.
    rho : float
        Born-Mayer screening length in Angstrom (default 0.25, a typical
        core-repulsion decay length for carbon).
    z1, z2 : float
        Atomic numbers (default carbon-carbon).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        r_on: float = 1.10,
        r_match: float = 0.80,
        rho: float = 0.25,
        z1: float = 6.0,
        z2: float = 6.0,
        skin: float = 0.3,
    ) -> None:
        super().__init__()
        self.r_on = float(r_on)
        self.r_match = float(r_match)
        self.z1 = float(z1)
        self.z2 = float(z2)
        self.skin = float(skin)
        self.A, self.rho, self.B, self.C = _solve_born_mayer(
            self.r_on, self.r_match, float(rho), self.z1, self.z2
        )

    def pair_energy(self, r: np.ndarray) -> np.ndarray:
        """Pair energy V(r); exactly zero for r >= r_on."""
        v = self.A * np.exp(-r / self.rho) + self.B * r + self.C
        return np.where(r < self.r_on, v, 0.0)

    def pair_dvdr(self, r: np.ndarray) -> np.ndarray:
        """dV/dr; exactly zero for r >= r_on."""
        dv = -self.A / self.rho * np.exp(-r / self.rho) + self.B
        return np.where(r < self.r_on, dv, 0.0)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        natoms = len(self.atoms)

        nl = NeighborList(
            [self.r_on / 2.0 + self.skin / 2.0] * natoms,
            self_interaction=False,
            bothways=False,
            skin=self.skin,
        )
        nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        energy = 0.0
        forces = np.zeros((natoms, 3))
        virial = np.zeros((3, 3))

        for ii in range(natoms):
            neighbors, offsets = nl.get_neighbors(ii)
            if len(neighbors) == 0:
                continue
            cells = offsets @ cell
            d_vecs = positions[neighbors] + cells - positions[ii]  # i -> j
            r = np.linalg.norm(d_vecs, axis=1)
            active = r < self.r_on
            if not np.any(active):
                continue
            d_vecs = d_vecs[active]
            r = r[active]

            energy += self.pair_energy(r).sum()
            # force on i: -dV/dr_i = dV/dr * r_hat_ij  (repulsive: pushes i away from j)
            f_scalars = self.pair_dvdr(r) / r  # du/dr / r
            f_vecs = f_scalars[:, None] * d_vecs
            forces[ii] += f_vecs.sum(axis=0)
            # virial following ASE's LennardJones convention (stress = W / V).
            virial += f_vecs.T @ d_vecs

        self.results["energy"] = float(energy)
        self.results["forces"] = forces
        if self.atoms.cell.rank == 3:
            self.results["stress"] = (
                full_3x3_to_voigt_6_stress(virial) / self.atoms.get_volume()
            )
        else:
            self.results["stress"] = np.zeros(6)


if __name__ == "__main__":
    # Quick numeric validation: smoothness at r_on, ZBL anchoring, dimer curve.
    wall = RepulsiveWallCC()
    print(f"Fitted Born-Mayer: A={wall.A:.4f} eV, rho={wall.rho:.4f} A, "
          f"B={wall.B:.4f} eV/A, C={wall.C:.4f} eV")
    print(f"{'r (A)':>7} {'V_wall (eV)':>12} {'F_wall (eV/A)':>14} {'V_ZBL (eV)':>11}")
    for r in [0.60, 0.70, 0.80, 0.90, 1.00, 1.05, 1.099, 1.10, 1.15, 1.30, 1.42]:
        v = float(wall.pair_energy(np.array([r]))[0])
        f = float(-wall.pair_dvdr(np.array([r]))[0])
        print(f"{r:7.3f} {v:12.4f} {f:14.4f} {zbl_energy(r, 6, 6):11.4f}")
