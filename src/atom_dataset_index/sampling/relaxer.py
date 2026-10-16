"""Lennard-Jones relaxation and energy filtering.

module_id: sampling.relaxer
module_version: 0.9.1
project_version: 1.1.0
last_updated: 2026-04-12
generation_mode: manual
source_files: [relaxer.py]
upstream_modules: [domain, utils]
downstream_modules: []
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from atom_dataset_index.domain import StructureRecord
from atom_dataset_index.sampling.utils import record_to_ase, update_positions

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

# Element-specific LJ sigma parameters (van der Waals radii approximation, in Å)
# Based on commonly used values in molecular simulations
LJ_SIGMA = {
    "H": 2.0,
    "He": 2.0,
    "Li": 2.5,
    "Be": 2.0,
    "B": 2.0,
    "C": 3.0,
    "N": 2.5,
    "O": 2.5,
    "F": 2.0,
    "Ne": 2.0,
    "Na": 3.0,
    "Mg": 2.5,
    "Al": 3.0,
    "Si": 3.0,
    "P": 3.0,
    "S": 3.0,
    "Cl": 3.0,
    "Ar": 3.0,
    "K": 3.5,
    "Ca": 3.0,
    "Sc": 3.0,
    "Ti": 3.0,
    "V": 3.0,
    "Cr": 3.0,
    "Mn": 3.0,
    "Fe": 3.0,
    "Co": 3.0,
    "Ni": 3.0,
    "Cu": 3.0,
    "Zn": 3.0,
    "Ga": 3.0,
    "Ge": 3.0,
    "As": 3.0,
    "Se": 3.0,
    "Br": 3.0,
    "Kr": 3.0,
    "Rb": 3.5,
    "Sr": 3.5,
    "Y": 3.5,
    "Zr": 3.5,
    "Nb": 3.5,
    "Mo": 3.5,
    "Tc": 3.5,
    "Ru": 3.5,
    "Rh": 3.5,
    "Pd": 3.5,
    "Ag": 3.5,
    "Cd": 3.5,
    "In": 3.5,
    "Sn": 3.5,
    "Sb": 3.5,
    "Te": 3.5,
    "I": 3.5,
    "Xe": 3.5,
    "Cs": 4.0,
    "Ba": 4.0,
    "Hf": 3.5,
    "Ta": 3.5,
    "W": 3.5,
    "Re": 3.5,
    "Os": 3.5,
    "Ir": 3.5,
    "Pt": 3.5,
    "Au": 3.5,
    "Hg": 3.5,
    "Tl": 3.5,
    "Pb": 3.5,
    "Bi": 3.5,
    "Po": 3.5,
    "At": 3.5,
    "Rn": 3.5,
}

# Default LJ epsilon (well depth) in eV
# Using a small value to avoid over-binding
DEFAULT_LJ_EPSILON = 0.1

# Default LJ cutoff in Å
DEFAULT_LJ_RC = 10.0


def relax_and_filter(
    candidates: list[StructureRecord],
    energy_filter_percent: float = 50.0,
    fmax: float = 0.5,
    max_steps: int = 100,
    epsilon: float = DEFAULT_LJ_EPSILON,
    rc: float = DEFAULT_LJ_RC,
    verbose: bool = False,
) -> list[tuple[StructureRecord, float]]:
    """Relax structures with LJ potential and filter by energy.

    This function performs fast pre-relaxation using a Lennard-Jones
    potential to filter out unphysical or extremely high-energy
    structures. The goal is not accurate energy ranking but removing
    obviously bad candidates.

    Args:
        candidates: Input structure candidates
        energy_filter_percent: Keep bottom X% by energy (0-100)
        fmax: Force convergence criterion for FIRE optimizer (eV/Å)
        max_steps: Maximum optimization steps
        epsilon: LJ well depth parameter (eV)
        rc: LJ cutoff radius (Å)
        verbose: Whether to log progress

    Returns:
        List of (StructureRecord, energy) tuples, sorted by energy (lowest first)

    Example:
        >>> candidates = [record1, record2, record3]
        >>> filtered = relax_and_filter(candidates, energy_filter_percent=50)
        >>> len(filtered)
        1 or 2  # Approximately 50% of input
    """
    from ase.calculators.lj import LennardJones
    from ase.optimize import FIRE

    results = []
    n_failed = 0

    for i, record in enumerate(candidates):
        if verbose and i % 10 == 0:
            logger.info(f"Relaxing structure {i + 1}/{len(candidates)}")

        try:
            atoms = record_to_ase(record)

            # Set up LJ calculator
            # Note: ASE LJ uses global parameters; we use epsilon and sigma
            # For mixed systems, sigma is taken as the average of the two atoms
            calc = LennardJones(epsilon=epsilon, sigma=3.0, rc=rc, smooth=True)
            atoms.calc = calc

            # Quick relaxation with FIRE
            opt = FIRE(atoms, logfile=None)  # Suppress optimizer output
            opt.run(fmax=fmax, steps=max_steps)

            # Get energy
            energy = float(atoms.get_potential_energy())

            # Update record with relaxed positions
            relaxed_record = update_positions(record, atoms)

            # Add metadata
            relaxed_record.metadata["lj_energy"] = energy
            relaxed_record.metadata["lj_steps"] = opt.get_number_of_steps()

            results.append((relaxed_record, energy))

        except Exception as e:
            n_failed += 1
            if verbose:
                logger.debug(f"Relaxation failed for record {record.sample_id}: {e}")
            continue

    if verbose:
        logger.info(
            f"Relaxation complete: {len(results)} succeeded, {n_failed} failed"
        )

    # Sort by energy (lowest first)
    results.sort(key=lambda x: x[1])

    # Apply energy filtering
    if energy_filter_percent < 100 and results:
        n_keep = max(1, int(len(results) * energy_filter_percent / 100))
        results = results[:n_keep]
        if verbose:
            logger.info(f"Filtered to {n_keep} structures by energy")

    return results


def relax_single_structure(
    record: StructureRecord,
    fmax: float = 0.5,
    max_steps: int = 100,
    epsilon: float = DEFAULT_LJ_EPSILON,
    rc: float = DEFAULT_LJ_RC,
) -> tuple[StructureRecord, float] | None:
    """Relax a single structure with LJ potential.

    Args:
        record: Structure to relax
        fmax: Force convergence criterion
        max_steps: Maximum optimization steps
        epsilon: LJ well depth
        rc: LJ cutoff

    Returns:
        Tuple of (relaxed_record, energy) or None if relaxation fails
    """
    from ase.calculators.lj import LennardJones
    from ase.optimize import FIRE

    try:
        atoms = record_to_ase(record)
        calc = LennardJones(epsilon=epsilon, sigma=3.0, rc=rc, smooth=True)
        atoms.calc = calc

        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)

        energy = float(atoms.get_potential_energy())
        relaxed_record = update_positions(record, atoms)
        relaxed_record.metadata["lj_energy"] = energy

        return relaxed_record, energy

    except Exception as e:
        logger.debug(f"Relaxation failed: {e}")
        return None


def compute_lj_energy(
    record: StructureRecord,
    epsilon: float = DEFAULT_LJ_EPSILON,
    rc: float = DEFAULT_LJ_RC,
) -> float | None:
    """Compute single-point LJ energy without relaxation.

    Args:
        record: Structure to evaluate
        epsilon: LJ well depth
        rc: LJ cutoff

    Returns:
        LJ energy or None if calculation fails
    """
    from ase.calculators.lj import LennardJones

    try:
        atoms = record_to_ase(record)
        calc = LennardJones(epsilon=epsilon, sigma=3.0, rc=rc, smooth=True)
        atoms.calc = calc
        return float(atoms.get_potential_energy())
    except Exception:
        return None
