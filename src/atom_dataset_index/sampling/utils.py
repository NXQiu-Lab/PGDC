"""Utility functions for structure sampling.

module_id: sampling.utils
module_version: 0.9.1
project_version: 1.1.0
last_updated: 2026-04-12
generation_mode: manual
source_files: [utils.py]
upstream_modules: [domain]
downstream_modules: [generator, relaxer]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from atom_dataset_index.domain import StructureRecord

if TYPE_CHECKING:
    from ase import Atoms


def pyxtal_to_record(crystal, sample_id: int = 0) -> StructureRecord:
    """Convert PyXtal crystal object to StructureRecord.

    Args:
        crystal: PyXtal crystal object
        sample_id: Sample ID for the record

    Returns:
        StructureRecord representation
    """
    # Convert to ASE first (proper way to extract structure data)
    ase_atoms = crystal.to_ase()
    
    # Extract data from ASE atoms
    atomic_numbers = tuple(int(z) for z in ase_atoms.get_atomic_numbers())
    positions = tuple(
        (float(x), float(y), float(z)) for x, y, z in ase_atoms.get_positions()
    )

    # Get cell matrix
    cell = None
    if ase_atoms.cell is not None and ase_atoms.cell.rank == 3:
        cell_matrix = ase_atoms.cell.array
        cell = tuple(
            (
                float(cell_matrix[i][0]),
                float(cell_matrix[i][1]),
                float(cell_matrix[i][2]),
            )
            for i in range(3)
        )

    pbc = tuple(bool(p) for p in ase_atoms.pbc)

    # Get space group number if available
    sg_number = None
    if hasattr(crystal, "group") and hasattr(crystal.group, "number"):
        sg_number = crystal.group.number

    # Get volume if available
    volume = None
    if hasattr(crystal, "lattice") and hasattr(crystal.lattice, "volume"):
        volume = float(crystal.lattice.volume)

    return StructureRecord(
        sample_id=sample_id,
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
        energy=None,
        forces=(),
        metadata={
            "source": "pyxtal",
            "space_group": sg_number,
            "volume": volume,
            "formula": crystal.formula if hasattr(crystal, "formula") else None,
        },
    )


def ase_to_record(atoms: "Atoms", sample_id: int = 0) -> StructureRecord:
    """Convert ASE Atoms object to StructureRecord.

    Args:
        atoms: ASE Atoms object
        sample_id: Sample ID for the record

    Returns:
        StructureRecord representation
    """
    atomic_numbers = tuple(int(z) for z in atoms.get_atomic_numbers())
    positions = tuple(
        (float(x), float(y), float(z)) for x, y, z in atoms.get_positions()
    )

    # Handle cell - may be None for non-periodic systems
    cell = None
    if atoms.cell is not None and atoms.cell.rank == 3:
        cell_matrix = atoms.cell.array
        cell = tuple(
            (
                float(cell_matrix[i][0]),
                float(cell_matrix[i][1]),
                float(cell_matrix[i][2]),
            )
            for i in range(3)
        )

    pbc = tuple(bool(p) for p in atoms.pbc)

    # Extract energy/forces from attached calculator if available
    energy: float | None = None
    forces: tuple[Vector3, ...] = ()
    if atoms.calc is not None:
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            energy = None
        try:
            raw_forces = atoms.get_forces()
            forces = tuple(
                (float(f[0]), float(f[1]), float(f[2])) for f in raw_forces
            )
        except Exception:
            forces = ()

    return StructureRecord(
        sample_id=sample_id,
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
        energy=energy,
        forces=forces,
        metadata={"source": "ase"},
    )


def record_to_ase(record: StructureRecord) -> "Atoms":
    """Convert StructureRecord to ASE Atoms object.

    Args:
        record: StructureRecord to convert

    Returns:
        ASE Atoms object
    """
    from ase import Atoms

    # Handle cluster (0D) case - no cell
    if record.cell is None or not any(record.pbc):
        return Atoms(
            numbers=list(record.atomic_numbers),
            positions=list(record.positions) if record.positions else None,
            pbc=False,
        )

    return Atoms(
        numbers=list(record.atomic_numbers),
        positions=list(record.positions) if record.positions else None,
        cell=list(record.cell),
        pbc=record.pbc,
    )


def update_positions(record: StructureRecord, atoms: "Atoms") -> StructureRecord:
    """Create new record with updated positions from ASE Atoms.

    Args:
        record: Original StructureRecord
        atoms: ASE Atoms with new positions

    Returns:
        New StructureRecord with updated positions and cell
    """
    new_positions = tuple(
        (float(x), float(y), float(z)) for x, y, z in atoms.get_positions()
    )

    # Update cell if available
    new_cell = None
    if atoms.cell is not None and atoms.cell.rank == 3:
        cell_matrix = atoms.cell.array
        new_cell = tuple(
            (
                float(cell_matrix[i][0]),
                float(cell_matrix[i][1]),
                float(cell_matrix[i][2]),
            )
            for i in range(3)
        )

    return StructureRecord(
        sample_id=record.sample_id,
        atomic_numbers=record.atomic_numbers,
        positions=new_positions,
        cell=new_cell,
        pbc=tuple(bool(p) for p in atoms.pbc),
        energy=record.energy,
        forces=record.forces,
        metadata=dict(record.metadata),
    )


def compute_min_distance(record: StructureRecord) -> float:
    """Compute minimum interatomic distance in a structure.

    Args:
        record: Structure to analyze

    Returns:
        Minimum distance between any two atoms (Å)
    """
    if len(record.positions) < 2:
        return float("inf")

    min_dist = float("inf")
    positions = record.positions

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            min_dist = min(min_dist, dist)

    return min_dist


def basic_validation(record: StructureRecord, min_distance: float = 0.8) -> bool:
    """Perform basic validation on a structure.

    Checks:
    - Minimum interatomic distance >= min_distance
    - Cell volume is positive (if periodic)
    - Positions match atomic_numbers count

    Args:
        record: Structure to validate
        min_distance: Minimum allowed interatomic distance (Å)

    Returns:
        True if structure passes validation
    """
    # Check positions count
    if record.positions and len(record.positions) != record.natoms:
        return False

    # Check minimum distance
    if record.positions and len(record.positions) >= 2:
        if compute_min_distance(record) < min_distance:
            return False

    # Check cell volume for periodic systems
    if record.cell is not None and any(record.pbc):
        volume = compute_cell_volume(record.cell)
        if volume <= 0:
            return False
        # Reasonable volume per atom check
        min_volume_per_atom = 5.0  # Å³
        if volume < min_volume_per_atom * record.natoms:
            return False

    return True


def compute_cell_volume(cell: tuple[tuple[float, float, float], ...]) -> float:
    """Compute volume of a unit cell.

    Args:
        cell: 3x3 cell matrix (vectors as rows)

    Returns:
        Cell volume (Å³)
    """
    # Volume = |a · (b × c)|
    a = cell[0]
    b = cell[1]
    c = cell[2]

    # Cross product b × c
    bc_x = b[1] * c[2] - b[2] * c[1]
    bc_y = b[2] * c[0] - b[0] * c[2]
    bc_z = b[0] * c[1] - b[1] * c[0]

    # Dot product a · (b × c)
    volume = abs(a[0] * bc_x + a[1] * bc_y + a[2] * bc_z)

    return volume


def perturb_atoms(atoms: "Atoms", std: float = 0.05) -> "Atoms":
    """Add random perturbation to atomic positions.

    Args:
        atoms: ASE Atoms object
        std: Standard deviation for perturbation (Å)

    Returns:
        Perturbed Atoms object (modified in place)
    """
    import numpy as np

    positions = atoms.get_positions()
    noise = np.random.normal(0, std, positions.shape)
    atoms.set_positions(positions + noise)
    return atoms


def enumerate_miller_indices(max_index: int) -> list[tuple[int, int, int]]:
    """Enumerate Miller indices up to max_index.

    Generates unique Miller indices (hkl) where at least one index
    is equal to max_index and others are <= max_index.

    Args:
        max_index: Maximum Miller index value

    Returns:
        List of (h, k, l) tuples
    """
    indices = []
    for h in range(max_index + 1):
        for k in range(max_index + 1):
            for l in range(max_index + 1):
                # Skip (0, 0, 0)
                if h == 0 and k == 0 and l == 0:
                    continue
                # Only include if max of indices equals current max_index
                # This avoids duplicates when iterating multiple max values
                if max(h, k, l) == max_index or max_index == 1:
                    indices.append((h, k, l))
    return indices


def parse_composition(formula: str) -> dict[str, int]:
    """Parse chemical formula string to composition dict.

    Args:
        formula: Chemical formula (e.g., "Hf4O8", "NaCl", "Au55")

    Returns:
        Dictionary of {element: count}

    Raises:
        ValueError: If formula cannot be parsed
    """
    import re

    # Pattern: Element (capital + optional lowercase) followed by optional count
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)

    composition = {}
    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        composition[element] = composition.get(element, 0) + count

    if not composition:
        raise ValueError(f"Could not parse chemical formula: {formula}")

    return composition


def composition_to_string(composition: dict[str, int]) -> str:
    """Convert composition dict to string representation.

    Args:
        composition: Dictionary of {element: count}

    Returns:
        Chemical formula string
    """
    parts = []
    for element, count in sorted(composition.items()):
        if count == 1:
            parts.append(element)
        else:
            parts.append(f"{element}{count}")
    return "".join(parts)
