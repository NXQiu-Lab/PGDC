"""Reader for VASP POSCAR format structures.

This module provides utilities to read VASP POSCAR files and convert them
to the internal StructureRecord representation used by the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atom_dataset_index.domain import StructureRecord


class PoscarReadError(RuntimeError):
    """Exception raised for POSCAR read failures."""


class PoscarReader:
    """Read VASP POSCAR files into StructureRecord objects.

    This reader uses ASE as the underlying parser for robustness, but converts
    the output to the package's internal StructureRecord representation for
    consistency with other dataset sources.
    """

    def __init__(self) -> None:
        """Initialize the reader."""
        self._check_ase_available()

    def read(self, path: str | Path, *, sample_id: int = 0) -> StructureRecord:
        """Read a single POSCAR file into a StructureRecord.

        Args:
            path: Path to the POSCAR file.
            sample_id: Sample ID to assign to the structure. Defaults to 0.

        Returns:
            StructureRecord representing the structure.

        Raises:
            PoscarReadError: If the file cannot be read or parsed.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"POSCAR file not found: {path}")

        try:
            from ase import Atoms
            from ase.io import read
        except ImportError as error:
            raise PoscarReadError(
                "ASE is required to read POSCAR files. "
                "Install with: pip install ase"
            ) from error

        try:
            atoms: Atoms = read(str(path), format="vasp")
        except Exception as error:
            raise PoscarReadError(f"Failed to parse POSCAR file: {path}") from error

        return self._atoms_to_record(atoms, sample_id=sample_id)

    def read_directory(
        self,
        directory: str | Path,
        pattern: str = "POSCAR*",
        sort: bool = True,
    ) -> list[StructureRecord]:
        """Read all POSCAR files in a directory.

        Args:
            directory: Path to the directory containing POSCAR files.
            pattern: Glob pattern to match POSCAR files. Defaults to "POSCAR*".
            sort: Whether to sort files by name before reading. Defaults to True.

        Returns:
            List of StructureRecord objects, one per file.

        Raises:
            PoscarReadError: If the directory cannot be read.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        files = list(directory.glob(pattern))
        if sort:
            files.sort()

        records: list[StructureRecord] = []
        for sample_id, file_path in enumerate(files):
            try:
                record = self.read(file_path, sample_id=sample_id)
                # Store the original path in metadata for traceability
                record.metadata["poscar_path"] = str(file_path)
                record.metadata["poscar_filename"] = file_path.name
                records.append(record)
            except PoscarReadError:
                # Log warning but continue with other files
                continue

        return records

    def _atoms_to_record(self, atoms: Any, *, sample_id: int) -> StructureRecord:
        """Convert an ASE Atoms object to StructureRecord.

        Args:
            atoms: ASE Atoms object.
            sample_id: Sample ID to assign.

        Returns:
            StructureRecord representation.
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

        # PBC flags
        pbc = tuple(bool(p) for p in atoms.pbc)

        # Energy and forces are not typically in POSCAR, set to None
        # They will come from subsequent AIMD calculations
        return StructureRecord(
            sample_id=sample_id,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
            energy=None,
            forces=(),
            metadata={"source_format": "poscar"},
        )

    @staticmethod
    def _check_ase_available() -> None:
        """Check if ASE is available.

        Raises:
            PoscarReadError: If ASE is not installed.
        """
        try:
            import ase  # noqa: F401
        except ImportError as error:
            raise PoscarReadError(
                "ASE is required to read POSCAR files. "
                "Install with: pip install ase"
            ) from error


def read_poscar(path: str | Path, *, sample_id: int = 0) -> StructureRecord:
    """Convenience function to read a single POSCAR file.

    Args:
        path: Path to the POSCAR file.
        sample_id: Sample ID to assign. Defaults to 0.

    Returns:
        StructureRecord representing the structure.
    """
    return PoscarReader().read(path, sample_id=sample_id)


def read_poscar_directory(
    directory: str | Path,
    pattern: str = "POSCAR*",
) -> list[StructureRecord]:
    """Convenience function to read all POSCAR files in a directory.

    Args:
        directory: Path to the directory.
        pattern: Glob pattern to match files. Defaults to "POSCAR*".

    Returns:
        List of StructureRecord objects.
    """
    return PoscarReader().read_directory(directory, pattern=pattern)
