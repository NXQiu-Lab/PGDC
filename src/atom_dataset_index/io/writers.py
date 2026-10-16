"""Generic result writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atom_dataset_index.domain import StructureRecord


class DatasetWriter:
    """Write structured data to repository-friendly JSON files."""

    @staticmethod
    def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return output

    @staticmethod
    def write_text(path: str | Path, payload: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        return output

    @staticmethod
    def write_xyz(path: str | Path, record: StructureRecord) -> Path:
        """Write a single StructureRecord to an ASE extended XYZ file."""
        import ase.io
        from ase import Atoms

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        cell = list(record.cell) if record.cell is not None else None
        atoms = Atoms(
            numbers=list(record.atomic_numbers),
            positions=list(record.positions),
            cell=cell,
            pbc=list(record.pbc),
        )
        if record.energy is not None:
            atoms.info["energy"] = record.energy
        if record.forces:
            import numpy as np

            atoms.arrays["forces"] = np.asarray(record.forces, dtype=np.float64)
        ase.io.write(str(output), atoms, format="extxyz")
        return output


class ManifestWriter:
    """Write task manifests consumed by downstream presentation steps."""

    @staticmethod
    def write(path: str | Path, manifest: dict[str, Any]) -> Path:
        return DatasetWriter.write_json(path, manifest)
