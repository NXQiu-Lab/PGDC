"""Descriptor adapters used by diversity-style metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import TYPE_CHECKING, Any

from atom_dataset_index.domain import StructureRecord

if TYPE_CHECKING:
    import numpy as np


class DescriptorAdapter(ABC):
    """Abstract adapter for converting structures into vector descriptors."""

    @abstractmethod
    def encode(self, record: StructureRecord) -> tuple[float, ...]:
        """Encode a structure as a numeric descriptor."""

    def encode_batch(self, records: list[StructureRecord]) -> "list[tuple[float, ...]]":
        """Encode multiple structures as numeric descriptors.
        
        Default implementation iterates over records. Subclasses may override
        with more efficient batch implementations.
        
        Args:
            records: List of structure records to encode.
            
        Returns:
            List of descriptor tuples, one per input record.
        """
        return [self.encode(record) for record in records]


class IdentityDescriptorAdapter(DescriptorAdapter):
    """Simple adapter based on flattened positions for bootstrap workflows."""

    def encode(self, record: StructureRecord) -> tuple[float, ...]:
        values: list[float] = []
        for vector in record.positions:
            values.extend(vector)
        if values:
            return tuple(values)
        return tuple(float(number) for number in record.atomic_numbers)


class SOAPDescriptorAdapter(DescriptorAdapter):
    """Structure-level descriptor with SOAP-first semantics and deterministic fallback.

    If a SOAP backend is not available in the environment, this adapter falls back to a
    fixed-length periodic radial fingerprint so v1 profiling can still run end-to-end.
    
    Supports both single-structure and batch encoding for improved performance.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        n_max: int = 4,
        l_max: int = 4,
        radial_bins: int = 12,
        include_cell: bool = True,
    ) -> None:
        self.cutoff = cutoff
        self.n_max = n_max
        self.l_max = l_max
        self.radial_bins = radial_bins
        self.include_cell = include_cell
        self.backend_name = "soap_fallback_radial"
        self._soap_builders: dict[tuple[int, ...], Any] = {}
        self._try_initialize_backend()

    def encode(self, record: StructureRecord) -> tuple[float, ...]:
        if not record.positions:
            return tuple(float(number) for number in record.atomic_numbers)
        soap_descriptor = self._encode_with_dscribe(record)
        if soap_descriptor is not None:
            return soap_descriptor

        histogram = [0.0] * self.radial_bins
        natoms = len(record.positions)
        if natoms > 1:
            for left_index in range(natoms):
                left = record.positions[left_index]
                for right_index in range(left_index + 1, natoms):
                    right = record.positions[right_index]
                    distance = math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in range(3)))
                    if distance > self.cutoff:
                        continue
                    bucket = min(int(distance / self.cutoff * self.radial_bins), self.radial_bins - 1)
                    histogram[bucket] += 1.0

        scale = max(natoms * max(natoms - 1, 1) / 2.0, 1.0)
        normalized_histogram = tuple(value / scale for value in histogram)
        composition = self._encode_composition(record)
        cell_features = self._encode_cell(record) if self.include_cell else ()
        return normalized_histogram + composition + cell_features

    def _try_initialize_backend(self) -> None:
        try:
            from ase import Atoms  # noqa: F401
            from dscribe.descriptors import SOAP  # noqa: F401
        except Exception:  # noqa: BLE001
            self.backend_name = "soap_fallback_radial"
            return
        self.backend_name = "dscribe_soap"

    def _encode_with_dscribe(self, record: StructureRecord) -> tuple[float, ...] | None:
        if self.backend_name != "dscribe_soap":
            return None
        try:
            from ase import Atoms
            from dscribe.descriptors import SOAP
        except Exception:  # noqa: BLE001
            self.backend_name = "soap_fallback_radial"
            return None

        species = tuple(sorted(set(record.atomic_numbers)))
        soap = self._soap_builders.get(species)
        if soap is None:
            soap = SOAP(
                species=list(species),
                periodic=any(record.pbc),
                r_cut=self.cutoff,
                n_max=self.n_max,
                l_max=self.l_max,
                average="inner",
                sparse=False,
            )
            self._soap_builders[species] = soap
        atoms = Atoms(
            numbers=list(record.atomic_numbers),
            positions=list(record.positions),
            cell=list(record.cell) if record.cell is not None else None,
            pbc=tuple(record.pbc),
        )
        values = soap.create(atoms)
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, list):
            return tuple(float(value) for value in values)
        return None

    def encode_batch(self, records: list[StructureRecord]) -> "list[tuple[float, ...]]":
        """Encode multiple structures using batch processing.
        
        When dscribe SOAP backend is available, uses vectorized batch creation
        for significantly better performance. Falls back to individual encoding
        for fallback radial backend.
        
        Args:
            records: List of structure records to encode.
            
        Returns:
            List of descriptor tuples, one per input record.
        """
        if not records:
            return []
        
        # Use optimized batch path when SOAP backend is available
        if self.backend_name == "dscribe_soap":
            return self._encode_batch_dscribe(records)
        
        # Fall back to default sequential encoding for fallback backend
        return [self.encode(record) for record in records]
    
    def _encode_batch_dscribe(
        self,
        records: list[StructureRecord],
        chunk_size: int = 2048,
    ) -> "list[tuple[float, ...]]":
        """Batch encode using dscribe SOAP backend with chunked processing."""
        from ase import Atoms
        from dscribe.descriptors import SOAP

        # Group records by species for efficient batching
        species_groups: dict[tuple[int, ...], list[tuple[int, StructureRecord]]] = {}
        for idx, record in enumerate(records):
            species = tuple(sorted(set(record.atomic_numbers)))
            if species not in species_groups:
                species_groups[species] = []
            species_groups[species].append((idx, record))

        results: list[tuple[float, ...] | None] = [None] * len(records)

        for species, indexed_records in species_groups.items():
            soap = self._soap_builders.get(species)
            if soap is None:
                pbc_values = [ir[1].pbc for ir in indexed_records]
                periodic = any(any(pbc) for pbc in pbc_values) if pbc_values else False

                soap = SOAP(
                    species=list(species),
                    periodic=periodic,
                    r_cut=self.cutoff,
                    n_max=self.n_max,
                    l_max=self.l_max,
                    average="inner",
                    sparse=False,
                )
                self._soap_builders[species] = soap

            atoms_list: list[Any] = []
            indices: list[int] = []
            for idx, record in indexed_records:
                if not record.positions:
                    results[idx] = tuple(float(n) for n in record.atomic_numbers)
                    continue

                atoms = Atoms(
                    numbers=list(record.atomic_numbers),
                    positions=list(record.positions),
                    cell=list(record.cell) if record.cell is not None else None,
                    pbc=tuple(record.pbc),
                )
                atoms_list.append(atoms)
                indices.append(idx)

                if len(atoms_list) >= chunk_size:
                    self._flush_batch(soap, atoms_list, indices, results, records)
                    atoms_list = []
                    indices = []

            if atoms_list:
                self._flush_batch(soap, atoms_list, indices, results, records)

        for i, result in enumerate(results):
            if result is None:
                results[i] = self.encode(records[i])

        return results  # type: ignore[return-value]

    def _flush_batch(
        self,
        soap: Any,
        atoms_list: list[Any],
        indices: list[int],
        results: list[tuple[float, ...] | None],
        records: list[StructureRecord],
    ) -> None:
        try:
            batch_values = soap.create(atoms_list, n_jobs=-1)
            for i, idx in enumerate(indices):
                values = batch_values[i]
                if hasattr(values, "tolist"):
                    values = values.tolist()
                if isinstance(values, list):
                    results[idx] = tuple(float(v) for v in values)
                else:
                    results[idx] = self.encode(records[idx])
        except Exception:
            for idx in indices:
                results[idx] = self.encode(records[idx])

    @staticmethod
    def _encode_composition(record: StructureRecord) -> tuple[float, ...]:
        unique = sorted(set(record.atomic_numbers))
        if not unique:
            return ()
        total = max(record.natoms, 1)
        counts = []
        for atomic_number in unique[:4]:
            fraction = sum(1 for number in record.atomic_numbers if number == atomic_number) / total
            counts.append(fraction)
        while len(counts) < 4:
            counts.append(0.0)
        return tuple(counts)

    @staticmethod
    def _encode_cell(record: StructureRecord) -> tuple[float, ...]:
        if record.cell is None:
            return (0.0, 0.0, 0.0)
        lengths = tuple(
            math.sqrt(sum(component * component for component in row))
            for row in record.cell
        )
        return lengths
