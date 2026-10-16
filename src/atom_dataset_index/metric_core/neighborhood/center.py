"""Helpers for center-atom selection."""

from __future__ import annotations

from atom_dataset_index.domain import StructureRecord


class CenterAtomSelector:
    """Select center atoms for local-environment metrics."""

    def select(self, record: StructureRecord, strategy: str = "all") -> tuple[int, ...]:
        if strategy == "all":
            return tuple(range(record.natoms))
        if strategy == "middle" and record.natoms:
            return (record.natoms // 2,)
        raise ValueError(f"Unsupported center selection strategy: {strategy}")

