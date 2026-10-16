"""Stable domain models used across package boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
PrimitiveMetadataValue = str | int | float | bool | None


class DomainValidationError(ValueError):
    """Raised when domain model inputs are inconsistent."""


@dataclass(slots=True)
class StructureRecord:
    """Single structure sample with optional arrays already decoded."""

    sample_id: int
    atomic_numbers: tuple[int, ...]
    positions: tuple[Vector3, ...] = ()
    cell: Matrix3 | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    energy: float | None = None
    forces: tuple[Vector3, ...] = ()
    metadata: dict[str, PrimitiveMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_id < 0:
            raise DomainValidationError("sample_id must be non-negative.")
        if not self.atomic_numbers:
            raise DomainValidationError("atomic_numbers must not be empty.")
        if self.positions and len(self.positions) != self.natoms:
            raise DomainValidationError("positions length must match atomic_numbers length.")
        if self.forces and len(self.forces) != self.natoms:
            raise DomainValidationError("forces length must match atomic_numbers length.")
        if self.cell is not None and any(len(row) != 3 for row in self.cell):
            raise DomainValidationError("cell must contain three 3D vectors.")
        if len(self.pbc) != 3:
            raise DomainValidationError("pbc must contain exactly three boolean flags.")
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise DomainValidationError("metadata keys must be strings.")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise DomainValidationError("metadata values must be primitive JSON-compatible types.")

    @property
    def natoms(self) -> int:
        return len(self.atomic_numbers)


@dataclass(slots=True)
class DatasetBundle:
    """Dataset-level container shared by profiling and benchmark tasks."""

    dataset_name: str
    records: tuple[StructureRecord, ...]
    metadata: dict[str, PrimitiveMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise DomainValidationError("dataset_name must not be empty.")
        sample_ids = [record.sample_id for record in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            raise DomainValidationError("dataset contains duplicated sample_id values.")
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise DomainValidationError("dataset metadata keys must be strings.")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise DomainValidationError(
                    "dataset metadata values must be primitive JSON-compatible types."
                )

    def energies(self) -> list[float]:
        return [record.energy for record in self.records if record.energy is not None]


@dataclass(slots=True)
class DatasetProfile:
    """Aggregated profile generated from metric results."""

    dataset_name: str
    metric_results: dict[str, object]
    summary: dict[str, float | int | str]
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingResult:
    """Training artifact summary for SchNetPack baselines."""

    model_name: str
    split_name: str
    metrics: dict[str, float]
    artifact_paths: dict[str, str]
    metadata: dict[str, PrimitiveMetadataValue] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkResult:
    """Joint profile and training comparison output."""

    benchmark_name: str
    profiles: dict[str, DatasetProfile]
    training_results: tuple[TrainingResult, ...]
    summary: dict[str, float | int | str]
