"""Abstract metric interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atom_dataset_index.domain import DatasetBundle, StructureRecord

if TYPE_CHECKING:
    from atom_dataset_index.metric_core.results import MetricResult


@dataclass(slots=True, frozen=True)
class MetricSpec:
    """Static metadata describing a metric implementation."""

    metric_id: str
    display_name: str
    required_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    description: str


@dataclass(slots=True)
class MetricContext:
    """Reusable metric runtime context."""

    reference_structure: StructureRecord | None = None
    center_atom_indices: dict[int, tuple[int, ...]] = field(default_factory=dict)
    descriptor_backend: str = "identity"
    config: dict[str, str | int | float | bool] = field(default_factory=dict)
    sample_descriptors: dict[int, tuple[float, ...]] = field(default_factory=dict)
    prototype_sample_ids: tuple[int, ...] = ()
    anchor_sample_ids: tuple[int, ...] = ()
    prototype_assignments: dict[int, int] = field(default_factory=dict)
    prototype_distances: dict[int, float] = field(default_factory=dict)
    applicability_checks: dict[str, bool] = field(default_factory=dict)
    # Pool-aware reference fields
    reference_sample_id: int | None = None
    stable_reference_sample_ids: tuple[int, ...] = ()
    diversity_reference_sample_ids: tuple[int, ...] = ()
    all_reference_sample_ids: tuple[int, ...] = ()
    reference_assignments: dict[int, int] = field(default_factory=dict)
    reference_distances: dict[int, float] = field(default_factory=dict)


class BaseMetricCalculator(ABC):
    """Shared interface for all metric algorithm implementations."""

    spec: MetricSpec

    @abstractmethod
    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> "MetricResult":
        """Run the metric and return a MetricResult."""
