"""Reusable metric result containers."""

from __future__ import annotations

from dataclasses import dataclass, field

from atom_dataset_index.metric_core.interfaces import MetricSpec


@dataclass(slots=True)
class SampleMetricRecord:
    """Sample-level metric output."""

    sample_id: int
    values: dict[str, float]
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(slots=True)
class AggregateMetricRecord:
    """Dataset-level summary for a metric."""

    values: dict[str, float]
    diagnostics: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    aggregation_policy: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MetricResult:
    """Unified result object returned by metric calculators."""

    spec: MetricSpec
    sample_records: tuple[SampleMetricRecord, ...]
    aggregate: AggregateMetricRecord

