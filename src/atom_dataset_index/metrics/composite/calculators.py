"""Composite metric placeholders."""

from __future__ import annotations

from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator, MetricContext, MetricSpec
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult


class CompositeMetricCalculator(BaseMetricCalculator):
    """Placeholder for profile-level composite metrics."""

    spec = MetricSpec(
        metric_id="composite.placeholder",
        display_name="Composite Placeholder",
        required_fields=(),
        output_fields=("composite_score",),
        description="Reserved placeholder for future profile-level compositions.",
    )

    def calculate(self, dataset, context: MetricContext) -> MetricResult:
        del dataset, context
        return MetricResult(
            spec=self.spec,
            sample_records=(),
            aggregate=AggregateMetricRecord(
                values={"composite_score": 0.0},
                aggregation_policy={},
            ),
        )
