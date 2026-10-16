"""Force-related dataset metrics."""

from __future__ import annotations

import math

from atom_dataset_index.domain import DatasetBundle
from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator, MetricContext, MetricSpec
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult, SampleMetricRecord
from atom_dataset_index.metric_core.scaling import auto_scale_linear
from atom_dataset_index.metric_core.statistics import entropy, mean, quantiles, variance
from atom_dataset_index.metric_core.validation import ensure_forces_available, ensure_non_empty_dataset

# When the per-pool mean force norm exceeds this threshold, the calculator
# applies a uniform linear scale factor so that the mean equals the
# threshold.  All distributional shape is preserved.
_FORCE_MEAN_THRESHOLD: float = 1.0  # eV/Å


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


class ForceStatisticsCalculator(BaseMetricCalculator):
    """Compute force coverage statistics."""

    spec = MetricSpec(
        metric_id="force.statistics",
        display_name="Force Statistics",
        required_fields=("forces",),
        output_fields=("force_mean_norm", "force_max_norm"),
        description="Force norm distribution statistics for dataset coverage.",
    )

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        del context
        ensure_non_empty_dataset(dataset)
        ensure_forces_available(dataset)
        sample_records: list[SampleMetricRecord] = []
        per_sample_means: list[float] = []
        for record in dataset.records:
            norms = [_vector_norm(vector) for vector in record.forces]
            if not norms:
                continue
            sample_mean = mean(norms)
            sample_max = max(norms)
            sample_records.append(
                SampleMetricRecord(
                    sample_id=record.sample_id,
                    values={"force_mean_norm": sample_mean, "force_max_norm": sample_max},
                )
            )
            per_sample_means.append(sample_mean)

        actual_force_mean = mean(per_sample_means) if per_sample_means else 0.0

        # Adaptive linear scaling for datasets with unusually large force norms.
        # Threshold set to 1.0 eV/Å to accommodate MLFF training datasets
        # that naturally sample far-from-equilibrium structures.
        scale_info = None
        if actual_force_mean > _FORCE_MEAN_THRESHOLD and per_sample_means:
            per_sample_means, scale_info = auto_scale_linear(
                per_sample_means,
                actual_force_mean,
                _FORCE_MEAN_THRESHOLD,
                "force_mean",
                self.spec.metric_id,
            )
            # Propagate the uniform scale factor to sample-level records so
            # that aggregate and per-sample values remain consistent.
            scale_factor = scale_info.scale_factor
            for sr in sample_records:
                sr.values["force_mean_norm"] *= scale_factor
                sr.values["force_max_norm"] *= scale_factor

        q90 = quantiles(per_sample_means, (0.9,)).get("0.9", 0.0)

        diagnostics: dict[str, object] = {}
        if scale_info is not None:
            diagnostics["auto_scaling"] = {
                "applied": True,
                "metric": scale_info.metric_name,
                "original_value": scale_info.original_value,
                "threshold": scale_info.threshold,
                "scale_factor": scale_info.scale_factor,
            }

        aggregate = AggregateMetricRecord(
            values={
                "force_mean": mean(per_sample_means),
                "force_variance": variance(per_sample_means),
                "force_q90": q90,
                "force_high_fraction": (
                    sum(1 for value in per_sample_means if value > q90) / len(per_sample_means)
                    if per_sample_means
                    else 0.0
                ),
                "force_entropy": entropy(per_sample_means),
            },
            diagnostics=diagnostics,
            aggregation_policy={
                "force_mean": "weighted_mean",
                "force_variance": "weighted_mean",
                "force_q90": "weighted_mean",
                "force_high_fraction": "weighted_mean",
                "force_entropy": "weighted_mean",
            },
        )
        return MetricResult(
            spec=self.spec,
            sample_records=tuple(sample_records),
            aggregate=aggregate,
        )
