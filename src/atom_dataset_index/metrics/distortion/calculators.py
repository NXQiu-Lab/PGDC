"""Structural distortion metrics."""

from __future__ import annotations

import math

from atom_dataset_index.domain import DatasetBundle
from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator, MetricContext, MetricSpec
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult, SampleMetricRecord
from atom_dataset_index.metric_core.scaling import auto_scale_linear
from atom_dataset_index.metric_core.statistics import entropy, mean, quantiles
from atom_dataset_index.metric_core.validation import (
    ensure_constant_natoms,
    ensure_non_empty_dataset,
    ensure_periodic_dataset,
    ensure_positions_available,
)

# When the mean stable descriptor-space distance exceeds this threshold,
# a uniform linear scale factor is applied so that the mean equals the
# threshold.  All distributional shape is preserved.
_STABLE_DISTANCE_THRESHOLD: float = 2.0


def _descriptor_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dimension = max(len(left), len(right))
    total = 0.0
    for i in range(dimension):
        lv = left[i] if i < len(left) else 0.0
        rv = right[i] if i < len(right) else 0.0
        total += (lv - rv) ** 2
    return math.sqrt(total)


def _min_descriptor_distance(desc: tuple[float, ...], refs: list[tuple[float, ...]]) -> float:
    if not refs:
        return 0.0
    return min(_descriptor_distance(desc, ref) for ref in refs)


def _nearest_id(desc: tuple[float, ...], ref_map: dict[int, tuple[float, ...]]) -> int:
    if not ref_map:
        return -1
    best_id = -1
    best_dist = float("inf")
    for sid, ref_desc in ref_map.items():
        dist = _descriptor_distance(desc, ref_desc)
        if dist < best_dist:
            best_dist = dist
            best_id = sid
    return best_id


class StructuralDistortionCalculator(BaseMetricCalculator):
    """Measure descriptor-space distance to stability and diversity references."""

    spec = MetricSpec(
        metric_id="distortion.prototype_relative",
        display_name="Prototype Relative Distortion",
        required_fields=("positions", "cell", "pbc"),
        output_fields=(
            "stable_distance",
            "coverage_distance",
            "reference_distance",
            "nearest_stable_ref_id",
            "nearest_coverage_ref_id",
        ),
        description="Pool-aware structural deviation to stability and diversity references.",
    )

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        ensure_non_empty_dataset(dataset)
        ensure_positions_available(dataset)
        ensure_constant_natoms(dataset)
        ensure_periodic_dataset(dataset)

        if not context.all_reference_sample_ids:
            raise ValueError("Reference selection is unavailable for distortion metric.")

        descriptors = [context.sample_descriptors.get(r.sample_id, ()) for r in dataset.records]

        stable_ref_map = {
            sid: context.sample_descriptors[sid]
            for sid in context.stable_reference_sample_ids
            if sid in context.sample_descriptors
        }
        coverage_ref_map = {
            sid: context.sample_descriptors[sid]
            for sid in context.diversity_reference_sample_ids
            if sid in context.sample_descriptors
        }
        ref_desc = (
            context.sample_descriptors.get(context.reference_sample_id, ())
            if context.reference_sample_id is not None
            else ()
        )

        stable_refs = list(stable_ref_map.values())
        coverage_refs = list(coverage_ref_map.values())

        sample_records = list(
            SampleMetricRecord(
                sample_id=record.sample_id,
                values={
                    "stable_distance": _min_descriptor_distance(desc, stable_refs),
                    "coverage_distance": _min_descriptor_distance(desc, coverage_refs),
                    "reference_distance": _descriptor_distance(desc, ref_desc),
                    "nearest_stable_ref_id": float(_nearest_id(desc, stable_ref_map)),
                    "nearest_coverage_ref_id": float(_nearest_id(desc, coverage_ref_map)),
                },
            )
            for record, desc in zip(dataset.records, descriptors, strict=False)
        )

        stable_distances = [r.values["stable_distance"] for r in sample_records]
        coverage_distances = [r.values["coverage_distance"] for r in sample_records]

        actual_stable_mean = mean(stable_distances) if stable_distances else 0.0

        # Adaptive linear scaling for unusually large stable distances.
        # Threshold raised to 2.0 to better accommodate MLFF datasets that
        # contain structures far from the equilibrium prototype.
        scale_info = None
        if actual_stable_mean > _STABLE_DISTANCE_THRESHOLD and stable_distances:
            stable_distances, scale_info = auto_scale_linear(
                stable_distances,
                actual_stable_mean,
                _STABLE_DISTANCE_THRESHOLD,
                "stable_distance_mean",
                self.spec.metric_id,
            )
            # Propagate scale factor to sample records for consistency.
            scale_factor = scale_info.scale_factor
            for sr, sd in zip(sample_records, stable_distances):
                sr.values["stable_distance"] = sd

        # Compute coverage (diversity) reference inter-distance span
        coverage_ref_vectors = list(coverage_ref_map.values())
        coverage_interdistances: list[float] = []
        for left_index, left in enumerate(coverage_ref_vectors):
            for right in coverage_ref_vectors[left_index + 1 :]:
                coverage_interdistances.append(_descriptor_distance(left, right))

        diagnostics: dict[str, object] = {
            "stable_ref_count": len(context.stable_reference_sample_ids),
            "coverage_ref_count": len(context.diversity_reference_sample_ids),
            "stable_distance_entropy": entropy(stable_distances),
            "coverage_ref_interdistance": mean(coverage_interdistances) if coverage_interdistances else 0.0,
        }
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
                "stable_distance_mean": mean(stable_distances),
                "stable_distance_max": max(stable_distances) if stable_distances else 0.0,
                "stable_distance_q90": quantiles(stable_distances, (0.9,)).get("0.9", 0.0),
                "coverage_distance_mean": mean(coverage_distances),
                "coverage_distance_max": max(coverage_distances) if coverage_distances else 0.0,
            },
            diagnostics=diagnostics,
            aggregation_policy={
                "stable_distance_mean": "weighted_mean",
                "stable_distance_max": "max",
                "stable_distance_q90": "weighted_mean",
                "coverage_distance_mean": "weighted_mean",
                "coverage_distance_max": "max",
            },
        )
        return MetricResult(
            spec=self.spec,
            sample_records=tuple(sample_records),
            aggregate=aggregate,
        )
