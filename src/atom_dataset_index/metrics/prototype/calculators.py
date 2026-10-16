"""Prototype coverage metrics."""

from __future__ import annotations

import math

from atom_dataset_index.domain import DatasetBundle
from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator, MetricContext, MetricSpec
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult, SampleMetricRecord
from atom_dataset_index.metric_core.statistics import mean
from atom_dataset_index.metric_core.validation import (
    ensure_constant_natoms,
    ensure_energy_available,
    ensure_non_empty_dataset,
    ensure_periodic_dataset,
    ensure_positions_available,
)


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dimension = max(len(left), len(right))
    total = 0.0
    for index in range(dimension):
        left_value = left[index] if index < len(left) else 0.0
        right_value = right[index] if index < len(right) else 0.0
        total += (left_value - right_value) ** 2
    return math.sqrt(total)


class PrototypeCoverageCalculator(BaseMetricCalculator):
    """Summarize prototype occupancy and spread."""

    spec = MetricSpec(
        metric_id="prototype.coverage",
        display_name="Prototype Coverage",
        required_fields=("positions", "cell", "pbc", "energy"),
        output_fields=("prototype_id", "prototype_distance"),
        description="Coverage of descriptor-space prototypes across the dataset.",
    )

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        ensure_non_empty_dataset(dataset)
        ensure_positions_available(dataset)
        ensure_energy_available(dataset)
        ensure_constant_natoms(dataset)
        ensure_periodic_dataset(dataset)
        if not context.prototype_sample_ids or not context.prototype_distances:
            raise ValueError("Prototype selection is unavailable for prototype coverage metric.")

        sample_records = tuple(
            SampleMetricRecord(
                sample_id=record.sample_id,
                values={
                    "prototype_id": float(context.prototype_assignments.get(record.sample_id, -1)),
                    "prototype_distance": context.prototype_distances.get(record.sample_id, 0.0),
                },
            )
            for record in dataset.records
        )
        occupancy: dict[int, int] = {sample_id: 0 for sample_id in context.prototype_sample_ids}
        for sample_id in context.prototype_assignments.values():
            occupancy[sample_id] = occupancy.get(sample_id, 0) + 1

        total = max(sum(occupancy.values()), 1)
        occupancy_probs = [count / total for count in occupancy.values() if count > 0]
        occupancy_entropy = -sum(prob * math.log(prob) for prob in occupancy_probs) if occupancy_probs else 0.0

        descriptor_vectors = context.sample_descriptors
        prototype_vectors = [
            descriptor_vectors[sample_id]
            for sample_id in context.prototype_sample_ids
            if sample_id in descriptor_vectors
        ]
        interdistances: list[float] = []
        for left_index, left in enumerate(prototype_vectors):
            for right in prototype_vectors[left_index + 1 :]:
                interdistances.append(_distance(left, right))

        prototype_id_set = set(context.prototype_sample_ids)
        prototype_energies = [
            float(record.energy) / max(record.natoms, 1)
            for record in dataset.records
            if record.sample_id in prototype_id_set and record.energy is not None
        ]

        aggregate = AggregateMetricRecord(
            values={
                "prototype_count": float(len(context.prototype_sample_ids)),
                "prototype_occupancy_entropy": occupancy_entropy,
                "prototype_min_occupancy": min(occupancy.values()) / total if occupancy else 0.0,
                "prototype_mean_interdistance": mean(interdistances),
                "prototype_energy_span": (
                    max(prototype_energies) - min(prototype_energies) if prototype_energies else 0.0
                ),
            },
            aggregation_policy={
                "prototype_count": "sum",
                "prototype_occupancy_entropy": "weighted_mean",
                "prototype_min_occupancy": "weighted_mean",
                "prototype_mean_interdistance": "weighted_mean",
                "prototype_energy_span": "max",
            },
        )
        return MetricResult(
            spec=self.spec,
            sample_records=sample_records,
            aggregate=aggregate,
        )
