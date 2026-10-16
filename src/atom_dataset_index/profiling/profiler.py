"""Dataset profiling workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from atom_dataset_index.domain import DatasetBundle, DatasetProfile
from atom_dataset_index.io import DatasetWriter
from atom_dataset_index.metric_core.descriptors import SOAPDescriptorAdapter
from atom_dataset_index.metric_core.interfaces import MetricContext
from atom_dataset_index.metric_core.registry import MetricRegistry
from atom_dataset_index.metric_core.reference import PrototypeStructureSelector, ReferenceStructureSelector
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult
from atom_dataset_index.metric_core.validation import (
    ensure_metric_result_consistent,
    ensure_non_empty_dataset,
    ensure_unique_sample_ids,
)
from atom_dataset_index.metrics.distortion import StructuralDistortionCalculator
from atom_dataset_index.metrics.diversity import DescriptorRedundancyCalculator, StructuralDiversityCalculator
from atom_dataset_index.metrics.energy import EnergyCoverageCalculator
from atom_dataset_index.metrics.force import ForceStatisticsCalculator
from atom_dataset_index.metrics.prototype import PrototypeCoverageCalculator
from atom_dataset_index.profiling.pools import DatasetPool, PoolBuilder


MetricFactory = Callable[[], object]


METRIC_FACTORIES: tuple[tuple[str, MetricFactory], ...] = (
    ("energy.coverage", EnergyCoverageCalculator),
    ("force.statistics", ForceStatisticsCalculator),
    ("distortion.prototype_relative", StructuralDistortionCalculator),
    ("prototype.coverage", PrototypeCoverageCalculator),
    ("diversity.structural", StructuralDiversityCalculator),
    ("diversity.redundancy", DescriptorRedundancyCalculator),
)

CORE_METRIC_IDS: tuple[str, ...] = (
    "energy.coverage",
    "force.statistics",
    "distortion.prototype_relative",
    "prototype.coverage",
    "diversity.structural",
    "diversity.redundancy",
)


def list_available_metric_ids() -> tuple[str, ...]:
    """Return all supported metric ids in default registration order."""

    return tuple(metric_id for metric_id, _ in METRIC_FACTORIES)


def build_default_registry(selected_metric_ids: tuple[str, ...] | None = None) -> MetricRegistry:
    """Build the default registry with optional metric filtering."""

    selected = selected_metric_ids or list_available_metric_ids()
    available = {metric_id: factory for metric_id, factory in METRIC_FACTORIES}
    unknown = [metric_id for metric_id in selected if metric_id not in available]
    if unknown:
        raise ValueError(f"Unknown metric ids: {', '.join(sorted(unknown))}")

    registry = MetricRegistry()
    registry.register_many(tuple(available[metric_id]() for metric_id in selected))
    return registry


def _sigmoid(value: float, mid: float, scale: float) -> float:
    if scale == 0:
        return 0.0 if value < mid else 1.0
    return 1.0 / (1.0 + math.exp(-(value - mid) / scale))


class DatasetProfiler:
    """Coordinate metric execution and assemble dataset profiles."""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        export_references_dir: str | Path | None = None,
        pool_builder: PoolBuilder | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.reference_selector = ReferenceStructureSelector()
        self.prototype_selector = PrototypeStructureSelector(adapter=SOAPDescriptorAdapter())
        self.export_references_dir = Path(export_references_dir) if export_references_dir is not None else None
        self.pool_builder = pool_builder or PoolBuilder()

    def profile(self, dataset: DatasetBundle) -> DatasetProfile:
        ensure_non_empty_dataset(dataset)
        ensure_unique_sample_ids(dataset)

        # 1. Build pools
        final_pools = self.pool_builder.build_pools(dataset)
        element_pools = self.pool_builder._split_by_composition(dataset)

        # 2. Element-pool level SOAP precompute
        element_pool_descriptors: dict[str, dict[int, tuple[float, ...]]] = {}
        for ep in element_pools:
            descriptors = self.prototype_selector.adapter.encode_batch(list(ep.records))
            element_pool_descriptors[ep.pool_id] = {
                sid: desc for sid, desc in zip([r.sample_id for r in ep.records], descriptors)
            }

        pool_results: list[dict[str, MetricResult]] = []
        all_skipped_metrics: dict[str, list[str]] = {}
        all_failed_metrics: dict[str, list[str]] = {}

        for pool in final_pools:
            pool_dataset = DatasetBundle(
                dataset_name=pool.pool_id,
                records=pool.records,
            )
            # Slice descriptors from element pool cache
            parent_pool_id = pool.parent_pool_id or pool.pool_id
            local_descriptors: dict[int, tuple[float, ...]] = {}
            if parent_pool_id in element_pool_descriptors:
                local_descriptors = {
                    sid: element_pool_descriptors[parent_pool_id][sid]
                    for sid in [r.sample_id for r in pool.records]
                    if sid in element_pool_descriptors[parent_pool_id]
                }

            context = MetricContext(reference_structure=self.reference_selector.select(pool_dataset))
            context.applicability_checks["constant_natoms"] = self._is_constant_natoms(pool_dataset)
            context.applicability_checks["periodic_dataset"] = self._is_periodic(pool_dataset)

            if context.applicability_checks["constant_natoms"] and context.applicability_checks["periodic_dataset"]:
                ref_result = self.prototype_selector.select_references(
                    pool_dataset,
                    precomputed_descriptors=local_descriptors if local_descriptors else None,
                )
                context.sample_descriptors = ref_result.sample_descriptors
                context.reference_sample_id = ref_result.reference_sample_id
                context.stable_reference_sample_ids = ref_result.stable_reference_sample_ids
                context.diversity_reference_sample_ids = ref_result.diversity_reference_sample_ids
                context.all_reference_sample_ids = ref_result.all_reference_sample_ids
                context.reference_assignments = ref_result.reference_assignments
                context.reference_distances = ref_result.reference_distances
                context.descriptor_backend = ref_result.descriptor_backend
                # Legacy fields for backward compatibility
                context.prototype_sample_ids = ref_result.all_reference_sample_ids
                context.anchor_sample_ids = ref_result.legacy_anchor_sample_ids
                context.prototype_assignments = ref_result.reference_assignments
                context.prototype_distances = ref_result.reference_distances

            if self.export_references_dir is not None:
                self._export_reference_structures(pool_dataset, context)

            results: dict[str, MetricResult] = {}
            skipped_metrics: dict[str, str] = {}
            failed_metrics: dict[str, str] = {}

            for metric_id in self.registry.list_metric_ids():
                calculator = self.registry.get(metric_id)
                try:
                    metric_result = calculator.calculate(pool_dataset, context)
                    ensure_metric_result_consistent(metric_result)
                except ValueError as error:
                    skipped_metrics[metric_id] = str(error)
                    continue
                except Exception as error:  # noqa: BLE001
                    failed_metrics[metric_id] = f"{type(error).__name__}: {error}"
                    continue
                results[metric_id] = metric_result

            for mid, reason in skipped_metrics.items():
                all_skipped_metrics.setdefault(mid, []).append(reason)
            for mid, reason in failed_metrics.items():
                all_failed_metrics.setdefault(mid, []).append(reason)

            pool_results.append(results)

        # 3. Aggregate pool results
        aggregated_results = self._aggregate_pool_results(dataset, final_pools, pool_results)

        # 4. Build scorecard (uses pool-level results for span metrics and effective phase count)
        scorecard = self._build_scorecard(dataset, final_pools, aggregated_results, pool_results)

        # 5. Build pool manifest for auditing
        pool_manifest = self._build_pool_manifest(final_pools, pool_results)

        summary = {
            "record_count": len(dataset.records),
            "pool_count": len(final_pools),
            "phase_count_effective": scorecard.get("phase_count_effective", len(final_pools)),
            "metric_count": len(aggregated_results),
            "skipped_metric_count": len(all_skipped_metrics),
            "failed_metric_count": len(all_failed_metrics),
            "stability_score": round(scorecard.get("stability_score", 0.0), 2),
            "compositional_diversity_score": round(scorecard.get("compositional_diversity_score", 0.0), 2),
            "structural_diversity_score": round(scorecard.get("structural_diversity_score", 0.0), 2),
            "uniformity_score": round(scorecard.get("uniformity_score", 0.0), 2),
            "coverage_score": round(scorecard.get("coverage_score", 0.0), 2),
            "redundancy_score": round(scorecard.get("redundancy_score", 0.0), 2),
        }

        # Collect anchor/prototype info across all pools for provenance (legacy compatibility)
        all_anchor_ids: list[int] = []
        all_proto_ids: list[int] = []
        descriptor_backend = "identity"
        # Re-run a lightweight global select just for provenance fields
        global_context = MetricContext(reference_structure=self.reference_selector.select(dataset))
        global_context.applicability_checks["constant_natoms"] = self._is_constant_natoms(dataset)
        global_context.applicability_checks["periodic_dataset"] = self._is_periodic(dataset)
        if global_context.applicability_checks["constant_natoms"] and global_context.applicability_checks["periodic_dataset"]:
            global_ref = self.prototype_selector.select_references(dataset)
            all_anchor_ids = list(global_ref.legacy_anchor_sample_ids)
            all_proto_ids = list(global_ref.all_reference_sample_ids)
            descriptor_backend = global_ref.descriptor_backend

        provenance = {
            "reference_strategy": "min_energy",
            "anchor_strategy": "pareto_energy_force",
            "prototype_strategy": "low_energy_force_filtered_medoids",
            "pool_count": str(len(final_pools)),
            "pool_ids": ",".join(p.pool_id for p in final_pools),
            "pool_types": ",".join(p.pool_type for p in final_pools),
            "pool_sizes": ",".join(str(len(p.records)) for p in final_pools),
            "selected_metrics": ",".join(self.registry.list_metric_ids()),
            "skipped_metrics": ",".join(sorted(all_skipped_metrics)) if all_skipped_metrics else "",
            "failed_metrics": ",".join(sorted(all_failed_metrics)) if all_failed_metrics else "",
            "anchor_sample_ids": ",".join(str(sid) for sid in all_anchor_ids),
            "prototype_sample_ids": ",".join(str(sid) for sid in all_proto_ids),
            "descriptor_backend": descriptor_backend,
            "applicability_checks_json": json.dumps({}, ensure_ascii=True, sort_keys=True),
            "skipped_reasons_json": json.dumps(
                {mid: reasons[0] for mid, reasons in all_skipped_metrics.items()},
                ensure_ascii=True, sort_keys=True,
            ),
            "failed_reasons_json": json.dumps(
                {mid: reasons[0] for mid, reasons in all_failed_metrics.items()},
                ensure_ascii=True, sort_keys=True,
            ),
            "pool_manifest_json": json.dumps(pool_manifest, ensure_ascii=True, sort_keys=False),
        }

        return DatasetProfile(
            dataset_name=dataset.dataset_name,
            metric_results=aggregated_results,
            summary=summary,
            provenance=provenance,
        )

    def _build_pool_manifest(
        self,
        pools: list[DatasetPool],
        pool_results: list[dict[str, MetricResult]],
    ) -> list[dict[str, object]]:
        """Construct per-pool detail records for external auditing."""
        manifest: list[dict[str, object]] = []
        for pool, results in zip(pools, pool_results):
            entry: dict[str, object] = {
                "pool_id": pool.pool_id,
                "pool_type": pool.pool_type,
                "parent_pool_id": pool.parent_pool_id,
                "size": len(pool.records),
                "metrics": {},
            }
            for metric_id, result in results.items():
                entry["metrics"][metric_id] = {
                    "aggregate_values": dict(result.aggregate.values),
                    "diagnostics": dict(result.aggregate.diagnostics),
                }
            manifest.append(entry)
        return manifest

    def _aggregate_pool_results(
        self,
        dataset: DatasetBundle,
        pools: list[DatasetPool],
        pool_results: list[dict[str, MetricResult]],
    ) -> dict[str, MetricResult]:
        """Aggregate per-pool metric results into dataset-level metrics."""
        aggregated: dict[str, MetricResult] = {}
        total_size = len(dataset.records)
        if total_size == 0:
            return aggregated

        for metric_id in self.registry.list_metric_ids():
            per_pool: list[tuple[DatasetPool, MetricResult]] = []
            for pool, results in zip(pools, pool_results):
                if metric_id in results:
                    per_pool.append((pool, results[metric_id]))

            if not per_pool:
                continue

            # Concatenate sample records
            all_samples = tuple(
                sample for _, result in per_pool for sample in result.sample_records
            )

            # Merge aggregate values by policy
            merged_values: dict[str, float] = {}
            all_keys = set()
            for _, result in per_pool:
                all_keys.update(result.aggregate.values.keys())

            for key in all_keys:
                policy = per_pool[0][1].aggregate.aggregation_policy.get(key, "weighted_mean")
                values_with_weight = [
                    (result.aggregate.values.get(key, 0.0), len(pool.records) / total_size)
                    for pool, result in per_pool
                    if key in result.aggregate.values
                ]
                if not values_with_weight:
                    continue

                if policy == "max":
                    merged_values[key] = max(v for v, _ in values_with_weight)
                elif policy == "sum":
                    merged_values[key] = sum(v for v, _ in values_with_weight)
                elif policy == "skip":
                    continue
                else:  # weighted_mean (default)
                    merged_values[key] = sum(v * w for v, w in values_with_weight)

            merged_diagnostics = dict(per_pool[0][1].aggregate.diagnostics)

            aggregated[metric_id] = MetricResult(
                spec=per_pool[0][1].spec,
                sample_records=all_samples,
                aggregate=AggregateMetricRecord(
                    values=merged_values,
                    diagnostics=merged_diagnostics,
                ),
            )

        return aggregated

    @staticmethod
    def _compute_effective_phases(
        pools: list[DatasetPool],
    ) -> tuple[int, list[int], dict[str, int]]:
        """Group chemically similar pools to avoid over-counting minor composition variants.

        Returns the number of effective phases, the list of record counts per phase,
        and a mapping from pool_id to its effective phase index.
        Uses a greedy ball-cover rather than transitive union-find so that a long chain
        of gradually varying solid-solution compositions does not collapse into a single
        effective phase.
        """
        if not pools:
            return 0, [], {}

        # Build element vectors over the union of all elements
        all_elements = sorted(
            {
                z
                for pool in pools
                for record in pool.records
                for z in record.atomic_numbers
            }
        )

        def pool_element_vector(pool: DatasetPool) -> list[float]:
            total_atoms = sum(record.natoms for record in pool.records)
            counts: dict[int, float] = {}
            for record in pool.records:
                for z in record.atomic_numbers:
                    counts[z] = counts.get(z, 0.0) + 1.0
            if total_atoms == 0:
                return [0.0] * len(all_elements)
            return [counts.get(z, 0.0) / total_atoms for z in all_elements]

        def jaccard(left_set: set[int], right_set: set[int]) -> float:
            inter = len(left_set & right_set)
            union = len(left_set | right_set)
            return inter / union if union > 0 else 0.0

        def euclidean(left: list[float], right: list[float]) -> float:
            return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5

        pool_sets = {
            pool.pool_id: {z for record in pool.records for z in record.atomic_numbers}
            for pool in pools
        }
        pool_vectors = {pool.pool_id: pool_element_vector(pool) for pool in pools}
        pool_sizes = {pool.pool_id: len(pool.records) for pool in pools}

        # Thresholds for treating two pools as the same structural family
        jaccard_threshold = 0.8
        euclidean_threshold = 0.05

        pool_ids = [p.pool_id for p in pools]
        assigned: set[str] = set()
        phase_counts: list[int] = []
        pool_to_phase: dict[str, int] = {}
        phase_index = 0

        # Sort by size so that larger pools become cluster centers first
        for center_pid in sorted(pool_ids, key=lambda pid: pool_sizes[pid], reverse=True):
            if center_pid in assigned:
                continue
            cluster_size = pool_sizes[center_pid]
            assigned.add(center_pid)
            pool_to_phase[center_pid] = phase_index
            for other_pid in pool_ids:
                if other_pid in assigned:
                    continue
                if jaccard(pool_sets[center_pid], pool_sets[other_pid]) < jaccard_threshold:
                    continue
                if euclidean(pool_vectors[center_pid], pool_vectors[other_pid]) < euclidean_threshold:
                    assigned.add(other_pid)
                    pool_to_phase[other_pid] = phase_index
                    cluster_size += pool_sizes[other_pid]
            phase_counts.append(cluster_size)
            phase_index += 1

        return len(phase_counts), phase_counts, pool_to_phase

    def _build_scorecard(
        self,
        dataset: DatasetBundle,
        pools: list[DatasetPool],
        aggregated_results: dict[str, MetricResult],
        pool_results: list[dict[str, MetricResult]],
    ) -> dict[str, float]:
        """Build dataset-level 6-dimensional scorecard."""
        total_size = len(dataset.records)
        phase_count_effective, phase_record_counts, _ = self._compute_effective_phases(pools)

        # Coverage fractions per effective phase (not raw pool)
        phase_fractions = [count / total_size for count in phase_record_counts] if total_size > 0 else []
        entropy_h = 0.0
        if phase_fractions and phase_count_effective > 1:
            for frac in phase_fractions:
                if frac > 0:
                    entropy_h -= frac * math.log(frac)
            uniformity_score = 100.0 * (entropy_h / math.log(phase_count_effective))
        elif len(pools) > 1:
            # Fallback to raw pool fractions when all pools collapse into one effective phase
            pool_fractions = [len(pool.records) / total_size for pool in pools]
            for frac in pool_fractions:
                if frac > 0:
                    entropy_h -= frac * math.log(frac)
            uniformity_score = 100.0 * (entropy_h / math.log(len(pools)))
        else:
            uniformity_score = 0.0

        # Extract aggregated inputs
        energy_agg = aggregated_results.get("energy.coverage", MetricResult(
            spec=EnergyCoverageCalculator.spec,
            sample_records=(),
            aggregate=AggregateMetricRecord(values={}),
        )).aggregate.values
        force_agg = aggregated_results.get("force.statistics", MetricResult(
            spec=ForceStatisticsCalculator.spec,
            sample_records=(),
            aggregate=AggregateMetricRecord(values={}),
        )).aggregate.values
        distortion_agg = aggregated_results.get("distortion.prototype_relative", MetricResult(
            spec=StructuralDistortionCalculator.spec,
            sample_records=(),
            aggregate=AggregateMetricRecord(values={}),
        )).aggregate.values
        structural_agg = aggregated_results.get("diversity.structural", MetricResult(
            spec=StructuralDiversityCalculator.spec,
            sample_records=(),
            aggregate=AggregateMetricRecord(values={}),
        )).aggregate.values
        redundancy_agg = aggregated_results.get("diversity.redundancy", MetricResult(
            spec=DescriptorRedundancyCalculator.spec,
            sample_records=(),
            aggregate=AggregateMetricRecord(values={}),
        )).aggregate.values

        mean_pool_force_mean = force_agg.get("force_mean", 0.0)
        mean_pool_force_variance = force_agg.get("force_variance", 0.0)
        mean_pool_stable_distance = distortion_agg.get("stable_distance_mean", 0.0)
        mean_pool_energy_range = energy_agg.get("energy_per_atom_range", 0.0)
        structural_diversity_score = structural_agg.get("structural_diversity_score", 0.0)
        redundancy_score = redundancy_agg.get("redundancy_score", 0.0)

        stability = mean([
            100.0 * (1.0 - _sigmoid(mean_pool_force_mean, mid=0.05, scale=0.02)),
            100.0 * (1.0 - _sigmoid(mean_pool_force_variance, mid=0.01, scale=0.005)),
            100.0 * (1.0 - _sigmoid(mean_pool_stable_distance, mid=0.30, scale=0.10)),
            100.0 * (1.0 - _sigmoid(mean_pool_energy_range, mid=0.50, scale=0.20)),
        ]) if any([mean_pool_force_mean, mean_pool_force_variance, mean_pool_stable_distance, mean_pool_energy_range]) else 0.0

        compositional_diversity_score = (
            100.0 * _sigmoid(float(phase_count_effective), mid=5.0, scale=2.0)
            if phase_count_effective > 0 else 0.0
        )

        # Coverage score couples dataset scale with uniformity
        scale_score = 100.0 * _sigmoid(float(total_size), mid=5000.0, scale=2500.0)
        coverage_score = scale_score * (uniformity_score / 100.0)

        return {
            "stability_score": stability,
            "compositional_diversity_score": compositional_diversity_score,
            "structural_diversity_score": structural_diversity_score,
            "uniformity_score": uniformity_score,
            "coverage_score": coverage_score,
            "redundancy_score": redundancy_score,
            "phase_count_effective": float(phase_count_effective),
            "mean_pool_force_mean": mean_pool_force_mean,
            "mean_pool_force_variance": mean_pool_force_variance,
            "mean_pool_stable_distance": mean_pool_stable_distance,
            "mean_pool_energy_range": mean_pool_energy_range,
        }

    def _export_reference_structures(self, dataset: DatasetBundle, context: MetricContext) -> None:
        if self.export_references_dir is None:
            return
        self.export_references_dir.mkdir(parents=True, exist_ok=True)
        record_by_id = {record.sample_id: record for record in dataset.records}

        if context.reference_structure is not None:
            ref = context.reference_structure
            DatasetWriter.write_xyz(self.export_references_dir / f"reference_{ref.sample_id}.xyz", ref)

        for index, sample_id in enumerate(context.stable_reference_sample_ids):
            record = record_by_id.get(sample_id)
            if record is not None:
                DatasetWriter.write_xyz(
                    self.export_references_dir / f"stable_ref_{index}_{sample_id}.xyz", record
                )

        for index, sample_id in enumerate(context.diversity_reference_sample_ids):
            record = record_by_id.get(sample_id)
            if record is not None:
                DatasetWriter.write_xyz(
                    self.export_references_dir / f"diversity_ref_{index}_{sample_id}.xyz", record
                )

    @staticmethod
    def _is_constant_natoms(dataset: DatasetBundle) -> bool:
        from atom_dataset_index.metric_core.validation import ensure_constant_natoms

        try:
            ensure_constant_natoms(dataset)
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_periodic(dataset: DatasetBundle) -> bool:
        from atom_dataset_index.metric_core.validation import ensure_periodic_dataset

        try:
            ensure_periodic_dataset(dataset)
        except ValueError:
            return False
        return True


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
