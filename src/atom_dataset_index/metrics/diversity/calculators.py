"""Descriptor diversity metrics."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from atom_dataset_index.domain import DatasetBundle
from atom_dataset_index.metric_core.descriptors import DescriptorAdapter, SOAPDescriptorAdapter
from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator, MetricContext, MetricSpec
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult, SampleMetricRecord
from atom_dataset_index.metric_core.statistics import mean, quantiles, variance
from atom_dataset_index.metric_core.validation import ensure_non_empty_dataset, ensure_positions_available


class DescriptorDiversityCalculator(BaseMetricCalculator):
    """Simple diversity metric based on descriptor spread."""

    spec = MetricSpec(
        metric_id="diversity.descriptor",
        display_name="Descriptor Diversity",
        required_fields=("positions", "atomic_numbers"),
        output_fields=("descriptor_radius",),
        description="Descriptor-space spread statistics.",
    )

    def __init__(self, adapter: DescriptorAdapter | None = None) -> None:
        self.adapter = adapter or SOAPDescriptorAdapter()

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        del context
        ensure_non_empty_dataset(dataset)
        ensure_positions_available(dataset)
        descriptors = [self.adapter.encode(record) for record in dataset.records]
        if not descriptors:
            raise ValueError("No descriptors available.")
        dimension = max(len(vector) for vector in descriptors)
        padded = [vector + (0.0,) * (dimension - len(vector)) for vector in descriptors]
        center = tuple(mean([vector[index] for vector in padded]) for index in range(dimension))

        sample_records = []
        radii = []
        for record, vector in zip(dataset.records, padded, strict=False):
            radius = math.sqrt(
                sum((value - center[index]) ** 2 for index, value in enumerate(vector))
            )
            radii.append(radius)
            sample_records.append(
                SampleMetricRecord(sample_id=record.sample_id, values={"descriptor_radius": radius})
            )

        aggregate = AggregateMetricRecord(
            values={
                "descriptor_radius_mean": mean(radii),
                "descriptor_radius_variance": variance(radii),
                "descriptor_radius_max": max(radii) if radii else 0.0,
            },
            aggregation_policy={
                "descriptor_radius_mean": "weighted_mean",
                "descriptor_radius_variance": "weighted_mean",
                "descriptor_radius_max": "max",
            },
        )
        return MetricResult(
            spec=self.spec,
            sample_records=tuple(sample_records),
            aggregate=aggregate,
        )


def _sigmoid(value: float, mid: float, scale: float) -> float:
    if scale == 0:
        return 0.0 if value < mid else 1.0
    return 1.0 / (1.0 + math.exp(-(value - mid) / scale))


def _descriptor_matrix_for_dataset(
    dataset: DatasetBundle,
    context: MetricContext,
    adapter: DescriptorAdapter,
) -> np.ndarray:
    """Return a padded descriptor matrix, preferring cached descriptors when available."""
    descriptors: list[tuple[float, ...]] = []
    if getattr(context, "sample_descriptors", None):
        descriptors = [
            context.sample_descriptors.get(record.sample_id, ())  # type: ignore[union-attr]
            for record in dataset.records
        ]
    if not descriptors or any(not d for d in descriptors):
        descriptors = adapter.encode_batch(list(dataset.records))

    dimension = max((len(d) for d in descriptors), default=0)
    matrix = np.zeros((len(descriptors), dimension), dtype=np.float64)
    for i, descriptor in enumerate(descriptors):
        matrix[i, : len(descriptor)] = descriptor
    return matrix


class StructuralDiversityCalculator(BaseMetricCalculator):
    """Measure structural diversity via descriptor-space clustering.

    Unlike composition-driven diversity, this metric is sensitive to local
    structure variation within a single phase.  It clusters the SOAP descriptor
    matrix of each pool and reports the normalized occupancy entropy of the
    resulting clusters.
    """

    spec = MetricSpec(
        metric_id="diversity.structural",
        display_name="Structural Diversity",
        required_fields=("positions", "atomic_numbers"),
        output_fields=("cluster_id", "cluster_distance"),
        description="Structural diversity measured by descriptor-space KMeans clustering.",
    )

    def __init__(
        self,
        min_clusters: int = 3,
        max_clusters: int = 12,
        cluster_threshold: int = 3000,
        min_cluster_fraction: float = 0.05,
        random_state: int = 42,
        adapter: DescriptorAdapter | None = None,
    ) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.cluster_threshold = cluster_threshold
        self.min_cluster_fraction = min_cluster_fraction
        self.random_state = random_state
        self.adapter = adapter or SOAPDescriptorAdapter()

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        ensure_non_empty_dataset(dataset)
        ensure_positions_available(dataset)

        matrix = _descriptor_matrix_for_dataset(dataset, context, self.adapter)
        n = matrix.shape[0]

        n_clusters = min(
            self.max_clusters,
            max(self.min_clusters, n // self.cluster_threshold),
            n,
        )

        if n_clusters <= 1:
            labels = np.zeros(n, dtype=int)
            effective_clusters = 1
            occupancy_entropy = 0.0
            normalized_entropy = 0.0
        else:
            try:
                from sklearn.cluster import KMeans

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        n_init="auto",
                    ).fit_predict(matrix)
            except Exception:  # noqa: BLE001
                labels = np.zeros(n, dtype=int)
                n_clusters = 1
                effective_clusters = 1
                occupancy_entropy = 0.0
                normalized_entropy = 0.0
            else:
                counts = np.bincount(labels, minlength=n_clusters)
                total = int(counts.sum())
                fractions = counts / total if total > 0 else counts
                effective_clusters = int(np.sum(fractions > self.min_cluster_fraction))
                nonzero = fractions[fractions > 0.0]
                occupancy_entropy = float(-np.sum(nonzero * np.log(nonzero)))
                normalized_entropy = (
                    occupancy_entropy / math.log(n_clusters) if n_clusters > 1 else 0.0
                )

        structural_diversity_score = 100.0 * normalized_entropy

        # Compute per-sample cluster distance to the cluster centroid.
        centroids: dict[int, np.ndarray] = {}
        for cluster in range(max(1, n_clusters)):
            mask = labels == cluster
            if np.any(mask):
                centroids[cluster] = matrix[mask].mean(axis=0)

        sample_records = []
        for i, record in enumerate(dataset.records):
            cluster = int(labels[i])
            centroid = centroids.get(cluster, matrix.mean(axis=0))
            distance = float(np.linalg.norm(matrix[i] - centroid))
            sample_records.append(
                SampleMetricRecord(
                    sample_id=record.sample_id,
                    values={
                        "cluster_id": float(cluster),
                        "cluster_distance": distance,
                    },
                )
            )

        aggregate = AggregateMetricRecord(
            values={
                "effective_cluster_count": float(effective_clusters),
                "cluster_occupancy_entropy": occupancy_entropy,
                "normalized_cluster_entropy": normalized_entropy,
                "structural_diversity_score": round(structural_diversity_score, 2),
            },
            diagnostics={
                "n_clusters": n_clusters,
                "min_cluster_fraction": self.min_cluster_fraction,
                "cluster_threshold": self.cluster_threshold,
            },
            aggregation_policy={
                "effective_cluster_count": "weighted_mean",
                "cluster_occupancy_entropy": "weighted_mean",
                "normalized_cluster_entropy": "weighted_mean",
                "structural_diversity_score": "weighted_mean",
            },
        )
        return MetricResult(
            spec=self.spec,
            sample_records=tuple(sample_records),
            aggregate=aggregate,
        )


def _kth_nearest_neighbor_distances(
    matrix: "np.ndarray",
    k: int,
    chunk_size: int = 500,
) -> "np.ndarray":
    """Return the k-th nearest-neighbor distance for each row (self excluded).

    Uses chunked ``cdist`` queries so the full pairwise distance matrix is never
    materialised, keeping memory use manageable for pools with tens of thousands
    of records.
    """
    n = matrix.shape[0]
    if n <= k:
        return np.zeros(n, dtype=np.float64)

    distances = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = cdist(matrix[start:end], matrix, metric="euclidean")
        rows = np.arange(end - start)
        cols = start + rows
        dists[rows, cols] = np.inf
        distances[start:end] = np.partition(dists, k - 1, axis=1)[:, k - 1]
    return distances


class DescriptorRedundancyCalculator(BaseMetricCalculator):
    """Measure descriptor-space redundancy via k-th nearest-neighbor distance.

    A low median k-NN distance indicates that many records are packed closely
    together in descriptor space, i.e. the dataset is locally redundant.  The
    calculator emits a raw distance and a 0-100 negative score:

    - ``redundancy_score``: high = redundant (negative indicator)
    """

    spec = MetricSpec(
        metric_id="diversity.redundancy",
        display_name="Descriptor Redundancy",
        required_fields=("positions", "atomic_numbers"),
        output_fields=("knn_distance",),
        description="Descriptor-space redundancy measured by k-th nearest neighbor distance.",
    )

    def __init__(
        self,
        k: int = 5,
        score_mid: float = 0.60,
        score_scale: float = 0.12,
        chunk_size: int = 500,
        adapter: DescriptorAdapter | None = None,
    ) -> None:
        self.k = k
        self.score_mid = score_mid
        self.score_scale = score_scale
        self.chunk_size = chunk_size
        self.adapter = adapter or SOAPDescriptorAdapter()

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        ensure_non_empty_dataset(dataset)
        ensure_positions_available(dataset)

        matrix = _descriptor_matrix_for_dataset(dataset, context, self.adapter)
        knn_distances = _kth_nearest_neighbor_distances(matrix, self.k, self.chunk_size)

        sample_records = tuple(
            SampleMetricRecord(
                sample_id=record.sample_id,
                values={"knn_distance": float(knn_distances[i])},
            )
            for i, record in enumerate(dataset.records)
        )

        distance_values = [float(value) for value in knn_distances]
        median_distance = quantiles(distance_values, (0.5,)).get("0.5", 0.0)
        mean_distance = mean(distance_values)

        redundancy = 100.0 * (1.0 - _sigmoid(median_distance, self.score_mid, self.score_scale))

        aggregate = AggregateMetricRecord(
            values={
                "median_knn_distance": median_distance,
                "mean_knn_distance": mean_distance,
                "redundancy_score": round(redundancy, 2),
            },
            diagnostics={
                "k_neighbors": self.k,
                "score_mid": self.score_mid,
                "score_scale": self.score_scale,
            },
            aggregation_policy={
                "median_knn_distance": "weighted_mean",
                "mean_knn_distance": "weighted_mean",
                "redundancy_score": "weighted_mean",
            },
        )
        return MetricResult(
            spec=self.spec,
            sample_records=sample_records,
            aggregate=aggregate,
        )
