"""Uncertainty estimator implementations for acquisition mode."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import DescriptorAdapter, SOAPDescriptorAdapter

from .interfaces import UncertaintyEstimator

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class DistanceBasedUncertaintyEstimator(UncertaintyEstimator):
    """Uncertainty estimator based on descriptor distance to dataset.

    This estimator uses the distance between a candidate structure and its
    nearest neighbor in the existing dataset as a proxy for model uncertainty.
    The intuition is that structures far from the training data are likely
    to have higher prediction uncertainty.

    This estimator does not require a trained model and can be used as a
    fallback when model checkpoints are not available.

    Uses scikit-learn's NearestNeighbors for efficient ANN queries, providing
    O(log N) nearest neighbor lookup compared to O(N) linear scan.

    Example:
        >>> from atom_dataset_index.acquisition import DistanceBasedUncertaintyEstimator
        >>> from atom_dataset_index.io import AseDbDatasetReader
        >>> 
        >>> dataset = AseDbDatasetReader("data/existing.db").read()
        >>> estimator = DistanceBasedUncertaintyEstimator(dataset)
        >>> 
        >>> uncertainty = estimator.estimate(structure)  # Returns 0-1
    """

    def __init__(
        self,
        dataset: DatasetBundle,
        adapter: DescriptorAdapter | None = None,
        normalize: bool = True,
        max_distance_sample_ratio: float = 0.1,
    ) -> None:
        """Initialize the estimator.

        Args:
            dataset: The existing dataset to compute distances against.
            adapter: Descriptor adapter for encoding structures. Defaults to
                SOAPDescriptorAdapter.
            normalize: Whether to normalize distances to [0, 1] range.
            max_distance_sample_ratio: Ratio of dataset to sample for estimating
                max distance (0.1 = 10% sample). Use 1.0 for exact computation.
                Lower values improve initialization speed with minimal impact
                on normalization quality.
        """
        self.dataset = dataset
        self.adapter = adapter or SOAPDescriptorAdapter()
        self.normalize = normalize
        self.max_distance_sample_ratio = max_distance_sample_ratio

        # Build dataset descriptors and ANN index
        self._dataset_descriptors: dict[int, tuple[float, ...]] = {}
        self._descriptor_dim: int = 0
        self._max_dataset_distance: float = 0.0
        self._nn_index: "NearestNeighborsIndex | None" = None
        self._build_dataset_index()

    def _build_dataset_index(self) -> None:
        """Build index of dataset descriptors and ANN search structure."""
        if not self.dataset.records:
            return

        # Batch compute descriptors for all dataset structures (Phase 1 optimization)
        descriptors_list = self.adapter.encode_batch(list(self.dataset.records))
        
        for record, descriptor in zip(self.dataset.records, descriptors_list):
            self._dataset_descriptors[record.sample_id] = descriptor
            self._descriptor_dim = max(self._descriptor_dim, len(descriptor))

        # Build ANN index for fast nearest neighbor queries
        if len(self._dataset_descriptors) > 0:
            self._nn_index = NearestNeighborsIndex(
                list(self._dataset_descriptors.values()),
                algorithm="auto",
            )

        # Estimate max distance using sampling (Phase 1 optimization)
        if self.normalize and len(self._dataset_descriptors) > 1:
            self._max_dataset_distance = self._estimate_max_distance_sampling()
        else:
            self._max_dataset_distance = 1.0

    def _estimate_max_distance_sampling(self) -> float:
        """Estimate maximum pairwise distance using sampling.
        
        Uses a random sample of the dataset to estimate the maximum distance,
        providing O((αN)²) complexity instead of O(N²) for exact computation,
        where α is the sample ratio (default 0.1).
        
        Returns:
            Estimated maximum distance, with 20% buffer for conservative
            normalization.
        """
        descriptors = list(self._dataset_descriptors.values())
        n_samples = len(descriptors)
        
        # Use exact computation for small datasets
        if n_samples <= 1000 or self.max_distance_sample_ratio >= 1.0:
            return self._compute_max_pairwise_distance_exact(descriptors)
        
        # Sample for large datasets
        sample_size = max(
            int(n_samples * self.max_distance_sample_ratio),
            100  # Minimum sample size
        )
        sample = random.sample(descriptors, min(sample_size, n_samples))
        
        max_dist = 0.0
        for i, desc_i in enumerate(sample):
            for desc_j in sample[i + 1 :]:
                dist = self._euclidean_distance(desc_i, desc_j)
                max_dist = max(max_dist, dist)
        
        # Add 20% buffer for conservative normalization
        return max(max_dist * 1.2, 1.0)

    def _compute_max_pairwise_distance_exact(
        self, 
        descriptors: list[tuple[float, ...]]
    ) -> float:
        """Compute exact maximum pairwise distance."""
        max_dist = 0.0
        for i, desc_i in enumerate(descriptors):
            for desc_j in descriptors[i + 1 :]:
                dist = self._euclidean_distance(desc_i, desc_j)
                max_dist = max(max_dist, dist)
        return max(max_dist, 1.0)

    def estimate(self, structure: StructureRecord) -> float:
        """Estimate uncertainty based on distance to nearest dataset sample.

        Uses ANN index for O(log N) query time instead of O(N) linear scan.

        Args:
            structure: The candidate structure to evaluate.

        Returns:
            Uncertainty score between 0 and 1, where higher values indicate
            the structure is farther from the existing dataset.
        """
        if not self._dataset_descriptors or self._nn_index is None:
            # No dataset, maximum uncertainty
            return 1.0

        # Encode candidate structure
        candidate_desc = self.adapter.encode(structure)

        # Query ANN index for nearest neighbor (O(log N) instead of O(N))
        min_distance = self._nn_index.query_nearest(candidate_desc)

        # Normalize to [0, 1]
        if self.normalize and self._max_dataset_distance > 0:
            normalized = min(min_distance / self._max_dataset_distance, 1.0)
        else:
            # Without normalization, use sigmoid-like compression
            normalized = 1.0 - math.exp(-min_distance)

        return normalized

    @staticmethod
    def _euclidean_distance(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        """Compute Euclidean distance between two descriptor vectors.

        Args:
            left: First descriptor vector.
            right: Second descriptor vector.

        Returns:
            Euclidean distance.
        """
        # Pad to same dimension
        dim = max(len(left), len(right))
        left_padded = left + (0.0,) * (dim - len(left))
        right_padded = right + (0.0,) * (dim - len(right))

        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(left_padded, right_padded))
        )


class NearestNeighborsIndex:
    """Approximate nearest neighbor index using scikit-learn.
    
    Wraps scikit-learn's NearestNeighbors to provide efficient O(log N)
    nearest neighbor queries for descriptor-based similarity search.
    
    Automatically selects optimal algorithm based on data characteristics:
    - BallTree: Efficient for low-dimensional data (default for < 50 dims)
    - KDTree: Good for moderate dimensions
    - Brute: Fallback for high dimensions
    """

    def __init__(
        self,
        descriptors: list[tuple[float, ...]],
        algorithm: str = "auto",
        metric: str = "euclidean",
    ) -> None:
        """Initialize the ANN index.

        Args:
            descriptors: List of descriptor vectors to index.
            algorithm: Algorithm to use ('auto', 'ball_tree', 'kd_tree', 'brute').
                'auto' lets scikit-learn choose based on data characteristics.
            metric: Distance metric to use.
        """
        self.descriptors = descriptors
        self.algorithm = algorithm
        self.metric = metric
        self._index: "NearestNeighbors | None" = None
        self._build_index()

    def _build_index(self) -> None:
        """Build the underlying nearest neighbors index."""
        if not self.descriptors:
            return

        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Convert to numpy array for scikit-learn
            import numpy as np
            X = np.array(self.descriptors, dtype=np.float32)
            
            self._index = NearestNeighbors(
                n_neighbors=1,
                algorithm=self.algorithm,
                metric=self.metric,
                n_jobs=-1,  # Use all CPU cores
            )
            self._index.fit(X)
        except ImportError:
            # scikit-learn not available, fall back to linear scan
            self._index = None

    def query_nearest(self, descriptor: tuple[float, ...]) -> float:
        """Find distance to nearest neighbor.

        Args:
            descriptor: Query descriptor vector.

        Returns:
            Distance to nearest neighbor in the index.
        """
        if self._index is not None:
            # Use ANN index for O(log N) query
            import numpy as np
            X = np.array([descriptor], dtype=np.float32)
            distances, _ = self._index.kneighbors(X)
            return float(distances[0][0])
        else:
            # Fall back to linear scan
            min_dist = float("inf")
            for desc in self.descriptors:
                dist = self._euclidean_distance(descriptor, desc)
                min_dist = min(min_dist, dist)
            return min_dist

    @staticmethod
    def _euclidean_distance(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        """Compute Euclidean distance."""
        dim = max(len(left), len(right))
        left_padded = left + (0.0,) * (dim - len(left))
        right_padded = right + (0.0,) * (dim - len(right))
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(left_padded, right_padded))
        )


class EnsembleUncertaintyEstimator(UncertaintyEstimator):
    """Uncertainty estimator based on ensemble of models (placeholder).

    This is a placeholder implementation for ensemble-based uncertainty.
    In practice, this would load multiple model checkpoints and compute
    prediction variance across the ensemble.

    Note: This requires SchNetPack model loading infrastructure which is
    not yet fully integrated. For now, use DistanceBasedUncertaintyEstimator.
    """

    def __init__(self, checkpoint_paths: list[str | Path]) -> None:
        """Initialize with model checkpoint paths.

        Args:
            checkpoint_paths: Paths to model checkpoint files.
        """
        self.checkpoint_paths = checkpoint_paths

    def estimate(self, structure: StructureRecord) -> float:
        """Estimate uncertainty using model ensemble.

        Currently raises NotImplementedError as ensemble loading is not
        yet integrated. Use DistanceBasedUncertaintyEstimator instead.

        Args:
            structure: The structure to evaluate.

        Raises:
            NotImplementedError: Ensemble uncertainty is not yet implemented.
        """
        raise NotImplementedError(
            "Ensemble uncertainty estimation requires SchNetPack model loading. "
            "Use DistanceBasedUncertaintyEstimator as a fallback."
        )
