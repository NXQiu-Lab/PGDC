"""Main acquisition evaluator for dynamic dataset supplementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import DescriptorAdapter, SOAPDescriptorAdapter

from .interfaces import AcquisitionValue, InformationGainCalculator, UncertaintyEstimator
from .report import AcquisitionRanking, AcquisitionReport
from .uncertainty import DistanceBasedUncertaintyEstimator, NearestNeighborsIndex

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Sequence

    from atom_dataset_index.io import IncarParams

_VALID_ELEMENT_CHECK_MODES = frozenset({"strict", "overlap", "none"})


def check_element_compatibility(
    candidate: StructureRecord,
    dataset_elements: frozenset[int],
    mode: str = "strict",
) -> tuple[bool, str]:
    """Check whether a candidate structure is chemically compatible with the dataset.

    Args:
        candidate: The candidate structure to check.
        dataset_elements: Frozen set of atomic numbers present in the dataset.
        mode: Compatibility mode.
            - ``"strict"``: candidate elements must be a subset of dataset elements.
              The candidate may contain fewer element types than the dataset, but
              cannot introduce any element not already in the dataset.
            - ``"overlap"``: candidate and dataset must share at least one element.
            - ``"none"``: no check, always compatible (original behaviour).

    Returns:
        A tuple ``(is_compatible, reason)`` where ``reason`` is an empty string
        when compatible and a human-readable explanation when incompatible.

    Raises:
        ValueError: If *mode* is not one of the recognised values.
    """
    if mode not in _VALID_ELEMENT_CHECK_MODES:
        raise ValueError(
            f"Unknown element_check mode {mode!r}. "
            f"Choose from {sorted(_VALID_ELEMENT_CHECK_MODES)}."
        )

    if mode == "none":
        return True, ""

    candidate_elements = frozenset(candidate.atomic_numbers)

    if mode == "strict":
        extra = candidate_elements - dataset_elements
        if extra:
            return (
                False,
                f"Candidate contains element(s) not in dataset (atomic numbers: "
                f"{sorted(extra)}). Dataset elements: {sorted(dataset_elements)}.",
            )
        return True, ""

    # mode == "overlap"
    shared = candidate_elements & dataset_elements
    if not shared:
        return (
            False,
            f"Candidate shares no elements with dataset. "
            f"Candidate: {sorted(candidate_elements)}, "
            f"Dataset: {sorted(dataset_elements)}.",
        )
    return True, ""


def parse_incar_optional(path: "str | Path | None") -> "IncarParams | None":
    """Helper to optionally parse INCAR."""
    if path is None:
        return None
    try:
        from atom_dataset_index.io import IncarParser
        return IncarParser().parse(path)
    except Exception:
        return None


class _InformationGainCalculator(InformationGainCalculator):
    """Concrete implementation of information gain calculation.
    
    Uses ANN (Approximate Nearest Neighbor) indexing for O(log N) coverage gap
    queries instead of O(N) linear scan, providing significant speedup for
    large datasets.
    """

    def __init__(
        self,
        dataset: DatasetBundle,
        uncertainty_estimator: UncertaintyEstimator,
        adapter: DescriptorAdapter | None = None,
        weights: dict[str, float] | None = None,
        max_distance_sample_ratio: float = 0.1,
    ) -> None:
        """Initialize calculator.

        Args:
            dataset: Original dataset for coverage gap calculation.
            uncertainty_estimator: Estimator for model uncertainty.
            adapter: Descriptor adapter. Defaults to SOAPDescriptorAdapter.
            weights: Component weights for total value. Defaults to
                {"uncertainty": 0.4, "coverage_gap": 0.4, "trajectory": 0.2}.
            max_distance_sample_ratio: Ratio for sampling-based max distance
                estimation. Lower values improve initialization speed.
        """
        self.dataset = dataset
        self.uncertainty_estimator = uncertainty_estimator
        self.adapter = adapter or SOAPDescriptorAdapter()
        self.weights = weights or {
            "uncertainty": 0.4,
            "coverage_gap": 0.4,
            "trajectory": 0.2,
        }
        self.max_distance_sample_ratio = max_distance_sample_ratio

        # Build dataset descriptor index and ANN index for coverage gap
        self._dataset_descriptors: list[tuple[float, ...]] = []
        self._max_dataset_distance = 0.0
        self._nn_index: NearestNeighborsIndex | None = None
        self._build_dataset_index()

    def _build_dataset_index(self) -> None:
        """Precompute dataset descriptors and build ANN index."""
        # Use batch encoding for better performance (Phase 1 optimization)
        self._dataset_descriptors = self.adapter.encode_batch(
            list(self.dataset.records)
        )

        # Build ANN index for fast nearest neighbor queries (Phase 1 optimization)
        if len(self._dataset_descriptors) > 0:
            self._nn_index = NearestNeighborsIndex(
                self._dataset_descriptors,
                algorithm="auto",
            )

        # Estimate max distance using sampling (Phase 1 optimization)
        if len(self._dataset_descriptors) > 1:
            self._max_dataset_distance = self._estimate_max_distance_sampling()
        else:
            self._max_dataset_distance = 1.0

    def _estimate_max_distance_sampling(self) -> float:
        """Estimate maximum pairwise distance using sampling.
        
        Uses random sampling to reduce complexity from O(N²) to O((αN)²).
        """
        import random
        
        n_samples = len(self._dataset_descriptors)
        
        # Use exact computation for small datasets
        if n_samples <= 1000 or self.max_distance_sample_ratio >= 1.0:
            return self._compute_max_distance_exact()
        
        # Sample for large datasets
        sample_size = max(
            int(n_samples * self.max_distance_sample_ratio),
            100
        )
        sample = random.sample(self._dataset_descriptors, min(sample_size, n_samples))
        
        max_dist = 0.0
        for i, desc_i in enumerate(sample):
            for desc_j in sample[i + 1 :]:
                dist = self._euclidean_distance(desc_i, desc_j)
                max_dist = max(max_dist, dist)
        
        # Add 20% buffer for conservative normalization
        return max(max_dist * 1.2, 1.0)

    def _compute_max_distance_exact(self) -> float:
        """Compute exact maximum pairwise distance."""
        max_dist = 0.0
        for i, desc_i in enumerate(self._dataset_descriptors):
            for desc_j in self._dataset_descriptors[i + 1 :]:
                dist = self._euclidean_distance(desc_i, desc_j)
                max_dist = max(max_dist, dist)
        return max(max_dist, 1.0)

    def compute(
        self,
        candidate: StructureRecord,
        trajectory_diversity: float = 0.5,
    ) -> AcquisitionValue:
        """Compute acquisition value for a candidate structure."""
        # 1. Uncertainty value
        uncertainty = self.uncertainty_estimator.estimate(candidate)

        # 2. Coverage gap (using ANN index for O(log N) query)
        coverage_gap = self._compute_coverage_gap(candidate)

        # 3. Trajectory diversity (already passed in)

        # 4. Combined value
        total = (
            self.weights["uncertainty"] * uncertainty +
            self.weights["coverage_gap"] * coverage_gap +
            self.weights["trajectory"] * trajectory_diversity
        )

        return AcquisitionValue(
            total=total,
            uncertainty=uncertainty,
            coverage_gap=coverage_gap,
            trajectory_diversity=trajectory_diversity,
        )

    def _compute_coverage_gap(self, candidate: StructureRecord) -> float:
        """Compute normalized distance to nearest dataset sample.
        
        Uses ANN index for O(log N) query instead of O(N) linear scan.
        """
        if not self._dataset_descriptors or self._nn_index is None:
            return 1.0  # Maximum gap if no dataset

        candidate_desc = self.adapter.encode(candidate)
        
        # Query ANN index for nearest neighbor (O(log N) instead of O(N))
        min_dist = self._nn_index.query_nearest(candidate_desc)

        # Normalize
        normalized = min(min_dist / self._max_dataset_distance, 1.0)
        return normalized

    @staticmethod
    def _euclidean_distance(
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        """Compute Euclidean distance."""
        import math
        dim = max(len(left), len(right))
        left_padded = left + (0.0,) * (dim - len(left))
        right_padded = right + (0.0,) * (dim - len(right))
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(left_padded, right_padded))
        )


class AcquisitionEvaluator:
    """Main evaluator for acquisition mode.

    This class orchestrates the evaluation of candidate structures for
    dataset supplementation. It combines uncertainty estimation, coverage
    gap analysis, and trajectory diversity prediction to rank candidates.

    Before scoring, candidates are optionally filtered by element compatibility
    to prevent chemically incompatible structures (e.g. H2O for an HfO2 dataset)
    from being incorrectly recommended due to artificially inflated descriptor
    distances.
    
    Phase 1 Performance Optimizations:
    - Batch descriptor encoding reduces Python call overhead
    - ANN (Approximate Nearest Neighbor) indexing provides O(log N) queries
    - Sampling-based max distance estimation reduces O(N²) to O((αN)²)

    Example:
        >>> from atom_dataset_index.acquisition import AcquisitionEvaluator
        >>> from atom_dataset_index.io import AseDbDatasetReader, PoscarReader
        >>> 
        >>> dataset = AseDbDatasetReader("data/existing.db"read()
        >>> evaluator = AcquisitionEvaluator(dataset, element_check="strict")
        >>> 
        >>> candidates = PoscarReader().read_directory("candidates/")
        >>> report = evaluator.evaluate(candidates, top_k=5)
        >>> 
        >>> for r in report.rankings:
        ...     print(f"{r.poscar_path}: {r.total_value:.3f}")
        >>> for item in report.incompatible_candidates:
        ...     print(f"REJECTED {item['poscar_path']}: {item['reason']}")
    """

    def __init__(
        self,
        original_dataset: DatasetBundle,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        adapter: DescriptorAdapter | None = None,
        weights: dict[str, float] | None = None,
        element_check: str = "strict",
        max_distance_sample_ratio: float = 0.1,
    ) -> None:
        """Initialize the evaluator.

        Args:
            original_dataset: The existing dataset to supplement.
            uncertainty_estimator: Uncertainty estimator. Defaults to
                DistanceBasedUncertaintyEstimator.
            adapter: Descriptor adapter. Defaults to SOAPDescriptorAdapter.
            weights: Component weights. Defaults to balanced weights.
            element_check: Element compatibility mode applied before scoring.
                ``"strict"`` (default) requires candidate elements ⊆ dataset
                elements. ``"overlap"`` requires at least one shared element.
                ``"none"`` disables the check (original behaviour).
            max_distance_sample_ratio: Ratio for sampling-based max distance
                estimation (0.1 = 10% sample). Lower values improve 
                initialization speed with minimal impact on results.

        Raises:
            ValueError: If *element_check* is not a recognised mode.
        """
        if element_check not in _VALID_ELEMENT_CHECK_MODES:
            raise ValueError(
                f"Unknown element_check mode {element_check!r}. "
                f"Choose from {sorted(_VALID_ELEMENT_CHECK_MODES)}."
            )
        self.dataset = original_dataset
        self.element_check = element_check
        self.adapter = adapter or SOAPDescriptorAdapter()
        self.max_distance_sample_ratio = max_distance_sample_ratio
        
        # Initialize uncertainty estimator with sampling optimization
        self.uncertainty_estimator = uncertainty_estimator or DistanceBasedUncertaintyEstimator(
            original_dataset, 
            adapter=self.adapter,
            max_distance_sample_ratio=max_distance_sample_ratio,
        )
        
        # Initialize calculator with sampling optimization
        self.calculator = _InformationGainCalculator(
            dataset=original_dataset,
            uncertainty_estimator=self.uncertainty_estimator,
            adapter=self.adapter,
            weights=weights,
            max_distance_sample_ratio=max_distance_sample_ratio,
        )
        
        # Precompute dataset element set once
        self._dataset_elements: frozenset[int] = frozenset(
            z for record in original_dataset.records for z in record.atomic_numbers
        )

    def evaluate(
        self,
        candidates: "Sequence[StructureRecord]",
        vasp_config_path: "str | Path | None" = None,
        top_k: int = 10,
    ) -> AcquisitionReport:
        """Evaluate candidate structures and generate acquisition report.

        Candidates that fail the element compatibility check are collected in
        ``AcquisitionReport.incompatible_candidates`` and excluded from scoring
        and ranking entirely.

        Args:
            candidates: List of candidate structures to evaluate.
            vasp_config_path: Optional path to INCAR file for trajectory
                diversity estimation.
            top_k: Number of top candidates to include in report.

        Returns:
            AcquisitionReport containing rankings, summary, and any
            incompatible candidates that were filtered out.
        """
        # Parse VASP config if provided
        vasp_config = parse_incar_optional(vasp_config_path)

        # Get trajectory diversity from INCAR or use default
        if vasp_config is not None:
            trajectory_diversity = vasp_config.estimate_diversity_proxy()
        else:
            trajectory_diversity = 0.5

        # Split candidates into compatible and incompatible
        compatible: list[StructureRecord] = []
        incompatible: list[dict] = []

        for candidate in candidates:
            poscar_path = str(candidate.metadata.get("poscar_path", f"sample_{candidate.sample_id}"))
            ok, reason = check_element_compatibility(
                candidate, self._dataset_elements, mode=self.element_check
            )
            if ok:
                compatible.append(candidate)
            else:
                incompatible.append({
                    "poscar_path": poscar_path,
                    "sample_id": candidate.sample_id,
                    "elements": sorted(set(candidate.atomic_numbers)),
                    "reason": reason,
                })

        # Compute value for each compatible candidate
        rankings: list[AcquisitionRanking] = []
        for candidate in compatible:
            value = self.calculator.compute(candidate, trajectory_diversity)
            poscar_path = str(candidate.metadata.get("poscar_path", f"sample_{candidate.sample_id}"))
            rankings.append(
                AcquisitionRanking(
                    poscar_path=poscar_path,
                    sample_id=candidate.sample_id,
                    total_value=value.total,
                    uncertainty=value.uncertainty,
                    coverage_gap=value.coverage_gap,
                    trajectory_diversity=value.trajectory_diversity,
                )
            )

        # Sort by total value descending
        rankings.sort(key=lambda r: r.total_value, reverse=True)

        # Generate summary
        summary = self._generate_summary(rankings, len(candidates), vasp_config)

        return AcquisitionReport(
            rankings=rankings[:top_k],
            summary=summary,
            original_dataset_name=self.dataset.dataset_name,
            original_sample_count=len(self.dataset.records),
            candidate_count=len(candidates),
            incompatible_candidates=incompatible,
        )

    def _generate_summary(
        self,
        rankings: list[AcquisitionRanking],
        total_candidates: int,
        vasp_config: IncarParams | None,
    ) -> dict:
        """Generate summary statistics."""
        if not rankings:
            return {
                "avg_total_value": 0.0,
                "avg_uncertainty": 0.0,
                "avg_coverage_gap": 0.0,
                "avg_trajectory_diversity": 0.0,
                "high_value_candidates": 0,
                "medium_value_candidates": 0,
                "low_value_candidates": 0,
            }

        # Count candidates by value range
        high_value = sum(1 for r in rankings if r.total_value >= 0.7)
        medium_value = sum(1 for r in rankings if 0.4 <= r.total_value < 0.7)
        low_value = sum(1 for r in rankings if r.total_value < 0.4)

        # Average values
        avg_total = sum(r.total_value for r in rankings) / len(rankings)
        avg_uncertainty = sum(r.uncertainty for r in rankings) / len(rankings)
        avg_coverage = sum(r.coverage_gap for r in rankings) / len(rankings)
        avg_trajectory = sum(r.trajectory_diversity for r in rankings) / len(rankings)

        summary = {
            "avg_total_value": round(avg_total, 4),
            "avg_uncertainty": round(avg_uncertainty, 4),
            "avg_coverage_gap": round(avg_coverage, 4),
            "avg_trajectory_diversity": round(avg_trajectory, 4),
            "high_value_candidates": high_value,
            "medium_value_candidates": medium_value,
            "low_value_candidates": low_value,
        }

        # Add VASP config info if available
        if vasp_config is not None:
            summary["vasp_temperature"] = vasp_config.effective_temperature
            summary["vasp_steps"] = vasp_config.effective_steps
            summary["vasp_total_time_ps"] = round(vasp_config.total_simulation_time, 2)

        return summary
