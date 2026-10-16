# atom_dataset_index.acquisition

## Metadata

- `module_id`: `atom_dataset_index.acquisition`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/acquisition/evaluator.py`, `src/atom_dataset_index/acquisition/interfaces.py`, `src/atom_dataset_index/acquisition/report.py`, `src/atom_dataset_index/acquisition/uncertainty.py`
- `upstream_modules`: `none`
- `downstream_modules`: `none`

## Module Purpose

Package leaf module for acquisition.

## Public Symbols

- `check_element_compatibility(candidate, dataset_elements, mode)`
  Check whether a candidate structure is chemically compatible with the dataset.

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
- `parse_incar_optional(path)`
  Helper to optionally parse INCAR.
- `class AcquisitionEvaluator`
  Main evaluator for acquisition mode.

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
- `class AcquisitionValue`
  Value metrics for a candidate structure.

Attributes:
    total: Combined acquisition value (0-1, higher is better).
    uncertainty: Model uncertainty proxy (0-1, higher means more uncertain).
    coverage_gap: Descriptor space coverage gap (0-1, higher means larger gap).
    trajectory_diversity: Estimated trajectory diversity (0-1, higher is better).
- `class UncertaintyEstimator`
  Abstract base class for MLIP prediction uncertainty estimation.

Implementations should provide a way to estimate how uncertain the model
is about its prediction for a given structure. Higher values indicate
greater uncertainty and potentially higher information gain.
- `class InformationGainCalculator`
  Abstract base class for information gain calculation.

Implementations compute the expected information gain from running an
AIMD simulation starting from a given candidate structure.
- `class AcquisitionRanking`
  Ranking information for a single candidate structure.

Attributes:
    poscar_path: Path to the POSCAR file for this candidate.
    sample_id: Sample ID assigned during reading.
    total_value: Combined acquisition value (0-1).
    uncertainty: Model uncertainty proxy (0-1).
    coverage_gap: Descriptor space coverage gap (0-1).
    trajectory_diversity: Estimated trajectory diversity (0-1).
- `class AcquisitionReport`
  Complete acquisition evaluation report.

Attributes:
    rankings: List of ranked candidates (sorted by total_value descending).
    summary: Dictionary of summary statistics.
    original_dataset_name: Name of the original dataset.
    original_sample_count: Number of samples in original dataset.
    candidate_count: Total number of candidates submitted (including rejected).
    incompatible_candidates: Candidates rejected by element compatibility check.
        Each entry is a dict with keys ``poscar_path``, ``sample_id``,
        ``elements`` (sorted list of atomic numbers), and ``reason``.
- `class ReportExporter`
  Exporter for acquisition reports in various formats.
- `class DistanceBasedUncertaintyEstimator`
  Uncertainty estimator based on descriptor distance to dataset.

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
- `class NearestNeighborsIndex`
  Approximate nearest neighbor index using scikit-learn.

Wraps scikit-learn's NearestNeighbors to provide efficient O(log N)
nearest neighbor queries for descriptor-based similarity search.

Automatically selects optimal algorithm based on data characteristics:
- BallTree: Efficient for low-dimensional data (default for < 50 dims)
- KDTree: Good for moderate dimensions
- Brute: Fallback for high dimensions
- `class EnsembleUncertaintyEstimator`
  Uncertainty estimator based on ensemble of models (placeholder).

This is a placeholder implementation for ensemble-based uncertainty.
In practice, this would load multiple model checkpoints and compute
prediction variance across the ensemble.

Note: This requires SchNetPack model loading infrastructure which is
not yet fully integrated. For now, use DistanceBasedUncertaintyEstimator.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
