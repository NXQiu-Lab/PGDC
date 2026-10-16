"""Acquisition mode for dynamic dataset supplementation guidance.

This module provides utilities to evaluate the information gain value of
candidate structures for active learning workflows. It estimates the value
of running AIMD simulations from given starting structures based on:

- Model uncertainty estimation
- Coverage gap analysis in descriptor space
- Trajectory diversity prediction from VASP INCAR parameters

Example usage:
    >>> from atom_dataset_index.acquisition import AcquisitionEvaluator
    >>> from atom_dataset_index.io import AseDbDatasetReader, PoscarReader
    >>> 
    >>> # Load original dataset
    >>> dataset = AseDbDatasetReader("data/existing.db").read()
    >>> 
    >>> # Load candidate structures
    >>> candidates = PoscarReader().read_directory("candidates/")
    >>> 
    >>> # Evaluate acquisition value
    >>> evaluator = AcquisitionEvaluator(dataset)
    >>> report = evaluator.evaluate(candidates, top_k=5)
    >>> 
    >>> # Get ranked recommendations
    >>> for ranking in report.rankings:
    ...     print(f"{ranking.poscar_path}: {ranking.total_value:.3f}")
"""

from .interfaces import AcquisitionValue, UncertaintyEstimator
from .report import AcquisitionRanking, AcquisitionReport
from .uncertainty import DistanceBasedUncertaintyEstimator
from .evaluator import AcquisitionEvaluator

__all__ = [
    # Interfaces
    "AcquisitionValue",
    "UncertaintyEstimator",
    # Report
    "AcquisitionRanking",
    "AcquisitionReport",
    # Uncertainty estimators
    "DistanceBasedUncertaintyEstimator",
    # Main evaluator
    "AcquisitionEvaluator",
]
