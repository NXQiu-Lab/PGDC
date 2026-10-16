"""Abstract interfaces for acquisition mode components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from atom_dataset_index.domain import StructureRecord


@dataclass(frozen=True, slots=True)
class AcquisitionValue:
    """Value metrics for a candidate structure.

    Attributes:
        total: Combined acquisition value (0-1, higher is better).
        uncertainty: Model uncertainty proxy (0-1, higher means more uncertain).
        coverage_gap: Descriptor space coverage gap (0-1, higher means larger gap).
        trajectory_diversity: Estimated trajectory diversity (0-1, higher is better).
    """

    total: float
    uncertainty: float
    coverage_gap: float
    trajectory_diversity: float


class UncertaintyEstimator(ABC):
    """Abstract base class for MLIP prediction uncertainty estimation.

    Implementations should provide a way to estimate how uncertain the model
    is about its prediction for a given structure. Higher values indicate
    greater uncertainty and potentially higher information gain.
    """

    @abstractmethod
    def estimate(self, structure: StructureRecord) -> float:
        """Estimate uncertainty for a single structure.

        Args:
            structure: The structure to evaluate.

        Returns:
            Uncertainty score between 0 and 1, where higher values indicate
            greater uncertainty.
        """
        ...


class InformationGainCalculator(ABC):
    """Abstract base class for information gain calculation.

    Implementations compute the expected information gain from running an
    AIMD simulation starting from a given candidate structure.
    """

    @abstractmethod
    def compute(
        self,
        candidate: StructureRecord,
        trajectory_diversity: float = 0.5,
    ) -> AcquisitionValue:
        """Compute acquisition value for a candidate structure.

        Args:
            candidate: The candidate structure to evaluate.
            trajectory_diversity: Estimated trajectory diversity from INCAR
                parameters (0-1), default 0.5.

        Returns:
            AcquisitionValue containing total value and components.
        """
        ...
