"""Registry for metric calculators."""

from __future__ import annotations

from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator


class MetricRegistry:
    """In-memory registry keyed by metric id."""

    def __init__(self) -> None:
        self._calculators: dict[str, BaseMetricCalculator] = {}

    def __contains__(self, metric_id: object) -> bool:
        return isinstance(metric_id, str) and metric_id in self._calculators

    def __len__(self) -> int:
        return len(self._calculators)

    def register(self, calculator: BaseMetricCalculator) -> None:
        metric_id = calculator.spec.metric_id
        if not metric_id.strip():
            raise ValueError("Metric id must not be blank.")
        if metric_id in self._calculators:
            raise ValueError(f"Metric already registered: {metric_id}")
        self._calculators[metric_id] = calculator

    def register_many(self, calculators: tuple[BaseMetricCalculator, ...]) -> None:
        for calculator in calculators:
            self.register(calculator)

    def is_registered(self, metric_id: str) -> bool:
        return metric_id in self._calculators

    def get(self, metric_id: str) -> BaseMetricCalculator:
        try:
            return self._calculators[metric_id]
        except KeyError as error:
            raise KeyError(f"Unknown metric: {metric_id}") from error

    def list_metric_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._calculators))
