"""Energy coverage metrics."""

from __future__ import annotations

from atom_dataset_index.domain import DatasetBundle
from atom_dataset_index.metric_core.interfaces import BaseMetricCalculator, MetricContext, MetricSpec
from atom_dataset_index.metric_core.results import AggregateMetricRecord, MetricResult, SampleMetricRecord
from atom_dataset_index.metric_core.scaling import auto_scale_linear
from atom_dataset_index.metric_core.statistics import mean, quantiles, variance
from atom_dataset_index.metric_core.validation import ensure_energy_available, ensure_non_empty_dataset

# When a per-pool energy range exceeds this threshold, the calculator
# assumes unusually large energy coverage and applies a uniform linear
# scale factor so that the range equals the threshold.  All
# distributional shape (variance ratios, quantiles, high-fraction
# proportions) is preserved.
_ENERGY_RANGE_THRESHOLD_EVA: float = 1.0  # eV/atom


class EnergyCoverageCalculator(BaseMetricCalculator):
    """Compute simple energy coverage indicators."""

    spec = MetricSpec(
        metric_id="energy.coverage",
        display_name="Energy Coverage",
        required_fields=("energy",),
        output_fields=("energy_per_atom", "delta_energy_per_atom"),
        description="Per-atom relative energy range, variance, and upper-tail coverage.",
    )

    def calculate(self, dataset: DatasetBundle, context: MetricContext) -> MetricResult:
        del context
        ensure_non_empty_dataset(dataset)
        ensure_energy_available(dataset)

        # Pair sample_id with per-atom energy while filtering missing values.
        paired = [
            (record.sample_id, float(record.energy) / max(record.natoms, 1))
            for record in dataset.records
            if record.energy is not None
        ]

        energies_per_atom = [e for _, e in paired]
        # Energy zero-point correction: shift energies so the minimum is at
        # zero before evaluating the range.  This prevents absolute energy
        # offsets (e.g., VASP total energies with core-electron
        # contributions) from causing spurious scaling decisions.
        min_energy = min(energies_per_atom)
        relative_energies = [e - min_energy for e in energies_per_atom]
        actual_range = max(relative_energies)  # equivalent to max - min

        # Adaptive linear scaling based on relative energy range.
        scale_info = None
        if actual_range > _ENERGY_RANGE_THRESHOLD_EVA:
            energies_per_atom, scale_info = auto_scale_linear(
                energies_per_atom,
                actual_range,
                _ENERGY_RANGE_THRESHOLD_EVA,
                "energy_per_atom_range",
                self.spec.metric_id,
            )
            # Re-pair scaled energies with their sample_ids.
            paired = [(sid, e) for (sid, _), e in zip(paired, energies_per_atom)]

        baseline = min(energies_per_atom)
        sample_records = tuple(
            SampleMetricRecord(
                sample_id=sid,
                values={
                    "energy_per_atom": e,
                    "delta_energy_per_atom": e - baseline,
                },
            )
            for sid, e in paired
        )

        relative = [e - baseline for e in energies_per_atom]
        q90 = quantiles(relative, (0.9,)).get("0.9", 0.0)

        diagnostics: dict[str, object] = {}
        if scale_info is not None:
            diagnostics["auto_scaling"] = {
                "applied": True,
                "metric": scale_info.metric_name,
                "original_range": scale_info.original_value,
                "threshold": scale_info.threshold,
                "scale_factor": scale_info.scale_factor,
            }

        aggregate = AggregateMetricRecord(
            values={
                "energy_per_atom_range": max(energies_per_atom) - min(energies_per_atom),
                "energy_per_atom_mean": mean(energies_per_atom),
                "energy_per_atom_variance": variance(energies_per_atom),
                "delta_energy_per_atom_q90": q90,
                "delta_energy_high_fraction": (
                    sum(1 for value in relative if value > q90) / len(relative) if relative else 0.0
                ),
            },
            diagnostics=diagnostics,
            aggregation_policy={
                "energy_per_atom_range": "max",
                "energy_per_atom_mean": "weighted_mean",
                "energy_per_atom_variance": "weighted_mean",
                "delta_energy_per_atom_q90": "weighted_mean",
                "delta_energy_high_fraction": "weighted_mean",
            },
        )
        return MetricResult(spec=self.spec, sample_records=sample_records, aggregate=aggregate)
