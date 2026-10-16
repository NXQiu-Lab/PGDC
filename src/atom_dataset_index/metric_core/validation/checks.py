"""Validation helpers."""

from __future__ import annotations

from atom_dataset_index.domain import DatasetBundle
from atom_dataset_index.metric_core.results import MetricResult


def ensure_non_empty_dataset(dataset: DatasetBundle) -> None:
    if not dataset.records:
        raise ValueError("Dataset contains no records.")


def ensure_energy_available(dataset: DatasetBundle) -> None:
    if not any(record.energy is not None for record in dataset.records):
        raise ValueError("Dataset contains no usable energy labels.")


def ensure_positions_available(dataset: DatasetBundle) -> None:
    if not any(record.positions for record in dataset.records):
        raise ValueError("Dataset contains no usable position arrays.")


def ensure_forces_available(dataset: DatasetBundle) -> None:
    if not any(record.forces for record in dataset.records):
        raise ValueError("Dataset contains no usable force arrays.")


def ensure_constant_natoms(dataset: DatasetBundle) -> None:
    natoms_values = {record.natoms for record in dataset.records}
    if len(natoms_values) > 1:
        raise ValueError("Dataset contains inconsistent atom counts; v1 metrics require constant natoms.")


def ensure_periodic_dataset(dataset: DatasetBundle) -> None:
    if not dataset.records:
        raise ValueError("Dataset contains no records.")
    for record in dataset.records:
        if record.cell is None or not any(record.pbc):
            raise ValueError("Dataset is not fully periodic; v1 prototype metrics require periodic structures.")


def ensure_unique_sample_ids(dataset: DatasetBundle) -> None:
    sample_ids = [record.sample_id for record in dataset.records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Dataset contains duplicated sample ids.")


def ensure_metric_result_consistent(result: MetricResult) -> None:
    expected = tuple(result.spec.output_fields)
    for record in result.sample_records:
        field_names = tuple(record.values)
        if field_names != expected:
            raise ValueError(
                f"Metric '{result.spec.metric_id}' produced sample fields {field_names}, "
                f"expected {expected}."
            )
