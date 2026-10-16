"""Tests for benchmark orchestration behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.benchmark import (
    BenchmarkRunner,
    DatasetSourceSpec,
    LoadedDatasetSpec,
    TrainingTaskSpec,
    list_registered_sources,
    register_source,
)
from atom_dataset_index.domain import BenchmarkResult, DatasetBundle, DatasetProfile, StructureRecord, TrainingResult


class _FakeRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls = 0

    def run(self, task: TrainingTaskSpec, loaded: LoadedDatasetSpec) -> TrainingResult:
        self.calls += 1
        return TrainingResult(
            model_name=self.model_name,
            split_name=task.split_name,
            metrics={"energy_mae": 0.1, "force_mae": 0.2},
            artifact_paths={"training_manifest": str(Path(task.output_dir) / "fake.json")},
            metadata={"head_mode": loaded.head_mode, "source_type": loaded.source_type},
        )


def _toy_context() -> tuple[DatasetBundle, DatasetProfile]:
    dataset = DatasetBundle(
        dataset_name="toy",
        records=(StructureRecord(sample_id=1, atomic_numbers=(1,), positions=((0.0, 0.0, 0.0),)),),
    )
    profile = DatasetProfile(dataset_name="toy", metric_results={}, summary={"metric_count": 0})
    return dataset, profile


class BenchmarkRunnerTest(unittest.TestCase):
    def test_default_sources_registered(self) -> None:
        names = set(list_registered_sources())
        self.assertTrue({"ase_db", "md17", "md22"}.issubset(names))

    def test_benchmark_rejects_unknown_model(self) -> None:
        dataset, profile = _toy_context()
        with self.assertRaises(ValueError):
            BenchmarkRunner().run(
                benchmark_name="sanity",
                dataset=dataset,
                profile=profile,
                tasks=(
                    TrainingTaskSpec(
                        model_name="UnknownModel",
                        split_name="train",
                        output_dir="/tmp",
                        source=DatasetSourceSpec(source_type="ase_db", datapath="toy.db"),
                        dry_run=True,
                    ),
                ),
            )

    def test_require_forces_rejects_missing_force_labels(self) -> None:
        source_name = "mock_no_forces_required"

        def _builder(spec: DatasetSourceSpec, task: TrainingTaskSpec) -> LoadedDatasetSpec:
            del spec, task
            return LoadedDatasetSpec(
                datamodule=object(),
                energy_key="energy",
                force_key=None,
                available_properties=("energy",),
                source_type=source_name,
                source_name="mock",
            )

        register_source(source_name, _builder)
        dataset, profile = _toy_context()
        fake_runner = _FakeRunner("SchNet")
        with self.assertRaises(ValueError):
            BenchmarkRunner(runners={"SchNet": fake_runner}).run(
                benchmark_name="no_forces",
                dataset=dataset,
                profile=profile,
                tasks=(
                    TrainingTaskSpec(
                        model_name="SchNet",
                        split_name="train",
                        output_dir="/tmp",
                        source=DatasetSourceSpec(source_type=source_name),
                    ),
                ),
            )
        self.assertEqual(fake_runner.calls, 0)

    def test_allow_energy_only_permits_run(self) -> None:
        source_name = "mock_allow_energy_only"

        def _builder(spec: DatasetSourceSpec, task: TrainingTaskSpec) -> LoadedDatasetSpec:
            del spec, task
            return LoadedDatasetSpec(
                datamodule=object(),
                energy_key="energy",
                force_key=None,
                available_properties=("energy",),
                source_type=source_name,
                source_name="mock",
            )

        register_source(source_name, _builder)
        dataset, profile = _toy_context()
        fake_runner = _FakeRunner("SchNet")
        result: BenchmarkResult = BenchmarkRunner(runners={"SchNet": fake_runner}).run(
            benchmark_name="energy_only",
            dataset=dataset,
            profile=profile,
            tasks=(
                TrainingTaskSpec(
                    model_name="SchNet",
                    split_name="train",
                    output_dir="/tmp",
                    source=DatasetSourceSpec(source_type=source_name),
                    allow_energy_only=True,
                ),
            ),
        )
        self.assertEqual(fake_runner.calls, 1)
        self.assertEqual(result.training_results[0].metadata["head_mode"], "energy-only")


if __name__ == "__main__":
    unittest.main()
