"""Tests for runtime task directory creation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.domain import BenchmarkResult, DatasetBundle, DatasetProfile, StructureRecord, TrainingResult
from atom_dataset_index.cli import main
from atom_dataset_index.tasks import build_task_name, get_task_definition, prepare_output_directory


class TasksTest(unittest.TestCase):
    @staticmethod
    def _mock_profile_dataset() -> DatasetBundle:
        return DatasetBundle(
            dataset_name="mock-profile",
            records=(
                StructureRecord(
                    sample_id=1,
                    atomic_numbers=(1,),
                    positions=((0.0, 0.0, 0.0),),
                    cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    pbc=(True, True, True),
                    energy=-1.0,
                    forces=((0.1, 0.0, 0.0),),
                ),
                StructureRecord(
                    sample_id=2,
                    atomic_numbers=(1,),
                    positions=((0.1, 0.0, 0.0),),
                    cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    pbc=(True, True, True),
                    energy=0.0,
                    forces=((0.2, 0.0, 0.0),),
                ),
            ),
        )

    def test_prepare_output_directory_creates_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = prepare_output_directory(tmpdir, "training", "schnet", run_id="run-001")
            self.assertEqual(output, Path(tmpdir) / "training" / "schnet" / "run-001")
            self.assertTrue(output.exists())

    def test_prepare_run_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main(
                [
                    "prepare-run",
                    "benchmark",
                    "--subtask",
                    "run",
                    "--output-root",
                    tmpdir,
                    "--date",
                    "2026-03-22",
                ]
            )
            self.assertEqual(exit_code, 0)
            run_root = Path(tmpdir) / "benchmark" / "run"
            manifests = list(run_root.glob("*/run_manifest.json"))
            self.assertEqual(len(manifests), 1)
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["task_name"], "benchmark.run")
            self.assertEqual(payload["command"], "prepare-run")
            self.assertTrue(payload["run_id"])

    def test_get_task_definition(self) -> None:
        definition = get_task_definition("dataset_profile")
        self.assertIn("build", definition.subtasks)

    def test_build_task_name(self) -> None:
        self.assertEqual(build_task_name("benchmark", "run"), "benchmark.run")
        self.assertEqual(build_task_name("benchmark", None), "benchmark.default")

    def test_benchmark_run_cli_writes_manifest_with_mock_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_result = BenchmarkResult(
                benchmark_name="mock",
                profiles={"mock": DatasetProfile(dataset_name="mock", metric_results={}, summary={})},
                training_results=(
                    TrainingResult(
                        model_name="SchNet",
                        split_name="default",
                        metrics={"energy_mae": 0.0, "force_mae": 0.0},
                        artifact_paths={
                            "training_manifest": str(Path(tmpdir) / "training_manifest.json"),
                            "metrics_file": str(Path(tmpdir) / "metrics.json"),
                            "energy_error_file": str(Path(tmpdir) / "energy.csv"),
                            "force_error_file": str(Path(tmpdir) / "force.csv"),
                        },
                        metadata={"head_mode": "energy+forces"},
                    ),
                ),
                summary={},
            )
            with patch("atom_dataset_index.cli.BenchmarkRunner.run", return_value=fake_result):
                exit_code = main(
                    [
                        "benchmark-run",
                        "--source-type",
                        "ase_db",
                        "--datapath",
                        "data/new_hfo2.db",
                        "--output-root",
                        tmpdir,
                        "--date",
                        "2026-03-22",
                    ]
                )
            self.assertEqual(exit_code, 0)
            run_root = Path(tmpdir) / "benchmark" / "run"
            manifests = list(run_root.glob("*/run_manifest.json"))
            self.assertEqual(len(manifests), 1)
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["command"], "benchmark-run")
            self.assertEqual(payload["task_name"], "benchmark.run")

    def test_dataset_profile_run_dry_run_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main(
                [
                    "dataset-profile-run",
                    "--source-type",
                    "ase_db",
                    "--datapath",
                    "data/new_hfo2.db",
                    "--dry-run",
                    "--output-root",
                    tmpdir,
                    "--date",
                    "2026-03-22",
                ]
            )
            self.assertEqual(exit_code, 0)
            run_root = Path(tmpdir) / "dataset_profile" / "build"
            manifests = list(run_root.glob("*/run_manifest.json"))
            self.assertEqual(len(manifests), 1)
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["command"], "dataset-profile-run")
            self.assertEqual(payload["task_name"], "dataset_profile.build")

    def test_dataset_profile_run_dry_run_supports_md17_and_md22(self) -> None:
        for source_type in ("md17", "md22"):
            with self.subTest(source_type=source_type):
                with tempfile.TemporaryDirectory() as tmpdir:
                    exit_code = main(
                        [
                            "dataset-profile-run",
                            "--source-type",
                            source_type,
                            "--molecule",
                            "benzene",
                            "--dry-run",
                            "--output-root",
                            tmpdir,
                            "--date",
                            "2026-03-22",
                        ]
                    )
                    self.assertEqual(exit_code, 0)
                    run_root = Path(tmpdir) / "dataset_profile" / "build"
                    manifests = list(run_root.glob("*/run_manifest.json"))
                    self.assertEqual(len(manifests), 1)
                    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
                    self.assertEqual(payload["inputs"]["source_type"], source_type)

    def test_dataset_profile_run_single_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "atom_dataset_index.cli._build_profile_from_source",
                return_value=self._mock_profile_dataset(),
            ):
                exit_code = main(
                    [
                        "dataset-profile-run",
                        "--source-type",
                        "ase_db",
                        "--datapath",
                        "data/new_hfo2.db",
                        "--metrics",
                        "energy.coverage",
                        "--output-root",
                        tmpdir,
                        "--date",
                        "2026-03-22",
                    ]
                )
            self.assertEqual(exit_code, 0)
            run_root = Path(tmpdir) / "dataset_profile" / "build"
            profile_files = list(run_root.glob("*/profile.json"))
            self.assertEqual(len(profile_files), 1)
            payload = json.loads(profile_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["metric_ids"], ["energy.coverage"])

    def test_dataset_profile_run_multiple_metric_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "atom_dataset_index.cli._build_profile_from_source",
                return_value=self._mock_profile_dataset(),
            ):
                exit_code = main(
                    [
                        "dataset-profile-run",
                        "--source-type",
                        "ase_db",
                        "--datapath",
                        "data/new_hfo2.db",
                        "--metrics",
                        "energy.coverage,distortion.prototype_relative",
                        "--metrics",
                        "diversity.structural",
                        "--output-root",
                        tmpdir,
                        "--date",
                        "2026-03-22",
                    ]
                )
            self.assertEqual(exit_code, 0)
            run_root = Path(tmpdir) / "dataset_profile" / "build"
            profile_files = list(run_root.glob("*/profile.json"))
            self.assertEqual(len(profile_files), 1)
            payload = json.loads(profile_files[0].read_text(encoding="utf-8"))
            self.assertCountEqual(
                payload["metric_ids"],
                ["energy.coverage", "distortion.prototype_relative", "diversity.structural"],
            )

    def test_dataset_profile_run_full_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "atom_dataset_index.cli._build_profile_from_source",
                return_value=self._mock_profile_dataset(),
            ):
                exit_code = main(
                    [
                        "dataset-profile-run",
                        "--source-type",
                        "ase_db",
                        "--datapath",
                        "data/new_hfo2.db",
                        "--preset",
                        "full",
                        "--output-root",
                        tmpdir,
                        "--date",
                        "2026-03-22",
                    ]
                )
            self.assertEqual(exit_code, 0)
            run_root = Path(tmpdir) / "dataset_profile" / "build"
            profile_files = list(run_root.glob("*/profile.json"))
            self.assertEqual(len(profile_files), 1)
            payload = json.loads(profile_files[0].read_text(encoding="utf-8"))
            self.assertEqual(len(payload["metric_ids"]), 6)
            summary_files = list(run_root.glob("*/summary.md"))
            self.assertEqual(len(summary_files), 1)
            self.assertIn("数据集质量画像总结", summary_files[0].read_text(encoding="utf-8"))
            figure_files = list(run_root.glob("*/figures/*.png"))
            self.assertGreaterEqual(len(figure_files), 1)

    def test_dataset_profile_run_rejects_unknown_metric_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                main(
                    [
                        "dataset-profile-run",
                        "--source-type",
                        "ase_db",
                        "--datapath",
                        "data/new_hfo2.db",
                        "--metrics",
                        "unknown.metric",
                        "--dry-run",
                        "--output-root",
                        tmpdir,
                        "--date",
                        "2026-03-22",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
