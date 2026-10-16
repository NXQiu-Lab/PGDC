"""Tests for config validation and ASE DB reader robustness."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.config import BenchmarkConfig, MetricConfig, PathConfig, PlotConfig
from atom_dataset_index.io import AseDbDatasetReader, DatasetSchemaError


def _blob_ints(values: list[int]) -> bytes:
    buffer = array("i", values)
    return buffer.tobytes()


def _blob_floats(values: list[float]) -> bytes:
    buffer = array("d", values)
    return buffer.tobytes()


class ConfigValidationTest(unittest.TestCase):
    def test_metric_config_validates_quantiles(self) -> None:
        with self.assertRaises(ValueError):
            MetricConfig(quantiles=(0.9, 0.5))

    def test_benchmark_config_validates_ratios(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(train_ratio=0.8, validation_ratio=0.3)

    def test_plot_config_validates_size(self) -> None:
        with self.assertRaises(ValueError):
            PlotConfig(width=0.0)

    def test_path_config_requires_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                PathConfig(
                    project_root=root,
                    data_dir=Path("data"),
                    output_dir=root / "outputs",
                    docs_dir=root / "docs",
                    tasks_dir=root / "tasks",
                )


class AseDbReaderTest(unittest.TestCase):
    def test_reader_decodes_minimal_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "toy.db"
            self._create_valid_db(db_path)
            dataset = AseDbDatasetReader(db_path).read()
            self.assertEqual(dataset.dataset_name, "toy")
            self.assertEqual(len(dataset.records), 1)
            self.assertEqual(dataset.records[0].sample_id, 1)

    def test_reader_validates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "broken.db"
            with sqlite3.connect(db_path):
                pass
            with self.assertRaises(DatasetSchemaError):
                AseDbDatasetReader(db_path).read()

    def test_reader_rejects_non_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "toy.db"
            self._create_valid_db(db_path)
            with self.assertRaises(ValueError):
                AseDbDatasetReader(db_path).read(limit=0)

    def test_reader_falls_back_to_data_blob_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "toy_data_blob.db"
            self._create_valid_db(db_path, use_data_blob_labels=True)
            dataset = AseDbDatasetReader(db_path).read()
            record = dataset.records[0]
            self.assertAlmostEqual(record.energy or 0.0, -2.5, places=6)
            self.assertEqual(record.forces, ((0.1, 0.2, 0.3),))

    @staticmethod
    def _create_valid_db(db_path: Path, use_data_blob_labels: bool = False) -> None:
        data_payload = None
        if use_data_blob_labels:
            from ase.db.core import object_to_bytes

            data_payload = object_to_bytes({"energy": [-2.5], "forces": [[0.1, 0.2, 0.3]]})
        with sqlite3.connect(db_path) as connection:
            connection.execute("CREATE TABLE information (key TEXT, value TEXT)")
            connection.execute(
                "CREATE TABLE systems ("
                "id INTEGER PRIMARY KEY, "
                "numbers BLOB, "
                "positions BLOB, "
                "cell BLOB, "
                "pbc INTEGER, "
                "energy REAL, "
                "forces BLOB, "
                "key_value_pairs TEXT, "
                "data BLOB)"
            )
            connection.execute(
                "INSERT INTO information (key, value) VALUES (?, ?)",
                ("source", "unit-test"),
            )
            connection.execute(
                "INSERT INTO systems (id, numbers, positions, cell, pbc, energy, forces, key_value_pairs, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    1,
                    _blob_ints([1]),
                    _blob_floats([0.0, 0.0, 0.0]),
                    _blob_floats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    1,
                    None if use_data_blob_labels else -1.0,
                    None if use_data_blob_labels else _blob_floats([0.1, 0.2, 0.3]),
                    '{"tag":"sample"}',
                    data_payload,
                ),
            )
            connection.commit()


if __name__ == "__main__":
    unittest.main()
