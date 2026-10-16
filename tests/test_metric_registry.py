"""Tests for metric registry and profiling bootstrap."""

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.registry import MetricRegistry
from atom_dataset_index.metrics.energy import EnergyCoverageCalculator
from atom_dataset_index.profiling import DatasetProfiler, build_default_registry, list_available_metric_ids


class MetricRegistryTest(unittest.TestCase):
    def test_registry_register_and_list(self) -> None:
        registry = MetricRegistry()
        registry.register(EnergyCoverageCalculator())
        self.assertEqual(registry.list_metric_ids(), ("energy.coverage",))
        self.assertTrue(registry.is_registered("energy.coverage"))
        self.assertIn("energy.coverage", registry)

    def test_registry_register_many(self) -> None:
        registry = MetricRegistry()
        registry.register_many((EnergyCoverageCalculator(),))
        self.assertEqual(len(registry), 1)

    def test_dataset_profiler_builds_profile(self) -> None:
        dataset = DatasetBundle(
            dataset_name="toy",
            records=(
                StructureRecord(sample_id=1, atomic_numbers=(1,), energy=-1.0),
                StructureRecord(sample_id=2, atomic_numbers=(1,), energy=0.0),
            ),
        )
        profile = DatasetProfiler().profile(dataset)
        self.assertEqual(profile.dataset_name, "toy")
        self.assertIn("energy.coverage", profile.metric_results)
        self.assertGreaterEqual(profile.summary["skipped_metric_count"], 1)
        self.assertTrue(profile.provenance["selected_metrics"])
        self.assertIn("prototype.coverage", list_available_metric_ids())

    def test_build_default_registry_with_selected_metric_ids(self) -> None:
        registry = build_default_registry(selected_metric_ids=("energy.coverage",))
        self.assertEqual(registry.list_metric_ids(), ("energy.coverage",))

    def test_build_default_registry_rejects_unknown_metric_id(self) -> None:
        with self.assertRaises(ValueError):
            build_default_registry(selected_metric_ids=("unknown.metric",))

    def test_list_available_metric_ids_contains_core_metric(self) -> None:
        self.assertIn("energy.coverage", list_available_metric_ids())
        self.assertIn("distortion.prototype_relative", list_available_metric_ids())
        self.assertIn("diversity.structural", list_available_metric_ids())
        self.assertIn("diversity.redundancy", list_available_metric_ids())


if __name__ == "__main__":
    unittest.main()
