"""Tests for the structural diversity metric."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import IdentityDescriptorAdapter
from atom_dataset_index.metric_core.interfaces import MetricContext
from atom_dataset_index.metrics.diversity import StructuralDiversityCalculator


class StructuralDiversityTest(unittest.TestCase):
    def _make_record(
        self,
        sample_id: int,
        offset: tuple[float, float, float],
    ) -> StructureRecord:
        return StructureRecord(
            sample_id=sample_id,
            atomic_numbers=(1, 1),
            positions=(
                (0.0 + offset[0], 0.0 + offset[1], 0.0 + offset[2]),
                (0.5 + offset[0], 0.5 + offset[1], 0.5 + offset[2]),
            ),
            cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            pbc=(True, True, True),
            energy=float(sample_id),
            forces=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )

    def test_redundant_cluster_has_low_structural_diversity(self) -> None:
        # 12 records with identical descriptors collapse to a single dominant cluster.
        records = tuple(self._make_record(i, (0.0, 0.0, 0.0)) for i in range(12))
        dataset = DatasetBundle(dataset_name="redundant", records=records)
        calculator = StructuralDiversityCalculator(adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=MetricContext())
        self.assertIn("structural_diversity_score", result.aggregate.values)
        self.assertLess(result.aggregate.values["structural_diversity_score"], 30.0)

    def test_spread_cluster_has_high_structural_diversity(self) -> None:
        # 12 records spaced far apart along x -> KMeans splits evenly -> high entropy.
        records = tuple(
            self._make_record(i, (float(i) * 10.0, 0.0, 0.0)) for i in range(12)
        )
        dataset = DatasetBundle(dataset_name="spread", records=records)
        calculator = StructuralDiversityCalculator(adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=MetricContext())
        self.assertGreater(result.aggregate.values["structural_diversity_score"], 70.0)

    def test_tiny_dataset_returns_zero_score(self) -> None:
        records = tuple(self._make_record(i, (0.0, 0.0, 0.0)) for i in range(1))
        dataset = DatasetBundle(dataset_name="tiny", records=records)
        calculator = StructuralDiversityCalculator(adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=MetricContext())
        self.assertEqual(result.aggregate.values["structural_diversity_score"], 0.0)

    def test_sample_records_contain_cluster_id(self) -> None:
        records = tuple(self._make_record(i, (0.0, 0.0, 0.0)) for i in range(4))
        dataset = DatasetBundle(dataset_name="ids", records=records)
        calculator = StructuralDiversityCalculator(adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=MetricContext())
        self.assertEqual(len(result.sample_records), 4)
        for sample in result.sample_records:
            self.assertIn("cluster_id", sample.values)
            self.assertIn("cluster_distance", sample.values)


if __name__ == "__main__":
    unittest.main()
