"""Tests for the descriptor redundancy metric."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import IdentityDescriptorAdapter
from atom_dataset_index.metrics.diversity import DescriptorRedundancyCalculator


class RedundancyMetricTest(unittest.TestCase):
    def _make_record(self, sample_id: int, offset: tuple[float, float, float]) -> StructureRecord:
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

    def test_redundant_cluster_has_high_redundancy_score(self) -> None:
        # 4 records at the same location -> very small k-NN distances -> high redundancy.
        records = tuple(self._make_record(i, (0.0, 0.0, 0.0)) for i in range(4))
        dataset = DatasetBundle(dataset_name="redundant", records=records)
        calculator = DescriptorRedundancyCalculator(k=2, adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=None)  # type: ignore[arg-type]
        self.assertIn("redundancy_score", result.aggregate.values)
        self.assertNotIn("novelty_score", result.aggregate.values)
        self.assertGreater(result.aggregate.values["redundancy_score"], 50.0)

    def test_spread_cluster_has_low_redundancy_score(self) -> None:
        # 4 records far apart -> large k-NN distances -> low redundancy.
        records = tuple(
            self._make_record(i, (float(i) * 10.0, 0.0, 0.0)) for i in range(4)
        )
        dataset = DatasetBundle(dataset_name="spread", records=records)
        calculator = DescriptorRedundancyCalculator(k=2, adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=None)  # type: ignore[arg-type]
        self.assertLess(result.aggregate.values["redundancy_score"], 50.0)

    def test_tiny_dataset_returns_zero_scores(self) -> None:
        records = tuple(self._make_record(i, (0.0, 0.0, 0.0)) for i in range(2))
        dataset = DatasetBundle(dataset_name="tiny", records=records)
        calculator = DescriptorRedundancyCalculator(k=5, adapter=IdentityDescriptorAdapter())
        result = calculator.calculate(dataset, context=None)  # type: ignore[arg-type]
        self.assertEqual(result.aggregate.values["median_knn_distance"], 0.0)
        self.assertGreaterEqual(result.aggregate.values["redundancy_score"], 99.0)


if __name__ == "__main__":
    unittest.main()
