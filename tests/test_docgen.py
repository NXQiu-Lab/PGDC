"""Tests for source-adjacent doc generation."""

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.utils.docgen import _build_leaf_doc, _iter_leaf_dirs


class DocgenTest(unittest.TestCase):
    def test_leaf_dirs_are_discovered(self) -> None:
        leaves = _iter_leaf_dirs()
        self.assertTrue(any(path.name == "statistics" for path in leaves))

    def test_leaf_doc_points_to_readme(self) -> None:
        statistics_dir = Path("src/atom_dataset_index/metric_core/statistics").resolve()
        leaf = _build_leaf_doc(statistics_dir, "0.5.0", "2026-03-22")
        self.assertEqual(leaf.readme_path.name, "README.md")
        self.assertTrue(leaf.module_id.endswith("metric_core.statistics"))


if __name__ == "__main__":
    unittest.main()
