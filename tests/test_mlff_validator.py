"""Tests for MLFF validation framework.

module_id: test_mlff_validator
module_version: 1.0.0
project_version: 1.0.5
last_updated: 2026-05-19
generation_mode: manual
source_files: [test_mlff_validator.py]
upstream_modules: [validation.mlff_validator]
downstream_modules: []
"""

from __future__ import annotations

import unittest

import numpy as np

from atom_dataset_index.validation.mlff_validator import (
    ClassEvaluationResult,
    ComparisonResult,
    aggregate_results,
    compare_class_errors,
)


class TestAggregateResults(unittest.TestCase):
    def test_empty(self):
        result = aggregate_results([], "EQ")
        self.assertEqual(result.class_label, "EQ")
        self.assertEqual(result.n_structures, 0)

    def test_basic_aggregation(self):
        per_struct = [
            {"idx": 0, "n_atoms": 4, "energy_error": 1.0, "energy_per_atom_error": 0.25,
             "force_errors": [0.1, 0.2], "force_mae": 0.15, "force_max_error": 0.2},
            {"idx": 1, "n_atoms": 4, "energy_error": 2.0, "energy_per_atom_error": 0.50,
             "force_errors": [0.3, 0.4], "force_mae": 0.35, "force_max_error": 0.4},
        ]
        result = aggregate_results(per_struct, "TP")
        self.assertEqual(result.n_structures, 2)
        self.assertEqual(result.n_atoms_total, 8)
        self.assertAlmostEqual(result.energy_mae_eV, 1.5)
        self.assertAlmostEqual(result.force_mae_eV_per_A, 0.25)
        self.assertAlmostEqual(result.force_max_error_eV_per_A, 0.4)


class TestCompareClassErrors(unittest.TestCase):
    def test_tp_higher_than_eq(self):
        tp = ClassEvaluationResult(
            class_label="TP",
            n_structures=10,
            energy_mae_eV=2.0,
            force_mae_eV_per_A=0.5,
            per_structure_errors=[{"force_mae": 0.5 + i * 0.01} for i in range(10)],
        )
        eq = ClassEvaluationResult(
            class_label="EQ",
            n_structures=10,
            energy_mae_eV=1.0,
            force_mae_eV_per_A=0.1,
            per_structure_errors=[{"force_mae": 0.1 + i * 0.005} for i in range(10)],
        )
        comp = compare_class_errors(tp, eq)
        self.assertAlmostEqual(comp.energy_mae_ratio, 2.0)
        self.assertAlmostEqual(comp.force_mae_ratio, 5.0)
        self.assertIsNotNone(comp.force_pvalue)
        self.assertIn("TP", comp.conclusion)

    def test_equal_errors(self):
        tp = ClassEvaluationResult(
            class_label="TP",
            n_structures=10,
            energy_mae_eV=1.0,
            force_mae_eV_per_A=0.1,
            per_structure_errors=[{"force_mae": 0.1} for _ in range(10)],
        )
        eq = ClassEvaluationResult(
            class_label="EQ",
            n_structures=10,
            energy_mae_eV=1.0,
            force_mae_eV_per_A=0.1,
            per_structure_errors=[{"force_mae": 0.1} for _ in range(10)],
        )
        comp = compare_class_errors(tp, eq)
        self.assertAlmostEqual(comp.energy_mae_ratio, 1.0)
        self.assertAlmostEqual(comp.force_mae_ratio, 1.0)

    def test_insufficient_data(self):
        tp = ClassEvaluationResult(
            class_label="TP",
            n_structures=2,
            per_structure_errors=[{"force_mae": 0.5}, {"force_mae": 0.6}],
        )
        eq = ClassEvaluationResult(
            class_label="EQ",
            n_structures=2,
            per_structure_errors=[{"force_mae": 0.1}, {"force_mae": 0.2}],
        )
        comp = compare_class_errors(tp, eq)
        self.assertIsNone(comp.force_pvalue)
        self.assertIn("Insufficient", comp.conclusion)


class TestComparisonResultDataclass(unittest.TestCase):
    def test_creation(self):
        comp = ComparisonResult(class_a="TP", class_b="EQ", force_mae_ratio=2.0)
        self.assertEqual(comp.class_a, "TP")
        self.assertEqual(comp.class_b, "EQ")
        self.assertEqual(comp.force_mae_ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
