"""Tests for dataset-aware candidate explorers.

module_id: test_explorers
module_version: 1.0.0
project_version: 1.0.5
last_updated: 2026-05-19
generation_mode: manual
source_files: [test_explorers.py]
upstream_modules: [sampling.explorers]
downstream_modules: []
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import IdentityDescriptorAdapter
from atom_dataset_index.sampling.explorers import (
    BoundaryExplorer,
    CandidateGenerator,
    StrainPathGenerator,
    _compute_centroid,
    _euclidean_distance,
    _linspace,
)
from atom_dataset_index.sampling.utils import ase_to_record, compute_cell_volume, record_to_ase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    positions: list[tuple[float, float, float]],
    numbers: list[int] | None = None,
    cell: tuple[tuple[float, float, float], ...] | None = None,
    sample_id: int = 0,
    metadata: dict | None = None,
) -> StructureRecord:
    """Create a StructureRecord with sensible defaults."""
    if numbers is None:
        numbers = [29] * len(positions)  # Cu
    if cell is None:
        # Default FCC-like cell for Cu
        cell = ((3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 3.6))
    return StructureRecord(
        sample_id=sample_id,
        atomic_numbers=tuple(numbers),
        positions=tuple(positions),
        cell=cell,
        pbc=(True, True, True),
        energy=None,
        forces=(),
        metadata=metadata or {},
    )


def _make_bundle(records: list[StructureRecord]) -> DatasetBundle:
    """Create a DatasetBundle from records."""
    return DatasetBundle(
        dataset_name="test_bundle",
        records=tuple(records),
        metadata={},
    )


# ---------------------------------------------------------------------------
# _euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance(unittest.TestCase):
    def test_identical_vectors(self):
        a = (1.0, 2.0, 3.0)
        self.assertAlmostEqual(_euclidean_distance(a, a), 0.0)

    def test_simple_3d(self):
        a = (0.0, 0.0, 0.0)
        b = (1.0, 2.0, 2.0)
        self.assertAlmostEqual(_euclidean_distance(a, b), 3.0)

    def test_different_lengths(self):
        a = (1.0, 2.0)
        b = (1.0, 2.0, 3.0)
        self.assertAlmostEqual(_euclidean_distance(a, b), 3.0)

    def test_empty_vectors(self):
        self.assertEqual(_euclidean_distance((), ()), 0.0)


# ---------------------------------------------------------------------------
# _compute_centroid
# ---------------------------------------------------------------------------

class TestComputeCentroid(unittest.TestCase):
    def test_simple(self):
        descriptors = [(1.0, 1.0), (3.0, 3.0)]
        centroid = _compute_centroid(descriptors)
        self.assertAlmostEqual(centroid[0], 2.0)
        self.assertAlmostEqual(centroid[1], 2.0)

    def test_empty(self):
        self.assertEqual(_compute_centroid([]), ())

    def test_single(self):
        self.assertEqual(_compute_centroid([(1.0, 2.0, 3.0)]), (1.0, 2.0, 3.0))


# ---------------------------------------------------------------------------
# _linspace
# ---------------------------------------------------------------------------

class TestLinspace(unittest.TestCase):
    def test_basic(self):
        result = _linspace(0.0, 1.0, 5)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b)

    def test_single(self):
        self.assertEqual(_linspace(0.0, 1.0, 1), [0.0])

    def test_negative(self):
        result = _linspace(-0.1, 0.1, 3)
        expected = [-0.1, 0.0, 0.1]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b)


# ---------------------------------------------------------------------------
# BoundaryExplorer
# ---------------------------------------------------------------------------

class TestBoundaryExplorer(unittest.TestCase):
    def _make_simple_dataset(self, n: int = 4) -> DatasetBundle:
        """Make a dataset where records have distinct descriptor-like positions."""
        records = []
        for i in range(n):
            # Spread positions so they are clearly separated in descriptor space
            offset = float(i) * 2.0
            record = _make_record(
                positions=[
                    (offset + 0.0, 0.0, 0.0),
                    (offset + 1.8, 0.0, 0.0),
                ],
                numbers=[29, 29],
                sample_id=i,
            )
            records.append(record)
        return _make_bundle(records)

    def test_empty_dataset(self):
        """Explorer with empty dataset should return empty list."""
        bundle = _make_bundle([])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(n_samples=5)
        self.assertEqual(result, [])

    def test_explore_returns_correct_count(self):
        """Should return exactly n_samples candidates."""
        bundle = self._make_simple_dataset(n=4)
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(n_samples=10)
        self.assertEqual(len(result), 10)

    def test_explore_copies_metadata(self):
        """Candidates should carry metadata from seed."""
        record = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            sample_id=0,
            metadata={"custom_key": "custom_value"},
        )
        bundle = _make_bundle([record])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(n_samples=1)
        self.assertEqual(len(result), 1)
        self.assertIn("be_seed_id", result[0].metadata)
        self.assertEqual(result[0].metadata.get("custom_key"), "custom_value")

    def test_explore_preserves_atomic_numbers(self):
        """Atomic numbers should be preserved in candidates."""
        record = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            numbers=[26, 29],  # Fe, Cu
            sample_id=0,
        )
        bundle = _make_bundle([record])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(n_samples=1)
        self.assertEqual(result[0].atomic_numbers, (26, 29))

    def test_explore_applies_strain(self):
        """Cell should change when strain is applied."""
        record = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            sample_id=0,
        )
        bundle = _make_bundle([record])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(n_samples=1, strain_range=(0.05, 0.05))

        original_volume = compute_cell_volume(record.cell)
        new_volume = compute_cell_volume(result[0].cell)
        # Volume should change with non-zero strain
        self.assertNotAlmostEqual(original_volume, new_volume, places=3)

    def test_explore_applies_position_noise(self):
        """Positions should differ from original after perturbation."""
        record = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            sample_id=0,
        )
        bundle = _make_bundle([record])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(n_samples=1, perturbation_std=0.1)

        # With noise, positions should not match exactly
        original_pos = np.array(record.positions)
        new_pos = np.array(result[0].positions)
        self.assertFalse(np.allclose(original_pos, new_pos, atol=1e-4))

    def test_boundary_explorer_metadata_content(self):
        """Metadata should contain perturbation details."""
        record = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            sample_id=42,
            metadata={"origin": "test"},
        )
        bundle = _make_bundle([record])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())
        result = explorer.explore(
            n_samples=1,
            perturbation_std=0.05,
            strain_range=(0.02, 0.02),
            temperature_k=600.0,
        )
        self.assertIn("be_seed_id", result[0].metadata)
        self.assertIn("be_perturbation_std", result[0].metadata)
        self.assertIn("be_strain", result[0].metadata)
        self.assertIn("be_temperature_k", result[0].metadata)
        self.assertEqual(result[0].metadata["be_seed_id"], 42)

    def test_temperature_scaling(self):
        """Higher temperature should result in larger perturbation_std."""
        record = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            sample_id=0,
        )
        bundle = _make_bundle([record])
        explorer = BoundaryExplorer(bundle, adapter=IdentityDescriptorAdapter())

        low_t = explorer.explore(n_samples=1, perturbation_std=0.1, temperature_k=300.0)
        high_t = explorer.explore(n_samples=1, perturbation_std=0.1, temperature_k=1200.0)

        low_std = low_t[0].metadata["be_perturbation_std"]
        high_std = high_t[0].metadata["be_perturbation_std"]

        # 1200K / 300K = 4.0, sqrt(4.0) = 2.0
        expected_ratio = math.sqrt(4.0)
        actual_ratio = high_std / low_std
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=4)


# ---------------------------------------------------------------------------
# StrainPathGenerator
# ---------------------------------------------------------------------------

class TestStrainPathGenerator(unittest.TestCase):
    def setUp(self):
        self.reference = _make_record(
            positions=[
                (0.0, 0.0, 0.0),
                (1.8, 1.8, 0.0),
                (1.8, 0.0, 1.8),
                (0.0, 1.8, 1.8),
            ],
            numbers=[29] * 4,
            cell=((3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 3.6)),
            sample_id=0,
        )
        self.generator = StrainPathGenerator()

    def test_generate_uniaxial_count(self):
        """Should produce exactly n_steps candidates."""
        result = self.generator.generate_uniaxial(
            self.reference, strain_range=(-0.1, 0.1), n_steps=5
        )
        self.assertEqual(len(result), 5)

    def test_generate_uniaxial_cell_changes(self):
        """Cell should change along strained axis."""
        result = self.generator.generate_uniaxial(
            self.reference, strain_range=(0.0, 0.1), n_steps=2, axis=0
        )
        # First has 0 strain, second has 0.1 strain
        v0 = compute_cell_volume(result[0].cell)
        v1 = compute_cell_volume(result[1].cell)
        # Uniaxial strain on x-axis: volume scales by (1+ε)
        self.assertAlmostEqual(v1 / v0, 1.1, places=4)

    def test_generate_uniaxial_metadata(self):
        """Metadata should contain strain info."""
        result = self.generator.generate_uniaxial(
            self.reference, strain_range=(0.0, 0.05), n_steps=2, axis=1
        )
        self.assertEqual(result[0].metadata.get("sp_type"), "uniaxial")
        self.assertEqual(result[0].metadata.get("sp_axis"), 1)
        self.assertIn("sp_scale_atoms", result[0].metadata)

    def test_generate_biaxial_count(self):
        result = self.generator.generate_biaxial(
            self.reference, strain_range=(0.0, 0.05), n_steps=3, axes=(0, 1)
        )
        self.assertEqual(len(result), 3)

    def test_generate_biaxial_volume(self):
        """Biaxial strain on x and y should scale volume by (1+ε)^2."""
        result = self.generator.generate_biaxial(
            self.reference, strain_range=(0.0, 0.1), n_steps=2, axes=(0, 1)
        )
        v0 = compute_cell_volume(result[0].cell)
        v1 = compute_cell_volume(result[1].cell)
        self.assertAlmostEqual(v1 / v0, 1.1 ** 2, places=4)

    def test_generate_shear_count(self):
        result = self.generator.generate_shear(
            self.reference, shear_range=(-0.05, 0.05), n_steps=5, plane=(0, 1)
        )
        self.assertEqual(len(result), 5)

    def test_generate_shear_cell_changes(self):
        """Shear should modify off-diagonal cell elements."""
        result = self.generator.generate_shear(
            self.reference, shear_range=(0.0, 0.1), n_steps=2, plane=(0, 1)
        )
        # Original cell should be diagonal
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertAlmostEqual(self.reference.cell[i][j], 0.0)

        # Sheared cell should have non-zero off-diagonal in row 0
        sheared = result[1].cell
        self.assertNotAlmostEqual(sheared[0][1], 0.0)

    def test_generate_hydrostatic_count(self):
        result = self.generator.generate_hydrostatic(
            self.reference, strain_range=(0.0, 0.1), n_steps=4
        )
        self.assertEqual(len(result), 4)

    def test_generate_hydrostatic_volume(self):
        """Hydrostatic strain should scale volume by (1+ε_v)."""
        result = self.generator.generate_hydrostatic(
            self.reference, strain_range=(0.0, 0.1), n_steps=2
        )
        v0 = compute_cell_volume(result[0].cell)
        v1 = compute_cell_volume(result[1].cell)
        self.assertAlmostEqual(v1 / v0, 1.1, places=4)

    def test_strain_preserves_atomic_numbers(self):
        """Atomic numbers should be preserved through strain."""
        ref = _make_record(
            positions=[(0.0, 0.0, 0.0), (1.8, 0.0, 0.0)],
            numbers=[26, 29],
            sample_id=0,
        )
        result = self.generator.generate_uniaxial(ref, n_steps=2)
        self.assertEqual(result[0].atomic_numbers, (26, 29))

    def test_strain_preserves_positions_count(self):
        """Number of atoms should remain constant."""
        result = self.generator.generate_uniaxial(self.reference, n_steps=3)
        for r in result:
            self.assertEqual(len(r.positions), len(self.reference.positions))

    def test_scale_atoms_false(self):
        """With scale_atoms=False, atoms should stay in absolute positions."""
        result = self.generator.generate_uniaxial(
            self.reference, strain_range=(0.0, 0.1), n_steps=2, scale_atoms=False
        )
        # First structure (0 strain) should have same positions
        pos0 = np.array(result[0].positions)
        orig_pos = np.array(self.reference.positions)
        np.testing.assert_allclose(pos0, orig_pos, atol=1e-6)

        # Second structure (10% strain on x) should have same absolute positions
        pos1 = np.array(result[1].positions)
        np.testing.assert_allclose(pos1, orig_pos, atol=1e-6)


# ---------------------------------------------------------------------------
# CandidateGenerator
# ---------------------------------------------------------------------------

class TestCandidateGenerator(unittest.TestCase):
    def _make_dataset(self, n: int = 3) -> DatasetBundle:
        records = []
        for i in range(n):
            record = _make_record(
                positions=[
                    (float(i) * 2.0, 0.0, 0.0),
                    (float(i) * 2.0 + 1.8, 0.0, 0.0),
                ],
                numbers=[29, 29],
                sample_id=i,
            )
            records.append(record)
        return _make_bundle(records)

    def test_init_without_dataset(self):
        """Should work without a dataset for random strategy."""
        gen = CandidateGenerator()
        self.assertIsNotNone(gen)

    def test_init_with_dataset(self):
        """Should create boundary explorer when dataset provided."""
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        self.assertIsNotNone(gen._boundary_explorer)

    def test_boundary_strategy_without_dataset_raises(self):
        """Boundary strategy should require a dataset."""
        gen = CandidateGenerator()
        with self.assertRaises(ValueError) as ctx:
            gen.generate("boundary", n_samples=5)
        self.assertIn("dataset", str(ctx.exception).lower())

    def test_boundary_strategy(self):
        """Boundary strategy should return candidates."""
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        result = gen.generate("boundary", n_samples=6)
        self.assertEqual(len(result), 6)

    def test_composite_strategy(self):
        """Composite strategy should mix boundary and strain."""
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        result = gen.generate("composite", n_samples=10)
        self.assertEqual(len(result), 10)

    def test_strain_uniaxial_strategy(self):
        """Uniaxial strain strategy should use dataset reference."""
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        result = gen.generate("strain_uniaxial", n_samples=4)
        self.assertEqual(len(result), 4)

    def test_strain_biaxial_strategy(self):
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        result = gen.generate("strain_biaxial", n_samples=4)
        self.assertEqual(len(result), 4)

    def test_strain_shear_strategy(self):
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        result = gen.generate("strain_shear", n_samples=4)
        self.assertEqual(len(result), 4)

    def test_strain_hydrostatic_strategy(self):
        dataset = self._make_dataset()
        gen = CandidateGenerator(dataset=dataset, adapter=IdentityDescriptorAdapter())
        result = gen.generate("strain_hydrostatic", n_samples=4)
        self.assertEqual(len(result), 4)

    def test_unknown_strategy_raises(self):
        gen = CandidateGenerator()
        with self.assertRaises(ValueError) as ctx:
            gen.generate("unknown", n_samples=5)
        self.assertIn("unknown", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
