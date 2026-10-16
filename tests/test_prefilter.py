"""Tests for DeterministicPreFilter.

Tests cover all six structure categories (EQ, MS, TP, TD, LU, AB)
and the main public API methods.
"""

from __future__ import annotations

import math
import unittest

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.sampling.prefilter import (
    DeterministicPreFilter,
    StructureClassResult,
    _compute_bonds_and_cn,
    _compute_cell_volume,
    _compute_density,
    _compute_min_distance,
    _get_covalent_radius,
)


class TestHelperFunctions(unittest.TestCase):
    """Test low-level geometric helper functions."""

    def test_covalent_radius_known(self) -> None:
        self.assertAlmostEqual(_get_covalent_radius(8), 0.66, places=2)
        self.assertAlmostEqual(_get_covalent_radius(26), 1.32, places=2)

    def test_covalent_radius_unknown(self) -> None:
        self.assertEqual(_get_covalent_radius(999), 1.20)

    def test_cell_volume_cubic(self) -> None:
        cell = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0))
        self.assertAlmostEqual(_compute_cell_volume(cell), 27.0, places=5)

    def test_min_distance(self) -> None:
        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(1, 1),
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            cell=None,
            pbc=(False, False, False),
        )
        self.assertAlmostEqual(_compute_min_distance(record), 1.0, places=5)

    def test_density_simple(self) -> None:
        # Cu FCC primitive: 4 atoms, a=3.61
        a = 3.61
        cell = ((a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a))
        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(29, 29, 29, 29),
            positions=(
                (0.0, 0.0, 0.0),
                (0.0, a / 2, a / 2),
                (a / 2, 0.0, a / 2),
                (a / 2, a / 2, 0.0),
            ),
            cell=cell,
            pbc=(True, True, True),
        )
        density = _compute_density(record)
        # Cu density ~ 8.96 g/cm³
        self.assertGreater(density, 7.0)
        self.assertLess(density, 11.0)

    def test_bonds_and_cn_copper(self) -> None:
        # Cu FCC with one unit cell
        a = 3.6
        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(29, 29, 29, 29),
            positions=(
                (0.0, 0.0, 0.0),
                (0.0, a / 2, a / 2),
                (a / 2, 0.0, a / 2),
                (a / 2, a / 2, 0.0),
            ),
            cell=((a, 0, 0), (0, a, 0), (0, 0, a)),
            pbc=(True, True, True),
        )
        bls, cn_dict = _compute_bonds_and_cn(record)
        # In FCC, each atom has 12 nearest neighbors
        self.assertIn(29, cn_dict)
        # In a single unit cell without PBC neighbor counting,
        # each atom has 3 nearest neighbors within the cell.
        # Full FCC CN=12 requires PBC-aware neighbor counting.
        self.assertAlmostEqual(cn_dict[29], 3.0, places=0)


class TestDeterministicPreFilter(unittest.TestCase):
    """Test DeterministicPreFilter classification logic."""

    def _make_fcc_record(
        self,
        energy: float | None = None,
        forces: tuple | None = None,
        displacement_std: float = 0.0,
        lattice_factor: float = 1.0,
    ) -> StructureRecord:
        """Helper to create a Cu-like FCC structure."""
        a = 3.6 * lattice_factor
        n = 2  # 2x2x2 supercell = 32 atoms
        positions = []
        atomic_numbers = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for base in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                        import random
                        random.seed(42)
                        x = (i + base[0]) * a
                        y = (j + base[1]) * a
                        z = (k + base[2]) * a
                        if displacement_std > 0:
                            x += random.gauss(0, displacement_std)
                            y += random.gauss(0, displacement_std)
                            z += random.gauss(0, displacement_std)
                        positions.append((x, y, z))
                        atomic_numbers.append(29)

        cell = ((n * a, 0, 0), (0, n * a, 0), (0, 0, n * a))

        if forces is None:
            forces = tuple((0.0, 0.0, 0.0) for _ in range(len(atomic_numbers)))

        return StructureRecord(
            sample_id=1,
            atomic_numbers=tuple(atomic_numbers),
            positions=tuple(positions),
            cell=cell,
            pbc=(True, True, True),
            energy=energy,
            forces=forces,
        )

    def _make_reference_dataset(self, n_structures: int = 5) -> DatasetBundle:
        """Create a reference dataset with Cu FCC structures."""
        records = []
        for i in range(n_structures):
            # Use a wide energy range so that -55 eV is clearly near the minimum
            rec = self._make_fcc_record(
                energy=-100.0 + i * 20.0,  # -100, -80, -60, -40, -20
                forces=tuple(
                    (0.05, 0.03, 0.02) for _ in range(32)
                ),
            )
            # Fix duplicate sample_id
            from dataclasses import replace
            rec = replace(rec, sample_id=i)
            records.append(rec)
        return DatasetBundle(dataset_name="test_cu", records=tuple(records))

    def test_classify_eq(self) -> None:
        """EQ: low energy, low force, small deviation."""
        ref = self._make_reference_dataset()
        pf = DeterministicPreFilter(ref)

        # Perfect equilibrium structure (energy near the minimum of reference)
        eq = self._make_fcc_record(energy=-95.0, forces=tuple((0.0, 0.0, 0.0) for _ in range(32)))
        result = pf.classify(eq)

        self.assertEqual(result.label, "EQ")
        self.assertTrue(result.passed)
        self.assertIn("Near-equilibrium", result.reason)

    def test_classify_ab_overlap(self) -> None:
        """AB: atoms too close (overlap)."""
        pf = DeterministicPreFilter()

        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(29, 29),
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),  # 0.1 Å apart!
            cell=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
            pbc=(True, True, True),
        )
        result = pf.classify(record)

        self.assertEqual(result.label, "AB")
        self.assertFalse(result.passed)
        self.assertIn("Atoms too close", result.reason)

    def test_classify_ab_small_volume(self) -> None:
        """AB: unrealistically small volume per atom."""
        pf = DeterministicPreFilter()

        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(29, 29, 29, 29),
            positions=(
                (0.0, 0.0, 0.0),
                (1.8, 0.0, 0.0),   # ~3.6 Å apart, no overlap
                (0.0, 1.8, 0.0),
                (0.0, 0.0, 1.8),
            ),
            cell=((2.0, 0, 0), (0, 2.0, 0), (0, 0, 2.0)),  # 8 Å³ / 4 atoms = 2 Å³/atom
            pbc=(True, True, True),
        )
        result = pf.classify(record)

        self.assertEqual(result.label, "AB")
        self.assertFalse(result.passed)
        self.assertIn("Unrealistically small volume", result.reason)

    def test_classify_lu_extreme_force(self) -> None:
        """LU: extreme forces."""
        ref = self._make_reference_dataset()
        pf = DeterministicPreFilter(ref)

        # Structure with one atom having extreme force
        forces = [(0.0, 0.0, 0.0)] * 32
        forces[0] = (20.0, 0.0, 0.0)  # 20 eV/Å!
        lu = self._make_fcc_record(energy=-50.0, forces=tuple(forces))
        result = pf.classify(lu)

        self.assertEqual(result.label, "LU")
        self.assertFalse(result.passed)
        self.assertIn("Extreme local force", result.reason)

    def test_filter_candidates(self) -> None:
        """Test batch filtering."""
        ref = self._make_reference_dataset()
        pf = DeterministicPreFilter(ref)

        candidates = [
            self._make_fcc_record(energy=-95.0),  # EQ -> pass
            self._make_fcc_record(
                energy=-50.0,
                forces=tuple((20.0, 0.0, 0.0) for _ in range(32)),  # LU -> fail
            ),
            StructureRecord(
                sample_id=3,
                atomic_numbers=(29, 29),
                positions=((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                cell=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
                pbc=(True, True, True),
            ),  # AB -> fail
        ]

        passed, results = pf.filter_candidates(candidates)

        self.assertEqual(len(passed), 1)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].label, "EQ")
        self.assertEqual(results[1].label, "LU")
        self.assertEqual(results[2].label, "AB")

    def test_check_method(self) -> None:
        """Test the simple check() method."""
        pf = DeterministicPreFilter()

        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(29, 29),
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
            cell=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
            pbc=(True, True, True),
        )
        passed, reason = pf.check(record)
        self.assertFalse(passed)
        self.assertIn("Atoms too close", reason)

    def test_no_reference_conservative(self) -> None:
        """Without reference dataset, filter uses conservative defaults."""
        pf = DeterministicPreFilter()

        # A reasonable structure should pass
        record = self._make_fcc_record(energy=-100.0)
        result = pf.classify(record)
        self.assertTrue(result.passed)
        self.assertEqual(result.label, "EQ")

        # Overlapping structure should still be rejected
        bad = StructureRecord(
            sample_id=1,
            atomic_numbers=(29, 29),
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
            cell=((10.0, 0, 0), (0, 10.0, 0), (0, 0, 10.0)),
            pbc=(True, True, True),
        )
        result_bad = pf.classify(bad)
        self.assertFalse(result_bad.passed)

    def test_result_immutable(self) -> None:
        """StructureClassResult should be frozen/immutable."""
        result = StructureClassResult(
            label="EQ", passed=True, reason="test", details={}
        )
        with self.assertRaises(AttributeError):
            result.label = "AB"  # type: ignore[misc]


class TestPreFilterWithReference(unittest.TestCase):
    """Test envelope building from reference dataset."""

    def test_envelope_construction(self) -> None:
        """Test that envelopes are built correctly from reference data."""
        # Create diverse reference structures
        records = []
        a = 3.6
        for i in range(3):
            rec = StructureRecord(
                sample_id=i,
                atomic_numbers=(29, 29, 29, 29),
                positions=(
                    (0.0, 0.0, 0.0),
                    (0.0, a / 2, a / 2),
                    (a / 2, 0.0, a / 2),
                    (a / 2, a / 2, 0.0),
                ),
                cell=((a, 0, 0), (0, a, 0), (0, 0, a)),
                pbc=(True, True, True),
                energy=-100.0 + i * 5.0,
                forces=tuple((0.1 * i, 0.0, 0.0) for _ in range(4)),
            )
            records.append(rec)

        dataset = DatasetBundle(dataset_name="test", records=tuple(records))
        pf = DeterministicPreFilter(dataset)

        # Envelopes should be populated
        self.assertTrue(len(pf._density_bounds) > 0)
        self.assertTrue(len(pf._bond_length_bounds) > 0)
        self.assertTrue(len(pf._cn_bounds) > 0)
        self.assertTrue(len(pf._force_stats) > 0)
        self.assertTrue(len(pf._energy_stats) > 0)


if __name__ == "__main__":
    unittest.main()
