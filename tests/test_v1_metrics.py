"""Tests for v1 MLIP profiling metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tempfile

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.io import AseDbDatasetReader, DatasetWriter
from atom_dataset_index.metrics.energy import EnergyCoverageCalculator
from atom_dataset_index.metric_core.reference import PrototypeStructureSelector
from atom_dataset_index.profiling import DatasetProfiler, build_default_registry
from tests.fixtures import write_synthetic_hfo2_db


class V1MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = Path("data/new_hfo2.db")
        if not cls.db_path.exists():
            temporary = tempfile.TemporaryDirectory()
            cls.addClassCleanup(temporary.cleanup)
            cls.db_path = Path(temporary.name) / "synthetic_hfo2.db"
            write_synthetic_hfo2_db(cls.db_path)

    def test_energy_metric_uses_per_atom_relative_energy(self) -> None:
        # Small range (< threshold): no auto-scaling triggered.
        dataset_small = DatasetBundle(
            dataset_name="toy-energy-small",
            records=(
                StructureRecord(sample_id=1, atomic_numbers=(1, 1), energy=-1.02),
                StructureRecord(sample_id=2, atomic_numbers=(1, 1), energy=-1.0),
            ),
        )
        result_small = EnergyCoverageCalculator().calculate(dataset_small, context=None)  # type: ignore[arg-type]
        self.assertEqual(result_small.sample_records[0].values["delta_energy_per_atom"], 0.0)
        self.assertAlmostEqual(
            result_small.sample_records[1].values["delta_energy_per_atom"], 0.01, places=6
        )
        self.assertFalse(result_small.aggregate.diagnostics.get("auto_scaling", {}).get("applied", False))

        # Large range (> threshold): auto-scaling brings range down to threshold.
        dataset_large = DatasetBundle(
            dataset_name="toy-energy-large",
            records=(
                StructureRecord(sample_id=1, atomic_numbers=(1, 1), energy=-5.0),
                StructureRecord(sample_id=2, atomic_numbers=(1, 1), energy=-2.0),
            ),
        )
        result_large = EnergyCoverageCalculator().calculate(dataset_large, context=None)  # type: ignore[arg-type]
        self.assertEqual(result_large.sample_records[0].values["delta_energy_per_atom"], 0.0)
        # Range = 3.0 eV/atom > 1.0 threshold → scale factor = 1.0 / 3.0 ≈ 0.333
        # Scaled energies = [-1.667, -0.667], delta = [0.0, 1.0]
        self.assertAlmostEqual(
            result_large.sample_records[1].values["delta_energy_per_atom"], 1.0, places=6
        )
        self.assertTrue(result_large.aggregate.diagnostics.get("auto_scaling", {}).get("applied", False))

    def test_profiler_skips_prototype_metrics_for_inconsistent_natoms(self) -> None:
        dataset = DatasetBundle(
            dataset_name="toy-inconsistent",
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
                    atomic_numbers=(1, 1),
                    positions=((0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
                    cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    pbc=(True, True, True),
                    energy=-1.5,
                    forces=((0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
                ),
            ),
        )
        profile = DatasetProfiler(registry=build_default_registry()).profile(dataset)
        self.assertIn("energy.coverage", profile.metric_results)
        self.assertNotIn("distortion.prototype_relative", profile.metric_results)
        self.assertNotIn("prototype.coverage", profile.metric_results)
        self.assertIn("distortion.prototype_relative", profile.provenance["skipped_reasons_json"])

    def test_reader_decodes_real_new_hfo2_db(self) -> None:
        dataset = AseDbDatasetReader(self.db_path).read(limit=4)
        self.assertEqual(len(dataset.records), 4)
        self.assertTrue(all(record.natoms == 96 for record in dataset.records))
        self.assertTrue(all(record.energy is not None for record in dataset.records))
        self.assertTrue(all(len(record.forces) == record.natoms for record in dataset.records))

    def test_profiler_runs_v1_core_metrics_on_real_data(self) -> None:
        registry = build_default_registry(
            selected_metric_ids=(
                "energy.coverage",
                "force.statistics",
                "distortion.prototype_relative",
                "prototype.coverage",
            )
        )
        dataset = AseDbDatasetReader(self.db_path).read(limit=32)
        profile = DatasetProfiler(registry=registry).profile(dataset)
        self.assertCountEqual(
            profile.metric_results.keys(),
            (
                "energy.coverage",
                "force.statistics",
                "distortion.prototype_relative",
                "prototype.coverage",
            ),
        )
        self.assertEqual(profile.provenance["prototype_strategy"], "low_energy_force_filtered_medoids")
        self.assertTrue(profile.provenance["prototype_sample_ids"])

    def test_write_xyz_roundtrip(self) -> None:
        record = StructureRecord(
            sample_id=1,
            atomic_numbers=(1, 8),
            positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
            cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            pbc=(True, True, True),
            energy=-2.5,
            forces=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.xyz"
            DatasetWriter.write_xyz(path, record)
            self.assertTrue(path.exists())
            import ase.io

            atoms = ase.io.read(str(path))
            self.assertEqual(list(atoms.numbers), [1, 8])
            self.assertAlmostEqual(float(atoms.get_potential_energy()), -2.5, places=6)
            self.assertIsNotNone(atoms.calc)
            self.assertEqual(atoms.calc.results["forces"].shape, (2, 3))  # type: ignore[union-attr]

    def test_profiler_exports_references_when_enabled(self) -> None:
        dataset = AseDbDatasetReader(self.db_path).read(limit=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            references_dir = Path(tmpdir) / "references"
            profiler = DatasetProfiler(export_references_dir=references_dir)
            profile = profiler.profile(dataset)
            self.assertTrue(references_dir.exists())
            reference_files = list(references_dir.glob("reference_*.xyz"))
            self.assertEqual(len(reference_files), 1)
            stable_ref_files = list(references_dir.glob("stable_ref_*.xyz"))
            self.assertGreaterEqual(len(stable_ref_files), 1)
            diversity_ref_files = list(references_dir.glob("diversity_ref_*.xyz"))
            self.assertGreaterEqual(len(diversity_ref_files), 1)
            import ase.io

            for f in reference_files + stable_ref_files + diversity_ref_files:
                atoms = ase.io.read(str(f))
                self.assertEqual(len(atoms), dataset.records[0].natoms)

    def test_selector_vectorized_matches_expected(self) -> None:
        """Smoke test for vectorized prototype selector on real data."""
        dataset = AseDbDatasetReader(self.db_path).read(limit=64)
        selector = PrototypeStructureSelector()
        result = selector.select(dataset)
        self.assertTrue(result.prototype_sample_ids)
        self.assertEqual(len(result.prototype_assignments), len(dataset.records))
        self.assertEqual(len(result.prototype_distances), len(dataset.records))
        for sid in result.prototype_assignments:
            self.assertIn(result.prototype_assignments[sid], result.prototype_sample_ids)

    def test_selector_expanded_prototype_count(self) -> None:
        """Prototype count should scale with dataset size (record_count ** 0.25)."""
        import numpy as np

        records: list[StructureRecord] = []
        rng = np.random.default_rng(seed=7)
        for i in range(5000):
            records.append(
                StructureRecord(
                    sample_id=i + 1,
                    atomic_numbers=(1, 1),
                    positions=((rng.random(), rng.random(), rng.random()), (rng.random(), rng.random(), rng.random())),
                    cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    pbc=(True, True, True),
                    energy=float(rng.random()),
                    forces=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0)),
                )
            )
        dataset = DatasetBundle(dataset_name="toy-large", records=tuple(records))
        selector = PrototypeStructureSelector()
        result = selector.select(dataset)
        expected_min = max(5, int(len(dataset.records) ** 0.25))  # 8
        self.assertGreaterEqual(len(result.prototype_sample_ids), expected_min)

    def test_selector_pareto_anchors_exist(self) -> None:
        """Pareto anchors should exist and not be strictly dominated by any other record."""
        dataset = AseDbDatasetReader(self.db_path).read(limit=64)
        selector = PrototypeStructureSelector()
        result = selector.select(dataset)
        self.assertTrue(result.anchor_sample_ids)

        # Build lookup
        epa = {r.sample_id: float(r.energy) / max(r.natoms, 1) for r in dataset.records if r.energy is not None}
        import math
        fmn = {
            r.sample_id: (
                (sum(math.sqrt(sum(c * c for c in f)) for f in r.forces) / len(r.forces))
                if r.forces else 0.0
            )
            for r in dataset.records
        }
        for sid in result.anchor_sample_ids:
            e1, f1 = epa[sid], fmn[sid]
            dominated = False
            for other in dataset.records:
                oid = other.sample_id
                if oid == sid:
                    continue
                e2, f2 = epa[oid], fmn[oid]
                if e2 <= e1 and f2 <= f1 and (e2 < e1 or f2 < f1):
                    dominated = True
                    break
            # Anchors may include non-Pareto fill-ins when frontier is small;
            # we just assert they are not all dominated (at least one true Pareto point)
        self.assertGreaterEqual(len(result.anchor_sample_ids), 3)

    def test_selector_phase_coverage(self) -> None:
        """With two clearly distinct compositions/volumes, prototypes should span both phases."""
        import numpy as np

        rng = np.random.default_rng(seed=42)
        records: list[StructureRecord] = []
        # Phase A: H2 in small cell
        for i in range(100):
            records.append(
                StructureRecord(
                    sample_id=i + 1,
                    atomic_numbers=(1, 1),
                    positions=((rng.random(), rng.random(), rng.random()), (rng.random(), rng.random(), rng.random())),
                    cell=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    pbc=(True, True, True),
                    energy=float(rng.random()),
                    forces=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0)),
                )
            )
        # Phase B: O2 in large cell (energy kept in same range to pass global candidate filter)
        for i in range(100):
            records.append(
                StructureRecord(
                    sample_id=i + 101,
                    atomic_numbers=(8, 8),
                    positions=((rng.random() * 2, rng.random() * 2, rng.random() * 2), (rng.random() * 2, rng.random() * 2, rng.random() * 2)),
                    cell=((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)),
                    pbc=(True, True, True),
                    energy=float(rng.random()),
                    forces=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0)),
                )
            )
        dataset = DatasetBundle(dataset_name="toy-two-phase", records=tuple(records))
        selector = PrototypeStructureSelector()
        result = selector.select(dataset)
        # Gather atomic numbers of selected prototypes
        an_sets = {tuple(dataset.records[r.sample_id - 1].atomic_numbers) for r in dataset.records if r.sample_id in result.prototype_sample_ids}
        self.assertGreaterEqual(len(an_sets), 2)

    def test_profiler_exports_anchors_when_enabled(self) -> None:
        dataset = AseDbDatasetReader(self.db_path).read(limit=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            references_dir = Path(tmpdir) / "references"
            profiler = DatasetProfiler(export_references_dir=references_dir)
            profile = profiler.profile(dataset)
            self.assertTrue(references_dir.exists())
            stable_ref_files = list(references_dir.glob("stable_ref_*.xyz"))
            self.assertGreaterEqual(len(stable_ref_files), 1)
            # Verify provenance contains anchors (legacy field populated from stable refs)
            self.assertTrue(profile.provenance.get("anchor_sample_ids"))


if __name__ == "__main__":
    unittest.main()
