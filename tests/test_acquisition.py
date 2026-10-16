"""Tests for acquisition mode functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

from atom_dataset_index.acquisition import (
    AcquisitionEvaluator,
    AcquisitionRanking,
    AcquisitionReport,
    AcquisitionValue,
    DistanceBasedUncertaintyEstimator,
)
from atom_dataset_index.acquisition.evaluator import check_element_compatibility
from atom_dataset_index.domain import DatasetBundle, StructureRecord


class MockDescriptorAdapter:
    """Mock descriptor adapter for testing.
    
    Returns fixed-length descriptors (16 dims) for ANN index compatibility.
    """

    def encode(self, record: StructureRecord) -> tuple[float, ...]:
        """Return a fixed-length descriptor based on atomic numbers."""
        # Create fixed-length descriptor (16 dims) for ANN index compatibility
        desc = [0.0] * 16
        for i, z in enumerate(record.atomic_numbers[:16]):
            desc[i] = float(z)
        # Add a position-based component for structure discrimination
        if record.positions:
            desc[0] += record.positions[0][0] * 0.01
        return tuple(desc)

    def encode_batch(self, records: list[StructureRecord]) -> list[tuple[float, ...]]:
        """Batch encode for Phase 1 optimization compatibility."""
        return [self.encode(record) for record in records]


def create_test_structure(sample_id: int, atomic_numbers: tuple[int, ...]) -> StructureRecord:
    """Create a simple test structure."""
    positions = tuple((0.0, 0.0, float(i)) for i in range(len(atomic_numbers)))
    return StructureRecord(
        sample_id=sample_id,
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
        pbc=(True, True, True),
        energy=-1.0,
        forces=tuple((0.0, 0.0, 0.0) for _ in atomic_numbers),
        metadata={},
    )


class TestAcquisitionValue:
    """Test AcquisitionValue dataclass."""

    def test_creation(self) -> None:
        """Test creating an AcquisitionValue."""
        value = AcquisitionValue(
            total=0.75,
            uncertainty=0.8,
            coverage_gap=0.7,
            trajectory_diversity=0.6,
        )
        assert value.total == 0.75
        assert value.uncertainty == 0.8
        assert value.coverage_gap == 0.7
        assert value.trajectory_diversity == 0.6


class TestAcquisitionRanking:
    """Test AcquisitionRanking dataclass."""

    def test_creation(self) -> None:
        """Test creating an AcquisitionRanking."""
        ranking = AcquisitionRanking(
            poscar_path="test/POSCAR",
            sample_id=0,
            total_value=0.8,
            uncertainty=0.7,
            coverage_gap=0.6,
            trajectory_diversity=0.5,
        )
        assert ranking.poscar_path == "test/POSCAR"
        assert ranking.total_value == 0.8

    def test_recommendation_high(self) -> None:
        """Test HIGH recommendation."""
        ranking = AcquisitionRanking(
            poscar_path="test/POSCAR",
            sample_id=0,
            total_value=0.75,
            uncertainty=0.7,
            coverage_gap=0.6,
            trajectory_diversity=0.5,
        )
        assert ranking.recommendation == "HIGH"

    def test_recommendation_medium(self) -> None:
        """Test MEDIUM recommendation."""
        ranking = AcquisitionRanking(
            poscar_path="test/POSCAR",
            sample_id=0,
            total_value=0.5,
            uncertainty=0.7,
            coverage_gap=0.6,
            trajectory_diversity=0.5,
        )
        assert ranking.recommendation == "MEDIUM"

    def test_recommendation_low(self) -> None:
        """Test LOW recommendation."""
        ranking = AcquisitionRanking(
            poscar_path="test/POSCAR",
            sample_id=0,
            total_value=0.3,
            uncertainty=0.7,
            coverage_gap=0.6,
            trajectory_diversity=0.5,
        )
        assert ranking.recommendation == "LOW"


class TestAcquisitionReport:
    """Test AcquisitionReport."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        rankings = [
            AcquisitionRanking(
                poscar_path="POSCAR_01",
                sample_id=0,
                total_value=0.8,
                uncertainty=0.7,
                coverage_gap=0.6,
                trajectory_diversity=0.5,
            ),
        ]
        report = AcquisitionReport(
            rankings=rankings,
            summary={"avg_total_value": 0.8},
            original_dataset_name="test_db",
            original_sample_count=100,
            candidate_count=5,
        )

        data = report.to_dict()
        assert "run_info" in data
        assert data["run_info"]["original_dataset"] == "test_db"
        assert len(data["rankings"]) == 1
        assert data["rankings"][0]["rank"] == 1

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        rankings = [
            AcquisitionRanking(
                poscar_path="POSCAR_01",
                sample_id=0,
                total_value=0.8,
                uncertainty=0.7,
                coverage_gap=0.6,
                trajectory_diversity=0.5,
            ),
        ]
        report = AcquisitionReport(
            rankings=rankings,
            summary={},
            original_dataset_name="test",
            original_sample_count=10,
            candidate_count=1,
        )

        json_str = report.to_json()
        assert '"POSCAR_01"' in json_str
        assert '"rank": 1' in json_str

    def test_export(self) -> None:
        """Test file export."""
        report = AcquisitionReport(
            rankings=[],
            summary={},
            original_dataset_name="test",
            original_sample_count=10,
            candidate_count=0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            report.export(temp_path)
            content = Path(temp_path).read_text()
            assert '"original_dataset": "test"' in content
        finally:
            Path(temp_path).unlink()


class TestDistanceBasedUncertaintyEstimator:
    """Test DistanceBasedUncertaintyEstimator."""

    def test_empty_dataset(self) -> None:
        """Test with empty dataset."""
        dataset = DatasetBundle(dataset_name="empty", records=())
        estimator = DistanceBasedUncertaintyEstimator(
            dataset, adapter=MockDescriptorAdapter()
        )

        candidate = create_test_structure(0, (1, 1))
        uncertainty = estimator.estimate(candidate)
        assert uncertainty == 1.0  # Maximum uncertainty for empty dataset

    def test_same_structure_zero_uncertainty(self) -> None:
        """Test that identical structures have zero uncertainty."""
        record = create_test_structure(0, (14, 14))  # Silicon
        dataset = DatasetBundle(dataset_name="si", records=(record,))
        estimator = DistanceBasedUncertaintyEstimator(
            dataset, adapter=MockDescriptorAdapter()
        )

        # Same structure should have low uncertainty
        candidate = create_test_structure(1, (14, 14))
        uncertainty = estimator.estimate(candidate)
        assert uncertainty == 0.0  # Same structure -> 0 distance

    def test_different_structure_higher_uncertainty(self) -> None:
        """Test that different structures have higher uncertainty."""
        record = create_test_structure(0, (14, 14))  # Silicon
        dataset = DatasetBundle(dataset_name="si", records=(record,))
        estimator = DistanceBasedUncertaintyEstimator(
            dataset, adapter=MockDescriptorAdapter()
        )

        # Different structure should have higher uncertainty
        candidate = create_test_structure(1, (79, 79))  # Gold
        uncertainty = estimator.estimate(candidate)
        assert uncertainty > 0.0  # Different -> higher distance


class TestAcquisitionEvaluator:
    """Test AcquisitionEvaluator."""

    def test_evaluate_candidates(self) -> None:
        """Test evaluating a list of candidates."""
        # Create a dataset
        dataset_records = [
            create_test_structure(0, (14, 14)),  # Silicon
            create_test_structure(1, (14, 14)),
        ]
        dataset = DatasetBundle(
            dataset_name="test_si", records=tuple(dataset_records)
        )

        # Create evaluator — element_check="none" keeps original cross-element behaviour
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="none",
        )

        # Create candidates
        candidates = [
            create_test_structure(100, (14, 14)),  # Same as dataset
            create_test_structure(101, (79, 79)),  # Different
        ]

        # Evaluate
        report = evaluator.evaluate(candidates, top_k=2)

        assert len(report.rankings) == 2
        assert report.original_dataset_name == "test_si"
        assert report.original_sample_count == 2
        assert report.candidate_count == 2
        assert "avg_total_value" in report.summary

        # Check that rankings are sorted by total_value
        for i in range(len(report.rankings) - 1):
            assert report.rankings[i].total_value >= report.rankings[i + 1].total_value

    def test_evaluate_empty_candidates(self) -> None:
        """Test evaluating empty candidates list."""
        dataset = DatasetBundle(dataset_name="test", records=())
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
        )

        report = evaluator.evaluate([], top_k=10)
        assert len(report.rankings) == 0
        assert report.candidate_count == 0

    def test_top_k_limit(self) -> None:
        """Test that top_k limits the number of rankings."""
        dataset = DatasetBundle(
            dataset_name="test",
            records=(create_test_structure(0, (14, 14)),),
        )
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="none",
        )

        # Create many candidates
        candidates = [create_test_structure(i, (i, i)) for i in range(1, 20)]

        report = evaluator.evaluate(candidates, top_k=5)
        assert len(report.rankings) == 5


class TestCheckElementCompatibility:
    """Unit tests for the check_element_compatibility function."""

    def test_strict_subset_passes(self) -> None:
        """Candidate with subset of dataset elements should pass strict mode."""
        # H2O dataset: {H=1, O=8}; candidate is pure H2 (only H)
        candidate = create_test_structure(0, (1, 1))
        dataset_elements = frozenset({1, 8})
        ok, reason = check_element_compatibility(candidate, dataset_elements, mode="strict")
        assert ok is True
        assert reason == ""

    def test_strict_exact_match_passes(self) -> None:
        """Candidate with exactly the same elements as dataset passes strict mode."""
        candidate = create_test_structure(0, (1, 8))
        dataset_elements = frozenset({1, 8})
        ok, reason = check_element_compatibility(candidate, dataset_elements, mode="strict")
        assert ok is True

    def test_strict_extra_element_rejected(self) -> None:
        """H2O candidate against HfO2 dataset must be rejected in strict mode."""
        # H2O: {H=1, O=8}; HfO2 dataset: {O=8, Hf=72}
        candidate = create_test_structure(0, (1, 1, 8))  # H, H, O
        dataset_elements = frozenset({8, 72})             # O, Hf
        ok, reason = check_element_compatibility(candidate, dataset_elements, mode="strict")
        assert ok is False
        assert "1" in reason  # atomic number of H must appear in reason

    def test_overlap_shared_element_passes(self) -> None:
        """Candidate sharing one element with dataset passes overlap mode."""
        # H2O shares O(8) with HfO2
        candidate = create_test_structure(0, (1, 1, 8))
        dataset_elements = frozenset({8, 72})
        ok, reason = check_element_compatibility(candidate, dataset_elements, mode="overlap")
        assert ok is True

    def test_overlap_no_shared_element_rejected(self) -> None:
        """Candidate with no shared element is rejected in overlap mode."""
        # Pure H2 has no element in common with HfO2
        candidate = create_test_structure(0, (1, 1))
        dataset_elements = frozenset({8, 72})
        ok, reason = check_element_compatibility(candidate, dataset_elements, mode="overlap")
        assert ok is False
        assert reason != ""

    def test_none_mode_always_passes(self) -> None:
        """mode='none' disables the check; every candidate should pass."""
        candidate = create_test_structure(0, (1, 1, 8))  # H2O
        dataset_elements = frozenset({8, 72})             # HfO2
        ok, reason = check_element_compatibility(candidate, dataset_elements, mode="none")
        assert ok is True
        assert reason == ""

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised mode should raise ValueError."""
        import unittest
        candidate = create_test_structure(0, (8, 8))
        dataset_elements = frozenset({8})
        with unittest.TestCase().assertRaises(ValueError):
            check_element_compatibility(candidate, dataset_elements, mode="unknown")


class TestElementCompatibilityInEvaluator:
    """Integration tests for element compatibility inside AcquisitionEvaluator."""

    def _make_hfo2_dataset(self) -> DatasetBundle:
        """Minimal HfO2-like dataset (O=8, Hf=72)."""
        records = (
            create_test_structure(0, (72, 8, 8)),   # HfO2 unit
            create_test_structure(1, (72, 72, 8, 8, 8, 8)),
        )
        return DatasetBundle(dataset_name="hfo2_test", records=records)

    def test_strict_rejects_h2o_candidate(self) -> None:
        """H2O candidate must land in incompatible_candidates under strict mode."""
        dataset = self._make_hfo2_dataset()
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="strict",
        )

        h2o = create_test_structure(100, (1, 1, 8))  # H2O
        report = evaluator.evaluate([h2o], top_k=10)

        assert len(report.rankings) == 0
        assert len(report.incompatible_candidates) == 1
        rejected = report.incompatible_candidates[0]
        assert rejected["sample_id"] == 100
        assert 1 in rejected["elements"]  # H is in the rejected elements list

    def test_strict_accepts_hfo2_candidate(self) -> None:
        """A chemically compatible HfO2 candidate must be scored and ranked."""
        dataset = self._make_hfo2_dataset()
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="strict",
        )

        compatible = create_test_structure(200, (72, 8, 8))  # HfO2
        report = evaluator.evaluate([compatible], top_k=10)

        assert len(report.rankings) == 1
        assert len(report.incompatible_candidates) == 0

    def test_mixed_candidates_split_correctly(self) -> None:
        """Compatible and incompatible candidates are correctly separated."""
        dataset = self._make_hfo2_dataset()
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="strict",
        )

        candidates = [
            create_test_structure(0, (72, 8, 8)),   # HfO2 — compatible
            create_test_structure(1, (1, 1, 8)),     # H2O  — incompatible
            create_test_structure(2, (72, 72, 8, 8, 8, 8)),  # HfO2 — compatible
        ]
        report = evaluator.evaluate(candidates, top_k=10)

        assert len(report.rankings) == 2
        assert len(report.incompatible_candidates) == 1
        assert report.candidate_count == 3

    def test_overlap_mode_accepts_h2o_for_hfo2(self) -> None:
        """In overlap mode, H2O (shares O) should pass for HfO2 dataset."""
        dataset = self._make_hfo2_dataset()
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="overlap",
        )

        h2o = create_test_structure(100, (1, 1, 8))
        report = evaluator.evaluate([h2o], top_k=10)

        assert len(report.rankings) == 1
        assert len(report.incompatible_candidates) == 0

    def test_report_dict_contains_incompatible_section(self) -> None:
        """to_dict() output must include incompatible_candidates key."""
        dataset = self._make_hfo2_dataset()
        evaluator = AcquisitionEvaluator(
            original_dataset=dataset,
            adapter=MockDescriptorAdapter(),
            element_check="strict",
        )
        h2o = create_test_structure(100, (1, 1, 8))
        report = evaluator.evaluate([h2o])

        data = report.to_dict()
        assert "incompatible_candidates" in data
        assert data["run_info"]["incompatible_count"] == 1
        assert data["run_info"]["scored_count"] == 0
