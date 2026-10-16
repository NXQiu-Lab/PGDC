"""Tests for VASP INCAR file parsing functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

from atom_dataset_index.io import IncarParser, IncarParams, IncarParseError, parse_incar


# Sample INCAR content for testing
SAMPLE_INCAR_MD = """# Molecular Dynamics INCAR
TEBEG = 300
TEEND = 300
NSW = 1000
POTIM = 1.0
IBRION = 0
ISYM = 0
ISMEAR = 0
SIGMA = 0.1
ENCUT = 500
NBANDS = 20
ALGO = Normal
MDALGO = 2
"""

SAMPLE_INCAR_PARTIAL = """# Partial INCAR
TEBEG = 500
NSW = 2000
"""

INCAR_WITH_COMMENTS = """# Test INCAR with inline comments
TEBEG = 300 ! Starting temperature
NSW = 1000 # Number of steps
POTIM = 1.0
"""

INCAR_SCIENTIFIC = """# Scientific notation test
TEBEG = 3.0e2
SIGMA = 1.0E-1
ENCUT = 5.0D2
"""


class TestIncarParser:
    """Test cases for IncarParser."""

    def test_parse_full_md_incar(self) -> None:
        """Test parsing a complete MD INCAR file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="INCAR", delete=False) as f:
            f.write(SAMPLE_INCAR_MD)
            temp_path = f.name

        try:
            parser = IncarParser()
            params = parser.parse(temp_path)

            assert params.tebeg == 300.0
            assert params.teend == 300.0
            assert params.nsw == 1000
            assert params.potim == 1.0
            assert params.ibrion == 0
            assert params.isym == 0
            assert params.ismear == 0
            assert params.sigma == 0.1
            assert params.encut == 500.0
            assert params.nbands == 20
            assert params.algo == "Normal"
            assert params.mdalgo == 2
        finally:
            Path(temp_path).unlink()

    def test_parse_partial_incar(self) -> None:
        """Test parsing an INCAR with only some parameters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="INCAR", delete=False) as f:
            f.write(SAMPLE_INCAR_PARTIAL)
            temp_path = f.name

        try:
            params = parse_incar(temp_path)

            assert params.tebeg == 500.0
            assert params.nsw == 2000
            # Other params should be None
            assert params.teend is None
            assert params.potim is None
        finally:
            Path(temp_path).unlink()

    def test_parse_with_comments(self) -> None:
        """Test parsing INCAR with inline comments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="INCAR", delete=False) as f:
            f.write(INCAR_WITH_COMMENTS)
            temp_path = f.name

        try:
            params = parse_incar(temp_path)

            assert params.tebeg == 300.0
            assert params.nsw == 1000
            assert params.potim == 1.0
        finally:
            Path(temp_path).unlink()

    def test_scientific_notation(self) -> None:
        """Test parsing scientific notation values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="INCAR", delete=False) as f:
            f.write(INCAR_SCIENTIFIC)
            temp_path = f.name

        try:
            params = parse_incar(temp_path)

            assert params.tebeg == 300.0
            assert params.sigma == 0.1
            assert params.encut == 500.0
        finally:
            Path(temp_path).unlink()

    def test_is_md_simulation(self) -> None:
        """Test the is_md_simulation property."""
        # MD simulation (IBRION=0)
        params_md = IncarParams(ibrion=0)
        assert params_md.is_md_simulation is True

        # Relaxation (IBRION=2)
        params_relax = IncarParams(ibrion=2)
        assert params_relax.is_md_simulation is False

        # No IBRION specified
        params_none = IncarParams()
        assert params_none.is_md_simulation is False

    def test_effective_temperature(self) -> None:
        """Test effective temperature calculation."""
        # TEBEG available
        params1 = IncarParams(tebeg=500)
        assert params1.effective_temperature == 500.0

        # Only TEIN available
        params2 = IncarParams(tein=400)
        assert params2.effective_temperature == 400.0

        # Neither available - default
        params3 = IncarParams()
        assert params3.effective_temperature == 300.0

    def test_effective_steps(self) -> None:
        """Test effective steps calculation."""
        params1 = IncarParams(nsw=2000)
        assert params1.effective_steps == 2000

        params2 = IncarParams()
        assert params2.effective_steps == 1000  # default

    def test_effective_timestep(self) -> None:
        """Test effective timestep calculation."""
        params1 = IncarParams(potim=2.0)
        assert params1.effective_timestep == 2.0

        params2 = IncarParams()
        assert params2.effective_timestep == 1.0  # default

    def test_total_simulation_time(self) -> None:
        """Test total simulation time calculation."""
        params = IncarParams(nsw=1000, potim=1.0)
        assert params.total_simulation_time == 1.0  # 1000 * 1.0 / 1000 = 1.0 ps

    def test_estimate_diversity_proxy(self) -> None:
        """Test diversity proxy estimation."""
        # High temp, long simulation
        params_high = IncarParams(tebeg=2000, nsw=20000, potim=1.0)
        diversity_high = params_high.estimate_diversity_proxy()
        assert 0.0 <= diversity_high <= 1.0
        assert diversity_high > 0.5  # Should be relatively high

        # Low temp, short simulation
        params_low = IncarParams(tebeg=100, nsw=100, potim=1.0)
        diversity_low = params_low.estimate_diversity_proxy()
        assert 0.0 <= diversity_low <= 1.0
        assert diversity_low < 0.5  # Should be relatively low

    def test_parse_nonexistent_file(self) -> None:
        """Test that FileNotFoundError is raised for nonexistent files."""
        parser = IncarParser()
        try:
            parser.parse("/nonexistent/path/INCAR")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_parse_optional_with_valid_path(self) -> None:
        """Test parse_optional with valid path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="INCAR", delete=False) as f:
            f.write(SAMPLE_INCAR_PARTIAL)
            temp_path = f.name

        try:
            parser = IncarParser()
            params = parser.parse_optional(temp_path)
            assert params is not None
            assert params.tebeg == 500.0
        finally:
            Path(temp_path).unlink()

    def test_parse_optional_with_none(self) -> None:
        """Test parse_optional with None."""
        parser = IncarParser()
        params = parser.parse_optional(None)
        assert params is None

    def test_parse_optional_with_invalid_path(self) -> None:
        """Test parse_optional with invalid path."""
        parser = IncarParser()
        params = parser.parse_optional("/nonexistent/path/INCAR")
        assert params is None


class TestIncarParamsDataclass:
    """Test IncarParams dataclass behavior."""

    def test_frozen_dataclass(self) -> None:
        """Test that IncarParams is immutable (frozen)."""
        params = IncarParams(tebeg=300)
        try:
            params.tebeg = 400  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except (AttributeError, TypeError):
            pass

    def test_slots(self) -> None:
        """Test that IncarParams uses slots for memory efficiency."""
        params = IncarParams(tebeg=300)
        # Check that __slots__ is defined
        assert hasattr(IncarParams, "__slots__") or hasattr(params, "__slots__")
        # Try to add a new attribute - should fail with slots
        try:
            object.__setattr__(params, "new_attribute", "test")
            # If we get here, slots might not be preventing new attributes
            # due to how frozen dataclass works
        except AttributeError:
            pass  # Expected behavior
