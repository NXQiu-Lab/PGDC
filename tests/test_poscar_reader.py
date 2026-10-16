"""Tests for POSCAR file reading functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

from atom_dataset_index.io import PoscarReader, PoscarReadError, read_poscar


# Sample POSCAR content for testing
SAMPLE_POSCAR = """POSCAR_test
   1.00000000000000
     5.4674000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.4674000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.4674000000000000
   Si
   2
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
"""


class TestPoscarReader:
    """Test cases for PoscarReader."""

    def test_read_single_poscar(self) -> None:
        """Test reading a single POSCAR file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="POSCAR", delete=False) as f:
            f.write(SAMPLE_POSCAR)
            temp_path = f.name

        try:
            reader = PoscarReader()
            record = reader.read(temp_path, sample_id=42)

            assert record.sample_id == 42
            assert len(record.atomic_numbers) == 2
            assert record.atomic_numbers == (14, 14)  # Silicon
            assert len(record.positions) == 2
            assert record.cell is not None
            assert record.pbc == (True, True, True)
            assert record.energy is None  # Not in POSCAR
            assert record.forces == ()  # Not in POSCAR
        finally:
            Path(temp_path).unlink()

    def test_read_directory(self) -> None:
        """Test reading multiple POSCAR files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple POSCAR files
            for i in range(3):
                path = Path(tmpdir) / f"POSCAR_{i:02d}"
                path.write_text(SAMPLE_POSCAR)

            reader = PoscarReader()
            records = reader.read_directory(tmpdir)

            assert len(records) == 3
            # Check that sample_ids are assigned sequentially
            for i, record in enumerate(records):
                assert record.sample_id == i
                assert "poscar_path" in record.metadata
                assert "poscar_filename" in record.metadata

    def test_read_nonexistent_file(self) -> None:
        """Test that FileNotFoundError is raised for nonexistent files."""
        reader = PoscarReader()
        try:
            reader.read("/nonexistent/path/POSCAR")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_read_directory_not_found(self) -> None:
        """Test that FileNotFoundError is raised for nonexistent directory."""
        reader = PoscarReader()
        try:
            reader.read_directory("/nonexistent/directory")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_convenience_function(self) -> None:
        """Test the read_poscar convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="POSCAR", delete=False) as f:
            f.write(SAMPLE_POSCAR)
            temp_path = f.name

        try:
            record = read_poscar(temp_path, sample_id=100)
            assert record.sample_id == 100
            assert len(record.atomic_numbers) == 2
        finally:
            Path(temp_path).unlink()


class TestPoscarReadError:
    """Test error handling for POSCAR reading."""

    def test_invalid_poscar_content(self) -> None:
        """Test that invalid POSCAR content raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="POSCAR", delete=False) as f:
            f.write("invalid content")
            temp_path = f.name

        try:
            reader = PoscarReader()
            try:
                reader.read(temp_path)
                assert False, "Should have raised PoscarReadError"
            except PoscarReadError:
                pass
        finally:
            Path(temp_path).unlink()
