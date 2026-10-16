"""Tests for scripts/md/ PaiNN ASE calculator and phonon workflow."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms, io

# scripts/md is not a package; add it to the path for importing the calculator.
_MD_DIR = Path(__file__).parent.parent / "scripts" / "md"
if str(_MD_DIR) not in sys.path:
    sys.path.insert(0, str(_MD_DIR))

from painn_calculator import PaiNNCalculator


class TestPaiNNCalculator(unittest.TestCase):
    """Basic tests for the ASE Calculator wrapper."""

    def setUp(self) -> None:
        self.model_path = Path("dataset/best_inference_model")
        if not self.model_path.exists():
            self.skipTest("Trained model not found at dataset/best_inference_model")

    def test_load_and_compute_sio2(self) -> None:
        """Calculator loads and returns consistent energy/forces for SiO2."""
        calc = PaiNNCalculator(self.model_path, device="cpu", cutoff=5.0)

        # Small periodic SiO2 cell.
        atoms = Atoms(
            symbols=["Si", "O", "O"],
            positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.6, 0.0]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        )
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        self.assertIsInstance(energy, float)
        self.assertTrue(np.isfinite(energy))
        self.assertEqual(forces.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(forces)))

    def test_pbc_guard_raises(self) -> None:
        """Non-periodic systems are rejected."""
        calc = PaiNNCalculator(self.model_path, device="cpu", cutoff=5.0)
        atoms = Atoms(
            symbols=["Si", "O"],
            positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=[False, False, False],
        )
        atoms.calc = calc
        with self.assertRaises(ValueError):
            atoms.get_potential_energy()


class TestPhononSiO2Script(unittest.TestCase):
    """End-to-end smoke test for scripts/md/phonon_sio2.py."""

    def setUp(self) -> None:
        self.script = _MD_DIR / "phonon_sio2.py"
        if not self.script.exists():
            self.skipTest("Phonon plotting script is excluded from the public distribution")
        self.model_path = Path("dataset/best_inference_model")
        self.structure_file = Path("scripts/md/alpha_quartz.poscar")
        if not self.model_path.exists():
            self.skipTest("Trained model not found")
        if not self.structure_file.exists():
            self.skipTest("Alpha-quartz POSCAR not found")

    def test_phonon_script_runs(self) -> None:
        """Short run of the phonon script produces the expected outputs."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "phonon_out"
            cmd = [
                "python",
                str(self.script),
                "--structure-file",
                str(self.structure_file),
                "--model-path",
                str(self.model_path),
                "--max-steps",
                "1",
                "--supercell",
                "2 0 0;0 2 0;0 0 2",
                "--output-dir",
                str(output_dir),
            ]
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"phonon_sio2.py failed:\nstdout={result.stdout}\nstderr={result.stderr}",
            )

            expected_files = [
                "relaxed_atoms.xyz",
                "phonopy_params.yaml",
                "FORCE_CONSTANTS",
                "band.yaml",
                "phonon_band.png",
                "phonon_sio2.log",
            ]
            for name in expected_files:
                self.assertTrue(
                    (output_dir / name).exists(),
                    f"Expected output file missing: {name}",
                )

            # The relaxed structure should still be a periodic SiO2 cell.
            relaxed = io.read(output_dir / "relaxed_atoms.xyz")
            self.assertEqual(relaxed.get_chemical_formula(), "O6Si3")
            self.assertTrue(relaxed.pbc.all())


if __name__ == "__main__":
    unittest.main()
