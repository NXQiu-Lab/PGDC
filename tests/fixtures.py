"""Deterministic synthetic structures, not research data or DFT labels."""

from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.spacegroup import crystal


def write_synthetic_hfo2_db(path: Path, count: int = 64) -> None:
    """Write periodic 96-atom HfO2 rows for reader and profiling tests."""
    reference = crystal(
        ["Hf", "O"],
        basis=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        spacegroup=225,
        cellpar=[5.08, 5.08, 5.08, 90, 90, 90],
    ).repeat((2, 2, 2))
    rng = np.random.default_rng(42)
    with connect(path) as database:
        for index in range(count):
            atoms = reference.copy()
            atoms.positions += rng.normal(0.0, 0.01, atoms.positions.shape)
            atoms.wrap()
            directions = rng.normal(size=(len(atoms), 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            forces = directions * (0.01 + 0.001 * index)
            atoms.calc = SinglePointCalculator(
                atoms, energy=-960.0 + 0.1 * index, forces=forces
            )
            database.write(atoms, data={"synthetic_test_fixture": True})
