"""Tests for trajectory event-driven mining.

module_id: test_trajectory_miner
module_version: 1.0.0
project_version: 1.0.5
last_updated: 2026-05-19
generation_mode: manual
source_files: [test_trajectory_miner.py]
upstream_modules: [sampling.trajectory_miner]
downstream_modules: []
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from atom_dataset_index.sampling.trajectory_miner import (
    Event,
    MiningResult,
    TrajectoryMiner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_energy_records(
    energies: list[float],
    base_step: int = 0,
) -> list[dict]:
    """Create energy records from a list of potential energies."""
    return [
        {
            "step": base_step + i,
            "time_fs": float(i),
            "kinetic_energy": 0.0,
            "temperature": 300.0,
            "potential_energy": float(e),
            "total_energy": float(e),
            "conserved_quantity": float(e),
            "pressure": 0.0,
        }
        for i, e in enumerate(energies)
    ]


def _make_frames(
    n_atoms: int,
    n_frames: int,
    positions_fn=None,
) -> list[dict]:
    """Create dummy trajectory frames."""
    frames = []
    for f in range(n_frames):
        if positions_fn is not None:
            pos = positions_fn(f)
        else:
            # Simple lattice
            pos = np.array([
                [float(i % 3) * 2.0, float((i // 3) % 3) * 2.0, float(i // 9) * 2.0]
                for i in range(n_atoms)
            ])
        frames.append({
            "n_atoms": n_atoms,
            "symbols": ["Cu"] * n_atoms,
            "positions": pos,
            "forces": None,
            "comment": "",
        })
    return frames


# ---------------------------------------------------------------------------
# Event tests
# ---------------------------------------------------------------------------

class TestEventDataclass(unittest.TestCase):
    def test_event_creation(self):
        ev = Event(frame_idx=10, event_type="energy_jump", magnitude=1.5)
        self.assertEqual(ev.frame_idx, 10)
        self.assertEqual(ev.event_type, "energy_jump")
        self.assertEqual(ev.magnitude, 1.5)


# ---------------------------------------------------------------------------
# Energy jump detection
# ---------------------------------------------------------------------------

class TestDetectEnergyJumps(unittest.TestCase):
    def test_no_events_flat(self):
        """Flat energy should produce no events."""
        energies = [0.0] * 20
        records = _make_energy_records(energies)
        frames = _make_frames(n_atoms=4, n_frames=20)
        miner = TrajectoryMiner(records, frames)
        events = miner.detect_energy_jumps()
        self.assertEqual(len(events), 0)

    def test_detects_single_jump(self):
        """A single large energy jump should be detected."""
        # Many zeros on both sides so the jump is isolated from baseline
        energies = [0.0] * 20 + [5.0] + [0.0] * 20
        records = _make_energy_records(energies)
        frames = _make_frames(n_atoms=4, n_frames=41)
        miner = TrajectoryMiner(records, frames)
        events = miner.detect_energy_jumps(threshold=2.0, window_size=9)
        # Both the rise (0→5) and fall (5→0) are detected as separate events
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "energy_jump")
        # Frame 21 corresponds to delta_e[20] = |E[21]-E[20]| = |0-5| = 5

    def test_respects_threshold(self):
        """Small jumps below threshold should be ignored."""
        energies = [0.0] * 10 + [0.1] + [0.0] * 10
        records = _make_energy_records(energies)
        frames = _make_frames(n_atoms=4, n_frames=21)
        miner = TrajectoryMiner(records, frames)
        events = miner.detect_energy_jumps(threshold=10.0)
        self.assertEqual(len(events), 0)

    def test_empty_records(self):
        miner = TrajectoryMiner([], [])
        events = miner.detect_energy_jumps()
        self.assertEqual(events, [])


# ---------------------------------------------------------------------------
# Volume change detection
# ---------------------------------------------------------------------------

class TestDetectVolumeChanges(unittest.TestCase):
    def test_no_volume_events_fixed_cell(self):
        """Fixed cell should produce no volume events."""
        records = _make_energy_records([0.0] * 10)
        frames = _make_frames(n_atoms=4, n_frames=10)
        cell = ((3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 3.6))
        miner = TrajectoryMiner(records, frames, cell=cell)
        events = miner.detect_volume_changes()
        self.assertEqual(len(events), 0)

    def test_detects_volume_jump(self):
        """Frames with different cell volumes should trigger event."""
        records = _make_energy_records([0.0] * 10)
        frames = _make_frames(n_atoms=4, n_frames=10)
        # Vary cell: first 5 frames small, last 5 large
        cell_small = ((3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 3.6))
        cell_large = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
        for i in range(5):
            frames[i]["comment"] = f'Lattice="{cell_small[0][0]} 0 0 0 {cell_small[1][1]} 0 0 0 {cell_small[2][2]}"'
        for i in range(5, 10):
            frames[i]["comment"] = f'Lattice="{cell_large[0][0]} 0 0 0 {cell_large[1][1]} 0 0 0 {cell_large[2][2]}"'

        miner = TrajectoryMiner(records, frames)
        events = miner.detect_volume_changes(threshold=0.1)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "volume_change")


# ---------------------------------------------------------------------------
# Coordination number detection
# ---------------------------------------------------------------------------

class TestDetectCoordinationChanges(unittest.TestCase):
    def test_no_cn_events_static(self):
        """Static positions should produce no CN events."""
        records = _make_energy_records([0.0] * 10)
        frames = _make_frames(n_atoms=8, n_frames=10)
        miner = TrajectoryMiner(records, frames)
        events = miner.detect_coordination_changes(cutoff=3.0)
        self.assertEqual(len(events), 0)

    def test_detects_cn_change(self):
        """Moving atoms closer/farther should trigger CN event."""
        records = _make_energy_records([0.0] * 10)

        def positions_fn(f):
            if f < 5:
                # Distant atoms → low CN
                return np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            else:
                # Close atoms → high CN
                return np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

        frames = _make_frames(n_atoms=2, n_frames=10, positions_fn=positions_fn)
        miner = TrajectoryMiner(records, frames)
        events = miner.detect_coordination_changes(cutoff=3.0, threshold=0.5)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "cn_change")


# ---------------------------------------------------------------------------
# Mine pipeline
# ---------------------------------------------------------------------------

class TestMinePipeline(unittest.TestCase):
    def test_basic_mine(self):
        """Mine should return sampled indices and events."""
        energies = [0.0] * 20 + [5.0] + [0.0] * 20
        records = _make_energy_records(energies)
        frames = _make_frames(n_atoms=4, n_frames=41)
        miner = TrajectoryMiner(records, frames)
        result = miner.mine(
            event_window=2,
            dense_stride=1,
            sparse_stride=5,
            energy_threshold=2.0,
            energy_window_size=9,
        )

        self.assertIsInstance(result, MiningResult)
        self.assertGreater(len(result.events), 0)
        self.assertGreater(len(result.sampled_indices), 0)
        # First and last frame should always be sampled
        self.assertIn(0, result.sampled_indices)
        self.assertIn(20, result.sampled_indices)
        # Event window frames should be in result
        for ev in result.events:
            for f in range(max(0, ev.frame_idx - 2), min(21, ev.frame_idx + 3)):
                self.assertIn(f, result.sampled_indices)

    def test_compression_ratio(self):
        """Compression ratio should be meaningful."""
        energies = list(range(100))
        records = _make_energy_records(energies)
        frames = _make_frames(n_atoms=4, n_frames=100)
        miner = TrajectoryMiner(records, frames)
        result = miner.mine(sparse_stride=10)
        self.assertIn("compression_ratio", result.stats)
        self.assertGreater(result.stats["compression_ratio"], 1.0)

    def test_empty_mine(self):
        """Mining empty data should not crash."""
        miner = TrajectoryMiner([], [])
        result = miner.mine()
        self.assertEqual(result.sampled_indices, [])
        self.assertEqual(result.events, [])


# ---------------------------------------------------------------------------
# Event merging
# ---------------------------------------------------------------------------

class TestEventMerging(unittest.TestCase):
    def test_merge_close_events(self):
        events = [
            Event(frame_idx=10, event_type="energy_jump", magnitude=1.0),
            Event(frame_idx=12, event_type="energy_jump", magnitude=2.0),
            Event(frame_idx=20, event_type="volume_change", magnitude=0.1),
        ]
        merged = TrajectoryMiner._merge_events(events, merge_distance=5)
        self.assertEqual(len(merged), 2)
        # First two merged, keeping the larger magnitude (2.0 at frame 12)
        self.assertEqual(merged[0].frame_idx, 12)
        self.assertEqual(merged[0].magnitude, 2.0)
        self.assertEqual(merged[1].frame_idx, 20)

    def test_no_merge_far_events(self):
        events = [
            Event(frame_idx=10, event_type="energy_jump", magnitude=1.0),
            Event(frame_idx=50, event_type="energy_jump", magnitude=2.0),
        ]
        merged = TrajectoryMiner._merge_events(events, merge_distance=5)
        self.assertEqual(len(merged), 2)


# ---------------------------------------------------------------------------
# Frame to atoms conversion
# ---------------------------------------------------------------------------

class TestFrameToAtoms(unittest.TestCase):
    def test_conversion(self):
        frame = {
            "n_atoms": 2,
            "symbols": ["Fe", "Ni"],
            "positions": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "forces": np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
            "comment": "",
        }
        cell = ((3.6, 0.0, 0.0), (0.0, 3.6, 0.0), (0.0, 0.0, 3.6))
        miner = TrajectoryMiner([], [], cell=cell)
        atoms = miner._frame_to_atoms(frame, 0)
        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms.get_chemical_symbols(), ["Fe", "Ni"])
        np.testing.assert_allclose(atoms.get_cell().array, np.array(cell))
        self.assertTrue("forces" in atoms.arrays)


if __name__ == "__main__":
    unittest.main()
