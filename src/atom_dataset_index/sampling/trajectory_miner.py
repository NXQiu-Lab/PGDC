"""Trajectory event-driven mining for CP2K AIMD outputs.

Detects transition-state events (energy jumps, volume changes, coordination
changes) in MD trajectories and applies intelligent subsampling:
- Dense sampling around events (transition states are information-dense)
- Sparse sampling in equilibrium regions (redundant frames)

module_id: sampling.trajectory_miner
module_version: 1.0.0
project_version: 1.1.0
last_updated: 2026-05-19
generation_mode: manual
source_files: [trajectory_miner.py]
upstream_modules: [domain, sampling.prefilter, sampling.utils]
downstream_modules: [io]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A detected event in a trajectory."""

    frame_idx: int
    event_type: str  # "energy_jump", "volume_change", "cn_change"
    magnitude: float
    description: str = ""


@dataclass
class MiningResult:
    """Result of trajectory mining."""

    sampled_indices: list[int] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
    frame_labels: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TrajectoryMiner
# ---------------------------------------------------------------------------


class TrajectoryMiner:
    """Mine CP2K AIMD trajectories for transition-state events.

    Combines three event detectors:
    1. **Energy jump**: Sudden potential energy changes (bond breaking/forming).
    2. **Volume change**: Cell volume jumps (phase transitions, NPT only).
    3. **Coordination change**: Local coordination number shifts (defect migration).

    Sampling strategy:
    - Event windows: dense sampling (every frame) around detected events.
    - Non-event regions: sparse sampling (every N frames) to avoid redundancy.

    Example:
        >>> miner = TrajectoryMiner(energy_records, trajectory_frames)
        >>> result = miner.mine(
        ...     event_window=5,
        ...     dense_stride=1,
        ...     sparse_stride=20,
        ... )
        >>> print(f"Detected {len(result.events)} events, "
        ...       f"sampled {len(result.sampled_indices)} frames")
    """

    def __init__(
        self,
        energy_records: list[dict],
        trajectory_frames: list[dict],
        cell: "tuple[tuple[float, float, float], ...] | None" = None,
    ) -> None:
        """Initialize miner with parsed CP2K outputs.

        Args:
            energy_records: List of dicts from `parse_cp2k_ener`.
                Each dict must have "step", "potential_energy".
            trajectory_frames: List of frames from `parse_cp2k_trajectory`.
                Each frame has "positions", "symbols", "n_atoms", etc.
            cell: Optional fixed cell matrix (3x3 tuple of tuples).
                Used when trajectory xyz does not carry lattice info.
        """
        self.energy_records = energy_records
        self.trajectory_frames = trajectory_frames
        self.cell = cell
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Check that energy and trajectory frames align."""
        n_ener = len(self.energy_records)
        n_traj = len(self.trajectory_frames)
        if n_ener > 0 and n_traj > 0 and n_ener != n_traj:
            logger.warning(
                f"TrajectoryMiner: energy records ({n_ener}) and "
                f"trajectory frames ({n_traj}) counts differ. "
                f"Using min({n_ener}, {n_traj}) for event detection."
            )

    # -----------------------------------------------------------------------
    # Event detectors
    # -----------------------------------------------------------------------

    def detect_energy_jumps(
        self,
        threshold: float | None = None,
        window_size: int = 3,
    ) -> list[Event]:
        """Detect sudden potential energy changes.

        Uses a rolling median to compute baseline, then flags frames where
        |ΔE| exceeds threshold × median(|ΔE|) in the local window.

        Args:
            threshold: Multiplier over local median energy jump.
                If None, auto-computed as 3.0.
            window_size: Rolling window for local baseline (frames).

        Returns:
            List of Event objects for energy jumps.
        """
        if not self.energy_records:
            return []

        energies = np.array([r["potential_energy"] for r in self.energy_records])
        n = len(energies)
        if n < window_size + 2:
            return []

        # Compute energy differences
        delta_e = np.abs(np.diff(energies))

        # Rolling median baseline
        half = window_size // 2
        events: list[Event] = []

        for i in range(len(delta_e)):
            # Exclude the current point from the local window
            left = max(0, i - half)
            right = min(len(delta_e), i + half + 1)
            local_vals = np.concatenate([delta_e[left:i], delta_e[i + 1:right]])
            if len(local_vals) == 0:
                continue
            local_median = float(np.median(local_vals))
            if local_median < 1e-6:
                local_median = 1e-6

            th = threshold if threshold is not None else 3.0
            ratio = delta_e[i] / local_median

            if ratio > th:
                frame_idx = i + 1  # delta_e[i] = E[i+1] - E[i]
                events.append(Event(
                    frame_idx=frame_idx,
                    event_type="energy_jump",
                    magnitude=float(delta_e[i]),
                    description=f"ΔE={delta_e[i]:.4f} eV, "
                                f"ratio={ratio:.2f}× local median",
                ))

        return events

    def detect_volume_changes(
        self,
        threshold: float | None = None,
    ) -> list[Event]:
        """Detect sudden cell volume changes (NPT trajectories only).

        Args:
            threshold: Fractional volume change threshold.
                If None, uses 0.05 (5%).

        Returns:
            List of Event objects for volume changes.
        """
        volumes = self._extract_volumes()
        if volumes is None or len(volumes) < 2:
            return []

        th = threshold if threshold is not None else 0.05
        events: list[Event] = []

        for i in range(1, len(volumes)):
            v0 = volumes[i - 1]
            v1 = volumes[i]
            if abs(v0) < 1e-6:
                continue
            frac_change = abs(v1 - v0) / abs(v0)
            if frac_change > th:
                events.append(Event(
                    frame_idx=i,
                    event_type="volume_change",
                    magnitude=frac_change,
                    description=f"ΔV/V={frac_change:.4f} "
                                f"({v0:.2f} → {v1:.2f} Å³)",
                ))

        return events

    def detect_coordination_changes(
        self,
        element: str | None = None,
        cutoff: float = 3.0,
        threshold: float | None = None,
    ) -> list[Event]:
        """Detect coordination number changes for selected atoms.

        Computes mean CN for atoms of `element` (or all atoms if None)
        in each frame, then flags frames with large CN jumps.

        Args:
            element: Target element symbol. None = average over all atoms.
            cutoff: Neighbor distance cutoff (Å).
            threshold: Absolute CN change threshold. If None, uses 0.5.

        Returns:
            List of Event objects for coordination changes.
        """
        if not self.trajectory_frames:
            return []

        cn_series = self._compute_coordination_numbers(element, cutoff)
        if cn_series is None or len(cn_series) < 2:
            return []

        th = threshold if threshold is not None else 0.5
        events: list[Event] = []

        for i in range(1, len(cn_series)):
            delta = abs(cn_series[i] - cn_series[i - 1])
            if delta > th:
                events.append(Event(
                    frame_idx=i,
                    event_type="cn_change",
                    magnitude=delta,
                    description=f"ΔCN={delta:.2f} "
                                f"({cn_series[i-1]:.2f} → {cn_series[i]:.2f})",
                ))

        return events

    # -----------------------------------------------------------------------
    # Main mining pipeline
    # -----------------------------------------------------------------------

    def mine(
        self,
        event_window: int = 5,
        dense_stride: int = 1,
        sparse_stride: int = 20,
        energy_threshold: float | None = None,
        energy_window_size: int = 5,
        volume_threshold: float | None = None,
        cn_threshold: float | None = None,
        cn_element: str | None = None,
        cn_cutoff: float = 3.0,
    ) -> MiningResult:
        """Run full mining pipeline.

        Args:
            event_window: Number of frames before/after each event to
                sample densely.
            dense_stride: Frame stride within event windows (1 = every frame).
            sparse_stride: Frame stride outside event windows.
            energy_threshold: Threshold for energy jump detection.
            volume_threshold: Threshold for volume change detection.
            cn_threshold: Threshold for coordination change detection.
            cn_element: Target element for CN analysis.
            cn_cutoff: Neighbor cutoff for CN analysis.

        Returns:
            MiningResult with sampled indices, events, and stats.
        """
        # Detect all event types
        energy_events = self.detect_energy_jumps(
            energy_threshold, window_size=energy_window_size
        )
        volume_events = self.detect_volume_changes(volume_threshold)
        cn_events = self.detect_coordination_changes(
            element=cn_element, cutoff=cn_cutoff, threshold=cn_threshold
        )

        all_events = sorted(
            energy_events + volume_events + cn_events,
            key=lambda e: e.frame_idx,
        )

        # Merge overlapping events
        merged_events = self._merge_events(all_events, merge_distance=event_window)

        # Build event regions
        n_frames = min(
            len(self.energy_records) if self.energy_records else float("inf"),
            len(self.trajectory_frames) if self.trajectory_frames else float("inf"),
        )
        if n_frames == float("inf"):
            n_frames = max(
                len(self.energy_records),
                len(self.trajectory_frames),
            )
        n_frames = int(n_frames)

        event_regions: set[int] = set()
        for ev in merged_events:
            for f in range(
                max(0, ev.frame_idx - event_window),
                min(n_frames, ev.frame_idx + event_window + 1),
            ):
                event_regions.add(f)

        # Sample
        sampled: set[int] = set()
        for f in range(n_frames):
            if f in event_regions:
                if f % dense_stride == 0:
                    sampled.add(f)
            else:
                if f % sparse_stride == 0:
                    sampled.add(f)

        # Ensure we always have at least the first and last frame
        if n_frames > 0:
            sampled.add(0)
            sampled.add(n_frames - 1)

        sampled_sorted = sorted(sampled)

        # Stats
        stats = {
            "n_frames_total": n_frames,
            "n_events": len(merged_events),
            "n_energy_events": len(energy_events),
            "n_volume_events": len(volume_events),
            "n_cn_events": len(cn_events),
            "n_sampled": len(sampled_sorted),
            "compression_ratio": round(n_frames / max(len(sampled_sorted), 1), 2),
        }

        # Label frames
        frame_labels: dict[int, str] = {}
        for f in sampled_sorted:
            if f in event_regions:
                frame_labels[f] = "event_window"
            else:
                frame_labels[f] = "sparse"

        return MiningResult(
            sampled_indices=sampled_sorted,
            events=merged_events,
            stats=stats,
            frame_labels=frame_labels,
        )

    def classify_frames(
        self,
        prefilter: "DeterministicPreFilter",
        frame_indices: list[int] | None = None,
    ) -> dict[int, dict]:
        """Classify selected frames using DeterministicPreFilter.

        Args:
            prefilter: DeterministicPreFilter instance.
            frame_indices: Frames to classify. If None, uses all sampled frames
                from the last `mine()` call (requires `mine()` to have been run).

        Returns:
            Dict mapping frame index → classification result dict.
        """
        from atom_dataset_index.domain import StructureRecord
        from atom_dataset_index.sampling.utils import ase_to_record

        if frame_indices is None:
            # This is a convenience; caller should pass indices explicitly
            raise ValueError(
                "frame_indices must be provided. Use result.sampled_indices "
                "from mine()."
            )

        results: dict[int, dict] = {}
        for idx in frame_indices:
            if idx >= len(self.trajectory_frames):
                continue
            frame = self.trajectory_frames[idx]
            atoms = self._frame_to_atoms(frame, idx)
            record = ase_to_record(atoms, sample_id=idx)
            result = prefilter.classify(record)
            results[idx] = {
                "label": result.label,
                "passed": result.passed,
                "reason": result.reason,
                "details": result.details,
            }

        return results

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract_volumes(self) -> np.ndarray | None:
        """Extract cell volume per frame from trajectory or fixed cell."""
        if not self.trajectory_frames:
            return None

        volumes = []
        for frame in self.trajectory_frames:
            # Try to get cell from frame (extxyz Lattice info)
            cell = self._get_frame_cell(frame)
            if cell is not None:
                vol = self._cell_volume(cell)
                volumes.append(vol)
            elif self.cell is not None:
                vol = self._cell_volume(self.cell)
                volumes.append(vol)
            else:
                # Cannot compute volume
                return None

        return np.array(volumes)

    @staticmethod
    def _get_frame_cell(frame: dict) -> tuple | None:
        """Try to extract cell from trajectory frame metadata."""
        # ASE extxyz may store cell in atoms.info or comment
        comment = frame.get("comment", "")
        # Simple parsing: look for Lattice= in comment
        if "Lattice=" in comment:
            import re
            match = re.search(r'Lattice="([^"]+)"', comment)
            if match:
                vals = [float(x) for x in match.group(1).split()]
                if len(vals) == 9:
                    return tuple(
                        tuple(vals[i * 3 + j] for j in range(3))
                        for i in range(3)
                    )
        # Check if frame has cell attribute (ASE Atoms object)
        if "cell" in frame:
            c = frame["cell"]
            if hasattr(c, "lengths"):
                # ASE Cell object
                arr = c.array
                return tuple(tuple(float(arr[i, j]) for j in range(3)) for i in range(3))
            try:
                arr = np.asarray(c, dtype=float)
                if arr.shape == (3, 3):
                    return tuple(tuple(float(arr[i, j]) for j in range(3)) for i in range(3))
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _cell_volume(cell: tuple[tuple[float, float, float], ...]) -> float:
        """Compute volume of a 3x3 cell matrix."""
        a = np.array(cell[0])
        b = np.array(cell[1])
        c = np.array(cell[2])
        return float(abs(np.dot(a, np.cross(b, c))))

    def _compute_coordination_numbers(
        self,
        element: str | None,
        cutoff: float,
    ) -> np.ndarray | None:
        """Compute mean coordination number per frame."""
        if not self.trajectory_frames:
            return None

        cn_list = []
        for frame in self.trajectory_frames:
            symbols = frame["symbols"]
            positions = frame["positions"]
            n_atoms = frame["n_atoms"]

            # Filter atoms by element if specified
            target_indices = list(range(n_atoms))
            if element is not None:
                target_indices = [i for i, s in enumerate(symbols) if s == element]
                if not target_indices:
                    cn_list.append(0.0)
                    continue

            # Compute CN for target atoms (simple distance cutoff)
            cns = []
            for i in target_indices:
                count = 0
                for j in range(n_atoms):
                    if i == j:
                        continue
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist < cutoff:
                        count += 1
                cns.append(count)
            cn_list.append(sum(cns) / len(cns))

        return np.array(cn_list)

    @staticmethod
    def _merge_events(events: list[Event], merge_distance: int = 5) -> list[Event]:
        """Merge events that are within merge_distance frames of each other."""
        if not events:
            return []

        merged: list[Event] = []
        current = events[0]
        for ev in events[1:]:
            if ev.frame_idx - current.frame_idx <= merge_distance:
                # Merge: keep the one with larger magnitude
                if ev.magnitude > current.magnitude:
                    current = ev
            else:
                merged.append(current)
                current = ev
        merged.append(current)
        return merged

    def _frame_to_atoms(self, frame: dict, frame_idx: int = 0) -> "Atoms":
        """Convert a trajectory frame dict to ASE Atoms."""
        from ase import Atoms

        positions = frame["positions"]
        symbols = frame["symbols"]
        cell = self._get_frame_cell(frame) or self.cell

        if cell is not None:
            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=cell,
                pbc=True,
            )
        else:
            atoms = Atoms(symbols=symbols, positions=positions)

        # Attach forces if available
        if frame.get("forces") is not None:
            atoms.arrays["forces"] = frame["forces"]

        return atoms


# ---------------------------------------------------------------------------
# Factory from file paths
# ---------------------------------------------------------------------------


def mine_cp2k_trajectory(
    pos_xyz_path: str | Path,
    ener_path: str | Path | None = None,
    frc_xyz_path: str | Path | None = None,
    cell: "tuple[tuple[float, float, float], ...] | None" = None,
    **kwargs: Any,
) -> MiningResult:
    """Convenience function: parse CP2K files and mine in one call.

    Args:
        pos_xyz_path: Path to CP2K position xyz file.
        ener_path: Optional path to .ener file.
        frc_xyz_path: Optional path to forces xyz file.
        cell: Fixed cell matrix (if xyz lacks lattice info).
        **kwargs: Passed to TrajectoryMiner.mine().

    Returns:
        MiningResult.
    """
    # Inline CP2K parsers to avoid cross-module path fragility
    def parse_cp2k_ener(ener_path):
        ener_path = Path(ener_path)
        if not ener_path.exists():
            return []
        records = []
        with ener_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    record = {
                        "step": int(parts[0]),
                        "time_fs": float(parts[1]),
                        "kinetic_energy": float(parts[2]),
                        "temperature": float(parts[3]),
                        "potential_energy": float(parts[4]),
                        "total_energy": float(parts[5]),
                        "conserved_quantity": float(parts[6]),
                    }
                    if len(parts) >= 8:
                        record["pressure"] = float(parts[7])
                    records.append(record)
                except (ValueError, IndexError):
                    continue
        return records

    def parse_cp2k_trajectory(pos_xyz_path, frc_xyz_path=None):
        pos_path = Path(pos_xyz_path)
        if not pos_path.exists():
            return []
        from ase.io import read
        pos_frames = read(str(pos_path), index=":")
        if not isinstance(pos_frames, list):
            pos_frames = [pos_frames]
        frc_frames = []
        if frc_xyz_path is not None:
            frc_path = Path(frc_xyz_path)
            if frc_path.exists():
                frc_frames = read(str(frc_path), index=":")
                if not isinstance(frc_frames, list):
                    frc_frames = [frc_frames]
        results = []
        for i, atoms in enumerate(pos_frames):
            frame = {
                "n_atoms": len(atoms),
                "symbols": atoms.get_chemical_symbols(),
                "positions": atoms.get_positions(),
                "forces": None,
                "comment": atoms.info.get("comment", ""),
            }
            if i < len(frc_frames):
                frame["forces"] = frc_frames[i].get_positions()
            results.append(frame)
        return results

    pos_xyz_path = Path(pos_xyz_path)
    if not pos_xyz_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {pos_xyz_path}")

    frames = parse_cp2k_trajectory(pos_xyz_path, frc_xyz_path)

    energy_records = []
    if ener_path is not None:
        ener_path = Path(ener_path)
        if ener_path.exists():
            energy_records = parse_cp2k_ener(ener_path)

    miner = TrajectoryMiner(energy_records, frames, cell=cell)
    return miner.mine(**kwargs)
