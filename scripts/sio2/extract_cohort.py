#!/usr/bin/env python3
"""
SiO₂ 非稳态结构追踪 cohort 提取脚本（纯只读后处理工具）

从 sio2_al_build/ 读取已完成的数据集和 AIMD 轨迹，提取 5 个追踪结构：
    S1(EQ) — 300K 平衡态
    S2(MS) — Annealing 淬火捕获的亚稳态
    S3(TP) — 轨迹事件帧（过渡态易感）
    S4(TD) — 2500K+ 高温热无序态
    S5(AB) — PreFilter 拒绝的物理荒谬结构

设计原则：
    - 只读 sio2_al_build/，绝不修改其中任何文件
    - 输出到独立的 outputs/ne_cohort/ 目录
    - 数据不完整时给出清晰的诊断信息，而非崩溃
    - 可在数据集制作流程的任何阶段安全运行

用法：
    uv run python scripts/sio2/extract_cohort.py \
        --build-dir sio2_al_build \
        --output-dir outputs/ne_cohort
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir.parent / "src"))
sys.path.insert(0, str(_scripts_dir))

from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.metric_core.descriptors import SOAPDescriptorAdapter
from atom_dataset_index.sampling import (
    DeterministicPreFilter,
    ase_to_record,
    record_to_ase,
)
from atom_dataset_index.sampling.trajectory_miner import mine_cp2k_trajectory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUILD_DIR_DEFAULT = "sio2_al_build"
OUTPUT_DIR_DEFAULT = "outputs/ne_cohort"
DB_NAME_DEFAULT = "dataset.db"

# Temperature windows for candidate screening
TEMP_EQ = 300.0
TEMP_TD = 2500.0
TEMP_TD_TOLERANCE = 200.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PoolStats:
    """Pool-level statistics needed for three-axis normalization."""

    e_min: float
    e_max: float
    e_range: float
    mu_f: float
    sigma_f: float
    d_max: float
    n_records: int


@dataclass
class CohortMember:
    """A single tracking cohort member."""

    sid: str
    label: str
    passed: bool
    dE_norm: float
    F_zscore: float
    D_norm: float
    temperature: float | None
    source: str
    reason: str | None
    filepath: str | None
    frame_idx: int | None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mean_force_norm(record) -> float:
    if not record.forces:
        return 0.0
    norms = [math.sqrt(sum(c * c for c in f)) for f in record.forces]
    return sum(norms) / len(norms)


def _compute_sio2_density(atoms: Atoms) -> float:
    """Compute mass density in g/cm³ for SiO₂."""
    if atoms.get_cell() is None or atoms.get_volume() <= 0:
        return 0.0
    volume_cm3 = atoms.get_volume() * 1e-24  # Å³ → cm³
    mass_g = sum(atoms.get_masses()) * 1.660539e-24  # amu → g
    return mass_g / volume_cm3 if volume_cm3 > 0 else 0.0


def _compute_sio2_bond_stats(atoms: Atoms) -> dict[str, float]:
    """Compute Si-O nearest-neighbor average bond length."""
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    si_o_dists: list[float] = []

    for i in range(len(atoms)):
        if numbers[i] != 14:  # Si
            continue
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if numbers[j] != 8:  # O
                continue
            d_vec = positions[j] + offset @ atoms.get_cell() - positions[i]
            dist = np.linalg.norm(d_vec)
            si_o_dists.append(dist)

    if not si_o_dists:
        return {"si_o_mean": 0.0, "si_o_min": 0.0, "si_o_max": 0.0}

    return {
        "si_o_mean": float(np.mean(si_o_dists)),
        "si_o_min": float(np.min(si_o_dists)),
        "si_o_max": float(np.max(si_o_dists)),
    }


def _print_diagnosis(msg: str) -> None:
    print(f"[DIAGNOSIS] {msg}")


def _parse_cp2k_ener_simple(ener_path: Path) -> list[dict[str, Any]]:
    """Minimal CP2K .ener parser (no external deps)."""
    records = []
    if not ener_path.exists():
        return records
    with ener_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                records.append({
                    "step": int(parts[0]),
                    "time_fs": float(parts[1]),
                    "potential_energy": float(parts[4]),
                })
            except (ValueError, IndexError):
                continue
    return records


# ---------------------------------------------------------------------------
# Pool statistics
# ---------------------------------------------------------------------------

def compute_pool_stats(records: list, adapter: SOAPDescriptorAdapter | None = None) -> PoolStats:
    """Compute pool-level statistics for three-axis normalization."""
    energies = [r.energy / r.natoms for r in records if r.energy is not None]
    if not energies:
        raise ValueError("No records with energy found.")

    e_min = float(min(energies))
    e_max = float(max(energies))
    e_range = e_max - e_min

    force_means = []
    for r in records:
        if r.forces:
            norms = [math.sqrt(sum(c * c for c in f)) for f in r.forces]
            force_means.append(sum(norms) / len(norms))

    mu_f = float(np.mean(force_means)) if force_means else 0.0
    sigma_f = float(np.std(force_means)) if len(force_means) > 1 else 1.0
    if sigma_f == 0:
        sigma_f = 1.0

    # SOAP descriptors for D_norm
    if adapter is None:
        adapter = SOAPDescriptorAdapter()
    descriptors = adapter.encode_batch(list(records))
    desc_matrix = np.array(descriptors)

    from scipy.spatial.distance import cdist

    distances = cdist(desc_matrix, desc_matrix, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    stable_distances = np.min(distances, axis=1)
    d_max = float(np.max(stable_distances)) if len(stable_distances) > 0 else 1.0

    return PoolStats(
        e_min=e_min,
        e_max=e_max,
        e_range=e_range,
        mu_f=mu_f,
        sigma_f=sigma_f,
        d_max=d_max,
        n_records=len(records),
    )


def compute_axes(record, record_idx: int, pool_stats: PoolStats, stable_distances: np.ndarray) -> dict[str, float]:
    """Compute three-axis metrics for a single record."""
    epa = record.energy / record.natoms if record.energy is not None else 0.0
    dE_norm = (epa - pool_stats.e_min) / pool_stats.e_range if pool_stats.e_range > 0 else 0.0

    if record.forces:
        norms = [math.sqrt(sum(c * c for c in f)) for f in record.forces]
        f_mean = sum(norms) / len(norms)
    else:
        f_mean = 0.0
    F_zscore = (f_mean - pool_stats.mu_f) / pool_stats.sigma_f

    D_norm = stable_distances[record_idx] / pool_stats.d_max if pool_stats.d_max > 0 else 0.0

    return {"dE_norm": float(dE_norm), "F_zscore": float(F_zscore), "D_norm": float(D_norm)}


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_s1_eq(records: list, pool_stats: PoolStats, stable_distances: np.ndarray) -> tuple[CohortMember, Atoms] | None:
    """Extract S1: Equilibrium structure from seed phase at ~300K."""
    candidates = []
    for idx, r in enumerate(records):
        temp = r.metadata.get("temperature", 0)
        phase = r.metadata.get("phase", "")
        if phase != "seed":
            continue
        if abs(temp - TEMP_EQ) > 50:
            continue
        f_mean = _mean_force_norm(r)
        candidates.append((f_mean, idx, r))

    if not candidates:
        _print_diagnosis("No seed-phase ~300K structures found for S1 (EQ).")
        return None

    # Pick the one with smallest force norm (most relaxed)
    candidates.sort(key=lambda x: x[0])
    _, idx, r = candidates[0]
    axes = compute_axes(r, idx, pool_stats, stable_distances)
    atoms = record_to_ase(r)

    member = CohortMember(
        sid="S1",
        label="EQ",
        passed=True,
        **axes,
        temperature=r.metadata.get("temperature"),
        source=r.metadata.get("run_name", "seed"),
        reason=None,
        filepath=None,
        frame_idx=None,
    )
    return member, atoms


def extract_s4_td(records: list, pool_stats: PoolStats, stable_distances: np.ndarray) -> tuple[CohortMember, Atoms] | None:
    """Extract S4: Thermally-disordered structure from ~2500K+ trajectories."""
    candidates = []
    for idx, r in enumerate(records):
        temp = r.metadata.get("temperature", 0)
        if abs(temp - TEMP_TD) > TEMP_TD_TOLERANCE and temp < TEMP_TD:
            continue
        f_mean = _mean_force_norm(r)
        candidates.append((f_mean, idx, r))

    if not candidates:
        _print_diagnosis(f"No ~{TEMP_TD}K structures found for S4 (TD).")
        return None

    # Pick one with high force (but not the absolute maximum, to avoid LU edge cases)
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Take the 5th-highest or similar, to get a "typical" TD rather than an outlier
    select_idx = min(4, len(candidates) - 1)
    _, idx, r = candidates[select_idx]
    axes = compute_axes(r, idx, pool_stats, stable_distances)
    atoms = record_to_ase(r)

    member = CohortMember(
        sid="S4",
        label="TD",
        passed=True,
        **axes,
        temperature=r.metadata.get("temperature"),
        source=r.metadata.get("run_name", f"T{int(r.metadata.get('temperature', 0))}"),
        reason=None,
        filepath=None,
        frame_idx=None,
    )
    return member, atoms


def extract_s2_s3_from_al(build_dir: Path, pf: DeterministicPreFilter, pool_stats: PoolStats) -> list[tuple[CohortMember, Atoms]]:
    """
    Scan AL iteration directories for Annealing trajectories.
    Extract S2 (MS) from quench/hold end and S3 (TP) from event frames.
    """
    # Ensure scripts.hea is importable for TrajectoryMiner (uv run isolates sys.path)
    import sys
    _project_root = Path(__file__).resolve().parent.parent.parent
    _scripts_dir = str(_project_root / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    results: list[tuple[CohortMember, Atoms]] = []
    al_loop_dir = build_dir / "phase_1_al_loop"

    if not al_loop_dir.exists():
        _print_diagnosis("No AL iterations found (phase_1_al_loop/ missing). S2 and S3 unavailable.")
        return results

    # Collect all trajectory files across all iterations
    traj_files: list[Path] = []
    for iter_dir in sorted(al_loop_dir.glob("iter_*")):
        selected_dir = iter_dir / "selected"
        if not selected_dir.exists():
            continue
        # Annealing: look for phase subdirectories
        for cfg_dir in selected_dir.iterdir():
            if not cfg_dir.is_dir():
                continue
            for phase_dir in cfg_dir.iterdir():
                if not phase_dir.is_dir():
                    continue
                pos_xyz = list(phase_dir.glob("*-pos-1.xyz"))
                ener_files = list(phase_dir.glob("*.ener"))
                if pos_xyz and ener_files:
                    traj_files.append((pos_xyz[0], ener_files[0], phase_dir.name))

    if not traj_files:
        _print_diagnosis("No Annealing trajectory files found in AL iterations. S2 and S3 unavailable.")
        return results

    # Find best event for S3 (TP)
    best_event = None
    best_traj = None
    best_frames = None

    for pos_xyz, ener_file, phase_name in traj_files:
        if "heating" not in phase_name and "high_temp" not in phase_name:
            continue
        try:
            mine_result = mine_cp2k_trajectory(
                pos_xyz_path=str(pos_xyz),
                ener_path=str(ener_file),
                event_window=5,
                sparse_stride=20,
            )
        except Exception as e:
            _print_diagnosis(f"TrajectoryMiner failed for {pos_xyz.name}: {e}")
            continue

        if mine_result.events:
            ev = max(mine_result.events, key=lambda e: e.magnitude)
            if best_event is None or ev.magnitude > best_event.magnitude:
                best_event = ev
                best_traj = (pos_xyz, ener_file)
                best_frames = read(str(pos_xyz), ":")

    # Extract S3 (TP)
    if best_event and best_frames:
        # Map energy-record frame index to xyz frame index (xyz is sparse output)
        ener_records = _parse_cp2k_ener_simple(best_traj[1]) if best_traj[1] else []
        if ener_records and len(best_frames) > 0:
            ratio = len(ener_records) / len(best_frames)
            frame_idx = min(int(best_event.frame_idx / ratio), len(best_frames) - 1)
        else:
            frame_idx = min(best_event.frame_idx, len(best_frames) - 1)

        atoms = best_frames[frame_idx]
        record = ase_to_record(atoms, sample_id=9999)
        result = pf.classify(record)

        member = CohortMember(
            sid="S3",
            label=result.label,
            passed=result.passed,
            dE_norm=0.0,  # placeholder — refined after full pool stats
            F_zscore=0.0,
            D_norm=0.0,
            temperature=None,
            source=f"AL_event:{best_traj[0].name}",
            reason=result.reason,
            filepath=str(best_traj[0]),
            frame_idx=frame_idx,
        )
        results.append((member, atoms))
        print(f"  S3 (TP): extracted from frame {frame_idx} of {best_traj[0].name}")

    # Extract S2 (MS) from quench/hold end
    s2_found = False
    for pos_xyz, ener_file, phase_name in traj_files:
        if "quench" not in phase_name and "hold" not in phase_name:
            continue
        try:
            frames = read(str(pos_xyz), ":")
        except Exception as e:
            _print_diagnosis(f"Failed to read {pos_xyz.name}: {e}")
            continue

        if len(frames) < 10:
            continue

        # Take last 10% of frames, pick the one with lowest energy (or lowest force)
        end_frames = frames[-max(1, len(frames) // 10):]
        # We'd need energy per frame — for now, just take the last frame
        atoms = end_frames[-1]
        record = ase_to_record(atoms, sample_id=9998)
        result = pf.classify(record)

        if result.label in ("EQ", "MS", "LU"):
            member = CohortMember(
                sid="S2",
                label=result.label,
                passed=result.passed,
                dE_norm=0.0,
                F_zscore=0.0,
                D_norm=0.0,
                temperature=None,
                source=f"AL_quench:{pos_xyz.name}",
                reason=result.reason,
                filepath=str(pos_xyz),
                frame_idx=len(frames) - 1,
            )
            results.append((member, atoms))
            s2_found = True
            print(f"  S2 (MS): extracted from end of {pos_xyz.name}")
            break

    if not s2_found:
        _print_diagnosis("No suitable quench/hold trajectory found for S2 (MS).")

    return results


def extract_s5_ab(build_dir: Path, output_dir: Path, pf: DeterministicPreFilter) -> tuple[CohortMember, Atoms] | None:
    """
    Extract S5: Absurd structure rejected by PreFilter.

    Because prefilter_candidates() deletes rejected files, we look for:
    1. Pre-saved rejected examples in output_dir/rejected_examples/
    2. Any remaining candidate files that still classify as AB
    """
    rejected_dir = output_dir / "rejected_examples"

    # Strategy 1: pre-saved rejected examples
    if rejected_dir.exists():
        for xyz_file in sorted(rejected_dir.glob("*.xyz")):
            try:
                atoms = read(str(xyz_file))
            except Exception:
                continue
            record = ase_to_record(atoms, sample_id=9997)
            result = pf.classify(record)
            if result.label in ("AB", "LU"):
                print(f"  S5 (AB): loaded from pre-saved example {xyz_file.name}")
                member = CohortMember(
                    sid="S5",
                    label="AB",
                    passed=False,
                    dE_norm=0.0,
                    F_zscore=0.0,
                    D_norm=0.0,
                    temperature=None,
                    source="rejected_examples",
                    reason=result.reason,
                    filepath=str(xyz_file),
                    frame_idx=None,
                )
                return member, atoms

    # Strategy 2: scan existing candidate directories (unlikely to find AB after prefilter)
    al_loop_dir = build_dir / "phase_1_al_loop"
    if al_loop_dir.exists():
        for iter_dir in sorted(al_loop_dir.glob("iter_*")):
            cand_dir = iter_dir / "candidates"
            if not cand_dir.exists():
                continue
            for poscar in sorted(cand_dir.glob("POSCAR_*")):
                try:
                    atoms = read(str(poscar), format="vasp")
                except Exception:
                    continue
                record = ase_to_record(atoms, sample_id=0)
                result = pf.classify(record)
                if result.label == "AB":
                    print(f"  S5 (AB): found in {poscar.name}")
                    member = CohortMember(
                        sid="S5",
                        label="AB",
                        passed=False,
                        dE_norm=0.0,
                        F_zscore=0.0,
                        D_norm=0.0,
                        temperature=None,
                        source=f"candidates:{poscar.name}",
                        reason=result.reason,
                        filepath=str(poscar),
                        frame_idx=None,
                    )
                    return member, atoms

    _print_diagnosis(
        "S5 (AB) not found. AB structures are deleted by prefilter_candidates().\n"
        "  To obtain S5, run the following BEFORE prefilter_candidates in your next AL iteration:\n"
        "    python scripts/sio2/save_rejected_candidates.py --build-dir sio2_al_build\n"
        "  Or manually save one rejected POSCAR to outputs/ne_cohort/rejected_examples/"
    )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SiO₂ non-equilibrium tracking cohort.")
    parser.add_argument("--build-dir", default=BUILD_DIR_DEFAULT, help="Path to sio2_al_build/")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output directory")
    parser.add_argument("--db-name", default=DB_NAME_DEFAULT, help="Database filename")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    output_dir = Path(args.output_dir)
    db_path = build_dir / args.db_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SiO₂ Tracking Cohort Extraction")
    print("=" * 60)
    print(f"Build dir : {build_dir}")
    print(f"Output dir: {output_dir}")
    print(f"DB path   : {db_path}")
    print()

    # --- Step 1: Load dataset ------------------------------------------------
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        print("       Please run 'seed-collect' (and AL collect if applicable) first.")
        return 1

    print("Loading dataset...")
    dataset = AseDbDatasetReader(db_path).read()
    records = list(dataset.records)
    print(f"  Loaded {len(records)} structures")

    # --- Step 2: Compute pool statistics -------------------------------------
    print("\nComputing pool statistics...")
    adapter = SOAPDescriptorAdapter()
    pool_stats = compute_pool_stats(records, adapter=adapter)
    print(f"  E_range: {pool_stats.e_range:.4f} eV/atom")
    print(f"  μ_F: {pool_stats.mu_f:.4f}, σ_F: {pool_stats.sigma_f:.4f} eV/Å")
    print(f"  D_max: {pool_stats.d_max:.4f}")

    # Precompute stable distances for D_norm
    descriptors = adapter.encode_batch(records)
    desc_matrix = np.array(descriptors)
    from scipy.spatial.distance import cdist

    distances = cdist(desc_matrix, desc_matrix, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    stable_distances = np.min(distances, axis=1)

    # Save pool stats for plotting scripts
    pool_stats_path = output_dir / "pool_stats.json"
    with open(pool_stats_path, "w") as f:
        json.dump(asdict(pool_stats), f, indent=2)
    print(f"  Saved pool_stats.json")

    # --- Step 3: Initialize PreFilter ----------------------------------------
    print("\nInitializing DeterministicPreFilter...")
    pf = DeterministicPreFilter(dataset)
    print("  PreFilter ready")

    # --- Step 4: Extract cohort members --------------------------------------
    cohort: dict[str, tuple[CohortMember, Atoms]] = {}

    # S1: EQ
    print("\nExtracting S1 (EQ)...")
    s1 = extract_s1_eq(records, pool_stats, stable_distances)
    if s1:
        cohort["S1"] = s1
        print(f"  ✓ S1: {s1[0].label}, dE={s1[0].dE_norm:.3f}, Fz={s1[0].F_zscore:.3f}")

    # S4: TD
    print("\nExtracting S4 (TD)...")
    s4 = extract_s4_td(records, pool_stats, stable_distances)
    if s4:
        cohort["S4"] = s4
        print(f"  ✓ S4: {s4[0].label}, dE={s4[0].dE_norm:.3f}, Fz={s4[0].F_zscore:.3f}")

    # S2 & S3: from AL trajectories
    print("\nScanning AL trajectories for S2 (MS) and S3 (TP)...")
    al_results = extract_s2_s3_from_al(build_dir, pf, pool_stats)
    for member, atoms in al_results:
        cohort[member.sid] = (member, atoms)

    # Patch missing cell info for trajectory-extracted structures (CP2K xyz lacks cell)
    ref_atoms = record_to_ase(records[0])
    ref_cell = ref_atoms.cell
    ref_pbc = ref_atoms.pbc
    for sid in ("S2", "S3"):
        if sid in cohort:
            _, atoms = cohort[sid]
            if atoms.cell is None or atoms.cell.rank == 0:
                atoms.set_cell(ref_cell, scale_atoms=False)
                atoms.set_pbc(ref_pbc)

    # Recompute axes for S2/S3 now that we have full pool stats
    # (They were extracted from raw trajectories, not from the DB)
    for sid in ("S2", "S3"):
        if sid not in cohort:
            continue
        member, atoms = cohort[sid]

        # Energy from ASE info (CP2K xyz comment line)
        energy = atoms.info.get("E")
        natoms = len(atoms)
        if energy is not None and pool_stats.e_range > 0:
            dE_norm = (float(energy) / natoms - pool_stats.e_min) / pool_stats.e_range
        else:
            dE_norm = 0.0

        # Forces from corresponding frc file
        f_mean = 0.0
        if member.filepath and member.frame_idx is not None:
            frc_path = Path(member.filepath).parent / Path(member.filepath).name.replace("-pos-1.xyz", "-frc-1.xyz")
            if frc_path.exists():
                try:
                    frc_frames = read(str(frc_path), index=":")
                    if member.frame_idx < len(frc_frames):
                        forces = frc_frames[member.frame_idx].get_positions()
                        norms = [math.sqrt(sum(c * c for c in f)) for f in forces]
                        f_mean = sum(norms) / len(norms)
                except Exception:
                    pass

        # D_norm via SOAP descriptor
        record = ase_to_record(atoms, sample_id=0)
        desc = adapter.encode_batch([record])
        dists_to_pool = cdist(np.array(desc), desc_matrix, metric="euclidean")
        d_min = float(np.min(dists_to_pool))
        D_norm = d_min / pool_stats.d_max if pool_stats.d_max > 0 else 0.0

        F_zscore = (f_mean - pool_stats.mu_f) / pool_stats.sigma_f if pool_stats.sigma_f > 0 else 0.0

        member.dE_norm = dE_norm
        member.F_zscore = F_zscore
        member.D_norm = D_norm
        print(f"  ✓ {sid}: {member.label}, dE={dE_norm:.3f}, Fz={F_zscore:.3f}, D={D_norm:.3f}")

    # S5: AB
    print("\nExtracting S5 (AB)...")
    s5 = extract_s5_ab(build_dir, output_dir, pf)
    if s5:
        cohort["S5"] = s5
        print(f"  ✓ S5: {s5[0].label}, reason={s5[0].reason}")

    # --- Step 5: Compute bond/density stats for each member ------------------
    print("\nComputing physical properties...")
    physical_stats: dict[str, dict[str, Any]] = {}
    for sid, (member, atoms) in cohort.items():
        stats = _compute_sio2_bond_stats(atoms)
        stats["density"] = _compute_sio2_density(atoms)
        physical_stats[sid] = stats
        print(f"  {sid}: ρ={stats['density']:.2f} g/cm³, Si-O_mean={stats['si_o_mean']:.2f} Å")

    # --- Step 6: Save outputs ------------------------------------------------
    print("\nSaving outputs...")

    # JSON metadata
    cohort_meta = {}
    for sid, (member, _) in cohort.items():
        cohort_meta[sid] = {
            **asdict(member),
            **physical_stats.get(sid, {}),
        }

    meta_path = output_dir / "cohort.json"
    with open(meta_path, "w") as f:
        json.dump(cohort_meta, f, indent=2, default=str)
    print(f"  {meta_path}")

    # XYZ files
    for sid, (_, atoms) in cohort.items():
        out_path = output_dir / f"cohort_{sid}.xyz"
        write(str(out_path), atoms)
        print(f"  {out_path}")

    # --- Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for sid in ("S1", "S2", "S3", "S4", "S5"):
        if sid in cohort:
            m = cohort[sid][0]
            print(f"  {sid} | {m.label:2s} | dE={m.dE_norm:.3f} | Fz={m.F_zscore:.3f} | D={m.D_norm:.3f} | {m.source}")
        else:
            print(f"  {sid} | -- | NOT FOUND")

    missing = [sid for sid in ("S1", "S2", "S3", "S4", "S5") if sid not in cohort]
    if missing:
        print(f"\nWARNING: Missing {missing}. Data may be incomplete.")
        return 2

    print("\nAll cohort members extracted successfully.")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
