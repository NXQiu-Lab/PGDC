#!/usr/bin/env python3
"""
SiO2 dataset builder with Active Learning loop — REMOTE SLURM edition.

All non-CP2K work (structure generation, profiling, acquisition, DB merging)
runs locally. CP2K AIMD jobs are uploaded to a remote SLURM cluster, submitted
via SSH, and results are pulled back later via SCP.

The workflow is asynchronous and resumable:

  seed-submit  →  (disconnect)  →  seed-collect
  al-submit    →  (disconnect)  →  al-collect

A local JSON manifest (remote_manifest.json) persists job state.

# ============================================================================
# Complete Workflow Commands
# ============================================================================
#
# 1. Seed phase — generate initial dataset with wide temperature coverage
# ---------------------------------------------------------------------------
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --seed-temps "300,500,700,900,1000,1200,1500,2000,2500,3000" \
    --seed-runs-per-temp 2 \
    seed-submit

# ... wait for jobs to finish (check with squeue) ...

uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    seed-collect

# 2. AL loop — iterative active learning (repeat submit/collect until convergence)
# -------------------------------------------------------------------------------
# Single-phase AL (NVT MD at target temperatures) — default md mode
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode md \
    --al-runs-per-iter 6 \
    al-submit

# NPT pressure-driven AL
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode npt \
    --npt-pressures-gpa "0,10,20,40" \
    --al-runs-per-iter 6 \
    al-submit

# Strain-induced AL
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode strain \
    --strain-values=-0.10,-0.05,0.05,0.10,0.15 \
    --al-runs-per-iter 6 \
    al-submit

# Ballistic-induced AL
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode ballistic \
    --ballistic-velocity-temps "1500,2000,2500" \
    --al-runs-per-iter 6 \
    al-submit

# ... wait for jobs to finish ...

uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    al-collect

# 3. AL loop with 5-phase annealing (heating → high-T → quench → hold)
# --------------------------------------------------------------------
# Backward-compatible annealing alias
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-use-annealing \
    --al-runs-per-iter 6 \
    --event-driven \
    al-submit

# Explicit annealing mode with custom temperature grid
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode annealing \
    --annealing-temps-high "2000,2500,3000,3500" \
    --al-annealing-heating-temp 1500.0 \
    --al-runs-per-iter 6 \
    --event-driven \
    al-submit

# ... wait for jobs to finish ...

uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode annealing \
    al-collect

# 4. Auto mode — blocking loop that submits and collects automatically
# --------------------------------------------------------------------
uv run python scripts/sio2/build_sio2_dataset_remote.py \
    --remote-config scripts/sio2/remote_config.json \
    --al-induction-mode annealing \
    --event-driven \
    auto

# 5. Status check
# ---------------
uv run python scripts/sio2/build_sio2_dataset_remote.py status \
    --build-dir sio2_al_build

# ============================================================================
# Important Notes
# ============================================================================
# - SiO2 is a wide-bandgap insulator (~9 eV). OT is used for all SCF runs.
# - Smearing is NOT used (incompatible with OT, unnecessary for insulators).
# - MD runs use NVT ensemble with CSVR thermostat by default; NPT_I is used
#   when --al-induction-mode npt is selected.
# - The workflow is resumable: if interrupted, rerun the same collect/submit
#   command and it will skip already-completed steps.
# - remote_manifest.json in the build directory tracks all job state.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure project src and scripts dir are on path
_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "sio2"))
sys.path.insert(0, str(_scripts_dir.parent / "src"))

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.sampling import (
    CandidateGenerator,
    DeterministicPreFilter,
    ase_to_record,
    record_to_ase,
)

from common.remote_slurm_utils import (
    RemoteConfig,
    RemoteSLURMClient,
    load_remote_config,
)
from sio2_utils import (
    attach_force_field_calculator,
    build_cp2k_input_sio2,
    generate_sio2_structure,
    get_db_stats,
    load_al_manifest,
    load_json,
    merge_trajectory_to_db,
    parse_cp2k_cell,
    parse_cp2k_ener,
    parse_cp2k_single_point_out,
    save_al_manifest,
    save_json,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BuilderConfig:
    """SiO2 数据集构建器的顶层配置。"""

    # 构建工作目录与数据库文件名
    build_dir: str = "sio2_al_build"
    db_name: str = "dataset.db"

    # SiO2 体系参数
    n_atoms: int = 72
    structure_types: tuple[str, ...] = ("alpha_quartz", "beta_cristobalite")
    structure_weights: tuple[float, ...] = (0.85, 0.15)  # 大部分 α-quartz，少量 β-cristobalite

    # 种子阶段（Phase 0）——覆盖宽温区的 NVT AIMD 基础数据集
    seed_temperatures: tuple[float, ...] = (
        300, 500, 700, 900, 1000, 1200, 1500, 2000, 2500, 3000,
    )
    seed_runs_per_temp: int = 2     # 每个温度重复运行次数
    seed_n_steps: int = 5000        # 每次 AIMD 步数（1 fs 步长 = 5 ps）
    seed_timestep: float = 1.0      # 积分步长 (fs)
    seed_output_stride: int = 20    # 每隔多少步输出一帧轨迹

    # 主动学习阶段（Phase 1）——迭代筛选高信息量候选结构
    al_max_iterations: int = 20      # 最大 AL 轮次
    al_runs_per_iteration: int = 6   # 每轮提交的 CP2K 作业数
    al_n_steps: int = 10000          # 每次 AL AIMD 步数（单段）
    al_output_stride: int = 20       # AL 轨迹输出步长
    al_top_k: int = 10               # 每轮从采集评分中选取的最优候选数
    al_candidate_pool_size: int = 50 # 候选结构池大小

    # 候选结构预筛选：默认使用 ASE 内置力场做快速能量/力估计，避免为每个候选
    # 结构提交 CP2K 单点计算。关闭后将恢复原始的 CP2K ENERGY_FORCE 单点预筛选。
    use_force_field: bool = True
    force_field_type: str = "morse"   # auto | emt | lj | tersoff | morse
    force_field_param_file: str = ""  # LAMMPS-style Tersoff potential file

    # 5-phase annealing 循环
    al_use_annealing: bool = False           # True: 启用 5-phase 升温-淬火
    al_annealing_relax_steps: int = 2000     # 阶段 1: 300 K 弛豫步数
    al_annealing_heating_steps: int = 3000   # 阶段 2: 升温步数
    al_annealing_heating_temp: float = 3000.0 # 阶段 2: 升温目标温度 (K)
    al_annealing_peak_temp: float = 3500.0   # 阶段 3: 高温保持温度 (K)
    al_annealing_high_temp_steps: int = 2000 # 阶段 3: 高温保持步数
    al_annealing_quench_steps: int = 3000    # 阶段 4: 淬火步数
    al_annealing_hold_steps: int = 5000      # 阶段 5: 300 K 保持步数

    # Profile-guided induction mode for the AL loop
    al_induction_mode: str = "md"            # md | annealing | npt | strain | ballistic
    ballistic_velocity_temps: tuple[float, ...] = (1500.0, 2000.0, 2500.0)
    npt_pressures_gpa: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0)
    npt_temperatures: tuple[float, ...] = (300.0, 1000.0, 2000.0)
    annealing_temps_high: tuple[float, ...] = (2000.0, 2500.0, 3000.0, 3500.0)
    strain_axes: tuple[str, ...] = ("x", "y", "z")
    strain_values: tuple[float, ...] = (-0.10, -0.05, 0.05, 0.10, 0.15)

    def __post_init__(self) -> None:
        if self.al_use_annealing:
            self.al_induction_mode = "annealing"

    # 终止阈值
    target_size: int = 20000
    min_coverage: float = 0.85
    min_uniformity: float = 0.80
    min_diversity: float = 0.75
    min_stability: float = 0.65

    # CP2K 计算参数 / 远程集群资源
    cp2k_nodes: int = 1
    cp2k_ppn: int = 64
    cp2k_nprocs: int = 64
    slurm_partition: str = "compute"
    cp2k_cutoff: int = 500
    cp2k_basis_set: str = "DZVP-MOLOPT-SR-GTH"
    cp2k_data_dir: str | None = None

    # 远程 SSH/SCP 配置
    remote_config_path: str = ""

    # 候选结构生成策略
    candidate_strategy: str = "random"
    candidate_strategy_params: dict[str, Any] = field(default_factory=dict)

    # 轨迹采样策略
    event_driven_sampling: bool = False
    event_window: int = 5
    sparse_stride: int = 20

    # 调试 / 跳过开关
    dry_run: bool = False
    skip_seed: bool = False
    skip_al: bool = False
    verbose: bool = False


# ---------------------------------------------------------------------------
# Remote manifest helpers
# ---------------------------------------------------------------------------

REMOTE_MANIFEST_NAME = "remote_manifest.json"


def load_remote_manifest(build_dir: str | Path) -> dict[str, Any]:
    path = Path(build_dir) / REMOTE_MANIFEST_NAME
    if path.exists():
        return load_json(path)
    return {
        "build_dir": str(build_dir),
        "remote_work_dir": "",
        "seed": {"submitted": False, "jobs": []},
        "al_iterations": [],
        "termination": {"stopped": False, "reason": None},
    }


def save_remote_manifest(build_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(build_dir) / REMOTE_MANIFEST_NAME
    save_json(path, manifest)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _run_cli(cmd: list[str], cwd: str | Path | None = None, check: bool = False) -> subprocess.CompletedProcess:
    full_cmd = ["uv", "run", "python", "-m", "atom_dataset_index"] + cmd
    return subprocess.run(full_cmd, capture_output=True, text=True, cwd=cwd, check=check)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cleanup_iteration_intermediates(build_dir: Path, iteration: int) -> None:
    iter_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}"
    if not iter_dir.exists():
        return
    cand_dir = iter_dir / "candidates"
    if cand_dir.exists():
        removed = 0
        for xyz_file in cand_dir.glob("*.xyz"):
            xyz_file.unlink()
            removed += 1
        if removed:
            print(f"  Cleaned {removed} candidate XYZs")
        try:
            cand_dir.rmdir()
        except OSError:
            pass
    selected_dir = iter_dir / "selected"
    if selected_dir.exists():
        removed_files = 0
        for run_dir in selected_dir.iterdir():
            if run_dir.is_dir():
                for fname in ("input.inp", "init.xyz"):
                    fp = run_dir / fname
                    if fp.exists():
                        fp.unlink()
                        removed_files += 1
        if removed_files:
            print(f"  Cleaned {removed_files} intermediate input files")


def _write_submit_script(
    run_dir: Path,
    job_name: str,
    nodes: int,
    ppn: int,
    nprocs: int,
    partition: str,
    with_geo_opt: bool = True,
    geo_restart_file: str = "",
) -> None:
    """Write a local submit.slurm for GEO_OPT → MD (or MD-only) workflow."""
    # nprocs is ignored under SLURM; srun uses the ntasks-per-node allocation.

    if with_geo_opt:
        check_block = (
            f'if [ ! -f "./{geo_restart_file}" ]; then\n'
            f'    echo "ERROR: GEO_OPT did not produce {geo_restart_file}; aborting."\n'
            f'    exit 1\n'
            f'fi\n'
        ) if geo_restart_file else ""
        run_block = (
            'echo "=== Step 1: GEO_OPT ==="\n'
            "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./geo_opt.inp\n"
            + check_block
            + "\n"
            'echo "=== Step 2: MD ==="\n'
            "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./input.inp\n"
        )
    else:
        run_block = "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./input.inp\n"

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --nodes={nodes}\n"
        f"#SBATCH --ntasks-per-node={ppn}\n"
        f"#SBATCH --partition={partition}\n"
        "#SBATCH --output=out\n"
        "#SBATCH --error=err\n"
        "\n"
        "# Handle path difference between head (/home/work) and compute nodes (/work)\n"
        'SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"\n'
        'HEAD_DIR="$SUBMIT_DIR"\n'
        'NODE_DIR="${SUBMIT_DIR/\\/home\\/work/\\/work}"\n'
        'if [ -d "$HEAD_DIR" ]; then\n'
        '    cd "$HEAD_DIR"\n'
        'elif [ -d "$NODE_DIR" ]; then\n'
        '    cd "$NODE_DIR"\n'
        "else\n"
        '    echo "ERROR: Cannot find work directory"\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        "source /opt/software/cp2k/tools/toolchain/install/setup\n"
        "export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH\n"
        "export PMIX_MCA_psec=native\n"
        "\n"
        + run_block
    )
    (run_dir / "submit.slurm").write_text(script)


def _extract_profile_scores(profile_dir: Path) -> dict[str, float]:
    summary_path = profile_dir / "summary.json"
    scores = {}
    if summary_path.exists():
        summary = load_json(summary_path)
    else:
        summary = {}
    for key in (
        "coverage_score", "uniformity_score", "stability_score",
        "compositional_diversity_score", "structural_diversity_score", "redundancy_score",
    ):
        scores[key] = float(summary.get("summary", {}).get(key, 0.0))
    return scores


# ---------------------------------------------------------------------------
# Profile / Acquisition helpers
# ---------------------------------------------------------------------------

def profile_dataset(config: BuilderConfig, iteration: int | None = None) -> tuple[Path, dict[str, Any]]:
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name
    label = f"iter_{iteration:03d}" if iteration is not None else "seed"
    profile_dir = build_dir / "profiles" / label
    profile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning profile for {label}...")

    if config.dry_run:
        print(f"  [DRY-RUN] Would run: dataset-profile-run --source-type ase_db --datapath {db_path}")
        return profile_dir, {}

    result = _run_cli([
        "dataset-profile-run",
        "--source-type", "ase_db",
        "--datapath", str(db_path),
        "--output-root", str(profile_dir),
        "--preset", "core",
    ])

    if result.returncode != 0:
        print(f"  Profile stderr: {result.stderr}")
        raise RuntimeError(f"Profile failed: {result.returncode}")

    output_line = result.stdout.strip().split("\n")[-1]
    actual_dir = Path(output_line) if output_line else profile_dir
    scores = _extract_profile_scores(actual_dir)
    print(f"  Scores: {scores}")
    return actual_dir, scores


def run_acquisition(config: BuilderConfig, candidates_dir: Path, iteration: int) -> Path:
    build_dir = Path(config.build_dir)
    db_path = build_dir / config.db_name
    acq_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}" / "acquisition"
    acq_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning acquisition for iteration {iteration}...")

    if config.dry_run:
        print(f"  [DRY-RUN] Would run: acquisition-mode --original-dataset {db_path} --candidates {candidates_dir}")
        return acq_dir

    result = _run_cli([
        "acquisition-mode",
        "--original-dataset", str(db_path),
        "--candidates", str(candidates_dir),
        "--top-k", str(config.al_top_k),
        "--element-check", "strict",
        "--output-root", str(acq_dir),
    ])

    if result.returncode != 0:
        print(f"  Acquisition stderr: {result.stderr}")
        raise RuntimeError(f"Acquisition failed: {result.returncode}")

    output_line = result.stdout.strip().split("\n")[-1]
    actual_dir = Path(output_line) if output_line else acq_dir
    print(f"  Acquisition report: {actual_dir}")
    return actual_dir


def _generate_candidates_random(
    config: BuilderConfig,
    iteration: int,
    target_temperatures: list[float],
    cand_dir: Path,
) -> int:
    """Random candidate generation strategy for SiO2."""
    count = 0
    for i in range(config.al_candidate_pool_size):
        base_temp = target_temperatures[i % len(target_temperatures)]
        temp = base_temp + random.gauss(0, 10.0)

        # Pick structure type with weighted random choice
        struct_type = random.choices(
            config.structure_types,
            weights=config.structure_weights,
            k=1,
        )[0]

        atoms = generate_sio2_structure(
            structure_type=struct_type,
            n_atoms=config.n_atoms,
            temperature=temp,
            seed=hash(f"iter{iteration}_cand{i}") % 2**31,
        )
        xyz_path = cand_dir / f"candidate_{count:04d}.xyz"
        write(str(xyz_path), atoms, format="extxyz")
        count += 1

    return count


def _generate_candidates_dataset_aware(
    config: BuilderConfig,
    iteration: int,
    target_temperatures: list[float],
    cand_dir: Path,
) -> int:
    """Dataset-aware candidate generation using CandidateGenerator."""
    from atom_dataset_index.io import AseDbDatasetReader
    from atom_dataset_index.metric_core.descriptors import SOAPDescriptorAdapter

    db_path = Path(config.build_dir) / config.db_name

    dataset = None
    if db_path.exists():
        try:
            dataset = AseDbDatasetReader(db_path).read()
            print(f"  CandidateGen: loaded {len(dataset.records)} reference structures")
        except Exception as e:
            print(f"  CandidateGen: failed to load reference dataset ({e}), falling back to random")

    if dataset is None or not dataset.records:
        print("  CandidateGen: no reference data, falling back to random generation")
        return _generate_candidates_random(config, iteration, target_temperatures, cand_dir)

    adapter = SOAPDescriptorAdapter()
    gen = CandidateGenerator(dataset=dataset, adapter=adapter)

    strategy = config.candidate_strategy
    params = dict(config.candidate_strategy_params)

    avg_temp = sum(target_temperatures) / len(target_temperatures) if target_temperatures else 300.0
    if strategy in ("boundary", "composite") and "temperature_k" not in params:
        params["temperature_k"] = avg_temp

    try:
        records = gen.generate(
            strategy=strategy,
            n_samples=config.al_candidate_pool_size,
            **params,
        )
    except Exception as e:
        print(f"  CandidateGen: {strategy} failed ({e}), falling back to random")
        return _generate_candidates_random(config, iteration, target_temperatures, cand_dir)

    count = 0
    for record in records:
        atoms = record_to_ase(record)
        xyz_path = cand_dir / f"candidate_{count:04d}.xyz"
        write(str(xyz_path), atoms, format="extxyz")
        count += 1

    return count


def generate_candidates(config: BuilderConfig, iteration: int, target_temperatures: list[float]) -> Path:
    build_dir = Path(config.build_dir)
    cand_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {config.al_candidate_pool_size} candidates (strategy: {config.candidate_strategy})...")

    if config.candidate_strategy == "random":
        count = _generate_candidates_random(config, iteration, target_temperatures, cand_dir)
    else:
        count = _generate_candidates_dataset_aware(config, iteration, target_temperatures, cand_dir)

    print(f"  Written {count} candidates to {cand_dir}")
    return cand_dir


def prefilter_candidates(
    config: BuilderConfig,
    iteration: int,
    cand_dir: Path,
) -> tuple[Path, int, dict[str, int]]:
    """Pre-filter generated candidates using DeterministicPreFilter."""
    db_path = Path(config.build_dir) / config.db_name
    xyz_files = sorted(cand_dir.glob("*.xyz"))
    if not xyz_files:
        return cand_dir, 0, {}

    pf: DeterministicPreFilter | None = None
    if db_path.exists():
        try:
            ref_dataset = AseDbDatasetReader(db_path).read()
            pf = DeterministicPreFilter(ref_dataset)
            print(f"  PreFilter: loaded {len(ref_dataset.records)} reference structures")
        except Exception as e:
            print(f"  PreFilter: failed to load reference dataset ({e}), using conservative defaults")
            pf = DeterministicPreFilter()
    else:
        print("  PreFilter: no reference dataset found, using conservative defaults")
        pf = DeterministicPreFilter()

    assert pf is not None

    passed: list[Path] = []
    rejected: list[tuple[Path, str, str]] = []

    for xyz_path in xyz_files:
        try:
            atoms = read(str(xyz_path))
            record = ase_to_record(atoms, sample_id=0)
            result = pf.classify(record)
            if result.passed:
                passed.append(xyz_path)
            else:
                rejected.append((xyz_path, result.label, result.reason))
        except Exception as e:
            rejected.append((xyz_path, "AB", f"Parse/convert error: {e}"))

    # If ALL candidates are rejected, keep the unfiltered pool so acquisition
    # can still run (common with small seed datasets where force envelopes are
    # too tight).
    if not passed and rejected:
        print(
            f"  WARNING: PreFilter rejected ALL {len(rejected)} candidates. "
            "Keeping unfiltered pool for this iteration."
        )
        return cand_dir, 0, {}

    # ------------------------------------------------------------------
    # Save AB/LU rejected examples for cohort extraction.
    # Writes to outputs/ne_cohort/ — never touches build_dir.
    # Failures are silently ignored so the main workflow is never blocked.
    # ------------------------------------------------------------------
    try:
        cohort_rejected_dir = Path("outputs/ne_cohort/rejected_examples")
        cohort_rejected_dir.mkdir(parents=True, exist_ok=True)
        for xyz_path, label, reason in rejected:
            if label in ("AB", "LU"):
                atoms = read(str(xyz_path))
                out_name = f"{xyz_path.stem}_{label}_iter{iteration:03d}.xyz"
                write(str(cohort_rejected_dir / out_name), atoms)
    except Exception:
        pass  # Cohort saving is best-effort; never block the AL loop.

    for xyz_path, label, reason in rejected:
        try:
            xyz_path.unlink()
        except OSError:
            pass

    for i, xyz_path in enumerate(sorted(passed)):
        new_name = xyz_path.parent / f"candidate_{i:04d}.xyz"
        if xyz_path != new_name:
            xyz_path.rename(new_name)

    stats: dict[str, int] = {}
    for _, label, _ in rejected:
        stats[label] = stats.get(label, 0) + 1

    print(
        f"  PreFilter: {len(passed)}/{len(xyz_files)} passed, "
        f"{len(rejected)} rejected (stats: {stats})"
    )

    return cand_dir, len(rejected), stats


def _run_force_field_prescreen(
    config: BuilderConfig,
    iteration: int,
    cand_dir: Path,
) -> Path:
    """Use a fast ASE built-in force field to estimate energy/forces for candidates.

    This replaces the CP2K ENERGY_FORCE single-point pre-screening step. The
    estimated values are written into the candidate XYZ files as SinglePointCalculator
    results so that the downstream prefilter and acquisition steps see energy/force
    tags. Production AIMD trajectories are still generated with CP2K DFT.
    """
    xyz_files = sorted(cand_dir.glob("*.xyz"))
    if not xyz_files:
        return cand_dir

    if config.dry_run:
        print(
            f"\n[DRY-RUN] Would run force-field pre-screening for "
            f"{len(xyz_files)} candidates."
        )
        return cand_dir

    print(
        f"\nForce-field pre-screening {len(xyz_files)} candidates "
        f"(type: {config.force_field_type})..."
    )
    n_ok = 0
    n_fail = 0

    for xyz_path in xyz_files:
        try:
            atoms = read(str(xyz_path))
            atoms.set_cell(atoms.get_cell(), scale_atoms=True)
            atoms = attach_force_field_calculator(
                atoms,
                force_field_type=config.force_field_type,
                force_field_param_file=config.force_field_param_file or None,
            )
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=forces,
            )
            write(str(xyz_path), atoms, format="extxyz")
            n_ok += 1
        except Exception as exc:
            print(f"  WARNING: force-field pre-screen failed for {xyz_path.name}: {exc}")
            n_fail += 1

    print(f"  Force-field pre-screen: {n_ok} succeeded, {n_fail} failed")
    return cand_dir


def run_candidate_single_points(
    config: BuilderConfig,
    client: RemoteSLURMClient,
    iteration: int,
    cand_dir: Path,
    ntasks_per_node: int = 16,
) -> Path:
    """Run CP2K ENERGY_FORCE single-point calculations for all candidates.

    If ``config.use_force_field`` is True (the default), this function instead
    runs a fast local force-field pre-screen and skips CP2K entirely.

    Each candidate gets its own small SLURM job. The function blocks until all
    jobs finish, then downloads the CP2K ``out`` files, parses energy and
    forces, and rewrites the local candidate XYZ files with the computed
    values so that downstream prefilter / acquisition see real DFT data.
    """
    if config.use_force_field:
        return _run_force_field_prescreen(config, iteration, cand_dir)

    xyz_files = sorted(cand_dir.glob("*.xyz"))
    if not xyz_files:
        return cand_dir

    if config.dry_run:
        print(
            f"\n[DRY-RUN] Would run CP2K single-point calculations for "
            f"{len(xyz_files)} candidates."
        )
        return cand_dir

    build_dir = Path(config.build_dir)
    sp_root = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}" / "single_point"
    sp_root.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning CP2K single-point calculations for {len(xyz_files)} candidates...")
    print(f"  Root directory: {sp_root}")
    print(f"  Cores per job:  {ntasks_per_node}")

    job_infos: list[dict[str, Any]] = []
    already_done: list[Path] = []

    for i, xyz_path in enumerate(xyz_files):
        atoms = read(str(xyz_path))
        if atoms.calc is not None:
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                if (
                    energy is not None
                    and forces is not None
                    and len(forces) == len(atoms)
                ):
                    already_done.append(xyz_path)
                    continue
            except Exception:
                pass

        run_name = f"sp_iter{iteration:03d}_cand{i:04d}"
        local_run_dir = sp_root / f"cand{i:04d}"
        local_run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(xyz_path, local_run_dir / "init.xyz")
        atoms = read(str(xyz_path))
        atoms.set_cell(atoms.get_cell(), scale_atoms=True)

        inp = build_cp2k_input_sio2(
            atoms=atoms,
            project_name=run_name,
            temperature=300.0,
            n_steps=1,
            timestep=1.0,
            output_stride=1,
            cutoff=config.cp2k_cutoff,
            basis_set=config.cp2k_basis_set,
            scf_guess="ATOMIC",
            electronic_temperature=None,
            cp2k_data_dir=config.cp2k_data_dir,
            run_type="ENERGY_FORCE",
        )
        with (local_run_dir / "input.inp").open("w") as f:
            f.write(inp)

        _write_submit_script(
            run_dir=local_run_dir,
            job_name=run_name,
            nodes=1,
            ppn=ntasks_per_node,
            nprocs=ntasks_per_node,
            partition=config.slurm_partition,
            with_geo_opt=False,
        )

        remote_subdir = (
            "phase_1_al_loop",
            f"iter_{iteration:03d}",
            "single_point",
            f"cand{i:04d}",
        )
        client.upload_run(local_run_dir, *remote_subdir)
        job_id = client.submit_existing_script(*remote_subdir, script_name="submit.slurm")

        job_infos.append({
            "job_id": job_id,
            "local_run_dir": local_run_dir,
            "remote_subdir": remote_subdir,
            "xyz_path": xyz_path,
            "run_name": run_name,
        })
        print(f"  Submitted {run_name} -> job {job_id}")

    job_ids = [j["job_id"] for j in job_infos]
    print(f"\nWaiting for {len(job_ids)} single-point jobs to finish...")
    client.wait_for_jobs(job_ids, poll_interval=30)

    n_success = 0
    n_failed = 0
    for info in job_infos:
        client.download_results(
            info["local_run_dir"],
            *info["remote_subdir"],
            patterns=["out", "err"],
        )
        out_path = info["local_run_dir"] / "out"
        result = parse_cp2k_single_point_out(out_path)
        if result is None:
            print(f"  WARNING: failed to parse single-point result for {info['run_name']}")
            n_failed += 1
            continue

        atoms = read(str(info["xyz_path"]))
        atoms.set_cell(atoms.get_cell(), scale_atoms=True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=result["energy"],
            forces=result["forces"],
        )
        write(str(info["xyz_path"]), atoms, format="extxyz")
        n_success += 1

    print(
        f"  Single-point calculations: {n_success} succeeded, {n_failed} failed, "
        f"{len(already_done)} already had energy/forces"
    )
    return cand_dir


def select_target_temperatures(
    config: BuilderConfig,
    profile_scores: dict[str, float],
    prev_targets: list[float] | None = None,
) -> list[float]:
    all_temps = list(config.seed_temperatures)
    extended = []
    for i in range(len(all_temps) - 1):
        extended.append(all_temps[i])
        mid = (all_temps[i] + all_temps[i + 1]) / 2
        extended.append(mid)
    extended.append(all_temps[-1])

    if prev_targets:
        extended = [t + random.choice([-20, -10, 10, 20]) for t in extended]

    selected = []
    step = max(1, len(extended) // config.al_runs_per_iteration)
    for i in range(0, len(extended), step):
        selected.append(extended[i])
        if len(selected) >= config.al_runs_per_iteration:
            break

    return selected[:config.al_runs_per_iteration]


def _choose_induction_parameters(
    config: BuilderConfig,
    iteration: int,
    run_idx: int,
) -> dict[str, Any]:
    """Choose induction parameters for a given AL run.

    Returns mode-specific kwargs such as temperature, pressure, strain axis,
    and strain value. Uses a deterministic random seed derived from iteration
    and run index so that dry-runs are reproducible.
    """
    rng = random.Random(iteration * 1000 + run_idx)
    mode = config.al_induction_mode

    if mode == "annealing":
        return {"t_high": rng.choice(config.annealing_temps_high)}
    if mode == "ballistic":
        return {
            "v_temp": rng.choice(config.ballistic_velocity_temps),
            "ensemble": rng.choice(["NVE", "NVT"]),
        }
    if mode == "npt":
        return {
            "pressure": rng.choice(config.npt_pressures_gpa),
            "temperature": rng.choice(config.npt_temperatures),
        }
    if mode == "strain":
        return {
            "strain_axis": rng.choice(config.strain_axes),
            "strain_value": rng.choice(config.strain_values),
            "temperature": rng.choice(config.npt_temperatures),
        }
    return {"temperature": config.seed_temperatures[run_idx % len(config.seed_temperatures)]}


# ---------------------------------------------------------------------------
# Seed Phase — Remote
# ---------------------------------------------------------------------------

def run_seed_submit(config: BuilderConfig, client: RemoteSLURMClient) -> None:
    """在本地生成种子配置文件，上传到远程集群并提交 SLURM 作业。"""
    print("\n" + "=" * 60)
    print("PHASE 0: SEED DATASET GENERATION (REMOTE SUBMIT)")
    print("=" * 60)

    build_dir = _ensure_dir(config.build_dir)
    seed_dir = _ensure_dir(build_dir / "phase_0_seed" / "configs")
    manifest = load_remote_manifest(build_dir)

    if manifest["seed"].get("submitted") and not config.dry_run:
        print("\nWARNING: Seed jobs have already been submitted.")
        print("  Re-submitting will create duplicate SLURM jobs.")

    jobs_to_submit: list[tuple[str, Path, tuple[str, ...]]] = []
    for temp in config.seed_temperatures:
        for run_idx in range(1, config.seed_runs_per_temp + 1):
            run_name = f"T{int(temp):04d}_run{run_idx:02d}"
            run_dir = seed_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            struct_type = random.choices(
                config.structure_types,
                weights=config.structure_weights,
                k=1,
            )[0]
            atoms = generate_sio2_structure(
                structure_type=struct_type,
                n_atoms=config.n_atoms,
                temperature=temp,
                seed=hash(run_name) % 2**31,
            )
            init_xyz = run_dir / "init.xyz"
            write(str(init_xyz), atoms)

            # Step 1: GEO_OPT
            geo_project = f"{run_name}_geo"
            geo_inp = build_cp2k_input_sio2(
                atoms=atoms,
                project_name=geo_project,
                temperature=temp,
                n_steps=config.seed_n_steps,
                timestep=config.seed_timestep,
                output_stride=config.seed_output_stride,
                cutoff=config.cp2k_cutoff,
                basis_set=config.cp2k_basis_set,
                scf_guess="ATOMIC",
                electronic_temperature=None,
                cp2k_data_dir=config.cp2k_data_dir,
                run_type="GEO_OPT",
            )
            with (run_dir / "geo_opt.inp").open("w") as f:
                f.write(geo_inp)

            # Step 2: MD
            # SiO2 is an insulator; use OT without smearing for all temperatures.
            inp_content = build_cp2k_input_sio2(
                atoms=atoms,
                project_name=run_name,
                temperature=temp,
                n_steps=config.seed_n_steps,
                timestep=config.seed_timestep,
                output_stride=config.seed_output_stride,
                cutoff=config.cp2k_cutoff,
                basis_set=config.cp2k_basis_set,
                scf_guess="RESTART",
                use_ot=True,
                wfn_restart_file=f"{geo_project}-RESTART.wfn",
                ext_restart_file=f"{geo_project}-1.restart",
                cp2k_data_dir=config.cp2k_data_dir,
            )
            with (run_dir / "input.inp").open("w") as f:
                f.write(inp_content)

            _write_submit_script(
                run_dir,
                job_name=run_name,
                nodes=config.cp2k_nodes,
                ppn=config.cp2k_ppn,
                nprocs=config.cp2k_nprocs,
                partition=config.slurm_partition,
                with_geo_opt=True,
                geo_restart_file=f"{geo_project}-1.restart",
            )

            if config.dry_run:
                print(f"  [DRY-RUN] Would upload and submit {run_name}")
                continue

            remote_subdir = ("seed", run_name)
            jobs_to_submit.append((run_name, run_dir, remote_subdir))

    if not jobs_to_submit:
        print("\nNo jobs to submit (dry-run or already done).")
        return

    print(f"\nSubmitting {len(jobs_to_submit)} jobs concurrently (max 8 workers)...")
    manifest_lock = threading.Lock()
    total_jobs = 0

    def _submit_one(job: tuple[str, Path, tuple[str, ...]]) -> str | None:
        run_name, run_dir, remote_subdir = job
        try:
            client.upload_run(run_dir, *remote_subdir)
        except RuntimeError as exc:
            print(f"  ERROR uploading {run_name}: {exc}")
            return None

        try:
            job_id = client.submit_existing_script(*remote_subdir)
        except RuntimeError as exc:
            print(f"  ERROR submitting {run_name}: {exc}")
            return None

        with manifest_lock:
            manifest["seed"]["jobs"].append({
                "job_id": job_id,
                "run_name": run_name,
                "remote_dir": "/".join(remote_subdir),
                "local_dir": str(run_dir.relative_to(build_dir)),
                "status": "pending",
                "n_frames": 0,
            })
        print(f"  Submitted {run_name} -> job {job_id}")
        return job_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_submit_one, job): job[0] for job in jobs_to_submit}
        for future in as_completed(futures):
            run_name = futures[future]
            try:
                if future.result() is not None:
                    total_jobs += 1
            except Exception as exc:
                print(f"  ERROR in {run_name}: {exc}")

    if total_jobs > 0:
        manifest["seed"]["submitted"] = True
    save_remote_manifest(build_dir, manifest)
    print(f"\nSeed submit complete. {total_jobs} jobs submitted.")
    print(f"Manifest saved to {build_dir / REMOTE_MANIFEST_NAME}")


def run_seed_collect(config: BuilderConfig, client: RemoteSLURMClient) -> int:
    """从远程集群下载完成的种子作业结果，并合并到本地 ASE 数据库。"""
    print("\n" + "=" * 60)
    print("PHASE 0: SEED DATASET COLLECTION (REMOTE)")
    print("=" * 60)

    build_dir = _ensure_dir(config.build_dir)
    db_path = build_dir / config.db_name
    manifest = load_remote_manifest(build_dir)

    if not manifest["seed"].get("submitted"):
        print("ERROR: No seed jobs have been submitted yet. Run 'seed-submit' first.")
        return 0

    jobs = manifest["seed"]["jobs"]
    total_added = 0

    for job in jobs:
        if job.get("status") == "collected":
            print(f"  {job['run_name']}: already collected ({job.get('n_frames', 0)} frames), skipping...")
            continue

        run_name = job["run_name"]
        run_dir = build_dir / job["local_dir"]
        remote_dir_parts = job["remote_dir"].split("/")

        if config.dry_run:
            print(f"  [DRY-RUN] Would collect {run_name}")
            continue

        job_id = job["job_id"]
        try:
            running = client.is_job_running(job_id)
        except Exception as exc:
            print(f"  WARNING: Could not query status for {run_name} ({job_id}): {exc}")
            running = True

        if running:
            job["status"] = "running"
            print(f"  {run_name} ({job_id}): still running, skipping...")
            continue

        print(f"  {run_name} ({job_id}): downloading results...")
        try:
            downloaded = client.download_results(run_dir, *remote_dir_parts)
            print(f"    Downloaded: {downloaded}")
        except RuntimeError as exc:
            print(f"    WARNING: download failed: {exc}")

        pos_xyz = run_dir / f"{run_name}-pos-1.xyz"
        frc_xyz = run_dir / f"{run_name}-frc-1.xyz"
        ener_file = run_dir / f"{run_name}-1.ener"
        if not ener_file.exists():
            ener_file = run_dir / f"{run_name}.ener"

        if not pos_xyz.exists():
            print(f"    WARNING: No trajectory found for {run_name}")
            job["status"] = "completed"
            continue

        metadata = {
            "temperature": float(run_name.split("_")[0][1:]),
            "phase": "seed",
            "run_name": run_name,
        }
        cell = parse_cp2k_cell(run_dir / "out")
        if cell is None:
            print(f"    WARNING: Could not parse cell from {run_name}/out")
        added = merge_trajectory_to_db(
            db_path=db_path,
            pos_xyz_path=pos_xyz,
            frc_xyz_path=frc_xyz if frc_xyz.exists() else None,
            ener_path=ener_file if ener_file.exists() else None,
            metadata=metadata,
            subsample=1,
            event_driven=config.event_driven_sampling,
            event_window=config.event_window,
            sparse_stride=config.sparse_stride,
            cell=cell,
        )
        total_added += added
        job["status"] = "collected"
        job["n_frames"] = added
        print(f"    {run_name}: added {added} frames")

    save_remote_manifest(build_dir, manifest)
    print(f"\nSeed collect complete. Total frames added this run: {total_added}")
    return total_added


# ---------------------------------------------------------------------------
# AL Loop — Remote
# ---------------------------------------------------------------------------

def run_al_submit(config: BuilderConfig, client: RemoteSLURMClient) -> None:
    """执行一轮主动学习迭代：数据集剖析 → 候选生成 → 采集评分 → 提交远程作业。"""
    print("\n" + "=" * 60)
    print("PHASE 1: ACTIVE LEARNING SUBMIT (REMOTE)")
    print("=" * 60)

    build_dir = _ensure_dir(config.build_dir)
    db_path = build_dir / config.db_name
    manifest = load_remote_manifest(build_dir)

    existing_iters = manifest.get("al_iterations", [])
    iteration = len(existing_iters) + 1

    if iteration > config.al_max_iterations:
        print(f"Max iterations ({config.al_max_iterations}) reached. Stopping.")
        manifest["termination"] = {"stopped": True, "reason": "max_iterations"}
        save_remote_manifest(build_dir, manifest)
        return

    print(f"\n--- AL Iteration {iteration} / {config.al_max_iterations} ---")
    profile_dir, scores = profile_dataset(config, iteration=iteration)

    db_stats = get_db_stats(db_path)
    dataset_size = db_stats.get("n_structures", 0)

    coverage = scores.get("coverage_score", 0.0)
    uniformity = scores.get("uniformity_score", 0.0)
    diversity = scores.get("structural_diversity_score", 0.0)
    stability = scores.get("stability_score", 0.0)

    print(f"\nCurrent status:")
    print(f"  dataset_size:    {dataset_size} / {config.target_size}")
    print(f"  coverage:        {coverage:.3f} / {config.min_coverage}")
    print(f"  uniformity:      {uniformity:.3f} / {config.min_uniformity}")
    print(f"  diversity:       {diversity:.3f} / {config.min_diversity}")
    print(f"  stability:       {stability:.3f} / {config.min_stability}")

    should_stop = (
        dataset_size >= config.target_size
        and coverage >= config.min_coverage
        and uniformity >= config.min_uniformity
        and diversity >= config.min_diversity
        and stability >= config.min_stability
    )

    if should_stop:
        print("\n*** ALL TERMINATION CONDITIONS MET ***")
        manifest["termination"] = {"stopped": True, "reason": "all_conditions_met"}
        save_remote_manifest(build_dir, manifest)
        return

    prev_targets = existing_iters[-1].get("target_temperatures") if existing_iters else None
    target_temps = select_target_temperatures(config, scores, prev_targets)
    print(f"\nTarget temperatures: {target_temps}")

    cand_dir = generate_candidates(config, iteration, target_temps)

    # Run real CP2K single-point energy/force calculations for every candidate
    # before prefilter / acquisition so that downstream steps use DFT data
    # instead of harmonic force estimates.
    cand_dir = run_candidate_single_points(config, client, iteration, cand_dir)

    cand_dir, n_rejected, rejection_stats = prefilter_candidates(
        config, iteration, cand_dir
    )

    acq_dir = run_acquisition(config, cand_dir, iteration)

    acq_report_path = acq_dir / "acquisition_report.json"
    selected_candidates = []
    if acq_report_path.exists():
        report = load_json(acq_report_path)
        rankings = report.get("rankings", [])
        for rank in rankings[:config.al_top_k]:
            poscar_name = Path(rank.get("poscar_path", "")).name
            poscar_path = cand_dir / poscar_name
            if poscar_path.exists():
                selected_candidates.append({
                    "poscar": poscar_path,
                    "temperature": target_temps[len(selected_candidates) % len(target_temps)],
                    "score": rank.get("total_value", 0.0),
                })

    if not selected_candidates:
        print("  No acquisition rankings, using random selection from candidates.")
        all_structs = sorted(cand_dir.glob("*.xyz"))
        for i, struct in enumerate(all_structs[:config.al_top_k]):
            selected_candidates.append({
                "poscar": struct,
                "temperature": target_temps[i % len(target_temps)],
                "score": 0.0,
            })

    print(f"  Selected {len(selected_candidates)} candidates for CP2K")

    # ------------------------------------------------------------------
    # Phase A: generate all local input files serially
    # ------------------------------------------------------------------
    al_jobs_to_submit: list[dict[str, Any]] = []
    for i, cand in enumerate(selected_candidates):
        base_name = f"iter{iteration:03d}_cfg{i:03d}"
        kwargs = _choose_induction_parameters(config, iteration, i)

        if config.al_induction_mode == "annealing":
            t_high = kwargs["t_high"]
            # Use a smaller timestep for the high-temperature hold phase to
            # improve stability when t_high exceeds ~2500 K.
            high_temp_timestep = 0.5 if t_high > 2500.0 else config.seed_timestep
            phases_def: list[dict[str, Any]] = [
                {"phase_name": "relax", "temperature": 300.0, "n_steps": config.al_annealing_relax_steps, "wfn_restart": None, "ext_restart": None, "induction_params": {"t_high": t_high}},
                {"phase_name": "heating", "temperature": config.al_annealing_heating_temp, "n_steps": config.al_annealing_heating_steps, "wfn_restart": f"../relax/{base_name}_relax_geo-RESTART.wfn", "ext_restart": f"../relax/{base_name}_relax-1.restart", "induction_params": {"t_high": t_high}},
                {"phase_name": "high_temp", "temperature": t_high, "n_steps": config.al_annealing_high_temp_steps, "timestep": high_temp_timestep, "wfn_restart": f"../heating/{base_name}_heating-RESTART.wfn", "ext_restart": f"../heating/{base_name}_heating-1.restart", "induction_params": {"t_high": t_high}},
                {"phase_name": "quench", "temperature": 300.0, "n_steps": config.al_annealing_quench_steps, "wfn_restart": f"../high_temp/{base_name}_high_temp-RESTART.wfn", "ext_restart": f"../high_temp/{base_name}_high_temp-1.restart", "induction_params": {"t_high": t_high}},
                {"phase_name": "hold", "temperature": 300.0, "n_steps": config.al_annealing_hold_steps, "wfn_restart": f"../quench/{base_name}_quench-RESTART.wfn", "ext_restart": f"../quench/{base_name}_quench-1.restart", "induction_params": {"t_high": t_high}},
            ]
        elif config.al_induction_mode == "npt":
            phases_def = [
                {"phase_name": "single", "temperature": kwargs["temperature"], "n_steps": config.al_n_steps, "wfn_restart": None, "ext_restart": None, "ensemble": "NPT_I", "pressure_gpa": kwargs["pressure"], "induction_params": {"pressure": kwargs["pressure"], "temperature": kwargs["temperature"]}},
            ]
        elif config.al_induction_mode == "strain":
            phases_def = [
                {"phase_name": "single", "temperature": kwargs["temperature"], "n_steps": config.al_n_steps, "wfn_restart": None, "ext_restart": None, "strain_axis": kwargs["strain_axis"], "strain_value": kwargs["strain_value"], "induction_params": {"strain_axis": kwargs["strain_axis"], "strain_value": kwargs["strain_value"], "temperature": kwargs["temperature"]}},
            ]
        elif config.al_induction_mode == "ballistic":
            # Ballistic runs start with high kinetic energy; use a smaller
            # timestep to maintain numerical stability.
            phases_def = [
                {"phase_name": "single", "temperature": cand["temperature"], "n_steps": config.al_n_steps, "timestep": 0.5, "wfn_restart": None, "ext_restart": None, "ballistic_velocity_temp": kwargs["v_temp"], "ballistic_ensemble": kwargs["ensemble"], "ensemble": kwargs["ensemble"], "induction_params": {"v_temp": kwargs["v_temp"], "ensemble": kwargs["ensemble"]}},
            ]
        else:  # md
            phases_def = [
                {"phase_name": "single", "temperature": cand["temperature"], "n_steps": config.al_n_steps, "wfn_restart": None, "ext_restart": None, "induction_params": {"temperature": cand["temperature"]}},
            ]

        job_def: dict[str, Any] = {"base_name": base_name, "cand": cand, "phases": []}

        for phase in phases_def:
            phase_name = phase["phase_name"]
            temp = phase["temperature"]
            n_steps = phase["n_steps"]
            wfn_restart = phase.get("wfn_restart")
            ext_restart = phase.get("ext_restart")

            if phase_name != "single":
                run_name = f"{base_name}_{phase_name}"
                run_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}" / "selected" / base_name / phase_name
                remote_subdir = ("phase_1_al_loop", f"iter_{iteration:03d}", "selected", base_name, phase_name)
            else:
                run_name = base_name
                run_dir = build_dir / "phase_1_al_loop" / f"iter_{iteration:03d}" / "selected" / base_name
                remote_subdir = ("phase_1_al_loop", f"iter_{iteration:03d}", "selected", base_name)

            run_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(cand["poscar"], run_dir / "init.xyz")
            atoms = read(str(cand["poscar"]))
            atoms.set_cell(atoms.get_cell(), scale_atoms=True)

            # Apply directional mechanical strain for strain induction mode
            if config.al_induction_mode == "strain":
                strain_axis = phase["strain_axis"]
                strain_value = phase["strain_value"]
                cell = atoms.get_cell()
                if strain_axis == "x":
                    cell[0] *= 1.0 + strain_value
                elif strain_axis == "y":
                    cell[1] *= 1.0 + strain_value
                elif strain_axis == "z":
                    cell[2] *= 1.0 + strain_value
                atoms.set_cell(cell, scale_atoms=True)

            # GEO_OPT only for the first phase (no prior restart)
            need_geo_opt = wfn_restart is None
            if need_geo_opt:
                geo_project = f"{run_name}_geo"
                geo_inp = build_cp2k_input_sio2(
                    atoms=atoms,
                    project_name=geo_project,
                    temperature=temp,
                    n_steps=n_steps,
                    timestep=config.seed_timestep,
                    output_stride=config.al_output_stride,
                    cutoff=config.cp2k_cutoff,
                    basis_set=config.cp2k_basis_set,
                    scf_guess="ATOMIC",
                    electronic_temperature=None,
                    cp2k_data_dir=config.cp2k_data_dir,
                    run_type="GEO_OPT",
                )
                with (run_dir / "geo_opt.inp").open("w") as f:
                    f.write(geo_inp)
                md_ext_restart = f"{geo_project}-1.restart"
                wfn_restart_file = f"{geo_project}-RESTART.wfn"
            else:
                wfn_restart_file = wfn_restart
                md_ext_restart = ext_restart

            # Production MD: NVT for md/annealing/strain, NPT_I for npt
            ensemble = phase.get("ensemble", "NVT")
            pressure_gpa = phase.get("pressure_gpa", 0.0)
            phase_timestep = phase.get("timestep", config.seed_timestep)
            inp_content = build_cp2k_input_sio2(
                atoms=atoms,
                project_name=run_name,
                temperature=temp,
                n_steps=n_steps,
                timestep=phase_timestep,
                output_stride=config.al_output_stride,
                cutoff=config.cp2k_cutoff,
                basis_set=config.cp2k_basis_set,
                scf_guess="RESTART",
                use_ot=True,
                wfn_restart_file=wfn_restart_file,
                ext_restart_file=md_ext_restart,
                cp2k_data_dir=config.cp2k_data_dir,
                ensemble=ensemble,
                pressure_gpa=pressure_gpa,
                ballistic_velocity_temp=phase.get("ballistic_velocity_temp"),
                ballistic_ensemble=phase.get("ballistic_ensemble", "NVE"),
            )
            with (run_dir / "input.inp").open("w") as f:
                f.write(inp_content)

            _write_submit_script(
                run_dir,
                job_name=run_name,
                nodes=config.cp2k_nodes,
                ppn=config.cp2k_ppn,
                nprocs=config.cp2k_nprocs,
                partition=config.slurm_partition,
                with_geo_opt=need_geo_opt,
                geo_restart_file=md_ext_restart if need_geo_opt else "",
            )

            job_def["phases"].append({
                "phase_name": phase_name,
                "run_name": run_name,
                "run_dir": run_dir,
                "remote_subdir": remote_subdir,
                "temperature": temp,
                "wfn_restart": wfn_restart,
                "ext_restart": ext_restart,
                "need_geo_opt": need_geo_opt,
                "induction_params": phase.get("induction_params", {}),
            })

        if config.dry_run:
            print(f"  [DRY-RUN] Would upload and submit {base_name} ({config.al_induction_mode})")
            continue

        al_jobs_to_submit.append(job_def)

    iter_jobs: list[dict[str, Any]] = []
    if al_jobs_to_submit:
        print(f"\nSubmitting {len(al_jobs_to_submit)} AL job chains concurrently (max 8 workers)...")
        jobs_lock = threading.Lock()

        def _submit_al_one(job_def: dict[str, Any]) -> list[dict[str, Any]] | None:
            base_name = job_def["base_name"]
            cand = job_def["cand"]
            phases = job_def["phases"]
            phase_results: list[dict[str, Any]] = []
            prev_job_id: str | None = None

            for phase in phases:
                run_name = phase["run_name"]
                run_dir = phase["run_dir"]
                remote_subdir = phase["remote_subdir"]

                try:
                    client.upload_run(run_dir, *remote_subdir)
                except RuntimeError as exc:
                    print(f"  ERROR uploading {run_name}: {exc}")
                    return None

                try:
                    job_id = client.submit_existing_script(
                        *remote_subdir,
                        depend_job_id=prev_job_id,
                    )
                except RuntimeError as exc:
                    print(f"  ERROR submitting {run_name}: {exc}")
                    return None

                phase_results.append({
                    "job_id": job_id,
                    "run_name": run_name,
                    "phase": phase["phase_name"],
                    "remote_dir": "/".join(remote_subdir),
                    "local_dir": str(run_dir.relative_to(build_dir)),
                    "status": "pending",
                    "n_frames": 0,
                    "temperature": phase["temperature"],
                    "depend_job_id": prev_job_id,
                    "induction_mode": config.al_induction_mode,
                    "induction_params": phase.get("induction_params", {}),
                })
                dep_info = f" (afterok:{prev_job_id})" if prev_job_id else ""
                print(f"  Submitted {run_name} -> job {job_id}{dep_info}")
                prev_job_id = job_id

            if len(phase_results) == 1:
                entry: dict[str, Any] = {**phase_results[0], "score": cand["score"]}
            else:
                entry = {
                    "annealing_chain": True,
                    "base_name": base_name,
                    "phases": phase_results,
                    "temperature": cand["temperature"],
                    "score": cand["score"],
                    "induction_mode": config.al_induction_mode,
                }

            with jobs_lock:
                iter_jobs.append(entry)
            return phase_results

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_submit_al_one, job): job["base_name"] for job in al_jobs_to_submit}
            for future in as_completed(futures):
                base_name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"  ERROR in {base_name}: {exc}")

    if not config.dry_run:
        manifest["al_iterations"].append({
            "iteration": iteration,
            "submitted": True,
            "collected": False,
            "jobs": iter_jobs,
            "target_temperatures": target_temps,
            "induction_mode": config.al_induction_mode,
            "n_new_frames": 0,
            "metrics_after": {
                "dataset_size": dataset_size,
                "coverage_score": coverage,
                "uniformity_score": uniformity,
                "compositional_diversity_score": scores.get("compositional_diversity_score", 0.0),
                "structural_diversity_score": diversity,
                "stability_score": stability,
                "redundancy_score": scores.get("redundancy_score", 0.0),
            },
        })

        save_remote_manifest(build_dir, manifest)

    print(f"\nAL iteration {iteration} submit complete. {len(iter_jobs)} jobs submitted.")


def run_al_collect(config: BuilderConfig, client: RemoteSLURMClient) -> int:
    """下载最近一轮 AL 迭代的完成结果并合并到数据库。"""
    print("\n" + "=" * 60)
    print("PHASE 1: ACTIVE LEARNING COLLECTION (REMOTE)")
    print("=" * 60)

    build_dir = _ensure_dir(config.build_dir)
    db_path = build_dir / config.db_name
    manifest = load_remote_manifest(build_dir)
    al_iters = manifest.get("al_iterations", [])

    if not al_iters:
        print("ERROR: No AL iterations have been submitted yet. Run 'al-submit' first.")
        return 0

    pending_iter = None
    for it in reversed(al_iters):
        if not it.get("collected"):
            pending_iter = it
            break

    if pending_iter is None:
        print("All AL iterations are already collected.")
        return 0

    iteration = pending_iter["iteration"]
    print(f"\nCollecting AL iteration {iteration}...")

    jobs = pending_iter["jobs"]
    total_added = 0
    all_done = True

    def _collect_phase(phase: dict[str, Any]) -> tuple[int, bool]:
        if phase.get("status") == "collected":
            return 0, True

        run_name = phase["run_name"]
        run_dir = build_dir / phase["local_dir"]
        remote_dir_parts = phase["remote_dir"].split("/")

        if config.dry_run:
            print(f"  [DRY-RUN] Would collect {run_name}")
            return 0, False

        job_id = phase["job_id"]
        try:
            running = client.is_job_running(job_id)
        except Exception as exc:
            print(f"  WARNING: Could not query status for {run_name} ({job_id}): {exc}")
            running = True

        if running:
            phase["status"] = "running"
            print(f"  {run_name} ({job_id}): still running, skipping...")
            return 0, False

        print(f"  {run_name} ({job_id}): downloading results...")
        try:
            downloaded = client.download_results(run_dir, *remote_dir_parts)
            print(f"    Downloaded: {downloaded}")
        except RuntimeError as exc:
            print(f"    WARNING: download failed: {exc}")

        pos_xyz = run_dir / f"{run_name}-pos-1.xyz"
        frc_xyz = run_dir / f"{run_name}-frc-1.xyz"
        ener_file = run_dir / f"{run_name}-1.ener"
        if not ener_file.exists():
            ener_file = run_dir / f"{run_name}.ener"

        if not pos_xyz.exists():
            print(f"    WARNING: No trajectory for {run_name}")
            phase["status"] = "completed"
            return 0, True

        metadata = {
            "temperature": phase.get("temperature", 300.0),
            "phase": f"al_iter{iteration}",
            "run_name": run_name,
            "acquisition_score": phase.get("score", 0.0),
            "induction_mode": phase.get("induction_mode", config.al_induction_mode),
            "induction_params": phase.get("induction_params", {}),
        }
        cell = parse_cp2k_cell(run_dir / "out")
        if cell is None:
            print(f"    WARNING: Could not parse cell from {run_name}/out")
        added = merge_trajectory_to_db(
            db_path=db_path,
            pos_xyz_path=pos_xyz,
            frc_xyz_path=frc_xyz if frc_xyz.exists() else None,
            ener_path=ener_file if ener_file.exists() else None,
            metadata=metadata,
            subsample=1,
            event_driven=config.event_driven_sampling,
            event_window=config.event_window,
            sparse_stride=config.sparse_stride,
            cell=cell,
        )
        phase["status"] = "collected"
        phase["n_frames"] = added
        print(f"    {run_name}: added {added} frames")
        return added, True

    for job in jobs:
        if job.get("status") == "collected":
            continue

        if job.get("annealing_chain"):
            chain_done = True
            chain_frames = 0
            for phase in job["phases"]:
                frames, done = _collect_phase(phase)
                chain_frames += frames
                if not done:
                    chain_done = False
            if chain_done:
                job["status"] = "collected"
                job["n_frames"] = chain_frames
                total_added += chain_frames
            else:
                all_done = False
        else:
            frames, done = _collect_phase(job)
            total_added += frames
            if not done:
                all_done = False

    pending_iter["n_new_frames"] = pending_iter.get("n_new_frames", 0) + total_added

    if all_done:
        pending_iter["collected"] = True
        if not config.dry_run:
            _cleanup_iteration_intermediates(build_dir, iteration)

    db_stats = get_db_stats(db_path)
    dataset_size = db_stats.get("n_structures", 0)
    pending_iter["metrics_after"]["dataset_size"] = dataset_size

    save_remote_manifest(build_dir, manifest)
    print(f"\nAL iteration {iteration} collect complete. Added {total_added} frames this run.")
    print(f"Total dataset size: {dataset_size}")
    return total_added


# ---------------------------------------------------------------------------
# Auto / Status
# ---------------------------------------------------------------------------

def run_auto(config: BuilderConfig, client: RemoteSLURMClient) -> None:
    """Automatically run the full pipeline, blocking between phases."""
    print("\n" + "=" * 60)
    print("AUTO MODE (REMOTE)")
    print("=" * 60)
    print("This mode will block while waiting for remote jobs.")
    print("Press Ctrl-C to stop safely.\n")

    manifest = load_remote_manifest(config.build_dir)
    if not manifest["seed"].get("submitted"):
        run_seed_submit(config, client)

    try:
        while True:
            manifest = load_remote_manifest(config.build_dir)

            if manifest["termination"].get("stopped"):
                print("\nTermination condition met. Stopping auto mode.")
                break

            # Seed collection
            seed_jobs = manifest["seed"].get("jobs", [])
            uncollected_seed = [j for j in seed_jobs if j.get("status") != "collected"]
            if uncollected_seed:
                run_seed_collect(config, client)
                manifest = load_remote_manifest(config.build_dir)
                still_uncollected = [j for j in manifest["seed"]["jobs"] if j.get("status") != "collected"]
                if still_uncollected:
                    job_ids = [j["job_id"] for j in still_uncollected]
                    print(f"\nWaiting for {len(job_ids)} seed jobs to finish...")
                    client.wait_for_jobs(job_ids)
                    run_seed_collect(config, client)

            # AL collection (latest iteration)
            al_iters = manifest.get("al_iterations", [])
            if al_iters and not al_iters[-1].get("collected"):
                run_al_collect(config, client)
                manifest = load_remote_manifest(config.build_dir)
                al_iters = manifest.get("al_iterations", [])
                if al_iters and not al_iters[-1].get("collected"):
                    job_ids = []
                    for j in al_iters[-1]["jobs"]:
                        if j.get("status") != "collected":
                            if j.get("annealing_chain"):
                                job_ids.extend([p["job_id"] for p in j["phases"] if p.get("status") != "collected"])
                            else:
                                job_ids.append(j["job_id"])
                    if job_ids:
                        print(f"\nWaiting for {len(job_ids)} AL jobs to finish...")
                        client.wait_for_jobs(job_ids)
                        run_al_collect(config, client)
                continue

            if manifest["termination"].get("stopped"):
                print("\nTermination condition met. Stopping auto mode.")
                break

            # Submit next AL iteration
            run_al_submit(config, client)
            manifest = load_remote_manifest(config.build_dir)
            al_iters = manifest.get("al_iterations", [])
            if al_iters and al_iters[-1].get("jobs"):
                job_ids = []
                for j in al_iters[-1]["jobs"]:
                    if j.get("annealing_chain"):
                        job_ids.extend([p["job_id"] for p in j["phases"]])
                    else:
                        job_ids.append(j["job_id"])
                print(f"\nWaiting for {len(job_ids)} AL jobs to finish...")
                client.wait_for_jobs(job_ids)

    except KeyboardInterrupt:
        print("\n\nAuto mode interrupted by user. Safe to resume later.")


def run_status(config: BuilderConfig) -> None:
    """Print a read-only summary of the remote manifest."""
    build_dir = Path(config.build_dir)
    manifest = load_remote_manifest(build_dir)

    print(f"Remote manifest: {build_dir / REMOTE_MANIFEST_NAME}")
    print(f"Remote work dir: {manifest.get('remote_work_dir', 'N/A')}")

    seed = manifest.get("seed", {})
    print(f"\nSEED:")
    print(f"  Submitted: {seed.get('submitted', False)}")
    jobs = seed.get("jobs", [])
    status_counts: dict[str, int] = {}
    for j in jobs:
        s = j.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Jobs: {len(jobs)} total")
    for s, c in sorted(status_counts.items()):
        print(f"    {s}: {c}")

    al_iters = manifest.get("al_iterations", [])
    print(f"\nAL ITERATIONS: {len(al_iters)}")
    for it in al_iters:
        i = it["iteration"]
        collected = it.get("collected", False)
        jobs = it.get("jobs", [])
        mode = it.get("induction_mode", "md")
        n_chains = sum(1 for j in jobs if j.get("annealing_chain"))
        n_singles = len(jobs) - n_chains
        n_phases = 0
        for j in jobs:
            if j.get("annealing_chain"):
                n_phases += len(j.get("phases", []))
            else:
                n_phases += 1
        n_frames = it.get("n_new_frames", 0)
        metrics = it.get("metrics_after", {})
        ds = metrics.get("dataset_size", "?")
        chain_info = f", annealing_chains={n_chains}" if n_chains else ""
        print(f"  Iter {i}: collected={collected}, mode={mode}, jobs={len(jobs)}, phases={n_phases}{chain_info}, frames={n_frames}, dataset_size={ds}")

    term = manifest.get("termination", {})
    if term.get("stopped"):
        print(f"\nTERMINATED: {term.get('reason', 'unknown')}")
    else:
        print(f"\nStatus: running")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_config_from_args(args: argparse.Namespace) -> BuilderConfig:
    structure_types = tuple(s.strip() for s in args.structure_types.split(","))
    structure_weights = None
    if args.structure_weights:
        structure_weights = tuple(float(w) for w in args.structure_weights.split(","))

    induction_mode = args.al_induction_mode
    if args.al_use_annealing:
        induction_mode = "annealing"

    return BuilderConfig(
        build_dir=args.build_dir,
        db_name=args.db_name,
        n_atoms=args.n_atoms,
        structure_types=structure_types,
        structure_weights=structure_weights,
        target_size=args.target_size,
        seed_temperatures=tuple(float(t) for t in args.seed_temps.split(",")),
        seed_runs_per_temp=args.seed_runs_per_temp,
        seed_n_steps=args.seed_n_steps,
        al_max_iterations=args.al_max_iter,
        al_runs_per_iteration=args.al_runs_per_iter,
        al_candidate_pool_size=args.al_candidate_pool_size,
        use_force_field=not args.no_force_field,
        force_field_type=args.force_field_type,
        force_field_param_file=args.force_field_param_file,
        al_n_steps=args.al_n_steps,
        al_use_annealing=args.al_use_annealing,
        al_induction_mode=induction_mode,
        npt_pressures_gpa=tuple(float(v) for v in args.npt_pressures_gpa.split(",")),
        npt_temperatures=tuple(float(v) for v in args.npt_temperatures.split(",")),
        annealing_temps_high=tuple(float(v) for v in args.annealing_temps_high.split(",")),
        ballistic_velocity_temps=tuple(float(v) for v in args.ballistic_velocity_temps.split(",")),
        strain_axes=tuple(a.strip() for a in args.strain_axes.split(",")),
        strain_values=tuple(float(v) for v in args.strain_values.split(",")),
        al_annealing_relax_steps=args.al_annealing_relax_steps,
        al_annealing_heating_steps=args.al_annealing_heating_steps,
        al_annealing_heating_temp=args.al_annealing_heating_temp,
        al_annealing_peak_temp=args.al_annealing_peak_temp,
        al_annealing_high_temp_steps=args.al_annealing_high_temp_steps,
        al_annealing_quench_steps=args.al_annealing_quench_steps,
        al_annealing_hold_steps=args.al_annealing_hold_steps,
        cp2k_cutoff=args.cp2k_cutoff,
        cp2k_ppn=args.cp2k_ppn,
        cp2k_nprocs=args.cp2k_nprocs,
        slurm_partition=args.slurm_partition,
        cp2k_data_dir=args.cp2k_data_dir,
        remote_config_path=args.remote_config,
        candidate_strategy=args.candidate_strategy,
        event_driven_sampling=args.event_driven,
        event_window=args.event_window,
        sparse_stride=args.sparse_stride,
        dry_run=args.dry_run,
        skip_seed=args.skip_seed,
        skip_al=args.skip_al,
        verbose=args.verbose,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SiO2 dataset via Active Learning loop with REMOTE CP2K AIMD.",
    )
    parser.add_argument(
        "--remote-config",
        required=True,
        help="Path to remote server JSON config (host, username, password, remote_work_dir).",
    )
    parser.add_argument(
        "--build-dir",
        default="sio2_al_build",
        help="Root directory for build outputs.",
    )
    parser.add_argument(
        "--db-name",
        default="dataset.db",
        help="ASE database file name.",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=72,
        help="Target number of atoms per supercell.",
    )
    parser.add_argument(
        "--structure-types",
        default="alpha_quartz,beta_cristobalite",
        help="Comma-separated structure types for candidates.",
    )
    parser.add_argument(
        "--structure-weights",
        default="",
        help="Comma-separated weights for structure types. Empty = equal weights.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=50000,
        help="Target total number of frames.",
    )
    parser.add_argument(
        "--seed-temps",
        default="300,500,700,900,1000,1200,1500,2000,2500,3000",
        help="Comma-separated seed temperatures (K).",
    )
    parser.add_argument(
        "--seed-runs-per-temp",
        type=int,
        default=2,
        help="Number of independent runs per seed temperature.",
    )
    parser.add_argument(
        "--seed-n-steps",
        type=int,
        default=5000,
        help="MD steps for seed trajectories.",
    )
    parser.add_argument(
        "--al-max-iter",
        type=int,
        default=20,
        help="Max AL iterations.",
    )
    parser.add_argument(
        "--al-runs-per-iter",
        type=int,
        default=6,
        help="Number of CP2K runs per AL iteration.",
    )
    parser.add_argument(
        "--al-candidate-pool-size",
        type=int,
        default=50,
        help="Number of candidate structures generated per AL iteration.",
    )
    parser.add_argument(
        "--no-force-field",
        action="store_true",
        default=False,
        help="Disable fast force-field pre-screening and run CP2K single-point calculations for each candidate.",
    )
    parser.add_argument(
        "--force-field-type",
        default="morse",
        choices=["auto", "emt", "lj", "tersoff", "morse"],
        help="ASE built-in force field for candidate pre-screening (default: morse).",
    )
    parser.add_argument(
        "--force-field-param-file",
        default="",
        help="Path to a LAMMPS-style Tersoff potential file (required when --force-field-type=tersoff).",
    )
    parser.add_argument(
        "--al-n-steps",
        type=int,
        default=10000,
        help="MD steps for AL trajectories (single-phase mode).",
    )
    parser.add_argument(
        "--cp2k-cutoff",
        type=int,
        default=500,
        help="CP2K plane-wave cutoff (Ry).",
    )
    parser.add_argument(
        "--cp2k-ppn",
        type=int,
        default=64,
        help="Processors per node for SLURM (--ntasks-per-node=N).",
    )
    parser.add_argument(
        "--cp2k-nprocs",
        type=int,
        default=64,
        help="Number of MPI ranks for CP2K (srun uses allocation).",
    )
    parser.add_argument(
        "--slurm-partition",
        default="compute",
        help="SLURM partition name.",
    )
    parser.add_argument(
        "--cp2k-data-dir",
        default="/opt/software/cp2k/data",
        help="CP2K data directory (remote path).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without submitting jobs.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Skip seed phase (assume existing DB).",
    )
    parser.add_argument(
        "--skip-al",
        action="store_true",
        help="Skip AL loop (run seed only).",
    )
    parser.add_argument(
        "--candidate-strategy",
        default="random",
        choices=["random", "boundary", "composite",
                 "strain_uniaxial", "strain_biaxial",
                 "strain_shear", "strain_hydrostatic"],
        help="Candidate generation strategy.",
    )
    parser.add_argument(
        "--event-driven",
        action="store_true",
        default=False,
        help="Use event-driven TrajectoryMiner sampling.",
    )
    parser.add_argument(
        "--event-window",
        type=int,
        default=5,
        help="Frames before/after each detected event to sample densely.",
    )
    parser.add_argument(
        "--sparse-stride",
        type=int,
        default=20,
        help="Frame stride for non-event regions in event-driven mode.",
    )
    parser.add_argument(
        "--al-induction-mode",
        default="md",
        choices=["md", "annealing", "npt", "strain", "ballistic"],
        help="Induction mode for the AL loop: md (NVT), annealing (5-phase), npt (NPT_I pressure), strain (directional strain), ballistic (high-energy initial velocities).",
    )
    parser.add_argument(
        "--npt-pressures-gpa",
        default="0.0,10.0,20.0,40.0",
        help="Comma-separated pressure grid for NPT_I induction (GPa).",
    )
    parser.add_argument(
        "--npt-temperatures",
        default="300.0,1000.0,2000.0",
        help="Comma-separated temperature grid for NPT/strain induction (K).",
    )
    parser.add_argument(
        "--annealing-temps-high",
        default="2000.0,2500.0,3000.0,3500.0",
        help="Comma-separated high-temperature choices for annealing induction (K).",
    )
    parser.add_argument(
        "--ballistic-velocity-temps",
        default="1500,2000,2500",
        help="Comma-separated initial velocity temperatures for ballistic induction (K).",
    )
    parser.add_argument(
        "--strain-axes",
        default="x,y,z",
        help="Comma-separated strain axes for strain induction.",
    )
    parser.add_argument(
        "--strain-values",
        default="-0.10,-0.05,0.05,0.10,0.15",
        help=(
            "Comma-separated strain values (fractional) for strain induction. "
            "Use --strain-values=-0.10,-0.05,... (with '=') when values start with a minus sign, "
            "otherwise argparse treats them as additional options."
        ),
    )
    parser.add_argument(
        "--al-use-annealing",
        action="store_true",
        default=False,
        help="Enable 5-phase annealing (relax → heating → high_temp → quench → hold) for AL candidates. Alias for --al-induction-mode annealing.",
    )
    parser.add_argument(
        "--al-annealing-relax-steps",
        type=int,
        default=2000,
        help="MD steps for the 300 K relaxation phase (default: 2000).",
    )
    parser.add_argument(
        "--al-annealing-heating-steps",
        type=int,
        default=3000,
        help="MD steps for the heating phase (default: 3000).",
    )
    parser.add_argument(
        "--al-annealing-heating-temp",
        type=float,
        default=3000.0,
        help="Target temperature for the heating phase in K (default: 3000).",
    )
    parser.add_argument(
        "--al-annealing-peak-temp",
        type=float,
        default=3500.0,
        help="Peak temperature for high-temp phase (default: 3500 K).",
    )
    parser.add_argument(
        "--al-annealing-high-temp-steps",
        type=int,
        default=2000,
        help="MD steps for the high-temp phase (default: 2000).",
    )
    parser.add_argument(
        "--al-annealing-quench-steps",
        type=int,
        default=3000,
        help="MD steps for the quenching phase (default: 3000).",
    )
    parser.add_argument(
        "--al-annealing-hold-steps",
        type=int,
        default=5000,
        help="MD steps for the 300 K hold phase (default: 5000).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    subparsers.add_parser("seed-submit", help="Generate seed configs and submit remote SLURM jobs.")
    subparsers.add_parser("seed-collect", help="Collect finished seed results from remote.")
    subparsers.add_parser("al-submit", help="Run one AL iteration and submit remote SLURM jobs.")
    subparsers.add_parser("al-collect", help="Collect finished results for the latest AL iteration.")
    subparsers.add_parser("auto", help="Run the full pipeline automatically (blocks while waiting).")
    subparsers.add_parser("status", help="Print current job status from manifest.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "status":
        config = _build_config_from_args(args)
        run_status(config)
        return 0

    if not args.remote_config:
        print("ERROR: --remote-config is required for this command.")
        return 1

    remote_cfg = load_remote_config(args.remote_config)
    config = _build_config_from_args(args)

    manifest = load_remote_manifest(config.build_dir)
    manifest["remote_work_dir"] = remote_cfg.remote_work_dir
    save_remote_manifest(config.build_dir, manifest)

    if not config.dry_run:
        client = RemoteSLURMClient(remote_cfg)
    else:
        client = None  # type: ignore[assignment]

    if args.command == "seed-submit":
        run_seed_submit(config, client)
    elif args.command == "seed-collect":
        run_seed_collect(config, client)
    elif args.command == "al-submit":
        run_al_submit(config, client)
    elif args.command == "al-collect":
        run_al_collect(config, client)
    elif args.command == "auto":
        run_auto(config, client)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
