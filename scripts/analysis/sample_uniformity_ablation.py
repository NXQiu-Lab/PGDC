#!/usr/bin/env python3
"""Sample uniformity ablation subsets from an ASE DB dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect

# ASE 3.23+ made metadata require an open connection, breaking SchNetPack which
# uses connect() without a context manager. Patch the getter to use
# managed_connection() so it works with or without an explicit connection.
import ase.db.sqlite as _ase_sqlite


def _ase_metadata_compat_getter(self):
    if self._metadata is None:
        with self.managed_connection() as con:
            self._initialize(con)
    return self._metadata.copy()


_ase_sqlite.SQLite3Database.metadata = property(
    _ase_metadata_compat_getter,
    _ase_sqlite.SQLite3Database.metadata.fset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atom_dataset_index.domain import StructureRecord
from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.profiling import (
    CORE_METRIC_IDS,
    DatasetProfiler,
    PoolBuilder,
    build_default_registry,
)


def _sigmoid(value: float, mid: float, scale: float) -> float:
    if scale == 0:
        return 0.0 if value < mid else 1.0
    return 1.0 / (1.0 + math.exp(-(value - mid) / scale))


def _compute_uniformity(phase_counts: list[int]) -> float:
    total = sum(phase_counts)
    if total == 0 or len(phase_counts) <= 1:
        return 100.0
    entropy = 0.0
    for c in phase_counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    return 100.0 * entropy / math.log(len(phase_counts))


def _compute_coverage_score(total_size: int, phase_counts: list[int]) -> float:
    uniformity = _compute_uniformity(phase_counts)
    scale_score = 100.0 * _sigmoid(float(total_size), mid=20000.0, scale=10000.0)
    return scale_score * (uniformity / 100.0)


def _stratified_test_split(
    pool_to_phase: dict[str, int],
    pools: list,
    test_fraction: float,
    seed: int,
) -> tuple[set[int], dict[int, list[StructureRecord]]]:
    """Return test sample_ids and available records per phase after removing test set."""
    rng = random.Random(seed)

    phase_to_records: dict[int, list[StructureRecord]] = {
        i: [] for i in set(pool_to_phase.values())
    }
    for pool in pools:
        phase_idx = pool_to_phase[pool.pool_id]
        phase_to_records[phase_idx].extend(pool.records)

    for recs in phase_to_records.values():
        rng.shuffle(recs)

    phase_ids = sorted(phase_to_records.keys())
    phase_sizes = [len(phase_to_records[i]) for i in phase_ids]
    total = sum(phase_sizes)
    target_test = max(1, int(total * test_fraction))

    # Ensure at least 1 sample per phase remains for training
    max_test = [max(0, s - 1) for s in phase_sizes]
    allocations = [min(int(s * test_fraction), max_test[i]) for i, s in enumerate(phase_sizes)]
    remaining = target_test - sum(allocations)

    if remaining > 0:
        fractional = sorted(
            [
                (phase_sizes[i] * test_fraction - int(phase_sizes[i] * test_fraction), i)
                for i in range(len(phase_ids))
            ],
            reverse=True,
        )
        for _, idx in fractional:
            if remaining <= 0:
                break
            can_give = max_test[idx] - allocations[idx]
            if can_give > 0:
                give = min(can_give, remaining)
                allocations[idx] += give
                remaining -= give

    if remaining > 0:
        for idx in sorted(range(len(phase_ids)), key=lambda j: phase_sizes[j], reverse=True):
            if remaining <= 0:
                break
            can_give = phase_sizes[idx] - allocations[idx]
            if can_give > 0:
                give = min(can_give, remaining)
                allocations[idx] += give
                remaining -= give

    available_records: dict[int, list[StructureRecord]] = {}
    test_ids: set[int] = set()
    for i, pid in enumerate(phase_ids):
        recs = phase_to_records[pid]
        n_test = allocations[i]
        test_ids.update(r.sample_id for r in recs[:n_test])
        available_records[pid] = recs[n_test:]

    return test_ids, available_records


def _build_high_uniformity_subset(
    available_records: dict[int, list[StructureRecord]],
    target_train_size: int,
    seed: int,
) -> tuple[list[StructureRecord], float]:
    rng = random.Random(seed)
    phase_ids = sorted(available_records.keys())
    phase_shuffled: dict[int, list[StructureRecord]] = {}
    for pid in phase_ids:
        recs = list(available_records[pid])
        rng.shuffle(recs)
        phase_shuffled[pid] = recs

    available = [len(phase_shuffled[pid]) for pid in phase_ids]
    n_phases = len(phase_ids)
    base = target_train_size // n_phases
    allocated = [min(base, a) for a in available]
    remaining = target_train_size - sum(allocated)

    while remaining > 0:
        deficits = [
            (available[i] - allocated[i], i)
            for i in range(n_phases)
            if available[i] > allocated[i]
        ]
        if not deficits:
            break
        deficits.sort(reverse=True, key=lambda x: x[0])
        _, idx = deficits[0]
        allocated[idx] += 1
        remaining -= 1

    subset: list[StructureRecord] = []
    phase_counts: list[int] = []
    for i, pid in enumerate(phase_ids):
        n = allocated[i]
        chosen = phase_shuffled[pid][:n]
        subset.extend(chosen)
        phase_counts.append(len(chosen))

    return subset, _compute_uniformity(phase_counts), phase_counts


def _build_low_uniformity_subset(
    available_records: dict[int, list[StructureRecord]],
    target_train_size: int,
    alpha: float,
    seed: int,
) -> tuple[list[StructureRecord], float, list[int]]:
    rng = random.Random(seed)
    phase_ids = sorted(available_records.keys())
    phase_shuffled: dict[int, list[StructureRecord]] = {}
    for pid in phase_ids:
        recs = list(available_records[pid])
        rng.shuffle(recs)
        phase_shuffled[pid] = recs

    available = [len(phase_shuffled[pid]) for pid in phase_ids]
    weights = [float(a) ** alpha for a in available]
    total_w = sum(weights)

    allocated: list[int] = []
    for i, a in enumerate(available):
        t = max(1, int(target_train_size * weights[i] / total_w))
        allocated.append(min(t, a))

    remaining = target_train_size - sum(allocated)
    if remaining > 0:
        while remaining > 0:
            deficits = [
                (available[i] - allocated[i], i)
                for i in range(len(phase_ids))
                if available[i] > allocated[i]
            ]
            if not deficits:
                break
            deficits.sort(reverse=True, key=lambda x: x[0])
            _, idx = deficits[0]
            allocated[idx] += 1
            remaining -= 1
    elif remaining < 0:
        deficit = -remaining
        while deficit > 0:
            surpluses = [
                (allocated[i] - 1, i)
                for i in range(len(phase_ids))
                if allocated[i] > 1
            ]
            if not surpluses:
                break
            surpluses.sort(reverse=True, key=lambda x: x[0])
            _, idx = surpluses[0]
            allocated[idx] -= 1
            deficit -= 1

    subset: list[StructureRecord] = []
    phase_counts: list[int] = []
    for i, pid in enumerate(phase_ids):
        n = min(allocated[i], available[i])
        chosen = phase_shuffled[pid][:n]
        subset.extend(chosen)
        phase_counts.append(len(chosen))

    return subset, _compute_uniformity(phase_counts), phase_counts


def _write_ase_db(
    records: list[StructureRecord],
    output_path: str,
    energy_key: str = "energy",
    force_key: str = "forces",
    distance_unit: str = "Angstrom",
    property_unit_dict: dict[str, str] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    property_unit_dict = property_unit_dict or {
        energy_key: "eV",
        force_key: "eV/Angstrom",
    }

    with connect(str(output), append=False) as db:
        for rec in records:
            cell = list(rec.cell) if rec.cell is not None else None
            atoms = Atoms(
                numbers=list(rec.atomic_numbers),
                positions=list(rec.positions),
                cell=cell,
                pbc=list(rec.pbc),
            )
            data: dict[str, object] = {}
            if rec.energy is not None:
                data[energy_key] = np.array([rec.energy], dtype=np.float64)
            if rec.forces:
                data[force_key] = np.asarray(rec.forces, dtype=np.float64)
            db.write(atoms, data=data)

        db.metadata = {
            "_distance_unit": distance_unit,
            "_property_unit_dict": property_unit_dict,
            "atomrefs": {},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample uniformity ablation subsets.")
    parser.add_argument("--datapath", required=True, help="Path to input ASE DB file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--target-train-size", type=int, default=10000, help="Target training set size per subset"
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.20, help="Fraction reserved for test set"
    )
    parser.add_argument(
        "--low-uniformity-target", type=float, default=80.0, help="Target uniformity for low subset"
    )
    parser.add_argument(
        "--high-uniformity-target", type=float, default=90.0, help="Target uniformity for high subset"
    )
    parser.add_argument(
        "--low-alpha", type=float, default=3.0, help="Power-law exponent for low-uniformity sampling"
    )
    parser.add_argument(
        "--adaptive-alpha", action="store_true", help="Auto-increase low-alpha until low_uniformity is sufficiently below high_uniformity"
    )
    parser.add_argument(
        "--min-uniformity-gap", type=float, default=5.0, help="Minimum desired gap between high and low uniformity"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--energy-key", default="energy", help="ASE DB energy field name")
    parser.add_argument("--force-key", default="forces", help="ASE DB force field name")
    parser.add_argument(
        "--distance-unit", default="Angstrom", help="Distance unit for SchNetPack (default: Angstrom)"
    )
    parser.add_argument(
        "--property-units",
        default=None,
        help='Property units as JSON dict for SchNetPack, e.g. \'{"energy":"eV","forces":"eV/Angstrom"}\' (default: inferred from energy_key and force_key)',
    )
    args = parser.parse_args()

    property_unit_dict: dict[str, str] | None = None
    if args.property_units is not None:
        property_unit_dict = json.loads(args.property_units)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/uniformity_ablation/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载数据集: {args.datapath} ...")
    dataset = AseDbDatasetReader(args.datapath).read()
    original_size = len(dataset.records)
    print(f"原始数据集大小: {original_size}")

    print("正在分池 ...")
    pool_builder = PoolBuilder()
    pools = pool_builder.build_pools(dataset)
    print(f"共分出 {len(pools)} 个池。")

    print("正在运行 profiling（仅核心指标）以获取有效相 ...")
    profiler = DatasetProfiler(registry=build_default_registry(CORE_METRIC_IDS))
    phase_count_effective, phase_record_counts, pool_to_phase = profiler._compute_effective_phases(
        pools
    )
    print(f"有效相数量: {phase_count_effective}")

    print("正在构建分层测试集 ...")
    test_ids, available_records = _stratified_test_split(
        pool_to_phase, pools, args.test_fraction, args.seed
    )
    print(f"测试集大小: {len(test_ids)}")

    target_train_size = min(args.target_train_size, original_size - len(test_ids))
    print(f"每个训练子集的目标大小: {target_train_size}")

    print("正在构建高均匀性子集 ...")
    high_records, high_uniformity, high_phase_counts = _build_high_uniformity_subset(
        available_records, target_train_size, args.seed
    )
    high_coverage = _compute_coverage_score(len(high_records), high_phase_counts)
    print(
        f"高均匀性子集: 大小={len(high_records)}, "
        f"uniformity={high_uniformity:.2f}, coverage={high_coverage:.2f}"
    )

    print("正在构建低均匀性子集 ...")
    low_alpha = args.low_alpha
    low_records, low_uniformity, low_phase_counts = _build_low_uniformity_subset(
        available_records, target_train_size, low_alpha, args.seed + 1
    )
    low_coverage = _compute_coverage_score(len(low_records), low_phase_counts)
    print(
        f"低均匀性子集: 大小={len(low_records)}, "
        f"uniformity={low_uniformity:.2f}, coverage={low_coverage:.2f}"
    )

    if args.adaptive_alpha:
        max_attempts = 10
        for attempt in range(max_attempts):
            gap = high_uniformity - low_uniformity
            if gap >= args.min_uniformity_gap:
                break
            low_alpha += 1.0
            print(f"  自适应尝试 {attempt + 1}: 使用 alpha={low_alpha:.1f} 重试 ...")
            low_records, low_uniformity, low_phase_counts = _build_low_uniformity_subset(
                available_records, target_train_size, low_alpha, args.seed + 1
            )
            low_coverage = _compute_coverage_score(len(low_records), low_phase_counts)
            print(
                f"  -> uniformity={low_uniformity:.2f}, "
                f"coverage={low_coverage:.2f} (差距={high_uniformity - low_uniformity:.2f})"
            )
        else:
            print(
                f"警告: 经过 {max_attempts} 次尝试仍未能达到目标均匀性差距。"
                f"最终差距={high_uniformity - low_uniformity:.2f}。"
                f"该数据集的有效相分布可能本身就很均匀，难以拉开大的均匀性差异。"
            )

    if abs(high_uniformity - args.high_uniformity_target) > 5.0:
        print(
            f"警告: 高均匀性得分 ({high_uniformity:.2f}) 与目标值 "
            f"({args.high_uniformity_target}) 偏差超过 5 分。"
        )
    if low_uniformity >= high_uniformity - 2.0:
        print(
            f"警告: 低均匀性得分 ({low_uniformity:.2f}) 过于接近高均匀性得分 "
            f"({high_uniformity:.2f})。建议增大 --low-alpha 或使用 --adaptive-alpha。"
        )

    test_path = output_dir / "test.db"
    high_path = output_dir / "train_high_uniformity.db"
    low_path = output_dir / "train_low_uniformity.db"

    print("正在写入 ASE DB 文件 ...")
    _write_ase_db(
        [r for r in dataset.records if r.sample_id in test_ids],
        str(test_path),
        args.energy_key,
        args.force_key,
        args.distance_unit,
        property_unit_dict,
    )
    _write_ase_db(
        high_records, str(high_path), args.energy_key, args.force_key, args.distance_unit, property_unit_dict
    )
    _write_ase_db(
        low_records, str(low_path), args.energy_key, args.force_key, args.distance_unit, property_unit_dict
    )

    manifest = {
        "source_datapath": str(Path(args.datapath).resolve()),
        "seed": args.seed,
        "original_size": original_size,
        "test_size": len(test_ids),
        "target_train_size": target_train_size,
        "phase_count_effective": phase_count_effective,
        "subsets": {
            "high": {
                "datapath": str(high_path.name),
                "actual_size": len(high_records),
                "uniformity_score": round(high_uniformity, 2),
                "coverage_score": round(high_coverage, 2),
            },
            "low": {
                "datapath": str(low_path.name),
                "actual_size": len(low_records),
                "uniformity_score": round(low_uniformity, 2),
                "coverage_score": round(low_coverage, 2),
            },
        },
    }
    manifest_path = output_dir / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"清单文件已写入: {manifest_path}")
    print(f"完成。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
