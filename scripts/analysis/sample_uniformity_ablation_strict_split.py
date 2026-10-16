#!/usr/bin/env python3
"""生成 uniform-phase 测试集 + 统一加权采样框架的 uniformity ablation 子集。

与 sample_uniformity_ablation.py 的核心差异：
1. 测试集改用均匀相分布（每相取固定数量），而非按比例分层抽样。
   原脚本按原始比例抽 → 测试集天然偏向高频相，low 训练集也偏向高频相 →
   对比不公平。本脚本让测试集每相样本数相同，成为中立裁判。
2. high/low 训练子集改用同一套 _build_weighted_subset 框架，仅权重向量不同。
   原脚本是两套完全不同的分配算法（均分补余 vs 幂律权重），可能引入算法偏差。
   本脚本统一底层逻辑，差异仅体现在 weights 参数上：
   - high: weights = [1.0, 1.0, ..., 1.0]（等权重，均匀分布）
   - low: weights = [available^alpha]（幂律权重，偏向高频相）
"""

from __future__ import annotations

import argparse
import json
import math
import random
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


def _write_ase_db(
    records: list,
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


def _uniform_phase_test_split(
    pool_to_phase: dict[str, int],
    pools: list,
    samples_per_phase: int,
    seed: int,
    min_train_per_phase: int = 1,
):
    """均匀相分布测试集分割。

    与按比例分层抽样不同，本函数从每个相中抽取固定数量（samples_per_phase）
    的样本作为测试集。这样测试集在所有相上的分布是均匀的，不会天然偏向
    高频相，从而避免对 low-uniformity 训练集的不公平优势。

    参数:
        pool_to_phase: 池ID到相索引的映射，由 profiling._compute_effective_phases 生成。
        pools: PoolBuilder 分出的所有池列表。
        samples_per_phase: 每个相要抽取的测试样本数。所有相取相同值。
        seed: 随机种子，保证可复现。
        min_train_per_phase: 每个相至少保留多少个样本给训练（默认1个）。
            若某相总样本数 <= min_train_per_phase，则该相不抽取测试样本。

    返回:
        test_ids: 测试集样本ID集合。
        available_records: 扣除测试集后，每个相剩余可用于训练的样本。
        phase_ids: 相索引列表（已排序）。
        sizes: 每个相的原始样本总数。
        allocations: 每个相实际抽取的测试样本数（通常等于 samples_per_phase，
            但小相可能更少）。
    """
    rng = random.Random(seed)

    # 按相索引把池中的记录汇总到各相
    phase_to_records: dict[int, list] = {i: [] for i in set(pool_to_phase.values())}
    for pool in pools:
        phase_idx = pool_to_phase[pool.pool_id]
        phase_to_records[phase_idx].extend(pool.records)

    # 每相内部随机打乱，保证抽样随机性
    for recs in phase_to_records.values():
        rng.shuffle(recs)

    phase_ids = sorted(phase_to_records.keys())
    sizes = [len(phase_to_records[pid]) for pid in phase_ids]

    test_ids: set[int] = set()
    available_records: dict[int, list] = {}
    allocations: list[int] = []

    for pid in phase_ids:
        recs = phase_to_records[pid]
        # 确保扣除测试集后，该相至少还剩 min_train_per_phase 个样本给训练
        max_test = max(0, len(recs) - min_train_per_phase)
        # 实际抽取数 = min(目标数, 该相可承受的最大数)
        n_test = min(samples_per_phase, max_test)
        test_ids.update(r.sample_id for r in recs[:n_test])
        available_records[pid] = recs[n_test:]
        allocations.append(n_test)

    return test_ids, available_records, phase_ids, sizes, allocations


def _build_weighted_subset(
    available_records: dict[int, list],
    target_train_size: int,
    weights: list[float],
    seed: int,
) -> tuple[list, float, list[int]]:
    """统一加权采样框架，用于构建 high/low 均匀性子集。

    本函数是 high 和 low 子集的通用构建器。两者的差异 ONLY 在于传入的
    weights（权重向量）不同：
    - high uniformity: 所有相权重相等 → 各相分配到的样本数趋于一致
    - low uniformity: 权重按 available^alpha 计算 → 高频相获得更多样本，
      低频相获得更少样本，整体均匀性下降。

    分配流程：
    1. 按权重比例计算每相的初始目标样本数，但不超过该相可用样本数。
    2. 若总分配数不足 target_train_size，从还有容量的相中继续补充，
       优先补充权重高的相（对 low 来说就是高频相）。
    3. 若总分配数超额，从权重低的相中回收（对 low 来说就是低频相）。

    参数:
        available_records: 扣除测试集后，每个相剩余的可用于训练的样本。
        target_train_size: 目标训练子集总大小。
        weights: 每个相的权重列表，长度等于相数。high 用等权重，low 用
            幂律权重。
        seed: 随机种子。

    返回:
        subset: 选中的样本列表。
        uniformity: 该子集的均匀性得分（熵归一化，0-100）。
        phase_counts: 每个相实际选中的样本数列表。
    """
    rng = random.Random(seed)
    phase_ids = sorted(available_records.keys())
    n_phases = len(phase_ids)

    # 每相内部随机打乱
    phase_shuffled: dict[int, list] = {}
    for pid in phase_ids:
        recs = list(available_records[pid])
        rng.shuffle(recs)
        phase_shuffled[pid] = recs

    # 每相可用的样本总数
    available = [len(phase_shuffled[pid]) for pid in phase_ids]
    total_weight = sum(weights)

    # 初始分配：按权重比例，但不超过可用数，且至少分配1个（如果该相有样本）
    allocated: list[int] = []
    for i in range(n_phases):
        if total_weight > 0 and available[i] > 0:
            target = int(target_train_size * weights[i] / total_weight)
            allocated.append(min(max(1, target), available[i]))
        else:
            allocated.append(0)

    remaining = target_train_size - sum(allocated)

    # 若剩余 > 0：还有配额没用完，从还有容量的相中继续分配
    # 分配优先级：按权重从高到低（对 low 优先给高频相）
    if remaining > 0:
        while remaining > 0:
            deficits = [
                (available[i] - allocated[i], i)
                for i in range(n_phases)
                if available[i] > allocated[i]
            ]
            if not deficits:
                break
            deficits.sort(reverse=True, key=lambda x: weights[x[1]])
            _, idx = deficits[0]
            allocated[idx] += 1
            remaining -= 1
    elif remaining < 0:
        # 若剩余 < 0：超配额了，从权重低的相中回收样本
        deficit = -remaining
        while deficit > 0:
            surpluses = [
                (allocated[i], i)
                for i in range(n_phases)
                if allocated[i] > 1
            ]
            if not surpluses:
                break
            surpluses.sort(key=lambda x: weights[x[1]])
            _, idx = surpluses[0]
            allocated[idx] -= 1
            deficit -= 1

    # 根据最终分配数，从每相的打乱列表中取出样本
    subset: list = []
    phase_counts: list[int] = []
    for i, pid in enumerate(phase_ids):
        n = min(allocated[i], available[i])
        chosen = phase_shuffled[pid][:n]
        subset.extend(chosen)
        phase_counts.append(len(chosen))

    return subset, _compute_uniformity(phase_counts), phase_counts


def main() -> None:
    """主函数：解析参数、加载数据、生成均匀测试集和 high/low 训练子集。"""
    parser = argparse.ArgumentParser(
        description="生成 uniformity ablation 子集（均匀相分布测试集 + 统一加权采样框架）"
    )
    parser.add_argument("--datapath", required=True, help="输入 ASE DB 文件路径")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认为 outputs/uniformity_ablation_strict/时间戳")
    parser.add_argument(
        "--target-train-size",
        type=int,
        default=10000,
        help="每个训练子集的目标大小（high 和 low 各自的大小，默认10000）",
    )
    parser.add_argument(
        "--test-samples-per-phase",
        type=int,
        default=20,
        help="测试集中每个相抽取的样本数（默认20）。所有相取相同值，"
             "确保测试集分布均匀、不偏向高频相。设为0则跳过测试集生成。",
    )
    parser.add_argument(
        "--low-alpha",
        type=float,
        default=3.0,
        help="低均匀性子集的幂律指数（默认3.0）。越大则 low 子集越偏向高频相，"
             "uniformity 越低。若 high/low uniformity 差距不足，可增大此值。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证结果可复现")
    parser.add_argument("--energy-key", default="energy", help="ASE DB 中能量字段名")
    parser.add_argument("--force-key", default="forces", help="ASE DB 中力字段名")
    parser.add_argument("--distance-unit", default="Angstrom", help="距离单位，用于 SchNetPack")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = Path(args.datapath).stem
    output_dir = Path(
        args.output_dir
        or f"outputs/uniformity_ablation_strict/"
        f"{dataset_name}_N{args.target_train_size}_T{args.test_samples_per_phase}_S{args.seed}_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载数据: {args.datapath}")
    dataset = AseDbDatasetReader(args.datapath).read()
    original_size = len(dataset.records)
    print(f"原始数据量: {original_size}")

    # 分池 + 使用实际池数作为相（避免 _compute_effective_phases 把化学相似的池合并为单一相）
    pool_builder = PoolBuilder()
    pools = pool_builder.build_pools(dataset)
    # 每个池视为一个独立的相，确保均匀性计算反映真实的池分布差异
    pool_to_phase = {pool.pool_id: i for i, pool in enumerate(pools)}
    phase_count_effective = len(pools)
    print(f"实际池数量（作为相）: {phase_count_effective}")

    # --- 均匀相分布测试集分割 ---
    test_ids, available_records, phase_ids, sizes, allocations = _uniform_phase_test_split(
        pool_to_phase=pool_to_phase,
        pools=pools,
        samples_per_phase=args.test_samples_per_phase,
        seed=args.seed,
        min_train_per_phase=1,
    )
    print(f"测试集大小: {len(test_ids)}")
    print(f"每相测试样本数: min={min(allocations)}, max={max(allocations)}")
    print(f"每相训练保留: min={min(sizes[i] - allocations[i] for i in range(len(sizes)))}")

    # 训练子集实际大小 = 目标值与扣除测试集后剩余总量中的较小值
    target_train_size = min(args.target_train_size, original_size - len(test_ids))
    print(f"训练子集目标大小: {target_train_size}")

    # --- 高均匀性子集：等权重 ---
    n_phases = len(available_records)
    weights_high = [1.0] * n_phases
    high_records, high_uniformity, high_phase_counts = _build_weighted_subset(
        available_records, target_train_size, weights_high, args.seed
    )
    high_coverage = _compute_coverage_score(len(high_records), high_phase_counts)
    print(
        f"高均匀性子集: 大小={len(high_records)}, "
        f"uniformity={high_uniformity:.2f}, coverage={high_coverage:.2f}"
    )

    # --- 低均匀性子集：幂律权重（偏向高频相） ---
    # 计算每相可用样本数，作为幂律权重的基础
    available_counts = [len(available_records[pid]) for pid in sorted(available_records.keys())]
    weights_low = [float(a) ** args.low_alpha for a in available_counts]
    low_records, low_uniformity, low_phase_counts = _build_weighted_subset(
        available_records, target_train_size, weights_low, args.seed + 1
    )
    low_coverage = _compute_coverage_score(len(low_records), low_phase_counts)
    print(
        f"低均匀性子集: 大小={len(low_records)}, "
        f"uniformity={low_uniformity:.2f}, coverage={low_coverage:.2f}"
    )

    # 如果差距太小，提示用户增大 alpha
    if low_uniformity >= high_uniformity - 2.0:
        print(
            f"警告: 低均匀性得分 ({low_uniformity:.2f}) 过于接近高均匀性得分 "
            f"({high_uniformity:.2f})。建议增大 --low-alpha。"
        )

    # 写入 ASE DB 文件
    test_path = output_dir / "test.db"
    high_path = output_dir / "train_high_uniformity.db"
    low_path = output_dir / "train_low_uniformity.db"

    _write_ase_db([r for r in dataset.records if r.sample_id in test_ids], str(test_path))
    _write_ase_db(high_records, str(high_path))
    _write_ase_db(low_records, str(low_path))

    # 生成清单文件
    manifest = {
        "source_datapath": str(Path(args.datapath).resolve()),
        "seed": args.seed,
        "original_size": original_size,
        "test_size": len(test_ids),
        "target_train_size": target_train_size,
        "phase_count_effective": phase_count_effective,
        "uniform_split": {
            "test_samples_per_phase": args.test_samples_per_phase,
            "min_train_per_phase": 1,
            "phase_ids": phase_ids,
            "phase_sizes": sizes,
            "phase_test_allocations": allocations,
        },
        "subsets": {
            "high": {
                "datapath": high_path.name,
                "actual_size": len(high_records),
                "uniformity_score": round(high_uniformity, 2),
                "coverage_score": round(high_coverage, 2),
                "weight_strategy": "equal",
            },
            "low": {
                "datapath": low_path.name,
                "actual_size": len(low_records),
                "uniformity_score": round(low_uniformity, 2),
                "coverage_score": round(low_coverage, 2),
                "weight_strategy": f"power_law_alpha_{args.low_alpha}",
            },
        },
    }

    manifest_path = output_dir / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"完成，输出目录: {output_dir}")
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()
