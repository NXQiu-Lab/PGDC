#!/usr/bin/env python3
"""
测试脚本：加载 high 和 low 两个 PaiNN 模型，在 test.db 上计算能量和力的 MAE。

用法:
    uv run python test_models_on_test_db.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from ase.db import connect
from schnetpack.interfaces import AtomsConverter
from schnetpack.transform import ASENeighborList
import schnetpack as spk
from schnetpack.representation.painn import PaiNN

# 路径配置
TEST_DB = "outputs/uniformity_ablation_strict/20260423-183745/test.db"
HIGH_MODEL_PATH = "moftest/2026_04_23_high_index_inter_8_F_64_painn_fleix/best_inference_model"
LOW_MODEL_PATH = "moftest/2026_04_25_low_index_inter_8_F_64_painn_fleix/best_inference_model"
CUTOFF = 5.0


def _fix_painn_electronic_embeddings(model):
    """递归修复旧版 PaiNN 模型缺少 electronic_embeddings 的问题。"""
    for module in model.modules():
        if isinstance(module, PaiNN) and not hasattr(module, "electronic_embeddings"):
            module.electronic_embeddings = nn.ModuleList([])


def load_model(model_path: str, device: str = "cpu"):
    """加载 schnetpack 模型并设为 eval 模式。"""
    print(f"加载模型: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    _fix_painn_electronic_embeddings(model)
    model.eval()
    model.to(device)
    return model


def evaluate_model(model, test_db_path: str, device: str = "cpu"):
    """
    在测试集上运行推理，返回能量和力的绝对平均误差。
    """
    converter = AtomsConverter(
        neighbor_list=ASENeighborList(cutoff=CUTOFF),
        device=device,
    )

    db = connect(test_db_path)
    energy_errors = []
    force_errors = []
    energy_per_atom_errors = []
    n_atoms_list = []

    total = len(list(db.select()))
    print(f"  测试集共 {total} 条记录，开始推理 ...")

    for idx, row in enumerate(db.select()):
        atoms = row.toatoms()
        inputs = converter(atoms)
        # Forces 计算需要 positions 的梯度
        inputs["_positions"].requires_grad_(True)

        with torch.set_grad_enabled(True):
            result = model(inputs)

        pred_energy = result[spk.properties.energy].detach().cpu().numpy().flatten()
        pred_forces = result[spk.properties.forces].detach().cpu().numpy()

        gt_energy = np.asarray(row.data["energy"]).flatten()
        gt_forces = np.asarray(row.data["forces"])

        n_atoms = len(atoms)
        n_atoms_list.append(n_atoms)

        # 能量绝对误差 (eV)
        e_err = np.abs(pred_energy - gt_energy).item()
        energy_errors.append(e_err)
        energy_per_atom_errors.append(e_err / n_atoms)

        # 力绝对误差 (eV/Å)，逐分量
        f_err = np.abs(pred_forces - gt_forces)
        force_errors.extend(f_err.flatten().tolist())

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"    已处理 {idx + 1}/{total} ...")

    results = {
        "n_structures": total,
        "n_atoms_total": sum(n_atoms_list),
        "energy_mae_eV": float(np.mean(energy_errors)),
        "energy_mae_std_eV": float(np.std(energy_errors)),
        "energy_mae_per_atom_meV": float(np.mean(energy_per_atom_errors) * 1000),
        "force_mae_eV_per_A": float(np.mean(force_errors)),
        "force_mae_std_eV_per_A": float(np.std(force_errors)),
    }
    return results


def print_results(model_name: str, results: dict):
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    print(f"测试结构数      : {results['n_structures']}")
    print(f"总原子数        : {results['n_atoms_total']}")
    print(f"能量 MAE        : {results['energy_mae_eV']:.6f} eV")
    print(f"能量 MAE (std)  : {results['energy_mae_std_eV']:.6f} eV")
    print(f"能量 MAE/atom   : {results['energy_mae_per_atom_meV']:.4f} meV/atom")
    print(f"力 MAE          : {results['force_mae_eV_per_A']:.6f} eV/Å")
    print(f"力 MAE (std)    : {results['force_mae_std_eV_per_A']:.6f} eV/Å")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 评估 high uniformity 模型
    high_model = load_model(HIGH_MODEL_PATH, device=device)
    high_results = evaluate_model(high_model, TEST_DB, device=device)
    print_results("high_index_inter_8_F_64_painn_fleix", high_results)

    # 评估 low uniformity 模型
    low_model = load_model(LOW_MODEL_PATH, device=device)
    low_results = evaluate_model(low_model, TEST_DB, device=device)
    print_results("low_index_inter_8_F_64_painn_fleix", low_results)

    # 汇总对比
    print(f"\n{'='*60}")
    print("对比汇总")
    print(f"{'='*60}")
    print(f"{'指标':<25} {'high_index_painn':>18} {'low_index_painn':>18}")
    print("-" * 65)
    print(
        f"{'能量 MAE (eV)':<25} "
        f"{high_results['energy_mae_eV']:>18.6f} "
        f"{low_results['energy_mae_eV']:>18.6f}"
    )
    print(
        f"{'能量 MAE/atom (meV)':<25} "
        f"{high_results['energy_mae_per_atom_meV']:>18.4f} "
        f"{low_results['energy_mae_per_atom_meV']:>18.4f}"
    )
    print(
        f"{'力 MAE (eV/Å)':<25} "
        f"{high_results['force_mae_eV_per_A']:>18.6f} "
        f"{low_results['force_mae_eV_per_A']:>18.6f}"
    )
    print()


if __name__ == "__main__":
    main()
