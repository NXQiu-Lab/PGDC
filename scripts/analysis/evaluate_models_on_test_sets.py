#!/usr/bin/env python3
"""Evaluate 1.1.2 ablation models on held-out test sets."""

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ase.db import connect
from schnetpack.interfaces import AtomsConverter
from schnetpack.transform import ASENeighborList
import schnetpack as spk
from schnetpack.representation.painn import PaiNN


CUTOFF = 5.0

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "training_runs 2"
TEST_DB_DIR = ROOT / "outputs" / "1_1_2_final_datasets"

CONFIGS = [
    ("cs_compositional_diversity_low_N10000", TEST_DB_DIR / "cs_test.db"),
    ("cs_compositional_diversity_mid_N10000", TEST_DB_DIR / "cs_test.db"),
    ("cs_compositional_diversity_high_N10000", TEST_DB_DIR / "cs_test.db"),
    ("sio2_random_10k", TEST_DB_DIR / "sio2_test.db"),
    ("sio2_adaptive_eq_preserve_10k", TEST_DB_DIR / "sio2_test.db"),
    ("sio2_structural_diversity_low_N5000", TEST_DB_DIR / "sio2_test.db"),
    ("sio2_structural_diversity_high_N5000", TEST_DB_DIR / "sio2_test.db"),
]


def _fix_painn_electronic_embeddings(model):
    for module in model.modules():
        if isinstance(module, PaiNN) and not hasattr(module, "electronic_embeddings"):
            module.electronic_embeddings = nn.ModuleList([])


def load_model(model_path: str, device: str):
    model = torch.load(model_path, map_location=device, weights_only=False)
    _fix_painn_electronic_embeddings(model)
    model.eval()
    model.to(device)
    return model


def evaluate_model(model, test_db_path: Path, device: str):
    converter = AtomsConverter(
        neighbor_list=ASENeighborList(cutoff=CUTOFF),
        device=device,
    )
    db = connect(str(test_db_path))
    energy_errors = []
    force_errors = []
    energy_per_atom_errors = []
    n_atoms_list = []

    total = len(list(db.select()))
    for idx, row in enumerate(db.select()):
        atoms = row.toatoms()
        inputs = converter(atoms)
        inputs["_positions"].requires_grad_(True)

        with torch.set_grad_enabled(True):
            result = model(inputs)

        pred_energy = result[spk.properties.energy].detach().cpu().numpy().flatten()
        pred_forces = result[spk.properties.forces].detach().cpu().numpy()

        gt_energy = np.asarray(row.data["energy"]).flatten()
        gt_forces = np.asarray(row.data["forces"])

        n_atoms = len(atoms)
        n_atoms_list.append(n_atoms)

        e_err = np.abs(pred_energy - gt_energy).item()
        energy_errors.append(e_err)
        energy_per_atom_errors.append(e_err / n_atoms)

        f_err = np.abs(pred_forces - gt_forces)
        force_errors.extend(f_err.flatten().tolist())

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  processed {idx + 1}/{total}")

    return {
        "n_structures": total,
        "n_atoms_total": int(sum(n_atoms_list)),
        "energy_mae_eV": float(np.mean(energy_errors)),
        "energy_mae_std_eV": float(np.std(energy_errors)),
        "energy_mae_per_atom_meV": float(np.mean(energy_per_atom_errors) * 1000),
        "force_mae_eV_per_A": float(np.mean(force_errors)),
        "force_mae_std_eV_per_A": float(np.std(force_errors)),
    }


def get_best_val_epoch(train_dir: Path):
    metrics_path = train_dir / "lightning_logs" / "version_0" / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = list(csv.DictReader(open(metrics_path)))
    best_epoch = None
    best_val = float("inf")
    for r in rows:
        if r.get("val_forces_MAE"):
            v = float(r["val_forces_MAE"])
            if v < best_val:
                best_val = v
                best_epoch = int(r["epoch"])
    return best_epoch, best_val


def main():
    device = "cpu"  # MPS scatter ops are unreliable; CPU is safe.
    print(f"Using device: {device}")

    output_dir = ROOT / "outputs" / "1_1_2_test_evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name, test_db_path in CONFIGS:
        train_dir = TRAIN_ROOT / model_name
        model_path = train_dir / "best_inference_model"
        if not model_path.exists():
            print(f"Skipping {model_name}: model not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Test DB: {test_db_path}")
        best_epoch, best_val = get_best_val_epoch(train_dir)
        print(f"Best val epoch: {best_epoch}, best val force MAE: {best_val:.6f}")

        model = load_model(str(model_path), device=device)
        res = evaluate_model(model, test_db_path, device=device)
        results[model_name] = res

        print(f"Test energy MAE: {res['energy_mae_eV']:.6f} eV")
        print(f"Test energy MAE/atom: {res['energy_mae_per_atom_meV']:.4f} meV/atom")
        print(f"Test force MAE: {res['force_mae_eV_per_A']:.6f} eV/Å")

    out_path = output_dir / "test_evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
