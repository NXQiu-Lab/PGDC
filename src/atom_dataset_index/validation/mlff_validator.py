"""MLFF validation framework for closed-loop hypothesis testing.

Provides model loading, per-class error evaluation, and statistical comparison
between structure classes (EQ vs TP). Designed to validate the core hypothesis
that Transition-Prone (TP) structures are the hardest for MLFF to learn.

module_id: validation.mlff_validator
module_version: 1.0.0
project_version: 1.1.0
last_updated: 2026-05-19
generation_mode: manual
source_files: [mlff_validator.py]
upstream_modules: [domain, sampling.prefilter, io]
downstream_modules: [scripts.analysis]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClassEvaluationResult:
    """Evaluation results for a single structure class."""

    class_label: str
    n_structures: int = 0
    n_atoms_total: int = 0
    energy_mae_eV: float = 0.0
    energy_mae_std_eV: float = 0.0
    energy_mae_per_atom_meV: float = 0.0
    force_mae_eV_per_A: float = 0.0
    force_mae_std_eV_per_A: float = 0.0
    force_max_error_eV_per_A: float = 0.0
    per_structure_errors: list[dict[str, float]] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Statistical comparison between two structure classes."""

    class_a: str
    class_b: str
    energy_mae_ratio: float = 0.0
    force_mae_ratio: float = 0.0
    energy_pvalue: float | None = None
    force_pvalue: float | None = None
    conclusion: str = ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_schnetpack_model(model_path: str | Path, device: str = "cpu") -> Any:
    """Load a SchNetPack / PaiNN model from disk.

    Handles the common ``electronic_embeddings`` compatibility issue
    for older PaiNN checkpoints.

    Args:
        model_path: Path to saved torch model.
        device: "cpu" or "cuda".

    Returns:
        Loaded torch model in eval mode.
    """
    import torch
    import torch.nn as nn
    from schnetpack.representation.painn import PaiNN

    model = torch.load(
        str(model_path),
        map_location=device,
        weights_only=False,
    )

    # Fix missing electronic_embeddings in older PaiNN models
    for module in model.modules():
        if isinstance(module, PaiNN) and not hasattr(module, "electronic_embeddings"):
            module.electronic_embeddings = nn.ModuleList([])

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Per-structure evaluation
# ---------------------------------------------------------------------------


def evaluate_on_db(
    model: Any,
    db_path: str | Path,
    device: str = "cpu",
    cutoff: float = 5.0,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate a model on all structures in an ASE DB.

    Args:
        model: Loaded torch model.
        db_path: Path to ASE database (.db).
        device: "cpu" or "cuda".
        cutoff: Neighbor list cutoff (Å).
        batch_size: Batch size for inference (currently only 1 supported).

    Returns:
        List of per-structure result dicts with keys:
        ``idx``, ``n_atoms``, ``energy_error``, ``energy_per_atom_error``,
        ``force_errors``, ``force_mae``, ``force_max_error``.
    """
    from ase.db import connect
    from schnetpack.interfaces import AtomsConverter
    from schnetpack.transform import ASENeighborList
    import schnetpack as spk
    import torch

    converter = AtomsConverter(
        neighbor_list=ASENeighborList(cutoff=cutoff),
        device=device,
    )

    db = connect(str(db_path))
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(db.select()):
        atoms = row.toatoms()
        inputs = converter(atoms)
        inputs["_positions"].requires_grad_(True)

        with torch.set_grad_enabled(True):
            pred = model(inputs)

        pred_energy = pred[spk.properties.energy].detach().cpu().numpy().flatten()
        pred_forces = pred[spk.properties.forces].detach().cpu().numpy()

        # Ground truth
        gt_energy = np.asarray(row.data.get("energy", row.energy)).flatten()
        gt_forces = np.asarray(row.data.get("forces", row.get("forces")))

        n_atoms = len(atoms)
        e_err = float(np.abs(pred_energy - gt_energy).item())
        f_err = np.abs(pred_forces - gt_forces)
        f_mae = float(np.mean(f_err))
        f_max = float(np.max(f_err))

        results.append({
            "idx": idx,
            "db_id": row.id,
            "n_atoms": n_atoms,
            "energy_error": e_err,
            "energy_per_atom_error": e_err / max(n_atoms, 1),
            "force_errors": f_err.flatten().tolist(),
            "force_mae": f_mae,
            "force_max_error": f_max,
        })

    return results


# ---------------------------------------------------------------------------
# Dataset splitting by PreFilter class
# ---------------------------------------------------------------------------


def split_db_by_class(
    db_path: str | Path,
    prefilter: "DeterministicPreFilter",
) -> dict[str, list[int]]:
    """Split ASE DB structures by DeterministicPreFilter class labels.

    Args:
        db_path: Path to ASE database.
        prefilter: DeterministicPreFilter instance.

    Returns:
        Dict mapping class label → list of DB row indices (0-based).
    """
    from ase.db import connect
    from atom_dataset_index.sampling.utils import ase_to_record

    db_path = Path(db_path)
    classes: dict[str, list[int]] = {}

    with connect(str(db_path)) as db:
        for idx, row in enumerate(db.select()):
            atoms = row.toatoms()
            record = ase_to_record(atoms, sample_id=idx)
            result = prefilter.classify(record)
            label = result.label
            if label not in classes:
                classes[label] = []
            classes[label].append(idx)

    return classes


# ---------------------------------------------------------------------------
# Aggregation and comparison
# ---------------------------------------------------------------------------


def aggregate_results(
    per_structure_results: list[dict[str, Any]],
    class_label: str,
) -> ClassEvaluationResult:
    """Aggregate per-structure errors into class-level statistics."""
    if not per_structure_results:
        return ClassEvaluationResult(class_label=class_label)

    energy_errors = [r["energy_error"] for r in per_structure_results]
    energy_per_atom = [r["energy_per_atom_error"] for r in per_structure_results]
    force_maes = [r["force_mae"] for r in per_structure_results]
    force_maxes = [r["force_max_error"] for r in per_structure_results]
    n_atoms_total = sum(r["n_atoms"] for r in per_structure_results)

    return ClassEvaluationResult(
        class_label=class_label,
        n_structures=len(per_structure_results),
        n_atoms_total=n_atoms_total,
        energy_mae_eV=float(np.mean(energy_errors)),
        energy_mae_std_eV=float(np.std(energy_errors)),
        energy_mae_per_atom_meV=float(np.mean(energy_per_atom) * 1000),
        force_mae_eV_per_A=float(np.mean(force_maes)),
        force_mae_std_eV_per_A=float(np.std(force_maes)),
        force_max_error_eV_per_A=float(np.max(force_maxes)),
        per_structure_errors=[
            {"idx": r["idx"], "energy_error": r["energy_error"], "force_mae": r["force_mae"]}
            for r in per_structure_results
        ],
    )


def compare_class_errors(
    result_a: ClassEvaluationResult,
    result_b: ClassEvaluationResult,
) -> ComparisonResult:
    """Statistically compare errors between two structure classes.

    Uses a simple Welch's t-test on per-structure force MAE values.
    Energy comparison uses ratio (no test due to variable natoms).

    Args:
        result_a: First class result.
        result_b: Second class result.

    Returns:
        ComparisonResult with ratios and statistical test.
    """
    from scipy import stats

    a_force = [e["force_mae"] for e in result_a.per_structure_errors]
    b_force = [e["force_mae"] for e in result_b.per_structure_errors]

    energy_ratio = (
        result_a.energy_mae_eV / result_b.energy_mae_eV
        if result_b.energy_mae_eV > 1e-9
        else float("inf")
    )
    force_ratio = (
        result_a.force_mae_eV_per_A / result_b.force_mae_eV_per_A
        if result_b.force_mae_eV_per_A > 1e-9
        else float("inf")
    )

    pvalue = None
    if len(a_force) >= 3 and len(b_force) >= 3:
        _, pvalue = stats.ttest_ind(a_force, b_force, equal_var=False)

    if pvalue is not None and pvalue < 0.05:
        if force_ratio > 1.2:
            conclusion = f"{result_a.class_label} force errors significantly higher than {result_b.class_label} (p={pvalue:.4f})"
        elif force_ratio < 0.83:
            conclusion = f"{result_b.class_label} force errors significantly higher than {result_a.class_label} (p={pvalue:.4f})"
        else:
            conclusion = f"No significant difference in force errors (p={pvalue:.4f})"
    else:
        conclusion = f"Insufficient data or no significant difference (p={pvalue})"

    return ComparisonResult(
        class_a=result_a.class_label,
        class_b=result_b.class_label,
        energy_mae_ratio=energy_ratio,
        force_mae_ratio=force_ratio,
        energy_pvalue=None,
        force_pvalue=pvalue,
        conclusion=conclusion,
    )


# ---------------------------------------------------------------------------
# High-level validator
# ---------------------------------------------------------------------------


class MLFFValidator:
    """High-level validator for MLFF closed-loop hypothesis testing.

    Orchestrates the full pipeline:
    1. Load model
    2. Classify DB structures (EQ / MS / TP / TD / LU / AB)
    3. Evaluate per-class errors
    4. Compare TP vs EQ statistically

    Example:
        >>> validator = MLFFValidator(model_path="best_model", db_path="dataset.db")
        >>> validator.run(device="cuda")
        >>> print(validator.comparison)
    """

    def __init__(
        self,
        model_path: str | Path,
        db_path: str | Path,
        prefilter: "DeterministicPreFilter | None" = None,
        device: str = "cpu",
        cutoff: float = 5.0,
    ) -> None:
        """Initialize validator.

        Args:
            model_path: Path to saved SchNetPack / PaiNN model.
            db_path: Path to ASE database.
            prefilter: DeterministicPreFilter for class splitting.
                If None, a default conservative filter is used.
            device: "cpu" or "cuda".
            cutoff: Neighbor list cutoff (Å).
        """
        self.model_path = Path(model_path)
        self.db_path = Path(db_path)
        self.device = device
        self.cutoff = cutoff
        self.prefilter = prefilter
        self.model: Any | None = None
        self.all_results: list[dict[str, Any]] = []
        self.class_results: dict[str, ClassEvaluationResult] = {}
        self.comparison: ComparisonResult | None = None

    def run(self) -> None:
        """Execute full validation pipeline."""
        logger.info("Loading model from %s", self.model_path)
        self.model = load_schnetpack_model(self.model_path, device=self.device)

        logger.info("Evaluating on %s", self.db_path)
        self.all_results = evaluate_on_db(
            self.model, self.db_path, device=self.device, cutoff=self.cutoff
        )

        # Split by class if prefilter available
        if self.prefilter is not None:
            logger.info("Splitting by PreFilter classes")
            class_indices = split_db_by_class(self.db_path, self.prefilter)
            for label, indices in class_indices.items():
                subset = [self.all_results[i] for i in indices if i < len(self.all_results)]
                if subset:
                    self.class_results[label] = aggregate_results(subset, label)
                    logger.info(
                        "  %s: %d structures, E_MAE=%.4f meV/atom, F_MAE=%.4f eV/Å",
                        label,
                        self.class_results[label].n_structures,
                        self.class_results[label].energy_mae_per_atom_meV,
                        self.class_results[label].force_mae_eV_per_A,
                    )

            # Run TP vs EQ comparison if both exist
            if "TP" in self.class_results and "EQ" in self.class_results:
                self.comparison = compare_class_errors(
                    self.class_results["TP"],
                    self.class_results["EQ"],
                )
                logger.info("Comparison: %s", self.comparison.conclusion)
        else:
            logger.warning("No PreFilter provided; skipping per-class analysis")
            # Aggregate everything as "all"
            self.class_results["all"] = aggregate_results(self.all_results, "all")

    def get_report(self) -> dict[str, Any]:
        """Generate a JSON-serializable report dict."""
        report: dict[str, Any] = {
            "model_path": str(self.model_path),
            "db_path": str(self.db_path),
            "n_total": len(self.all_results),
            "classes": {},
        }
        for label, result in self.class_results.items():
            report["classes"][label] = {
                "n_structures": result.n_structures,
                "energy_mae_eV": round(result.energy_mae_eV, 6),
                "energy_mae_per_atom_meV": round(result.energy_mae_per_atom_meV, 4),
                "force_mae_eV_per_A": round(result.force_mae_eV_per_A, 6),
                "force_mae_std_eV_per_A": round(result.force_mae_std_eV_per_A, 6),
            }
        if self.comparison is not None:
            report["comparison"] = {
                "class_a": self.comparison.class_a,
                "class_b": self.comparison.class_b,
                "energy_mae_ratio": round(self.comparison.energy_mae_ratio, 3),
                "force_mae_ratio": round(self.comparison.force_mae_ratio, 3),
                "force_pvalue": (
                    round(self.comparison.force_pvalue, 6)
                    if self.comparison.force_pvalue is not None
                    else None
                ),
                "conclusion": self.comparison.conclusion,
            }
        return report
