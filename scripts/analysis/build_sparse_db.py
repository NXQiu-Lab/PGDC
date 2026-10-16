#!/usr/bin/env python3
"""Build a sparsified dataset using descriptor-space k-center greedy sampling.

Supports two descriptor backends:
    * soap  — full SOAP descriptor via SOAPDescriptorAdapter (slow for >10k samples)
    * lowdim — lightweight ~30-dimensional fingerprint (RDF + composition + cell + energy/force stats)

Usage examples:
    uv run python scripts/analysis/build_sparse_db.py \
        --src outputs/ablation_no_relax/dataset_skip_iter2.db \
        --dst outputs/ablation_sparse/dataset_sparse_10000_global.db \
        --n-select 10000 --mode global --descriptor-type lowdim

    uv run python scripts/analysis/build_sparse_db.py \
        --src outputs/ablation_no_relax/dataset_skip_iter2.db \
        --dst outputs/ablation_sparse/dataset_sparse_10000_phase.db \
        --n-select 10000 --mode phase --descriptor-type lowdim
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import ase.db
import numpy as np
from scipy.spatial.distance import cdist

from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.metric_core.descriptors.base import SOAPDescriptorAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse dataset construction via k-center greedy")
    parser.add_argument("--src", required=True, help="Source ASE DB path")
    parser.add_argument("--dst", required=True, help="Output ASE DB path")
    parser.add_argument("--n-select", type=int, required=True, help="Target number of structures")
    parser.add_argument(
        "--mode",
        choices=["global", "phase", "adaptive"],
        default="adaptive",
        help="Sampling mode: global k-center, proportional per-phase, or density-adaptive per-phase",
    )
    parser.add_argument(
        "--density-k",
        type=int,
        default=10,
        help="Number of nearest neighbors used for phase density estimation (adaptive mode)",
    )
    parser.add_argument(
        "--descriptor-type",
        choices=["soap", "lowdim"],
        default="lowdim",
        help="Descriptor used for sparse sampling",
    )
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--run-profile", action="store_true", help="Run dataset-profile-run after build")
    return parser.parse_args()


def _records_and_phases(src: Path) -> tuple[list[Any], list[Any], list[str]]:
    bundle = AseDbDatasetReader(src).read()
    records = list(bundle.records)
    db = ase.db.connect(str(src))
    rows = list(db.select())
    phases = [str(r.get("phase", "")) for r in rows]
    return records, rows, phases


def _compute_descriptors(
    records: list[Any],
    descriptor_type: str,
    cutoff: float,
    n_max: int,
    l_max: int,
) -> np.ndarray:
    if descriptor_type == "soap":
        adapter = SOAPDescriptorAdapter(cutoff=cutoff, n_max=n_max, l_max=l_max)
        logger.info("Computing SOAP descriptors (%s) for %d records...", adapter.backend_name, len(records))
        feats = adapter.encode_batch(records)
        return np.asarray(feats, dtype=np.float64)

    logger.info("Computing low-dimensional fingerprints for %d records...", len(records))
    feats = [_lowdim_fingerprint(r) for r in records]
    matrix = np.asarray(feats, dtype=np.float64)
    # Z-score normalize each column so all feature groups contribute comparably.
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def _lowdim_fingerprint(record: Any) -> list[float]:
    """Lightweight structure fingerprint (~28 dim).

    Combines:
      * radial distribution function (RDF) histogram (20 bins)
      * composition counts (2 dims for Si/O)
      * cell features (3 dims)
      * energy and force statistics (3 dims)

    Uses vectorised numpy distance calculation; PBC is ignored for speed because
    the fingerprint is only used for coarse-grained density estimation.
    """
    positions = np.asarray(record.positions, dtype=np.float64)
    numbers = list(record.atomic_numbers)
    cell = record.cell
    n_atoms = len(numbers)

    features: list[float] = []

    # 1. RDF histogram (20 bins, cutoff 6.0 Å)
    r_cut = 6.0
    n_bins = 20
    rdf = np.zeros(n_bins, dtype=np.float64)
    if n_atoms > 1:
        diff = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # upper triangle, exclude self
        triu_idx = np.triu_indices(n_atoms, k=1)
        pair_dists = dists[triu_idx]
        pair_dists = pair_dists[pair_dists < r_cut]
        if pair_dists.size:
            bin_idx = (pair_dists / r_cut * n_bins).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            np.add.at(rdf, bin_idx, 1.0)

    pair_count = max(n_atoms * (n_atoms - 1) / 2.0, 1.0)
    features.extend((rdf / pair_count).tolist())

    # 2. Composition counts (Si=14, O=8)
    si_count = numbers.count(14)
    o_count = numbers.count(8)
    features.extend([si_count / max(n_atoms, 1), o_count / max(n_atoms, 1)])

    # 3. Cell features
    if cell is not None:
        cell_arr = np.asarray(cell, dtype=np.float64)
        volume = float(np.linalg.det(cell_arr)) if cell_arr.shape == (3, 3) else 0.0
        lengths = [float(np.linalg.norm(v)) for v in cell_arr]
    else:
        volume = 0.0
        lengths = [0.0, 0.0, 0.0]
    features.extend([volume / max(n_atoms, 1), sum(lengths) / 3.0, max(lengths) - min(lengths)])

    # 4. Energy / force statistics
    energy = float(record.energy) if record.energy is not None else 0.0
    if record.forces:
        force_norms = [float(np.linalg.norm(f)) for f in record.forces]
        f_mean = sum(force_norms) / len(force_norms)
        f_max = max(force_norms)
    else:
        f_mean = 0.0
        f_max = 0.0
    features.extend([energy / max(n_atoms, 1), f_mean, f_max])

    return features


def _k_center_greedy(matrix: np.ndarray, k: int) -> list[int]:
    """Return indices of k structures selected by k-center greedy algorithm.

    Uses incremental distance updates so the total cost is O(k * n * dim)
    instead of O(k^2 * n * dim).
    """
    n = matrix.shape[0]
    if k >= n:
        return list(range(n))

    # First pick: point with minimum average distance to others (most central).
    # Computing the full distance matrix once for the first pick is acceptable
    # because n is at most a few thousand for individual phases.
    full_dist = cdist(matrix, matrix, metric="euclidean")
    mean_dists = (full_dist.sum(axis=1) - np.diag(full_dist)) / max(1, n - 1)
    selected = [int(np.argmin(mean_dists))]

    # Incremental farthest-point sampling for the remaining picks.
    min_dist_to_selected = np.full(n, np.inf, dtype=np.float64)
    # Update distances for the first selected point.
    min_dist_to_selected = full_dist[selected[0]]
    min_dist_to_selected[selected[0]] = -1.0  # exclude selected

    for _ in range(1, k):
        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)
        min_dist_to_selected[next_idx] = -1.0  # exclude selected
        # Update only against the newly added point.
        new_dists = np.linalg.norm(matrix - matrix[next_idx], axis=1)
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_dists)

    return selected


def _select_global(matrix: np.ndarray, k: int) -> list[int]:
    return _k_center_greedy(matrix, k)


def _select_phase_stratified(
    matrix: np.ndarray, phases: list[str], k: int
) -> list[int]:
    """Stratified k-center: allocate quota per phase proportionally and sample within each phase."""
    phase_counts = Counter(phases)
    total = len(phases)
    phase_order = list(phase_counts.keys())

    # Proportional quotas, rounding down then fill remainders.
    quotas: dict[str, int] = {}
    remaining = k
    for p in phase_order:
        q = math.floor(k * phase_counts[p] / total)
        quotas[p] = q
        remaining -= q
    # Distribute leftover to phases with highest fractional parts.
    if remaining:
        fractions = {
            p: (k * phase_counts[p] / total) - quotas[p] for p in phase_order
        }
        for p in sorted(fractions, key=fractions.get, reverse=True)[:remaining]:
            quotas[p] += 1

    logger.info("Phase quotas: %s", quotas)

    selected: list[int] = []
    for phase in phase_order:
        phase_indices = [i for i, p in enumerate(phases) if p == phase]
        q = min(quotas[phase], len(phase_indices))
        if q <= 0:
            continue
        phase_matrix = matrix[phase_indices]
        local_selected = _k_center_greedy(phase_matrix, q)
        selected.extend(phase_indices[i] for i in local_selected)

    return selected


def _phase_density_score(matrix: np.ndarray, k: int = 10) -> float:
    """Return median k-NN distance for a phase matrix; larger == more dispersed."""
    n = matrix.shape[0]
    if n <= 1:
        return 1.0
    k_eff = min(k, n - 1)
    # Compute squared distances first to avoid sqrt for all pairs.
    dists = cdist(matrix, matrix, metric="euclidean")
    # Sort each row; skip the zero self-distance.
    sorted_dists = np.sort(dists, axis=1)[:, 1 : k_eff + 1]
    return float(np.median(sorted_dists))


def _select_adaptive_stratified(
    matrix: np.ndarray,
    phases: list[str],
    k: int,
    density_k: int = 10,
) -> list[int]:
    """Density-adaptive phase stratified k-center.

    Quotas are allocated so that phases with larger median k-NN distance
    (more dispersed / lower redundancy) receive a larger share of the budget.
    Within each phase, k-center greedy removes local redundancy.
    """
    phase_counts = Counter(phases)
    phase_order = list(phase_counts.keys())

    density_scores: dict[str, float] = {}
    for phase in phase_order:
        phase_indices = [i for i, p in enumerate(phases) if p == phase]
        density_scores[phase] = _phase_density_score(
            matrix[phase_indices], k=density_k
        )

    total_score = sum(density_scores.values())
    if total_score <= 0:
        total_score = 1.0

    # Allocate quotas proportional to density score, capped at phase size.
    raw_quotas = {p: k * density_scores[p] / total_score for p in phase_order}
    # First pass: floor and enforce cap.
    quotas: dict[str, int] = {}
    remaining = k
    for p in phase_order:
        q = min(math.floor(raw_quotas[p]), phase_counts[p])
        quotas[p] = q
        remaining -= q
    # Distribute leftover to phases with largest fractional residuals.
    if remaining > 0:
        residuals = {
            p: raw_quotas[p] - quotas[p] for p in phase_order if quotas[p] < phase_counts[p]
        }
        for p in sorted(residuals, key=residuals.get, reverse=True)[:remaining]:
            quotas[p] += 1

    logger.info("Adaptive phase density scores: %s", density_scores)
    logger.info("Adaptive phase quotas: %s", quotas)

    selected: list[int] = []
    for phase in phase_order:
        phase_indices = [i for i, p in enumerate(phases) if p == phase]
        q = min(quotas[phase], len(phase_indices))
        if q <= 0:
            continue
        phase_matrix = matrix[phase_indices]
        local_selected = _k_center_greedy(phase_matrix, q)
        selected.extend(phase_indices[i] for i in local_selected)

    return selected


def _write_subset(src: Path, dst: Path, selected_indices: list[int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    db_in = ase.db.connect(str(src))
    db_out = ase.db.connect(str(dst))
    rows = list(db_in.select())
    for idx in selected_indices:
        row = rows[idx]
        db_out.write(row.toatoms(), **row.key_value_pairs)
    logger.info("Wrote %d structures to %s", len(selected_indices), dst)


def _run_profile(dst: Path) -> None:
    import subprocess

    output_root = dst.parent / f"profile_{dst.stem}"
    cmd = [
        "uv", "run", "atom-dataset-index", "dataset-profile-run",
        "--source-type", "ase_db",
        "--datapath", str(dst),
        "--preset", "core",
        "--output-root", str(output_root),
    ]
    logger.info("Running profile: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    records, _rows, phases = _records_and_phases(src)
    logger.info("Loaded %d records from %s", len(records), src)
    logger.info("Phase distribution: %s", dict(Counter(phases).most_common()))

    matrix = _compute_descriptors(
        records,
        args.descriptor_type,
        args.cutoff,
        args.n_max,
        args.l_max,
    )
    logger.info("Descriptor matrix shape: %s", matrix.shape)

    if args.mode == "global":
        selected = _select_global(matrix, args.n_select)
    elif args.mode == "phase":
        selected = _select_phase_stratified(matrix, phases, args.n_select)
    else:
        selected = _select_adaptive_stratified(
            matrix, phases, args.n_select, density_k=args.density_k
        )

    logger.info("Selected %d structures", len(selected))
    _write_subset(src, dst, selected)

    if args.run_profile:
        _run_profile(dst)


if __name__ == "__main__":
    main()
