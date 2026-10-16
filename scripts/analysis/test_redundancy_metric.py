#!/usr/bin/env python3
"""
Prototype test for a redundancy/novelty metric (Scheme A: k-NN distance).

Computes descriptor-space median k-th nearest-neighbor distance for a list of
ASE databases and emits both raw distances and 0-100 scores.  The expectation is
that heavily redundant datasets (full AL dataset with relax frames) score worse
(lower novelty / higher redundancy) than sparse/adaptive subsets.

Run from repo root:
    uv run python scripts/analysis/test_redundancy_metric.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

# Repo path setup
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from atom_dataset_index.io import AseDbDatasetReader
from atom_dataset_index.metric_core.descriptors import SOAPDescriptorAdapter


# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
K_NEIGHBORS = 5
CHUNK_SIZE = 500
DESCRIPTOR_ADAPTER = SOAPDescriptorAdapter()

# Datasets to compare (paths relative to repo root)
DATASETS: list[tuple[str, Path]] = [
    ("full_dataset", _REPO_ROOT / "sio2_al_build" / "dataset.db"),
    ("no_relax", _REPO_ROOT / "outputs" / "ablation_no_relax" / "dataset_no_relax.db"),
    ("skip_iter2", _REPO_ROOT / "outputs" / "ablation_no_relax" / "dataset_skip_iter2.db"),
    ("sparse_phase", _REPO_ROOT / "outputs" / "ablation_sparse" / "dataset_sparse_10000_phase.db"),
    ("sparse_adaptive", _REPO_ROOT / "outputs" / "ablation_sparse" / "dataset_sparse_10000_adaptive.db"),
]

# Sigmoid parameters for the 0-100 score.  Calibrated on the SiO2 ablation
# datasets: the full/unsampled set has the smallest median k-NN distance, while
# the density-adaptive sparse subset has the largest.  Values are in the same
# descriptor-space units as the SOAP output.
SCORE_MID = 0.60
SCORE_SCALE = 0.12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigmoid(value: float, mid: float, scale: float) -> float:
    if scale == 0:
        return 0.0 if value < mid else 1.0
    return 1.0 / (1.0 + math.exp(-(value - mid) / scale))


def descriptor_matrix(records: list) -> np.ndarray:
    """Encode records and return a 2-D numpy array of descriptors."""
    print(f"  Encoding {len(records)} records with {DESCRIPTOR_ADAPTER.backend_name} ...")
    t0 = time.time()
    descs = DESCRIPTOR_ADAPTER.encode_batch(records)
    elapsed = time.time() - t0
    print(f"    encoded in {elapsed:.1f}s")

    dim = max((len(d) for d in descs), default=0)
    matrix = np.zeros((len(descs), dim), dtype=np.float64)
    for i, d in enumerate(descs):
        matrix[i, : len(d)] = d
    return matrix


def median_knn_distance(matrix: np.ndarray, k: int = K_NEIGHBORS, chunk_size: int = CHUNK_SIZE) -> float:
    """Exact median k-th nearest-neighbor distance (excluding self).

    Uses chunked ``cdist`` queries so the full NxN distance matrix is never
    materialised.
    """
    n = matrix.shape[0]
    if n <= k:
        return 0.0

    knn_dists: list[float] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # (chunk, n)
        dists = cdist(matrix[start:end], matrix, metric="euclidean")
        # Exclude self-distance by setting diagonal of this chunk to +inf.
        rows = np.arange(end - start)
        cols = start + rows
        dists[rows, cols] = np.inf
        # k-th nearest neighbor distance for each query
        kth = np.partition(dists, k - 1, axis=1)[:, k - 1]
        knn_dists.extend(kth.tolist())

    return float(np.median(knn_dists))


def redundancy_scores(median_dist: float, mid: float = SCORE_MID, scale: float = SCORE_SCALE) -> dict[str, float]:
    """Convert raw median k-NN distance into a 0-100 negative score.

    - redundancy_score: high = bad (lots of redundancy)
    """
    redundancy = 100.0 * (1.0 - _sigmoid(median_dist, mid=mid, scale=scale))
    return {
        "median_knn_distance": median_dist,
        "redundancy_score": round(redundancy, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    results: list[dict[str, object]] = []

    for name, path in DATASETS:
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue

        print(f"\n[{name}] {path}")
        t0 = time.time()
        dataset = AseDbDatasetReader(path).read()
        load_time = time.time() - t0
        print(f"  loaded {len(dataset.records)} records in {load_time:.1f}s")

        matrix = descriptor_matrix(list(dataset.records))
        median_dist = median_knn_distance(matrix, k=K_NEIGHBORS)
        scores = redundancy_scores(median_dist)

        row = {"name": name, "path": str(path), "n_records": len(dataset.records), **scores}
        results.append(row)
        print(f"  {scores}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Dataset':<20} {'N':>8} {'med_knn':>10} {'Redundancy':>12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<20} {r['n_records']:>8} "
            f"{r['median_knn_distance']:>10.4f} "
            f"{r['redundancy_score']:>12.2f}"
        )

    out_dir = _REPO_ROOT / "outputs" / "ablation_sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "redundancy_prototype_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_neighbors": K_NEIGHBORS,
                "adapter": DESCRIPTOR_ADAPTER.backend_name,
                "score_mid": SCORE_MID,
                "score_scale": SCORE_SCALE,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
