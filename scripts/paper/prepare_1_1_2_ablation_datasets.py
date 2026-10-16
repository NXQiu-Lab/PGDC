#!/usr/bin/env python3
"""Prepare ablation datasets for paper Section 1.1.2.

This script generates the CsSnClBrI3 and SiO2 subsets used to study the
physical meaning and system-dependence of the six-dimensional dataset
scorecard (Stability, Compositional Diversity, Structural Diversity,
Uniformity, Coverage, Redundancy). It performs dataset splits, writes ASE
databases, and validates each output with the full DatasetProfiler.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms
from ase.db import connect

# ASE >=3.23 compatibility patch for SchNetPack-style metadata access.
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
from atom_dataset_index.metric_core.descriptors.base import SOAPDescriptorAdapter
from atom_dataset_index.profiling import DatasetProfiler, PoolBuilder
from atom_dataset_index.sampling import DeterministicPreFilter
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[2]
CS_SOURCE = ROOT / "data" / "CsSnClBrI3_perovskite.db"
SIO2_SOURCE = ROOT / "sio2_al_build" / "dataset.db"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "1_1_2_final_datasets"


def _build_kwargs(record: object) -> dict[str, Any]:
    """Build ASE DB keyword arguments from record metadata (primitive values only)."""
    kwargs: dict[str, Any] = {}
    for key, value in record.metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            kwargs[key] = value
    return kwargs


def _write_ase_db(
    records: list,
    output_path: str | Path,
    energy_key: str = "energy",
    force_key: str = "forces",
    extra_kv: dict[str, Any] | Callable[[object], dict[str, Any]] | None = None,
    distance_unit: str = "Angstrom",
    property_unit_dict: dict[str, str] | None = None,
) -> Path:
    """Write StructureRecords to an ASE DB with SchNetPack-compatible metadata."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    property_unit_dict = property_unit_dict or {
        energy_key: "eV",
        force_key: "eV/Angstrom",
    }
    extra_kv = extra_kv or {}

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
            kwargs = _build_kwargs(rec)
            if callable(extra_kv):
                kwargs.update(extra_kv(rec))
            else:
                kwargs.update(extra_kv)
            db.write(atoms, data=data, **kwargs)

        db.metadata = {
            "_distance_unit": distance_unit,
            "_property_unit_dict": property_unit_dict,
            "atomrefs": {},
        }
    return output


def _uniform_phase_test_split(
    pool_to_phase: dict[str, int],
    pools: list,
    samples_per_phase: int,
    seed: int,
    min_train_per_phase: int = 1,
) -> tuple[set[int], dict[int, list], list[int], list[int], list[int]]:
    """Uniform-phase test split (mirrors sample_uniformity_ablation_strict_split.py)."""
    rng = random.Random(seed)
    phase_to_records: dict[int, list] = {i: [] for i in set(pool_to_phase.values())}
    for pool in pools:
        phase_idx = pool_to_phase[pool.pool_id]
        phase_to_records[phase_idx].extend(pool.records)
    for recs in phase_to_records.values():
        rng.shuffle(recs)

    phase_ids = sorted(phase_to_records.keys())
    sizes = [len(phase_to_records[pid]) for pid in phase_ids]
    test_ids: set[int] = set()
    available_records: dict[int, list] = {}
    allocations: list[int] = []

    for pid in phase_ids:
        recs = phase_to_records[pid]
        max_test = max(0, len(recs) - min_train_per_phase)
        n_test = min(samples_per_phase, max_test)
        test_ids.update(r.sample_id for r in recs[:n_test])
        available_records[pid] = recs[n_test:]
        allocations.append(n_test)

    return test_ids, available_records, phase_ids, sizes, allocations


def _compositional_subset(
    available_records: dict[int, list],
    n_phases: int,
    target_total: int,
    seed: int,
) -> tuple[list, list[int]]:
    """Build a compositional-diversity subset from the top-N largest phases."""
    rng = random.Random(seed)
    sorted_phases = sorted(
        available_records.keys(),
        key=lambda pid: len(available_records[pid]),
        reverse=True,
    )
    selected = sorted_phases[:n_phases]
    base = target_total // n_phases
    extra = target_total % n_phases

    chosen: list = []
    counts: list[int] = []
    for i, pid in enumerate(selected):
        quota = base + (1 if i < extra else 0)
        recs = list(available_records[pid])
        rng.shuffle(recs)
        take = min(quota, len(recs))
        chosen.extend(recs[:take])
        counts.append(take)
    return chosen, counts


def _lowdim_fingerprint(record: object) -> list[float]:
    """Lightweight structure fingerprint for SiO2 density-adaptive sampling."""
    positions = np.asarray(record.positions, dtype=np.float64)
    numbers = list(record.atomic_numbers)
    cell = record.cell
    n_atoms = len(numbers)

    features: list[float] = []
    r_cut = 6.0
    n_bins = 20
    rdf = np.zeros(n_bins, dtype=np.float64)
    if n_atoms > 1:
        diff = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        triu_idx = np.triu_indices(n_atoms, k=1)
        pair_dists = dists[triu_idx]
        pair_dists = pair_dists[pair_dists < r_cut]
        if pair_dists.size:
            bin_idx = (pair_dists / r_cut * n_bins).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            np.add.at(rdf, bin_idx, 1.0)

    pair_count = max(n_atoms * (n_atoms - 1) / 2.0, 1.0)
    features.extend((rdf / pair_count).tolist())

    si_count = numbers.count(14)
    o_count = numbers.count(8)
    features.extend([si_count / max(n_atoms, 1), o_count / max(n_atoms, 1)])

    if cell is not None:
        cell_arr = np.asarray(cell, dtype=np.float64)
        volume = float(np.linalg.det(cell_arr)) if cell_arr.shape == (3, 3) else 0.0
        lengths = [float(np.linalg.norm(v)) for v in cell_arr]
    else:
        volume = 0.0
        lengths = [0.0, 0.0, 0.0]
    features.extend([volume / max(n_atoms, 1), sum(lengths) / 3.0, max(lengths) - min(lengths)])

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


def _k_center_greedy(matrix: np.ndarray, k: int, seed: int) -> list[int]:
    """Incremental k-center greedy selection.

    Uses a deterministic random first pick controlled by ``seed`` and
    vectorized distance updates to handle class sizes up to ~15k efficiently.
    """
    n = matrix.shape[0]
    if k >= n:
        return list(range(n))
    rng = random.Random(seed)
    selected = [rng.randrange(n)]
    min_dist = np.full(n, np.inf, dtype=np.float64)
    min_dist[selected[0]] = -1.0

    for _ in range(1, k):
        new_dists = np.linalg.norm(matrix - matrix[selected[-1]], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
        min_dist[selected[-1]] = -1.0
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    return selected


def _k_center_sample_records(
    records: list,
    k: int,
    seed: int,
) -> list:
    """Return ``k`` records selected by k-center greedy on the lowdim fingerprint."""
    if k >= len(records):
        return records
    feats = np.asarray([_lowdim_fingerprint(r) for r in records], dtype=np.float64)
    means = feats.mean(axis=0)
    stds = feats.std(axis=0)
    stds[stds == 0] = 1.0
    normalized = (feats - means) / stds
    selected_indices = _k_center_greedy(normalized, k, seed)
    return [records[i] for i in selected_indices]


def _profile_db(db_path: Path, dataset_name: str) -> dict[str, Any]:
    """Run the default DatasetProfiler on an ASE DB and return the summary."""
    dataset = AseDbDatasetReader(db_path).read(dataset_name=dataset_name)
    profiler = DatasetProfiler()
    profile = profiler.profile(dataset)
    return dict(profile.summary)


def _prepare_cs_test_and_splits(
    cs_dataset: object,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Prepare CsSnClBrI3 test set and compositional/structural ablations."""
    pool_builder = PoolBuilder()
    pools = pool_builder.build_pools(cs_dataset)
    phase_count, phase_counts, pool_to_phase = DatasetProfiler._compute_effective_phases(pools)
    print(f"CsSnClBrI3: {len(cs_dataset.records)} records, {len(pools)} pools, "
          f"{phase_count} effective phases")

    test_ids, available_records, phase_ids, sizes, allocations = _uniform_phase_test_split(
        pool_to_phase=pool_to_phase,
        pools=pools,
        samples_per_phase=20,
        seed=seed,
        min_train_per_phase=1,
    )
    print(f"  test set: {len(test_ids)} records (target 20 per phase)")

    # --- Cs test set ---
    test_records = [r for r in cs_dataset.records if r.sample_id in test_ids]
    cs_test_path = _write_ase_db(test_records, output_dir / "cs_test.db")

    # --- Compositional diversity subsets ---
    # Determine the minimum number of phases whose cumulative available size
    # reaches the target, so the "low" subset can actually hit N=10000.
    sorted_phase_ids = sorted(
        available_records.keys(), key=lambda pid: len(available_records[pid]), reverse=True
    )
    cumulative = 0
    low_n_phases = phase_count
    for i, pid in enumerate(sorted_phase_ids):
        cumulative += len(available_records[pid])
        if cumulative >= 10000:
            low_n_phases = i + 1
            break

    comp_configs = [
        ("cs_compositional_diversity_low_N10000.db", low_n_phases, 10000),
        ("cs_compositional_diversity_mid_N10000.db", max(low_n_phases + 15, 40), 10000),
        ("cs_compositional_diversity_high_N10000.db", phase_count, 10000),
    ]
    comp_results: dict[str, Any] = {}
    for filename, n_phases, target_total in comp_configs:
        records, counts = _compositional_subset(
            available_records, n_phases, target_total, seed
        )
        path = _write_ase_db(records, output_dir / filename)
        comp_results[filename] = {
            "path": str(path.relative_to(output_dir)),
            "actual_size": len(records),
            "target_size": target_total,
            "n_phases_requested": n_phases,
            "n_phases_with_samples": sum(1 for c in counts if c > 0),
            "phase_counts": counts,
        }
        print(f"  {filename}: {len(records)} records from "
              f"{sum(1 for c in counts if c > 0)} phases")

    return {
        "phase_count_effective": phase_count,
        "test_set": {
            "path": str(cs_test_path.relative_to(output_dir)),
            "size": len(test_records),
            "test_samples_per_phase": 20,
            "min_train_per_phase": 1,
            "phase_ids": phase_ids,
            "phase_sizes": sizes,
            "phase_test_allocations": allocations,
        },
        "compositional_diversity": comp_results,
    }


def _build_structural_diversity_subsets(
    records: list,
    target_n: int,
    seed: int,
    n_clusters: int = 8,
) -> dict[str, Any]:
    """Build low/high structural-diversity subsets from a single-composition pool.

    Low: sample only from the largest SOAP-space cluster.
    High: sample proportionally from all clusters.
    """
    rng = random.Random(seed)
    soap = SOAPDescriptorAdapter(cutoff=6.0, n_max=4, l_max=4)
    raw_descriptors = soap.encode_batch(records)
    max_len = max(len(d) for d in raw_descriptors)
    descriptors = np.zeros((len(records), max_len), dtype=np.float64)
    for i, desc in enumerate(raw_descriptors):
        descriptors[i, : len(desc)] = desc

    n_clusters = min(n_clusters, len(records))
    if n_clusters < 2:
        labels = np.zeros(len(records), dtype=int)
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(descriptors)

    clusters: dict[int, list] = defaultdict(list)
    for rec, label in zip(records, labels):
        clusters[int(label)].append(rec)
    cluster_sizes = {label: len(recs) for label, recs in clusters.items()}

    largest_cluster_label = max(clusters.keys(), key=lambda lbl: len(clusters[lbl]))
    largest_cluster_records = list(clusters[largest_cluster_label])

    low_target = min(target_n, len(largest_cluster_records))
    rng.shuffle(largest_cluster_records)
    low_records = largest_cluster_records[:low_target]

    high_target = min(target_n, len(records))
    sorted_clusters = sorted(clusters.keys(), key=lambda lbl: len(clusters[lbl]), reverse=True)
    quotas: dict[int, int] = {}
    for lbl in sorted_clusters:
        quotas[lbl] = int(high_target * len(clusters[lbl]) / len(records))
    remaining = high_target - sum(quotas.values())
    for lbl in sorted_clusters:
        if remaining <= 0:
            break
        if quotas[lbl] < len(clusters[lbl]):
            quotas[lbl] += 1
            remaining -= 1

    high_records: list = []
    high_counts: list[int] = []
    for lbl in sorted_clusters:
        q = min(quotas[lbl], len(clusters[lbl]))
        recs = list(clusters[lbl])
        rng.shuffle(recs)
        high_records.extend(recs[:q])
        high_counts.append(q)

    return {
        "clusters": clusters,
        "cluster_sizes": dict(sorted(cluster_sizes.items())),
        "low": {
            "records": low_records,
            "target_size": target_n,
            "actual_size": len(low_records),
            "cluster_label": int(largest_cluster_label),
        },
        "high": {
            "records": high_records,
            "target_size": target_n,
            "actual_size": len(high_records),
            "per_cluster_counts": high_counts,
        },
    }


def _prepare_sio2(
    sio2_dataset: object,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Prepare SiO2 test set, random 10k, EQ-preserve adaptive 10k, and structural-diversity subsets."""
    print(f"SiO2: {len(sio2_dataset.records)} records")
    print("  classifying with DeterministicPreFilter ...")
    prefilter = DeterministicPreFilter(reference_dataset=sio2_dataset)
    labels = [prefilter.classify(r).label for r in sio2_dataset.records]
    label_counts = Counter(labels)
    print(f"  classification: {dict(sorted(label_counts.items()))}")

    rng = random.Random(seed)
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    for idx_list in label_to_indices.values():
        rng.shuffle(idx_list)

    # Stratified test set.  Target ~1000 total while leaving TD for training.
    # The source data has 753 TD structures, so we sample 250 rather than
    # exhausting them, preserving the TD class for the training subsets.
    test_targets = {
        "EQ": 400,
        "MS": 400,
        "TD": 250,
    }
    test_indices: list[int] = []
    other_budget = 0
    for label, target in test_targets.items():
        available = label_to_indices.get(label, [])
        take = min(target, len(available))
        test_indices.extend(available[:take])
        label_to_indices[label] = available[take:]
    # Small samples from any other present classes (TP/LU/AB), expected zero for SiO2.
    for label in sorted(label_to_indices.keys()):
        if label not in test_targets:
            available = label_to_indices[label]
            take = min(10, len(available))
            test_indices.extend(available[:take])
            label_to_indices[label] = available[take:]
            other_budget += take

    sample_id_to_label = {sio2_dataset.records[i].sample_id: labels[i] for i in test_indices}
    test_records = [sio2_dataset.records[i] for i in test_indices]
    test_path = _write_ase_db(
        test_records,
        output_dir / "sio2_test.db",
        extra_kv=lambda rec: {"prefilter_label": sample_id_to_label.get(rec.sample_id, "UNKNOWN")},
    )
    print(f"  test set: {len(test_records)} records")

    # Random 10k from remaining records.
    remaining_indices = [
        i for i in range(len(sio2_dataset.records)) if i not in set(test_indices)
    ]
    rng.shuffle(remaining_indices)
    random_10k_indices = remaining_indices[:10000]
    random_10k_records = [sio2_dataset.records[i] for i in random_10k_indices]
    sample_id_to_label = {sio2_dataset.records[i].sample_id: labels[i] for i in random_10k_indices}
    random_path = _write_ase_db(
        random_10k_records,
        output_dir / "sio2_random_10k.db",
        extra_kv=lambda rec: {"prefilter_label": sample_id_to_label.get(rec.sample_id, "UNKNOWN")},
    )
    print(f"  random 10k: {len(random_10k_records)} records")

    # EQ-preserve adaptive 10k: density-adaptive k-center per class with
    # quotas matching the full-dataset EQ/MS/TD proportions.
    non_test_labels = [labels[i] for i in remaining_indices]
    non_test_records = [sio2_dataset.records[i] for i in remaining_indices]
    class_records: dict[str, list] = defaultdict(list)
    for rec, label in zip(non_test_records, non_test_labels):
        class_records[label].append(rec)

    total = len(sio2_dataset.records)
    eq_frac = label_counts.get("EQ", 0) / total
    ms_frac = label_counts.get("MS", 0) / total
    td_frac = label_counts.get("TD", 0) / total
    class_fracs = {"EQ": eq_frac, "MS": ms_frac, "TD": td_frac}

    # Initial quotas based on full-dataset proportions.
    quotas: dict[str, int] = {}
    for label, frac in class_fracs.items():
        quotas[label] = min(int(round(10000 * frac)), len(class_records.get(label, [])))

    # Redistribute any shortfall to classes that still have slack.
    for _ in range(10):
        current_total = sum(quotas.values())
        if current_total >= 10000:
            break
        remaining = 10000 - current_total
        slack = {
            label: len(class_records.get(label, [])) - quotas.get(label, 0)
            for label in class_fracs
        }
        total_slack = sum(max(0, s) for s in slack.values())
        if total_slack == 0:
            break
        for label in class_fracs:
            add = min(
                int(round(remaining * class_fracs[label])),
                max(0, slack[label]),
            )
            quotas[label] = quotas.get(label, 0) + add

    print(f"  adaptive quotas: {quotas}")
    adaptive_records: list = []
    adaptive_label_map: dict[int, str] = {}
    for label, quota in quotas.items():
        recs = class_records.get(label, [])
        if quota <= 0 or not recs:
            continue
        selected = _k_center_sample_records(recs, min(quota, len(recs)), seed)
        adaptive_records.extend(selected)
        for rec in selected:
            adaptive_label_map[rec.sample_id] = label
        print(f"    {label}: selected {len(selected)} / {len(recs)}")

    adaptive_path = _write_ase_db(
        adaptive_records,
        output_dir / "sio2_adaptive_eq_preserve_10k.db",
        extra_kv=lambda rec: {"prefilter_label": adaptive_label_map.get(rec.sample_id, "UNKNOWN")},
    )
    print(f"  adaptive 10k: {len(adaptive_records)} records")

    # Structural diversity subsets: cluster the full non-test SiO2 pool in SOAP space.
    sd_result = _build_structural_diversity_subsets(non_test_records, target_n=5000, seed=seed)
    low_sd_records = sd_result["low"]["records"]
    high_sd_records = sd_result["high"]["records"]
    sd_low_path = _write_ase_db(
        low_sd_records,
        output_dir / "sio2_structural_diversity_low_N5000.db",
        extra_kv=lambda rec: {"prefilter_label": adaptive_label_map.get(rec.sample_id, "UNKNOWN")},
    )
    sd_high_path = _write_ase_db(
        high_sd_records,
        output_dir / "sio2_structural_diversity_high_N5000.db",
        extra_kv=lambda rec: {"prefilter_label": adaptive_label_map.get(rec.sample_id, "UNKNOWN")},
    )
    print(f"  structural diversity low: {len(low_sd_records)} records "
          f"(cluster {sd_result['low']['cluster_label']})")
    print(f"  structural diversity high: {len(high_sd_records)} records "
          f"across {len(sd_result['high']['per_cluster_counts'])} clusters")

    return {
        "classification_counts": dict(sorted(label_counts.items())),
        "full_dataset_fractions": {
            "EQ": round(eq_frac, 4),
            "MS": round(ms_frac, 4),
            "TD": round(td_frac, 4),
        },
        "test_set": {
            "path": str(test_path.relative_to(output_dir)),
            "size": len(test_records),
            "targets_per_class": test_targets,
            "other_classes_sampled": other_budget,
        },
        "random_10k": {
            "path": str(random_path.relative_to(output_dir)),
            "size": len(random_10k_records),
        },
        "adaptive_eq_preserve_10k": {
            "path": str(adaptive_path.relative_to(output_dir)),
            "size": len(adaptive_records),
            "quotas": quotas,
        },
        "structural_diversity": {
            "cluster_sizes": sd_result["cluster_sizes"],
            "low": {
                "path": str(sd_low_path.relative_to(output_dir)),
                "size": len(low_sd_records),
                "cluster_label": sd_result["low"]["cluster_label"],
            },
            "high": {
                "path": str(sd_high_path.relative_to(output_dir)),
                "size": len(high_sd_records),
                "per_cluster_counts": sd_result["high"]["per_cluster_counts"],
            },
        },
    }


def _profile_existing(output_dir: Path, seed: int) -> dict[str, Any]:
    """Profile existing output files and write/overwrite the manifest."""
    output_files: dict[str, Path] = {
        "cs_test": output_dir / "cs_test.db",
        "cs_compositional_diversity_low_N10000": output_dir / "cs_compositional_diversity_low_N10000.db",
        "cs_compositional_diversity_mid_N10000": output_dir / "cs_compositional_diversity_mid_N10000.db",
        "cs_compositional_diversity_high_N10000": output_dir / "cs_compositional_diversity_high_N10000.db",
        "sio2_structural_diversity_low_N5000": output_dir / "sio2_structural_diversity_low_N5000.db",
        "sio2_structural_diversity_high_N5000": output_dir / "sio2_structural_diversity_high_N5000.db",
        "sio2_test": output_dir / "sio2_test.db",
        "sio2_random_10k": output_dir / "sio2_random_10k.db",
        "sio2_adaptive_eq_preserve_10k": output_dir / "sio2_adaptive_eq_preserve_10k.db",
    }

    file_entries: list[dict[str, Any]] = []
    print("Profiling output datasets (this may take several minutes) ...")
    for name, path in output_files.items():
        if not path.exists():
            print(f"  skipping {name}: file not found")
            continue
        print(f"  profiling {name} ...", flush=True)
        try:
            summary = _profile_db(path, name)
        except Exception as exc:
            print(f"    profiling failed: {exc}")
            summary = {"error": str(exc)}
        file_entries.append({
            "name": name,
            "path": str(path.relative_to(output_dir)),
            "absolute_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "record_count": summary.get("record_count") if isinstance(summary, dict) else None,
            "profile_summary": summary,
        })
    print()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "output_dir": str(output_dir.resolve()),
        "note": "Generated by --profile-only; construction details were retained from the original run.",
        "files": file_entries,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 70)
    print("Generated files")
    print("=" * 70)
    for entry in file_entries:
        size_mb = entry["size_bytes"] / (1024 * 1024)
        rec_count = entry.get("record_count") or "?"
        print(f"  {entry['name']:<40} {rec_count:>6} records  {size_mb:>6.2f} MB")
    print()

    print("Six-dimensional scorecard summary")
    print("=" * 70)
    score_keys = [
        "stability_score",
        "compositional_diversity_score",
        "structural_diversity_score",
        "uniformity_score",
        "coverage_score",
        "redundancy_score",
        "phase_count_effective",
    ]
    header = f"{'dataset':<40}" + "".join(f"{k[:8]:>10}" for k in score_keys)
    print(header)
    print("-" * len(header))
    for entry in file_entries:
        summary = entry.get("profile_summary") or {}
        row = f"{entry['name']:<40}"
        for key in score_keys:
            value = summary.get(key)
            if value is None:
                row += f"{'-':>10}"
            else:
                row += f"{value:>10.2f}"
        print(row)
    print()

    print(f"Manifest written to: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare 1.1.2 ablation datasets for the six-dimensional scorecard paper."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated datasets and manifest.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the final DatasetProfiler validation (faster, no scores).",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip generation and only (re-)profile existing output files.",
    )
    args = parser.parse_args()

    if args.skip_profile and args.profile_only:
        print("Error: --skip-profile and --profile-only are mutually exclusive.", file=__import__("sys").stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_only:
        _profile_existing(output_dir, args.seed)
        return 0

    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()

    if not CS_SOURCE.exists():
        print(f"Error: CsSnClBrI3 source not found: {CS_SOURCE}", file=__import__("sys").stderr)
        return 1
    if not SIO2_SOURCE.exists():
        print(f"Error: SiO2 source not found: {SIO2_SOURCE}", file=__import__("sys").stderr)
        return 1

    cs_dataset = AseDbDatasetReader(CS_SOURCE).read(dataset_name="cs_full")
    sio2_dataset = AseDbDatasetReader(SIO2_SOURCE).read(dataset_name="sio2_full")

    cs_info = _prepare_cs_test_and_splits(cs_dataset, output_dir, args.seed)
    print()
    sio2_info = _prepare_sio2(sio2_dataset, output_dir, args.seed)
    print()

    # Collect all output files and profile them.
    output_files: dict[str, Path] = {
        "cs_test": output_dir / "cs_test.db",
        "cs_compositional_diversity_low_N10000": output_dir / "cs_compositional_diversity_low_N10000.db",
        "cs_compositional_diversity_mid_N10000": output_dir / "cs_compositional_diversity_mid_N10000.db",
        "cs_compositional_diversity_high_N10000": output_dir / "cs_compositional_diversity_high_N10000.db",
        "sio2_structural_diversity_low_N5000": output_dir / "sio2_structural_diversity_low_N5000.db",
        "sio2_structural_diversity_high_N5000": output_dir / "sio2_structural_diversity_high_N5000.db",
        "sio2_test": output_dir / "sio2_test.db",
        "sio2_random_10k": output_dir / "sio2_random_10k.db",
        "sio2_adaptive_eq_preserve_10k": output_dir / "sio2_adaptive_eq_preserve_10k.db",
    }

    file_entries: list[dict[str, Any]] = []
    if not args.skip_profile:
        print("Profiling output datasets (this may take several minutes) ...")
        for name, path in output_files.items():
            if not path.exists():
                continue
            print(f"  profiling {name} ...", flush=True)
            try:
                summary = _profile_db(path, name)
            except Exception as exc:
                print(f"    profiling failed: {exc}")
                summary = {"error": str(exc)}
            file_entries.append({
                "name": name,
                "path": str(path.relative_to(output_dir)),
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "record_count": summary.get("record_count") if isinstance(summary, dict) else None,
                "profile_summary": summary,
            })
        print()
    else:
        for name, path in output_files.items():
            if not path.exists():
                continue
            file_entries.append({
                "name": name,
                "path": str(path.relative_to(output_dir)),
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "record_count": None,
                "profile_summary": None,
            })

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "output_dir": str(output_dir.resolve()),
        "inputs": {
            "cs_source": str(CS_SOURCE.resolve()),
            "sio2_source": str(SIO2_SOURCE.resolve()),
        },
        "cs_info": cs_info,
        "sio2_info": sio2_info,
        "files": file_entries,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Console summary
    print("=" * 70)
    print("Generated files")
    print("=" * 70)
    for entry in file_entries:
        size_mb = entry["size_bytes"] / (1024 * 1024)
        rec_count = entry.get("record_count") or "?"
        print(f"  {entry['name']:<40} {rec_count:>6} records  {size_mb:>6.2f} MB")
    print()

    if not args.skip_profile:
        print("Six-dimensional scorecard summary")
        print("=" * 70)
        score_keys = [
            "stability_score",
            "compositional_diversity_score",
            "structural_diversity_score",
            "uniformity_score",
            "coverage_score",
            "redundancy_score",
            "phase_count_effective",
        ]
        header = f"{'dataset':<40}" + "".join(f"{k[:8]:>10}" for k in score_keys)
        print(header)
        print("-" * len(header))
        for entry in file_entries:
            summary = entry.get("profile_summary") or {}
            row = f"{entry['name']:<40}"
            for key in score_keys:
                value = summary.get(key)
                if value is None:
                    row += f"{'-':>10}"
                else:
                    row += f"{value:>10.2f}"
            print(row)
        print()

    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
