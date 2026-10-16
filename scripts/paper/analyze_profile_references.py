#!/usr/bin/env python3
"""Analyze profile reference structures for CsSnClBrI3 (14056015).

Loads the raw 14056015 dataset, runs DatasetProfiler with reference export,
and produces a structured report on:

1.  **Relaxation snapshot hit rate**: how many relaxation-related structures
    (al_data + relax_test_set snapshots) are selected as references.
2.  **Space-group coverage**: distribution of reference structures across the
    four perovskite lattice types (pm3m, p4mbm, i4mcm, pnma).
3.  **Non-equilibrium hit rate**: hit rates for high-force / high-energy
    structures, demonstrating the "stable-anchor" design philosophy.

Outputs are written to ``outputs/reference_analysis/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ase.io import read

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.profiling import DatasetProfiler, build_default_registry


def _max_force_norm(record: StructureRecord) -> float:
    if not record.forces:
        return 0.0
    return max(math.sqrt(sum(c * c for c in f)) for f in record.forces)


def _mean_force_norm(record: StructureRecord) -> float:
    if not record.forces:
        return 0.0
    norms = [math.sqrt(sum(c * c for c in f)) for f in record.forces]
    return sum(norms) / len(norms)


def _make_record(
    sample_id: int,
    atoms,
    meta: dict[str, Any],
    idx: int,
    source: str,
) -> StructureRecord:
    """Build a StructureRecord with primitive-only metadata."""
    forces = None
    if atoms.calc is not None:
        forces = atoms.get_forces()
    else:
        forces = atoms.arrays.get("forces", None)

    metadata: dict[str, str | int | float | bool | None] = {"source": source}

    # Copy known metadata keys if present for this index
    for key in [
        "lattice_type",
        "n_cl",
        "n_br",
        "n_i",
        "relaxed",
        "relaxation_iteration",
        "al_iteration",
        "al_batch_index",
        "batch",
    ]:
        if key in meta and str(idx) in meta[key]:
            val = meta[key][str(idx)]
            # Normalise numpy / JSON types to primitives
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, bool):
                metadata[key] = val
            elif isinstance(val, int):
                metadata[key] = val
            elif isinstance(val, float):
                metadata[key] = val
            elif isinstance(val, str):
                metadata[key] = val
            else:
                metadata[key] = str(val)

    return StructureRecord(
        sample_id=sample_id,
        atomic_numbers=tuple(int(z) for z in atoms.get_atomic_numbers()),
        positions=tuple(map(tuple, atoms.get_positions())),
        cell=tuple(map(tuple, atoms.get_cell())) if atoms.get_cell() is not None else None,
        pbc=tuple(bool(x) for x in atoms.get_pbc()),
        energy=float(atoms.get_potential_energy()),
        forces=tuple(map(tuple, forces)) if forces is not None else (),
        metadata=metadata,
    )


def load_dataset(base_dir: Path) -> DatasetBundle:
    """Load all 14056015 subsets into a single DatasetBundle."""
    records: list[StructureRecord] = []
    sid = 0

    # --- sp_train_set: 16000 ---
    atoms = read(str(base_dir / "sp_train_set.xyz"), ":")
    with open(base_dir / "sp_train_set.json") as f:
        meta = json.load(f)
    # JSON keys are in insertion order matching ASE read order
    json_keys = list(meta["total_energy"].keys())
    for i, at in enumerate(atoms):
        sid += 1
        records.append(_make_record(sid, at, meta, json_keys[i], "sp_train_set"))

    # --- sp_test_set: 2600 ---
    atoms = read(str(base_dir / "sp_test_set.xyz"), ":")
    with open(base_dir / "sp_test_set.json") as f:
        meta = json.load(f)
    json_keys = list(meta["total_energy"].keys())
    for i, at in enumerate(atoms):
        sid += 1
        records.append(_make_record(sid, at, meta, json_keys[i], "sp_test_set"))

    # --- al_data: 4230 ---
    atoms = read(str(base_dir / "al_data.xyz"), ":")
    with open(base_dir / "al_data.json") as f:
        meta = json.load(f)
    json_keys = list(meta["total_energy"].keys())
    for i, at in enumerate(atoms):
        sid += 1
        records.append(_make_record(sid, at, meta, json_keys[i], "al_data"))

    # --- relax_test_set: 100 initial + 100 relaxed ---
    atoms_init = read(str(base_dir / "relax_test_set_initial.xyz"), ":")
    atoms_rel = read(str(base_dir / "relax_test_set_relaxed.xyz"), ":")
    with open(base_dir / "relax_test_set.json") as f:
        meta = json.load(f)
    json_keys = list(meta["total_energy"].keys())
    for i, at in enumerate(atoms_init):
        sid += 1
        records.append(_make_record(sid, at, meta, json_keys[i], "relax_test_set_initial"))
    for i, at in enumerate(atoms_rel):
        sid += 1
        records.append(_make_record(sid, at, meta, json_keys[i], "relax_test_set_relaxed"))

    return DatasetBundle(
        dataset_name="CsSnClBrI3_14056015",
        records=tuple(records),
    )


def parse_reference_ids_from_exports(export_dir: Path) -> dict[str, set[int]]:
    """Parse exported reference XYZ filenames to recover sample IDs."""
    refs: dict[str, set[int]] = {
        "primary": set(),
        "stable": set(),
        "diversity": set(),
    }
    if not export_dir.exists():
        return refs

    for path in export_dir.iterdir():
        if not path.suffix == ".xyz":
            continue
        name = path.stem
        if name.startswith("reference_"):
            sid_str = name.replace("reference_", "")
            refs["primary"].add(int(sid_str))
        elif name.startswith("stable_ref_"):
            parts = name.split("_")
            refs["stable"].add(int(parts[-1]))
        elif name.startswith("diversity_ref_"):
            parts = name.split("_")
            refs["diversity"].add(int(parts[-1]))

    return refs


def analyze_references(
    profile,
    dataset: DatasetBundle,
    ref_ids: dict[str, set[int]],
    output_dir: Path,
    base_dir: Path,
) -> None:
    """Cross-reference selected structures with metadata and write report."""
    record_by_id = {r.sample_id: r for r in dataset.records}

    # Build unified sets
    all_ref_ids = ref_ids["primary"] | ref_ids["stable"] | ref_ids["diversity"]
    stable_ref_ids = ref_ids["stable"]
    diversity_ref_ids = ref_ids["diversity"]

    categories = {
        "all_references": all_ref_ids,
        "primary_reference": ref_ids["primary"],
        "stable_references": stable_ref_ids,
        "diversity_references": diversity_ref_ids,
    }

    # ------------------------------------------------------------------
    # 1. Source distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. REFERENCE SOURCE DISTRIBUTION")
    print("=" * 60)
    source_dist: dict[str, dict[str, int]] = {}
    for cat_name, ids in categories.items():
        counts: dict[str, int] = {}
        for sid in ids:
            rec = record_by_id.get(sid)
            if rec is None:
                continue
            src = str(rec.metadata.get("source", "unknown"))
            counts[src] = counts.get(src, 0) + 1
        source_dist[cat_name] = counts
        print(f"\n{cat_name} (n={len(ids)}):")
        for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {src:30s}: {cnt}")

    # ------------------------------------------------------------------
    # 2. Space-group coverage
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. SPACE-GROUP (LATTICE TYPE) COVERAGE")
    print("=" * 60)
    sg_dist: dict[str, dict[str, int]] = {}
    for cat_name, ids in categories.items():
        counts: dict[str, int] = {}
        for sid in ids:
            rec = record_by_id.get(sid)
            if rec is None:
                continue
            lt = str(rec.metadata.get("lattice_type", "unknown"))
            counts[lt] = counts.get(lt, 0) + 1
        sg_dist[cat_name] = counts
        print(f"\n{cat_name} (n={len(ids)}):")
        for lt, cnt in sorted(counts.items()):
            print(f"  {lt:10s}: {cnt}")

    # ------------------------------------------------------------------
    # 3. Relaxation snapshot hit rate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. RELAXATION SNAPSHOT HIT RATE")
    print("=" * 60)

    relax_sources = {"al_data", "relax_test_set_initial", "relax_test_set_relaxed"}
    relax_ids = {
        r.sample_id for r in dataset.records
        if str(r.metadata.get("source", "")) in relax_sources
    }

    # Sub-categories within relaxation data
    al_data_ids = {r.sample_id for r in dataset.records if r.metadata.get("source") == "al_data"}
    relax_init_ids = {r.sample_id for r in dataset.records if r.metadata.get("source") == "relax_test_set_initial"}
    relax_rel_ids = {r.sample_id for r in dataset.records if r.metadata.get("source") == "relax_test_set_relaxed"}

    # Relaxed vs unrelaxed within al_data
    al_unrelaxed = {
        r.sample_id for r in dataset.records
        if r.metadata.get("source") == "al_data" and r.metadata.get("relaxed") is False
    }
    al_relaxed = {
        r.sample_id for r in dataset.records
        if r.metadata.get("source") == "al_data" and r.metadata.get("relaxed") is True
    }

    print(f"\nTotal relaxation-related structures: {len(relax_ids)}")
    print(f"  al_data total              : {len(al_data_ids)}")
    print(f"  al_data unrelaxed          : {len(al_unrelaxed)}")
    print(f"  al_data relaxed            : {len(al_relaxed)}")
    print(f"  relax_test_set_initial     : {len(relax_init_ids)}")
    print(f"  relax_test_set_relaxed     : {len(relax_rel_ids)}")

    print("\nHit rates by category:")
    for label, pool in [
        ("all relaxation-related", relax_ids),
        ("al_data total", al_data_ids),
        ("al_data unrelaxed", al_unrelaxed),
        ("al_data relaxed", al_relaxed),
        ("relax_test_set_initial", relax_init_ids),
        ("relax_test_set_relaxed", relax_rel_ids),
    ]:
        for cat_name, ids in categories.items():
            hit = ids & pool
            pct = 100.0 * len(hit) / len(pool) if pool else 0.0
            print(f"  {cat_name:25s} vs {label:25s}: {len(hit):4d} / {len(pool):4d} ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # 4. Non-equilibrium (high-force / high-energy) hit rate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. NON-EQUILIBRIUM STRUCTURE HIT RATE")
    print("=" * 60)

    # Collect per-structure metrics
    energies_per_atom: dict[int, float] = {}
    max_forces: dict[int, float] = {}
    mean_forces: dict[int, float] = {}

    min_energy_pa = float("inf")
    for r in dataset.records:
        if r.energy is not None and r.natoms > 0:
            epa = r.energy / r.natoms
            energies_per_atom[r.sample_id] = epa
            if epa < min_energy_pa:
                min_energy_pa = epa
        max_forces[r.sample_id] = _max_force_norm(r)
        mean_forces[r.sample_id] = _mean_force_norm(r)

    # Define non-equilibrium sets
    ne_sets: dict[str, set[int]] = {}

    # High max force
    ne_sets["max_force_gt_1.0"] = {sid for sid, f in max_forces.items() if f > 1.0}
    ne_sets["max_force_gt_0.5"] = {sid for sid, f in max_forces.items() if f > 0.5}
    ne_sets["max_force_gt_0.1"] = {sid for sid, f in max_forces.items() if f > 0.1}

    # High mean force
    ne_sets["mean_force_gt_0.5"] = {sid for sid, f in mean_forces.items() if f > 0.5}

    # High energy above minimum
    ne_sets["delta_e_pa_gt_0.05"] = {
        sid for sid, epa in energies_per_atom.items()
        if epa - min_energy_pa > 0.05
    }
    ne_sets["delta_e_pa_gt_0.01"] = {
        sid for sid, epa in energies_per_atom.items()
        if epa - min_energy_pa > 0.01
    }

    # Early relaxation steps (al_data only)
    early_relax = {
        r.sample_id for r in dataset.records
        if r.metadata.get("source") == "al_data"
        and isinstance(r.metadata.get("relaxation_iteration"), int)
        and r.metadata["relaxation_iteration"] <= 2
    }
    ne_sets["al_early_relaxation"] = early_relax

    print(f"\nNon-equilibrium set sizes:")
    for label, ids in sorted(ne_sets.items()):
        print(f"  {label:30s}: {len(ids):5d}")

    print("\nHit rates (reference hits / non-equilibrium set):")
    for ne_label, ne_ids in sorted(ne_sets.items()):
        for cat_name, ids in categories.items():
            hit = ids & ne_ids
            pct = 100.0 * len(hit) / len(ne_ids) if ne_ids else 0.0
            print(f"  {cat_name:25s} vs {ne_label:30s}: {len(hit):4d} / {len(ne_ids):5d} ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # 5. Cross-analysis: space-group × source for references
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. REFERENCE CROSS-TABULATION (space-group × source)")
    print("=" * 60)

    crosstab: dict[str, dict[str, dict[str, int]]] = {}
    for cat_name, ids in categories.items():
        table: dict[str, dict[str, int]] = {}
        for sid in ids:
            rec = record_by_id.get(sid)
            if rec is None:
                continue
            lt = str(rec.metadata.get("lattice_type", "unknown"))
            src = str(rec.metadata.get("source", "unknown"))
            table.setdefault(lt, {})
            table[lt][src] = table[lt].get(src, 0) + 1
        crosstab[cat_name] = table
        print(f"\n{cat_name}:")
        for lt in sorted(table.keys()):
            print(f"  {lt}:")
            for src, cnt in sorted(table[lt].items(), key=lambda x: -x[1]):
                print(f"    {src:30s}: {cnt}")

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "dataset_name": dataset.dataset_name,
        "total_structures": len(dataset.records),
        "reference_counts": {k: len(v) for k, v in categories.items()},
        "source_distribution": source_dist,
        "space_group_distribution": sg_dist,
        "relaxation_hits": {},
        "non_equilibrium_hits": {},
        "crosstab": crosstab,
    }

    # Populate relaxation hits
    for label, pool in [
        ("all_relaxation", relax_ids),
        ("al_data", al_data_ids),
        ("al_unrelaxed", al_unrelaxed),
        ("al_relaxed", al_relaxed),
        ("relax_test_initial", relax_init_ids),
        ("relax_test_relaxed", relax_rel_ids),
    ]:
        report["relaxation_hits"][label] = {
            cat: len(ids & pool) for cat, ids in categories.items()
        }
        report["relaxation_hits"][label]["total"] = len(pool)

    # Populate non-equilibrium hits
    for ne_label, ne_ids in ne_sets.items():
        report["non_equilibrium_hits"][ne_label] = {
            cat: len(ids & ne_ids) for cat, ids in categories.items()
        }
        report["non_equilibrium_hits"][ne_label]["total"] = len(ne_ids)

    # ------------------------------------------------------------------
    # 6. Export hit structures for material analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. EXPORTING HIT STRUCTURES FOR MATERIAL ANALYSIS")
    print("=" * 60)

    hit_dir = output_dir / "hit_structures"
    hit_dir.mkdir(parents=True, exist_ok=True)

    from ase.io import read, write
    import numpy as np

    # Load all source structures into memory (indexed by sample_id)
    all_atoms: dict[int, object] = {}
    all_metas: dict[int, dict[str, Any]] = {}

    def _load_subset(bd: Path, xyz_name: str, json_name: str, sid_start: int, meta_keys: list[str]) -> int:
        atoms = read(str(bd / xyz_name), ":")
        with open(bd / json_name) as f:
            meta = json.load(f)
        keys = list(meta["total_energy"].keys())
        for i, at in enumerate(atoms):
            sid = sid_start + i
            all_atoms[sid] = at
            m: dict[str, Any] = {"source": xyz_name.replace(".xyz", "")}
            for k in meta_keys:
                if k in meta and keys[i] in meta[k]:
                    m[k] = meta[k][keys[i]]
            all_metas[sid] = m
        return len(atoms)

    n1 = _load_subset(base_dir, "sp_train_set.xyz", "sp_train_set.json", 1,
                       ["lattice_type", "n_cl", "n_br", "n_i", "batch"])
    n2 = _load_subset(base_dir, "sp_test_set.xyz", "sp_test_set.json", 1 + n1,
                       ["lattice_type", "n_cl", "n_br", "n_i"])
    n3 = _load_subset(base_dir, "al_data.xyz", "al_data.json", 1 + n1 + n2,
                       ["lattice_type", "n_cl", "n_br", "n_i", "relaxed", "relaxation_iteration", "al_iteration"])
    _load_subset(base_dir, "relax_test_set_initial.xyz", "relax_test_set.json", 1 + n1 + n2 + n3,
                 ["lattice_type", "n_cl", "n_br", "n_i"])
    _load_subset(base_dir, "relax_test_set_relaxed.xyz", "relax_test_set.json", 1 + n1 + n2 + n3 + 100,
                 ["lattice_type", "n_cl", "n_br", "n_i"])

    # Define export groups
    export_groups: dict[str, set[int]] = {}

    # By reference category + relaxation status
    relax_sources = {"al_data", "relax_test_set_initial", "relax_test_set_relaxed"}
    for cat_name, ids in categories.items():
        for src in relax_sources:
            sids = {sid for sid in ids if record_by_id.get(sid, {}).metadata.get("source") == src}
            if sids:
                export_groups[f"{cat_name}__{src}"] = sids

    # Also export by combined relaxation status
    al_relaxed_hit = {sid for sid in all_ref_ids
                      if record_by_id.get(sid, {}).metadata.get("source") == "al_data"
                      and record_by_id.get(sid, {}).metadata.get("relaxed") is True}
    al_unrelaxed_hit = {sid for sid in all_ref_ids
                        if record_by_id.get(sid, {}).metadata.get("source") == "al_data"
                        and record_by_id.get(sid, {}).metadata.get("relaxed") is False}
    rt_init_hit = {sid for sid in all_ref_ids
                   if record_by_id.get(sid, {}).metadata.get("source") == "relax_test_set_initial"}
    rt_rel_hit = {sid for sid in all_ref_ids
                  if record_by_id.get(sid, {}).metadata.get("source") == "relax_test_set_relaxed"}

    if al_relaxed_hit:
        export_groups["al_data_relaxed__all_refs"] = al_relaxed_hit
    if al_unrelaxed_hit:
        export_groups["al_data_unrelaxed__all_refs"] = al_unrelaxed_hit
    if rt_init_hit:
        export_groups["relax_test_set_initial__all_refs"] = rt_init_hit
    if rt_rel_hit:
        export_groups["relax_test_set_relaxed__all_refs"] = rt_rel_hit

    # Write XYZ files with metadata
    for group_name, sids in sorted(export_groups.items()):
        atoms_list = []
        for sid in sorted(sids):
            at = all_atoms.get(sid)
            if at is None:
                continue
            m = all_metas.get(sid, {})
            info = at.info.copy() if hasattr(at, "info") else {}
            info["sample_id"] = sid
            info["source"] = m.get("source", "unknown")
            info["lattice_type"] = m.get("lattice_type", "unknown")
            info["relaxed"] = str(m.get("relaxed", "N/A"))
            info["relaxation_iteration"] = str(m.get("relaxation_iteration", "N/A"))
            at.info = info
            atoms_list.append(at)

        if atoms_list:
            out_path = hit_dir / f"{group_name}.xyz"
            write(str(out_path), atoms_list, format="extxyz")
            print(f"  {group_name}: {len(atoms_list)} structures -> {out_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reference_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Report saved to: {report_path}")
    print(f"Exported refs  : {output_dir / 'exported_refs'}")
    print(f"Hit structures : {hit_dir}")


def main() -> None:
    base_dir = Path("14056015")
    output_dir = Path("outputs/reference_analysis")
    export_dir = output_dir / "exported_refs"

    print("Loading dataset from 14056015 ...")
    dataset = load_dataset(base_dir)
    print(f"  Total structures: {len(dataset.records)}")

    # Quick metadata sanity check
    sources: dict[str, int] = {}
    lts: dict[str, int] = {}
    for r in dataset.records:
        src = str(r.metadata.get("source", "unknown"))
        sources[src] = sources.get(src, 0) + 1
        lt = str(r.metadata.get("lattice_type", "unknown"))
        lts[lt] = lts.get(lt, 0) + 1
    print(f"  Source breakdown: {sources}")
    print(f"  Lattice types   : {dict(sorted(lts.items()))}")

    print("\nRunning DatasetProfiler (this may take a few minutes) ...")
    profiler = DatasetProfiler(
        registry=build_default_registry(),
        export_references_dir=export_dir,
    )
    profile = profiler.profile(dataset)
    print(f"  Profile complete: {profile.dataset_name}")
    print(f"  Pools           : {profile.summary.get('pool_count')}")
    print(f"  Stability score : {profile.summary.get('stability_score')}")
    print(f"  Compositional diversity score : {profile.summary.get('compositional_diversity_score')}")
    print(f"  Structural diversity score    : {profile.summary.get('structural_diversity_score')}")

    print("\nParsing exported reference structures ...")
    ref_ids = parse_reference_ids_from_exports(export_dir)
    print(f"  Primary refs   : {len(ref_ids['primary'])}")
    print(f"  Stable refs    : {len(ref_ids['stable'])}")
    print(f"  Diversity refs : {len(ref_ids['diversity'])}")

    analyze_references(profile, dataset, ref_ids, output_dir, base_dir)


if __name__ == "__main__":
    main()
