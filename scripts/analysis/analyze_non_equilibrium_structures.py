#!/usr/bin/env python3
"""Fine-grained analysis of non-equilibrium structure hit rates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ase.io import read


def _max_force_norm(atoms) -> float:
    forces = atoms.get_forces() if atoms.calc is not None else atoms.arrays.get("forces", None)
    if forces is None:
        return 0.0
    return max(math.sqrt(sum(c * c for c in f)) for f in forces)


def parse_reference_ids(export_dir: Path) -> dict[str, set[int]]:
    refs: dict[str, set[int]] = {"primary": set(), "stable": set(), "diversity": set()}
    if not export_dir.exists():
        return refs
    for path in export_dir.iterdir():
        if path.suffix != ".xyz":
            continue
        name = path.stem
        if name.startswith("reference_"):
            refs["primary"].add(int(name.replace("reference_", "")))
        elif name.startswith("stable_ref_"):
            refs["stable"].add(int(name.split("_")[-1]))
        elif name.startswith("diversity_ref_"):
            refs["diversity"].add(int(name.split("_")[-1]))
    return refs


def main() -> None:
    base_dir = Path("14056015")
    export_dir = Path("outputs/reference_analysis/exported_refs")
    output_dir = Path("outputs/reference_analysis")

    ref_ids = parse_reference_ids(export_dir)
    all_ref_ids = ref_ids["primary"] | ref_ids["stable"] | ref_ids["diversity"]
    categories = {
        "all_references": all_ref_ids,
        "primary_reference": ref_ids["primary"],
        "stable_references": ref_ids["stable"],
        "diversity_references": ref_ids["diversity"],
    }

    atoms_init = read(str(base_dir / "relax_test_set_initial.xyz"), ":")
    atoms_rel = read(str(base_dir / "relax_test_set_relaxed.xyz"), ":")
    with open(base_dir / "relax_test_set.json") as f:
        rt_meta = json.load(f)

    RT_INIT_START = 22831
    RT_REL_START = 22931

    print("=" * 70)
    print("A. FAR-FROM-EQUILIBRIUM INITIAL STRUCTURES (relax_test_set_initial)")
    print("=" * 70)

    far_eq_structs: dict[str, list[dict[str, Any]]] = {}

    tier_defs = [
        ("strict (delta>0.5 eV AND force>10 eV/A)", lambda d, f: d > 0.5 and f > 10.0),
        ("moderate (delta>0.2 eV AND force>5 eV/A)", lambda d, f: d > 0.2 and f > 5.0),
        ("lenient (delta>0.05 eV)", lambda d, f: d > 0.05),
    ]

    for tier_name, tier_fn in tier_defs:
        structs = []
        for i in range(len(atoms_init)):
            delta_e = atoms_init[i].get_potential_energy() - atoms_rel[i].get_potential_energy()
            max_f = _max_force_norm(atoms_init[i])
            if tier_fn(delta_e, max_f):
                structs.append({
                    "idx": i,
                    "sid": RT_INIT_START + i,
                    "lt": rt_meta["lattice_type"][str(i)],
                    "delta_e": delta_e,
                    "max_f": max_f,
                })
        far_eq_structs[tier_name] = structs

    print("")
    print("Total relax_test_set_initial: " + str(len(atoms_init)))
    for tier_name, structs in far_eq_structs.items():
        sids = {s["sid"] for s in structs}
        print("")
        print("  Tier: " + tier_name + " -> " + str(len(structs)) + " structures")
        for cat_name, ids in categories.items():
            hit = ids & sids
            pct = 100.0 * len(hit) / len(sids) if sids else 0.0
            print("    " + cat_name + ": " + str(len(hit)) + " / " + str(len(sids)) + " (" + f"{pct:.1f}" + "%)")

    print("")
    print("  Top 10 initial structures by energy drop:")
    all_init = []
    for i in range(len(atoms_init)):
        delta_e = atoms_init[i].get_potential_energy() - atoms_rel[i].get_potential_energy()
        max_f = _max_force_norm(atoms_init[i])
        in_strict = any(s["idx"] == i for s in far_eq_structs["strict (delta>0.5 eV AND force>10 eV/A)"])
        all_init.append({"idx": i, "delta_e": delta_e, "max_f": max_f, "lt": rt_meta["lattice_type"][str(i)], "in_strict": in_strict})
    all_init.sort(key=lambda x: -x["delta_e"])
    for s in all_init[:10]:
        marker = " *" if s["in_strict"] else "  "
        print("    " + marker + " idx=" + str(s["idx"]).rjust(3) + " " + s["lt"].ljust(6) + " delta=" + f"{s['delta_e']:.4f}" + " max_f=" + f"{s['max_f']:.2f}")

    print("")
    print("=" * 70)
    print("B. FAILED-RELAXATION / METASTABLE STRUCTURES (relax_test_set_relaxed)")
    print("=" * 70)

    failed_structs: dict[str, list[dict[str, Any]]] = {}

    failed_defs = [
        ("strict (residual force > 0.05 eV/A)", lambda f: f > 0.05),
        ("moderate (residual force > 0.1 eV/A)", lambda f: f > 0.1),
        ("severe (residual force > 0.5 eV/A)", lambda f: f > 0.5),
        ("extreme (residual force > 1.0 eV/A)", lambda f: f > 1.0),
    ]

    for tier_name, tier_fn in failed_defs:
        structs = []
        for i in range(len(atoms_rel)):
            max_f = _max_force_norm(atoms_rel[i])
            if tier_fn(max_f):
                structs.append({
                    "idx": i,
                    "sid": RT_REL_START + i,
                    "lt": rt_meta["lattice_type"][str(i)],
                    "max_f": max_f,
                    "delta_e": atoms_init[i].get_potential_energy() - atoms_rel[i].get_potential_energy(),
                })
        failed_structs[tier_name] = structs

    print("")
    print("Total relax_test_set_relaxed: " + str(len(atoms_rel)))
    for tier_name, structs in failed_structs.items():
        sids = {s["sid"] for s in structs}
        print("")
        print("  Tier: " + tier_name + " -> " + str(len(structs)) + " structures")
        for cat_name, ids in categories.items():
            hit = ids & sids
            pct = 100.0 * len(hit) / len(sids) if sids else 0.0
            print("    " + cat_name + ": " + str(len(hit)) + " / " + str(len(sids)) + " (" + f"{pct:.1f}" + "%)")

    print("")
    print("  Top 10 relaxed structures by residual force:")
    all_rel = []
    for i in range(len(atoms_rel)):
        max_f = _max_force_norm(atoms_rel[i])
        delta_e = atoms_init[i].get_potential_energy() - atoms_rel[i].get_potential_energy()
        all_rel.append({"idx": i, "max_f": max_f, "delta_e": delta_e, "lt": rt_meta["lattice_type"][str(i)]})
    all_rel.sort(key=lambda x: -x["max_f"])
    for s in all_rel[:10]:
        print("      idx=" + str(s["idx"]).rjust(3) + " " + s["lt"].ljust(6) + " residual_f=" + f"{s['max_f']:6.2f}" + " delta_e=" + f"{s['delta_e']:.4f}")

    print("")
    print("=" * 70)
    print("C. AL_DATA RELAXATION SNAPSHOTS")
    print("=" * 70)

    atoms_al = read(str(base_dir / "al_data.xyz"), ":")
    with open(base_dir / "al_data.json") as f:
        al_meta = json.load(f)

    AL_START = 18601

    al_unrelaxed = []
    al_relaxed_high_force = []
    al_relaxed_low_force = []

    keys = list(al_meta["total_energy"].keys())
    for k in keys:
        idx = keys.index(k)
        sid = AL_START + idx
        relaxed = al_meta.get("relaxed", {}).get(k, False)
        max_f = _max_force_norm(atoms_al[idx])
        if not relaxed:
            al_unrelaxed.append({"sid": sid, "max_f": max_f})
        else:
            if max_f > 0.05:
                al_relaxed_high_force.append({"sid": sid, "max_f": max_f})
            else:
                al_relaxed_low_force.append({"sid": sid, "max_f": max_f})

    al_groups = {
        "al_unrelaxed": al_unrelaxed,
        "al_relaxed_residual>0.05": al_relaxed_high_force,
        "al_relaxed_residual<=0.05": al_relaxed_low_force,
    }

    for group_name, group in al_groups.items():
        sids = {s["sid"] for s in group}
        print("")
        print("  " + group_name + ": " + str(len(group)) + " structures")
        for cat_name, ids in categories.items():
            hit = ids & sids
            pct = 100.0 * len(hit) / len(sids) if sids else 0.0
            print("    " + cat_name + ": " + str(len(hit)) + " / " + str(len(sids)) + " (" + f"{pct:.1f}" + "%)")

    report = {
        "far_from_equilibrium_initial": {},
        "failed_relaxation_metastable": {},
        "al_data_groups": {},
    }

    for k, v in far_eq_structs.items():
        sids = {s["sid"] for s in v}
        report["far_from_equilibrium_initial"][k] = {
            "total": len(sids),
            "hits": {cat: len(ids & sids) for cat, ids in categories.items()},
        }

    for k, v in failed_structs.items():
        sids = {s["sid"] for s in v}
        report["failed_relaxation_metastable"][k] = {
            "total": len(sids),
            "hits": {cat: len(ids & sids) for cat, ids in categories.items()},
        }

    for k, v in al_groups.items():
        sids = {s["sid"] for s in v}
        report["al_data_groups"][k] = {
            "total": len(sids),
            "hits": {cat: len(ids & sids) for cat, ids in categories.items()},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "non_equilibrium_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("")
    print("=" * 70)
    print("Report saved to: " + str(output_dir / "non_equilibrium_analysis.json"))


if __name__ == "__main__":
    main()
