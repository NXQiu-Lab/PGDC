#!/usr/bin/env python3
"""
Recover failed annealing jobs for a specific AL iteration.

This script:
  1. Reads remote_manifest.json to find the target iteration.
  2. For each annealing chain, SSH-checks whether the remote relax phase
     successfully produced *_relax_geo-RESTART.wfn.
  3. For chains where relax succeeded:
     - Cancels old SLURM jobs for heating/high_temp/quench/hold (if still queued).
     - Regenerates correct input.inp / submit.sh locally using the FIXED logic.
     - Uploads to the remote cluster (overwrites bad inputs).
     - Resubmits the chain from heating onwards.
  4. For chains where relax failed or is not finished: reports and skips.

Usage:
    uv run python scripts/sio2/recover_annealing_iteration.py \
        --remote-config scripts/sio2/remote_config.json \
        --iteration 2

    uv run python scripts/sio2/recover_annealing_iteration.py \
        --remote-config scripts/sio2/remote_config.json \
        --iteration 2 \
        --max-workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup (mirrors build_sio2_dataset_remote.py)
# ---------------------------------------------------------------------------
_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "sio2"))

from ase.io import read

from common.remote_slurm_utils import RemoteConfig, RemoteSLURMClient, load_remote_config
from sio2_utils import build_cp2k_input_sio2


# ---------------------------------------------------------------------------
# Helpers copied from build_sio2_dataset_remote.py (exact same logic)
# ---------------------------------------------------------------------------

def _write_submit_script(
    run_dir: Path,
    job_name: str,
    nodes: int,
    ppn: int,
    nprocs: int,
    queue: str,
    with_geo_opt: bool = True,
    geo_restart_file: str = "",
) -> None:
    env_block = (
        "source /opt/software/cp2k/tools/toolchain/install/setup\n"
        "export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH\n"
        "export PMIX_MCA_psec=native\n"
    )
    if with_geo_opt:
        check_block = (
            f'if [ ! -f "./{geo_restart_file}" ]; then\n'
            f'    echo "ERROR: GEO_OPT did not produce {geo_restart_file}; aborting."\n'
            f'    exit 1\n'
            f'fi\n'
        ) if geo_restart_file else ""
        run_block = (
            'echo "=== Step 1: GEO_OPT ==="\n'
            "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./geo_opt.inp\n"
            + check_block
            + "\n"
            'echo "=== Step 2: MD ==="\n'
            "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./input.inp\n"
        )
    else:
        run_block = "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./input.inp\n"

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --nodes={nodes}\n"
        f"#SBATCH --ntasks-per-node={ppn}\n"
        f"#SBATCH --partition={queue}\n"
        "#SBATCH --output=out\n"
        "#SBATCH --error=err\n\n"
        "# Handle path difference between head (/home/work) and compute nodes (/work)\n"
        'SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"\n'
        'HEAD_DIR="$SUBMIT_DIR"\n'
        'NODE_DIR="${SUBMIT_DIR/\\/home\\/work/\\/work}"\n'
        'if [ -d "$HEAD_DIR" ]; then\n'
        '    cd "$HEAD_DIR"\n'
        'elif [ -d "$NODE_DIR" ]; then\n'
        '    cd "$NODE_DIR"\n'
        "else\n"
        '    echo "ERROR: Cannot find work directory"\n'
        "    exit 1\n"
        "fi\n\n"
        + env_block
        + "\n"
        + run_block
    )
    (run_dir / "submit.slurm").write_text(script)


# ---------------------------------------------------------------------------
# Remote status checker
# ---------------------------------------------------------------------------

def _remote_relax_ok(client: RemoteSLURMClient, remote_relax_dir: str, base_name: str) -> bool:
    """Return True if the remote relax dir has both required restart files."""
    wfn = f"{remote_relax_dir}/{base_name}_relax_geo-RESTART.wfn"
    rst = f"{remote_relax_dir}/{base_name}_relax-1.restart"
    result = client._ssh(f"test -f {wfn} && test -f {rst} && echo OK || echo MISSING")
    return "OK" in result.stdout


def _cancel_old_jobs(client: RemoteSLURMClient, job_ids: list[str]) -> None:
    """Best-effort scancel for a list of job IDs. Ignores 'Invalid job id' errors."""
    for jid in job_ids:
        if not jid:
            continue
        result = client._ssh(f"scancel {jid} 2>&1 || true")
        # Silently ignore errors — job may have already finished.


# ---------------------------------------------------------------------------
# Input regeneration
# ---------------------------------------------------------------------------

def regenerate_chain_inputs(
    base_name: str,
    poscar_path: Path,
    build_dir: Path,
    iteration: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Regenerate correct local inputs for a 5-phase annealing chain.

    Returns a list of phase dicts (same fields as the original manifest).
    """
    atoms = read(str(poscar_path))
    atoms.set_cell(atoms.get_cell(), scale_atoms=True)

    # Config defaults
    cutoff = config.get("cp2k_cutoff", 500)
    basis_set = config.get("cp2k_basis_set", "DZVP-MOLOPT-SR-GTH")
    cp2k_data_dir = config.get("cp2k_data_dir", None)
    timestep = config.get("seed_timestep", 1.0)
    output_stride = config.get("al_output_stride", 20)
    nodes = config.get("cp2k_nodes", 1)
    ppn = config.get("cp2k_ppn", 64)
    nprocs = config.get("cp2k_nprocs", 64)
    queue = config.get("cp2k_queue", "compute")

    # Annealing parameters
    phases_def = [
        ("relax", 300.0, config.get("al_annealing_relax_steps", 2000), None, None),
        ("heating", config.get("al_annealing_heating_temp", 3000.0), config.get("al_annealing_heating_steps", 3000),
         f"../relax/{base_name}_relax_geo-RESTART.wfn",
         f"../relax/{base_name}_relax-1.restart"),
        ("high_temp", config.get("al_annealing_peak_temp", 3500.0), config.get("al_annealing_high_temp_steps", 2000),
         f"../heating/{base_name}_heating-RESTART.wfn",
         f"../heating/{base_name}_heating-1.restart"),
        ("quench", 300.0, config.get("al_annealing_quench_steps", 3000),
         f"../high_temp/{base_name}_high_temp-RESTART.wfn",
         f"../high_temp/{base_name}_high_temp-1.restart"),
        ("hold", 300.0, config.get("al_annealing_hold_steps", 5000),
         f"../quench/{base_name}_quench-RESTART.wfn",
         f"../quench/{base_name}_quench-1.restart"),
    ]

    phase_results: list[dict[str, Any]] = []

    for phase_name, temp, n_steps, wfn_restart, ext_restart in phases_def:
        run_name = f"{base_name}_{phase_name}"
        run_dir = (
            build_dir
            / "phase_1_al_loop"
            / f"iter_{iteration:03d}"
            / "selected"
            / base_name
            / phase_name
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        remote_subdir = (
            "phase_1_al_loop",
            f"iter_{iteration:03d}",
            "selected",
            base_name,
            phase_name,
        )

        need_geo_opt = wfn_restart is None

        if need_geo_opt:
            geo_project = f"{run_name}_geo"
            geo_inp = build_cp2k_input_sio2(
                atoms=atoms,
                project_name=geo_project,
                temperature=temp,
                n_steps=n_steps,
                timestep=timestep,
                output_stride=output_stride,
                cutoff=cutoff,
                basis_set=basis_set,
                scf_guess="ATOMIC",
                electronic_temperature=None,
                cp2k_data_dir=cp2k_data_dir,
                run_type="GEO_OPT",
            )
            with (run_dir / "geo_opt.inp").open("w") as f:
                f.write(geo_inp)
            md_ext_restart = f"{geo_project}-1.restart"
            wfn_restart_file = f"{geo_project}-RESTART.wfn"
        else:
            wfn_restart_file = wfn_restart
            md_ext_restart = ext_restart

        inp_content = build_cp2k_input_sio2(
            atoms=atoms,
            project_name=run_name,
            temperature=temp,
            n_steps=n_steps,
            timestep=timestep,
            output_stride=output_stride,
            cutoff=cutoff,
            basis_set=basis_set,
            scf_guess="RESTART",
            use_ot=True,
            wfn_restart_file=wfn_restart_file,
            ext_restart_file=md_ext_restart,
            cp2k_data_dir=cp2k_data_dir,
        )
        with (run_dir / "input.inp").open("w") as f:
            f.write(inp_content)

        _write_submit_script(
            run_dir,
            job_name=run_name,
            nodes=nodes,
            ppn=ppn,
            nprocs=nprocs,
            queue=queue,
            with_geo_opt=need_geo_opt,
            geo_restart_file=md_ext_restart if need_geo_opt else "",
        )

        phase_results.append({
            "phase_name": phase_name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "remote_subdir": remote_subdir,
            "temperature": temp,
            "need_geo_opt": need_geo_opt,
        })

    return phase_results


# ---------------------------------------------------------------------------
# Main recovery workflow
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover failed annealing chains for a given AL iteration."
    )
    parser.add_argument(
        "--remote-config",
        required=True,
        help="Path to remote server JSON config.",
    )
    parser.add_argument(
        "--build-dir",
        default="sio2_al_build",
        help="Root build directory (contains remote_manifest.json).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=2,
        help="AL iteration number to recover.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max concurrent upload+submit workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without uploading or submitting.",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    manifest_path = build_dir / "remote_manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    al_iters = manifest.get("al_iterations", [])
    target_iter = next((it for it in al_iters if it["iteration"] == args.iteration), None)
    if target_iter is None:
        print(f"ERROR: Iteration {args.iteration} not found in manifest.")
        sys.exit(1)

    # Hard-coded defaults (match BuilderConfig defaults used for iteration 2)
    config = {
        "al_annealing_relax_steps": 2000,
        "al_annealing_heating_steps": 3000,
        "al_annealing_heating_temp": 3000.0,
        "al_annealing_peak_temp": 3500.0,
        "al_annealing_high_temp_steps": 2000,
        "al_annealing_quench_steps": 3000,
        "al_annealing_hold_steps": 5000,
        "cp2k_cutoff": 500,
        "cp2k_basis_set": "DZVP-MOLOPT-SR-GTH",
        "cp2k_data_dir": "/opt/software/cp2k/data",
        "seed_timestep": 1.0,
        "al_output_stride": 20,
        "cp2k_nodes": 1,
        "cp2k_ppn": 64,
        "cp2k_nprocs": 64,
        "cp2k_queue": "compute",
    }

    remote_cfg = load_remote_config(args.remote_config)
    client = RemoteSLURMClient(remote_cfg)

    remote_work_dir = manifest.get("remote_work_dir", remote_cfg.remote_work_dir)
    iter_remote_base = f"{remote_work_dir}/phase_1_al_loop/iter_{args.iteration:03d}/selected"

    jobs = target_iter.get("jobs", [])
    chains_to_recover: list[dict[str, Any]] = []

    print(f"\nScanning iteration {args.iteration} ({len(jobs)} jobs)...")

    for job in jobs:
        if not job.get("annealing_chain"):
            continue

        base_name = job["base_name"]
        phases = job["phases"]
        relax_phase = phases[0]
        remote_relax_dir = f"{remote_work_dir}/{relax_phase['remote_dir']}"

        print(f"\n  {base_name}:")
        print(f"    Remote relax dir: {remote_relax_dir}")

        relax_ok = _remote_relax_ok(client, remote_relax_dir, base_name)
        if not relax_ok:
            print(f"    -> Relax NOT ready (missing restart files). SKIPPING.")
            continue

        print(f"    -> Relax SUCCESS.")

        # Locate init.xyz locally (copied to each phase dir during original submit)
        relax_local = build_dir / relax_phase["local_dir"]
        poscar_path = relax_local / "init.xyz"
        if not poscar_path.exists():
            for ph in phases[1:]:
                alt = build_dir / ph["local_dir"] / "init.xyz"
                if alt.exists():
                    poscar_path = alt
                    break

        if not poscar_path.exists():
            print(f"    -> WARNING: No local init.xyz found. SKIPPING.")
            continue

        # Gather old job IDs to cancel before resubmitting
        old_job_ids = [ph.get("job_id", "") for ph in phases[1:] if ph.get("job_id")]
        print(f"    -> Old jobs to cancel: {old_job_ids}")

        chains_to_recover.append({
            "base_name": base_name,
            "poscar": poscar_path,
            "old_job_ids": old_job_ids,
        })

    if not chains_to_recover:
        print("\nNo chains need recovery (all relax phases incomplete or no candidates).")
        client.close()
        return

    print(f"\n{'='*60}")
    print(f"Will recover {len(chains_to_recover)} annealing chains.")
    if args.dry_run:
        print("  [DRY-RUN] No actual upload/submit will happen.")
    print(f"{'='*60}\n")

    lock = threading.Lock()
    recovered: list[dict[str, Any]] = []

    def _recover_one(chain: dict[str, Any]) -> dict[str, Any] | None:
        base_name = chain["base_name"]
        poscar = chain["poscar"]
        old_ids = chain["old_job_ids"]

        try:
            # 1. Cancel old jobs (best effort)
            if not args.dry_run and old_ids:
                _cancel_old_jobs(client, old_ids)
                time.sleep(0.5)

            # 2. Regenerate all local inputs (including relax, for completeness)
            phase_results = regenerate_chain_inputs(
                base_name=base_name,
                poscar_path=poscar,
                build_dir=build_dir,
                iteration=args.iteration,
                config=config,
            )

            # 3. Upload + submit from heating onwards (skip index 0 = relax)
            new_jobs: list[dict[str, Any]] = []
            prev_job_id: str | None = None

            for phase in phase_results[1:]:
                run_dir = Path(phase["run_dir"])
                remote_subdir = phase["remote_subdir"]

                if not args.dry_run:
                    client.upload_run(run_dir, *remote_subdir)
                    job_id = client.submit_existing_script(
                        *remote_subdir,
                        depend_job_id=prev_job_id,
                    )
                else:
                    job_id = f"DRYRUN_{phase['phase_name']}"

                dep_str = f" (afterok:{prev_job_id})" if prev_job_id else ""
                print(f"  [{base_name}] {phase['phase_name']}: submitted {job_id}{dep_str}")

                new_jobs.append({
                    "phase": phase["phase_name"],
                    "job_id": job_id,
                    "depend_job_id": prev_job_id,
                })
                prev_job_id = job_id

            return {"base_name": base_name, "new_jobs": new_jobs}

        except Exception as exc:
            print(f"  ERROR recovering {base_name}: {exc}")
            import traceback
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_recover_one, ch): ch["base_name"] for ch in chains_to_recover}
        for future in as_completed(futures):
            base_name = futures[future]
            try:
                result = future.result()
                if result:
                    with lock:
                        recovered.append(result)
            except Exception as exc:
                print(f"  UNEXPECTED ERROR in {base_name}: {exc}")

    print(f"\n{'='*60}")
    print(f"Recovery complete: {len(recovered)}/{len(chains_to_recover)} chains resubmitted.")
    print(f"{'='*60}\n")

    for r in recovered:
        print(f"{r['base_name']}:")
        for nj in r["new_jobs"]:
            dep = f"  (depends on {nj['depend_job_id']})" if nj.get("depend_job_id") else "  (no dependency)"
            print(f"  {nj['phase']:9s}: {nj['job_id']}{dep}")

    print("\nNext steps:")
    print("  1. Monitor new jobs with:  squeue -u $(whoami)")
    print("  2. When they finish, run:  uv run python scripts/sio2/build_sio2_dataset_remote.py \\")
    print("                               --remote-config scripts/sio2/remote_config.json \\")
    print("                               al-collect")

    client.close()


if __name__ == "__main__":
    main()
