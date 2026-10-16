#!/usr/bin/env python3
"""
Add DFT stress tensors to an existing ASE database via remote CP2K single points.

This script follows the submit/collect pattern used by the amorphous carbon
remote dataset builder. It is designed for intermittent network connections:

  1. ``submit``  - generates CP2K ENERGY_FORCE inputs with STRESS_TENSOR
                   ANALYTICAL, uploads them to a remote SLURM cluster, submits
                   the jobs, and saves a local manifest. The local machine can
                   then go offline.
  2. ``collect`` - reconnects to the cluster, downloads finished outputs,
                   parses the stress tensor, writes it back to the ASE DB, and
                   cleans up remote/local scratch. Can be run repeatedly until
                   all jobs are done.
  3. ``status``  - prints current progress without touching remote files.
  4. ``auto``     - continuously collects finished jobs and submits new ones
                    until the database is complete. Useful when each single
                    point is short.

The script never modifies original CP2K inputs/outputs on the server. It only
adds labels to the local ASE DB and creates new scratch directories for the
single-point runs.

Relabel workflow (energy + forces + stress, server-driven queue)
----------------------------------------------------------------
Rows that already have labels can be recomputed with ``--force-recompute``
(e.g. the mlip4_al rows whose CP2K settings did not match the project
standard). Combine with ``--source mlip4_al`` to select rows by keypoint.

Use ``submit --prepare-only`` to generate inputs and SLURM scripts and upload
them without submitting; then start the server-side driver inside tmux so the
queue stays filled even when the local machine is offline::

    uv run python scripts/amorphous_carbon/compute_stress_for_db.py submit \
        --db ... --remote-config ... --work-dir ... \
        --source mlip4_al --force-recompute --prepare-only \
        --job-prefix relabel_sp
    ssh <username>@<host> 'tmux new-session -d -s relabel \
        "bash ~/slurm_queue_driver.sh <remote_work_dir>/relabel 16 60"'

``collect`` then reads the per-directory ``state`` file written by the driver
(``submitted <jobid>`` / ``done`` / ``failed ...``), downloads finished
outputs, parses energy/forces/stress, audits units, and writes all three
labels back to the DB (reserved columns and ``row.data`` copies).

CP2K input generation reuses the project-specific carbon templates in
``amorphous_carbon_utils.py``. The placement of ``STRESS_TENSOR ANALYTICAL``
under ``&FORCE_EVAL`` and ``RUN_TYPE ENERGY_FORCE`` in ``&GLOBAL`` follows the
CP2K keyword hierarchy follows the CP2K input reference.

Example workflow
----------------

Submit up to 1000 stress single points in batches of 100, using 16 cores
per job and keeping at most 4 jobs in the SLURM queue at once::

    uv run python scripts/amorphous_carbon/compute_stress_for_db.py submit \
        --db amorphous_carbon_al_build/amorphous_carbon_dataset_final_10000.db \
        --remote-config scripts/amorphous_carbon/remote_config.json \
        --work-dir amorphous_carbon_al_build/stress_single_points \
        --batch-size 100 \
        --max-jobs 1000 \
        --ntasks-per-node 16 \
        --max-concurrent 4

The local machine can be shut down immediately after ``submit`` returns: jobs
run on the remote SLURM cluster and their state is persisted in the local
manifest. To continue submitting more jobs once slots free up, simply re-run
the same ``submit`` command later.

Later (can be run from a different session/machine)::

    uv run python scripts/amorphous_carbon/compute_stress_for_db.py collect \
        --db amorphous_carbon_al_build/amorphous_carbon_dataset_final_10000.db \
        --remote-config scripts/amorphous_carbon/remote_config.json \
        --work-dir amorphous_carbon_al_build/stress_single_points

Run ``collect`` periodically until it reports that everything is done, or
use the ``auto`` command to let the script manage the queue continuously::

    uv run python scripts/amorphous_carbon/compute_stress_for_db.py auto \
        --db amorphous_carbon_al_build/amorphous_carbon_dataset_final_10000.db \
        --remote-config scripts/amorphous_carbon/remote_config_local.json \
        --work-dir amorphous_carbon_al_build/stress_single_points \
        --ntasks-per-node 16 \
        --max-concurrent 4 \
        --poll-interval 30
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io import write

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "amorphous_carbon"))
sys.path.insert(0, str(_scripts_dir / "sio2"))
sys.path.insert(0, str(_scripts_dir.parent / "src"))

from amorphous_carbon_utils import build_cp2k_input_carbon
from common.cp2k_utils import HA_TO_EV, parse_cp2k_stress
from common.remote_slurm_utils import (
    REMOTE_SLURM_TEMPLATE,
    RemoteConfig,
    RemoteSLURMClient,
    load_remote_config,
)
from common.unit_audit import audit_frame_values
from sio2_utils import parse_cp2k_single_point_out

logger = logging.getLogger(__name__)

STRESS_UNIT = "eV/Angstrom/Angstrom/Angstrom"
GPa_TO_EV_PER_ANGSTROM3 = 1.0 / 160.2176567
HA_BOHR_TO_EV_ANG = 51.42208619083232


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    """State for a single stress single-point job."""

    row_id: int
    remote_dir: str
    job_id: str | None = None
    # pending | prepared (uploaded, driver submits) | submitted | running |
    # completed | failed
    status: str = "pending"
    error: str | None = None


@dataclass
class Manifest:
    """Persistent state for the stress computation workflow."""

    db_path: str
    remote_work_dir: str
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    n_completed: int = 0
    n_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "remote_work_dir": self.remote_work_dir,
            "jobs": {
                k: {
                    "row_id": v.row_id,
                    "remote_dir": v.remote_dir,
                    "job_id": v.job_id,
                    "status": v.status,
                    "error": v.error,
                }
                for k, v in self.jobs.items()
            },
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        manifest = cls(
            db_path=data["db_path"],
            remote_work_dir=data["remote_work_dir"],
            n_completed=data.get("n_completed", 0),
            n_failed=data.get("n_failed", 0),
        )
        for k, v in data.get("jobs", {}).items():
            manifest.jobs[k] = JobRecord(
                row_id=v["row_id"],
                remote_dir=v["remote_dir"],
                job_id=v.get("job_id"),
                status=v.get("status", "pending"),
                error=v.get("error"),
            )
        return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_path(work_dir: Path) -> Path:
    return work_dir / "stress_manifest.json"


def load_manifest(work_dir: Path) -> Manifest:
    path = _manifest_path(work_dir)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}. Run submit first.")
    with path.open("r") as f:
        return Manifest.from_dict(json.load(f))


def save_manifest(work_dir: Path, manifest: Manifest) -> None:
    path = _manifest_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def _row_key(row_id: int) -> str:
    return f"row_{row_id}"


def _local_row_dir(work_dir: Path, row_id: int) -> Path:
    return work_dir / "local" / f"row_{row_id:07d}"


def _remote_row_dir(remote_work_dir: str, row_id: int, prefix: str = "stress_sp") -> str:
    return f"{remote_work_dir.rstrip('/')}/{prefix}_row_{row_id:07d}"


def _remote_row_rel(row_id: int, prefix: str = "stress_sp") -> str:
    return f"{prefix}_row_{row_id:07d}"


def _sync_manifest_db_path(manifest: Manifest, args: argparse.Namespace, work_dir: Path) -> None:
    """
    Update manifest.db_path to match the --db argument.

    This allows a workflow to be resumed on a different machine where the
    project directory has a different absolute path.
    """
    db_path = str(Path(args.db).resolve())
    if db_path != manifest.db_path:
        manifest.db_path = db_path
        save_manifest(work_dir, manifest)


def _rel_remote_dir(remote_work_dir: str, remote_dir: str) -> str:
    """
    Convert an absolute remote path to a path relative to ``remote_work_dir``.

    The upload/download helpers in ``RemoteSLURMClient`` already prepend
    ``remote_work_dir``, so they must receive relative parts. Manifests still
    store absolute paths for easy human inspection and cleanup.
    """
    base = remote_work_dir.rstrip("/")
    if remote_dir.startswith(base + "/"):
        return remote_dir[len(base) + 1 :]
    return remote_dir.lstrip("/")


def _rows_missing_stress(db_path: str) -> list[int]:
    """Return ASE DB row IDs that do not yet have a valid stress tensor."""
    missing: list[int] = []
    with connect(db_path) as db:
        for row in db.select():
            data = row.data if hasattr(row, "data") and row.data else {}
            stress = data.get("stress")
            # Treat missing or None/empty stress as not yet computed.
            if stress is None:
                missing.append(row.id)
            else:
                arr = np.asarray(stress)
                if arr.size == 0 or np.isnan(arr).all():
                    missing.append(row.id)
    return missing


def _update_db_stress(db_path: str, row_id: int, stress: np.ndarray) -> None:
    """Write a (1, 3, 3) stress tensor into the ASE DB without touching other data."""
    stress_arr = np.asarray(stress, dtype=np.float64)
    if stress_arr.shape == (3, 3):
        stress_arr = stress_arr[np.newaxis, :, :]
    if stress_arr.shape != (1, 3, 3):
        raise ValueError(f"Stress shape must be (1, 3, 3) or (3, 3), got {stress_arr.shape}")

    with connect(db_path) as db:
        db.update(row_id, data={"stress": stress_arr})

        md = db.metadata
        unit_dict = md.get("_property_unit_dict", {})
        if "stress" not in unit_dict:
            unit_dict = dict(unit_dict)
            unit_dict["stress"] = STRESS_UNIT
            md["_property_unit_dict"] = unit_dict
            db.metadata = md


def _rows_to_compute(
    db_path: str,
    source: str | None = None,
    force_recompute: bool = False,
) -> list[int]:
    """Return ASE DB row IDs to (re)compute.

    By default this is identical to :func:`_rows_missing_stress`. With
    ``source`` only rows whose ``source`` keypoint matches are considered;
    with ``force_recompute`` rows are selected even if a stress tensor is
    already present (used to relabel rows computed with non-standard CP2K
    settings).
    """
    if source is None and not force_recompute:
        return _rows_missing_stress(db_path)

    selected: list[int] = []
    with connect(db_path) as db:
        for row in db.select():
            if source is not None and row.key_value_pairs.get("source") != source:
                continue
            if force_recompute:
                selected.append(row.id)
                continue
            data = row.data if hasattr(row, "data") and row.data else {}
            stress = data.get("stress")
            if stress is None:
                selected.append(row.id)
            else:
                arr = np.asarray(stress)
                if arr.size == 0 or np.isnan(arr).all():
                    selected.append(row.id)
    return selected


def _parse_energy_forces(out_file: Path) -> tuple[float, np.ndarray] | None:
    """Parse total energy (eV) and forces (eV/Å) from a CP2K single-point out.

    Primary path: the ``FORCES| Atomic forces [eV/angstrom]`` block produced
    when the input sets ``FORCE_UNIT eV/angstrom`` (parsed by
    :func:`sio2_utils.parse_cp2k_single_point_out`, which also converts the
    Hartree total energy to eV). Fallback: the legacy
    ``ATOMIC FORCES in [a.u.]`` block, converted from Ha/bohr.
    """
    text = out_file.read_text(errors="ignore")
    if "FORCES| Atomic forces" in text:
        header_line = next(
            ln for ln in text.splitlines() if "FORCES| Atomic forces" in ln
        )
        if "ev/angstrom" not in header_line.lower():
            raise ValueError(f"unexpected FORCES unit in {out_file}: {header_line!r}")
        parsed = parse_cp2k_single_point_out(out_file)
        if parsed is None:
            return None
        return parsed["energy"], np.asarray(parsed["forces"], dtype=np.float64)

    if "ATOMIC FORCES in [a.u.]" not in text:
        return None
    energy_hartree = None
    for line in text.splitlines():
        if line.strip().startswith("ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]"):
            energy_hartree = float(line.split()[-1])
    if energy_hartree is None:
        return None
    forces: list[tuple[float, float, float]] = []
    in_block = False
    for line in text.splitlines():
        if "ATOMIC FORCES in [a.u.]" in line:
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if stripped.startswith("SUM OF ATOMIC FORCES"):
            break
        parts = stripped.split()
        if len(parts) >= 5 and parts[0].isdigit():
            try:
                forces.append((float(parts[-3]), float(parts[-2]), float(parts[-1])))
            except ValueError:
                continue
    if not forces:
        return None

    return energy_hartree * HA_TO_EV, np.asarray(forces, dtype=np.float64) * HA_BOHR_TO_EV_ANG


def _update_db_labels(
    db_path: str,
    row_id: int,
    energy_ev: float,
    forces_ev_ang: np.ndarray,
    stress_ev_ang3: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    """Write energy/forces/stress labels into reserved columns and row.data.

    Reserved columns (energy, forces) are encoded via the Atoms calculator
    results, so they are rewritten through ``atoms=``; the ``data`` copies
    (which schnetpack actually reads) are updated in the same call. The
    existing dtype/shape of each ``data`` entry is preserved.
    """
    stress_arr = np.asarray(stress_ev_ang3, dtype=np.float64)
    if stress_arr.shape == (3, 3):
        stress_arr = stress_arr[np.newaxis, :, :]
    if stress_arr.shape != (1, 3, 3):
        raise ValueError(f"Stress shape must be (1, 3, 3) or (3, 3), got {stress_arr.shape}")
    forces_arr = np.asarray(forces_ev_ang, dtype=np.float64)

    with connect(db_path) as db:
        row = db.get(id=row_id)
        atoms = row.toatoms()
        if atoms.calc is None:
            raise ValueError(f"row {row_id} has no calculator results to update")
        atoms.calc.results["energy"] = float(energy_ev)
        atoms.calc.results["forces"] = forces_arr
        if "free_energy" in atoms.calc.results:
            atoms.calc.results["free_energy"] = float(energy_ev)

        data = dict(row.data) if row.data else {}
        for key, value in (
            ("energy", np.float64(energy_ev)),
            ("forces", forces_arr),
            ("stress", stress_arr),
        ):
            old = data.get(key)
            if old is None:
                data[key] = value
            else:
                old_arr = np.asarray(old, dtype=np.float64)
                data[key] = np.asarray(value, dtype=np.float64).reshape(old_arr.shape)
        data["_sp_relabel"] = provenance
        db.update(row_id, atoms=atoms, data=data)

        md = db.metadata
        unit_dict = md.get("_property_unit_dict", {})
        if "stress" not in unit_dict:
            unit_dict = dict(unit_dict)
            unit_dict["stress"] = STRESS_UNIT
            md["_property_unit_dict"] = unit_dict
            db.metadata = md


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def _write_local_run(
    local_dir: Path,
    atoms: Atoms,
    row_id: int,
    cutoff: int,
    rel_cutoff: int,
    basis_set: str,
    xc_functional: str,
    cp2k_data_dir: str,
) -> None:
    """Generate CP2K input for an ENERGY_FORCE single point with stress."""
    local_dir.mkdir(parents=True, exist_ok=True)

    # Write structure as XYZ for reference; coordinates are also inlined in the input.
    write(local_dir / "structure.xyz", atoms, format="extxyz")

    project_name = f"stress_sp_{row_id}"
    inp_content = build_cp2k_input_carbon(
        atoms=atoms,
        project_name=project_name,
        run_type="ENERGY_FORCE",
        cutoff=cutoff,
        rel_cutoff=rel_cutoff,
        basis_set=basis_set,
        xc_functional=xc_functional,
        scf_guess="ATOMIC",
        cp2k_data_dir=cp2k_data_dir,
        # n_steps/temperature are irrelevant for ENERGY_FORCE but kept to satisfy signature.
        n_steps=1,
        temperature=300.0,
    )

    (local_dir / "input.inp").write_text(inp_content)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def _slurm_script_content(
    remote_dir: str,
    job_name: str,
    ntasks_per_node: int,
    partition: str,
) -> str:
    """Render the SLURM script for a single-point job (no submission).

    Mirrors ``RemoteSLURMClient.write_and_submit_slurm`` so that a script
    generated locally for ``--prepare-only`` is identical to what the client
    would have written remotely.
    """
    work_dir_node = remote_dir.replace("/home/work", "/work")
    return REMOTE_SLURM_TEMPLATE.format(
        job_name=job_name,
        ntasks_per_node=ntasks_per_node,
        partition=partition,
        depend_line="",
        work_dir_head=remote_dir,
        work_dir_node=work_dir_node,
        run_block="srun /opt/software/cp2k/install/bin/cp2k.psmp -i input.inp",
    )


def cmd_submit(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    remote_config = load_remote_config(args.remote_config)

    # Load or create manifest.
    manifest_path = _manifest_path(work_dir)
    if manifest_path.exists():
        manifest = load_manifest(work_dir)
        _sync_manifest_db_path(manifest, args, work_dir)
        print(f"Resuming from existing manifest: {manifest_path}")
    else:
        manifest = Manifest(
            db_path=str(db_path),
            remote_work_dir=remote_config.remote_work_dir,
        )

    # Determine which rows need (re)computation and are not already tracked.
    all_missing = _rows_to_compute(
        str(db_path),
        source=args.source,
        force_recompute=args.force_recompute,
    )
    already_tracked = {r.row_id for r in manifest.jobs.values()}
    to_submit = [rid for rid in all_missing if rid not in already_tracked]

    print(f"DB rows to compute: {len(all_missing)}")
    print(f"Already tracked in manifest: {len(already_tracked)}")

    if not to_submit:
        print("Nothing to submit.")
        return 0

    # Dry-run is a purely local preview: do not connect to the remote cluster.
    if args.dry_run:
        n_to_submit_dry = min(len(to_submit), args.max_jobs, args.max_concurrent)
        print(f"Dry run: would submit up to {n_to_submit_dry} rows")
        print(f"  (concurrency limit {args.max_concurrent}, max-jobs {args.max_jobs})")
        print("Would submit the following rows:")
        for rid in to_submit[:n_to_submit_dry]:
            print(f"  row {rid}")
        return 0

    # Only connect to the remote cluster when we actually intend to submit jobs.
    remote_client = RemoteSLURMClient(remote_config)
    prepare_only = getattr(args, "prepare_only", False)

    if prepare_only:
        # The server-side driver (slurm_queue_driver.sh, typically inside tmux)
        # submits the prepared jobs, so no local concurrency accounting here.
        available_slots = len(to_submit)
        active_jobs = 0
    else:
        # Enforce the concurrent-job limit before starting new submissions.
        active_jobs = _count_active_jobs(remote_client, manifest)
        available_slots = max(0, args.max_concurrent - active_jobs)
        print(
            f"Active stress jobs already in SLURM queue: {active_jobs} "
            f"(max-concurrent limit: {args.max_concurrent})"
        )
        print(f"Free slots available now: {available_slots}")

    n_to_submit = min(len(to_submit), args.max_jobs, available_slots)
    mode = "prepare (no sbatch)" if prepare_only else "submit"
    print(f"This submit call will {mode} {n_to_submit} new job(s).")

    if n_to_submit == 0:
        print(
            "Concurrency limit reached. Re-run submit later when slots free up, "
            "or run collect to retrieve finished results."
        )
        return 0

    n_submitted = 0
    with connect(str(db_path)) as db:
        for row_id in to_submit:
            if n_submitted >= n_to_submit:
                break

            row = db.get(id=row_id)
            atoms = row.toatoms()
            atoms.set_pbc(True)

            local_dir = _local_row_dir(work_dir, row_id)
            _write_local_run(
                local_dir,
                atoms,
                row_id,
                cutoff=args.cutoff,
                rel_cutoff=args.rel_cutoff,
                basis_set=args.basis_set,
                xc_functional=args.xc_functional,
                cp2k_data_dir=args.cp2k_data_dir,
            )

            remote_dir = _remote_row_dir(
                remote_config.remote_work_dir, row_id, prefix=args.job_prefix
            )
            remote_rel = _remote_row_rel(row_id, prefix=args.job_prefix)
            job_name = f"csp_{row_id:07d}"
            job_id = None
            try:
                if prepare_only:
                    (local_dir / "submit.slurm").write_text(
                        _slurm_script_content(
                            remote_dir, job_name, args.ntasks_per_node, args.partition
                        )
                    )
                    # upload_run expects paths relative to remote_work_dir; it
                    # prepends the base internally.
                    remote_client.upload_run(local_dir, remote_rel)
                else:
                    # upload_run/write_and_submit_slurm expect paths relative to
                    # remote_work_dir; they prepend it internally.
                    remote_client.upload_run(local_dir, remote_rel)
                    job_id = remote_client.write_and_submit_slurm(
                        remote_rel,
                        job_name=job_name,
                        input_file="input.inp",
                        ntasks_per_node=args.ntasks_per_node,
                        partition=args.partition,
                    )
            except Exception as exc:
                logger.error("Failed to submit row %d: %s", row_id, exc)
                # Clean up local scratch so the row can be retried later.
                shutil.rmtree(local_dir, ignore_errors=True)
                continue

            key = _row_key(row_id)
            manifest.jobs[key] = JobRecord(
                row_id=row_id,
                remote_dir=remote_dir,
                job_id=job_id,
                status="prepared" if prepare_only else "submitted",
            )
            n_submitted += 1

            if n_submitted % args.batch_size == 0:
                save_manifest(work_dir, manifest)
                print(f"  submitted {n_submitted}/{n_to_submit}")

    save_manifest(work_dir, manifest)
    print(f"\nSubmitted {n_submitted} new jobs.")
    print(f"Manifest saved to {manifest_path}")
    if prepare_only:
        print(
            "All inputs uploaded. Start the server-side driver (tmux) to run "
            "the queue, e.g.:\n"
            f"  ssh {remote_config.username}@{remote_config.host} -p {remote_config.port} "
            f"'tmux new-session -d -s relabel \"bash slurm_queue_driver.sh "
            f"{remote_config.remote_work_dir} <max_concurrent> 60 {args.job_prefix}_row_\"'"
        )
    else:
        print("You can now disconnect. Run 'collect' later to retrieve results.")
    return 0


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


def _download_and_parse(
    remote_client: RemoteSLURMClient,
    work_dir: Path,
    record: JobRecord,
) -> tuple[dict[str, Any] | None, str | None]:
    """Download a finished job's output and parse energy/forces/stress labels."""
    local_dir = _local_row_dir(work_dir, record.row_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_rel = _rel_remote_dir(
        remote_client.config.remote_work_dir, record.remote_dir
    )
    remote_parts = remote_rel.split("/")
    downloaded = remote_client.download_results(
        local_dir,
        *remote_parts,
        patterns=["out", "err", "*.inp"],
    )

    out_file = local_dir / "out"
    if not out_file.exists():
        return None, "output file 'out' not downloaded"

    # Quick sanity check: SCF must have converged.
    out_text = out_file.read_text(errors="ignore")
    if "SCF run converged" not in out_text:
        return None, "SCF did not converge"

    ef = _parse_energy_forces(out_file)
    if ef is None:
        return None, "energy/forces not found in output"
    energy_ev, forces = ef

    stress = parse_cp2k_stress(out_file)
    if stress is None:
        return None, "stress tensor not found in output"

    return {"energy": energy_ev, "forces": forces, "stress": stress}, None


def _remote_state(remote_client: RemoteSLURMClient, remote_dir: str) -> str | None:
    """Read the driver-managed ``state`` file of a prepared job directory."""
    result = remote_client._ssh(f"cat {remote_dir}/state 2>/dev/null || true")
    state = result.stdout.strip()
    return state or None


def _backup_db_once_per_day(db_path: str) -> None:
    """Copy the DB to a .bak_relabel_<timestamp> file, at most once per day."""
    p = Path(db_path)
    day = datetime.now().strftime("%Y%m%d")
    if list(p.parent.glob(p.name + f".bak_relabel_{day}*")):
        return
    backup = p.with_name(f"{p.name}.bak_relabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(p, backup)
    print(f"Backup: {backup}")


def _remote_job_state(remote_client: RemoteSLURMClient, job_id: str) -> str:
    """
    Query SLURM for a job's state.

    Returns one of: RUNNING, PENDING, COMPLETED, FAILED, UNKNOWN.
    UNKNOWN usually means the job is too new for the scheduler to report, or
    it finished and has been purged from scontrol.
    """
    # Active jobs are visible in squeue.
    result = remote_client._ssh(f"squeue -j {job_id} -h -o '%T' 2>&1 || true")
    state = result.stdout.strip()
    # squeue may return an error such as "slurm_load_jobs error: Invalid job id
    # specified" for purged jobs; treat that as an unknown state.
    if state and not state.lower().startswith("slurm_load_jobs error"):
        return state

    # Completed/failed jobs may still be in scontrol.
    result = remote_client._ssh(f"scontrol show job {job_id} 2>&1 || true")
    stdout = result.stdout
    if "slurm_load_jobs error" in stdout.lower():
        return "UNKNOWN"
    if "JobState" in stdout:
        for line in stdout.split("\n"):
            if "JobState" in line:
                return line.split("=")[1].split()[0]
    return "UNKNOWN"


def _remote_file_exists(remote_client: RemoteSLURMClient, remote_path: str) -> bool:
    """Check whether a file exists on the remote host."""
    result = remote_client._ssh(f"test -f {remote_path} && echo yes || echo no")
    return result.stdout.strip() == "yes"


def _count_active_jobs(remote_client: RemoteSLURMClient, manifest: Manifest) -> int:
    """
    Count manifest jobs that are still active in the SLURM queue.

    Jobs marked as ``submitted`` or ``running`` are queried; only PENDING,
    RUNNING, CONFIGURING, COMPLETING or SUSPENDED states count against the
    concurrency limit. Completed/failed/unknown jobs are treated as finished
    for the purpose of allowing new submissions, even if they have not yet
    been collected.
    """
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
    active = 0
    for record in manifest.jobs.values():
        if record.status not in ("submitted", "running") or record.job_id is None:
            continue
        state = _remote_job_state(remote_client, record.job_id)
        if state in active_states:
            active += 1
    return active


def cmd_collect(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).resolve()
    manifest = load_manifest(work_dir)
    _sync_manifest_db_path(manifest, args, work_dir)
    remote_config = load_remote_config(args.remote_config)
    remote_client = RemoteSLURMClient(remote_config)

    pending_jobs = {
        k: v for k, v in manifest.jobs.items()
        if v.status in ("prepared", "submitted", "running")
    }

    if not pending_jobs:
        print("No prepared/submitted/running jobs to collect.")
        print(
            f"Manifest: {manifest.n_completed} completed, "
            f"{manifest.n_failed} failed, {len(manifest.jobs)} total tracked."
        )
        return 0

    print(f"Checking {len(pending_jobs)} unfinished jobs...")

    # Pass 1: determine readiness, download and parse finished jobs.
    parsed: list[tuple[JobRecord, dict[str, Any]]] = []
    newly_failed = 0
    still_pending = 0

    for key, record in pending_jobs.items():
        if record.job_id is None:
            # Driver-managed (prepare-only) job: the server-side
            # slurm_queue_driver.sh tracks the lifecycle in a `state` file.
            state = _remote_state(remote_client, record.remote_dir)
            if state is None:
                still_pending += 1
                continue
            if state.startswith("failed"):
                record.status = "failed"
                record.error = f"driver: {state}"
                manifest.n_failed += 1
                newly_failed += 1
                logger.error("Row %d failed: driver reports %s", record.row_id, state)
                continue
            if state != "done":
                # "pending" or "submitted <jobid>"; keep polling the state file.
                still_pending += 1
                continue
        else:
            state = _remote_job_state(remote_client, record.job_id)
            out_path = f"{record.remote_dir}/out"

            if state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"):
                print(f"  row {record.row_id}: job {record.job_id} {state}")
                if record.status != "running":
                    record.status = "running"
                still_pending += 1
                continue

            if state == "FAILED":
                record.status = "failed"
                record.error = "SLURM job state FAILED"
                manifest.n_failed += 1
                newly_failed += 1
                logger.error("Row %d failed: SLURM job %s FAILED", record.row_id, record.job_id)
                continue

            # state in ("COMPLETED", "UNKNOWN"): job no longer scheduled. Try
            # to collect; if output is missing, leave it for the next collect.
            if not _remote_file_exists(remote_client, out_path):
                print(
                    f"  row {record.row_id}: job {record.job_id} not in queue but "
                    "output not yet available; will retry"
                )
                still_pending += 1
                continue

        try:
            labels, err = _download_and_parse(remote_client, work_dir, record)
            if labels is None:
                record.status = "failed"
                record.error = err or "unknown parse error"
                manifest.n_failed += 1
                newly_failed += 1
                logger.error("Row %d failed: %s", record.row_id, record.error)
                continue
            parsed.append((record, labels))
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            manifest.n_failed += 1
            newly_failed += 1
            logger.error("Row %d failed during collection: %s", record.row_id, exc)

    # Unit audit gate: audit the whole batch before writing anything.
    if parsed:
        report = audit_frame_values(
            energies=[labels["energy"] for _, labels in parsed],
            natoms=[int(np.asarray(labels["forces"]).shape[0]) for _, labels in parsed],
            forces=[np.asarray(labels["forces"]) for _, labels in parsed],
        )
        print("\nUnit audit of collected batch:")
        print(report.format())
        if not report.ok:
            save_manifest(work_dir, manifest)
            print("Unit audit FAILED — nothing was written to the DB.")
            return 1

        _backup_db_once_per_day(manifest.db_path)

        # Pass 2: write labels (reserved columns + row.data) and clean up.
        provenance = {
            "type": "cp2k_sp_relabel",
            "method": "CP2K Quickstep PBE / DZVP-MOLOPT-SR-GTH, "
                      "ENERGY_FORCE (zero-step MD), STRESS_TENSOR ANALYTICAL",
            "units": {"energy": "eV", "forces": "eV/Angstrom", "stress": STRESS_UNIT},
            "date": datetime.now().isoformat(timespec="seconds"),
        }
        newly_completed = 0
        for record, labels in parsed:
            try:
                _update_db_labels(
                    manifest.db_path,
                    record.row_id,
                    labels["energy"],
                    labels["forces"],
                    labels["stress"],
                    provenance,
                )
            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)
                manifest.n_failed += 1
                newly_failed += 1
                logger.error("Row %d failed during DB write: %s", record.row_id, exc)
                continue
            record.status = "completed"
            record.error = None
            manifest.n_completed += 1
            newly_completed += 1

            # Clean up scratch.
            if not args.keep_scratch:
                local_dir = _local_row_dir(work_dir, record.row_id)
                shutil.rmtree(local_dir, ignore_errors=True)
                remote_client._ssh(f"rm -rf {record.remote_dir}")
    else:
        newly_completed = 0

    save_manifest(work_dir, manifest)

    remaining = sum(
        1 for r in manifest.jobs.values()
        if r.status in ("prepared", "submitted", "running")
    )
    print(
        f"\nCompleted this run: {newly_completed} success, {newly_failed} failed. "
        f"Still pending: {still_pending}."
    )
    print(
        f"Total: {manifest.n_completed} completed, {manifest.n_failed} failed, "
        f"{len(manifest.jobs)} tracked."
    )

    if remaining == 0:
        print("\nAll tracked jobs finished.")
    else:
        print("Run 'collect' again later to check remaining jobs.")

    return 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).resolve()
    manifest_path = _manifest_path(work_dir)
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}.")
        print("Run 'submit' first to initialize the workflow.")
        return 0
    manifest = load_manifest(work_dir)
    _sync_manifest_db_path(manifest, args, work_dir)

    total = len(manifest.jobs)
    completed = sum(1 for r in manifest.jobs.values() if r.status == "completed")
    failed = sum(1 for r in manifest.jobs.values() if r.status == "failed")
    running = sum(1 for r in manifest.jobs.values() if r.status == "running")
    submitted = sum(1 for r in manifest.jobs.values() if r.status == "submitted")
    prepared = sum(1 for r in manifest.jobs.values() if r.status == "prepared")

    print(f"Manifest: {work_dir / 'stress_manifest.json'}")
    print(f"DB path: {manifest.db_path}")
    print(f"Total tracked jobs: {total}")
    print(f"  completed : {completed}")
    print(f"  failed    : {failed}")
    print(f"  running   : {running}")
    print(f"  submitted : {submitted}")
    print(f"  prepared  : {prepared} (waiting for server-side driver)")

    db_missing = len(_rows_missing_stress(manifest.db_path))
    print(f"DB rows still missing stress: {db_missing}")

    # Query SLURM for the actual number of active stress jobs.
    try:
        remote_config = load_remote_config(args.remote_config)
        remote_client = RemoteSLURMClient(remote_config)
        active = _count_active_jobs(remote_client, manifest)
        print(f"Active stress jobs in SLURM queue: {active}")
    except Exception as exc:
        print(f"Could not query SLURM for active job count: {exc}")

    return 0


# ---------------------------------------------------------------------------
# Auto loop
# ---------------------------------------------------------------------------


def cmd_auto(args: argparse.Namespace) -> int:
    """
    Continuously collect finished jobs and submit new ones until done.

    This is useful when each single-point stress calculation is short (e.g.
    ~40 s): the loop keeps the SLURM queue filled up to ``--max-concurrent``
    jobs without manual intervention.
    """
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    remote_config = load_remote_config(args.remote_config)

    print(f"Auto mode: collect + submit loop every {args.poll_interval} s")
    print("Press Ctrl+C to stop. The local machine can still be shut down;")
    print("re-run the same 'auto' command later to resume.")

    try:
        while True:
            print("\n" + "=" * 60)
            print("[auto] Running collect...")
            cmd_collect(args)

            print("\n[auto] Running submit...")
            cmd_submit(args)

            manifest = load_manifest(work_dir)
            _sync_manifest_db_path(manifest, args, work_dir)
            missing = len(_rows_missing_stress(manifest.db_path))

            try:
                remote_client = RemoteSLURMClient(remote_config)
                active = _count_active_jobs(remote_client, manifest)
            except Exception as exc:
                print(f"[auto] Could not query active job count: {exc}")
                active = None

            if active is not None:
                print(f"\n[auto] Status: {missing} rows missing stress, "
                      f"{active} active remote jobs")
                if missing == 0 and active == 0:
                    print("[auto] All done.")
                    return 0
            else:
                print(f"\n[auto] {missing} rows missing stress")

            print(f"[auto] Sleeping {args.poll_interval} s...")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[auto] Stopped by user.")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to ASE database to augment with stress tensors.",
    )
    parser.add_argument(
        "--remote-config",
        type=str,
        required=True,
        help="Path to remote SLURM config JSON.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Local working directory for manifests and scratch.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add DFT stress tensors to an ASE database via remote CP2K single points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = subparsers.add_parser("submit", help="Generate inputs and submit remote jobs.")
    _add_common_args(p_submit)
    p_submit.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of jobs to submit before saving the manifest.",
    )
    p_submit.add_argument(
        "--max-jobs",
        type=int,
        default=10000,
        help="Maximum number of new jobs to submit in this invocation.",
    )
    p_submit.add_argument(
        "--ntasks-per-node",
        type=int,
        default=16,
        help="CP2K MPI tasks per SLURM job (number of cores used).",
    )
    p_submit.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of stress jobs allowed in the SLURM queue at once.",
    )
    p_submit.add_argument(
        "--partition",
        type=str,
        default="compute",
        help="SLURM partition.",
    )
    p_submit.add_argument(
        "--cutoff",
        type=int,
        default=500,
        help="CP2K plane-wave cutoff in Ry.",
    )
    p_submit.add_argument(
        "--rel-cutoff",
        type=int,
        default=50,
        help="CP2K relative cutoff in Ry.",
    )
    p_submit.add_argument(
        "--basis-set",
        type=str,
        default="DZVP-MOLOPT-SR-GTH",
        help="CP2K basis set.",
    )
    p_submit.add_argument(
        "--xc-functional",
        type=str,
        default="PBE",
        help="CP2K XC functional.",
    )
    p_submit.add_argument(
        "--cp2k-data-dir",
        type=str,
        default="/opt/software/cp2k/data",
        help="CP2K data directory on the remote cluster (BASIS, POTENTIAL files).",
    )
    p_submit.add_argument(
        "--dry-run",
        action="store_true",
        help="List rows that would be submitted without submitting.",
    )
    p_submit.add_argument(
        "--source",
        type=str,
        default=None,
        help="Only consider rows whose 'source' keypoint matches this value.",
    )
    p_submit.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute rows even if a stress tensor is already present.",
    )
    p_submit.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate inputs + submit.slurm and upload them without sbatch; "
             "a server-side driver (slurm_queue_driver.sh in tmux) submits them.",
    )
    p_submit.add_argument(
        "--job-prefix",
        type=str,
        default="stress_sp",
        help="Prefix for remote job directories ('<prefix>_row_<id>').",
    )

    # collect
    p_collect = subparsers.add_parser("collect", help="Poll remote jobs and write stress back to DB.")
    _add_common_args(p_collect)
    p_collect.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep local and remote scratch directories after successful collection.",
    )

    # status
    p_status = subparsers.add_parser("status", help="Print current progress.")
    _add_common_args(p_status)

    # auto
    p_auto = subparsers.add_parser(
        "auto",
        help="Continuously collect and submit until all stress labels are computed.",
    )
    _add_common_args(p_auto)
    p_auto.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of jobs to submit before saving the manifest.",
    )
    p_auto.add_argument(
        "--max-jobs",
        type=int,
        default=10000,
        help="Maximum number of new jobs to submit per loop iteration.",
    )
    p_auto.add_argument(
        "--ntasks-per-node",
        type=int,
        default=16,
        help="CP2K MPI tasks per SLURM job (number of cores used).",
    )
    p_auto.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of stress jobs allowed in the SLURM queue at once.",
    )
    p_auto.add_argument(
        "--partition",
        type=str,
        default="compute",
        help="SLURM partition.",
    )
    p_auto.add_argument(
        "--cutoff",
        type=int,
        default=500,
        help="CP2K plane-wave cutoff in Ry.",
    )
    p_auto.add_argument(
        "--rel-cutoff",
        type=int,
        default=50,
        help="CP2K relative cutoff in Ry.",
    )
    p_auto.add_argument(
        "--basis-set",
        type=str,
        default="DZVP-MOLOPT-SR-GTH",
        help="CP2K basis set.",
    )
    p_auto.add_argument(
        "--xc-functional",
        type=str,
        default="PBE",
        help="CP2K XC functional.",
    )
    p_auto.add_argument(
        "--cp2k-data-dir",
        type=str,
        default="/opt/software/cp2k/data",
        help="CP2K data directory on the remote cluster (BASIS, POTENTIAL files).",
    )
    p_auto.add_argument(
        "--dry-run",
        action="store_true",
        help="(Ignored in auto mode.)",
    )
    p_auto.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds to wait between collect/submit cycles.",
    )
    p_auto.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep local and remote scratch directories after successful collection.",
    )
    p_auto.add_argument(
        "--source",
        type=str,
        default=None,
        help="Only consider rows whose 'source' keypoint matches this value.",
    )
    p_auto.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute rows even if a stress tensor is already present.",
    )
    p_auto.add_argument(
        "--job-prefix",
        type=str,
        default="stress_sp",
        help="Prefix for remote job directories ('<prefix>_row_<id>').",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "auto":
        return cmd_auto(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
