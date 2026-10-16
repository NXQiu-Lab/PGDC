"""
Cluster job submission and monitoring utilities (Slurm / PBS compatible).

Default: Slurm (sbatch / squeue / scancel).
Falls back to PBS (qsub / qstat) if Slurm commands are not available.

Usage:
  submit_cp2k_job(...)  → writes submit script + sbatch
  squeue() / qstat()    → query job status
  is_job_running(...)   → check if a job is active
  wait_for_jobs(...)    → block until all jobs finish
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Submission script templates
# ---------------------------------------------------------------------------

SLURM_TEMPLATE = """#!/bin/bash -l
{partition_line}{account_line}#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={nprocs}
{cpus_line}{mem_line}#SBATCH --output={work_dir}/slurm-%j.out
#SBATCH --error={work_dir}/slurm-%j.err
#SBATCH --chdir={work_dir}

{env_lines}
{cp2k_cmd}
"""

PBS_TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -o {stdout_path}
#PBS -e {stderr_path}
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -q {queue}

cd "$PBS_O_WORKDIR"

source /opt/rh/devtoolset-11/enable
export CP2K_PATH=/opt/public/cp2k.24.1/
export PATH=$CP2K_PATH/exe/local:$PATH
source $CP2K_PATH/tools/toolchain/install/setup
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

mpirun -np {nprocs} cp2k.popt -i ./{input_file}
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ClusterJob:
    """Represents a submitted cluster job."""

    job_id: str
    job_name: str
    work_dir: Path
    input_file: Path
    submitted_at: float
    scheduler: str  # "slurm" or "pbs"


# ---------------------------------------------------------------------------
# Scheduler detection
# ---------------------------------------------------------------------------

def _detect_scheduler() -> str:
    """Auto-detect whether Slurm or PBS is available."""
    if shutil.which("sbatch") is not None:
        return "slurm"
    if shutil.which("qsub") is not None:
        return "pbs"
    raise RuntimeError(
        "No cluster scheduler found. Expected 'sbatch' (Slurm) or 'qsub' (PBS)."
    )


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def write_slurm_script(
    work_dir: str | Path,
    job_name: str,
    input_file: str,
    nodes: int = 1,
    nprocs: int = 76,
    queue: str = "comp",
    cpus_per_task: int = 1,
    mem: str = "",
    extra_env: dict[str, str] | None = None,
    cp2k_cmd: str | None = None,
    account: str = "",
) -> Path:
    """Write a Slurm submission script (sub.sh).

    Args:
        queue: Slurm partition name. Empty string omits the ``-p`` directive.
        account: Slurm account name. Empty string omits the ``--account`` directive.
        cp2k_cmd: Custom CP2K execution command. If None, defaults to
            ``mpirun -np {nprocs} cp2k.popt {input_file} > output.log``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "sub.sh"

    partition_line = f"#SBATCH -p {queue}\n" if queue else ""
    account_line = f"#SBATCH --account={account}\n" if account else ""
    cpus_line = f"#SBATCH --cpus-per-task={cpus_per_task}\n" if cpus_per_task > 1 else ""
    mem_line = f"#SBATCH --mem={mem}\n" if mem else ""

    env_lines = ""
    if extra_env:
        for key, value in extra_env.items():
            env_lines += f"export {key}={value}\n"
        if env_lines:
            env_lines += "\n"

    if cp2k_cmd is None:
        cp2k_cmd = f"mpirun -np {nprocs} cp2k.popt {input_file} > output.log"

    script = SLURM_TEMPLATE.format(
        partition_line=partition_line,
        account_line=account_line,
        nodes=nodes,
        nprocs=nprocs,
        cpus_line=cpus_line,
        mem_line=mem_line,
        env_lines=env_lines,
        cp2k_cmd=cp2k_cmd,
        work_dir=str(work_dir.resolve()),
    )

    with script_path.open("w") as f:
        f.write(script)

    return script_path


def write_pbs_script(
    work_dir: str | Path,
    job_name: str,
    input_file: str,
    nodes: int = 1,
    ppn: int = 32,
    nprocs: int = 48,
    queue: str = "vasp",
    stdout_path: str = "out",
    stderr_path: str = "err",
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Write a PBS submission script (submit.sh)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "submit.sh"

    env_lines = ""
    if extra_env:
        for key, value in extra_env.items():
            env_lines += f"export {key}={value}\n"

    script = PBS_TEMPLATE.format(
        job_name=job_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        nodes=nodes,
        ppn=ppn,
        queue=queue,
        nprocs=nprocs,
        input_file=input_file,
    )

    if env_lines:
        script = script.replace(
            "export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH",
            f"export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH\n{env_lines}",
        )

    with script_path.open("w") as f:
        f.write(script)

    return script_path


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def sbatch(script_path: str | Path) -> str:
    """Submit a Slurm script via sbatch.

    Returns:
        Job ID string (e.g. "12345").
    """
    script_path = Path(script_path).resolve()
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Parse "Submitted batch job 12345"
    parts = result.stdout.strip().split()
    job_id = parts[-1] if parts else ""
    if not job_id:
        raise RuntimeError(f"sbatch returned empty job ID. stdout={result.stdout}")
    return job_id


def qsub(script_path: str | Path) -> str:
    """Submit a PBS script via qsub.

    Returns:
        Job ID string.
    """
    script_path = Path(script_path)
    result = subprocess.run(
        ["qsub", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed: {result.stderr}")

    job_id = result.stdout.strip().split(".")[0]
    if not job_id:
        raise RuntimeError(f"qsub returned empty job ID. stdout={result.stdout}")
    return job_id


# ---------------------------------------------------------------------------
# Job status queries
# ---------------------------------------------------------------------------

def squeue(job_id: str | None = None) -> list[dict[str, Any]]:
    """Query Slurm job status.

    Returns:
        List of dicts with keys: job_id, name, user, state, partition.
    """
    cmd = ["squeue", "--noheader", "--format=%i %j %u %t %P"]
    if job_id:
        cmd.extend(["-j", job_id])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if "Invalid job id" in result.stderr:
            return []
        raise RuntimeError(f"squeue failed: {result.stderr}")

    jobs: list[dict[str, Any]] = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            jobs.append({
                "job_id": parts[0],
                "name": parts[1],
                "user": parts[2],
                "state": parts[3],
                "queue": parts[4],
            })
    return jobs


def qstat(job_id: str | None = None) -> list[dict[str, Any]]:
    """Query PBS job status.

    Returns:
        List of dicts with keys: job_id, name, user, state, queue.
    """
    cmd = ["qstat"]
    if job_id:
        cmd.append(job_id)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if "Unknown Job Id" in result.stderr:
            return []
        raise RuntimeError(f"qstat failed: {result.stderr}")

    lines = result.stdout.strip().split("\n")
    if len(lines) < 3:
        return []

    jobs: list[dict[str, Any]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 5:
            jobs.append({
                "job_id": parts[0].split(".")[0],
                "name": parts[1],
                "user": parts[2],
                "state": parts[4] if len(parts) > 4 else "?",
                "queue": parts[3] if len(parts) > 3 else "?",
            })
    return jobs


# ---------------------------------------------------------------------------
# Running-state helpers
# ---------------------------------------------------------------------------

def is_job_running(job_id: str, scheduler: str | None = None) -> bool:
    """Check if a job is still running or queued.

    Args:
        job_id: Job ID to check.
        scheduler: "slurm" or "pbs". Auto-detected if None.
    """
    sched = scheduler or _detect_scheduler()
    if sched == "slurm":
        jobs = squeue(job_id)
        if not jobs:
            return False
        state = jobs[0].get("state", "")
        # R=running, PD=pending, CG=completing, CF=configuring
        return state in ("R", "PD", "CG", "CF", "S", "ST")
    else:
        jobs = qstat(job_id)
        if not jobs:
            return False
        state = jobs[0].get("state", "")
        return state in ("R", "Q", "H", "T", "W", "S")


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------

def wait_for_jobs(
    job_ids: list[str],
    poll_interval: int = 60,
    timeout: float | None = None,
    scheduler: str | None = None,
    callback: Any | None = None,
) -> dict[str, str]:
    """Wait for a list of jobs to complete.

    Args:
        job_ids: List of job IDs to wait for.
        poll_interval: Seconds between status checks.
        timeout: Max total seconds to wait. None = no timeout.
        scheduler: "slurm" or "pbs". Auto-detected if None.
        callback: Optional callable(job_id, state) called each poll.

    Returns:
        Dict mapping job_id -> final state ("completed", "timeout", "unknown").
    """
    sched = scheduler or _detect_scheduler()
    start_time = time.time()
    remaining = set(job_ids)
    results: dict[str, str] = {}

    print(f"Waiting for {len(remaining)} {sched.upper()} jobs: {sorted(remaining)}")

    while remaining:
        elapsed = time.time() - start_time
        if timeout is not None and elapsed > timeout:
            for jid in remaining:
                results[jid] = "timeout"
            print(f"Timeout after {timeout}s. Remaining: {remaining}")
            return results

        still_running: set[str] = set()
        for jid in remaining:
            if is_job_running(jid, scheduler=sched):
                still_running.add(jid)
                if callback:
                    callback(jid, "running")
            else:
                results[jid] = "completed"
                print(f"  Job {jid} completed.")
                if callback:
                    callback(jid, "completed")

        remaining = still_running
        if remaining:
            print(f"  {len(remaining)} jobs still running, next check in {poll_interval}s...")
            time.sleep(poll_interval)

    return results


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def submit_cp2k_job(
    work_dir: str | Path,
    job_name: str,
    input_file: str = "input.inp",
    nodes: int = 1,
    ppn: int = 32,
    nprocs: int = 48,
    queue: str = "comp",
    extra_env: dict[str, str] | None = None,
    cp2k_cmd: str | None = None,
    cpus_per_task: int = 1,
    mem: str = "",
    account: str = "",
) -> ClusterJob:
    """Write submission script + submit in one call.

    Auto-detects Slurm vs PBS and uses the appropriate backend.
    For Slurm, *ppn* is ignored and *nprocs* maps to #SBATCH --ntasks-per-node.

    Returns:
        ClusterJob object.
    """
    work_dir = Path(work_dir)
    sched = _detect_scheduler()

    if sched == "slurm":
        script_path = write_slurm_script(
            work_dir=work_dir,
            job_name=job_name,
            input_file=input_file,
            nodes=nodes,
            nprocs=nprocs,
            queue=queue,
            cpus_per_task=cpus_per_task,
            mem=mem,
            extra_env=extra_env,
            cp2k_cmd=cp2k_cmd,
            account=account,
        )
        job_id = sbatch(script_path)
    else:
        script_path = write_pbs_script(
            work_dir=work_dir,
            job_name=job_name,
            input_file=input_file,
            nodes=nodes,
            ppn=ppn,
            nprocs=nprocs,
            queue=queue,
            extra_env=extra_env,
        )
        job_id = qsub(script_path)

    return ClusterJob(
        job_id=job_id,
        job_name=job_name,
        work_dir=work_dir,
        input_file=work_dir / input_file,
        submitted_at=time.time(),
        scheduler=sched,
    )
