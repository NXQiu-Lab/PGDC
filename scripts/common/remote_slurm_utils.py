"""
Remote SLURM job submission and monitoring via SSH/SCP.

Mirrors the interface of ``scripts/common/remote_job_utils.RemotePBSClient`` but targets
the existing SLURM cluster used by ``run_iter003_on_slurm.py``.

Uses SSH ControlMaster to keep a persistent connection open so that:
- Only ONE password prompt is needed (the first connection)
- All subsequent SSH/SCP commands reuse the same authenticated channel

Requires ``ssh`` and ``scp`` on the local machine.
If a password is provided in the config, ``sshpass`` is also required.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SLURM script template (matches run_iter003_on_slurm.py environment)
# ---------------------------------------------------------------------------

REMOTE_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --partition={partition}
#SBATCH --output=out
#SBATCH --error=err
{depend_line}

# Handle path difference between head (/home/work) and compute nodes (/work)
if [ -d {work_dir_head} ]; then
    cd {work_dir_head}
elif [ -d {work_dir_node} ]; then
    cd {work_dir_node}
else
    echo "ERROR: Cannot find work directory"
    exit 1
fi

source /opt/software/cp2k/tools/toolchain/install/setup
export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH
export PMIX_MCA_psec=native

{run_block}
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RemoteConfig:
    """Connection details for the remote SLURM head node."""

    host: str
    port: int = 22
    username: str = ""
    password: str = ""
    remote_work_dir: str = ""


def load_remote_config(path: str | Path) -> RemoteConfig:
    """Load remote config from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return RemoteConfig(
        host=data.get("host", ""),
        port=data.get("port", 22),
        username=data.get("username", ""),
        password=data.get("password", ""),
        remote_work_dir=data.get("remote_work_dir", ""),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RemoteSLURMClient:
    """Handles SSH/SCP interactions with a remote SLURM head node via ControlMaster."""

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config
        self._target = f"{config.username}@{config.host}" if config.username else config.host

        # ControlMaster socket path
        # Use /tmp (short) plus a short hash of the target to stay within the
        # Unix domain socket path length limit (<=104 bytes on macOS).
        target_key = f"{config.host}:{config.port}:{config.username}"
        target_hash = hashlib.shake_128(target_key.encode()).hexdigest(8)
        self._control_path = Path("/tmp") / f"ssh_mux_{target_hash}"

        # Base SSH options (shared)
        self._ssh_opts = [
            "-p", str(config.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ControlPath={self._control_path}",
        ]

        if config.password:
            if not self._has_sshpass():
                raise RuntimeError(
                    "Config contains a password but 'sshpass' is not installed.\n"
                    "Install it with: brew install sshpass\n"
                    "Or remove the 'password' field from the config to use interactive authentication."
                )

        # Open ControlMaster connection (single auth prompt)
        self._open_control_master()
        atexit.register(self.close)

    def close(self) -> None:
        """Close the ControlMaster connection and clean up the socket."""
        self._close_control_master()

    # -- ControlMaster -------------------------------------------------------

    def _open_control_master(self) -> None:
        """Open a persistent SSH ControlMaster connection."""
        # If a socket already exists, check if it's alive
        if self._control_path.exists():
            test_cmd = self._wrap_with_sshpass(
                ["ssh"] + self._ssh_opts + ["-O", "check", self._target]
            )
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return  # Already alive
            # Stale socket, remove it
            try:
                self._control_path.unlink()
            except OSError:
                pass

        print(f"Establishing SSH connection to {self._target} (ControlMaster)...")
        print("  (You may be prompted for a password once.)")

        master_cmd = self._wrap_with_sshpass(
            ["ssh"] + self._ssh_opts + ["-o", "ControlMaster=yes", self._target, "-f", "-N"]
        )
        result = subprocess.run(master_cmd)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to establish SSH ControlMaster connection.\n"
                "Please check your host, username, and password."
            )

        # Give it a moment to settle
        time.sleep(0.5)

    def _close_control_master(self) -> None:
        """Send exit signal to the ControlMaster."""
        if not self._control_path.exists():
            return
        try:
            cmd = self._wrap_with_sshpass(
                ["ssh"] + self._ssh_opts + ["-O", "exit", self._target]
            )
            subprocess.run(cmd, capture_output=True, text=True)
        except Exception:
            pass
        try:
            if self._control_path.exists():
                self._control_path.unlink()
        except OSError:
            pass

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _has_sshpass() -> bool:
        return shutil.which("sshpass") is not None

    def _wrap_with_sshpass(self, cmd: list[str]) -> list[str]:
        if self.config.password and self._has_sshpass():
            return ["sshpass", "-p", self.config.password] + cmd
        return cmd

    def _ssh(self, remote_cmd: str, capture: bool = True) -> subprocess.CompletedProcess:
        """
        Execute a command on the remote host via SSH (reuses ControlMaster).
        Uses ``bash -l -c`` to ensure login shell environment is loaded
        (so tools like sbatch are available in PATH).
        """
        quoted = shlex.quote(remote_cmd)
        full_cmd = self._wrap_with_sshpass(
            ["ssh"] + self._ssh_opts + [self._target, f"bash -l -c {quoted}"]
        )
        kwargs: dict[str, Any] = {}
        if capture:
            kwargs["capture_output"] = True
            kwargs["text"] = True
        return subprocess.run(full_cmd, **kwargs)

    def _scp_up(self, local_path: Path, remote_path: str) -> subprocess.CompletedProcess:
        """Upload a file or directory (recursive) to the remote host."""
        dest = f"{self._target}:{remote_path}"
        cmd = self._wrap_with_sshpass(
            ["scp"] + ["-P", str(self.config.port), "-o", f"ControlPath={self._control_path}"] + ["-r", str(local_path), dest]
        )
        return subprocess.run(cmd, capture_output=True, text=True)

    def _scp_down(self, remote_path: str, local_path: Path) -> subprocess.CompletedProcess:
        """Download a file or directory (recursive) from the remote host."""
        src = f"{self._target}:{remote_path}"
        cmd = self._wrap_with_sshpass(
            ["scp"] + ["-P", str(self.config.port), "-o", f"ControlPath={self._control_path}"] + ["-r", src, str(local_path)]
        )
        return subprocess.run(cmd, capture_output=True, text=True)

    def _remote_path(self, *parts: str) -> str:
        """Join path components on the remote side (POSIX)."""
        base = self.config.remote_work_dir.rstrip("/")
        return "/".join([base] + list(parts))

    # -- directory management ------------------------------------------------

    def ensure_remote_dir(self, *parts: str) -> None:
        """Create a directory on the remote host (mkdir -p)."""
        remote_dir = self._remote_path(*parts)
        result = self._ssh(f"mkdir -p {remote_dir}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create remote dir {remote_dir}: {result.stderr}")

    # -- upload / download ---------------------------------------------------

    def upload_run(self, local_run_dir: Path, *remote_parts: str) -> None:
        """Upload a local run directory to the remote server."""
        remote_dest = self._remote_path(*remote_parts)
        # Ensure the parent directory exists on the remote host; scp does not
        # create intermediate directories automatically.
        remote_parent = "/".join(remote_dest.split("/")[:-1])
        if remote_parent:
            self._ssh(f"mkdir -p {remote_parent}")
        # Remove the remote dir first if it exists, so scp -r creates it cleanly
        # instead of nesting the local dir inside the existing remote dir.
        self._ssh(f"rm -rf {remote_dest} 2>/dev/null || true")
        result = self._scp_up(local_run_dir, remote_dest)
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP upload failed for {local_run_dir} -> {remote_dest}\n"
                f"stderr: {result.stderr}"
            )

    def download_results(
        self,
        local_run_dir: Path,
        *remote_parts: str,
        patterns: list[str] | None = None,
    ) -> list[str]:
        """
        Download result files from remote to local.

        Args:
            local_run_dir: Local directory to download into.
            remote_parts: Path components on the remote (relative to remote_work_dir).
            patterns: Glob patterns for files to download. Defaults to common CP2K outputs.

        Returns:
            List of downloaded file names (relative to local_run_dir).
        """
        if patterns is None:
            patterns = ["*-pos-1.xyz", "*-frc-1.xyz", "*.ener", "*-RESTART.wfn", "out", "err"]

        remote_dir = self._remote_path(*remote_parts)
        local_run_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []

        for pat in patterns:
            # List matching files on remote
            list_cmd = f"ls {remote_dir}/{pat} 2>/dev/null || true"
            result = self._ssh(list_cmd)
            if result.returncode != 0:
                continue

            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            for remote_file in files:
                basename = remote_file.split("/")[-1]
                local_file = local_run_dir / basename
                res = self._scp_down(remote_file, local_file)
                if res.returncode == 0:
                    downloaded.append(basename)

        return downloaded

    # -- job submission ------------------------------------------------------

    def write_and_submit_slurm(
        self,
        *remote_parts: str,
        job_name: str,
        input_file: str = "input.inp",
        ntasks_per_node: int = 64,
        partition: str = "compute",
        depend_job_id: str | None = None,
        run_block: str | None = None,
    ) -> str:
        """
        Write a SLURM submission script on the remote host and submit it.

        Args:
            depend_job_id: If given, add ``--dependency=afterok:<job_id>`` so
                that this job only starts after the dependency succeeds.
            run_block: Optional shell block to execute instead of the default
                ``srun cp2k.psmp -i <input_file>``.

        Returns:
            The SLURM job ID (numeric string).
        """
        remote_dir = self._remote_path(*remote_parts)
        self.ensure_remote_dir(*remote_parts)

        work_dir_head = remote_dir
        work_dir_node = remote_dir.replace("/home/work", "/work")

        depend_line = f"#SBATCH --dependency=afterok:{depend_job_id}" if depend_job_id else ""
        if run_block is None:
            run_block = f"srun /opt/software/cp2k/install/bin/cp2k.psmp -i {input_file}"

        script_content = REMOTE_SLURM_TEMPLATE.format(
            job_name=job_name,
            ntasks_per_node=ntasks_per_node,
            partition=partition,
            depend_line=depend_line,
            work_dir_head=work_dir_head,
            work_dir_node=work_dir_node,
            run_block=run_block,
        )

        # Write script via SSH heredoc
        script_path = f"{remote_dir}/submit.slurm"
        write_cmd = (
            f"cat > {script_path} << 'EOFCP2KSLURM'\n"
            f"{script_content}\n"
            "EOFCP2KSLURM"
        )
        result = self._ssh(write_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to write SLURM script on remote: {result.stderr}")

        # Submit
        sbatch_cmd = f"cd {remote_dir} && sbatch --parsable"
        if depend_job_id:
            sbatch_cmd += f" --dependency=afterok:{depend_job_id}"
        sbatch_cmd += " submit.slurm"
        result = self._ssh(sbatch_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed on remote: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0].strip()
        if not job_id:
            raise RuntimeError(f"sbatch returned empty job ID. stdout: {result.stdout}")
        return job_id

    def submit_existing_script(
        self,
        *remote_parts: str,
        script_name: str = "submit.slurm",
        depend_job_id: str | None = None,
    ) -> str:
        """
        Submit an already-uploaded SLURM script on the remote host.

        Args:
            depend_job_id: If given, add ``--dependency=afterok:<job_id>`` so
                that this job only starts after the dependency succeeds.

        Returns:
            The SLURM job ID (numeric string).
        """
        remote_dir = self._remote_path(*remote_parts)
        sbatch_cmd = f"cd {remote_dir} && sbatch --parsable"
        if depend_job_id:
            sbatch_cmd += f" --dependency=afterok:{depend_job_id}"
        sbatch_cmd += f" {script_name}"
        result = self._ssh(sbatch_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed on remote: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0].strip()
        if not job_id:
            raise RuntimeError(f"sbatch returned empty job ID. stdout: {result.stdout}")
        return job_id

    # -- job status ----------------------------------------------------------

    def is_job_running(self, job_id: str) -> bool:
        """Check if a SLURM job is still queued or running."""
        # squeue only shows jobs in the scheduling queue; completed jobs disappear.
        result = self._ssh(f"squeue -j {job_id} -h -o '%T' 2>&1 || true")
        state = result.stdout.strip()
        if not state:
            # Job not in squeue anymore; use scontrol as fallback to distinguish
            # COMPLETED from FAILED/CANCELLED (all are "not running").
            result = self._ssh(f"scontrol show job {job_id} 2>&1 || true")
            stdout = result.stdout
            if "JobState" in stdout:
                for line in stdout.split("\n"):
                    if "JobState" in line:
                        job_state = line.split("=")[1].split()[0]
                        return job_state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED")
            return False
        return state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED")

    def wait_for_jobs(
        self,
        job_ids: list[str],
        poll_interval: int = 60,
        timeout: float | None = None,
    ) -> dict[str, str]:
        """
        Block until all jobs finish. Only intended for ``auto`` mode.

        Returns:
            Dict mapping job_id -> final state ("completed" or "timeout").
        """
        start_time = time.time()
        remaining = set(job_ids)
        results: dict[str, str] = {}

        print(f"Waiting for {len(remaining)} remote SLURM jobs: {sorted(remaining)}")

        while remaining:
            elapsed = time.time() - start_time
            if timeout is not None and elapsed > timeout:
                for jid in remaining:
                    results[jid] = "timeout"
                print(f"Timeout after {timeout}s. Remaining: {remaining}")
                return results

            still_running: set[str] = set()
            for jid in remaining:
                if self.is_job_running(jid):
                    still_running.add(jid)
                else:
                    results[jid] = "completed"
                    print(f"  Job {jid} completed.")

            remaining = still_running
            if remaining:
                print(f"  {len(remaining)} jobs still running, next check in {poll_interval}s...")
                time.sleep(poll_interval)

        return results
