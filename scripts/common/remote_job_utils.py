"""Remote PBS job submission and monitoring via SSH/SCP.

Previously lived in ``scripts/water/remote_pbs_utils.py``; extracted here so
that ``scripts/water`` can be removed while keeping the client required by
``scripts/sio2``.

Uses SSH ControlMaster to keep a persistent connection open so that:
- Only ONE password prompt is needed (the first connection)
- All subsequent SSH/SCP commands reuse the same authenticated channel

Requires ``ssh`` and ``scp`` on the local machine.
If a password is provided in the config, ``sshpass`` is also required.
"""

from __future__ import annotations

import atexit
import json
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PBS script template (matches the target CentOS 7 server environment)
# ---------------------------------------------------------------------------

REMOTE_PBS_TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -q {queue}

cd "$PBS_O_WORKDIR"

# Redirect stdout/stderr to workdir so we can tail -f out in real-time
exec > out 2> err

source /opt/rh/devtoolset-11/enable
export CP2K_PATH=/opt/public/cp2k.24.1/
export PATH=$CP2K_PATH/exe/local:$PATH
source $CP2K_PATH/tools/toolchain/install/setup
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

mpirun -np {nprocs} cp2k.popt -i ./{input_file}
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RemoteConfig:
    """Connection details for the remote PBS server."""

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

class RemotePBSClient:
    """Handles SSH/SCP interactions with a remote PBS cluster via ControlMaster."""

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config
        self._target = f"{config.username}@{config.host}" if config.username else config.host

        # ControlMaster socket path
        self._control_path = Path(tempfile.gettempdir()) / f"ssh_mux_{config.host}_{config.port}_{config.username}"

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
        (so tools like qsub are available in PATH).
        """
        # Pass the whole "bash -l -c '...'" as a SINGLE argument to ssh so that
        # bash receives the quoted string as the -c operand.
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

    def write_and_submit_pbs(
        self,
        *remote_parts: str,
        job_name: str,
        input_file: str = "input.inp",
        nodes: int = 1,
        ppn: int = 16,
        nprocs: int = 16,
        queue: str = "cp2k",
        depend_job_id: str | None = None,
    ) -> str:
        """
        Write a PBS submission script on the remote host and submit it.

        Args:
            depend_job_id: If given, add ``-W depend=afterok:<job_id>`` so that
                this job only starts after the dependency succeeds.

        Returns:
            The PBS job ID (e.g. "12345.server").
        """
        remote_dir = self._remote_path(*remote_parts)
        self.ensure_remote_dir(*remote_parts)

        script_content = REMOTE_PBS_TEMPLATE.format(
            job_name=job_name,
            nodes=nodes,
            ppn=ppn,
            nprocs=nprocs,
            queue=queue,
            input_file=input_file,
        )

        # Write script via SSH heredoc
        script_path = f"{remote_dir}/submit.sh"
        write_cmd = (
            f"cat > {script_path} << 'EOF'\n"
            f"{script_content}\n"
            f"EOF"
        )
        result = self._ssh(write_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to write PBS script on remote: {result.stderr}")

        # Submit
        qsub_cmd = f"cd {remote_dir} && qsub"
        if depend_job_id:
            qsub_cmd += f" -W depend=afterok:{depend_job_id}"
        qsub_cmd += " submit.sh"
        result = self._ssh(qsub_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"qsub failed on remote: {result.stderr}")

        job_id = result.stdout.strip()
        if not job_id:
            raise RuntimeError(f"qsub returned empty job ID. stdout: {result.stdout}")
        return job_id

    def submit_existing_script(
        self,
        *remote_parts: str,
        script_name: str = "submit.sh",
        depend_job_id: str | None = None,
    ) -> str:
        """
        Submit an already-uploaded PBS script on the remote host.

        Args:
            depend_job_id: If given, add ``-W depend=afterok:<job_id>`` so that
                this job only starts after the dependency succeeds.

        Returns:
            The PBS job ID (e.g. "12345.server").
        """
        remote_dir = self._remote_path(*remote_parts)
        qsub_cmd = f"cd {remote_dir} && qsub"
        if depend_job_id:
            qsub_cmd += f" -W depend=afterok:{depend_job_id}"
        qsub_cmd += f" {script_name}"
        result = self._ssh(qsub_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"qsub failed on remote: {result.stderr}")

        job_id = result.stdout.strip()
        if not job_id:
            raise RuntimeError(f"qsub returned empty job ID. stdout: {result.stdout}")
        return job_id

    # -- job status ----------------------------------------------------------

    def is_job_running(self, job_id: str) -> bool:
        """Check if a PBS job is still queued or running."""
        result = self._ssh(f"qstat {job_id} 2>&1 || true")
        stdout = result.stdout
        if "Unknown Job Id" in stdout or "qstat: Unknown Job Id" in stdout:
            return False
        lines = [l.strip() for l in stdout.split("\n") if l.strip()]
        if len(lines) < 3:
            return False
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 5:
                state = parts[4] if len(parts) > 4 else "?"
                return state in ("R", "Q", "H", "T", "W", "S")
        return False

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

        print(f"Waiting for {len(remaining)} remote PBS jobs: {sorted(remaining)}")

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
