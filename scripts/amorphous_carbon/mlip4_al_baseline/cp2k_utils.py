#!/usr/bin/env python3
"""CP2K single-point execution via local mpirun or SLURM, plus output parsing."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = 51.42208619083232


def build_cp2k_input(
    atoms: Atoms,
    params: dict[str, Any],
    project_name: str,
    run_type: str = "ENERGY_FORCE",
) -> str:
    """Build a CP2K ENERGY_FORCE input for a single configuration."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    xc = params["xc_functional"].upper()

    coord_lines = []
    for sym, pos in zip(symbols, positions):
        coord_lines.append(
            "      {:2s} {:>18.10f} {:>18.10f} {:>18.10f}".format(sym, pos[0], pos[1], pos[2])
        )

    scf_guess = params.get("scf_guess", "ATOMIC")
    use_smear = params.get("use_smear", True)

    lines = [
        "&GLOBAL",
        "  PROJECT {}".format(project_name),
        "  RUN_TYPE {}".format(run_type),
        "  PRINT_LEVEL MEDIUM",
        "&END GLOBAL",
        "",
        "&FORCE_EVAL",
        "  METHOD Quickstep",
        "  &PRINT",
        "    &FORCES ON",
        "    &END FORCES",
        "  &END PRINT",
        "  &DFT",
        "    BASIS_SET_FILE_NAME {}".format(params["basis_set_file"]),
        "    POTENTIAL_FILE_NAME {}".format(params["potential_file"]),
        "    &QS",
        "      EPS_DEFAULT {}".format(params.get("eps_default", "1.0E-12")),
        "      EXTRAPOLATION PS",
        "      EXTRAPOLATION_ORDER 3",
        "    &END QS",
        "    &MGRID",
        "      CUTOFF {}".format(params["cutoff"]),
        "      REL_CUTOFF {}".format(params.get("rel_cutoff", 50)),
        "      NGRIDS {}".format(params.get("ngrids", 4)),
        "    &END MGRID",
        "    &XC",
        "      &XC_FUNCTIONAL",
        "        &{}".format(xc),
        "        &END {}".format(xc),
        "      &END XC_FUNCTIONAL",
        "    &END XC",
        "    &SCF",
        "      SCF_GUESS {}".format(scf_guess),
        "      EPS_SCF {}".format(params.get("scf_eps", "1.0E-5")),
        "      MAX_SCF {}".format(params.get("scf_max_iter", 300)),
        "      ADDED_MOS 100",
    ]
    if use_smear:
        lines.extend([
            "      &SMEAR ON",
            "        METHOD FERMI_DIRAC",
            "        ELECTRONIC_TEMPERATURE [K] 1000.0",
            "      &END SMEAR",
        ])
    lines.extend([
        "      &DIAGONALIZATION ON",
        "        ALGORITHM STANDARD",
        "      &END DIAGONALIZATION",
        "      &MIXING",
        "        METHOD BROYDEN_MIXING",
        "        ALPHA 0.2",
        "        NBUFFER 8",
        "      &END MIXING",
        "    &END SCF",
        "  &END DFT",
        "  &SUBSYS",
        "    &CELL",
    ])

    for i, label in enumerate(["A", "B", "C"]):
        vec = cell[i]
        lines.append(
            "      {} {:.10f} {:.10f} {:.10f}".format(label, vec[0], vec[1], vec[2])
        )
    lines.extend([
        "      PERIODIC XYZ",
        "    &END CELL",
        "    &COORD",
    ])
    lines.extend(coord_lines)
    lines.extend([
        "    &END COORD",
        "    &KIND C",
        "      BASIS_SET {}".format(params["basis_set"]),
        "      POTENTIAL {}".format(params["potential"]),
        "    &END KIND",
        "  &END SUBSYS",
        "&END FORCE_EVAL",
    ])

    return "\n".join(lines)


def parse_cp2k_energy(out_text: str) -> float | None:
    """Extract total energy in eV from CP2K output text."""
    for line in out_text.splitlines():
        m = re.search(r"Total energy:\s+([-\d.]+)\s+Ha", line)
        if m:
            return float(m.group(1)) * HA_TO_EV
        m = re.search(r"ENERGY\|.*energy \(a\.u\.\):\s+([-\d.]+)", line)
        if m:
            return float(m.group(1)) * HA_TO_EV
        m = re.search(r"ENERGY\|.*energy \[hartree\]\s+([-\d.]+)", line)
        if m:
            return float(m.group(1)) * HA_TO_EV
    return None


def parse_cp2k_forces(out_text: str, natoms: int) -> np.ndarray | None:
    """Extract atomic forces in eV/A from CP2K output text.

    Supports the 'FORCES| Atomic forces [hartree/bohr]' block produced by
    CP2K's &PRINT / &FORCES ON section for ENERGY_FORCE runs.
    """
    forces: list[list[float]] = []
    in_forces = False
    for line in out_text.splitlines():
        if "ATOMIC FORCES in [a.u.]" in line or "FORCES| Atomic forces" in line:
            in_forces = True
            forces = []
            continue
        if not in_forces:
            continue
        stripped = line.strip()
        if stripped.startswith("FORCES|"):
            stripped = stripped[7:].strip()
        # Skip header, separator, and summary lines
        if stripped == "" or stripped.startswith("Atom") or stripped.startswith("x"):
            continue
        if stripped.startswith("Sum") or stripped.startswith("Total atomic force"):
            continue
        parts = stripped.split()
        if len(parts) >= 4:
            try:
                # First token is the atom index, then x/y/z components
                fx = float(parts[1])
                fy = float(parts[2])
                fz = float(parts[3])
                forces.append([fx, fy, fz])
            except ValueError:
                pass
        if len(forces) >= natoms:
            break

    if len(forces) == natoms:
        return np.array(forces) * HA_BOHR_TO_EV_ANG
    return None


def cp2k_finished_normally(out_text: str) -> bool:
    return "PROGRAM ENDED AT" in out_text or "NORMAL COMPLETION" in out_text


def _setup_cp2k_env(cp2k_config: dict[str, Any], env: dict[str, str]) -> dict[str, str]:
    """Set PATH, LD_LIBRARY_PATH, CP2K_DATA_DIR for CP2K execution."""
    paths = []
    if cp2k_config.get("cp2k_path"):
        paths.append(cp2k_config["cp2k_path"])
    mpirun = cp2k_config.get("mpirun", "mpirun")
    mpirun_parent = Path(mpirun).parent
    if str(mpirun_parent) != ".":
        paths.append(str(mpirun_parent))
    paths.extend(["/usr/bin", "/bin"])
    env["PATH"] = ":".join(paths)

    if cp2k_config.get("cp2k_data"):
        env["CP2K_DATA_DIR"] = cp2k_config["cp2k_data"]
    if cp2k_config.get("cp2k_lib"):
        env["LD_LIBRARY_PATH"] = "{}:{}".format(cp2k_config["cp2k_lib"], env.get("LD_LIBRARY_PATH", ""))
    return env


def _run_cp2k_local(
    inp_path: Path,
    out_path: Path,
    cp2k_config: dict[str, Any],
    work_dir: Path,
) -> tuple[bool, str]:
    """Run CP2K locally via mpirun. Returns (finished_normally, out_text)."""
    n_cores = cp2k_config.get("n_cores", 1)
    cmd = [
        cp2k_config.get("mpirun", "mpirun"),
        "-np", str(n_cores),
    ]
    if cp2k_config.get("mpirun_extra_args"):
        cmd.extend(cp2k_config["mpirun_extra_args"].split())
    cmd.extend([
        cp2k_config.get("cp2k_cmd", "cp2k.psmp"),
        "-o", str(out_path),
        str(inp_path),
    ])

    env = _setup_cp2k_env(cp2k_config, os.environ.copy())
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = "512M"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work_dir)
    out_text = out_path.read_text(encoding="utf-8") if out_path.exists() else proc.stdout
    return cp2k_finished_normally(out_text) and proc.returncode == 0, out_text


def _build_slurm_script(
    cp2k_config: dict[str, Any],
    label: str,
    work_dir: Path,
    out_path: Path,
    inp_path: Path,
) -> Path:
    """Write a SLURM submission script for CP2K and return its path."""
    n_cores = cp2k_config.get("n_cores", 1)
    partition = cp2k_config.get("partition", "compute")
    time_limit = cp2k_config.get("time_limit", "01:00:00")
    nodes = cp2k_config.get("nodes", 1)
    nodelist = cp2k_config.get("nodelist", "")
    exclude = cp2k_config.get("exclude", "")
    ntasks_per_node = n_cores // nodes
    mpirun = cp2k_config.get("mpirun", "mpirun")
    cp2k_cmd = cp2k_config.get("cp2k_cmd", "cp2k.psmp")
    cp2k_path = cp2k_config.get("cp2k_path", "")
    cp2k_data = cp2k_config.get("cp2k_data", "")
    cp2k_lib = cp2k_config.get("cp2k_lib", "")

    script_path = work_dir / "{}.slurm".format(label)
    script = "#!/bin/bash\n"
    script += "#SBATCH --job-name={}\n".format(label)
    script += "#SBATCH --partition={}\n".format(partition)
    script += "#SBATCH --nodes={}\n".format(nodes)
    script += "#SBATCH --ntasks-per-node={}\n".format(ntasks_per_node)
    if nodelist:
        script += "#SBATCH --nodelist={}\n".format(nodelist)
    if exclude:
        script += "#SBATCH --exclude={}\n".format(exclude)
    script += "#SBATCH --time={}\n".format(time_limit)
    script += "#SBATCH --output={}\n".format(work_dir / "{}.slurm.log".format(label))
    script += "\n"
    script += "export PATH={}:{}:/usr/bin:/bin:$PATH\n".format(cp2k_path, Path(mpirun).parent)
    script += "export CP2K_DATA_DIR={}\n".format(cp2k_data)
    # Source the toolchain setup so that OpenBLAS/ELPA/HDF5/etc. are found,
    # then prepend the CP2K install lib directory.
    script += "source /opt/software/cp2k/tools/toolchain/install/setup\n"
    script += "export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH\n".format(cp2k_lib)
    # CP2K psmp uses both MPI and OpenMP; prevent catastrophic oversubscription
    # by pinning each MPI rank to a single OpenMP thread.  Total cores = n_cores.
    script += "export OMP_NUM_THREADS=1\n"
    script += "export OMP_STACKSIZE=512M\n"
    script += "\n"
    script += "{} -np {} --bind-to none {} -o {} {}\n".format(mpirun, n_cores, cp2k_cmd, out_path, inp_path)
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _submit_slurm_script(script_path: Path) -> str:
    """Submit a SLURM script and return the job ID."""
    submit = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = None
    for line in submit.stdout.splitlines():
        if "Submitted batch job" in line:
            job_id = line.strip().split()[-1]
            break
    if job_id is None:
        raise RuntimeError("Failed to parse SLURM job ID: {}".format(submit.stdout))
    return job_id


def _run_cp2k_slurm(
    inp_path: Path,
    out_path: Path,
    cp2k_config: dict[str, Any],
    work_dir: Path,
    label: str,
) -> tuple[bool, str]:
    """Submit CP2K via SLURM sbatch and poll until completion."""
    script_path = _build_slurm_script(cp2k_config, label, work_dir, out_path, inp_path)
    job_id = _submit_slurm_script(script_path)
    logger.info("  Submitted SLURM job {} for {}".format(job_id, label))

    # Poll until job leaves queue
    while True:
        time.sleep(10)
        squeue = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True,
            text=True,
        )
        if squeue.returncode != 0 or not squeue.stdout.strip():
            # Job no longer in queue; wait for file flush
            time.sleep(5)
            break

    out_text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    return cp2k_finished_normally(out_text), out_text


def run_cp2k_single_point(
    atoms: Atoms,
    cp2k_config: dict[str, Any],
    work_dir: Path,
    label: str,
    timing: Any | None = None,
    iteration: int | None = None,
) -> dict[str, Any]:
    """Run a CP2K ENERGY_FORCE single-point calculation (local or SLURM).

    Returns a dict with energy, forces, converged flag, wall time, and core-hours.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    params = cp2k_config.get("input_params", {})
    inp_text = build_cp2k_input(atoms, params, project_name=label)
    inp_path = work_dir / "{}.inp".format(label)
    out_path = work_dir / "{}.out".format(label)
    inp_path.write_text(inp_text, encoding="utf-8")

    n_cores = cp2k_config.get("n_cores", 1)
    mode = cp2k_config.get("mode", "local").lower()

    logger.info("  Running CP2K ({}): {} ({} atoms, {} cores)".format(mode, label, len(atoms), n_cores))
    if timing is not None:
        timing.start("cp2k_dft", n_cores=n_cores, iteration=iteration, label=label)
    t0 = time.perf_counter()

    if mode == "slurm":
        converged, out_text = _run_cp2k_slurm(inp_path, out_path, cp2k_config, work_dir, label)
    else:
        converged, out_text = _run_cp2k_local(inp_path, out_path, cp2k_config, work_dir)

    wall = time.perf_counter() - t0
    if timing is not None:
        timing.stop("cp2k_dft", iteration=iteration, label=label)

    energy = parse_cp2k_energy(out_text) if converged else None
    forces = parse_cp2k_forces(out_text, len(atoms)) if converged else None

    core_hours = wall * n_cores / 3600.0
    energy_str = "E={:.4f} eV".format(energy) if energy is not None else "E=None"
    logger.info("  {}: converged={}, {}, wall={:.1f}s, core_hours={:.4f}".format(
        label, converged, energy_str, wall, core_hours))

    return {
        "converged": converged,
        "energy": energy,
        "forces": forces,
        "wall_time_seconds": wall,
        "core_hours": core_hours,
        "n_cores": n_cores,
        "output_file": str(out_path),
    }


def run_cp2k_single_points_batch(
    frames: list[Atoms],
    cp2k_config: dict[str, Any],
    work_dir: Path,
    labels: list[str],
    timing: Any | None = None,
    iteration: int | None = None,
    max_parallel: int = 4,
) -> list[dict[str, Any]]:
    """Run many CP2K single-points via SLURM with bounded parallelism.

    Submits SLURM jobs one by one but keeps at most `max_parallel` jobs in the
    queue/running state at any time.  This avoids flooding the scheduler while
    still getting much higher throughput than sequential submission.

    Returns a list of result dicts in the same order as `frames`/`labels`.
    """
    if len(frames) != len(labels):
        raise ValueError("frames and labels must have the same length")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    params = cp2k_config.get("input_params", {})
    n_cores = cp2k_config.get("n_cores", 1)
    mode = cp2k_config.get("mode", "local").lower()

    if mode != "slurm":
        raise ValueError("run_cp2k_single_points_batch only supports SLURM mode")

    # Prepare all inputs and scripts
    jobs: list[dict[str, Any]] = []
    for atoms, label in zip(frames, labels):
        sub_dir = work_dir / label
        sub_dir.mkdir(parents=True, exist_ok=True)
        inp_path = sub_dir / "{}.inp".format(label)
        out_path = sub_dir / "{}.out".format(label)
        inp_text = build_cp2k_input(atoms, params, project_name=label)
        inp_path.write_text(inp_text, encoding="utf-8")
        script_path = _build_slurm_script(cp2k_config, label, sub_dir, out_path, inp_path)
        jobs.append({
            "label": label,
            "atoms": atoms,
            "inp_path": inp_path,
            "out_path": out_path,
            "script_path": script_path,
        })

    logger.info("  Batch CP2K: {} jobs, max_parallel={}".format(len(jobs), max_parallel))

    # Maintain active jobs and results
    pending = list(jobs)
    active: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    job_submit_time: dict[str, float] = {}

    def submit_next() -> None:
        if not pending:
            return
        info = pending.pop(0)
        job_id = _submit_slurm_script(info["script_path"])
        active[job_id] = info
        job_submit_time[job_id] = time.perf_counter()
        logger.info("    Submitted SLURM job {} for {}".format(job_id, info["label"]))

    # Seed initial batch
    for _ in range(min(max_parallel, len(pending))):
        submit_next()

    # Poll until all jobs finish, submitting new ones as slots free up
    while active:
        time.sleep(10)
        squeue = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h"],
            capture_output=True,
            text=True,
        )
        running_ids = set()
        for line in squeue.stdout.splitlines():
            parts = line.split()
            if parts:
                running_ids.add(parts[0])

        finished = [jid for jid in active if jid not in running_ids]
        for jid in finished:
            info = active.pop(jid)
            wall = time.perf_counter() - job_submit_time.pop(jid)
            out_text = info["out_path"].read_text(encoding="utf-8") if info["out_path"].exists() else ""
            converged = cp2k_finished_normally(out_text)
            natoms = len(info["atoms"])
            energy = parse_cp2k_energy(out_text) if converged else None
            forces = parse_cp2k_forces(out_text, natoms) if converged else None
            core_hours = wall * n_cores / 3600.0

            if timing is not None:
                timing.record(
                    "cp2k_dft",
                    n_cores=n_cores,
                    wall_time_seconds=wall,
                    iteration=iteration,
                    label=info["label"],
                )

            energy_str = "E={:.4f} eV".format(energy) if energy is not None else "E=None"
            logger.info("    {}: converged={}, {}, wall={:.1f}s, core_hours={:.4f}".format(
                info["label"], converged, energy_str, wall, core_hours))

            results[info["label"]] = {
                "converged": converged,
                "energy": energy,
                "forces": forces,
                "wall_time_seconds": wall,
                "core_hours": core_hours,
                "n_cores": n_cores,
                "output_file": str(info["out_path"]),
            }

            # Fill the slot with the next pending job
            submit_next()

    # Return in original order
    return [results[j["label"]] for j in jobs]
