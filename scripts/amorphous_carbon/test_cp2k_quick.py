#!/usr/bin/env python3
"""快速验证单个远程 CP2K 任务（无定形碳体系，64 原子）。

用法:
    uv run python scripts/amorphous_carbon/test_cp2k_quick.py \
        --remote-config scripts/amorphous_carbon/remote_config.json

脚本行为:
    1. 在 cp2k_test_verify_amorphous_carbon/ 目录下生成一个 64 原子无定形碳测试结构
    2. 生成短 CP2K 输入文件（默认 1 ps NVT @ 300 K，可改为短退火）
    3. 通过 SCP 上传到远程服务器的 <remote_work_dir>/cp2k_test_verify_amorphous_carbon/
    4. 通过 SSH 在远程提交 SLURM 任务
    5. 输出 job_id 和远程/本地检查命令
    6. （可选 --wait）等待任务完成后自动下载结果并验证

注意:
    默认提交后即退出，适合本地不能一直挂机的场景。
    使用 --wait 可以在本地挂机等待任务完成并自动收集结果。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "amorphous_carbon"))
sys.path.insert(0, str(_scripts_dir.parent / "src"))

from ase.io import write

from amorphous_carbon_utils import (
    build_cp2k_input_carbon,
    generate_amorphous_carbon_seed,
    get_density,
    get_sp3_fraction,
)
from common.remote_slurm_utils import RemoteSLURMClient, load_remote_config


def _write_slurm_script(test_dir: Path, input_file: str = "input.inp", nprocs: int = 64) -> None:
    """Write a SLURM submit.slurm for a single CP2K run."""
    script = f"""#!/bin/bash
#SBATCH --job-name=ac_verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --partition=compute
#SBATCH --output=out
#SBATCH --error=err

# Handle path difference between head (/home/work) and compute nodes (/work)
SUBMIT_DIR="${{SLURM_SUBMIT_DIR:-$(pwd)}}"
HEAD_DIR="$SUBMIT_DIR"
NODE_DIR="${{SUBMIT_DIR/\\/home\\/work/\\/work}}"
if [ -d "$HEAD_DIR" ]; then
    cd "$HEAD_DIR"
elif [ -d "$NODE_DIR" ]; then
    cd "$NODE_DIR"
else
    echo "ERROR: Cannot find work directory"
    exit 1
fi

source /opt/software/cp2k/tools/toolchain/install/setup
export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH
export PMIX_MCA_psec=native

srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./{input_file}
"""
    (test_dir / "submit.slurm").write_text(script)
    (test_dir / "submit.slurm").chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quick remote CP2K test job for amorphous carbon.")
    parser.add_argument(
        "--remote-config",
        required=True,
        help="Path to remote server JSON config.",
    )
    parser.add_argument(
        "--test-dir",
        default="cp2k_test_verify_amorphous_carbon",
        help="Local test directory name.",
    )
    parser.add_argument(
        "--run-type",
        default="MD",
        choices=["MD", "ANNEALING", "NPT", "BALLISTIC"],
        help="CP2K run type for the quick test.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1000,
        help="Number of MD steps for the quick test (default 1000 = 1 ps).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Target temperature (K).",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=10.0,
        help="Target pressure (GPa) for NPT run type.",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=64,
        help="Number of MPI processes for SLURM job.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the job to finish, then download results and verify.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without connecting to remote.",
    )
    args = parser.parse_args(argv)

    # -----------------------------------------------------------------------
    # 1. Load remote config
    # -----------------------------------------------------------------------
    remote_cfg = load_remote_config(args.remote_config)

    # -----------------------------------------------------------------------
    # 2. Generate test system locally
    # -----------------------------------------------------------------------
    test_dir = Path(args.test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    atoms = generate_amorphous_carbon_seed(
        n_atoms=64,
        target_density_g_cm3=2.0,
        temperature=args.temperature,
        seed=42,
    )
    write(str(test_dir / "init.xyz"), atoms)
    print(f"Generated test system: {len(atoms)} atoms")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Density: {get_density(atoms):.3f} g/cm³")
    print(f"  sp3 fraction (seed): {get_sp3_fraction(atoms):.3f}")

    # -----------------------------------------------------------------------
    # 3. Generate CP2K input
    # -----------------------------------------------------------------------
    if args.run_type == "MD":
        project_name = "ac_verify_md"
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=project_name,
            run_type="MD",
            temperature=args.temperature,
            n_steps=args.n_steps,
        )
    elif args.run_type == "ANNEALING":
        project_name = "ac_verify_anneal"
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=project_name,
            run_type="ANNEALING",
            temperature=args.temperature,
            n_steps=args.n_steps,
            annealing_start_temp=args.temperature,
            annealing_stop_temp=300.0,
        )
    elif args.run_type == "NPT":
        project_name = "ac_verify_npt"
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=project_name,
            run_type="NPT",
            ensemble="NPT_I",
            temperature=args.temperature,
            pressure_gpa=args.pressure,
            n_steps=args.n_steps,
        )
    elif args.run_type == "BALLISTIC":
        project_name = "ac_verify_ballistic"
        inp = build_cp2k_input_carbon(
            atoms,
            project_name=project_name,
            run_type="BALLISTIC",
            temperature=args.temperature,
            ballistic_velocity_temp=2000.0,
            ballistic_ensemble="NVE",
            n_steps=args.n_steps,
        )
    else:
        raise ValueError(f"Unsupported run type: {args.run_type}")

    input_file = "input.inp"
    (test_dir / input_file).write_text(inp)
    print(f"  Written CP2K input: {test_dir / input_file}")

    # -----------------------------------------------------------------------
    # 4. Write SLURM script
    # -----------------------------------------------------------------------
    _write_slurm_script(test_dir, input_file=input_file, nprocs=args.nprocs)

    if args.dry_run:
        print("\n[DRY-RUN] Would upload and submit:")
        print(f"  Local dir: {test_dir}")
        print(f"  Remote subdir: cp2k_test_verify_amorphous_carbon")
        return 0

    # -----------------------------------------------------------------------
    # 5. Upload and submit
    # -----------------------------------------------------------------------
    client = RemoteSLURMClient(remote_cfg)
    try:
        remote_subdir = "cp2k_test_verify_amorphous_carbon"
        client.upload_run(test_dir, remote_subdir)
        job_id = client.submit_existing_script(remote_subdir, script_name="submit.slurm")

        print(f"\nSubmitted test job: {job_id}")
        print(f"  Remote subdir: {remote_cfg.remote_work_dir}/{remote_subdir}")
        print(f"  Local dir: {test_dir}")
        print("\nCheck remote status with:")
        print(f"  ssh {remote_cfg.username}@{remote_cfg.host} 'squeue -j {job_id}'")
        print("\nCheck remote output with:")
        print(f"  ssh {remote_cfg.username}@{remote_cfg.host} 'tail -f {remote_cfg.remote_work_dir}/{remote_subdir}/out'")

        # -------------------------------------------------------------------
        # 6. Optional wait + download + verify
        # -------------------------------------------------------------------
        if args.wait:
            print("\nWaiting for job to finish...")
            while client.is_job_running(job_id):
                time.sleep(30)

            print("Job finished. Downloading results...")
            downloaded = client.download_results(
                test_dir,
                remote_subdir,
                patterns=["*-pos-1.xyz", "*-frc-1.xyz", "*.ener", "*-RESTART.wfn", "out", "err"],
            )
            print(f"Downloaded: {downloaded}")

            pos_file = test_dir / f"{project_name}-pos-1.xyz"
            ener_file = test_dir / f"{project_name}-1.ener"
            if not ener_file.exists():
                ener_file = test_dir / f"{project_name}.ener"
            if pos_file.exists():
                frames = read(str(pos_file), index=":")
                print(f"  Trajectory frames: {len(frames)}")
                print(f"  Final density: {get_density(frames[-1]):.3f} g/cm³")
                print(f"  Final sp3 fraction: {get_sp3_fraction(frames[-1]):.3f}")
            if ener_file.exists():
                from common.cp2k_utils import parse_cp2k_ener
                records = parse_cp2k_ener(ener_file)
                if records:
                    e_final = records[-1]["potential_energy"]
                    print(f"  Final potential energy: {e_final:.3f} eV")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
