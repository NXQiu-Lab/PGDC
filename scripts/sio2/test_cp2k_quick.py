#!/usr/bin/env python3
"""
快速验证单个远程 CP2K 任务 (SiO2 体系，4 阶段 annealing)。

用法:
    uv run python scripts/sio2/test_cp2k_quick.py --remote-config scripts/sio2/remote_config.json

脚本行为:
    1. 在 cp2k_test_verify_sio2/ 目录下生成一个小的测试体系 (SiO2, 72 atoms)
    2. 生成 4 阶段 annealing CP2K 输入文件 (GEO_OPT → heating → high_temp → quench → hold)
    3. 通过 SCP 上传到远程服务器的 <remote_work_dir>/cp2k_test_verify_sio2/
    4. 通过 SSH 在远程提交 SLURM 任务
    5. 输出 job_id 和远程/本地检查命令
    6. （可选 --wait）等待任务完成后自动下载结果并验证

注意：
    默认提交后即退出，适合本地不能一直挂机的场景。
    使用 --wait 可以在本地挂机等待任务完成并自动收集结果。
"""

import argparse
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_scripts_dir / "sio2"))
sys.path.insert(0, str(_scripts_dir.parent / "src"))

from ase.io import write

from common.remote_slurm_utils import RemoteSLURMClient, load_remote_config
from sio2_utils import build_cp2k_input_sio2, generate_sio2_structure


def _write_slurm_script(test_dir: Path, nprocs: int = 64) -> None:
    """Write a SLURM submit.slurm for the 5-phase annealing chain."""
    env_block = (
        "source /opt/software/cp2k/tools/toolchain/install/setup\n"
        "export LD_LIBRARY_PATH=/opt/software/cp2k/install/lib:$LD_LIBRARY_PATH\n"
        "export PMIX_MCA_psec=native\n"
    )

    run_block = (
        'echo "=== Step 1: GEO_OPT (relax) ==="\n'
        "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./geo_opt.inp\n"
        'if [ ! -f "./sio2_relax_geo-1.restart" ]; then\n'
        '    echo "ERROR: GEO_OPT did not produce sio2_relax_geo-1.restart; aborting."\n'
        '    exit 1\n'
        'fi\n\n'
        'echo "=== Step 2: MD heating @ 3000K ==="\n'
        "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./heating.inp\n\n"
        'echo "=== Step 3: MD high-temp @ 3500K ==="\n'
        "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./high_temp.inp\n\n"
        'echo "=== Step 4: MD quench @ 300K ==="\n'
        "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./quench.inp\n\n"
        'echo "=== Step 5: MD hold @ 300K ==="\n'
        "srun /opt/software/cp2k/install/bin/cp2k.psmp -i ./hold.inp\n"
    )

    script = (
        "#!/bin/bash\n"
        "#SBATCH --job-name=sio2_verify\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node={nprocs}\n"
        "#SBATCH --partition=compute\n"
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
    (test_dir / "submit.slurm").write_text(script)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quick remote CP2K test job for SiO2 (5-phase annealing).")
    parser.add_argument(
        "--remote-config",
        required=True,
        help="Path to remote server JSON config.",
    )
    parser.add_argument(
        "--test-dir",
        default="cp2k_test_verify_sio2",
        help="Local test directory name.",
    )
    parser.add_argument(
        "--structure",
        default="alpha_quartz",
        choices=["alpha_quartz", "beta_cristobalite"],
        help="SiO2 structure type for test.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="MD steps per annealing phase (default: 500, short for quick test).",
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

    atoms = generate_sio2_structure(
        structure_type=args.structure,
        n_atoms=72,
        temperature=300.0,
        seed=42,
    )
    write(str(test_dir / "init.xyz"), atoms)
    print(f"Generated test system: {len(atoms)} atoms")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Structure: {atoms.info.get('structure_type')}")

    # -----------------------------------------------------------------------
    # 3. Generate CP2K inputs: 5-phase annealing chain
    # -----------------------------------------------------------------------
    cp2k_data_dir = "/opt/software/cp2k/data"
    n_steps = args.n_steps

    # Phase 1: GEO_OPT (relax)
    geo_opt_inp = build_cp2k_input_sio2(
        atoms=atoms,
        project_name="sio2_relax_geo",
        temperature=300.0,
        n_steps=n_steps,
        timestep=1.0,
        output_stride=20,
        cutoff=500,
        basis_set="DZVP-MOLOPT-SR-GTH",
        scf_guess="ATOMIC",
        cp2k_data_dir=cp2k_data_dir,
        run_type="GEO_OPT",
    )

    # Phase 2: heating @ 3000K (skip relax, use GEO_OPT restart directly)
    heating_inp = build_cp2k_input_sio2(
        atoms=atoms,
        project_name="sio2_heating",
        temperature=3000.0,
        n_steps=n_steps,
        timestep=1.0,
        output_stride=20,
        cutoff=500,
        basis_set="DZVP-MOLOPT-SR-GTH",
        scf_guess="RESTART",
        use_ot=True,
        wfn_restart_file="sio2_relax_geo-RESTART.wfn",
        ext_restart_file="sio2_relax_geo-1.restart",
        cp2k_data_dir=cp2k_data_dir,
    )

    # Phase 4: high_temp @ 3500K
    high_temp_inp = build_cp2k_input_sio2(
        atoms=atoms,
        project_name="sio2_high_temp",
        temperature=3500.0,
        n_steps=n_steps,
        timestep=1.0,
        output_stride=20,
        cutoff=500,
        basis_set="DZVP-MOLOPT-SR-GTH",
        scf_guess="RESTART",
        use_ot=True,
        wfn_restart_file="sio2_heating-RESTART.wfn",
        ext_restart_file="sio2_heating-1.restart",
        cp2k_data_dir=cp2k_data_dir,
    )

    # Phase 5: quench @ 300K
    quench_inp = build_cp2k_input_sio2(
        atoms=atoms,
        project_name="sio2_quench",
        temperature=300.0,
        n_steps=n_steps,
        timestep=1.0,
        output_stride=20,
        cutoff=500,
        basis_set="DZVP-MOLOPT-SR-GTH",
        scf_guess="RESTART",
        use_ot=True,
        wfn_restart_file="sio2_high_temp-RESTART.wfn",
        ext_restart_file="sio2_high_temp-1.restart",
        cp2k_data_dir=cp2k_data_dir,
    )

    # Phase 6: hold @ 300K
    hold_inp = build_cp2k_input_sio2(
        atoms=atoms,
        project_name="sio2_hold",
        temperature=300.0,
        n_steps=n_steps,
        timestep=1.0,
        output_stride=20,
        cutoff=500,
        basis_set="DZVP-MOLOPT-SR-GTH",
        scf_guess="RESTART",
        use_ot=True,
        wfn_restart_file="sio2_quench-RESTART.wfn",
        ext_restart_file="sio2_quench-1.restart",
        cp2k_data_dir=cp2k_data_dir,
    )

    (test_dir / "geo_opt.inp").write_text(geo_opt_inp)
    (test_dir / "heating.inp").write_text(heating_inp)
    (test_dir / "high_temp.inp").write_text(high_temp_inp)
    (test_dir / "quench.inp").write_text(quench_inp)
    (test_dir / "hold.inp").write_text(hold_inp)

    _write_slurm_script(test_dir)

    print(f"\n测试目录: {test_dir.resolve()}")
    print("生成文件:")
    for f in ("geo_opt.inp", "heating.inp", "high_temp.inp", "quench.inp", "hold.inp", "submit.slurm"):
        print(f"  {test_dir / f}")

    if args.dry_run:
        print("\n[DRY-RUN] Would upload and submit remote SLURM job.")
        print(f"  Local dir : {test_dir.resolve()}")
        print(f"  Remote dir: {remote_cfg.remote_work_dir}/cp2k_test_verify_sio2")
        return 0

    # -----------------------------------------------------------------------
    # 4. Create client and connect to remote (only when not dry-run)
    # -----------------------------------------------------------------------
    client = RemoteSLURMClient(remote_cfg)

    # -----------------------------------------------------------------------
    # 5. Upload to remote
    # -----------------------------------------------------------------------
    print("\nUploading test files to remote...")
    try:
        client.upload_run(test_dir, "cp2k_test_verify_sio2")
        print("  Upload complete.")
    except RuntimeError as exc:
        print(f"  ERROR: Upload failed: {exc}")
        return 1

    # -----------------------------------------------------------------------
    # 6. Submit SLURM job on remote
    # -----------------------------------------------------------------------
    print("\nSubmitting remote SLURM job (5-phase annealing)...")
    try:
        job_id = client.submit_existing_script("cp2k_test_verify_sio2", script_name="submit.slurm")
    except RuntimeError as exc:
        print(f"  ERROR: Job submission failed: {exc}")
        return 1

    print(f"✅ 任务已提交: job_id = {job_id}")
    print(f"   远程工作目录: {remote_cfg.remote_work_dir}/cp2k_test_verify_sio2")

    # -----------------------------------------------------------------------
    # 7. Print check commands
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("远程检查命令 (在服务器上执行):")
    print(f"   squeue -j {job_id}")
    print(f"   tail -f {remote_cfg.remote_work_dir}/cp2k_test_verify_sio2/out")
    print(f"   cat {remote_cfg.remote_work_dir}/cp2k_test_verify_sio2/err")

    print("\n本地检查命令:")
    print(f"   ssh {remote_cfg.username}@{remote_cfg.host} 'squeue -j {job_id}'")

    print("\n成功标志 (约 30~60 分钟后检查):")
    print("   1. out 文件中依次出现 GEO_OPT → heating → high_temp → quench → hold 的输出")
    print("   2. 远程目录下生成 sio2_hold-pos-1.xyz 轨迹文件")
    print("   3. err 文件为空或只有警告")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 8. Optional: wait and collect
    # -----------------------------------------------------------------------
    if args.wait:
        print("\n--wait 模式: 等待任务完成并自动下载结果...")
        client.wait_for_jobs([job_id], poll_interval=30)

        print("\nDownloading results...")
        try:
            downloaded = client.download_results(test_dir, "cp2k_test_verify_sio2")
            print(f"  Downloaded: {downloaded}")
        except RuntimeError as exc:
            print(f"  WARNING: download failed: {exc}")

        pos_xyz = test_dir / "sio2_hold-pos-1.xyz"
        ener_file = test_dir / "sio2_hold.ener"

        if pos_xyz.exists():
            print(f"\n✅ 轨迹文件已下载: {pos_xyz}")
        else:
            print(f"\n⚠️  未找到轨迹文件: {pos_xyz}")

        if ener_file.exists():
            print(f"✅ 能量文件已下载: {ener_file}")
        else:
            print(f"⚠️  未找到能量文件: {ener_file}")

        print("\n快速验证完成。")
    else:
        print("\n脚本已退出。任务在远程运行中。")
        print("稍后运行以下命令收集结果:")
        print(f"   uv run python scripts/sio2/test_cp2k_quick.py ")
        print(f"       --remote-config {args.remote_config} --wait")
        print("\n或者手动下载:")
        print(f"   scp -r {remote_cfg.username}@{remote_cfg.host}:{remote_cfg.remote_work_dir}/cp2k_test_verify_sio2/* {test_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
