"""Unified CLI entrypoints for project tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

from .acquisition import AcquisitionEvaluator
from .benchmark import BenchmarkRunner, DatasetSourceSpec, TrainingTaskSpec
from .domain import DatasetBundle, DatasetProfile, StructureRecord
from .io import AseDbDatasetReader, DatasetWriter, PoscarReader
from .sampling.utils import ase_to_record
from .metric_core.results import MetricResult
from .presentation import PresentationExporter, RunManifest
from .profiling import CORE_METRIC_IDS, DatasetProfiler, PoolBuilder, build_default_registry, list_available_metric_ids
from .sampling.cli import sampling_mode_command
from .tasks import build_task_name, get_task_definition, list_task_families, prepare_output_directory
from .utils.docgen import main as generate_docs_main

DEFAULT_PROJECT_VERSION = "1.3.0"


def _normalize_metric_ids(raw_metrics: list[str] | None) -> tuple[str, ...]:
    if not raw_metrics:
        return ()
    flattened: list[str] = []
    for raw in raw_metrics:
        for item in raw.split(","):
            metric_id = item.strip()
            if metric_id:
                flattened.append(metric_id)
    deduped: list[str] = []
    seen: set[str] = set()
    for metric_id in flattened:
        if metric_id not in seen:
            deduped.append(metric_id)
            seen.add(metric_id)
    return tuple(deduped)


def _resolve_selected_metrics(raw_metrics: list[str] | None, preset: str) -> tuple[str, ...]:
    selected = _normalize_metric_ids(raw_metrics)
    if selected:
        return selected
    if preset == "core":
        return CORE_METRIC_IDS
    if preset == "full":
        return list_available_metric_ids()
    raise ValueError(f"Unsupported metric preset: {preset}")


def _build_profile_from_source(
    source_type: str,
    datapath: str | None,
    molecule: str | None,
    limit: int | None,
) -> DatasetBundle:
    normalized_source = source_type.strip()
    if normalized_source == "ase_db":
        if not datapath:
            raise ValueError("ase_db source requires --datapath.")
        return AseDbDatasetReader(datapath).read(limit=limit)

    if normalized_source in {"md17", "md22"}:
        if not molecule:
            raise ValueError(f"{normalized_source} source requires --molecule.")
        import schnetpack.transform as trn
        from schnetpack.datasets import MD17, MD22

        dataset_class = MD17 if normalized_source == "md17" else MD22
        energy_key = dataset_class.energy
        force_key = dataset_class.forces
        resolved_datapath = datapath or str(Path("data") / f"{normalized_source}_{molecule}.db")
        datamodule = dataset_class(
            datapath=resolved_datapath,
            molecule=molecule,
            batch_size=1,
            num_train=0.8,
            num_val=0.1,
            num_test=0.1,
            transforms=[trn.CastTo32()],
            num_workers=0,
            pin_memory=False,
        )
        datamodule.prepare_data()
        datamodule.setup()
        dataset = datamodule.dataset
        total = len(dataset) if limit is None else min(limit, len(dataset))
        records: list[StructureRecord] = []
        for index in range(total):
            sample = dataset[index]
            atomic_numbers = _extract_1d_ints(sample, ("_atomic_numbers", "atomic_numbers", "numbers", "Z"))
            if not atomic_numbers:
                continue
            positions = _extract_2d_vec3(sample, ("_positions", "positions", "R"))
            forces = _extract_2d_vec3(sample, (force_key, "forces", "_forces"))
            energy = _extract_scalar(sample, (energy_key, "energy", "E"))
            cell = _extract_2d_vec3(sample, ("_cell", "cell"))[:3]
            pbc_values = _extract_1d_bool(sample, ("_pbc", "pbc"))
            records.append(
                StructureRecord(
                    sample_id=index + 1,
                    atomic_numbers=atomic_numbers,
                    positions=positions,
                    cell=tuple(cell) if cell else None,
                    pbc=pbc_values if pbc_values else (False, False, False),
                    energy=energy,
                    forces=forces,
                )
            )
        return DatasetBundle(
            dataset_name=f"{normalized_source}:{molecule}",
            records=tuple(records),
            metadata={
                "source_type": normalized_source,
                "datapath": resolved_datapath,
                "molecule": molecule,
            },
        )

    raise ValueError(f"Unsupported source type: {source_type}")


def _extract_1d_ints(sample: dict[str, Any], keys: tuple[str, ...]) -> tuple[int, ...]:
    value = _pick_first(sample, keys)
    if value is None:
        return ()
    flattened = _to_python(value)
    if not isinstance(flattened, list):
        flattened = [flattened]
    return tuple(int(item) for item in flattened)


def _extract_1d_bool(sample: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, bool, bool]:
    value = _pick_first(sample, keys)
    if value is None:
        return ()
    flattened = _to_python(value)
    if not isinstance(flattened, list):
        flattened = [flattened]
    values = tuple(bool(item) for item in flattened[:3])
    if len(values) == 3:
        return values
    return ()


def _extract_2d_vec3(sample: dict[str, Any], keys: tuple[str, ...]) -> tuple[tuple[float, float, float], ...]:
    value = _pick_first(sample, keys)
    if value is None:
        return ()
    matrix = _to_python(value)
    if not isinstance(matrix, list):
        return ()
    vectors: list[tuple[float, float, float]] = []
    for row in matrix:
        if isinstance(row, list) and len(row) >= 3:
            vectors.append((float(row[0]), float(row[1]), float(row[2])))
    return tuple(vectors)


def _extract_scalar(sample: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    value = _pick_first(sample, keys)
    if value is None:
        return None
    payload = _to_python(value)
    if isinstance(payload, list):
        if not payload:
            return None
        payload = payload[0]
    try:
        return float(payload)
    except (TypeError, ValueError):
        return None


def _pick_first(sample: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _serialize_metric_result(result: MetricResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["sample_count"] = len(result.sample_records)
    return payload


def _split_csv_metrics(csv_value: str) -> tuple[str, ...]:
    if not csv_value:
        return ()
    return tuple(item for item in csv_value.split(",") if item)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atom-dataset-index",
        description="Unified task runner for atom dataset indexing workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    docs_parser = subparsers.add_parser(
        "generate-docs",
        help="Generate leaf README files and package-level aggregated docs.",
    )
    docs_parser.add_argument(
        "--project-version",
        default=DEFAULT_PROJECT_VERSION,
        help="Project version written into generated documentation.",
    )
    docs_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="ISO date used in generated documentation headers.",
    )

    subparsers.add_parser(
        "list-tasks",
        help="Show supported task families exposed by the repository.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-run",
        help="Create a runtime output directory for a task family/subtask.",
    )
    prepare_parser.add_argument("family", help="Task family name.")
    prepare_parser.add_argument(
        "--subtask",
        default=None,
        help="Optional subtask name registered under the task family.",
    )
    prepare_parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory used for runtime task outputs.",
    )
    prepare_parser.add_argument(
        "--project-version",
        default=DEFAULT_PROJECT_VERSION,
        help="Project version written into the run manifest.",
    )
    prepare_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date string written into the run manifest.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark-run",
        help="Run SchNetPack baseline training for a selected dataset source.",
    )
    benchmark_parser.add_argument(
        "--benchmark-name",
        default="default_benchmark",
        help="Benchmark run label.",
    )
    benchmark_parser.add_argument(
        "--model",
        default="SchNet",
        choices=("SchNet", "PaiNN"),
        help="Baseline model name.",
    )
    benchmark_parser.add_argument(
        "--split-name",
        default="default",
        help="Split label recorded in result metadata.",
    )
    benchmark_parser.add_argument(
        "--source-type",
        required=True,
        choices=("ase_db", "md17", "md22"),
        help="Dataset source type.",
    )
    benchmark_parser.add_argument(
        "--datapath",
        default=None,
        help="Path to ASE DB (required for ase_db, optional cache path for md17/md22).",
    )
    benchmark_parser.add_argument(
        "--molecule",
        default=None,
        help="Molecule name for md17/md22 source.",
    )
    benchmark_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training datamodule.",
    )
    benchmark_parser.add_argument(
        "--num-train",
        type=float,
        default=0.8,
        help="Train split size (ratio or absolute count accepted by SchNetPack).",
    )
    benchmark_parser.add_argument(
        "--num-val",
        type=float,
        default=0.1,
        help="Validation split size (ratio or absolute count accepted by SchNetPack).",
    )
    benchmark_parser.add_argument(
        "--num-test",
        type=float,
        default=None,
        help="Optional test split size.",
    )
    benchmark_parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    benchmark_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    benchmark_parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Neighbor list cutoff radius.",
    )
    benchmark_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loader worker count.",
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source and configuration without launching fit loops.",
    )
    benchmark_parser.add_argument(
        "--allow-energy-only",
        action="store_true",
        help="Allow energy-only training when forces labels are unavailable.",
    )
    benchmark_parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory used for benchmark outputs.",
    )
    benchmark_parser.add_argument(
        "--project-version",
        default=DEFAULT_PROJECT_VERSION,
        help="Project version written into run manifest.",
    )
    benchmark_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date string written into run manifest.",
    )

    profile_parser = subparsers.add_parser(
        "dataset-profile-run",
        help="Run dataset quality profile calculation for selected metric subsets.",
    )
    profile_parser.add_argument(
        "--source-type",
        required=True,
        choices=("ase_db", "md17", "md22"),
        help="Dataset source type.",
    )
    profile_parser.add_argument(
        "--datapath",
        default=None,
        help="Path to source DB (required for ase_db; optional cache path for md17/md22).",
    )
    profile_parser.add_argument(
        "--molecule",
        default=None,
        help="Molecule name for md17/md22 source.",
    )
    profile_parser.add_argument(
        "--metrics",
        action="append",
        default=None,
        help="Metric id to run; repeatable, and comma-separated values are also accepted.",
    )
    profile_parser.add_argument(
        "--preset",
        default="core",
        choices=("core", "full"),
        help="Metric preset used when --metrics is not provided.",
    )
    profile_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for source loading.",
    )
    profile_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source and selected metrics without running calculations.",
    )
    profile_parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory used for profile outputs.",
    )
    profile_parser.add_argument(
        "--project-version",
        default=DEFAULT_PROJECT_VERSION,
        help="Project version written into run manifest.",
    )
    profile_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date string written into run manifest.",
    )
    profile_parser.add_argument(
        "--export-references",
        action="store_true",
        help="Export selected reference and prototype structures as individual XYZ files.",
    )
    profile_parser.add_argument(
        "--pool-min-size",
        type=int,
        default=None,
        help="Override the minimum absolute size for an element-composition pool to be considered large (default: 15).",
    )
    profile_parser.add_argument(
        "--subpool-min-size",
        type=int,
        default=None,
        help="Override the minimum absolute size for a structure sub-pool to be retained (default: 10).",
    )

    # Acquisition mode subcommand
    acquisition_parser = subparsers.add_parser(
        "acquisition-mode",
        help="Evaluate candidate structures for dataset supplementation (active learning).",
    )
    acquisition_parser.add_argument(
        "--original-dataset",
        required=True,
        help="Path to the original ASE database (.db file).",
    )
    acquisition_parser.add_argument(
        "--candidates",
        required=True,
        help="Path to POSCAR file or directory containing candidate structures.",
    )
    acquisition_parser.add_argument(
        "--vasp-config",
        default=None,
        help="Optional path to VASP INCAR file for trajectory diversity estimation.",
    )
    acquisition_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to include in report. Defaults to 10.",
    )
    acquisition_parser.add_argument(
        "--element-check",
        choices=["strict", "overlap", "none"],
        default="strict",
        help=(
            "Element compatibility mode applied before scoring candidates. "
            "'strict' (default): candidate elements must be a subset of dataset elements. "
            "'overlap': at least one shared element required. "
            "'none': no check, all candidates are scored (original behaviour)."
        ),
    )
    acquisition_parser.add_argument(
        "--max-distance-sample-ratio",
        type=float,
        default=0.1,
        help=(
            "Sampling ratio for max distance estimation (0.0-1.0). "
            "Lower values improve initialization speed with minimal impact on results. "
            "Use 1.0 for exact computation (slower but slightly more accurate). "
            "Default: 0.1 (10%% sampling, ~100x speedup for large datasets)."
        ),
    )
    acquisition_parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory used for acquisition mode outputs.",
    )
    acquisition_parser.add_argument(
        "--project-version",
        default=DEFAULT_PROJECT_VERSION,
        help="Project version written into run manifest.",
    )
    acquisition_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date string written into run manifest.",
    )

    # Sampling mode subcommand
    sampling_parser = subparsers.add_parser(
        "sampling-mode",
        help="Generate candidate structures for dataset supplementation.",
    )
    sampling_parser.add_argument(
        "--system-type",
        required=True,
        choices=["bulk", "surface", "cluster"],
        help="Type of system to generate.",
    )
    sampling_parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Target number of structures to generate.",
    )
    sampling_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated structures.",
    )
    sampling_parser.add_argument(
        "--composition",
        help="Chemical formula (e.g., 'Hf4O8') for bulk/cluster.",
    )
    sampling_parser.add_argument(
        "--seed",
        help="Path to seed POSCAR for surface generation.",
    )
    sampling_parser.add_argument(
        "--miller-max",
        type=int,
        default=2,
        help="Maximum Miller index for surfaces (default: 2).",
    )
    sampling_parser.add_argument(
        "--layers",
        help="Comma-separated layer counts for surfaces (default: 4,6,8).",
    )
    sampling_parser.add_argument(
        "--energy-filter-percent",
        type=float,
        default=50.0,
        help="Keep bottom X%% by energy after relaxation (default: 50).",
    )
    sampling_parser.add_argument(
        "--lj-fmax",
        type=float,
        default=0.5,
        help="LJ relaxation force convergence (default: 0.5 eV/Å).",
    )
    sampling_parser.add_argument(
        "--lj-max-steps",
        type=int,
        default=100,
        help="Max LJ relaxation steps (default: 100).",
    )
    sampling_parser.add_argument(
        "--min-distance",
        type=float,
        default=0.8,
        help="Minimum allowed interatomic distance (default: 0.8 Å).",
    )
    sampling_parser.add_argument(
        "--config",
        help="Path to YAML configuration file.",
    )
    sampling_parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    sampling_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output.",
    )
    sampling_parser.add_argument(
        "--project-version",
        default=DEFAULT_PROJECT_VERSION,
        help="Project version written into run manifest.",
    )
    sampling_parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date string written into run manifest.",
    )

    return parser


def _load_candidates(path: Path) -> list[StructureRecord]:
    """Load candidate structures from a file or directory using ASE.

    Supports common structure formats (.xyz, .poscar, .vasp, .cif, etc.).
    Falls back to POSCAR* globs for backward compatibility.
    """
    from ase.io import read

    files: list[Path] = []
    if path.is_dir():
        for ext in (".xyz", ".poscar", ".vasp", ".cif", ".extxyz"):
            files.extend(path.glob(f"*{ext}"))
        files.extend(path.glob("POSCAR*"))
        files = sorted(set(files))
    else:
        files = [path]

    records: list[StructureRecord] = []
    for sample_id, file_path in enumerate(files):
        try:
            atoms = read(str(file_path))
            record = ase_to_record(atoms, sample_id=sample_id)
            record.metadata["poscar_path"] = str(file_path)
            record.metadata["poscar_filename"] = file_path.name
            records.append(record)
        except Exception:
            continue
    return records


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-docs":
        return generate_docs_main(
            project_version=args.project_version,
            today=args.date,
        )

    if args.command == "list-tasks":
        for definition in list_task_families():
            subtasks = ", ".join(definition.subtasks)
            print(f"{definition.family}: {subtasks}")
        return 0

    if args.command == "prepare-run":
        task_definition = get_task_definition(args.family)
        task_name = build_task_name(args.family, args.subtask)
        output_dir = prepare_output_directory(
            output_root=args.output_root,
            family=args.family,
            subtask=args.subtask,
        )
        exporter = PresentationExporter()
        manifest = RunManifest(
            run_id=output_dir.name,
            task_name=task_name,
            project_version=args.project_version,
            generated_at=args.date,
            command="prepare-run",
            inputs={
                "family": args.family,
                "subtask": args.subtask or "",
            },
            outputs={
                "run_directory": str(output_dir),
            },
            metadata={
                "family_description": task_definition.description,
                "subtask_count": len(task_definition.subtasks),
            },
        )
        manifest_path = Path(output_dir) / "run_manifest.json"
        exporter.export_manifest(str(manifest_path), manifest)
        print(output_dir)
        return 0

    if args.command == "benchmark-run":
        output_dir = prepare_output_directory(
            output_root=args.output_root,
            family="benchmark",
            subtask="run",
        )
        source = DatasetSourceSpec(
            source_type=args.source_type,
            datapath=args.datapath,
            molecule=args.molecule,
        )
        task = TrainingTaskSpec(
            model_name=args.model,
            split_name=args.split_name,
            output_dir=str(output_dir),
            source=source,
            batch_size=args.batch_size,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            cutoff=args.cutoff,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            require_forces=not args.allow_energy_only,
            allow_energy_only=args.allow_energy_only,
        )
        result = BenchmarkRunner().run(
            benchmark_name=args.benchmark_name,
            dataset=DatasetBundle(dataset_name=args.benchmark_name, records=()),
            profile=DatasetProfile(
                dataset_name=args.benchmark_name,
                metric_results={},
                summary={"record_count": 0, "metric_count": 0},
            ),
            tasks=(task,),
        )
        training = result.training_results[0]
        manifest = RunManifest(
            run_id=output_dir.name,
            task_name="benchmark.run",
            project_version=args.project_version,
            generated_at=args.date,
            command="benchmark-run",
            inputs={
                "source_type": args.source_type,
                "datapath": args.datapath or "",
                "molecule": args.molecule or "",
                "model_name": args.model,
                "allow_energy_only": str(args.allow_energy_only),
                "dry_run": str(args.dry_run),
            },
            outputs={
                "run_directory": str(output_dir),
                "training_manifest": training.artifact_paths.get("training_manifest", ""),
                "metrics_file": training.artifact_paths.get("metrics_file", ""),
                "energy_error_file": training.artifact_paths.get("energy_error_file", ""),
                "force_error_file": training.artifact_paths.get("force_error_file", ""),
            },
            metadata={
                "head_mode": str(training.metadata.get("head_mode", "")),
                "require_forces": not args.allow_energy_only,
                "source_type": args.source_type,
            },
        )
        PresentationExporter().export_manifest(str(Path(output_dir) / "run_manifest.json"), manifest)
        print(output_dir)
        return 0

    if args.command == "dataset-profile-run":
        selected_metric_ids = _resolve_selected_metrics(args.metrics, args.preset)
        registry = build_default_registry(selected_metric_ids=selected_metric_ids)
        output_dir = prepare_output_directory(
            output_root=args.output_root,
            family="dataset_profile",
            subtask="build",
        )
        profile_path = Path(output_dir) / "profile.json"
        summary_path = Path(output_dir) / "summary.md"
        summary_json_path = Path(output_dir) / "summary.json"
        metrics_dir = Path(output_dir) / "metrics"
        figures_dir = Path(output_dir) / "figures"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        start = perf_counter()

        if args.dry_run:
            if args.source_type == "ase_db" and not args.datapath:
                raise ValueError("ase_db source requires --datapath.")
            if args.source_type in {"md17", "md22"} and not args.molecule:
                raise ValueError(f"{args.source_type} source requires --molecule.")
            payload = {
                "mode": "dry-run",
                "source_type": args.source_type,
                "datapath": args.datapath or "",
                "molecule": args.molecule or "",
                "selected_metrics": list(selected_metric_ids),
                "metric_count": len(selected_metric_ids),
            }
            DatasetWriter.write_json(profile_path, payload)
            elapsed = perf_counter() - start
            manifest = RunManifest(
                run_id=Path(output_dir).name,
                task_name="dataset_profile.build",
                project_version=args.project_version,
                generated_at=args.date,
                command="dataset-profile-run",
                inputs={
                    "source_type": args.source_type,
                    "datapath": args.datapath or "",
                    "molecule": args.molecule or "",
                    "preset": args.preset,
                    "selected_metrics": ",".join(selected_metric_ids),
                    "dry_run": "True",
                },
                outputs={
                    "run_directory": str(output_dir),
                    "profile_file": str(profile_path),
                    "summary_file": str(summary_path),
                    "summary_json_file": str(summary_json_path),
                    "metrics_directory": str(metrics_dir),
                    "figures_directory": str(figures_dir),
                },
                metadata={
                    "selected_metric_count": len(selected_metric_ids),
                    "success_metric_count": 0,
                    "skipped_metric_count": 0,
                    "failed_metric_count": 0,
                    "execution_seconds": round(elapsed, 6),
                },
            )
            PresentationExporter().export_manifest(str(Path(output_dir) / "run_manifest.json"), manifest)
            print(output_dir)
            return 0

        dataset = _build_profile_from_source(
            source_type=args.source_type,
            datapath=args.datapath,
            molecule=args.molecule,
            limit=args.limit,
        )
        references_dir: Path | None = None
        if args.export_references:
            references_dir = Path(output_dir) / "references"
        pool_builder_kwargs: dict[str, int] = {}
        if args.pool_min_size is not None:
            pool_builder_kwargs["min_abs_pool"] = args.pool_min_size
        if args.subpool_min_size is not None:
            pool_builder_kwargs["min_abs_subpool"] = args.subpool_min_size
        pool_builder = PoolBuilder(**pool_builder_kwargs) if pool_builder_kwargs else PoolBuilder()
        profile = DatasetProfiler(
            registry=registry,
            export_references_dir=references_dir,
            pool_builder=pool_builder,
        ).profile(dataset)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metric_results = profile.metric_results
        for metric_id, metric_result in metric_results.items():
            metric_path = metrics_dir / f"{metric_id.replace('.', '_')}.json"
            if isinstance(metric_result, MetricResult):
                DatasetWriter.write_json(metric_path, _serialize_metric_result(metric_result))
            else:
                DatasetWriter.write_json(metric_path, {"metric_id": metric_id, "payload": str(metric_result)})

        profile_payload = {
            "dataset_name": profile.dataset_name,
            "summary": profile.summary,
            "provenance": profile.provenance,
            "metric_ids": list(metric_results),
            "applicability_checks": json.loads(
                profile.provenance.get("applicability_checks_json", "{}")
            ),
            "aggregates": {
                metric_id: (
                    metric_result.aggregate.values if isinstance(metric_result, MetricResult) else {}
                )
                for metric_id, metric_result in metric_results.items()
            },
            "sample_level_fields": {
                metric_id: [
                    {
                        "sample_id": sample_record.sample_id,
                        "values": sample_record.values,
                    }
                    for sample_record in metric_result.sample_records[:10]
                ]
                for metric_id, metric_result in metric_results.items()
                if isinstance(metric_result, MetricResult)
            },
        }
        DatasetWriter.write_json(profile_path, profile_payload)

        # Write pool manifest if available
        pool_manifest_json = profile.provenance.get("pool_manifest_json", "")
        pool_manifest_path = Path(output_dir) / "pool_manifest.json"
        if pool_manifest_json:
            try:
                pool_manifest_data = json.loads(pool_manifest_json)
                DatasetWriter.write_json(pool_manifest_path, pool_manifest_data)
            except json.JSONDecodeError:
                pool_manifest_path = Path(output_dir) / "pool_manifest.json"
                DatasetWriter.write_json(pool_manifest_path, {"error": "Invalid pool manifest JSON"})
        exporter = PresentationExporter()
        exporter.export_profile_summary(summary_path, profile, dataset.dataset_name)
        exporter.export_profile_summary_json(summary_json_path, profile)
        generated_figures = exporter.export_profile_figures(figures_dir, profile)

        skipped_metrics = _split_csv_metrics(profile.provenance.get("skipped_metrics", ""))
        failed_metrics = _split_csv_metrics(profile.provenance.get("failed_metrics", ""))
        elapsed = perf_counter() - start
        manifest = RunManifest(
            run_id=Path(output_dir).name,
            task_name="dataset_profile.build",
            project_version=args.project_version,
            generated_at=args.date,
            command="dataset-profile-run",
            inputs={
                "source_type": args.source_type,
                "datapath": args.datapath or "",
                "molecule": args.molecule or "",
                "preset": args.preset,
                "selected_metrics": ",".join(selected_metric_ids),
                "dry_run": "False",
                "export_references": str(args.export_references),
            },
            outputs={
                "run_directory": str(output_dir),
                "profile_file": str(profile_path),
                "summary_file": str(summary_path),
                "summary_json_file": str(summary_json_path),
                "metrics_directory": str(metrics_dir),
                "figures_directory": str(figures_dir),
                "references_directory": str(references_dir) if references_dir is not None else "",
                "pool_manifest_file": str(pool_manifest_path) if pool_manifest_json else "",
            },
            metadata={
                "selected_metric_count": len(selected_metric_ids),
                "success_metric_count": len(metric_results),
                "skipped_metric_count": len(skipped_metrics),
                "failed_metric_count": len(failed_metrics),
                "figure_count": len(generated_figures),
                "execution_seconds": round(elapsed, 6),
            },
        )
        exporter.export_manifest(str(Path(output_dir) / "run_manifest.json"), manifest)
        print(output_dir)
        return 0

    if args.command == "sampling-mode":
        return sampling_mode_command(
            [
                "--system-type",
                args.system_type,
                "--n-samples",
                str(args.n_samples),
                "--output-dir",
                args.output_dir,
                "--miller-max",
                str(args.miller_max),
                "--energy-filter-percent",
                str(args.energy_filter_percent),
                "--lj-fmax",
                str(args.lj_fmax),
                "--lj-max-steps",
                str(args.lj_max_steps),
                "--min-distance",
                str(args.min_distance),
            ]
            + (["--composition", args.composition] if args.composition else [])
            + (["--seed", args.seed] if args.seed else [])
            + (["--layers", args.layers] if args.layers else [])
            + (["--config", args.config] if args.config else [])
            + (["--random-seed", str(args.random_seed)] if args.random_seed else [])
            + (["--verbose"] if args.verbose else [])
        )

    if args.command == "acquisition-mode":
        output_dir = prepare_output_directory(
            output_root=args.output_root,
            family="acquisition",
            subtask="evaluate",
        )
        report_path = Path(output_dir) / "acquisition_report.json"
        summary_path = Path(output_dir) / "acquisition_summary.md"
        start = perf_counter()

        # Load original dataset
        original_dataset = AseDbDatasetReader(args.original_dataset).read()

        # Load candidates (generic ASE format support)
        candidates_path = Path(args.candidates)
        candidates = _load_candidates(candidates_path)

        if not candidates:
            raise ValueError(f"No valid structure files found at: {args.candidates}")

        # Evaluate acquisition value
        evaluator = AcquisitionEvaluator(
            original_dataset,
            element_check=args.element_check,
            max_distance_sample_ratio=args.max_distance_sample_ratio,
        )
        report = evaluator.evaluate(
            candidates=candidates,
            vasp_config_path=args.vasp_config,
            top_k=args.top_k,
        )

        # Export report
        report.export(report_path)
        from .acquisition.report import ReportExporter
        ReportExporter.export_markdown(report, summary_path)

        # Print summary to stdout
        report.print_summary()

        elapsed = perf_counter() - start
        manifest = RunManifest(
            run_id=output_dir.name,
            task_name="acquisition.evaluate",
            project_version=args.project_version,
            generated_at=args.date,
            command="acquisition-mode",
            inputs={
                "original_dataset": args.original_dataset,
                "candidates": args.candidates,
                "vasp_config": args.vasp_config or "",
                "top_k": str(args.top_k),
                "element_check": args.element_check,
                "max_distance_sample_ratio": str(args.max_distance_sample_ratio),
            },
            outputs={
                "run_directory": str(output_dir),
                "report_json": str(report_path),
                "report_markdown": str(summary_path),
            },
            metadata={
                "original_sample_count": report.original_sample_count,
                "candidate_count": report.candidate_count,
                "incompatible_count": len(report.incompatible_candidates),
                "top_k": len(report.rankings),
                "execution_seconds": round(elapsed, 6),
            },
        )
        PresentationExporter().export_manifest(str(Path(output_dir) / "run_manifest.json"), manifest)
        print(output_dir)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
