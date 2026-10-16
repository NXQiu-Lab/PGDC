"""Benchmark orchestration with real SchNetPack training runners."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from atom_dataset_index.domain import BenchmarkResult, DatasetBundle, DatasetProfile, TrainingResult
from atom_dataset_index.io import DatasetWriter


@dataclass(slots=True)
class DatasetSourceSpec:
    """Description of a training data source consumed by benchmark runners."""

    source_type: str
    datapath: str | None = None
    molecule: str | None = None
    extra_args: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_type.strip():
            raise ValueError("source_type must not be blank.")


@dataclass(slots=True)
class TrainingTaskSpec:
    """Training task specification for baseline runners."""

    model_name: str
    split_name: str
    output_dir: str
    source: DatasetSourceSpec | None = None
    batch_size: int = 32
    num_train: float | int = 0.8
    num_val: float | int = 0.1
    num_test: float | int | None = None
    max_epochs: int = 1
    learning_rate: float = 1e-4
    cutoff: float = 5.0
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 7
    dry_run: bool = False
    require_forces: bool = True
    allow_energy_only: bool = False
    device: str = "cpu"
    precision: str = "32"
    extra_args: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank.")
        if not self.split_name.strip():
            raise ValueError("split_name must not be blank.")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be blank.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than zero.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be greater than zero.")
        if self.allow_energy_only:
            self.require_forces = False


@dataclass(slots=True)
class LoadedDatasetSpec:
    """Loaded data module and resolved property keys for a training run."""

    datamodule: Any
    energy_key: str
    force_key: str | None
    available_properties: tuple[str, ...]
    source_type: str
    source_name: str
    head_mode: str = "energy+forces"


SourceBuilder = Callable[[DatasetSourceSpec, TrainingTaskSpec], LoadedDatasetSpec]
_SOURCE_REGISTRY: dict[str, SourceBuilder] = {}


def register_source(source_type: str, builder: SourceBuilder) -> None:
    """Register a dataset source builder for benchmark workflows."""

    normalized = source_type.strip()
    if not normalized:
        raise ValueError("source_type must not be blank.")
    _SOURCE_REGISTRY[normalized] = builder


def list_registered_sources() -> tuple[str, ...]:
    """List all registered data source names."""

    _ensure_default_sources_registered()
    return tuple(sorted(_SOURCE_REGISTRY))


def _import_schnetpack_data_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import schnetpack as spk
        import schnetpack.transform as trn
        from schnetpack.data.datamodule import AtomsDataModule
        from schnetpack.datasets import MD17, MD22
    except ImportError as error:
        raise RuntimeError(
            "SchnetPack dataset stack is unavailable. Please install schnetpack and dependencies."
        ) from error
    return spk, trn, AtomsDataModule, MD17, MD22


def _import_schnetpack_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import pytorch_lightning as pl
        import schnetpack as spk
        import schnetpack.transform as trn
        import torch
    except ImportError as error:
        raise RuntimeError(
            "SchnetPack training stack is unavailable. Please install pytorch_lightning and torch."
        ) from error
    return spk, trn, pl, torch


def _resolve_head_mode(task: TrainingTaskSpec, force_key: str | None) -> tuple[str, str | None]:
    if force_key is not None:
        return "energy+forces", force_key
    if task.require_forces and not task.allow_energy_only:
        raise ValueError(
            "forces labels are required by default. Use allow_energy_only=True to explicitly downgrade."
        )
    return "energy-only", None


def _build_transforms(trn: Any, energy_key: str, cutoff: float) -> list[Any]:
    return [
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets(energy_key, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32(),
    ]


def _collect_available_properties(datamodule: Any) -> tuple[str, ...]:
    sample = datamodule.dataset[0]
    return tuple(str(name) for name in sample.keys())


def _build_datamodule(
    atoms_data_module_cls: Any,
    datapath: str,
    task: TrainingTaskSpec,
    transforms: list[Any],
) -> Any:
    kwargs: dict[str, Any] = {
        "datapath": datapath,
        "batch_size": task.batch_size,
        "num_train": task.num_train,
        "num_val": task.num_val,
        "transforms": transforms,
        "num_workers": task.num_workers,
        "pin_memory": task.pin_memory,
    }
    if task.num_test is not None:
        kwargs["num_test"] = task.num_test
    return atoms_data_module_cls(**kwargs)


def _setup_datamodule(datamodule: Any, source_name: str) -> None:
    try:
        datamodule.prepare_data()
        datamodule.setup()
    except Exception as error:  # noqa: BLE001 - upstream libraries raise heterogeneous exceptions
        raise RuntimeError(f"Failed to initialize SchNetPack datamodule for source: {source_name}") from error


def _convert_ase_db_to_schnetpack(
    source_path: str,
    target_path: str,
    energy_key: str,
    force_key: str,
) -> str:
    from ase.db import connect
    import numpy as np

    spk, _, _, _, _ = _import_schnetpack_data_stack()
    atoms_buffer: list[Any] = []
    properties_buffer: list[dict[str, Any]] = []
    with_forces = False
    with connect(source_path) as db:
        for row in db.select():
            atoms = row.toatoms()
            data_payload = dict(getattr(row, "data", {}) or {})
            energy = row.get(energy_key)
            if energy is None and hasattr(row, energy_key):
                energy = getattr(row, energy_key)
            if energy is None:
                energy = data_payload.get(energy_key)
            if energy is None:
                continue
            energy_array = np.asarray(energy, dtype=np.float32).reshape(-1)
            if energy_array.size == 0:
                continue
            payload: dict[str, Any] = {energy_key: np.asarray(float(energy_array[0]), dtype=np.float32)}
            forces = row.get(force_key)
            if forces is None and hasattr(row, force_key):
                forces = getattr(row, force_key)
            if forces is None:
                forces = data_payload.get(force_key)
            if forces is not None:
                payload[force_key] = np.asarray(forces, dtype=np.float32)
                with_forces = True
            atoms_buffer.append(atoms)
            properties_buffer.append(payload)
    if not atoms_buffer:
        raise ValueError(f"No usable energy labels found in ASE DB: {source_path}")
    property_unit_dict = {energy_key: "eV"}
    if with_forces:
        property_unit_dict[force_key] = "eV/Ang"
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    dataset = spk.data.ASEAtomsData.create(
        datapath=str(target),
        distance_unit="Ang",
        property_unit_dict=property_unit_dict,
    )
    dataset.add_systems(property_list=properties_buffer, atoms_list=atoms_buffer)
    return str(target)


def _build_ase_db_source(spec: DatasetSourceSpec, task: TrainingTaskSpec) -> LoadedDatasetSpec:
    if not spec.datapath:
        raise ValueError("ase_db source requires datapath.")
    energy_key = str(spec.extra_args.get("energy_key", "energy"))
    force_key = str(spec.extra_args.get("force_key", "forces"))
    if task.dry_run:
        from ase.db import connect

        with connect(spec.datapath) as db:
            row = db.get(1)
            data_payload = dict(getattr(row, "data", {}) or {})
            energy = row.get(energy_key)
            if energy is None and hasattr(row, energy_key):
                energy = getattr(row, energy_key)
            if energy is None:
                energy = data_payload.get(energy_key)
            forces = row.get(force_key)
            if forces is None and hasattr(row, force_key):
                forces = getattr(row, force_key)
            if forces is None:
                forces = data_payload.get(force_key)
            has_energy = energy is not None
            has_forces = forces is not None
            has_positions = True
            record_count = int(db.count())
        properties: list[str] = ["atomic_numbers"]
        if has_positions:
            properties.append("positions")
        if has_energy:
            properties.append(energy_key)
        if has_forces:
            properties.append(force_key)
        resolved_force_key = force_key if has_forces else None
        head_mode, resolved_force_key = _resolve_head_mode(task, resolved_force_key)
        return LoadedDatasetSpec(
            datamodule={"record_count": record_count},
            energy_key=energy_key,
            force_key=resolved_force_key,
            available_properties=tuple(properties),
            source_type=spec.source_type,
            source_name=spec.datapath,
            head_mode=head_mode,
        )
    _, trn, AtomsDataModule, _, _ = _import_schnetpack_data_stack()
    transforms = _build_transforms(trn, energy_key, task.cutoff)
    datamodule = _build_datamodule(AtomsDataModule, spec.datapath, task, transforms)
    source_name = spec.datapath
    try:
        _setup_datamodule(datamodule, source_name)
    except RuntimeError:
        converted_path = str(Path(task.output_dir) / "converted_schnetpack_source.db")
        try:
            converted_path = _convert_ase_db_to_schnetpack(
                source_path=spec.datapath,
                target_path=converted_path,
                energy_key=energy_key,
                force_key=force_key,
            )
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "Failed to convert ASE DB into SchNetPack-ready format. "
                "Please check dataset fields or run benchmark in --dry-run mode."
            ) from error
        datamodule = _build_datamodule(AtomsDataModule, converted_path, task, transforms)
        _setup_datamodule(datamodule, converted_path)
        source_name = converted_path
    properties = _collect_available_properties(datamodule)
    if energy_key not in properties:
        raise ValueError(f"Energy property '{energy_key}' not found in source '{source_name}'.")
    resolved_force_key = force_key if force_key in properties else None
    head_mode, resolved_force_key = _resolve_head_mode(task, resolved_force_key)
    return LoadedDatasetSpec(
        datamodule=datamodule,
        energy_key=energy_key,
        force_key=resolved_force_key,
        available_properties=properties,
        source_type=spec.source_type,
        source_name=source_name,
        head_mode=head_mode,
    )


def _build_md17_source(spec: DatasetSourceSpec, task: TrainingTaskSpec) -> LoadedDatasetSpec:
    if not spec.molecule:
        raise ValueError("md17 source requires molecule name.")
    _, trn, _, MD17, _ = _import_schnetpack_data_stack()
    datapath = spec.datapath or str(Path("data") / f"md17_{spec.molecule}.db")
    kwargs: dict[str, Any] = {
        "datapath": datapath,
        "molecule": spec.molecule,
        "batch_size": task.batch_size,
        "num_train": task.num_train,
        "num_val": task.num_val,
        "transforms": _build_transforms(trn, MD17.energy, task.cutoff),
        "num_workers": task.num_workers,
        "pin_memory": task.pin_memory,
    }
    if task.num_test is not None:
        kwargs["num_test"] = task.num_test
    datamodule = MD17(**kwargs)
    _setup_datamodule(datamodule, datapath)
    properties = _collect_available_properties(datamodule)
    resolved_force_key = MD17.forces if MD17.forces in properties else None
    head_mode, resolved_force_key = _resolve_head_mode(task, resolved_force_key)
    return LoadedDatasetSpec(
        datamodule=datamodule,
        energy_key=MD17.energy,
        force_key=resolved_force_key,
        available_properties=properties,
        source_type=spec.source_type,
        source_name=spec.molecule,
        head_mode=head_mode,
    )


def _build_md22_source(spec: DatasetSourceSpec, task: TrainingTaskSpec) -> LoadedDatasetSpec:
    if not spec.molecule:
        raise ValueError("md22 source requires molecule name.")
    _, trn, _, _, MD22 = _import_schnetpack_data_stack()
    datapath = spec.datapath or str(Path("data") / f"md22_{spec.molecule}.db")
    kwargs: dict[str, Any] = {
        "datapath": datapath,
        "molecule": spec.molecule,
        "batch_size": task.batch_size,
        "num_train": task.num_train,
        "num_val": task.num_val,
        "transforms": _build_transforms(trn, MD22.energy, task.cutoff),
        "num_workers": task.num_workers,
        "pin_memory": task.pin_memory,
    }
    if task.num_test is not None:
        kwargs["num_test"] = task.num_test
    datamodule = MD22(**kwargs)
    _setup_datamodule(datamodule, datapath)
    properties = _collect_available_properties(datamodule)
    resolved_force_key = MD22.forces if MD22.forces in properties else None
    head_mode, resolved_force_key = _resolve_head_mode(task, resolved_force_key)
    return LoadedDatasetSpec(
        datamodule=datamodule,
        energy_key=MD22.energy,
        force_key=resolved_force_key,
        available_properties=properties,
        source_type=spec.source_type,
        source_name=spec.molecule,
        head_mode=head_mode,
    )


def _ensure_default_sources_registered() -> None:
    if "ase_db" not in _SOURCE_REGISTRY:
        register_source("ase_db", _build_ase_db_source)
    if "md17" not in _SOURCE_REGISTRY:
        register_source("md17", _build_md17_source)
    if "md22" not in _SOURCE_REGISTRY:
        register_source("md22", _build_md22_source)


class SchNetPackDatasetAdapter:
    """Resolve a source spec into a SchNetPack datamodule and key mapping."""

    def __init__(self) -> None:
        _ensure_default_sources_registered()

    def adapt(self, source: DatasetSourceSpec, task: TrainingTaskSpec) -> LoadedDatasetSpec:
        normalized = source.source_type.strip()
        if normalized not in _SOURCE_REGISTRY:
            available = ", ".join(list_registered_sources())
            raise ValueError(f"Unsupported source type: {normalized}. Available: {available}")
        loaded = _SOURCE_REGISTRY[normalized](source, task)
        head_mode, force_key = _resolve_head_mode(task, loaded.force_key)
        loaded.head_mode = head_mode
        loaded.force_key = force_key
        return loaded


class _BaseSchnetpackRunner:
    """Shared training implementation for SchNetPack-based models."""

    model_name = ""

    def _build_atomistic_task(self, task: TrainingTaskSpec, loaded: LoadedDatasetSpec) -> Any:
        spk, trn, _, torch = _import_schnetpack_training_stack()
        n_atom_basis = int(task.extra_args.get("n_atom_basis", 64))
        n_interactions = int(task.extra_args.get("n_interactions", 3))
        n_rbf = int(task.extra_args.get("n_rbf", 20))

        radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=task.cutoff)
        if self.model_name == "SchNet":
            representation = spk.representation.SchNet(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                radial_basis=radial_basis,
                cutoff_fn=spk.nn.CosineCutoff(task.cutoff),
            )
        elif self.model_name == "PaiNN":
            representation = spk.representation.PaiNN(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                radial_basis=radial_basis,
                cutoff_fn=spk.nn.CosineCutoff(task.cutoff),
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        pairwise_distance = spk.atomistic.PairwiseDistances()
        pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=loaded.energy_key)
        output_modules = [pred_energy]
        task_outputs = [
            spk.task.ModelOutput(
                name=loaded.energy_key,
                loss_fn=torch.nn.MSELoss(),
                loss_weight=0.01 if loaded.force_key else 1.0,
                metrics={},
            )
        ]
        if loaded.force_key is not None:
            pred_forces = spk.atomistic.Forces(
                energy_key=loaded.energy_key,
                force_key=loaded.force_key,
            )
            output_modules.append(pred_forces)
            task_outputs.append(
                spk.task.ModelOutput(
                    name=loaded.force_key,
                    loss_fn=torch.nn.MSELoss(),
                    loss_weight=0.99,
                    metrics={},
                )
            )

        nnpot = spk.model.NeuralNetworkPotential(
            representation=representation,
            input_modules=[pairwise_distance],
            output_modules=output_modules,
            postprocessors=[
                trn.CastTo64(),
                trn.AddOffsets(loaded.energy_key, add_mean=True, add_atomrefs=False),
            ],
        )
        return spk.task.AtomisticTask(
            model=nnpot,
            outputs=task_outputs,
            optimizer_cls=torch.optim.AdamW,
            optimizer_args={"lr": task.learning_rate},
        )

    @staticmethod
    def _flatten_metrics(*groups: list[dict[str, Any]]) -> dict[str, float]:
        merged: dict[str, float] = {}
        for group in groups:
            for payload in group:
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        merged[key] = float(value)
        return merged

    @staticmethod
    def _find_metric(metrics: dict[str, float], *parts: str) -> float:
        lowered_parts = tuple(part.lower() for part in parts)
        for key, value in metrics.items():
            normalized = key.lower()
            if all(part in normalized for part in lowered_parts):
                return value
        return 0.0

    @staticmethod
    def _record_count(datamodule: Any) -> int:
        if isinstance(datamodule, dict):
            count = datamodule.get("record_count")
            if isinstance(count, int):
                return count
        dataset = getattr(datamodule, "dataset", None)
        if dataset is not None:
            try:
                return len(dataset)
            except TypeError:
                return 0
        records = getattr(datamodule, "records", None)
        if records is not None:
            return len(records)
        return 0

    def run(self, task: TrainingTaskSpec, loaded: LoadedDatasetSpec) -> TrainingResult:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_tag = self.model_name.lower()
        metrics_path = output_dir / f"{model_tag}_metrics.json"
        training_manifest_path = output_dir / f"{model_tag}_training_manifest.json"
        energy_error_file = output_dir / f"{model_tag}_energy_errors.csv"
        force_error_file = output_dir / f"{model_tag}_force_errors.csv"

        if task.dry_run:
            payload = {
                "mode": "dry-run",
                "model_name": self.model_name,
                "source_type": loaded.source_type,
                "source_name": loaded.source_name,
                "head_mode": loaded.head_mode,
                "available_properties": loaded.available_properties,
                "require_forces": task.require_forces,
                "allow_energy_only": task.allow_energy_only,
            }
            DatasetWriter.write_json(metrics_path, payload)
            self._write_error_csv(energy_error_file, [])
            self._write_error_csv(force_error_file, [])
            DatasetWriter.write_json(
                training_manifest_path,
                {
                    "status": "dry-run",
                    "model_name": self.model_name,
                    "head_mode": loaded.head_mode,
                    "output_dir": str(output_dir),
                },
            )
            return TrainingResult(
                model_name=self.model_name,
                split_name=task.split_name,
                metrics={"energy_mae": 0.0, "force_mae": 0.0},
                artifact_paths={
                    "training_manifest": str(training_manifest_path),
                    "metrics_file": str(metrics_path),
                    "energy_error_file": str(energy_error_file),
                    "force_error_file": str(force_error_file),
                },
                metadata={
                    "source_type": loaded.source_type,
                    "head_mode": loaded.head_mode,
                    "dry_run": True,
                    "record_count": self._record_count(loaded.datamodule),
                },
            )

        spk, _, pl, _ = _import_schnetpack_training_stack()
        atomistic_task = self._build_atomistic_task(task, loaded)
        checkpoint_path = output_dir / f"{model_tag}_best_inference_model"
        callbacks = [
            spk.train.ModelCheckpoint(
                model_path=str(checkpoint_path),
                save_top_k=1,
                monitor="val_loss",
            )
        ]
        trainer = pl.Trainer(
            default_root_dir=str(output_dir),
            max_epochs=task.max_epochs,
            accelerator="cpu" if task.device == "cpu" else "auto",
            devices=1,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
            precision=task.precision,
        )
        trainer.fit(atomistic_task, datamodule=loaded.datamodule)
        validation_metrics = trainer.validate(atomistic_task, datamodule=loaded.datamodule, verbose=False)
        test_metrics = trainer.test(atomistic_task, datamodule=loaded.datamodule, verbose=False)
        merged_metrics = self._flatten_metrics(validation_metrics, test_metrics)
        energy_mae = self._find_metric(merged_metrics, loaded.energy_key, "mae")
        if energy_mae == 0.0:
            energy_mae = self._find_metric(merged_metrics, "energy", "mae")
        force_mae = self._find_metric(merged_metrics, "forces", "mae")
        if loaded.force_key is not None and force_mae == 0.0:
            force_mae = self._find_metric(merged_metrics, loaded.force_key, "mae")

        DatasetWriter.write_json(
            metrics_path,
            {
                "model_name": self.model_name,
                "split_name": task.split_name,
                "head_mode": loaded.head_mode,
                "metrics": merged_metrics,
            },
        )
        self._write_error_csv(energy_error_file, [("validation_or_test", energy_mae)])
        self._write_error_csv(force_error_file, [("validation_or_test", force_mae)])
        DatasetWriter.write_json(
            training_manifest_path,
            {
                "status": "completed",
                "model_name": self.model_name,
                "source_type": loaded.source_type,
                "source_name": loaded.source_name,
                "head_mode": loaded.head_mode,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        return TrainingResult(
            model_name=self.model_name,
            split_name=task.split_name,
            metrics={"energy_mae": energy_mae, "force_mae": force_mae},
            artifact_paths={
                "training_manifest": str(training_manifest_path),
                "metrics_file": str(metrics_path),
                "energy_error_file": str(energy_error_file),
                "force_error_file": str(force_error_file),
                "checkpoint_path": str(checkpoint_path),
            },
            metadata={
                "source_type": loaded.source_type,
                "head_mode": loaded.head_mode,
                "dry_run": False,
                "record_count": self._record_count(loaded.datamodule),
            },
        )

    @staticmethod
    def _write_error_csv(path: Path, rows: list[tuple[str, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("split", "mae"))
            writer.writerows(rows)


class SchNetRunner(_BaseSchnetpackRunner):
    """Train SchNet baseline tasks via SchNetPack."""

    model_name = "SchNet"


class PaiNNRunner(_BaseSchnetpackRunner):
    """Train PaiNN baseline tasks via SchNetPack."""

    model_name = "PaiNN"


class BenchmarkRunner:
    """Coordinate profile outputs with SchNetPack baseline runners."""

    def __init__(
        self,
        adapter: SchNetPackDatasetAdapter | None = None,
        runners: dict[str, _BaseSchnetpackRunner] | None = None,
    ) -> None:
        self.adapter = adapter or SchNetPackDatasetAdapter()
        self.runners = runners or {"SchNet": SchNetRunner(), "PaiNN": PaiNNRunner()}

    def run(
        self,
        benchmark_name: str,
        dataset: DatasetBundle,
        profile: DatasetProfile,
        tasks: tuple[TrainingTaskSpec, ...],
    ) -> BenchmarkResult:
        if not tasks:
            raise ValueError("At least one training task is required.")
        unknown_models = sorted({task.model_name for task in tasks if task.model_name not in self.runners})
        if unknown_models:
            raise ValueError(f"Unsupported benchmark model(s): {', '.join(unknown_models)}")
        training_results: list[TrainingResult] = []
        for task in tasks:
            if task.source is None:
                raise ValueError("TrainingTaskSpec.source is required for benchmark execution.")
            loaded = self.adapter.adapt(task.source, task)
            result = self.runners[task.model_name].run(task, loaded)
            training_results.append(result)
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            profiles={dataset.dataset_name: profile},
            training_results=tuple(training_results),
            summary={
                "task_count": len(tasks),
                "model_count": len(training_results),
                "record_count": len(dataset.records),
                "profile_metric_count": int(profile.summary.get("metric_count", 0)),
                "source_types": ",".join(sorted({task.source.source_type for task in tasks if task.source})),
            },
        )
