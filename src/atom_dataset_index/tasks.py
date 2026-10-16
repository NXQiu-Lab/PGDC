"""Runtime task registry and output directory preparation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Definition of a supported task family and optional subtask names."""

    family: str
    subtasks: tuple[str, ...]
    description: str


TASK_DEFINITIONS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        family="dataset_profile",
        subtasks=("build", "validate", "export"),
        description="Dataset evaluation and profile generation tasks.",
    ),
    TaskDefinition(
        family="dataset_sampling",
        subtasks=("split", "subset", "inspect"),
        description="Sampling, split, and subset management tasks.",
    ),
    TaskDefinition(
        family="training",
        subtasks=("schnet", "painn", "compare"),
        description="SchNetPack baseline training tasks.",
    ),
    TaskDefinition(
        family="benchmark",
        subtasks=("run", "collect", "summarize"),
        description="Benchmark execution and comparison tasks.",
    ),
    TaskDefinition(
        family="acquisition",
        subtasks=("evaluate",),
        description="Active learning candidate evaluation tasks.",
    ),
    TaskDefinition(
        family="docs",
        subtasks=("generate_leaf", "generate_package", "generate_top"),
        description="Documentation generation tasks.",
    ),
)


def list_task_families() -> tuple[TaskDefinition, ...]:
    """Return all registered task families."""

    return TASK_DEFINITIONS


def get_task_definition(family: str) -> TaskDefinition:
    """Return a task family definition."""

    normalized_family = family.strip()
    if not normalized_family:
        raise KeyError("Task family must not be blank.")
    for definition in TASK_DEFINITIONS:
        if definition.family == normalized_family:
            return definition
    raise KeyError(f"Unknown task family: {family}")


def build_task_name(family: str, subtask: str | None = None) -> str:
    """Build a normalized task name for run manifests."""

    definition = get_task_definition(family)
    normalized_subtask = (subtask or "default").strip()
    if normalized_subtask != "default" and normalized_subtask not in definition.subtasks:
        raise ValueError(f"Unsupported subtask '{subtask}' for family '{family}'.")
    return f"{definition.family}.{normalized_subtask}"


def build_run_id(now: datetime | None = None) -> str:
    """Build a sortable run identifier."""

    current = now or datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def prepare_output_directory(
    output_root: str | Path,
    family: str,
    subtask: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Create a runtime output directory for a task family/subtask."""

    definition = get_task_definition(family)
    normalized_subtask = None if subtask is None else subtask.strip()
    if normalized_subtask is not None and normalized_subtask not in definition.subtasks:
        raise ValueError(f"Unsupported subtask '{subtask}' for family '{family}'.")

    resolved_run_id = run_id or build_run_id()
    root = Path(output_root)
    path = root / family
    if normalized_subtask:
        path = path / normalized_subtask
    path = path / resolved_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
