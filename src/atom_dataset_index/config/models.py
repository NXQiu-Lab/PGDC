"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathConfig:
    """Repository path layout used by tasks and output exporters."""

    project_root: Path
    data_dir: Path
    output_dir: Path
    docs_dir: Path
    tasks_dir: Path

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.docs_dir = Path(self.docs_dir)
        self.tasks_dir = Path(self.tasks_dir)
        if not self.project_root.exists():
            raise ValueError(f"project_root does not exist: {self.project_root}")
        if not self.project_root.is_dir():
            raise ValueError(f"project_root must be a directory: {self.project_root}")
        for name, path in (
            ("data_dir", self.data_dir),
            ("output_dir", self.output_dir),
            ("docs_dir", self.docs_dir),
            ("tasks_dir", self.tasks_dir),
        ):
            if not path.is_absolute():
                raise ValueError(f"{name} must be an absolute path: {path}")


@dataclass(slots=True)
class MetricConfig:
    """Parameters that control metric execution and descriptor behavior."""

    descriptor_backend: str = "identity"
    cutoff_radius: float = 6.0
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    high_energy_quantile: float = 0.9
    cluster_count: int = 8
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cutoff_radius <= 0:
            raise ValueError("cutoff_radius must be > 0.")
        if not self.quantiles:
            raise ValueError("quantiles must not be empty.")
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be sorted in ascending order.")
        if len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError("quantiles must not contain duplicate values.")
        for quantile in self.quantiles:
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(f"quantile out of range [0, 1]: {quantile}")
        if not 0.0 <= self.high_energy_quantile <= 1.0:
            raise ValueError("high_energy_quantile must be within [0, 1].")
        if self.cluster_count <= 0:
            raise ValueError("cluster_count must be > 0.")


@dataclass(slots=True)
class BenchmarkConfig:
    """Parameters used for subset construction and training orchestration."""

    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    random_seed: int = 7
    max_samples: int | None = None
    model_names: tuple[str, ...] = ("SchNet", "PaiNN")

    def __post_init__(self) -> None:
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be within (0, 1).")
        if not 0.0 <= self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be within [0, 1).")
        if self.train_ratio + self.validation_ratio >= 1.0:
            raise ValueError("train_ratio + validation_ratio must be < 1.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided.")
        if not self.model_names:
            raise ValueError("model_names must not be empty.")
        if any(not name.strip() for name in self.model_names):
            raise ValueError("model_names must not include blank names.")


@dataclass(slots=True)
class PlotConfig:
    """Rendering defaults for presentation outputs."""

    figure_format: str = "png"
    width: float = 8.0
    height: float = 5.0
    dpi: int = 200

    def __post_init__(self) -> None:
        if not self.figure_format.strip():
            raise ValueError("figure_format must not be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0.")
        if self.dpi <= 0:
            raise ValueError("dpi must be > 0.")
