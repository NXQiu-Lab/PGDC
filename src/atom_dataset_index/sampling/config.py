"""Sampling configuration for structure generation.

module_id: sampling
module_version: 0.9.1
project_version: 1.1.0
last_updated: 2026-04-12
generation_mode: manual
source_files: [config.py]
upstream_modules: []
downstream_modules: [generator, relaxer]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SystemType = Literal["bulk", "surface", "cluster"]


@dataclass
class SamplingConfig:
    """Configuration for structure sampling.

    Supports three system types:
    - bulk: Periodic crystals using PyXtal
    - surface: Surfaces/slabs using ASE
    - cluster: Nanoclusters using PyXtal 0D mode

    Example:
        >>> # Bulk sampling
        >>> config = SamplingConfig(
        ...     system_type="bulk",
        ...     composition={"Hf": 4, "O": 8},
        ...     n_samples=500,
        ...     energy_filter_percent=50,
        ... )
        >>>
        >>> # Surface sampling
        >>> config = SamplingConfig(
        ...     system_type="surface",
        ...     seed_path="POSCAR_bulk",
        ...     n_samples=200,
        ...     miller_max=2,
        ...     layers=[4, 6],
        ... )
    """

    # Common parameters
    system_type: SystemType = "bulk"
    n_samples: int = 100
    energy_filter_percent: float = 50.0
    min_distance: float = 0.8  # Å, minimum allowed interatomic distance
    random_seed: int | None = None

    # Bulk/cluster parameters
    composition: dict[str, int] | None = None
    spacegroup_range: tuple[int, int] = (2, 230)
    lattice_factor_range: tuple[float, float] = (0.8, 1.5)

    # Surface parameters
    seed_path: str | None = None
    miller_max: int = 2
    layers: list[int] | None = None
    vacuum: float = 15.0  # Å
    add_perturbation: bool = True
    perturbation_std: float = 0.05  # Å

    # LJ relaxation parameters
    lj_fmax: float = 0.5  # eV/Å, force convergence criterion
    lj_max_steps: int = 100
    lj_epsilon: float = 0.1  # eV, LJ well depth
    lj_rc: float = 10.0  # Å, LJ cutoff

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if not 0 < self.energy_filter_percent <= 100:
            raise ValueError("energy_filter_percent must be in (0, 100]")
        if self.min_distance <= 0:
            raise ValueError("min_distance must be positive")
        if self.spacegroup_range[0] < 1 or self.spacegroup_range[1] > 230:
            raise ValueError("spacegroup_range must be within [1, 230]")

        # Validate system_type specific requirements
        if self.system_type in ("bulk", "cluster"):
            if not self.composition:
                raise ValueError(
                    f"composition is required for system_type={self.system_type}"
                )
        elif self.system_type == "surface":
            if not self.seed_path:
                raise ValueError("seed_path is required for system_type=surface")


@dataclass
class SamplingReport:
    """Report generated after sampling run.

    Attributes:
        system_type: Type of system sampled
        composition: Chemical composition (for bulk/cluster)
        n_requested: Number of samples requested
        n_generated: Number of structures successfully generated
        n_relaxed: Number of structures that completed LJ relaxation
        n_final: Number of structures after energy filtering
        energy_filter_percent: Percentage of structures kept
        energy_range: Min and max energy after relaxation
        generation_time_seconds: Total time for generation
        structures: List of generated structures with metadata
    """

    system_type: SystemType
    composition: dict[str, int] | None
    n_requested: int
    n_generated: int
    n_relaxed: int
    n_final: int
    energy_filter_percent: float
    energy_range: dict[str, float]
    generation_time_seconds: float
    structures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "system_type": self.system_type,
            "composition": self.composition,
            "n_requested": self.n_requested,
            "n_generated": self.n_generated,
            "n_relaxed": self.n_relaxed,
            "n_final": self.n_final,
            "energy_filter_percent": self.energy_filter_percent,
            "energy_range": self.energy_range,
            "generation_time_seconds": self.generation_time_seconds,
            "structures": self.structures,
        }


def config_from_yaml(path: str | Path) -> SamplingConfig:
    """Load sampling configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        SamplingConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle nested config format
    if "sampling" in data:
        data = data["sampling"]

    return SamplingConfig(**data)


def config_to_yaml(config: SamplingConfig, path: str | Path) -> None:
    """Save sampling configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    import yaml

    path = Path(path)
    data = {
        "system_type": config.system_type,
        "n_samples": config.n_samples,
        "energy_filter_percent": config.energy_filter_percent,
        "min_distance": config.min_distance,
        "random_seed": config.random_seed,
        "composition": config.composition,
        "spacegroup_range": list(config.spacegroup_range),
        "lattice_factor_range": list(config.lattice_factor_range),
        "seed_path": config.seed_path,
        "miller_max": config.miller_max,
        "layers": config.layers,
        "vacuum": config.vacuum,
        "add_perturbation": config.add_perturbation,
        "perturbation_std": config.perturbation_std,
        "lj_fmax": config.lj_fmax,
        "lj_max_steps": config.lj_max_steps,
        "lj_epsilon": config.lj_epsilon,
        "lj_rc": config.lj_rc,
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump({"sampling": data}, f, default_flow_style=False, sort_keys=True)
