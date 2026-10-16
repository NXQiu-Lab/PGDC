"""Structure sampling module for generating candidate pools.

This module provides functionality for generating candidate structures
for MLIP dataset construction without requiring external MD software.
It supports bulk crystals, surfaces, and clusters using PyXtal and ASE.

module_id: sampling
module_version: 0.9.1
project_version: 1.1.0
last_updated: 2026-04-12
generation_mode: manual
source_files: [__init__.py, config.py, generator.py, relaxer.py, utils.py, cli.py]
upstream_modules: [domain, io]
downstream_modules: [acquisition]

Example:
    >>> from atom_dataset_index.sampling import generate_structures, SamplingConfig
    >>> config = SamplingConfig(
    ...     system_type="bulk",
    ...     composition={"Hf": 4, "O": 8},
    ...     n_samples=100,
    ...     energy_filter_percent=50,
    ... )
    >>> candidates = generate_structures(
    ...     "bulk", 100, config, composition={"Hf": 4, "O": 8}
    ... )
    >>> len(candidates)
    100
"""

from atom_dataset_index.sampling.config import (
    SamplingConfig,
    SamplingReport,
    SystemType,
    config_from_yaml,
    config_to_yaml,
)
from atom_dataset_index.sampling.generator import generate_structures
from atom_dataset_index.sampling.explorers import (
    BoundaryExplorer,
    CandidateGenerator,
    StrainPathGenerator,
)
from atom_dataset_index.sampling.prefilter import (
    DeterministicPreFilter,
    StructureClassResult,
)
from atom_dataset_index.sampling.relaxer import relax_and_filter
from atom_dataset_index.sampling.utils import (
    ase_to_record,
    basic_validation,
    composition_to_string,
    parse_composition,
    pyxtal_to_record,
    record_to_ase,
)

__all__ = [
    # Configuration
    "SamplingConfig",
    "SamplingReport",
    "SystemType",
    "config_from_yaml",
    "config_to_yaml",
    # Generation
    "generate_structures",
    # Dataset-aware explorers
    "BoundaryExplorer",
    "StrainPathGenerator",
    "CandidateGenerator",
    # Pre-filter (replaces LJ relaxer)
    "DeterministicPreFilter",
    "StructureClassResult",
    # Relaxation
    "relax_and_filter",
    # Utilities
    "ase_to_record",
    "basic_validation",
    "composition_to_string",
    "parse_composition",
    "pyxtal_to_record",
    "record_to_ase",
]
