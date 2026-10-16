"""Structure generation for bulk/surface/cluster systems.

module_id: sampling.generator
module_version: 0.9.1
project_version: 1.1.0
last_updated: 2026-04-12
generation_mode: manual
source_files: [generator.py]
upstream_modules: [domain, utils, config]
downstream_modules: [relaxer]
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from atom_dataset_index.domain import StructureRecord
from atom_dataset_index.sampling.config import SamplingConfig, SystemType
from atom_dataset_index.sampling.utils import (
    ase_to_record,
    basic_validation,
    enumerate_miller_indices,
    perturb_atoms,
    pyxtal_to_record,
)

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)


def generate_structures(
    system_type: SystemType,
    n_samples: int,
    config: SamplingConfig,
    composition: dict[str, int] | None = None,
    seed_path: str | None = None,
    verbose: bool = False,
) -> list[StructureRecord]:
    """Generate candidate structures based on system type.

    This is the unified entry point for structure generation.
    Routes to the appropriate generator based on system_type.

    Args:
        system_type: Type of system ("bulk", "surface", "cluster")
        n_samples: Target number of structures to generate
        config: Sampling configuration
        composition: Chemical composition dict (for bulk/cluster)
        seed_path: Path to seed structure (for surface)
        verbose: Whether to log progress

    Returns:
        List of generated StructureRecord candidates

    Raises:
        ValueError: If system_type is unknown or required params missing
        ImportError: If PyXtal is required but not installed

    Example:
        >>> config = SamplingConfig(system_type="bulk")
        >>> candidates = generate_structures(
        ...     "bulk", 100, config, composition={"Hf": 4, "O": 8}
        ... )
    """
    if system_type == "bulk":
        if not composition:
            raise ValueError("composition is required for bulk generation")
        return _generate_bulk(composition, n_samples, config, verbose)

    elif system_type == "surface":
        if not seed_path:
            raise ValueError("seed_path is required for surface generation")
        return _generate_surface(seed_path, n_samples, config, verbose)

    elif system_type == "cluster":
        if not composition:
            raise ValueError("composition is required for cluster generation")
        return _generate_cluster(composition, n_samples, config, verbose)

    else:
        raise ValueError(f"Unknown system_type: {system_type}")


def _generate_bulk(
    composition: dict[str, int],
    n_samples: int,
    config: SamplingConfig,
    verbose: bool = False,
) -> list[StructureRecord]:
    """Generate bulk crystal structures using PyXtal.

    Uses PyXtal's random crystal generation with space group constraints.
    Structures are generated in random space groups within the specified
    range and filtered for validity.

    Args:
        composition: Dictionary of {element: count}
        n_samples: Target number of structures
        config: Sampling configuration
        verbose: Whether to log progress

    Returns:
        List of valid StructureRecord candidates
    """
    try:
        from pyxtal import pyxtal
    except ImportError as e:
        raise ImportError(
            "PyXtal is required for bulk sampling. "
            "Install with: pip install pyxtal"
        ) from e

    candidates = []
    max_attempts = n_samples * 5  # Allow for failures
    sg_min, sg_max = config.spacegroup_range

    if verbose:
        logger.info(
            f"Generating {n_samples} bulk structures "
            f"(space groups {sg_min}-{sg_max})"
        )

    for attempt in range(max_attempts):
        if len(candidates) >= n_samples:
            break

        # Random space group
        sg = random.randint(sg_min, sg_max)

        # Random lattice factor
        factor = random.uniform(*config.lattice_factor_range)

        try:
            crystal = pyxtal()
            crystal.from_random(
                dim=3,
                group=sg,
                species=list(composition.keys()),
                numIons=list(composition.values()),
                factor=factor,
            )

            if crystal.valid:
                record = pyxtal_to_record(crystal, sample_id=len(candidates))
                record.metadata["space_group"] = sg
                record.metadata["lattice_factor"] = factor

                if basic_validation(record, config.min_distance):
                    candidates.append(record)
                    if verbose and len(candidates) % 10 == 0:
                        logger.info(f"Generated {len(candidates)}/{n_samples}")

        except Exception as e:
            if verbose:
                logger.debug(f"Generation failed for SG {sg}: {e}")
            continue

    if verbose:
        logger.info(f"Generated {len(candidates)} bulk structures")

    return candidates


def _generate_surface(
    seed_path: str,
    n_samples: int,
    config: SamplingConfig,
    verbose: bool = False,
) -> list[StructureRecord]:
    """Generate surface/slab structures using ASE.

    Builds surfaces from a bulk seed structure using ASE's surface
    builder. Enumerates Miller indices and layer counts.

    Args:
        seed_path: Path to seed bulk structure (POSCAR format)
        n_samples: Target number of structures
        config: Sampling configuration
        verbose: Whether to log progress

    Returns:
        List of StructureRecord surface structures
    """
    from ase.io import read

    try:
        bulk = read(seed_path)
    except Exception as e:
        raise ValueError(f"Failed to read seed structure from {seed_path}: {e}")

    candidates = []
    layer_range = config.layers or [4, 6, 8]

    if verbose:
        logger.info(
            f"Generating surface structures from {seed_path}, "
            f"Miller max={config.miller_max}, layers={layer_range}"
        )

    # Enumerate Miller indices
    miller_list = enumerate_miller_indices(config.miller_max)

    for miller in miller_list:
        for n_layers in layer_range:
            try:
                # Use ASE's surface function
                from ase.build import surface

                slab = surface(
                    bulk,
                    miller,
                    n_layers,
                    vacuum=config.vacuum,
                )

                # Optional perturbation
                if config.add_perturbation:
                    slab = perturb_atoms(slab, std=config.perturbation_std)

                record = ase_to_record(slab, sample_id=len(candidates))
                # Convert tuple to list for JSON compatibility
                record.metadata["miller_indices"] = f"({miller[0]},{miller[1]},{miller[2]})"
                record.metadata["n_layers"] = n_layers
                record.metadata["vacuum"] = config.vacuum
                record.metadata["seed"] = seed_path

                candidates.append(record)

                if verbose and len(candidates) % 10 == 0:
                    logger.info(f"Generated {len(candidates)} surfaces")

            except Exception as e:
                if verbose:
                    logger.debug(f"Failed for {miller} with {n_layers} layers: {e}")
                continue

    # If we generated too many, randomly sample
    if len(candidates) > n_samples:
        candidates = random.sample(candidates, n_samples)
        if verbose:
            logger.info(f"Sampled {n_samples} surfaces from {len(candidates)} generated")

    if verbose:
        logger.info(f"Generated {len(candidates)} surface structures")

    return candidates


def _generate_cluster(
    composition: dict[str, int],
    n_samples: int,
    config: SamplingConfig,
    verbose: bool = False,
) -> list[StructureRecord]:
    """Generate cluster/nanoparticle structures using PyXtal 0D mode.

    Uses PyXtal's 0D (cluster) mode with point group symmetry.
    Suitable for metal nanoclusters and quantum dots.

    Args:
        composition: Dictionary of {element: count}
        n_samples: Target number of structures
        config: Sampling configuration
        verbose: Whether to log progress

    Returns:
        List of StructureRecord cluster structures
    """
    try:
        from pyxtal import pyxtal
    except ImportError as e:
        raise ImportError(
            "PyXtal is required for cluster sampling. "
            "Install with: pip install pyxtal"
        ) from e

    candidates = []
    max_attempts = n_samples * 5

    # PyXtal point groups (1-32)
    point_groups = list(range(1, 33))

    if verbose:
        logger.info(f"Generating {n_samples} cluster structures")

    for attempt in range(max_attempts):
        if len(candidates) >= n_samples:
            break

        # Random point group
        pg = random.choice(point_groups)

        try:
            cluster = pyxtal()
            cluster.from_random(
                dim=0,  # 0D = cluster
                group=pg,
                species=list(composition.keys()),
                numIons=list(composition.values()),
            )

            if cluster.valid:
                record = pyxtal_to_record(cluster, sample_id=len(candidates))
                record.metadata["point_group"] = pg

                if basic_validation(record, config.min_distance):
                    candidates.append(record)
                    if verbose and len(candidates) % 10 == 0:
                        logger.info(f"Generated {len(candidates)}/{n_samples}")

        except Exception as e:
            if verbose:
                logger.debug(f"Generation failed for PG {pg}: {e}")
            continue

    if verbose:
        logger.info(f"Generated {len(candidates)} cluster structures")

    return candidates
