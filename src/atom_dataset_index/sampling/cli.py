"""CLI implementation for sampling-mode command.

module_id: sampling.cli
module_version: 0.9.1
project_version: 1.1.0
last_updated: 2026-04-12
generation_mode: manual
source_files: [cli.py]
upstream_modules: [config, generator, relaxer, utils]
downstream_modules: [io]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

from atom_dataset_index.io import DatasetWriter
from atom_dataset_index.sampling.config import (
    SamplingConfig,
    SamplingReport,
    config_from_yaml,
)
from atom_dataset_index.sampling.generator import generate_structures
from atom_dataset_index.sampling.relaxer import relax_and_filter
from atom_dataset_index.sampling.utils import parse_composition

logger = logging.getLogger(__name__)


def build_sampling_parser() -> argparse.ArgumentParser:
    """Build argument parser for sampling-mode command."""
    parser = argparse.ArgumentParser(
        prog="sampling-mode",
        description="Generate candidate structures for dataset supplementation.",
    )

    # Required arguments
    parser.add_argument(
        "--system-type",
        required=True,
        choices=["bulk", "surface", "cluster"],
        help="Type of system to generate (bulk, surface, cluster)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Target number of structures to generate",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated structures",
    )

    # System-type specific arguments
    parser.add_argument(
        "--composition",
        help="Chemical formula (e.g., 'Hf4O8', 'NaCl', 'Au55') for bulk/cluster",
    )
    parser.add_argument(
        "--seed",
        help="Path to seed POSCAR for surface generation",
    )

    # Surface-specific arguments
    parser.add_argument(
        "--miller-max",
        type=int,
        default=2,
        help="Maximum Miller index for surfaces (default: 2)",
    )
    parser.add_argument(
        "--layers",
        help="Comma-separated layer counts for surfaces (default: 4,6,8)",
    )

    # Energy filtering
    parser.add_argument(
        "--energy-filter-percent",
        type=float,
        default=50.0,
        help="Keep bottom X%% by energy after relaxation (default: 50)",
    )

    # LJ relaxation parameters
    parser.add_argument(
        "--lj-fmax",
        type=float,
        default=0.5,
        help="LJ relaxation force convergence (eV/Å, default: 0.5)",
    )
    parser.add_argument(
        "--lj-max-steps",
        type=int,
        default=100,
        help="Max LJ relaxation steps (default: 100)",
    )

    # Validation
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.8,
        help="Minimum allowed interatomic distance (Å, default: 0.8)",
    )

    # General options
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file (overrides other args)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


def sampling_mode_command(argv: list[str] | None = None) -> int:
    """Execute sampling-mode command.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = build_sampling_parser()
    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    start_time = perf_counter()

    # Load or build config
    if args.config:
        try:
            config = config_from_yaml(args.config)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
    else:
        # Parse composition if provided
        composition = None
        if args.composition:
            try:
                composition = parse_composition(args.composition)
            except ValueError as e:
                logger.error(f"Invalid composition: {e}")
                return 1

        # Parse layers if provided
        layers = None
        if args.layers:
            try:
                layers = [int(x.strip()) for x in args.layers.split(",")]
            except ValueError:
                logger.error("Invalid layers format, expected comma-separated integers")
                return 1

        config = SamplingConfig(
            system_type=args.system_type,
            n_samples=args.n_samples,
            energy_filter_percent=args.energy_filter_percent,
            min_distance=args.min_distance,
            random_seed=args.random_seed,
            composition=composition,
            seed_path=args.seed,
            miller_max=args.miller_max,
            layers=layers,
            lj_fmax=args.lj_fmax,
            lj_max_steps=args.lj_max_steps,
        )

    # Set random seed if provided
    if config.random_seed is not None:
        import random

        random.seed(config.random_seed)
        logger.info(f"Random seed set to {config.random_seed}")

    # Validate configuration
    try:
        if config.system_type in ("bulk", "cluster") and not config.composition:
            logger.error(f"--composition is required for {config.system_type}")
            return 1
        if config.system_type == "surface" and not config.seed_path:
            logger.error("--seed is required for surface generation")
            return 1
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Generate structures
    logger.info(f"Generating {config.n_samples} {config.system_type} structures...")
    try:
        candidates = generate_structures(
            system_type=config.system_type,
            n_samples=config.n_samples,
            config=config,
            composition=config.composition,
            seed_path=config.seed_path,
            verbose=args.verbose,
        )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return 1
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1

    n_generated = len(candidates)
    logger.info(f"Generated {n_generated} structures")

    if n_generated == 0:
        logger.error("No structures generated")
        return 1

    # Step 2: LJ relaxation and energy filtering
    logger.info("Performing LJ relaxation and energy filtering...")
    try:
        relaxed = relax_and_filter(
            candidates=candidates,
            energy_filter_percent=config.energy_filter_percent,
            fmax=config.lj_fmax,
            max_steps=config.lj_max_steps,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Relaxation failed: {e}")
        return 1

    n_relaxed = len(relaxed)
    n_final = len(relaxed)
    logger.info(f"After relaxation: {n_relaxed}, after filtering: {n_final}")

    # Step 3: Write output structures
    logger.info(f"Writing {n_final} structures to {output_dir}...")

    structure_metadata = []
    for i, (record, energy) in enumerate(relaxed):
        filename = f"POSCAR_{i + 1:04d}"
        filepath = output_dir / filename

        # Write POSCAR using ASE
        try:
            from ase.io import write

            atoms = record.to_ase_atoms() if hasattr(record, "to_ase_atoms") else None
            if atoms is None:
                from atom_dataset_index.sampling.utils import record_to_ase

                atoms = record_to_ase(record)

            # For clusters (0D), add a bounding box for POSCAR format
            if args.system_type == "cluster":
                from ase import Atoms
                import numpy as np

                positions = atoms.get_positions()
                if len(positions) > 0:
                    # Calculate bounding box with padding
                    min_pos = positions.min(axis=0)
                    max_pos = positions.max(axis=0)
                    center = (min_pos + max_pos) / 2
                    size = (max_pos - min_pos).max() + 10.0  # Add 10Å padding

                    # Center atoms and set cell
                    atoms.center(about=center)
                    atoms.set_cell([size, size, size])
                    atoms.set_pbc(False)

            write(str(filepath), atoms, format="vasp")

            structure_metadata.append(
                {
                    "filename": filename,
                    "energy": round(energy, 4),
                    "source": record.metadata.get("source", "unknown"),
                    "space_group": record.metadata.get("space_group"),
                    "point_group": record.metadata.get("point_group"),
                    "miller_indices": record.metadata.get("miller_indices"),
                    "n_atoms": record.natoms,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to write {filename}: {e}")
            continue

    # Generate report
    elapsed = perf_counter() - start_time
    energy_values = [e for _, e in relaxed]

    report = SamplingReport(
        system_type=config.system_type,
        composition=config.composition,
        n_requested=config.n_samples,
        n_generated=n_generated,
        n_relaxed=n_relaxed,
        n_final=n_final,
        energy_filter_percent=config.energy_filter_percent,
        energy_range={
            "min": round(min(energy_values), 4) if energy_values else 0.0,
            "max": round(max(energy_values), 4) if energy_values else 0.0,
        },
        generation_time_seconds=round(elapsed, 2),
        structures=structure_metadata,
    )

    # Write report
    report_path = output_dir / "sampling_report.json"
    DatasetWriter.write_json(report_path, report.to_dict())
    logger.info(f"Report written to {report_path}")

    # Print summary
    print(f"\nSampling complete:")
    print(f"  Generated: {n_generated}")
    print(f"  Relaxed: {n_relaxed}")
    print(f"  Final (after filtering): {n_final}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_dir}")

    return 0


def main() -> int:
    """Entry point for sampling-mode command."""
    return sampling_mode_command()


if __name__ == "__main__":
    sys.exit(main())
