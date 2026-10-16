# atom_dataset_index.sampling

## Metadata

- `module_id`: `atom_dataset_index.sampling`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/sampling/cli.py`, `src/atom_dataset_index/sampling/config.py`, `src/atom_dataset_index/sampling/explorers.py`, `src/atom_dataset_index/sampling/generator.py`, `src/atom_dataset_index/sampling/prefilter.py`, `src/atom_dataset_index/sampling/relaxer.py`, `src/atom_dataset_index/sampling/trajectory_miner.py`, `src/atom_dataset_index/sampling/utils.py`
- `upstream_modules`: `none`
- `downstream_modules`: `none`

## Module Purpose

Package leaf module for sampling.

## Public Symbols

- `build_sampling_parser()`
  Build argument parser for sampling-mode command.
- `sampling_mode_command(argv)`
  Execute sampling-mode command.

Args:
    argv: Command line arguments (defaults to sys.argv[1:])

Returns:
    Exit code (0 for success, 1 for error)
- `main()`
  Entry point for sampling-mode command.
- `class SamplingConfig`
  Configuration for structure sampling.

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
- `class SamplingReport`
  Report generated after sampling run.

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
- `config_from_yaml(path)`
  Load sampling configuration from YAML file.

Args:
    path: Path to YAML configuration file

Returns:
    SamplingConfig instance

Raises:
    FileNotFoundError: If config file doesn't exist
    ValueError: If config is invalid
- `config_to_yaml(config, path)`
  Save sampling configuration to YAML file.

Args:
    config: Configuration to save
    path: Output file path
- `class BoundaryExplorer`
  Explore the boundary of a dataset by perturbing far-field structures.

Uses descriptor-space distance to identify structures that are farthest
from the dataset centroid, then applies small physical perturbations
to generate new candidates near the boundary.

This implements the "explore周边 + 填补缺口" strategy: rather than
jumping to completely unknown regions, we nudge the existing boundary
outward step by step.

Example:
    >>> explorer = BoundaryExplorer(dataset)
    >>> candidates = explorer.explore(n_samples=20, perturbation_std=0.05)
- `class StrainPathGenerator`
  Generate a sequence of strained structures for transition-state induction.

Creates structures along deformation paths (tensile, compressive, shear)
that are designed to cross energy barriers during short AIMD runs.

Key insight: we do not need to run to fracture. The non-linear region
of the stress-strain curve (deviation from Hooke's law) is where atomic
bonds begin to reorder and transition states are densest.

Example:
    >>> generator = StrainPathGenerator()
    >>> candidates = generator.generate_uniaxial(
    ...     reference_atoms, strain_range=(-0.10, 0.15), n_steps=12
    ... )
- `class CandidateGenerator`
  Unified candidate generator supporting multiple strategies.

Replaces the monolithic generate_structures() with a strategy-pattern
design that can switch between:
    - "random": Original PyXtal/ASE random generation
    - "boundary": Dataset-aware boundary exploration
    - "strain": Strain-path generation for transition-state induction
    - "composite": Combined strategy (boundary + strain)

Example:
    >>> gen = CandidateGenerator(dataset=ref_dataset)
    >>> candidates = gen.generate(
    ...     strategy="boundary",
    ...     n_samples=20,
    ...     perturbation_std=0.05,
    ... )
- `generate_structures(system_type, n_samples, config, composition, seed_path, verbose)`
  Generate candidate structures based on system type.

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
- `class StructureClassResult`
  Result of structure classification.
- `class DeterministicPreFilter`
  Data-driven pre-filter for candidate structures.

Builds envelopes from a reference dataset and classifies candidates
into six categories:
    EQ: Equilibrium (near-stable)
    MS: Metastable
    TP: Transition-Prone (core target)
    TD: Thermally-Disordered
    LU: Locally-Unstable (rejected)
    AB: Absurd/physically impossible (rejected)

The filter does NOT require a trained MLIP. It uses:
    - Density envelope (per composition group)
    - Bond length envelope (per element pair)
    - Coordination number envelope (per element)
    - Force z-score (relative to reference pool statistics)
    - Structural distance metrics

Example:
    >>> from atom_dataset_index.io import AseDbDatasetReader
    >>> from atom_dataset_index.sampling import DeterministicPreFilter
    >>> dataset = AseDbDatasetReader("data/existing.db").read()
    >>> pf = DeterministicPreFilter(dataset)
    >>> result = pf.classify(candidate_record)
    >>> print(result.label, result.passed, result.reason)
- `relax_and_filter(candidates, energy_filter_percent, fmax, max_steps, epsilon, rc, verbose)`
  Relax structures with LJ potential and filter by energy.

This function performs fast pre-relaxation using a Lennard-Jones
potential to filter out unphysical or extremely high-energy
structures. The goal is not accurate energy ranking but removing
obviously bad candidates.

Args:
    candidates: Input structure candidates
    energy_filter_percent: Keep bottom X% by energy (0-100)
    fmax: Force convergence criterion for FIRE optimizer (eV/Å)
    max_steps: Maximum optimization steps
    epsilon: LJ well depth parameter (eV)
    rc: LJ cutoff radius (Å)
    verbose: Whether to log progress

Returns:
    List of (StructureRecord, energy) tuples, sorted by energy (lowest first)

Example:
    >>> candidates = [record1, record2, record3]
    >>> filtered = relax_and_filter(candidates, energy_filter_percent=50)
    >>> len(filtered)
    1 or 2  # Approximately 50% of input
- `relax_single_structure(record, fmax, max_steps, epsilon, rc)`
  Relax a single structure with LJ potential.

Args:
    record: Structure to relax
    fmax: Force convergence criterion
    max_steps: Maximum optimization steps
    epsilon: LJ well depth
    rc: LJ cutoff

Returns:
    Tuple of (relaxed_record, energy) or None if relaxation fails
- `compute_lj_energy(record, epsilon, rc)`
  Compute single-point LJ energy without relaxation.

Args:
    record: Structure to evaluate
    epsilon: LJ well depth
    rc: LJ cutoff

Returns:
    LJ energy or None if calculation fails
- `class Event`
  A detected event in a trajectory.
- `class MiningResult`
  Result of trajectory mining.
- `class TrajectoryMiner`
  Mine CP2K AIMD trajectories for transition-state events.

Combines three event detectors:
1. **Energy jump**: Sudden potential energy changes (bond breaking/forming).
2. **Volume change**: Cell volume jumps (phase transitions, NPT only).
3. **Coordination change**: Local coordination number shifts (defect migration).

Sampling strategy:
- Event windows: dense sampling (every frame) around detected events.
- Non-event regions: sparse sampling (every N frames) to avoid redundancy.

Example:
    >>> miner = TrajectoryMiner(energy_records, trajectory_frames)
    >>> result = miner.mine(
    ...     event_window=5,
    ...     dense_stride=1,
    ...     sparse_stride=20,
    ... )
    >>> print(f"Detected {len(result.events)} events, "
    ...       f"sampled {len(result.sampled_indices)} frames")
- `mine_cp2k_trajectory(pos_xyz_path, ener_path, frc_xyz_path, cell)`
  Convenience function: parse CP2K files and mine in one call.

Args:
    pos_xyz_path: Path to CP2K position xyz file.
    ener_path: Optional path to .ener file.
    frc_xyz_path: Optional path to forces xyz file.
    cell: Fixed cell matrix (if xyz lacks lattice info).
    **kwargs: Passed to TrajectoryMiner.mine().

Returns:
    MiningResult.
- `pyxtal_to_record(crystal, sample_id)`
  Convert PyXtal crystal object to StructureRecord.

Args:
    crystal: PyXtal crystal object
    sample_id: Sample ID for the record

Returns:
    StructureRecord representation
- `ase_to_record(atoms, sample_id)`
  Convert ASE Atoms object to StructureRecord.

Args:
    atoms: ASE Atoms object
    sample_id: Sample ID for the record

Returns:
    StructureRecord representation
- `record_to_ase(record)`
  Convert StructureRecord to ASE Atoms object.

Args:
    record: StructureRecord to convert

Returns:
    ASE Atoms object
- `update_positions(record, atoms)`
  Create new record with updated positions from ASE Atoms.

Args:
    record: Original StructureRecord
    atoms: ASE Atoms with new positions

Returns:
    New StructureRecord with updated positions and cell
- `compute_min_distance(record)`
  Compute minimum interatomic distance in a structure.

Args:
    record: Structure to analyze

Returns:
    Minimum distance between any two atoms (Å)
- `basic_validation(record, min_distance)`
  Perform basic validation on a structure.

Checks:
- Minimum interatomic distance >= min_distance
- Cell volume is positive (if periodic)
- Positions match atomic_numbers count

Args:
    record: Structure to validate
    min_distance: Minimum allowed interatomic distance (Å)

Returns:
    True if structure passes validation
- `compute_cell_volume(cell)`
  Compute volume of a unit cell.

Args:
    cell: 3x3 cell matrix (vectors as rows)

Returns:
    Cell volume (Å³)
- `perturb_atoms(atoms, std)`
  Add random perturbation to atomic positions.

Args:
    atoms: ASE Atoms object
    std: Standard deviation for perturbation (Å)

Returns:
    Perturbed Atoms object (modified in place)
- `enumerate_miller_indices(max_index)`
  Enumerate Miller indices up to max_index.

Generates unique Miller indices (hkl) where at least one index
is equal to max_index and others are <= max_index.

Args:
    max_index: Maximum Miller index value

Returns:
    List of (h, k, l) tuples
- `parse_composition(formula)`
  Parse chemical formula string to composition dict.

Args:
    formula: Chemical formula (e.g., "Hf4O8", "NaCl", "Au55")

Returns:
    Dictionary of {element: count}

Raises:
    ValueError: If formula cannot be parsed
- `composition_to_string(composition)`
  Convert composition dict to string representation.

Args:
    composition: Dictionary of {element: count}

Returns:
    Chemical formula string

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
