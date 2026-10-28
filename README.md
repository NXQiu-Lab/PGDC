# PGDC

Profile-Guided Dataset Construction for Machine Learning Force Fields.

PGDC provides composition-aware dataset profiling, reference-structure
selection, candidate generation, physical prefiltering, trajectory mining,
and SchNetPack benchmark orchestration. The six profile dimensions are
Stability, Compositional Diversity, Structural Diversity, Uniformity,
Coverage, and Redundancy. Scores are diagnostic quantities, not universal
optimization objectives; higher Redundancy indicates stronger repetition.

Repository: https://github.com/NXQiu-Lab/PGDC

## Installation

Use Python 3.12 and a current version of [uv](https://docs.astral.sh/uv/).
The Python package remains named `atom-dataset-index` and its import namespace
is `atom_dataset_index`, preserving the existing public interfaces.

```bash
uv sync --locked --group test
uv run atom-dataset-index --help
```

The lockfile retains the tested dependency versions and uses the public
Aliyun PyPI mirror. It contains no private package-server address.
Core dependencies include ASE, DScribe, NumPy, SciPy, scikit-learn and
SchNetPack/PyTorch. Matplotlib remains a dependency of the core result
exporter; all plotting scripts are excluded from this distribution.

Random crystal generation additionally requires the sampling extra:

```bash
uv sync --locked --group test --extra sampling
```

## Dataset Profiling

The archived 17,540-structure SiO2 Round 3 snapshot matching the dataset size
in manuscript Figure 5 is available as a [GitHub Release](https://github.com/NXQiu-Lab/PGDC/releases/tag/fig5-sio2-round3-17540-v1).
Download `sio2_round3_17540_ev.db` for ASE-compatible data in eV and
eV/Angstrom; the release includes provenance, a full-database unit audit,
and SHA-256 checksums. This release provides the database snapshot; plotting
scripts and exact historical candidate coordinates are not included.

For other datasets, provide your own ASE database. Research databases and
trained models are not bundled in the source checkout. Periodic records need
atomic numbers, positions, cell, PBC, energy and forces. Dataset energies must
be in eV, forces in eV/Angstrom, and stress in eV/Angstrom^3 (positive tensile).
CP2K labels are converted at parse time.

```bash
uv run atom-dataset-index dataset-profile-run \
  --source-type ase_db \
  --datapath data/dataset.db \
  --preset core \
  --export-references
```

Outputs are created under `outputs/`, including `profile.json`,
`summary.json`, `summary.md`, per-metric results, the pool manifest and run
manifest. The built-in exporter can produce diagnostic figures. No training
model is required for static profiling. SOAP provenance records the backend
used, including any fallback descriptor.

Database audit:

```bash
uv run python scripts/common/unit_audit.py --db data/dataset.db
```

Inspect subcommand help for sampling, acquisition and benchmark options.
The core benchmark module includes SchNet/PaiNN training runners even though
the standalone plotting-dependent training script is not distributed.

## Tests

Run the full suite with pytest. Some test classes are pytest-style and are
not collected by unittest, so unittest alone is not the full verification.

```bash
uv run --group test python -m pytest tests -q
uv run python -m unittest discover -s tests
```

Calculator integration tests require your trained model at
`dataset/best_inference_model` and otherwise skip. The original phonon
integration test is retained but explicitly skips because its plotting
script is excluded.

Six legacy profiling tests use a deterministic temporary synthetic HfO2
database when the optional local research database is absent. Synthetic
energies and forces only exercise code paths; they are not physical labels.

## Public Distribution Scope

- `src/`: complete runtime package and source-adjacent documentation.
- `scripts/`: non-plotting profiling, dataset-processing, MD-calculator,
  CP2K, scheduling and baseline helpers.
- `tests/`: all original test modules, plus a discovery package marker.
- Packaging metadata, dependency lockfile, MIT license and public docs.

Excluded: all scripts containing plotting functionality, private remote
configurations, hard-coded server-address scripts, deployment-specific
recovery wrappers, research databases/models/run outputs, manuscript
materials, editor settings, local tool integrations and repository history.
See `scripts/README.md` for the supported script groups and caveats.

Remote workflows require SSH, CP2K and your own PBS/SLURM environment.
`config/remote_config.example.json` is a nonfunctional example, not a real
connection configuration. Copy it to an ignored local configuration and
provide your own host, user and working directory. Inspect generated job
templates before submission: some retained CP2K templates assume
`/opt/software/cp2k` and a head/compute-node path mapping.

The MLIP-4 baseline additionally requires the separately installed `mlip_4`
Python module and CP2K/MPI. These external executables and site-specific
jobs are not covered by the offline unit tests.

## License

MIT. See `LICENSE`. Third-party dependencies retain their own licenses.
