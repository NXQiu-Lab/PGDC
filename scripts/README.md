# Public Scripts

Run scripts from the repository root with `uv run python <script>`. Consult
each script's module documentation or `--help` where provided. Scripts
require user-supplied input data and do not include research outputs.

| Group | Purpose |
| --- | --- |
| `run_task.py`, `generate_docs.py` | CLI task dispatch and source documentation |
| `common/` | CP2K parsing, unit audit/conversion, database backfill, SSH clients |
| `analysis/` | Subset selection, sparsification, reference analysis, model evaluation |
| `sio2/` | Structure generation, CP2K dataset construction, recovery and cohort extraction |
| `amorphous_carbon/` | CP2K construction, stress relabeling, label synchronization and merging |
| `amorphous_carbon/mlip4_al_baseline/` | MTP training, MD exploration, CP2K labeling and dataset conversion |
| `md/` | PaiNN calculator, repulsive-wall/force-cap calculators and NPT smoke script |
| `paper/` | Non-plotting database conversion, ablation preparation and reference analysis |
| `pbs_utils.py`, `slurm_utils.py` | Local scheduler submission helpers |

## Exclusions

All scripts that import plotting libraries or generate plots are excluded,
including combined computation/plotting workflows. The standalone
`painn.py` is excluded because it imports Matplotlib. Core training remains
available in `src/atom_dataset_index/benchmark/`.

Private remote configurations, the hard-coded SiO2 iteration-deployment
script and the deployment-specific MLIP recovery shell wrapper are not
included. Generic recovery helpers remain available. Original research
script READMEs were replaced to avoid exposing deployment details or
advertising excluded workflows.

## External Requirements

CP2K workflows require CP2K, its basis/potential files and MPI as appropriate.
Remote jobs additionally need SSH/SCP and PBS or SLURM. Site-specific CP2K
installation paths and partition/node settings in templates must be adapted
to your own cluster. The clients support key-based authentication; do not
commit passwords or private keys.

The MLIP baseline requires `mlip_4`, installed separately under its own
license. `al_config.yaml` is a public example with relative paths and local
CP2K mode; configure it before a real run. `prepare_seed.py` and
`convert_to_db.py` accept path arguments with generic relative defaults.

The evaluation and legacy-repair scripts target particular research layouts
and expect matching local input files. They are not automatic benchmarks
on a fresh installation. Back up databases before using any repair script.

All database labels must use eV, eV/Angstrom and eV/Angstrom^3. Use
`common/unit_audit.py` before training or merging labels; legacy conversions
are provided by `common/fix_db_units_to_ev.py`.
