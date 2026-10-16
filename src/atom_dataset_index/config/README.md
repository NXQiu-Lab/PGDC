# atom_dataset_index.config

## Metadata

- `module_id`: `atom_dataset_index.config`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/config/models.py`
- `upstream_modules`: `none`
- `downstream_modules`: `none`

## Module Purpose

Package leaf module for config.

## Public Symbols

- `class PathConfig`
  Repository path layout used by tasks and output exporters.
- `class MetricConfig`
  Parameters that control metric execution and descriptor behavior.
- `class BenchmarkConfig`
  Parameters used for subset construction and training orchestration.
- `class PlotConfig`
  Rendering defaults for presentation outputs.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
