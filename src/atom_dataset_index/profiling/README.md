# atom_dataset_index.profiling

## Metadata

- `module_id`: `atom_dataset_index.profiling`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/profiling/pools.py`, `src/atom_dataset_index/profiling/profiler.py`
- `upstream_modules`: `atom_dataset_index.metrics`, `atom_dataset_index.metric_core`
- `downstream_modules`: `atom_dataset_index.benchmark`, `atom_dataset_index.presentation`

## Module Purpose

Package leaf module for profiling.

## Public Symbols

- `class DatasetPool`
  A flattened pool ready for independent metric evaluation.
- `class PoolBuilder`
  Build evaluation pools via two-level stratification with small-pool merging.
- `list_available_metric_ids()`
  Return all supported metric ids in default registration order.
- `build_default_registry(selected_metric_ids)`
  Build the default registry with optional metric filtering.
- `class DatasetProfiler`
  Coordinate metric execution and assemble dataset profiles.
- `mean(values)`
  No docstring provided.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
