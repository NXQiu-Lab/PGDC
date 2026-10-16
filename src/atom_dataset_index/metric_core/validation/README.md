# atom_dataset_index.metric_core.validation

## Metadata

- `module_id`: `atom_dataset_index.metric_core.validation`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/metric_core/validation/checks.py`
- `upstream_modules`: `atom_dataset_index.domain`
- `downstream_modules`: `atom_dataset_index.metrics`, `atom_dataset_index.profiling`

## Module Purpose

Shared metric infrastructure reused by algorithm-level metrics.

## Public Symbols

- `ensure_non_empty_dataset(dataset)`
  No docstring provided.
- `ensure_energy_available(dataset)`
  No docstring provided.
- `ensure_positions_available(dataset)`
  No docstring provided.
- `ensure_forces_available(dataset)`
  No docstring provided.
- `ensure_constant_natoms(dataset)`
  No docstring provided.
- `ensure_periodic_dataset(dataset)`
  No docstring provided.
- `ensure_unique_sample_ids(dataset)`
  No docstring provided.
- `ensure_metric_result_consistent(result)`
  No docstring provided.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
