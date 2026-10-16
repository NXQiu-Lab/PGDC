# atom_dataset_index.metric_core.reference

## Metadata

- `module_id`: `atom_dataset_index.metric_core.reference`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/metric_core/reference/selector.py`
- `upstream_modules`: `atom_dataset_index.domain`
- `downstream_modules`: `atom_dataset_index.metrics`, `atom_dataset_index.profiling`

## Module Purpose

Shared metric infrastructure reused by algorithm-level metrics.

## Public Symbols

- `class ReferenceStructureSelector`
  Choose a reference structure for distortion metrics.
- `class ReferenceSelectionResult`
  Unified reference selection payload reused by multiple metrics.
- `class PrototypeStructureSelector`
  Select unified reference structures (stability + diversity) for pool-aware profiling.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
