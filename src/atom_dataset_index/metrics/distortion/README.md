# atom_dataset_index.metrics.distortion

## Metadata

- `module_id`: `atom_dataset_index.metrics.distortion`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/metrics/distortion/calculators.py`
- `upstream_modules`: `atom_dataset_index.metric_core`, `atom_dataset_index.domain`
- `downstream_modules`: `atom_dataset_index.profiling`, `atom_dataset_index.benchmark`

## Module Purpose

Algorithm-level metric implementations.

## Public Symbols

- `class StructuralDistortionCalculator`
  Measure descriptor-space distance to stability and diversity references.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
