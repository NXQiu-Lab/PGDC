# atom_dataset_index.presentation

## Metadata

- `module_id`: `atom_dataset_index.presentation`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/presentation/exporters.py`
- `upstream_modules`: `atom_dataset_index.benchmark`, `atom_dataset_index.profiling`
- `downstream_modules`: `none`

## Module Purpose

Presentation outputs including exports and manifests.

## Public Symbols

- `class RunManifest`
  Manifest written by task entrypoints.
- `class PresentationExporter`
  Export raw figures metadata, tables, and run manifests.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
