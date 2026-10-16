# atom_dataset_index.benchmark

## Metadata

- `module_id`: `atom_dataset_index.benchmark`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/benchmark/runners.py`
- `upstream_modules`: `atom_dataset_index.profiling`, `atom_dataset_index.domain`
- `downstream_modules`: `atom_dataset_index.presentation`

## Module Purpose

Package leaf module for benchmark.

## Public Symbols

- `class DatasetSourceSpec`
  Description of a training data source consumed by benchmark runners.
- `class TrainingTaskSpec`
  Training task specification for baseline runners.
- `class LoadedDatasetSpec`
  Loaded data module and resolved property keys for a training run.
- `register_source(source_type, builder)`
  Register a dataset source builder for benchmark workflows.
- `list_registered_sources()`
  List all registered data source names.
- `class SchNetPackDatasetAdapter`
  Resolve a source spec into a SchNetPack datamodule and key mapping.
- `class SchNetRunner`
  Train SchNet baseline tasks via SchNetPack.
- `class PaiNNRunner`
  Train PaiNN baseline tasks via SchNetPack.
- `class BenchmarkRunner`
  Coordinate profile outputs with SchNetPack baseline runners.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
