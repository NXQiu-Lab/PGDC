# atom_dataset_index.domain

## Metadata

- `module_id`: `atom_dataset_index.domain`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/domain/models.py`
- `upstream_modules`: `none`
- `downstream_modules`: `none`

## Module Purpose

Package leaf module for domain.

## Public Symbols

- `class DomainValidationError`
  Raised when domain model inputs are inconsistent.
- `class StructureRecord`
  Single structure sample with optional arrays already decoded.
- `class DatasetBundle`
  Dataset-level container shared by profiling and benchmark tasks.
- `class DatasetProfile`
  Aggregated profile generated from metric results.
- `class TrainingResult`
  Training artifact summary for SchNetPack baselines.
- `class BenchmarkResult`
  Joint profile and training comparison output.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
