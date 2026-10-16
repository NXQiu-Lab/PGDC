# atom_dataset_index.metrics.diversity

## Metadata

- `module_id`: `atom_dataset_index.metrics.diversity`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/metrics/diversity/calculators.py`
- `upstream_modules`: `atom_dataset_index.metric_core`, `atom_dataset_index.domain`
- `downstream_modules`: `atom_dataset_index.profiling`, `atom_dataset_index.benchmark`

## Module Purpose

Algorithm-level metric implementations.

## Public Symbols

- `class DescriptorDiversityCalculator`
  Simple diversity metric based on descriptor spread.
- `class StructuralDiversityCalculator`
  Measure structural diversity via descriptor-space clustering.

Unlike composition-driven diversity, this metric is sensitive to local
structure variation within a single phase.  It clusters the SOAP descriptor
matrix of each pool and reports the normalized occupancy entropy of the
resulting clusters.
- `class DescriptorRedundancyCalculator`
  Measure descriptor-space redundancy via k-th nearest-neighbor distance.

A low median k-NN distance indicates that many records are packed closely
together in descriptor space, i.e. the dataset is locally redundant.  The
calculator emits a raw distance and a 0-100 negative score:

- ``redundancy_score``: high = redundant (negative indicator)

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
