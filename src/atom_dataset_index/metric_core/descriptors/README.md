# atom_dataset_index.metric_core.descriptors

## Metadata

- `module_id`: `atom_dataset_index.metric_core.descriptors`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/metric_core/descriptors/base.py`
- `upstream_modules`: `atom_dataset_index.domain`
- `downstream_modules`: `atom_dataset_index.metrics`, `atom_dataset_index.profiling`

## Module Purpose

Shared metric infrastructure reused by algorithm-level metrics.

## Public Symbols

- `class DescriptorAdapter`
  Abstract adapter for converting structures into vector descriptors.
- `class IdentityDescriptorAdapter`
  Simple adapter based on flattened positions for bootstrap workflows.
- `class SOAPDescriptorAdapter`
  Structure-level descriptor with SOAP-first semantics and deterministic fallback.

If a SOAP backend is not available in the environment, this adapter falls back to a
fixed-length periodic radial fingerprint so v1 profiling can still run end-to-end.

Supports both single-structure and batch encoding for improved performance.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
