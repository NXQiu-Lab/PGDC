# atom_dataset_index.validation

## Metadata

- `module_id`: `atom_dataset_index.validation`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/validation/mlff_validator.py`
- `upstream_modules`: `none`
- `downstream_modules`: `none`

## Module Purpose

Package leaf module for validation.

## Public Symbols

- `class ClassEvaluationResult`
  Evaluation results for a single structure class.
- `class ComparisonResult`
  Statistical comparison between two structure classes.
- `load_schnetpack_model(model_path, device)`
  Load a SchNetPack / PaiNN model from disk.

Handles the common ``electronic_embeddings`` compatibility issue
for older PaiNN checkpoints.

Args:
    model_path: Path to saved torch model.
    device: "cpu" or "cuda".

Returns:
    Loaded torch model in eval mode.
- `evaluate_on_db(model, db_path, device, cutoff, batch_size)`
  Evaluate a model on all structures in an ASE DB.

Args:
    model: Loaded torch model.
    db_path: Path to ASE database (.db).
    device: "cpu" or "cuda".
    cutoff: Neighbor list cutoff (Å).
    batch_size: Batch size for inference (currently only 1 supported).

Returns:
    List of per-structure result dicts with keys:
    ``idx``, ``n_atoms``, ``energy_error``, ``energy_per_atom_error``,
    ``force_errors``, ``force_mae``, ``force_max_error``.
- `split_db_by_class(db_path, prefilter)`
  Split ASE DB structures by DeterministicPreFilter class labels.

Args:
    db_path: Path to ASE database.
    prefilter: DeterministicPreFilter instance.

Returns:
    Dict mapping class label → list of DB row indices (0-based).
- `aggregate_results(per_structure_results, class_label)`
  Aggregate per-structure errors into class-level statistics.
- `compare_class_errors(result_a, result_b)`
  Statistically compare errors between two structure classes.

Uses a simple Welch's t-test on per-structure force MAE values.
Energy comparison uses ratio (no test due to variable natoms).

Args:
    result_a: First class result.
    result_b: Second class result.

Returns:
    ComparisonResult with ratios and statistical test.
- `class MLFFValidator`
  High-level validator for MLFF closed-loop hypothesis testing.

Orchestrates the full pipeline:
1. Load model
2. Classify DB structures (EQ / MS / TP / TD / LU / AB)
3. Evaluate per-class errors
4. Compare TP vs EQ statistically

Example:
    >>> validator = MLFFValidator(model_path="best_model", db_path="dataset.db")
    >>> validator.run(device="cuda")
    >>> print(validator.comparison)

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
