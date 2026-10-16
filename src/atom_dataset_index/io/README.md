# atom_dataset_index.io

## Metadata

- `module_id`: `atom_dataset_index.io`
- `module_version`: `1.3.0-r1`
- `project_version`: `1.3.0`
- `last_updated`: `2026-06-17`
- `generation_mode`: `auto`
- `source_files`: `src/atom_dataset_index/io/ase_db.py`, `src/atom_dataset_index/io/incar_parser.py`, `src/atom_dataset_index/io/poscar_reader.py`, `src/atom_dataset_index/io/writers.py`
- `upstream_modules`: `atom_dataset_index.domain`
- `downstream_modules`: `none`

## Module Purpose

Package leaf module for io.

## Public Symbols

- `class DatasetReadError`
  Base exception raised for dataset read failures.
- `class DatasetSchemaError`
  Raised when required ASE schema columns are unavailable.
- `class DatasetDecodeError`
  Raised when ASE binary payloads cannot be decoded safely.
- `class AseDbDatasetReader`
  Read a subset of ASE DB fields into package domain objects.
- `class IncarParams`
  Parsed INCAR parameters relevant to AIMD simulations.

Attributes:
    tebeg: Starting temperature in Kelvin (TEBEG).
    teend: Final temperature in Kelvin (TEEND).
    nsw: Number of ionic steps (NSW).
    potim: Time step in femtoseconds (POTIM).
    nbands: Number of bands (NBANDS).
    encut: Plane-wave cutoff energy in eV (ENCUT).
    isym: Symmetry treatment flag (ISYM).
    ismear: Smearing method (ISMEAR).
    sigma: Smearing width in eV (SIGMA).
    algo: Algorithm selection (ALGO).
    ibrion: Ionic relaxation method (IBRION), 0 for MD.
    mdalgo: MD algorithm (MDALGO).
    andersen_prob: Collision probability for Andersen thermostat.
    tein: Initial temperature if different from TEBEG.
    smass: Nose mass for Nose-Hoover thermostat.
- `class IncarParseError`
  Exception raised for INCAR parsing failures.
- `class IncarParser`
  Parse VASP INCAR files into structured parameters.

This parser handles the standard VASP INCAR format with key-value pairs.
It extracts parameters most relevant for AIMD simulation characterization.
- `parse_incar(path)`
  Convenience function to parse an INCAR file.

Args:
    path: Path to the INCAR file.

Returns:
    IncarParams containing the parsed parameters.
- `class PoscarReadError`
  Exception raised for POSCAR read failures.
- `class PoscarReader`
  Read VASP POSCAR files into StructureRecord objects.

This reader uses ASE as the underlying parser for robustness, but converts
the output to the package's internal StructureRecord representation for
consistency with other dataset sources.
- `read_poscar(path)`
  Convenience function to read a single POSCAR file.

Args:
    path: Path to the POSCAR file.
    sample_id: Sample ID to assign. Defaults to 0.

Returns:
    StructureRecord representing the structure.
- `read_poscar_directory(directory, pattern)`
  Convenience function to read all POSCAR files in a directory.

Args:
    directory: Path to the directory.
    pattern: Glob pattern to match files. Defaults to "POSCAR*".

Returns:
    List of StructureRecord objects.
- `class DatasetWriter`
  Write structured data to repository-friendly JSON files.
- `class ManifestWriter`
  Write task manifests consumed by downstream presentation steps.

## I/O Contract

- Inputs: Python domain objects, configuration objects, or primitive values defined by the source files listed above.
- Outputs: Stable objects or helper values returned by the public symbols in this directory.
- Format requirements: Downstream modules should depend on exported classes/functions rather than internal helper details.

## Update Rule

- Any interface change in this directory requires regenerating this `README.md` before higher-level docs are updated.
