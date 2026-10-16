"""
SiO2 dataset utilities for CP2K AIMD dataset building.

Provides:
  - Initial SiO2 configuration generation (α-quartz, β-cristobalite)
  - CP2K DFT AIMD input file generation for SiO2 (insulator, PBE/GTH)
  - CP2K output parsing and ASE database operations

Reuses generic CP2K parsers / DB helpers from scripts.common.cp2k_utils.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.spacegroup import crystal

# Re-use generic CP2K output parsers and DB helpers from scripts.common
_scripts_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_scripts_dir))
from common.cp2k_utils import (
    get_db_stats,
    load_json,
    load_al_manifest,
    merge_trajectory_to_db,
    parse_cp2k_cell,
    parse_cp2k_ener,
    parse_cp2k_trajectory,
    save_al_manifest,
    save_json,
)


# ---------------------------------------------------------------------------
# 1. Element properties for SiO2
# ---------------------------------------------------------------------------

ELEMENT_PROPS_SIO2: dict[str, dict[str, str]] = {
    "Si": {"potential": "GTH-PBE-q4", "basis": "DZVP-MOLOPT-SR-GTH"},
    "O":  {"potential": "GTH-PBE-q6", "basis": "DZVP-MOLOPT-SR-GTH"},
}

# ---------------------------------------------------------------------------
# 2. SiO2 structure generation
# ---------------------------------------------------------------------------

def generate_alpha_quartz(n_atoms_target: int = 72, seed: int | None = None) -> Atoms:
    """Generate α-quartz supercell from standard crystallographic data.

    Primitive cell has 3 Si + 6 O = 9 atoms.
    A 2×2×2 supercell gives 72 atoms.
    """
    # Standard α-quartz (P3_2 21) unit cell
    cell = np.array([
        [2.4574830883638521, -4.2564855677874682, 0.0],
        [2.4574830883638521,  4.2564855677874682, 0.0],
        [0.0,                  0.0,                 5.43130114],
    ])

    symbols = ["Si"] * 3 + ["O"] * 6
    frac = np.array([
        [0.53108859, 0.53108859, 0.0],
        [0.0,        0.46891141, 0.6666666666666661],
        [0.46891141, 0.0,        0.3333333333333330],
        [0.26922311, 0.41339429, 0.78489098],
        [0.58660571, 0.85582882, 0.1182243133333330],
        [0.14417118, 0.73077689, 0.4515576466666660],
        [0.41339429, 0.26922311, 0.2151090199999990],
        [0.73077689, 0.14417118, 0.5484423533333330],
        [0.85582882, 0.58660571, 0.8817756866666661],
    ])

    atoms = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)

    # Build supercell to reach target size
    n_prim = len(atoms)  # 9
    repeat = max(1, int(round((n_atoms_target / n_prim) ** (1.0 / 3.0))))
    while (repeat ** 3) * n_prim < n_atoms_target:
        repeat += 1
    atoms = atoms * (repeat, repeat, repeat)

    # Small random perturbation to break perfect symmetry
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pos = atoms.get_positions()
    noise = rng.normal(0, 0.05, pos.shape)
    atoms.set_positions(pos + noise)
    atoms.wrap()
    atoms.info["structure_type"] = "alpha_quartz"
    atoms.info["n_atoms_target"] = n_atoms_target
    atoms.set_pbc(True)
    return atoms


def generate_beta_cristobalite(n_atoms_target: int = 72, seed: int | None = None) -> Atoms:
    """Generate β-cristobalite supercell.

    Primitive cell (trigonal setting of P2_13) has 3 Si + 6 O = 9 atoms.
    A 2×2×2 supercell gives 72 atoms.
    """
    cell = np.array([
        [2.5302611141839888, -4.3825408061825062, 0.0],
        [2.5302611141839888,  4.3825408061825062, 0.0],
        [0.0,                  0.0,                 5.5413533099999999],
    ])

    symbols = ["Si"] * 3 + ["O"] * 6
    frac = np.array([
        [0.0,        0.5,        0.3333333333333330],
        [0.5,        0.5,        0.6666666666666661],
        [0.5,        0.0,        0.0],
        [0.58278926, 0.79139463, 0.8333333333333330],
        [0.79139463, 0.58278926, 0.5],
        [0.79139463, 0.20860537, 0.1666666666666660],
        [0.41721074, 0.20860537, 0.8333333333333330],
        [0.20860537, 0.41721074, 0.5],
        [0.20860537, 0.79139463, 0.1666666666666660],
    ])

    atoms = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)

    # Build supercell to reach target size
    n_prim = len(atoms)  # 9
    repeat = max(1, int(round((n_atoms_target / n_prim) ** (1.0 / 3.0))))
    while (repeat ** 3) * n_prim < n_atoms_target:
        repeat += 1
    atoms = atoms * (repeat, repeat, repeat)

    # Small random perturbation
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pos = atoms.get_positions()
    noise = rng.normal(0, 0.05, pos.shape)
    atoms.set_positions(pos + noise)
    atoms.wrap()
    atoms.info["structure_type"] = "beta_cristobalite"
    atoms.info["n_atoms_target"] = n_atoms_target
    atoms.set_pbc(True)
    return atoms


def generate_sio2_structure(
    structure_type: str = "alpha_quartz",
    n_atoms: int = 72,
    temperature: float = 300.0,
    seed: int | None = None,
) -> Atoms:
    """Generate a SiO2 supercell.

    Args:
        structure_type: "alpha_quartz" or "beta_cristobalite".
        n_atoms: Target number of atoms.
        temperature: Target temperature (K) for thermal disorder magnitude.
        seed: Random seed.

    Returns:
        ASE Atoms object with periodic boundary conditions.
    """
    if structure_type == "alpha_quartz":
        atoms = generate_alpha_quartz(n_atoms_target=n_atoms, seed=seed)
    elif structure_type == "beta_cristobalite":
        atoms = generate_beta_cristobalite(n_atoms_target=n_atoms, seed=seed)
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")

    # Add thermal disorder proportional to temperature
    if temperature > 0 and seed is not None:
        rng = np.random.default_rng(seed + 42)
        pos = atoms.get_positions()
        # RMS displacement ~ 0.02 Å at 300 K, scales with sqrt(T)
        # Keep small to avoid breaking coordination in layered supercells
        rms = 0.02 * (temperature / 300.0) ** 0.5
        if rms > 0.15:
            rms = 0.15
        noise = rng.normal(0, rms, pos.shape)
        atoms.set_positions(pos + noise)
        atoms.wrap()

    atoms.info["temperature"] = float(temperature)
    return atoms


# ---------------------------------------------------------------------------
# 3. CP2K DFT AIMD input generation for SiO2 (insulator)
# ---------------------------------------------------------------------------

CP2K_DFT_COMMON_SIO2 = """&GLOBAL
  PROJECT {project_name}
  RUN_TYPE {run_type}
  PRINT_LEVEL LOW
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME  {basis_set_path}
    POTENTIAL_FILE_NAME  {potential_path}
{wfn_restart_line}    &QS
      EPS_DEFAULT 1.0E-12
      EXTRAPOLATION PS
      EXTRAPOLATION_ORDER 3
    &END QS
    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF {rel_cutoff}
      NGRIDS 4
    &END MGRID
    &SCF
      SCF_GUESS {scf_guess}
      EPS_SCF {eps_scf}
      MAX_SCF {max_scf}
{smear_block}{scf_method_block}    &END SCF
    &XC
      &XC_FUNCTIONAL {xc_functional}
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      A {cell_ax:.6f} {cell_ay:.6f} {cell_az:.6f}
      B {cell_bx:.6f} {cell_by:.6f} {cell_bz:.6f}
      C {cell_cx:.6f} {cell_cy:.6f} {cell_cz:.6f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{velocity_block}{kinds}
    &TOPOLOGY
      CONNECTIVITY OFF
    &END TOPOLOGY
  &END SUBSYS
{force_eval_print_block}&END FORCE_EVAL
"""

CP2K_MD_MOTION_SIO2 = """
&MOTION
  &MD
    ENSEMBLE {ensemble}
    STEPS {n_steps}
    TIMESTEP {timestep}
    TEMPERATURE {temperature}
    &THERMOSTAT
      TYPE CSVR
      &CSVR
        TIMECON {thermostat_timecon}
      &END CSVR
    &END THERMOSTAT
{barostat_block}  &END MD
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        MD {output_stride}
      &END EACH
    &END TRAJECTORY
    &FORCES ON
      &EACH
        MD {output_stride}
      &END EACH
    &END FORCES
    &RESTART
      &EACH
        MD 1000
      &END EACH
    &END RESTART
  &END PRINT
&END MOTION
"""

CP2K_GEO_OPT_MOTION_SIO2 = """
&MOTION
  &GEO_OPT
    TYPE MINIMIZATION
    OPTIMIZER BFGS
    MAX_ITER 200
    MAX_FORCE 4.5E-4
  &END GEO_OPT
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        GEO_OPT 1
      &END EACH
    &END TRAJECTORY
    &FORCES ON
      &EACH
        GEO_OPT 1
      &END EACH
    &END FORCES
    &RESTART
      &EACH
        GEO_OPT 1
      &END EACH
    &END RESTART
  &END PRINT
&END MOTION
"""

CP2K_ENERGY_FORCE_MOTION_SIO2 = """
# No &MOTION block needed for RUN_TYPE ENERGY_FORCE.
"""

CP2K_BALLISTIC_MOTION_SIO2 = """
&MOTION
  &MD
    ENSEMBLE {ensemble}
    STEPS {n_steps}
    TIMESTEP {timestep}
    TEMPERATURE {temperature}
{thermostat_block}  &END MD
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        MD {output_stride}
      &END EACH
    &END TRAJECTORY
    &FORCES ON
      &EACH
        MD {output_stride}
      &END EACH
    &END FORCES
    &RESTART
      &EACH
        MD 1000
      &END EACH
    &END RESTART
  &END PRINT
&END MOTION
"""


def build_cp2k_input_sio2(
    atoms: Atoms,
    project_name: str = "sio2",
    temperature: float = 300.0,
    n_steps: int = 20000,
    timestep: float = 1.0,
    output_stride: int = 40,
    cutoff: int = 500,
    rel_cutoff: int = 50,
    xc_functional: str = "PBE",
    basis_set: str = "DZVP-MOLOPT-SR-GTH",
    scf_guess: str = "ATOMIC",
    use_ot: bool = True,
    electronic_temperature: float | None = None,
    wfn_restart_file: str | None = None,
    ext_restart_file: str | None = None,
    thermostat_timecon: float = 200.0,
    cp2k_data_dir: str | None = None,
    run_type: str = "MD",
    ensemble: str = "NVT",
    pressure_gpa: float = 0.0,
    barostat_timecon: float = 1000.0,
    ballistic_velocity_temp: float | None = None,
    ballistic_ensemble: str = "NVE",
) -> str:
    """Generate a CP2K DFT input file for SiO2 (insulator / wide-gap oxide).

    Differs from the metallic HEA template in that:
      - No ADDED_MOS (insulator).
      - No FERMI_DIRAC smearing by default (optional small smearing for
        high-temperature MD where the gap may shrink).
      - Uses OT (orbital transformation) by default for faster SCF.

    Args:
        atoms: ASE Atoms object.
        project_name: PROJECT name.
        temperature: Thermostat target temperature (K).
        n_steps: Number of MD steps.
        timestep: Time step (fs).
        output_stride: Write output every N steps.
        cutoff: Plane-wave cutoff (Ry).
        rel_cutoff: Relative cutoff (Ry).
        xc_functional: XC functional.
        basis_set: Basis set name.
        scf_guess: SCF guess type.
        use_ot: Use OT SCF solver (default True for insulators).
        electronic_temperature: If set, enables FERMI_DIRAC smearing at
            this electronic temperature (K). Useful for very high-T MD.
        wfn_restart_file: WFN restart file for &DFT block.
        ext_restart_file: EXT_RESTART file for geometry/velocities.
        thermostat_timecon: Thermostat time constant (fs).
        cp2k_data_dir: CP2K data directory. If None, inferred from env.
        run_type: "MD" or "GEO_OPT".
        ensemble: MD ensemble, "NVT" or "NPT_I".
        pressure_gpa: Target pressure for NPT_I (GPa). Converted to bar internally.
        barostat_timecon: Barostat time constant for NPT_I (fs).
        ballistic_velocity_temp: If set, inject a Maxwell-Boltzmann velocity
            block at this temperature (K) and run a short ballistic MD.
        ballistic_ensemble: "NVE" or "NVT" for ballistic mode.

    Returns:
        CP2K input file content as string.
    """
    if cp2k_data_dir is None:
        cp2k_data_dir = os.environ.get(
            "CP2K_DATA_DIR",
            os.environ.get("CP2K_PATH", "/opt/software/cp2k") + "/data",
        )

    basis_set_path = f"{cp2k_data_dir}/BASIS_MOLOPT"
    potential_path = f"{cp2k_data_dir}/GTH_POTENTIALS"

    wfn_restart_line = f"    WFN_RESTART_FILE_NAME {wfn_restart_file}\n" if wfn_restart_file else ""

    cell = atoms.get_cell()
    cell_ax, cell_ay, cell_az = cell[0]
    cell_bx, cell_by, cell_bz = cell[1]
    cell_cx, cell_cy, cell_cz = cell[2]

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    coord_lines = []
    for sym, pos in zip(symbols, positions):
        coord_lines.append(
            f"      {sym:4s} {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}"
        )
    coords_str = "\n".join(coord_lines)

    unique_elements = sorted(set(symbols))
    kind_lines = []
    for elem in unique_elements:
        props = ELEMENT_PROPS_SIO2.get(elem, {"potential": f"GTH-{xc_functional}", "basis": basis_set})
        kind_lines.append(
            f"    &KIND {elem}\n"
            f"      BASIS_SET {props['basis']}\n"
            f"      POTENTIAL {props['potential']}\n"
            f"    &END KIND"
        )
    kinds_str = "\n".join(kind_lines)

    is_ballistic = ballistic_velocity_temp is not None
    if run_type == "GEO_OPT":
        eps_scf = "1.0E-6"
        max_scf = "200"
        motion_block = CP2K_GEO_OPT_MOTION_SIO2
        force_eval_print_block = ""
    elif run_type == "ENERGY_FORCE":
        eps_scf = "1.0E-7"
        max_scf = "300"
        motion_block = CP2K_ENERGY_FORCE_MOTION_SIO2
        force_eval_print_block = """
  &PRINT
    &FORCES ON
      FORCE_UNIT eV/angstrom
      NDIGITS 8
    &END FORCES
  &END PRINT
"""
    else:
        eps_scf = "1.0E-6"
        max_scf = "300"
        if ensemble.upper() == "NPT_I":
            barostat_block = (
                "    &BAROSTAT\n"
                f"      PRESSURE {pressure_gpa * 10000.0:.1f}\n"
                f"      TIMECON {barostat_timecon}\n"
                "    &END BAROSTAT\n"
            )
        else:
            barostat_block = ""
        if is_ballistic:
            if ballistic_ensemble.upper() == "NVE":
                thermostat_block = ""
            else:
                thermostat_block = (
                    "    &THERMOSTAT\n"
                    "      TYPE CSVR\n"
                    "      &CSVR\n"
                    f"        TIMECON {thermostat_timecon}\n"
                    "      &END CSVR\n"
                    "    &END THERMOSTAT\n"
                )
            motion_block = CP2K_BALLISTIC_MOTION_SIO2.format(
                ensemble=ballistic_ensemble,
                n_steps=n_steps,
                timestep=timestep,
                temperature=temperature,
                thermostat_block=thermostat_block,
                output_stride=output_stride,
            )
        else:
            motion_block = CP2K_MD_MOTION_SIO2.format(
                ensemble=ensemble,
                n_steps=n_steps,
                timestep=timestep,
                temperature=temperature,
                thermostat_timecon=thermostat_timecon,
                barostat_block=barostat_block,
                output_stride=output_stride,
            )
        if ensemble.upper() == "NPT_I":
            force_eval_print_block = "  STRESS_TENSOR ANALYTICAL\n"
        else:
            force_eval_print_block = ""

    # Smearing block (optional, for high-T MD where gap may close)
    # CP2K requires ADDED_MOS when smearing is enabled.
    # IMPORTANT: OT and smearing are INCOMPATIBLE in CP2K. For insulators
    # like SiO2, OT should always be used without smearing.
    if electronic_temperature is not None and electronic_temperature > 0:
        if use_ot:
            raise ValueError(
                "CP2K does not support OT with smearing (ADDED_MOS). "
                "SiO2 is an insulator and should use OT without smearing. "
                "Remove electronic_temperature or set use_ot=False."
            )
        # Compute ADDED_MOS from valence electrons
        valence_map = {"Si": 4, "O": 6}
        n_electrons = sum(
            valence_map.get(sym, 4) for sym in atoms.get_chemical_symbols()
        )
        n_occ = n_electrons // 2  # spin-unpolarised
        added_mos = max(50, n_occ // 3)
        smear_block = (
            f"      ADDED_MOS {added_mos}\n"
            "      &SMEAR ON\n"
            "        METHOD FERMI_DIRAC\n"
            f"        ELECTRONIC_TEMPERATURE [K] {electronic_temperature}\n"
            "      &END SMEAR\n"
        )
    else:
        smear_block = ""

    # SCF method block
    if use_ot:
        scf_method_block = (
            "      &OT ON\n"
            "        PRECONDITIONER FULL_SINGLE_INVERSE\n"
            "        MINIMIZER DIIS\n"
            "      &END OT\n"
            "      &OUTER_SCF\n"
            "        MAX_SCF 8\n"
            f"        EPS_SCF {eps_scf}\n"
            "      &END OUTER_SCF\n"
        )
    else:
        scf_method_block = (
            "      &DIAGONALIZATION ON\n"
            "        ALGORITHM STANDARD\n"
            "      &END DIAGONALIZATION\n"
            "      &MIXING\n"
            "        METHOD BROYDEN_MIXING\n"
            "        ALPHA 0.2\n"
            "        NBUFFER 8\n"
            "      &END MIXING\n"
        )

    velocity_block = ""
    if is_ballistic:
        temp_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(temp_atoms, temperature_K=ballistic_velocity_temp)
        velocities = temp_atoms.get_velocities()
        # CP2K &VELOCITY expects atomic units: bohr/au_t
        _ANG_FS_TO_BOHR_AUT = 0.02418884 / 0.529177  # ≈ 0.04572
        velocities_bohr_aut = velocities * _ANG_FS_TO_BOHR_AUT
        vel_lines = ["    &VELOCITY"]
        for v in velocities_bohr_aut:
            vel_lines.append(
                f"      {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}"
            )
        vel_lines.append("    &END VELOCITY")
        velocity_block = "\n".join(vel_lines) + "\n"

    common_block = CP2K_DFT_COMMON_SIO2.format(
        project_name=project_name,
        run_type=run_type,
        basis_set_path=basis_set_path,
        potential_path=potential_path,
        cutoff=cutoff,
        rel_cutoff=rel_cutoff,
        scf_guess=scf_guess,
        eps_scf=eps_scf,
        max_scf=max_scf,
        smear_block=smear_block,
        scf_method_block=scf_method_block,
        xc_functional=xc_functional,
        wfn_restart_line=wfn_restart_line,
        force_eval_print_block=force_eval_print_block,
        velocity_block=velocity_block,
        cell_ax=cell_ax, cell_ay=cell_ay, cell_az=cell_az,
        cell_bx=cell_bx, cell_by=cell_by, cell_bz=cell_bz,
        cell_cx=cell_cx, cell_cy=cell_cy, cell_cz=cell_cz,
        coords=coords_str,
        kinds=kinds_str,
    )

    if ext_restart_file:
        restart_block = f"""&EXT_RESTART
  RESTART_FILE_NAME {ext_restart_file}
&END EXT_RESTART
"""
        common_block = restart_block + "\n" + common_block

    return common_block + motion_block


# ---------------------------------------------------------------------------
# 4. CP2K ENERGY_FORCE output parsing
# ---------------------------------------------------------------------------

_HARTREE_TO_EV = 27.211386245988
_AU_TO_EV_PER_ANGSTROM = 51.42208619083232


def parse_cp2k_single_point_out(out_path: str | Path) -> dict[str, Any] | None:
    """Parse energy and atomic forces from a CP2K ENERGY_FORCE output file.

    Args:
        out_path: Path to the CP2K ``out`` file.

    Returns:
        Dict with ``energy`` (eV) and ``forces`` (eV/Å, shape (n_atoms, 3))
        or ``None`` if parsing fails.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return None

    text = out_path.read_text()

    # Energy: take the last occurrence of the total FORCE_EVAL energy line.
    energy_hartree: float | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]"):
            parts = line.split()
            try:
                energy_hartree = float(parts[-1])
            except (ValueError, IndexError):
                continue

    if energy_hartree is None:
        return None

    energy_ev = energy_hartree * _HARTREE_TO_EV

    # Forces: CP2K prints a "FORCES|" block when &FORCE_EVAL / &PRINT / &FORCES
    # is enabled. The input sets FORCE_UNIT eV/angstrom, so values are eV/Å.
    # Format (CP2K 2026.1):
    #   FORCES| Atomic forces [eV/angstrom]
    #   FORCES|   Atom     x               y               z               |f|
    #   FORCES|      1  9.03163665E-01 -2.69201161E-01 -1.14025264E+00   1.47930725E+00
    #   ...
    #   FORCES| Sum    -3.15963429E-01 -6.27497969E-02 -2.81549446E-02
    forces: list[tuple[float, float, float]] = []
    in_forces = False
    header_seen = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FORCES| Atomic forces"):
            in_forces = True
            header_seen = False
            continue
        if not in_forces:
            continue
        if not stripped.startswith("FORCES|"):
            break
        body = stripped[len("FORCES|"):].strip()
        if not header_seen:
            if body.startswith("Atom"):
                header_seen = True
            continue
        if body.startswith("Sum") or body.startswith("Total atomic force"):
            break
        parts = body.split()
        # Expect: index fx fy fz [|f|]
        if len(parts) >= 4:
            try:
                fx = float(parts[-4])
                fy = float(parts[-3])
                fz = float(parts[-2])
                forces.append((fx, fy, fz))
            except (ValueError, IndexError):
                continue

    if not forces:
        return None

    return {"energy": energy_ev, "forces": tuple(forces)}


def attach_force_field_calculator(
    atoms: Atoms,
    force_field_type: str = "morse",
    force_field_param_file: str | Path | None = None,
) -> Atoms:
    """Attach a fast ASE built-in force field to *atoms* for candidate pre-screening.

    Supported ``force_field_type`` values:

    - ``morse`` (default): ASE MorsePotential (single-element parameters,
      works for any element but does not distinguish Si from O).
    - ``auto``: try the EMT potential; if the structure contains
      elements that EMT does not parameterize (e.g. Si), fall back to a generic
      Lennard-Jones potential.
    - ``emt``: ASE EMT potential (fast, no external files).
    - ``lj``: generic Lennard-Jones potential (chemically crude, works for any
      element).
    - ``tersoff``: ASE Tersoff calculator. Requires a LAMMPS-style Tersoff
      parameter file via ``force_field_param_file``.

    The force-field estimate is intended only to provide energy/force tags for
    the prefilter / acquisition steps; production AIMD trajectories are still
    generated with CP2K DFT.
    """
    force_field_type = force_field_type.lower()

    if force_field_type == "tersoff":
        if force_field_param_file is None:
            raise ValueError(
                "Tersoff force field requires --force-field-param-file "
                "(LAMMPS-style Tersoff potential file)."
            )
        from ase.calculators.tersoff import Tersoff

        atoms.calc = Tersoff.from_lammps(force_field_param_file)
        return atoms

    if force_field_type == "morse":
        from ase.calculators.morse import MorsePotential

        atoms.calc = MorsePotential()
        return atoms

    if force_field_type == "lj":
        from ase.calculators.lj import LennardJones

        # Generic LJ fallback. sigma=1.5 Å places the minimum near typical
        # Si-O distances; epsilon=0.1 eV gives moderate force magnitudes.
        atoms.calc = LennardJones(sigma=1.5, epsilon=0.1, rc=4.5, smooth=True)
        return atoms

    if force_field_type == "emt":
        from ase.calculators.emt import EMT

        atoms.calc = EMT()
        atoms.get_potential_energy()  # raises if unsupported element
        return atoms

    # auto: try EMT, fall back to generic LJ
    if force_field_type != "auto":
        raise ValueError(f"Unknown force_field_type: {force_field_type}")

    try:
        from ase.calculators.emt import EMT

        atoms.calc = EMT()
        atoms.get_potential_energy()
        return atoms
    except Exception:
        from ase.calculators.lj import LennardJones

        atoms.calc = LennardJones(sigma=1.5, epsilon=0.1, rc=4.5, smooth=True)
        return atoms
