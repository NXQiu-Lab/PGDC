"""Utilities for amorphous carbon dataset construction.

Provides:
  - Initial amorphous carbon configuration generation (melt-quench, density grid)
  - Candidate perturbation with geometry-aware screening
  - CP2K DFT AIMD input generation for carbon (semi-metal / small-gap)
  - Coordination-number / sp³-fraction / density helpers

Induction modes supported:
  - MD / GEO_OPT
  - ANNEALING (temperature ramp)
  - NPT (pressure-driven sp²→sp³ transition)
  - BALLISTIC (high-energy initial velocities)
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import ase.db
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Element properties for carbon
# ---------------------------------------------------------------------------

ELEMENT_PROPS_C: dict[str, dict[str, str]] = {
    "C": {"potential": "GTH-PBE-q4", "basis": "DZVP-MOLOPT-SR-GTH"},
}

# Covalent radius for carbon (Å), used for CN cutoff and overlap checks
_C_COVALENT_RADIUS: float = 0.76

# Carbon mass (amu)
_C_MASS: float = 12.011

# ---------------------------------------------------------------------------
# 2. Amorphous carbon structure generation
# ---------------------------------------------------------------------------


def get_density(atoms: Atoms) -> float:
    """Return density in g/cm³."""
    volume_ang3 = atoms.get_volume()
    if volume_ang3 <= 0:
        raise ValueError("Invalid cell volume")
    mass_amu = sum(atoms.get_masses())
    # amu/Å³ -> g/cm³
    return (mass_amu / volume_ang3) * 1.66053906660


def set_density(atoms: Atoms, target_density_g_cm3: float, scale_atoms: bool = True) -> Atoms:
    """Scale cell to target density while optionally scaling atomic positions."""
    current_density = get_density(atoms)
    scale = (current_density / target_density_g_cm3) ** (1.0 / 3.0)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=scale_atoms)
    return atoms


def generate_amorphous_carbon_seed(
    n_atoms: int = 64,
    target_density_g_cm3: float = 2.0,
    temperature: float = 300.0,
    seed: int | None = None,
) -> Atoms:
    """Generate an initial amorphous carbon seed structure.

    Strategy:
      1. Build a small diamond supercell (cubic, ordered).
      2. Scale to target density.
      3. Add thermal disorder scaled by sqrt(T/300K).

    This seed is NOT meant to be a realistic amorphous carbon yet; it will be
    melted and quenched in AIMD. The purpose is to provide a chemically
    reasonable starting point with correct density and no severe overlaps.

    Args:
        n_atoms: Number of carbon atoms. Should be a multiple of 8 for diamond
            conventional cell (2×2×2 supercell = 64 atoms).
        target_density_g_cm3: Target density in g/cm³. Amorphous carbon spans
            roughly 1.5–3.0 g/cm³; 2.0 g/cm³ is a good intermediate seed.
        temperature: Target temperature (K) used to scale initial disorder.
        seed: Random seed for reproducibility.

    Returns:
        ASE Atoms object with periodic boundary conditions.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        random.seed(seed)
    else:
        rng = np.random.default_rng()

    # Build diamond conventional cell (8 atoms) and replicate to target size
    atoms = bulk("C", "diamond", a=3.567, cubic=True)
    n_prim = len(atoms)
    repeat = max(1, int(round((n_atoms / n_prim) ** (1.0 / 3.0))))
    atoms = atoms * (repeat, repeat, repeat)

    # Scale to target density
    atoms = set_density(atoms, target_density_g_cm3)

    # Add thermal disorder
    if temperature > 0:
        rms = 0.02 * (temperature / 300.0) ** 0.5
        if rms > 0.15:
            rms = 0.15
        pos = atoms.get_positions()
        noise = rng.normal(0, rms, pos.shape)
        atoms.set_positions(pos + noise)
        atoms.wrap()

    atoms.info["target_density"] = float(target_density_g_cm3)
    atoms.info["temperature"] = float(temperature)
    atoms.info["source"] = "diamond_seed_perturbed"
    return atoms


def generate_liquid_carbon_seed(
    n_atoms: int = 64,
    target_density_g_cm3: float = 2.0,
    seed: int | None = None,
) -> Atoms:
    """Generate a random initial structure at target density for melt-quench.

    Uses a simple random placement with minimum distance constraint. This is
    intentionally more disordered than generate_amorphous_carbon_seed and is
    meant to be melted at high temperature before quenching.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Estimate cubic cell length from density
    volume = (n_atoms * _C_MASS) / (target_density_g_cm3 / 1.66053906660)
    L = volume ** (1.0 / 3.0)
    cell = np.eye(3) * L

    positions: list[np.ndarray] = []
    min_dist = 1.2  # Å, slightly larger than C-C bond to avoid huge forces
    max_trials = 10000

    for _ in range(n_atoms):
        trial = 0
        while trial < max_trials:
            pos = rng.random(3) * L
            if all(np.linalg.norm(pos - p) > min_dist for p in positions):
                positions.append(pos)
                break
            trial += 1
        else:
            raise RuntimeError(
                f"Failed to place {n_atoms} carbon atoms with min distance {min_dist} Å "
                f"at density {target_density_g_cm3} g/cm³. Try lower density or larger cell."
            )

    atoms = Atoms(
        symbols=["C"] * n_atoms,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    atoms.info["target_density"] = float(target_density_g_cm3)
    atoms.info["source"] = "liquid_random_seed"
    return atoms


# ---------------------------------------------------------------------------
# 3. Candidate perturbation and geometry screening
# ---------------------------------------------------------------------------


def compute_coordination_numbers(
    atoms: Atoms,
    cutoff_factor: float = 1.3,
) -> np.ndarray:
    """Compute coordination number for each atom using covalent-radius cutoff.

    Args:
        atoms: ASE Atoms object.
        cutoff_factor: CN cutoff = sum of covalent radii × factor.

    Returns:
        Array of integer coordination numbers.
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    natoms = len(atoms)

    # Build neighbor list via ASE if available, otherwise simple O(N²)
    try:
        from ase.neighborlist import neighbor_list

        i, j, d = neighbor_list("ijd", atoms, cutoff=(_C_COVALENT_RADIUS * 2 * cutoff_factor))
        cn = np.zeros(natoms, dtype=int)
        for jj in j:
            cn[jj] += 1
        return cn
    except Exception:
        # Fallback O(N²) without PBC
        cn = np.zeros(natoms, dtype=int)
        cutoff = (_C_COVALENT_RADIUS * 2) * cutoff_factor
        for i in range(natoms):
            for j in range(i + 1, natoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= cutoff:
                    cn[i] += 1
                    cn[j] += 1
        return cn


def get_sp3_fraction(atoms: Atoms, cn_cutoff_factor: float = 1.3) -> float:
    """Estimate sp³ fraction as fraction of atoms with CN == 4.

    This is a crude proxy. More accurate approaches (e.g., based on bond angles
    or common-neighbor analysis) can be added later.
    """
    cn = compute_coordination_numbers(atoms, cutoff_factor=cn_cutoff_factor)
    return float(np.sum(cn == 4) / len(atoms))


def get_min_distance(atoms: Atoms) -> float:
    """Return minimum interatomic distance (Å)."""
    try:
        from ase.neighborlist import neighbor_list

        d = neighbor_list("d", atoms, cutoff=3.0)
        return float(np.min(d)) if len(d) > 0 else float("inf")
    except Exception:
        positions = atoms.get_positions()
        natoms = len(atoms)
        min_dist = float("inf")
        for i in range(natoms):
            for j in range(i + 1, natoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_dist:
                    min_dist = dist
        return min_dist


def check_carbon_candidate(
    atoms: Atoms,
    density_min: float = 1.2,
    density_max: float = 3.5,
    cn_min: int = 2,
    cn_max: int = 5,
    min_dist_threshold: float = 1.0,
) -> tuple[bool, str]:
    """Lightweight geometry check for carbon candidates before AIMD.

    This is NOT a DFT-based classification. It only rejects physically absurd
    or locally unstable structures (LU/AB equivalents). The real EQ/MS/TP/TD
    classification is performed on DFT frames after AIMD.

    Returns:
        (passed, reason) tuple.
    """
    if atoms.cell is None or atoms.cell.rank != 3:
        return False, "No valid unit cell"

    # Density check
    density = get_density(atoms)
    if density < density_min or density > density_max:
        return False, f"Density {density:.2f} g/cm³ outside [{density_min}, {density_max}]"

    # Minimum distance check
    min_dist = get_min_distance(atoms)
    if min_dist < min_dist_threshold:
        return False, f"Minimum distance {min_dist:.2f} Å < {min_dist_threshold} Å"

    # Coordination number check
    cn = compute_coordination_numbers(atoms)
    if np.any(cn < cn_min) or np.any(cn > cn_max):
        bad = np.where((cn < cn_min) | (cn > cn_max))[0]
        return False, f"{len(bad)} atoms have CN outside [{cn_min}, {cn_max}]"

    return True, "Passed geometry check"


def perturb_amorphous_carbon(
    atoms: Atoms,
    position_std: float = 0.15,
    strain_range: tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> Atoms:
    """Generate a perturbed candidate from an existing amorphous carbon structure.

    Args:
        atoms: Parent ASE Atoms object.
        position_std: RMS position displacement in Å.
        strain_range: If given, apply a random strain in each cell direction
            sampled uniformly from [-s, +s] for each component. Use small
            values (0.03–0.10) to stay near the parent basin.
        seed: Random seed.

    Returns:
        New ASE Atoms object with perturbations applied.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    new_atoms = atoms.copy()

    # Position perturbation
    pos = new_atoms.get_positions()
    noise = rng.normal(0, position_std, pos.shape)
    new_atoms.set_positions(pos + noise)
    new_atoms.wrap()

    # Optional cell strain
    if strain_range is not None:
        sx, sy, sz = strain_range
        strain_matrix = np.eye(3)
        strain_matrix[0, 0] += rng.uniform(-abs(sx), abs(sx))
        strain_matrix[1, 1] += rng.uniform(-abs(sy), abs(sy))
        strain_matrix[2, 2] += rng.uniform(-abs(sz), abs(sz))
        new_atoms.set_cell(strain_matrix @ new_atoms.get_cell(), scale_atoms=True)

    new_atoms.info["parent_source"] = atoms.info.get("source", "unknown")
    new_atoms.info["perturbation_std"] = float(position_std)
    return new_atoms


# ---------------------------------------------------------------------------
# 4. CP2K DFT AIMD input generation for carbon
# ---------------------------------------------------------------------------

CP2K_ENERGY_FORCE_MOTION_C = """
&MOTION
  &MD
    ENSEMBLE NVE
    STEPS 0
    TIMESTEP 1.0
  &END MD
&END MOTION
"""


CP2K_DFT_COMMON_C = """&GLOBAL
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
      ADDED_MOS {added_mos}
      &SMEAR ON
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE [K] {electronic_temperature}
      &END SMEAR
{scf_method_block}    &END SCF
    &XC
      &XC_FUNCTIONAL {xc_functional}
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
{stress_tensor_block}  &SUBSYS
    &CELL
      A {cell_ax:.6f} {cell_ay:.6f} {cell_az:.6f}
      B {cell_bx:.6f} {cell_by:.6f} {cell_bz:.6f}
      C {cell_cx:.6f} {cell_cy:.6f} {cell_cz:.6f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{kinds}
{velocity_block}    &TOPOLOGY
      CONNECTIVITY OFF
    &END TOPOLOGY
  &END SUBSYS
&END FORCE_EVAL
"""

CP2K_MD_MOTION_C = """
&MOTION
  &MD
    ENSEMBLE NVT
    STEPS {n_steps}
    TIMESTEP {timestep}
    TEMPERATURE {temperature}
    &THERMOSTAT
      TYPE CSVR
      &CSVR
        TIMECON {thermostat_timecon}
      &END CSVR
    &END THERMOSTAT
  &END MD
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

CP2K_GEO_OPT_MOTION_C = """
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

CP2K_ANNEALING_MOTION_C = """
&MOTION
  &MD
    ENSEMBLE LANGEVIN
    STEPS {n_steps}
    TIMESTEP {timestep}
    TEMPERATURE {t_start}
    TEMPERATURE_ANNEALING {temperature_annealing_factor}
    &LANGEVIN
      GAMMA {langevin_gamma}
    &END LANGEVIN
  &END MD
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

CP2K_NPT_MOTION_C = """
&MOTION
  &MD
    ENSEMBLE {ensemble}
    STEPS {n_steps}
    TIMESTEP {timestep}
    TEMPERATURE {temperature}
    &BAROSTAT
      PRESSURE {pressure}
      TIMECON {barostat_timecon}
    &END BAROSTAT
    &THERMOSTAT
      TYPE CSVR
      &CSVR
        TIMECON {thermostat_timecon}
      &END CSVR
    &END THERMOSTAT
  &END MD
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
    &STRESS ON
      &EACH
        MD {output_stride}
      &END EACH
    &END STRESS
    &RESTART
      &EACH
        MD 1000
      &END EACH
    &END RESTART
  &END PRINT
&END MOTION
"""

CP2K_BALLISTIC_MOTION_C = """
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


def build_cp2k_input_carbon(
    atoms: Atoms,
    project_name: str = "amorphous_carbon",
    temperature: float = 300.0,
    n_steps: int = 20000,
    timestep: float = 1.0,
    output_stride: int = 40,
    cutoff: int = 500,
    rel_cutoff: int = 50,
    xc_functional: str = "PBE",
    basis_set: str = "DZVP-MOLOPT-SR-GTH",
    scf_guess: str = "ATOMIC",
    electronic_temperature: float = 1000.0,
    added_mos: int = 100,
    use_ot: bool = False,
    wfn_restart_file: str | None = None,
    ext_restart_file: str | None = None,
    thermostat_timecon: float = 50.0,
    langevin_gamma: float = 0.002,
    cp2k_data_dir: str | None = None,
    run_type: str = "MD",
    # Annealing-specific
    annealing_start_temp: float | None = None,
    annealing_stop_temp: float | None = None,
    # NPT-specific
    ensemble: str = "NVT",
    pressure_gpa: float | None = None,
    barostat_timecon: float = 1000.0,
    # Ballistic-specific
    ballistic_velocity_temp: float | None = None,
    ballistic_ensemble: str = "NVE",
) -> str:
    """Generate a CP2K DFT input file for amorphous carbon.

    Carbon can be metallic (sp²-rich) or semiconducting (sp³-rich) depending on
    density. Therefore diagonalization + Fermi-Dirac smearing is used by default.
    Set ``use_ot=True`` only if you know the system has a finite gap.

    Supported run_type values:
      - "MD": standard NVT molecular dynamics
      - "GEO_OPT": geometry optimization
      - "ANNEALING": Langevin MD with exponential temperature annealing
      - "NPT": NPT_I or NPT_F with barostat
      - "BALLISTIC": MD with high-energy initial velocities injected
      - "ENERGY_FORCE": single-point energy, forces and stress tensor

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
        electronic_temperature: Electronic smearing temperature (K).
        added_mos: Additional empty MOs for smearing.
        use_ot: Use OT SCF solver (default False for carbon).
        wfn_restart_file: WFN restart file for &DFT block.
        ext_restart_file: EXT_RESTART file for geometry/velocities.
        thermostat_timecon: CSVR thermostat time constant (fs).
        langevin_gamma: Langevin friction coefficient (fs^-1), used for annealing.
        cp2k_data_dir: CP2K data directory. If None, inferred from env.
        run_type: "MD", "GEO_OPT", "ANNEALING", "NPT", "BALLISTIC", or
        "ENERGY_FORCE".
        annealing_start_temp: Starting temperature for annealing ramp (K).
        annealing_stop_temp: Ending temperature for annealing ramp (K).
        ensemble: MD ensemble (NVT, NPT_I, NPT_F, NVE).
        pressure_gpa: Target pressure for NPT (GPa); converted to bar internally.
        barostat_timecon: Barostat time constant (fs).
        ballistic_velocity_temp: Temperature for initial velocity generation (K).
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
        props = ELEMENT_PROPS_C.get(elem, {"potential": f"GTH-{xc_functional}", "basis": basis_set})
        kind_lines.append(
            f"    &KIND {elem}\n"
            f"      BASIS_SET {props['basis']}\n"
            f"      POTENTIAL {props['potential']}\n"
            f"    &END KIND"
        )
    kinds_str = "\n".join(kind_lines)

    # SCF settings by run type
    if run_type == "GEO_OPT":
        eps_scf = "1.0E-6"
        max_scf = "200"
    else:
        eps_scf = "1.0E-5"
        max_scf = "300"

    # SCF method block
    if use_ot:
        if electronic_temperature and electronic_temperature > 0:
            raise ValueError("CP2K OT is incompatible with FERMI_DIRAC smearing. "
                             "Set use_ot=False or electronic_temperature=0.")
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

    # Motion block selection
    if run_type == "ENERGY_FORCE":
        # CP2K's ENERGY_FORCE run type does not evaluate the stress tensor.
        # Use a zero-step NVE MD run type instead; this performs a single-point
        # energy/force/stress evaluation at the input geometry.
        motion_block = CP2K_ENERGY_FORCE_MOTION_C
    elif run_type == "GEO_OPT":
        motion_block = CP2K_GEO_OPT_MOTION_C
    elif run_type == "ANNEALING":
        t_start = annealing_start_temp if annealing_start_temp is not None else temperature
        t_stop = annealing_stop_temp if annealing_stop_temp is not None else temperature
        if n_steps <= 0:
            raise ValueError("n_steps must be positive for annealing")
        # CP2K's TEMPERATURE_ANNEALING multiplies the target temperature each step.
        # T(n) = T_start * factor^n; set T(n_steps) = T_stop.
        temperature_annealing_factor = (t_stop / t_start) ** (1.0 / n_steps)
        motion_block = CP2K_ANNEALING_MOTION_C.format(
            n_steps=n_steps,
            timestep=timestep,
            output_stride=output_stride,
            t_start=t_start,
            temperature_annealing_factor=temperature_annealing_factor,
            langevin_gamma=langevin_gamma,
        )
    elif run_type == "NPT":
        if pressure_gpa is None:
            raise ValueError("pressure_gpa is required for run_type='NPT'")
        pt = pressure_gpa
        pressure_bar = pt * 10000.0  # GPa -> bar
        motion_block = CP2K_NPT_MOTION_C.format(
            ensemble=ensemble if ensemble.startswith("NPT") else "NPT_I",
            n_steps=n_steps,
            timestep=timestep,
            temperature=temperature,
            pressure=pressure_bar,
            barostat_timecon=barostat_timecon,
            thermostat_timecon=thermostat_timecon,
            output_stride=output_stride,
        )
    elif run_type == "BALLISTIC":
        if ballistic_ensemble == "NVE":
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
        motion_block = CP2K_BALLISTIC_MOTION_C.format(
            ensemble=ballistic_ensemble,
            n_steps=n_steps,
            timestep=timestep,
            temperature=temperature,
            thermostat_timecon=thermostat_timecon,
            output_stride=output_stride,
            thermostat_block=thermostat_block,
        )
    else:
        # Standard MD (NVT)
        motion_block = CP2K_MD_MOTION_C.format(
            n_steps=n_steps,
            timestep=timestep,
            temperature=temperature,
            thermostat_timecon=thermostat_timecon,
            output_stride=output_stride,
        )

    # Velocity block for ballistic mode
    velocity_block = ""
    if run_type == "BALLISTIC" and ballistic_velocity_temp is not None:
        temp_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(temp_atoms, temperature_K=ballistic_velocity_temp)
        velocities = temp_atoms.get_velocities()
        # CP2K &VELOCITY expects atomic units: bohr/au_t
        _ANG_FS_TO_BOHR_AUT = 0.02418884 / 0.529177  # ≈ 0.04572
        velocities_bohr_aut = velocities * _ANG_FS_TO_BOHR_AUT
        vel_lines = ["    &VELOCITY"]
        for v in velocities_bohr_aut:
            vel_lines.append(f"      {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}")
        vel_lines.append("    &END VELOCITY")
        velocity_block = "\n".join(vel_lines) + "\n"

    common_block = CP2K_DFT_COMMON_C.format(
        project_name=project_name,
        run_type="MD" if run_type in ("ANNEALING", "NPT", "BALLISTIC", "ENERGY_FORCE") else run_type,
        basis_set_path=basis_set_path,
        potential_path=potential_path,
        cutoff=cutoff,
        rel_cutoff=rel_cutoff,
        scf_guess=scf_guess,
        eps_scf=eps_scf,
        max_scf=max_scf,
        added_mos=added_mos,
        electronic_temperature=electronic_temperature,
        wfn_restart_line=wfn_restart_line,
        scf_method_block=scf_method_block,
        xc_functional=xc_functional,
        stress_tensor_block=(
            "  STRESS_TENSOR ANALYTICAL\n"
            "  &PRINT\n"
            "    &STRESS_TENSOR ON\n"
            "    &END STRESS_TENSOR\n"
            # ENERGY_FORCE single points also need atomic forces in the stdout
            # (the zero-step MD motion block does not print them there).
            # FORCE_UNIT makes the unit explicit so parsers can rely on it.
            + (
                "    &FORCES ON\n"
                "      FORCE_UNIT eV/angstrom\n"
                "      NDIGITS 8\n"
                "    &END FORCES\n"
                if run_type == "ENERGY_FORCE" else ""
            )
            + "  &END PRINT\n"
        ) if run_type in ("NPT", "ENERGY_FORCE") else "",
        coords=coords_str,
        kinds=kinds_str,
        velocity_block=velocity_block,
        cell_ax=cell_ax, cell_ay=cell_ay, cell_az=cell_az,
        cell_bx=cell_bx, cell_by=cell_by, cell_bz=cell_bz,
        cell_cx=cell_cx, cell_cy=cell_cy, cell_cz=cell_cz,
    )

    if ext_restart_file:
        restart_block = f"""&EXT_RESTART
  RESTART_FILE_NAME {ext_restart_file}
&END EXT_RESTART
"""
        common_block = restart_block + "\n" + common_block

    return common_block + motion_block


# ---------------------------------------------------------------------------
# 5. Convenience helpers for profile / analysis
# ---------------------------------------------------------------------------


def summarize_carbon_structure(atoms: Atoms) -> dict[str, Any]:
    """Return a dict of key structural descriptors for logging/profile."""
    density = get_density(atoms)
    cn = compute_coordination_numbers(atoms)
    sp3_frac = get_sp3_fraction(atoms)
    return {
        "n_atoms": len(atoms),
        "density_g_cm3": round(density, 3),
        "sp3_fraction": round(sp3_frac, 3),
        "mean_cn": round(float(np.mean(cn)), 2),
        "min_distance_A": round(get_min_distance(atoms), 3),
    }


# ---------------------------------------------------------------------------
# 6. Density-adaptive k-center sampling for final 10k dataset
# ---------------------------------------------------------------------------


def _lowdim_fingerprint_carbon(atoms: Atoms) -> np.ndarray:
    """Lightweight fingerprint for amorphous carbon sparse selection.

    Combines:
      * radial distribution function (RDF) histogram (16 bins)
      * density
      * sp3 fraction
      * energy and force statistics (if available)
    """
    positions = atoms.get_positions()
    n_atoms = len(atoms)
    features: list[float] = []

    # RDF histogram (16 bins, cutoff 6.0 Å)
    r_cut = 6.0
    n_bins = 16
    rdf = np.zeros(n_bins, dtype=np.float64)
    if n_atoms > 1:
        diff = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        triu_idx = np.triu_indices(n_atoms, k=1)
        pair_dists = dists[triu_idx]
        pair_dists = pair_dists[pair_dists < r_cut]
        if pair_dists.size:
            bin_idx = (pair_dists / r_cut * n_bins).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            np.add.at(rdf, bin_idx, 1.0)
    pair_count = max(n_atoms * (n_atoms - 1) / 2.0, 1.0)
    features.extend((rdf / pair_count).tolist())

    # Density and sp3 fraction
    features.append(get_density(atoms))
    features.append(get_sp3_fraction(atoms))

    # Energy / force statistics
    energy = float(atoms.calc.results.get("energy", 0.0)) if atoms.calc else 0.0
    forces = atoms.calc.results.get("forces", None) if atoms.calc else None
    if forces is not None:
        force_norms = np.linalg.norm(forces, axis=1)
        f_mean = float(np.mean(force_norms))
        f_max = float(np.max(force_norms))
    else:
        f_mean = 0.0
        f_max = 0.0
    features.extend([energy / max(n_atoms, 1), f_mean, f_max])

    return np.asarray(features, dtype=np.float64)


def _k_center_greedy(matrix: np.ndarray, k: int) -> list[int]:
    """Return indices of k structures selected by k-center greedy algorithm."""
    n = matrix.shape[0]
    if k >= n:
        return list(range(n))

    full_dist = cdist(matrix, matrix, metric="euclidean")
    mean_dists = (full_dist.sum(axis=1) - np.diag(full_dist)) / max(1, n - 1)
    selected = [int(np.argmin(mean_dists))]

    min_dist_to_selected = full_dist[selected[0]].copy()
    min_dist_to_selected[selected[0]] = -1.0

    for _ in range(1, k):
        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)
        min_dist_to_selected[next_idx] = -1.0
        new_dists = np.linalg.norm(matrix - matrix[next_idx], axis=1)
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_dists)

    return selected


def _density_bin_label(density: float, bin_edges: list[float]) -> str:
    """Return density bin label for a given density."""
    for i in range(len(bin_edges) - 1):
        if density < bin_edges[i + 1]:
            return f"density_{bin_edges[i]:.1f}_{bin_edges[i + 1]:.1f}"
    return f"density_>={bin_edges[-1]:.1f}"


def density_adaptive_kcenter_sample(
    src_db_path: str | Path,
    dst_db_path: str | Path,
    n_select: int,
    density_bin_edges: tuple[float, ...] = (1.2, 1.8, 2.4, 3.0, 3.5),
    density_k: int = 10,
) -> dict[str, Any]:
    """Build a sparsified amorphous carbon dataset using density-adaptive k-center.

    Args:
        src_db_path: Source ASE DB path.
        dst_db_path: Output ASE DB path.
        n_select: Target number of structures.
        density_bin_edges: Edges for density stratification (g/cm³).
        density_k: Number of nearest neighbors used for per-bin density estimation.

    Returns:
        Dict with source count, selected count, and bin quotas.
    """
    src_db = ase.db.connect(str(src_db_path))
    dst_db = ase.db.connect(str(dst_db_path))

    rows = list(src_db.select())
    if not rows:
        raise ValueError(f"No structures found in {src_db_path}")

    logger.info("Loaded %d structures from %s", len(rows), src_db_path)

    # Compute fingerprints
    fingerprints: list[np.ndarray] = []
    densities: list[float] = []
    for row in rows:
        atoms = row.toatoms()
        fingerprints.append(_lowdim_fingerprint_carbon(atoms))
        densities.append(get_density(atoms))

    matrix = np.asarray(fingerprints, dtype=np.float64)
    # Z-score normalize
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds

    # Assign density bins
    bin_labels = [_density_bin_label(d, list(density_bin_edges)) for d in densities]
    unique_bins = sorted(set(bin_labels))

    # Compute per-bin density scores (median k-NN distance; larger = more dispersed)
    bin_indices: dict[str, list[int]] = {b: [] for b in unique_bins}
    for i, b in enumerate(bin_labels):
        bin_indices[b].append(i)

    bin_scores: dict[str, float] = {}
    for b, indices in bin_indices.items():
        n = len(indices)
        if n <= 1:
            bin_scores[b] = 1.0
            continue
        k_eff = min(density_k, n - 1)
        dists = cdist(matrix[indices], matrix[indices], metric="euclidean")
        sorted_dists = np.sort(dists, axis=1)[:, 1 : k_eff + 1]
        bin_scores[b] = float(np.median(sorted_dists))

    total_score = sum(bin_scores.values()) or 1.0

    # Allocate quotas proportional to density score, capped by bin size
    raw_quotas = {b: n_select * bin_scores[b] / total_score for b in unique_bins}
    quotas: dict[str, int] = {}
    remaining = n_select
    for b in unique_bins:
        q = min(int(raw_quotas[b]), len(bin_indices[b]))
        quotas[b] = q
        remaining -= q

    # Distribute leftover to bins with largest fractional residuals.
    # Loop until all remaining slots are assigned or no bin has capacity.
    while remaining > 0:
        residuals = {
            b: raw_quotas[b] - quotas[b]
            for b in unique_bins
            if quotas[b] < len(bin_indices[b])
        }
        if not residuals:
            break
        for b in sorted(residuals, key=residuals.get, reverse=True):
            if remaining <= 0:
                break
            if quotas[b] < len(bin_indices[b]):
                quotas[b] += 1
                remaining -= 1

    logger.info("Density-adaptive quotas: %s", quotas)

    # Select within each bin using k-center greedy
    selected_global_indices: list[int] = []
    for b in unique_bins:
        indices = bin_indices[b]
        q = min(quotas[b], len(indices))
        if q <= 0:
            continue
        local_selected = _k_center_greedy(matrix[indices], q)
        selected_global_indices.extend(indices[i] for i in local_selected)

    # Write selected structures
    for idx in selected_global_indices:
        row = rows[idx]
        dst_db.write(row.toatoms(), **row.key_value_pairs)

    logger.info(
        "Wrote %d structures to %s",
        len(selected_global_indices),
        dst_db_path,
    )

    return {
        "n_source": len(rows),
        "n_selected": len(selected_global_indices),
        "bin_quotas": quotas,
        "bin_scores": bin_scores,
    }


__all__ = [
    "ELEMENT_PROPS_C",
    "get_density",
    "set_density",
    "generate_amorphous_carbon_seed",
    "generate_liquid_carbon_seed",
    "compute_coordination_numbers",
    "get_sp3_fraction",
    "get_min_distance",
    "check_carbon_candidate",
    "perturb_amorphous_carbon",
    "build_cp2k_input_carbon",
    "summarize_carbon_structure",
    "density_adaptive_kcenter_sample",
]
