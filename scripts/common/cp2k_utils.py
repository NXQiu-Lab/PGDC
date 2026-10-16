"""Generic CP2K output parsers and ASE database helpers.

Previously lived in ``scripts/hea/hea_utils.py``; extracted here so that
``scripts/hea`` can be removed while keeping the utilities required by
``scripts/sio2``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io import read

logger = logging.getLogger(__name__)

# CP2K writes energies in Hartree and forces in Hartree/bohr; this project
# standardises on eV and eV/Angstrom. All parsers below convert at read time.
HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = 51.42208619083232


def _cellpar_to_matrix(
    cell: tuple[list[float], list[float]],
) -> np.ndarray:
    """Convert cell lengths/angles to a Cartesian cell matrix."""
    lengths, angles = cell
    a, b, c = (float(x) for x in lengths)
    alpha, beta, gamma = (np.deg2rad(float(x)) for x in angles)
    sin_gamma = np.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("Invalid cell: gamma angle is too close to 0 or 180 degrees")

    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0] = [a, 0.0, 0.0]
    matrix[1] = [b * np.cos(gamma), b * sin_gamma, 0.0]
    matrix[2, 0] = c * np.cos(beta)
    matrix[2, 1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    matrix[2, 2] = np.sqrt(max(0.0, c * c - matrix[2, 0] ** 2 - matrix[2, 1] ** 2))
    return matrix


def parse_cp2k_ener(ener_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse CP2K .ener file.

    Energy columns (``Kin.``, ``Pot.``, ``Cons Qty.``) are printed by CP2K in
    Hartree (``[a.u.]``); they are converted to eV here. ``pressure`` is kept
    in bar as printed by CP2K MD.

    Returns list of dicts with keys:
        step, time_fs, kinetic_energy (eV), temperature (K),
        potential_energy (eV), total_energy (eV), conserved_quantity (eV),
        pressure (bar, when present)
    """
    ener_path = Path(ener_path)
    if not ener_path.exists():
        return []

    records = []
    with ener_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                record = {
                    "step": int(parts[0]),
                    "time_fs": float(parts[1]),
                    "kinetic_energy": float(parts[2]) * HA_TO_EV,
                    "temperature": float(parts[3]),
                    "potential_energy": float(parts[4]) * HA_TO_EV,
                    "total_energy": float(parts[5]) * HA_TO_EV,
                    "conserved_quantity": float(parts[6]) * HA_TO_EV,
                }
                if len(parts) >= 8:
                    record["pressure"] = float(parts[7])
                records.append(record)
            except (ValueError, IndexError):
                continue
    return records


def parse_cp2k_cell(out_path: str | Path) -> tuple[list[float], list[float]] | None:
    """Parse cell lengths [ang] and angles [deg] from CP2K stdout.

    Returns (lengths, angles) or None if not found.
    """
    import re

    text = Path(out_path).read_text()
    m_len = re.search(
        r"Cell lengths \[ang\]\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
        text,
    )
    m_ang = re.search(
        r"Cell angles \[deg\]\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
        text,
    )
    if not m_len or not m_ang:
        return None
    lengths = [float(x) for x in m_len.groups()]
    angles = [float(x) for x in m_ang.groups()]
    return lengths, angles


def parse_cp2k_md_cells(
    out_path: str | Path,
) -> dict[int, tuple[list[float], list[float]]]:
    """Parse the cell printed for each MD step from CP2K stdout.

    Standard CP2K XYZ trajectories do not carry lattice vectors.  NPT runs do,
    however, print the evolving cell in ``out`` after every MD step.  The
    returned mapping is keyed by the CP2K MD step, including step 0 when the
    initialization cell is available.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return {}

    text = out_path.read_text()
    angle_match = re.search(
        r"Cell angles \[deg\]\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
        text,
    )
    angles = [float(x) for x in angle_match.groups()] if angle_match else [90.0] * 3

    cells: dict[int, tuple[list[float], list[float]]] = {}
    current_step: int | None = None
    step_pattern = re.compile(r"^\s*MD\|\s+Step number\s+(\d+)", re.MULTILINE)
    length_pattern = re.compile(
        r"^\s*(?:MD_INI|MD)\|\s+Cell lengths \[ang\]\s+"
        r"([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
        re.MULTILINE,
    )

    for line in text.splitlines():
        step_match = step_pattern.match(line)
        if step_match:
            current_step = int(step_match.group(1))
            continue

        if line.lstrip().startswith("MD_INI|"):
            current_step = 0

        length_match = length_pattern.match(line)
        if length_match and current_step is not None:
            lengths = [float(x) for x in length_match.groups()]
            cells[current_step] = (lengths, list(angles))

    return cells


def parse_cp2k_stress(out_path: str | Path) -> np.ndarray | None:
    """
    Parse the stress tensor from a CP2K stdout.

    CP2K can print the stress tensor in several formats. The two most common
    are::

        STRESS TENSOR [GPa]
        X Y Z
        X -8233.84728056  8.05482610  0.90481573
        ...

    and::

        STRESS| Analytical stress tensor [bar]
        STRESS|                        x                   y                   z
        STRESS|      x       -4.37660932369E+05   8.08366623472E+04  -1.63488674490E+05
        ...

    The sign convention used by CP2K matches ASE (positive = tensile stress),
    so the returned tensor can be stored directly in ``row.data['stress']``
    after unit conversion.

    Args:
        out_path: Path to CP2K output file.

    Returns:
        Stress tensor as ndarray of shape (3, 3) in eV/Å³, or None if the
        stress block is not found or the output is not converged.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return None

    text = out_path.read_text()

    # Use the last occurrence of a stress-tensor header so that we pick up the
    # final result rather than an intermediate print.  We anchor the newer
    # CP2K format to the start of the "STRESS|" line so that we do not
    # accidentally match the trailing "Eigenvectors and eigenvalues of the
    # analytical stress tensor" block.
    header_pattern = re.compile(
        r"^\s*STRESS\s*\|\s*Analytical stress tensor\s*\[([^\]]+)\]",
        re.IGNORECASE | re.MULTILINE,
    )
    match = None
    for m in header_pattern.finditer(text):
        match = m

    if match is None:
        # Fallback to the older CP2K format without the "STRESS|" prefix.
        for m in re.finditer(r"^\s*STRESS TENSOR\s*\[([^\]]+)\]", text, re.IGNORECASE | re.MULTILINE):
            match = m

    if match is None:
        return None

    unit_str = match.group(1).strip().lower()
    start = match.end()
    lines = text[start:].splitlines()

    data_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Remove the leading "STRESS|" token if present.
        if stripped.upper().startswith("STRESS|"):
            stripped = stripped[7:].strip()
        parts = stripped.split()
        # Skip the "x y z" header line.
        if len(parts) == 3 and all(p.upper() in {"X", "Y", "Z"} for p in parts):
            continue
        # Data line: label + 3 numbers.
        if len(parts) >= 4 and parts[0].upper() in {"X", "Y", "Z"}:
            data_lines.append(stripped)
            if len(data_lines) == 3:
                break

    if len(data_lines) != 3:
        return None

    try:
        stress = np.array(
            [[float(p) for p in line.split()[1:4]] for line in data_lines],
            dtype=np.float64,
        )
    except (ValueError, IndexError):
        return None

    # Convert to eV/Å³. 1 eV/Å³ = 160.2176567 GPa.
    GPa_TO_EV_PER_ANGSTROM3 = 1.0 / 160.2176567
    if unit_str in ("gpa",):
        factor = GPa_TO_EV_PER_ANGSTROM3
    elif unit_str in ("bar", "bars"):
        factor = 1e-4 * GPa_TO_EV_PER_ANGSTROM3  # 1 bar = 1e-4 GPa
    elif unit_str in ("au", "a.u.", "atomic"):
        # CP2K atomic unit of stress is Hartree/Bohr^3
        # 1 Ha/Bohr^3 = 43529.744 GPa
        factor = 43529.744 * GPa_TO_EV_PER_ANGSTROM3
    else:
        # Default to GPa, which is the CP2K standard for this print block.
        factor = GPa_TO_EV_PER_ANGSTROM3

    return stress * factor


def parse_cp2k_md_stress(out_path: str | Path) -> dict[int, np.ndarray]:
    """Parse the per-MD-step analytical stress tensors from a CP2K stdout.

    With ``STRESS_TENSOR ANALYTICAL`` and the ``&PRINT/&STRESS_TENSOR``
    section enabled, CP2K prints one ``STRESS| Analytical stress tensor``
    block per MD step.  The block for step N appears right before the
    ``MD| Step number N`` summary line; the very first block (printed during
    initialisation) belongs to step 0.  Because the output is strictly
    sequential, the k-th block (0-indexed) corresponds to MD step k.

    The sign convention used by CP2K matches ASE (positive = tensile stress).

    Args:
        out_path: Path to CP2K output file.

    Returns:
        Mapping from MD step to a (3, 3) stress tensor in eV/Å³.
        Empty dict if no stress blocks are found.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return {}

    header_re = re.compile(
        r"^\s*STRESS\s*\|\s*Analytical stress tensor\s*\[([^\]]+)\]",
        re.IGNORECASE,
    )
    banner_re = re.compile(r"^\s*MD\|\s+Step number\s+(\d+)")

    lines = out_path.read_text().splitlines()

    GPa_TO_EV_PER_ANGSTROM3 = 1.0 / 160.2176567

    stresses: dict[int, np.ndarray] = {}
    max_step = -1
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1

        banner = banner_re.match(line)
        if banner:
            max_step = max(max_step, int(banner.group(1)))
            continue

        header = header_re.match(line)
        if header is None:
            continue

        step = len(stresses)  # block ordinal == MD step (see docstring)
        unit_str = header.group(1).strip().lower()
        if unit_str in ("gpa",):
            factor = GPa_TO_EV_PER_ANGSTROM3
        elif unit_str in ("bar", "bars"):
            factor = 1e-4 * GPa_TO_EV_PER_ANGSTROM3
        elif unit_str in ("au", "a.u.", "atomic"):
            factor = 43529.744 * GPa_TO_EV_PER_ANGSTROM3
        else:
            factor = GPa_TO_EV_PER_ANGSTROM3

        rows: list[list[float]] = []
        while i < len(lines) and len(rows) < 3:
            stripped = lines[i].strip()
            i += 1
            if not stripped:
                continue
            if stripped.upper().startswith("STRESS|"):
                stripped = stripped[7:].strip()
            parts = stripped.split()
            # Skip the "x y z" column header line.
            if len(parts) == 3 and all(p.upper() in {"X", "Y", "Z"} for p in parts):
                continue
            if len(parts) >= 4 and parts[0].upper() in {"X", "Y", "Z"}:
                try:
                    rows.append([float(p) for p in parts[1:4]])
                except ValueError:
                    break

        if len(rows) != 3:
            logger.warning(
                "Stress block %d in %s is incomplete; skipping it.", step, out_path
            )
            continue
        stresses[step] = np.array(rows, dtype=np.float64) * factor

    if stresses and max_step >= 0 and len(stresses) != max_step + 1:
        logger.warning(
            "Parsed %d stress blocks from %s but MD banners reach step %d; "
            "step alignment may be off.",
            len(stresses),
            out_path,
            max_step,
        )

    return stresses


def parse_cp2k_trajectory(
    pos_xyz_path: str | Path,
    frc_xyz_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Parse CP2K trajectory (positions) and optional forces.

    CP2K writes ``-frc-1.xyz`` force trajectories in Hartree/bohr; forces are
    converted to eV/Angstrom here. Positions remain in Angstrom as printed.

    Returns list of frames, each frame is:
        {
            "n_atoms": int,
            "symbols": list[str],
            "positions": ndarray (n_atoms, 3),  # Angstrom
            "forces": ndarray (n_atoms, 3) or None,  # eV/Angstrom
            "comment": str,
        }
    """
    pos_path = Path(pos_xyz_path)
    if not pos_path.exists():
        return []

    pos_frames = read(str(pos_path), index=":")
    if not isinstance(pos_frames, list):
        pos_frames = [pos_frames]

    frc_frames = []
    if frc_xyz_path is not None:
        frc_path = Path(frc_xyz_path)
        if frc_path.exists():
            frc_frames = read(str(frc_path), index=":")
            if not isinstance(frc_frames, list):
                frc_frames = [frc_frames]

    results = []
    for i, atoms in enumerate(pos_frames):
        frame = {
            "n_atoms": len(atoms),
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions(),
            "forces": None,
            "comment": atoms.info.get("comment", ""),
        }
        step = atoms.info.get("i")
        if step is not None:
            frame["step"] = int(step)
        if i < len(frc_frames):
            frame["forces"] = frc_frames[i].get_positions() * HA_BOHR_TO_EV_ANG
        results.append(frame)

    return results


def merge_trajectory_to_db(
    db_path: str | Path,
    pos_xyz_path: str | Path,
    frc_xyz_path: str | Path | None,
    ener_path: str | Path | None,
    metadata: dict[str, Any] | None = None,
    subsample: int = 1,
    event_driven: bool = False,
    event_window: int = 5,
    sparse_stride: int = 20,
    energy_threshold: float | None = None,
    cell: tuple[list[float], list[float]] | None = None,
    cell_by_step: dict[int, tuple[list[float], list[float]]] | None = None,
) -> int:
    """
    Parse CP2K outputs and append frames to ASE DB.

    The parsers convert CP2K atomic units at read time, so energies (eV) and
    forces (eV/Angstrom) are stored in true eV units. Before anything is
    written, the selected frames pass through the unit audit in
    ``scripts/common/unit_audit.py``; if the audit fails (e.g. values still
    look like Hartree), a RuntimeError is raised and no rows are added.

    Two sampling modes are supported:

    * **Uniform** (default, ``event_driven=False``): keep every ``subsample``-th
      frame — simple and predictable.
    * **Event-driven** (``event_driven=True``): use :class:`TrajectoryMiner` to
      detect transition events (energy jumps, volume changes, coordination changes)
      and apply dense sampling near events / sparse sampling elsewhere.  This
      mode is independent of ``subsample``.

    Args:
        db_path: Path to ASE database (.db).
        pos_xyz_path: Trajectory xyz file.
        frc_xyz_path: Forces xyz file (optional).
        ener_path: Energy file (optional).
        metadata: Extra metadata to store per frame.
        subsample: Keep every Nth frame (uniform mode only).
        event_driven: If True, use TrajectoryMiner for intelligent subsampling.
        event_window: Frames before/after each event to sample densely
            (event-driven mode only).
        sparse_stride: Frame stride for non-event regions
            (event-driven mode only).
        energy_threshold: Energy-jump detection threshold multiplier for
            TrajectoryMiner (None = auto).
        cell: Optional (lengths, angles) tuple in angstrom/degrees.  CP2K
            standard XYZ does not contain lattice vectors; pass the cell
            here to ensure the ASE DB stores periodic boundary info.
        cell_by_step: Optional mapping from CP2K MD step to (lengths, angles).
            When present, this takes precedence over ``cell`` for each frame and
            preserves the changing cell of NPT trajectories.

    Returns:
        Number of frames added.
    """
    from atom_dataset_index.sampling.trajectory_miner import TrajectoryMiner

    db_path = Path(db_path)
    metadata = metadata or {}

    frames = parse_cp2k_trajectory(pos_xyz_path, frc_xyz_path)
    for frame in frames:
        frame_cell = None
        if cell_by_step is not None:
            step = frame.get("step")
            if step is not None:
                frame_cell = cell_by_step.get(int(step))
        if frame_cell is None:
            frame_cell = cell
        if frame_cell is not None:
            frame["cell"] = _cellpar_to_matrix(frame_cell)
    energy_records = []
    if ener_path is not None and Path(ener_path).exists():
        energy_records = parse_cp2k_ener(ener_path)

    # CP2K writes energies every MD step but coordinates only at the
    # trajectory stride. Align records by the step embedded in XYZ comments.
    frame_energy_records = energy_records
    if energy_records and all(frame.get("step") is not None for frame in frames):
        energy_by_step = {int(record["step"]): record for record in energy_records}
        aligned = [
            energy_by_step.get(int(frame["step"]))
            for frame in frames
        ]
        if all(record is not None for record in aligned):
            frame_energy_records = [record for record in aligned if record is not None]
        else:
            logger.warning(
                "Could not align all trajectory frames with energy steps; "
                "falling back to positional energy matching."
            )

    # Determine which frame indices to write
    if event_driven and frame_energy_records:
        miner = TrajectoryMiner(frame_energy_records, frames)
        result = miner.mine(
            event_window=event_window,
            sparse_stride=sparse_stride,
            energy_threshold=energy_threshold,
        )
        frame_indices = set(result.sampled_indices)
        logger.info(
            "Event-driven sampling: %d events detected, "
            "%d / %d frames selected (compression %.1f×)",
            len(result.events),
            len(frame_indices),
            len(frames),
            result.stats.get("compression_ratio", 1.0),
        )
    else:
        if event_driven and not energy_records:
            logger.warning(
                "event_driven=True but no .ener file found; "
                "falling back to uniform subsampling."
            )
        frame_indices = set(range(0, len(frames), max(1, subsample)))

    # ------------------------------------------------------------------
    # Unit audit gate: the parsers above convert CP2K output to eV /
    # eV/Angstrom at read time. Audit the frames that are about to be
    # written; on failure nothing is written and the collect aborts.
    # ------------------------------------------------------------------
    from common.unit_audit import audit_frame_values

    audit_e, audit_n, audit_f = [], [], []
    for i in sorted(frame_indices):
        if i >= len(frames):
            continue
        frame = frames[i]
        energy = None
        if i < len(frame_energy_records):
            energy = frame_energy_records[i]["potential_energy"]
        if energy is not None and frame["forces"] is not None:
            audit_e.append(float(energy))
            audit_n.append(int(frame["n_atoms"]))
            audit_f.append(frame["forces"])
    if audit_e:
        report = audit_frame_values(audit_e, audit_n, audit_f)
        if not report.ok:
            raise RuntimeError(
                f"Unit audit failed for {pos_xyz_path} — refusing to write "
                f"frames to {db_path}:\n{report.format()}"
            )
        for warning in report.warnings:
            logger.warning("unit audit: %s (%s)", warning, pos_xyz_path)
    else:
        logger.warning(
            "unit audit skipped for %s: no frames with both energy and forces",
            pos_xyz_path,
        )

    from ase.calculators.singlepoint import SinglePointCalculator

    added = 0
    with connect(str(db_path), append=True) as db:
        for i, frame in enumerate(frames):
            if i not in frame_indices:
                continue

            atoms = Atoms(
                symbols=frame["symbols"],
                positions=frame["positions"],
            )
            frame_cell = frame.get("cell")
            if frame_cell is not None:
                atoms.set_cell(frame_cell)
                atoms.set_pbc(True)

            energy = None
            temperature = None
            if i < len(frame_energy_records):
                energy = frame_energy_records[i]["potential_energy"]
                temperature = frame_energy_records[i]["temperature"]

            if energy is None and frame["forces"] is None:
                logger.warning(
                    "Skipping frame %d: both energy and forces are missing.", i
                )
                continue

            calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=frame["forces"],
            )
            atoms.calc = calc

            kvp = {}
            for k, v in metadata.items():
                if isinstance(v, dict):
                    kvp[k] = json.dumps(v)
                else:
                    kvp[k] = v
            kvp["frame_index"] = i
            kvp["event_driven"] = event_driven
            if temperature is not None:
                kvp["temperature"] = temperature

            # schnetpack reads labels from row.data, not the reserved columns,
            # so mirror energy/forces into data (energy shape (1,), matching
            # the convention of the other ingest paths).
            frame_data: dict[str, Any] = {}
            if energy is not None:
                frame_data["energy"] = np.array([energy], dtype=np.float64)
            if frame["forces"] is not None:
                frame_data["forces"] = np.asarray(frame["forces"], dtype=np.float64)

            db.write(atoms, data=frame_data, **kvp)
            added += 1

    return added


def get_db_stats(db_path: str | Path) -> dict[str, Any]:
    """Get basic statistics from an ASE DB."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {"n_structures": 0}

    with connect(str(db_path)) as db:
        count = db.count()
        if count == 0:
            return {"n_structures": 0}

        temperatures = []
        energies = []
        for row in db.select():
            if "temperature" in row.key_value_pairs:
                temperatures.append(row.temperature)
            if "energy" in row.key_value_pairs:
                energies.append(row.energy)

        stats = {"n_structures": count}
        if temperatures:
            stats["temperature_min"] = min(temperatures)
            stats["temperature_max"] = max(temperatures)
            stats["temperature_mean"] = sum(temperatures) / len(temperatures)
        if energies:
            stats["energy_min"] = min(energies)
            stats["energy_max"] = max(energies)
            stats["energy_mean"] = sum(energies) / len(energies)

        return stats


def load_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2, default=str)


def load_al_manifest(build_dir: str | Path) -> dict[str, Any]:
    """Load or create AL build manifest."""
    path = Path(build_dir) / "al_manifest.json"
    if path.exists():
        return load_json(path)
    return {
        "seed_completed": False,
        "al_iterations": [],
        "total_frames": 0,
    }


def save_al_manifest(build_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(build_dir) / "al_manifest.json"
    save_json(path, manifest)
