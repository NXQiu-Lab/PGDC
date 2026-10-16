"""Parser for VASP INCAR configuration files.

This module provides utilities to parse VASP INCAR files and extract
key parameters relevant to AIMD simulations for acquisition value estimation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class IncarParams:
    """Parsed INCAR parameters relevant to AIMD simulations.

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
    """

    tebeg: float | None = None
    teend: float | None = None
    nsw: int | None = None
    potim: float | None = None
    nbands: int | None = None
    encut: float | None = None
    isym: int | None = None
    ismear: int | None = None
    sigma: float | None = None
    algo: str | None = None
    ibrion: int | None = None
    mdalgo: int | None = None
    andersen_prob: float | None = None
    tein: float | None = None
    smass: float | None = None

    @property
    def is_md_simulation(self) -> bool:
        """Check if the configuration is for MD simulation (IBRION=0)."""
        return self.ibrion == 0

    @property
    def effective_temperature(self) -> float:
        """Get the effective simulation temperature.

        Returns TEBEG if available, otherwise TEIN, otherwise 300K default.
        """
        if self.tebeg is not None:
            return self.tebeg
        if self.tein is not None:
            return self.tein
        return 300.0

    @property
    def effective_steps(self) -> int:
        """Get the effective number of MD steps.

        Returns NSW if available, otherwise 1000 default.
        """
        return self.nsw if self.nsw is not None else 1000

    @property
    def effective_timestep(self) -> float:
        """Get the effective time step in fs.

        Returns POTIM if available, otherwise 1.0 fs default.
        """
        return self.potim if self.potim is not None else 1.0

    @property
    def total_simulation_time(self) -> float:
        """Calculate total simulation time in picoseconds.

        Returns:
            Total simulation time (NSW * POTIM / 1000).
        """
        return self.effective_steps * self.effective_timestep / 1000.0

    def estimate_diversity_proxy(self) -> float:
        """Estimate trajectory diversity based on INCAR parameters.

        This is a simplified model that estimates the expected structural
        diversity from an AIMD run based on temperature and simulation length.

        Returns:
            A diversity score between 0 and 1, where higher values indicate
            potentially more diverse trajectories.
        """
        # Temperature factor: higher temp -> more diversity
        # Normalize to typical MD range (300K - 2000K)
        temp_factor = min(self.effective_temperature / 2000.0, 1.0)

        # Time factor: longer simulation -> more diversity
        # Normalize to typical range (1 ps - 20 ps)
        time_factor = min(self.total_simulation_time / 20.0, 1.0)

        # Combined score with equal weighting
        return 0.5 * temp_factor + 0.5 * time_factor


class IncarParseError(RuntimeError):
    """Exception raised for INCAR parsing failures."""


class IncarParser:
    """Parse VASP INCAR files into structured parameters.

    This parser handles the standard VASP INCAR format with key-value pairs.
    It extracts parameters most relevant for AIMD simulation characterization.
    """

    # Map of parameter names to their types
    _PARAM_TYPES: dict[str, type] = {
        "TEBEG": float,
        "TEEND": float,
        "NSW": int,
        "POTIM": float,
        "NBANDS": int,
        "ENCUT": float,
        "ISYM": int,
        "ISMEAR": int,
        "SIGMA": float,
        "ALGO": str,
        "IBRION": int,
        "MDALGO": int,
        "ANDERSEN_PROB": float,
        "TEIN": float,
        "SMASS": float,
    }

    def parse(self, path: str | Path) -> IncarParams:
        """Parse an INCAR file into IncarParams.

        Args:
            path: Path to the INCAR file.

        Returns:
            IncarParams containing the parsed parameters.

        Raises:
            FileNotFoundError: If the file does not exist.
            IncarParseError: If the file cannot be parsed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"INCAR file not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise IncarParseError(f"Failed to read INCAR file: {path}") from error

        return self._parse_content(content)

    def _parse_content(self, content: str) -> IncarParams:
        """Parse INCAR content string.

        Args:
            content: String content of the INCAR file.

        Returns:
            IncarParams with extracted values.
        """
        params: dict[str, Any] = {}

        for line in content.splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("!") or line.startswith("#"):
                continue

            # Remove inline comments
            if "!" in line:
                line = line[: line.index("!")].strip()
            if "#" in line:
                line = line[: line.index("#")].strip()

            # Parse key-value pair
            parsed = self._parse_line(line)
            if parsed:
                key, value = parsed
                # Convert key to lowercase for IncarParams field matching
                params[key.lower()] = value

        return IncarParams(**params)

    def _parse_line(self, line: str) -> tuple[str, Any] | None:
        """Parse a single INCAR line.

        Args:
            line: The line to parse.

        Returns:
            Tuple of (parameter_name, value) or None if not a valid assignment.
        """
        # Match pattern: KEY = VALUE or KEY=VALUE
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", line)
        if not match:
            return None

        key = match.group(1).upper()
        value_str = match.group(2).strip()

        # Get the expected type for this parameter
        param_type = self._PARAM_TYPES.get(key)
        if param_type is None:
            # Unknown parameter, skip it
            return None

        # Parse the value according to expected type
        try:
            if param_type == str:
                # Remove quotes if present
                if (value_str.startswith('"') and value_str.endswith('"')) or (
                    value_str.startswith("'") and value_str.endswith("'")
                ):
                    value_str = value_str[1:-1]
                return key, value_str
            elif param_type == int:
                return key, int(value_str)
            elif param_type == float:
                # Handle scientific notation and special values
                value_str = value_str.replace("d", "e").replace("D", "e")
                return key, float(value_str)
        except ValueError:
            # Failed to parse, skip this parameter
            return None

        return None

    def parse_optional(self, path: str | Path | None) -> IncarParams | None:
        """Optionally parse an INCAR file.

        Args:
            path: Path to the INCAR file, or None.

        Returns:
            IncarParams if path is provided and valid, None otherwise.
        """
        if path is None:
            return None
        try:
            return self.parse(path)
        except (FileNotFoundError, IncarParseError):
            return None


def parse_incar(path: str | Path) -> IncarParams:
    """Convenience function to parse an INCAR file.

    Args:
        path: Path to the INCAR file.

    Returns:
        IncarParams containing the parsed parameters.
    """
    return IncarParser().parse(path)
