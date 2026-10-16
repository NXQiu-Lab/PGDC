"""Deterministic pre-filter for candidate structures.

Replaces LJ-based relaxation with data-driven physical plausibility checks.
Classifies structures into EQ/MS/TP/TD/LU/AB categories.

module_id: sampling.prefilter
module_version: 1.0.0
project_version: 1.1.0
last_updated: 2026-05-19
generation_mode: manual
source_files: [prefilter.py]
upstream_modules: [domain]
downstream_modules: [generator, relaxer]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atom_dataset_index.domain import DatasetBundle, StructureRecord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Covalent radii (Å) for common elements, from ASE/CPK data
_COVALENT_RADII: dict[int, float] = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
    9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
    23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32,
    30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
    37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47,
    44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07,
    58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96,
    65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87,
    72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36,
    79: 1.36, 80: 1.32, 81: 1.45, 82: 1.47, 83: 1.46, 84: 1.48, 85: 1.40,
    86: 1.50,
}

# Default covalent radius for unknown elements
_DEFAULT_RADIUS: float = 1.20

# Minimum distance threshold as fraction of covalent radii sum
# Below this, atoms are considered overlapping (LU)
_MIN_DISTANCE_FACTOR: float = 0.5

# Maximum coordination number bounds (chemical plausibility)
# Key: element Z, Value: (min_cn, max_cn)
_CN_PLAUSIBILITY: dict[int, tuple[int, int]] = {
    # Transition metals (common in HEA)
    26: (4, 14),  # Fe
    27: (4, 14),  # Co
    28: (4, 14),  # Ni
    24: (4, 14),  # Cr
    25: (4, 14),  # Mn
    29: (4, 14),  # Cu
    22: (4, 14),  # Ti
    23: (4, 14),  # V
    40: (4, 14),  # Zr
    41: (4, 14),  # Nb
    42: (4, 14),  # Mo
    73: (4, 14),  # Ta
    74: (4, 14),  # W
    # Oxides / ceramics
    8: (1, 8),    # O
    13: (3, 12),  # Al
    14: (2, 8),   # Si
    72: (4, 14),  # Hf
}

# Default CN bounds for elements not in the table
_DEFAULT_CN_BOUNDS: tuple[int, int] = (1, 16)

# Force threshold for LU (absolute, in eV/Å)
_LU_MAX_FORCE: float = 10.0

# Harmonic spring constant for force estimation (eV/Å²)
_HARMONIC_K: float = 7.0


def _get_covalent_radius(z: int) -> float:
    """Get covalent radius for atomic number Z."""
    return _COVALENT_RADII.get(z, _DEFAULT_RADIUS)


def _composition_key(record: StructureRecord) -> str:
    """Generate a composition key for grouping."""
    counts: dict[int, int] = {}
    for z in record.atomic_numbers:
        counts[z] = counts.get(z, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return "empty"
    # Sort by atomic number, format as Z:fraction
    parts = []
    for z in sorted(counts):
        frac = counts[z] / total
        parts.append(f"{z}:{frac:.3f}")
    return "_".join(parts)


@dataclass(frozen=True)
class StructureClassResult:
    """Result of structure classification."""

    label: str  # EQ, MS, TP, TD, LU, AB
    passed: bool  # True for EQ, MS, TP, TD
    reason: str  # Human-readable explanation
    details: dict[str, object]  # Numerical details


class DeterministicPreFilter:
    """Data-driven pre-filter for candidate structures.

    Builds envelopes from a reference dataset and classifies candidates
    into six categories:
        EQ: Equilibrium (near-stable)
        MS: Metastable
        TP: Transition-Prone (core target)
        TD: Thermally-Disordered
        LU: Locally-Unstable (rejected)
        AB: Absurd/physically impossible (rejected)

    The filter does NOT require a trained MLIP. It uses:
        - Density envelope (per composition group)
        - Bond length envelope (per element pair)
        - Coordination number envelope (per element)
        - Force z-score (relative to reference pool statistics)
        - Structural distance metrics

    Example:
        >>> from atom_dataset_index.io import AseDbDatasetReader
        >>> from atom_dataset_index.sampling import DeterministicPreFilter
        >>> dataset = AseDbDatasetReader("data/existing.db").read()
        >>> pf = DeterministicPreFilter(dataset)
        >>> result = pf.classify(candidate_record)
        >>> print(result.label, result.passed, result.reason)
    """

    def __init__(self, reference_dataset: DatasetBundle | None = None) -> None:
        """Initialize filter, optionally building envelopes from a dataset.

        Args:
            reference_dataset: Existing dataset to derive statistical envelopes.
                If None, the filter operates with conservative default bounds.
        """
        self._density_bounds: dict[str, tuple[float, float]] = {}
        self._bond_length_bounds: dict[tuple[int, int], tuple[float, float]] = {}
        self._cn_bounds: dict[int, tuple[float, float]] = {}
        self._force_stats: dict[str, tuple[float, float]] = {}  # pool -> (mean, std)
        self._energy_stats: dict[str, tuple[float, float]] = {}  # pool -> (min, range)
        self._distance_stats: dict[str, tuple[float, float]] = {}  # pool -> (mean, max)

        if reference_dataset is not None:
            self._build_envelopes(reference_dataset)

    # ------------------------------------------------------------------
    # Envelope construction
    # ------------------------------------------------------------------

    def _build_envelopes(self, dataset: DatasetBundle) -> None:
        """Build statistical envelopes from reference dataset."""
        logger.info(f"Building envelopes from {len(dataset.records)} reference structures")

        # Group by composition
        groups: dict[str, list[StructureRecord]] = {}
        for record in dataset.records:
            key = _composition_key(record)
            groups.setdefault(key, []).append(record)

        for comp_key, records in groups.items():
            self._build_pool_envelopes(comp_key, records)

        logger.info(
            f"Envelopes built: {len(self._density_bounds)} density, "
            f"{len(self._bond_length_bounds)} bond, "
            f"{len(self._cn_bounds)} CN, "
            f"{len(self._force_stats)} force stats"
        )

    def _build_pool_envelopes(self, pool_id: str, records: list[StructureRecord]) -> None:
        """Build envelopes for a single composition pool."""
        densities: list[float] = []
        bond_lengths: dict[tuple[int, int], list[float]] = {}
        cns: dict[int, list[float]] = {}
        forces: list[float] = []
        energies: list[float] = []

        for record in records:
            # Density
            try:
                density = _compute_density(record)
                if density > 0:
                    densities.append(density)
            except Exception:
                pass

            # Bond lengths and coordination numbers
            try:
                bls, cn_dict = _compute_bonds_and_cn(record)
                for pair, dist_list in bls.items():
                    bond_lengths.setdefault(pair, []).extend(dist_list)
                for z, cn_val in cn_dict.items():
                    cns.setdefault(z, []).append(cn_val)
            except Exception:
                pass

            # Forces (if available)
            if record.forces:
                try:
                    mean_force = _compute_mean_force_norm(record)
                    if mean_force > 0:
                        forces.append(mean_force)
                except Exception:
                    pass

            # Energy (if available)
            if record.energy is not None and record.natoms > 0:
                try:
                    e_per_atom = float(record.energy) / record.natoms
                    energies.append(e_per_atom)
                except Exception:
                    pass

        # Store density bounds with 30% margin
        if densities:
            d_min, d_max = min(densities), max(densities)
            margin = (d_max - d_min) * 0.3 + d_min * 0.1
            self._density_bounds[pool_id] = (d_min - margin, d_max + margin)

        # Store bond length bounds with 20% margin
        for pair, dist_list in bond_lengths.items():
            if len(dist_list) >= 3:
                r_min, r_max = min(dist_list), max(dist_list)
                margin = (r_max - r_min) * 0.2 + r_min * 0.1
                self._bond_length_bounds[pair] = (max(0.1, r_min - margin), r_max + margin)

        # Store CN bounds
        for z, cn_list in cns.items():
            if len(cn_list) >= 3:
                cn_min, cn_max = min(cn_list), max(cn_list)
                self._cn_bounds[z] = (cn_min - 1.0, cn_max + 1.0)

        # Store force stats
        if forces:
            f_mean = sum(forces) / len(forces)
            f_std = math.sqrt(sum((f - f_mean) ** 2 for f in forces) / len(forces)) if len(forces) > 1 else 0.1
            if f_std < 0.01:
                f_std = 0.1  # Avoid division by zero
            self._force_stats[pool_id] = (f_mean, f_std)

        # Store energy stats
        if energies:
            e_min = min(energies)
            e_max = max(energies)
            e_range = max(e_max - e_min, 0.01)
            self._energy_stats[pool_id] = (e_min, e_range)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        record: StructureRecord,
        pool_id: str | None = None,
        estimate_forces: bool = True,
    ) -> StructureClassResult:
        """Classify a structure into EQ/MS/TP/TD/LU/AB.

        Args:
            record: Structure to classify
            pool_id: Composition pool identifier. If None, inferred from record.
            estimate_forces: Whether to estimate forces via harmonic approximation
                if no forces are present in the record.

        Returns:
            StructureClassResult with label, pass/fail, reason, and details
        """
        if pool_id is None:
            pool_id = _composition_key(record)

        details: dict[str, object] = {}

        # ===== Step 1: AB check (physical absurdity) =====
        ab_reason = self._check_absurd(record, pool_id, details)
        if ab_reason:
            return StructureClassResult(
                label="AB",
                passed=False,
                reason=ab_reason,
                details=details,
            )

        # ===== Step 2: LU check (local instability) =====
        lu_reason = self._check_locally_unstable(record, pool_id, estimate_forces, details)
        if lu_reason:
            return StructureClassResult(
                label="LU",
                passed=False,
                reason=lu_reason,
                details=details,
            )

        # ===== Step 3: Compute normalized metrics for fine classification =====
        dE_norm, F_zscore, D_norm = self._compute_normalized_metrics(
            record, pool_id, estimate_forces, details
        )

        details["dE_norm"] = round(dE_norm, 4)
        details["F_zscore"] = round(F_zscore, 4)
        details["D_norm"] = round(D_norm, 4)

        # ===== Step 4: Classify into EQ / MS / TP / TD =====
        if dE_norm < 0.1 and F_zscore < 1.0 and D_norm < 0.3:
            return StructureClassResult(
                label="EQ",
                passed=True,
                reason="Near-equilibrium structure: low energy offset, low force, close to stable references",
                details=details,
            )

        if dE_norm < 0.5 and F_zscore < 2.0 and D_norm < 0.5:
            return StructureClassResult(
                label="MS",
                passed=True,
                reason="Metastable structure: moderate deviation but physically sound",
                details=details,
            )

        if F_zscore >= 1.0 and F_zscore <= 3.0 and D_norm >= 0.3 and D_norm <= 1.0:
            return StructureClassResult(
                label="TP",
                passed=True,
                reason="Transition-prone structure: elevated force and structural deviation, likely to cross barriers in AIMD",
                details=details,
            )

        if F_zscore > 2.0 and dE_norm > 0.3:
            return StructureClassResult(
                label="TD",
                passed=True,
                reason="Thermally-disordered structure: high force and energy but coordinated, viable at elevated temperature",
                details=details,
            )

        # Fallback: if it passed AB/LU but doesn't fit above, classify as MS
        # (this handles edge cases near boundaries)
        return StructureClassResult(
            label="MS",
            passed=True,
            reason="Boundary case: does not strictly match any category, treated as metastable",
            details=details,
        )

    def check(self, record: StructureRecord, pool_id: str | None = None) -> tuple[bool, str]:
        """Quick pass/fail check (AB and LU are rejected, all others pass).

        Args:
            record: Structure to check
            pool_id: Composition pool identifier

        Returns:
            (is_passed, reason) tuple
        """
        result = self.classify(record, pool_id)
        return result.passed, result.reason

    def filter_candidates(
        self,
        candidates: list[StructureRecord],
        pool_id: str | None = None,
    ) -> tuple[list[StructureRecord], list[StructureClassResult]]:
        """Filter a list of candidates, returning passed ones and all results.

        Args:
            candidates: List of candidate structures
            pool_id: Pool identifier (if None, inferred per-candidate)

        Returns:
            (passed_records, all_results) tuple
        """
        passed: list[StructureRecord] = []
        results: list[StructureClassResult] = []

        for record in candidates:
            pid = pool_id if pool_id is not None else _composition_key(record)
            result = self.classify(record, pid)
            results.append(result)
            if result.passed:
                passed.append(record)

        return passed, results

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_absurd(
        self,
        record: StructureRecord,
        pool_id: str,
        details: dict[str, object],
    ) -> str:
        """Check for physically absurd structures. Returns reason string or empty."""
        # Check 1: Minimum distance (atom overlap)
        try:
            min_dist = _compute_min_distance(record)
            details["min_distance"] = round(min_dist, 4)
        except Exception:
            min_dist = float("inf")

        if min_dist < 0.5:
            return f"Atoms too close: minimum distance {min_dist:.2f} Å < 0.5 Å"

        # Check 2: Per-pair minimum distance against covalent radii
        for i in range(record.natoms):
            for j in range(i + 1, record.natoms):
                zi, zj = record.atomic_numbers[i], record.atomic_numbers[j]
                dist = _atom_distance(record, i, j)
                r_sum = _get_covalent_radius(zi) + _get_covalent_radius(zj)
                if dist < r_sum * _MIN_DISTANCE_FACTOR:
                    return (
                        f"Severe atom overlap: pair ({zi},{zj}) distance "
                        f"{dist:.2f} Å < {r_sum * _MIN_DISTANCE_FACTOR:.2f} Å "
                        f"(covalent radii sum × {_MIN_DISTANCE_FACTOR})"
                    )

        # Check 3: Density bounds (if reference data available)
        if pool_id in self._density_bounds:
            try:
                density = _compute_density(record)
                details["density"] = round(density, 4)
                d_min, d_max = self._density_bounds[pool_id]
                if density < d_min * 0.5 or density > d_max * 2.0:
                    return (
                        f"Density {density:.2f} g/cm³ far outside envelope "
                        f"[{d_min:.2f}, {d_max:.2f}]"
                    )
            except Exception:
                pass

        # Check 4: Cell volume reasonableness
        if record.cell is not None and any(record.pbc):
            try:
                volume = _compute_cell_volume(record.cell)
                vpa = volume / max(record.natoms, 1)
                if vpa < 3.0:
                    return f"Unrealistically small volume per atom: {vpa:.2f} Å³"
                if vpa > 500.0:
                    return f"Unrealistically large volume per atom: {vpa:.2f} Å³"
                details["volume_per_atom"] = round(vpa, 2)
            except Exception:
                pass

        # Check 5: Coordination number chemical plausibility
        try:
            _, cn_dict = _compute_bonds_and_cn(record)
            for z, cn in cn_dict.items():
                cn_min_hard, cn_max_hard = _CN_PLAUSIBILITY.get(z, _DEFAULT_CN_BOUNDS)
                if cn < cn_min_hard or cn > cn_max_hard:
                    elem = _z_to_symbol(z)
                    return (
                        f"Implausible coordination number for {elem}({z}): "
                        f"CN={cn:.1f}, allowed [{cn_min_hard}, {cn_max_hard}]"
                    )
        except Exception:
            pass

        return ""

    def _check_locally_unstable(
        self,
        record: StructureRecord,
        pool_id: str,
        estimate_forces: bool,
        details: dict[str, object],
    ) -> str:
        """Check for locally unstable structures. Returns reason string or empty."""
        # Check 1: Extreme forces
        mean_force = 0.0
        max_force = 0.0

        if record.forces:
            try:
                mean_force = _compute_mean_force_norm(record)
                max_force = max(
                    math.sqrt(sum(c * c for c in f)) for f in record.forces
                )
            except Exception:
                pass
        elif estimate_forces:
            try:
                mean_force = self._estimate_harmonic_force(record, pool_id)
            except Exception:
                pass

        details["mean_force"] = round(mean_force, 4)
        details["max_force"] = round(max_force, 4)

        if max_force > _LU_MAX_FORCE:
            return f"Extreme local force: max ||F|| = {max_force:.2f} eV/Å > {_LU_MAX_FORCE} eV/Å"

        # Check 2: Force z-score (if reference stats available)
        if pool_id in self._force_stats:
            f_mean, f_std = self._force_stats[pool_id]
            if f_std > 0:
                f_zscore = (mean_force - f_mean) / f_std
                if f_zscore > 5.0:
                    return (
                        f"Force extremely high relative to pool: "
                        f"z-score = {f_zscore:.2f} (mean={f_mean:.3f}, std={f_std:.3f})"
                    )

        # Check 3: Bond lengths far outside envelope (indicates severe distortion)
        try:
            bls, _ = _compute_bonds_and_cn(record)
            for pair, dist_list in bls.items():
                if pair in self._bond_length_bounds:
                    r_min, r_max = self._bond_length_bounds[pair]
                    for dist in dist_list:
                        if dist < r_min * 0.5:
                            return (
                                f"Severely compressed bond {pair}: "
                                f"{dist:.2f} Å < {r_min * 0.5:.2f} Å"
                            )
        except Exception:
            pass

        return ""

    def _compute_normalized_metrics(
        self,
        record: StructureRecord,
        pool_id: str,
        estimate_forces: bool,
        details: dict[str, object],
    ) -> tuple[float, float, float]:
        """Compute ΔE_norm, F_zscore, D_norm."""
        # ΔE_norm
        dE_norm = 0.0
        if record.energy is not None and record.natoms > 0:
            e_per_atom = float(record.energy) / record.natoms
            if pool_id in self._energy_stats:
                e_min, e_range = self._energy_stats[pool_id]
                dE_norm = (e_per_atom - e_min) / e_range
                dE_norm = max(0.0, dE_norm)
            else:
                # No reference: use very conservative estimate
                dE_norm = 0.0
        details["energy_per_atom"] = float(record.energy) / record.natoms if record.energy else None

        # F_zscore
        f_zscore = 0.0
        mean_force = 0.0
        if record.forces:
            mean_force = _compute_mean_force_norm(record)
        elif estimate_forces:
            mean_force = self._estimate_harmonic_force(record, pool_id)

        if pool_id in self._force_stats:
            f_mean, f_std = self._force_stats[pool_id]
            if f_std > 0:
                f_zscore = (mean_force - f_mean) / f_std
        else:
            # Conservative fallback: if force > 1.0 eV/Å, assume non-equilibrium
            f_zscore = mean_force / 0.5  # 0.5 eV/Å as reference std

        # D_norm: structural distance proxy (using density deviation as proxy)
        # In a full implementation, this would use SOAP descriptors.
        # Here we use a composite proxy: density deviation + CN deviation.
        d_norm = 0.0
        try:
            density = _compute_density(record)
            if pool_id in self._density_bounds:
                d_min, d_max = self._density_bounds[pool_id]
                d_center = (d_min + d_max) / 2.0
                d_span = max(d_max - d_min, 0.1)
                d_norm = abs(density - d_center) / d_span

            # Add CN deviation contribution
            _, cn_dict = _compute_bonds_and_cn(record)
            cn_dev = 0.0
            cn_count = 0
            for z, cn in cn_dict.items():
                if z in self._cn_bounds:
                    cn_min, cn_max = self._cn_bounds[z]
                    cn_center = (cn_min + cn_max) / 2.0
                    cn_span = max(cn_max - cn_min, 0.5)
                    cn_dev += abs(cn - cn_center) / cn_span
                    cn_count += 1
            if cn_count > 0:
                d_norm = 0.5 * d_norm + 0.5 * (cn_dev / cn_count)
        except Exception:
            pass

        return dE_norm, f_zscore, d_norm

    def _estimate_harmonic_force(
        self,
        record: StructureRecord,
        pool_id: str,
    ) -> float:
        """Estimate mean force via harmonic approximation using bond-length deviations.

        For each bond in the structure, computes the deviation from its "ideal"
        bond length (centre of the reference envelope, or sum of covalent radii as
        fallback), then estimates the restoring force as F ≈ k × |r − r_ideal|.

        This gives a physically meaningful proxy without requiring reference forces:
        a bond compressed/stretched by Δr experiences a force of k × Δr.
        The mean over all bonds is returned as F_mean.

        Args:
            record: Candidate structure (no forces needed).
            pool_id: Composition pool identifier.

        Returns:
            Estimated mean force norm (eV/Å).
        """
        try:
            bls, _ = _compute_bonds_and_cn(record)
        except Exception:
            return 0.0

        if not bls:
            return 0.0

        force_samples: list[float] = []
        for pair, dist_list in bls.items():
            # Determine ideal bond length for this pair
            if pair in self._bond_length_bounds:
                r_min, r_max = self._bond_length_bounds[pair]
                r_ideal = (r_min + r_max) / 2.0
            else:
                zi, zj = pair
                r_ideal = _get_covalent_radius(zi) + _get_covalent_radius(zj)

            for dist in dist_list:
                deviation = abs(dist - r_ideal)
                force_samples.append(_HARMONIC_K * deviation)

        if not force_samples:
            return 0.0
        return sum(force_samples) / len(force_samples)


# ----------------------------------------------------------------------
# Helper functions (geometry / chemistry)
# ----------------------------------------------------------------------

def _z_to_symbol(z: int) -> str:
    """Convert atomic number to symbol (limited set)."""
    symbols = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
        9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 22: "Ti", 23: "V",
        24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
        40: "Zr", 41: "Nb", 42: "Mo", 72: "Hf", 73: "Ta", 74: "W",
    }
    return symbols.get(z, f"Z{z}")


def _compute_density(record: StructureRecord) -> float:
    """Compute density in g/cm³."""
    if record.cell is None or not any(record.pbc):
        raise ValueError("Cannot compute density for non-periodic structure")

    volume = _compute_cell_volume(record.cell)
    if volume <= 0:
        raise ValueError("Invalid cell volume")

    # Total mass in amu
    total_mass = sum(_atomic_mass(z) for z in record.atomic_numbers)

    # 1 amu = 1.66053906660e-24 g
    # 1 Å³ = 1e-24 cm³
    # density = mass / volume (g/cm³)
    # amu/Å³ * 1.660539 = g/cm³
    AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
    density = (total_mass / volume) * AMU_PER_A3_TO_G_PER_CM3

    return density


def _atomic_mass(z: int) -> float:
    """Approximate atomic mass for common elements."""
    masses = {
        1: 1.008, 2: 4.003, 3: 6.94, 4: 9.012, 5: 10.81, 6: 12.01,
        7: 14.01, 8: 16.00, 9: 19.00, 10: 20.18, 11: 22.99, 12: 24.31,
        13: 26.98, 14: 28.09, 15: 30.97, 16: 32.06, 17: 35.45, 18: 39.95,
        19: 39.10, 20: 40.08, 21: 44.96, 22: 47.87, 23: 50.94, 24: 52.00,
        25: 54.94, 26: 55.85, 27: 58.93, 28: 58.69, 29: 63.55, 30: 65.38,
        31: 69.72, 32: 72.63, 33: 74.92, 34: 78.97, 35: 79.90, 36: 83.80,
        37: 85.47, 38: 87.62, 39: 88.91, 40: 91.22, 41: 92.91, 42: 95.95,
        43: 98.00, 44: 101.07, 45: 102.91, 46: 106.42, 47: 107.87,
        48: 112.41, 49: 114.82, 50: 118.71, 51: 121.76, 52: 127.60,
        53: 126.90, 54: 131.29, 55: 132.91, 56: 137.33, 57: 138.91,
        58: 140.12, 72: 178.49, 73: 180.95, 74: 183.84, 75: 186.21,
        76: 190.23, 77: 192.22, 78: 195.08, 79: 196.97, 80: 200.59,
    }
    return masses.get(z, float(z))  # Fallback: use Z as rough mass


def _compute_cell_volume(cell: tuple[tuple[float, float, float], ...]) -> float:
    """Compute volume of a unit cell."""
    a = cell[0]
    b = cell[1]
    c = cell[2]
    bc_x = b[1] * c[2] - b[2] * c[1]
    bc_y = b[2] * c[0] - b[0] * c[2]
    bc_z = b[0] * c[1] - b[1] * c[0]
    return abs(a[0] * bc_x + a[1] * bc_y + a[2] * bc_z)


def _compute_min_distance(record: StructureRecord) -> float:
    """Compute minimum interatomic distance."""
    if len(record.positions) < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(len(record.positions)):
        for j in range(i + 1, len(record.positions)):
            dist = _atom_distance(record, i, j)
            min_dist = min(min_dist, dist)
    return min_dist


def _atom_distance(record: StructureRecord, i: int, j: int) -> float:
    """Distance between atoms i and j (no PBC)."""
    pi = record.positions[i]
    pj = record.positions[j]
    return math.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2)


def _compute_bonds_and_cn(
    record: StructureRecord,
    cn_cutoff_factor: float = 1.3,
) -> tuple[dict[tuple[int, int], list[float]], dict[int, float]]:
    """Compute bond lengths and coordination numbers.

    Uses a distance cutoff based on covalent radii sum × factor.

    Returns:
        (bond_lengths, coordination_numbers) where:
        - bond_lengths: dict mapping (Z_i, Z_j) -> list of distances
        - coordination_numbers: dict mapping Z -> mean CN for that element
    """
    natoms = record.natoms
    if natoms < 2 or not record.positions:
        return {}, {}

    # Compute pairwise distances and determine bonds
    bonds_per_atom: dict[int, list[float]] = {i: [] for i in range(natoms)}
    bond_lengths: dict[tuple[int, int], list[float]] = {}

    for i in range(natoms):
        zi = record.atomic_numbers[i]
        ri = _get_covalent_radius(zi)
        for j in range(i + 1, natoms):
            zj = record.atomic_numbers[j]
            rj = _get_covalent_radius(zj)
            dist = _atom_distance(record, i, j)
            cutoff = (ri + rj) * cn_cutoff_factor
            if dist <= cutoff:
                bonds_per_atom[i].append(dist)
                bonds_per_atom[j].append(dist)
                pair = (min(zi, zj), max(zi, zj))
                bond_lengths.setdefault(pair, []).append(dist)

    # Compute mean CN per element
    cn_per_element: dict[int, list[int]] = {}
    for i in range(natoms):
        zi = record.atomic_numbers[i]
        cn = len(bonds_per_atom[i])
        cn_per_element.setdefault(zi, []).append(cn)

    cn_dict = {z: sum(cns) / len(cns) for z, cns in cn_per_element.items()}

    return bond_lengths, cn_dict


def _compute_mean_force_norm(record: StructureRecord) -> float:
    """Compute mean force norm."""
    if not record.forces:
        return 0.0
    norms = [math.sqrt(sum(c * c for c in f)) for f in record.forces]
    return sum(norms) / len(norms)
