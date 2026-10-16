"""Dataset-aware candidate explorers for active learning.

Provides structure generation strategies that use existing dataset
information to guide exploration toward under-sampled regions.

module_id: sampling.explorers
module_version: 1.0.0
project_version: 1.1.0
last_updated: 2026-05-19
generation_mode: manual
source_files: [explorers.py]
upstream_modules: [domain, metric_core.descriptors, sampling.utils]
downstream_modules: [generator, prefilter]
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import DescriptorAdapter, SOAPDescriptorAdapter
from atom_dataset_index.sampling.utils import ase_to_record, record_to_ase

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BoundaryExplorer
# ---------------------------------------------------------------------------


class BoundaryExplorer:
    """Explore the boundary of a dataset by perturbing far-field structures.

    Uses descriptor-space distance to identify structures that are farthest
    from the dataset centroid, then applies small physical perturbations
    to generate new candidates near the boundary.

    This implements the "explore周边 + 填补缺口" strategy: rather than
    jumping to completely unknown regions, we nudge the existing boundary
    outward step by step.

    Example:
        >>> explorer = BoundaryExplorer(dataset)
        >>> candidates = explorer.explore(n_samples=20, perturbation_std=0.05)
    """

    def __init__(
        self,
        dataset: DatasetBundle,
        adapter: DescriptorAdapter | None = None,
    ) -> None:
        """Initialize explorer with a reference dataset.

        Args:
            dataset: Existing dataset to define the boundary.
            adapter: Descriptor adapter for distance computation.
                Defaults to SOAPDescriptorAdapter.
        """
        self.dataset = dataset
        self.adapter = adapter or SOAPDescriptorAdapter()
        self._descriptors: dict[int, tuple[float, ...]] = {}
        self._centroid: tuple[float, ...] | None = None
        self._build_descriptor_index()

    def _build_descriptor_index(self) -> None:
        """Precompute descriptors for all dataset structures."""
        if not self.dataset.records:
            return

        descriptors_list = self.adapter.encode_batch(list(self.dataset.records))
        for record, desc in zip(self.dataset.records, descriptors_list):
            self._descriptors[record.sample_id] = desc

        # Compute centroid
        if descriptors_list:
            self._centroid = _compute_centroid(descriptors_list)

    def explore(
        self,
        n_samples: int,
        n_seeds: int | None = None,
        perturbation_std: float = 0.05,
        strain_range: tuple[float, float] = (-0.03, 0.03),
        temperature_k: float = 300.0,
    ) -> list[StructureRecord]:
        """Generate boundary candidates by perturbing far-field structures.

        Args:
            n_samples: Total number of candidates to generate.
            n_seeds: Number of boundary seed structures to select from the
                dataset. If None, uses max(3, n_samples // 5).
            perturbation_std: Standard deviation (Å) for atomic position
                Gaussian perturbation.
            strain_range: (min, max) fractional strain to apply to the cell.
            temperature_k: Target temperature for thermal disorder magnitude.
                Used to scale perturbation_std if the given value seems too
                small for the target temperature.

        Returns:
            List of candidate StructureRecord objects.
        """
        if not self.dataset.records or self._centroid is None:
            logger.warning("BoundaryExplorer: empty dataset or no descriptors")
            return []

        if n_seeds is None:
            n_seeds = max(3, n_samples // 5)

        # Select boundary seeds: structures farthest from centroid
        distances = [
            (sid, _euclidean_distance(desc, self._centroid))
            for sid, desc in self._descriptors.items()
        ]
        distances.sort(key=lambda x: x[1], reverse=True)

        seed_ids = [sid for sid, _ in distances[:n_seeds]]
        seed_records = {
            r.sample_id: r for r in self.dataset.records
            if r.sample_id in seed_ids
        }

        candidates: list[StructureRecord] = []
        samples_per_seed = max(1, n_samples // len(seed_ids))

        for seed_id in seed_ids:
            seed = seed_records.get(seed_id)
            if seed is None:
                continue

            for _ in range(samples_per_seed):
                if len(candidates) >= n_samples:
                    break

                perturbed = self._perturb_structure(
                    seed,
                    perturbation_std=perturbation_std,
                    strain_range=strain_range,
                    temperature_k=temperature_k,
                )
                candidates.append(perturbed)

        # Fill any remaining slots by reusing seeds with larger perturbations
        while len(candidates) < n_samples and seed_ids:
            seed_id = random.choice(seed_ids)
            seed = seed_records.get(seed_id)
            if seed is None:
                continue
            perturbed = self._perturb_structure(
                seed,
                perturbation_std=perturbation_std * 1.5,
                strain_range=(strain_range[0] * 1.5, strain_range[1] * 1.5),
                temperature_k=temperature_k,
            )
            candidates.append(perturbed)

        return candidates[:n_samples]

    def _perturb_structure(
        self,
        record: StructureRecord,
        perturbation_std: float,
        strain_range: tuple[float, float],
        temperature_k: float,
    ) -> StructureRecord:
        """Apply a composite perturbation to a structure.

        Perturbation includes:
        1. Atomic position jitter (Gaussian)
        2. Optional cell strain (random direction)
        3. Temperature-scaled displacement magnitude
        """
        import numpy as np

        atoms = record_to_ase(record)

        # 1. Thermal displacement scaled by temperature
        # At 300K, typical RMSD ~0.08 Å for metals; scale by sqrt(T/300)
        temp_scale = math.sqrt(temperature_k / 300.0)
        effective_std = perturbation_std * temp_scale

        positions = atoms.get_positions()
        noise = np.random.normal(0.0, effective_std, positions.shape)
        atoms.set_positions(positions + noise)
        atoms.wrap()

        # 2. Cell strain (random direction)
        strain = random.uniform(*strain_range)
        if random.random() < 0.5:
            # Isotropic volume strain
            cell = atoms.get_cell()
            factor = 1.0 + strain
            new_cell = cell * factor
            atoms.set_cell(new_cell, scale_atoms=True)
        else:
            # Uniaxial strain (random axis)
            axis = random.randint(0, 2)
            cell = atoms.get_cell()
            new_cell = cell.copy()
            new_cell[axis] *= (1.0 + strain)
            atoms.set_cell(new_cell, scale_atoms=True)

        # Create new record with updated positions and metadata
        new_record = ase_to_record(atoms, sample_id=record.sample_id)

        # Copy original metadata and add perturbation info
        metadata = dict(record.metadata) if record.metadata else {}
        metadata["be_seed_id"] = record.sample_id
        metadata["be_perturbation_std"] = round(effective_std, 4)
        metadata["be_strain"] = round(strain, 4)
        metadata["be_temperature_k"] = round(temperature_k, 1)

        # Use dataclass replace to set metadata
        from dataclasses import replace
        new_record = replace(new_record, metadata=metadata)

        return new_record


# ---------------------------------------------------------------------------
# StrainPathGenerator
# ---------------------------------------------------------------------------


class StrainPathGenerator:
    """Generate a sequence of strained structures for transition-state induction.

    Creates structures along deformation paths (tensile, compressive, shear)
    that are designed to cross energy barriers during short AIMD runs.

    Key insight: we do not need to run to fracture. The non-linear region
    of the stress-strain curve (deviation from Hooke's law) is where atomic
    bonds begin to reorder and transition states are densest.

    Example:
        >>> generator = StrainPathGenerator()
        >>> candidates = generator.generate_uniaxial(
        ...     reference_atoms, strain_range=(-0.10, 0.15), n_steps=12
        ... )
    """

    def __init__(self) -> None:
        """Initialize generator."""
        pass

    def generate_uniaxial(
        self,
        reference: StructureRecord,
        strain_range: tuple[float, float] = (-0.10, 0.10),
        n_steps: int = 10,
        axis: int | None = None,
        scale_atoms: bool = True,
    ) -> list[StructureRecord]:
        """Generate uniaxial strain path candidates.

        Args:
            reference: Reference structure.
            strain_range: (min, max) fractional strain.
            n_steps: Number of strain states to generate.
            axis: Strain axis (0=x, 1=y, 2=z). If None, chosen randomly.
            scale_atoms: Whether to scale atomic positions with cell.

        Returns:
            List of strained StructureRecord candidates.
        """
        atoms = record_to_ase(reference)

        if axis is None:
            axis = random.randint(0, 2)

        strains = _linspace(strain_range[0], strain_range[1], n_steps)
        candidates: list[StructureRecord] = []

        for i, strain in enumerate(strains):
            strained_atoms = atoms.copy()
            cell = strained_atoms.get_cell()
            new_cell = cell.copy()
            new_cell[axis] *= (1.0 + strain)
            strained_atoms.set_cell(new_cell, scale_atoms=scale_atoms)

            record = ase_to_record(strained_atoms, sample_id=i)
            metadata = dict(reference.metadata) if reference.metadata else {}
            metadata["sp_type"] = "uniaxial"
            metadata["sp_axis"] = axis
            metadata["sp_strain"] = round(strain, 4)
            metadata["sp_scale_atoms"] = scale_atoms
            from dataclasses import replace
            record = replace(record, metadata=metadata)
            candidates.append(record)

        return candidates

    def generate_biaxial(
        self,
        reference: StructureRecord,
        strain_range: tuple[float, float] = (-0.05, 0.05),
        n_steps: int = 8,
        axes: tuple[int, int] | None = None,
        scale_atoms: bool = True,
    ) -> list[StructureRecord]:
        """Generate biaxial strain path candidates.

        Args:
            reference: Reference structure.
            strain_range: (min, max) fractional strain (applied equally to both axes).
            n_steps: Number of strain states.
            axes: Tuple of two axes to strain. If None, chosen randomly.
            scale_atoms: Whether to scale atomic positions with cell.

        Returns:
            List of strained StructureRecord candidates.
        """
        atoms = record_to_ase(reference)

        if axes is None:
            all_axes = [(0, 1), (1, 2), (0, 2)]
            axes = random.choice(all_axes)

        strains = _linspace(strain_range[0], strain_range[1], n_steps)
        candidates: list[StructureRecord] = []

        for i, strain in enumerate(strains):
            strained_atoms = atoms.copy()
            cell = strained_atoms.get_cell()
            new_cell = cell.copy()
            for axis in axes:
                new_cell[axis] *= (1.0 + strain)
            strained_atoms.set_cell(new_cell, scale_atoms=scale_atoms)

            record = ase_to_record(strained_atoms, sample_id=i)
            metadata = dict(reference.metadata) if reference.metadata else {}
            metadata["sp_type"] = "biaxial"
            metadata["sp_axes"] = f"{axes[0]},{axes[1]}"
            metadata["sp_strain"] = round(strain, 4)
            metadata["sp_scale_atoms"] = scale_atoms
            from dataclasses import replace
            record = replace(record, metadata=metadata)
            candidates.append(record)

        return candidates

    def generate_shear(
        self,
        reference: StructureRecord,
        shear_range: tuple[float, float] = (-0.10, 0.10),
        n_steps: int = 10,
        plane: tuple[int, int] = (0, 1),
        scale_atoms: bool = True,
    ) -> list[StructureRecord]:
        """Generate shear strain path candidates.

        Applies simple shear γ to the cell matrix:
            cell[i,j] += γ * cell[j,j]  (for i != j)

        Args:
            reference: Reference structure.
            shear_range: (min, max) shear strain γ.
            n_steps: Number of shear states.
            plane: (direction, normal) indices defining shear plane.
            scale_atoms: Whether to scale atomic positions with cell.

        Returns:
            List of sheared StructureRecord candidates.
        """
        atoms = record_to_ase(reference)
        dir_idx, norm_idx = plane

        gammas = _linspace(shear_range[0], shear_range[1], n_steps)
        candidates: list[StructureRecord] = []

        for i, gamma in enumerate(gammas):
            sheared_atoms = atoms.copy()
            cell = sheared_atoms.get_cell()
            new_cell = cell.copy()
            # Simple shear: displacement in dir_idx proportional to norm_idx coordinate
            new_cell[dir_idx] += gamma * new_cell[norm_idx]
            sheared_atoms.set_cell(new_cell, scale_atoms=scale_atoms)

            record = ase_to_record(sheared_atoms, sample_id=i)
            metadata = dict(reference.metadata) if reference.metadata else {}
            metadata["sp_type"] = "shear"
            metadata["sp_plane"] = f"{plane[0]},{plane[1]}"
            metadata["sp_gamma"] = round(gamma, 4)
            metadata["sp_scale_atoms"] = scale_atoms
            from dataclasses import replace
            record = replace(record, metadata=metadata)
            candidates.append(record)

        return candidates

    def generate_hydrostatic(
        self,
        reference: StructureRecord,
        strain_range: tuple[float, float] = (-0.10, 0.10),
        n_steps: int = 10,
        scale_atoms: bool = True,
    ) -> list[StructureRecord]:
        """Generate hydrostatic (isotropic volume) strain path candidates.

        Args:
            reference: Reference structure.
            strain_range: (min, max) volumetric strain (fractional change in volume).
            n_steps: Number of strain states.
            scale_atoms: Whether to scale atomic positions with cell.

        Returns:
            List of volumetrically strained StructureRecord candidates.
        """
        atoms = record_to_ase(reference)

        # Volumetric strain ε_v → linear strain ε = (1 + ε_v)^(1/3) - 1
        eps_volumes = _linspace(strain_range[0], strain_range[1], n_steps)
        candidates: list[StructureRecord] = []

        for i, eps_v in enumerate(eps_volumes):
            strained_atoms = atoms.copy()
            linear_factor = (1.0 + eps_v) ** (1.0 / 3.0)
            cell = strained_atoms.get_cell()
            new_cell = cell * linear_factor
            strained_atoms.set_cell(new_cell, scale_atoms=scale_atoms)

            record = ase_to_record(strained_atoms, sample_id=i)
            metadata = dict(reference.metadata) if reference.metadata else {}
            metadata["sp_type"] = "hydrostatic"
            metadata["sp_volumetric_strain"] = round(eps_v, 4)
            metadata["sp_linear_factor"] = round(linear_factor, 4)
            metadata["sp_scale_atoms"] = scale_atoms
            from dataclasses import replace
            record = replace(record, metadata=metadata)
            candidates.append(record)

        return candidates


# ---------------------------------------------------------------------------
# Unified candidate generation with strategy switching
# ---------------------------------------------------------------------------


class CandidateGenerator:
    """Unified candidate generator supporting multiple strategies.

    Replaces the monolithic generate_structures() with a strategy-pattern
    design that can switch between:
        - "random": Original PyXtal/ASE random generation
        - "boundary": Dataset-aware boundary exploration
        - "strain": Strain-path generation for transition-state induction
        - "composite": Combined strategy (boundary + strain)

    Example:
        >>> gen = CandidateGenerator(dataset=ref_dataset)
        >>> candidates = gen.generate(
        ...     strategy="boundary",
        ...     n_samples=20,
        ...     perturbation_std=0.05,
        ... )
    """

    def __init__(
        self,
        dataset: DatasetBundle | None = None,
        adapter: DescriptorAdapter | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            dataset: Reference dataset for dataset-aware strategies.
                Required for "boundary" and "composite" strategies.
            adapter: Descriptor adapter. Defaults to SOAPDescriptorAdapter.
        """
        self.dataset = dataset
        self.adapter = adapter or SOAPDescriptorAdapter()
        self._boundary_explorer: BoundaryExplorer | None = None
        self._strain_generator: StrainPathGenerator | None = None

        if dataset is not None:
            self._boundary_explorer = BoundaryExplorer(dataset, adapter=self.adapter)
        self._strain_generator = StrainPathGenerator()

    def generate(
        self,
        strategy: str,
        n_samples: int,
        **kwargs: object,
    ) -> list[StructureRecord]:
        """Generate candidates using the specified strategy.

        Args:
            strategy: Generation strategy name.
                - "random": PyXtal/ASE random generation (requires
                  system_type, composition, etc. via kwargs)
                - "boundary": Perturb far-field structures from dataset
                - "strain_uniaxial", "strain_biaxial", "strain_shear",
                  "strain_hydrostatic": Strain paths
                - "composite": boundary + strain_uniaxial mixed
            n_samples: Number of candidates to generate.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of candidate StructureRecord objects.

        Raises:
            ValueError: If strategy is unknown or required params missing.
        """
        if strategy == "random":
            return self._generate_random(n_samples, **kwargs)

        elif strategy == "boundary":
            if self._boundary_explorer is None:
                raise ValueError("'boundary' strategy requires a reference dataset")
            return self._boundary_explorer.explore(
                n_samples=n_samples,
                perturbation_std=kwargs.get("perturbation_std", 0.05),
                strain_range=kwargs.get("strain_range", (-0.03, 0.03)),
                temperature_k=kwargs.get("temperature_k", 300.0),
            )

        elif strategy == "strain_uniaxial":
            return self._generate_strain(
                "uniaxial", n_samples, **kwargs
            )

        elif strategy == "strain_biaxial":
            return self._generate_strain(
                "biaxial", n_samples, **kwargs
            )

        elif strategy == "strain_shear":
            return self._generate_strain(
                "shear", n_samples, **kwargs
            )

        elif strategy == "strain_hydrostatic":
            return self._generate_strain(
                "hydrostatic", n_samples, **kwargs
            )

        elif strategy == "composite":
            return self._generate_composite(n_samples, **kwargs)

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Choose from: "
                f"random, boundary, strain_uniaxial, strain_biaxial, "
                f"strain_shear, strain_hydrostatic, composite"
            )

    def _generate_random(self, n_samples: int, **kwargs: object) -> list[StructureRecord]:
        """Delegate to original generate_structures."""
        from atom_dataset_index.sampling.generator import generate_structures
        from atom_dataset_index.sampling.config import SamplingConfig

        system_type = kwargs.get("system_type", "bulk")
        config = kwargs.get("config")
        if config is None:
            config = SamplingConfig(system_type=system_type)

        return generate_structures(
            system_type=system_type,
            n_samples=n_samples,
            config=config,
            composition=kwargs.get("composition"),
            seed_path=kwargs.get("seed_path"),
            verbose=kwargs.get("verbose", False),
        )

    def _generate_strain(
        self,
        strain_type: str,
        n_samples: int,
        **kwargs: object,
    ) -> list[StructureRecord]:
        """Generate strain-path candidates."""
        if self._strain_generator is None:
            raise RuntimeError("Strain generator not initialized")

        reference = kwargs.get("reference")
        if reference is None:
            if self.dataset is None or not self.dataset.records:
                raise ValueError("'reference' or 'dataset' required for strain strategies")
            # Use lowest-energy structure as reference
            reference = min(
                self.dataset.records,
                key=lambda r: r.energy if r.energy is not None else 0.0,
            )

        # Default to n_samples so the user gets exactly the requested count
        n_steps = kwargs.get("n_steps", n_samples)

        if strain_type == "uniaxial":
            return self._strain_generator.generate_uniaxial(
                reference=reference,
                strain_range=kwargs.get("strain_range", (-0.10, 0.10)),
                n_steps=n_steps,
                axis=kwargs.get("axis"),
                scale_atoms=kwargs.get("scale_atoms", True),
            )
        elif strain_type == "biaxial":
            return self._strain_generator.generate_biaxial(
                reference=reference,
                strain_range=kwargs.get("strain_range", (-0.05, 0.05)),
                n_steps=n_steps,
                axes=kwargs.get("axes"),
                scale_atoms=kwargs.get("scale_atoms", True),
            )
        elif strain_type == "shear":
            return self._strain_generator.generate_shear(
                reference=reference,
                shear_range=kwargs.get("shear_range", (-0.10, 0.10)),
                n_steps=n_steps,
                plane=kwargs.get("plane", (0, 1)),
                scale_atoms=kwargs.get("scale_atoms", True),
            )
        elif strain_type == "hydrostatic":
            return self._strain_generator.generate_hydrostatic(
                reference=reference,
                strain_range=kwargs.get("strain_range", (-0.10, 0.10)),
                n_steps=n_steps,
                scale_atoms=kwargs.get("scale_atoms", True),
            )
        else:
            raise ValueError(f"Unknown strain type: {strain_type}")

    def _generate_composite(self, n_samples: int, **kwargs: object) -> list[StructureRecord]:
        """Combine boundary exploration and strain paths.

        Splits n_samples 50/50 between boundary perturbations and strain paths.
        """
        if self._boundary_explorer is None or self._strain_generator is None:
            raise ValueError("'composite' strategy requires a reference dataset")

        n_boundary = n_samples // 2
        n_strain = n_samples - n_boundary

        boundary_candidates = self._boundary_explorer.explore(
            n_samples=n_boundary,
            perturbation_std=kwargs.get("perturbation_std", 0.05),
        )

        strain_candidates = self._generate_strain(
            "uniaxial",
            n_samples=n_strain,
            **kwargs,
        )

        return boundary_candidates + strain_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_centroid(descriptors: list[tuple[float, ...]]) -> tuple[float, ...]:
    """Compute centroid (mean vector) of a set of descriptors."""
    if not descriptors:
        return ()
    dim = max(len(d) for d in descriptors)
    sums = [0.0] * dim
    for desc in descriptors:
        for i, val in enumerate(desc):
            sums[i] += val
    n = len(descriptors)
    return tuple(s / n for s in sums)


def _euclidean_distance(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    """Compute Euclidean distance between two descriptor vectors."""
    dim = max(len(left), len(right))
    total = 0.0
    for i in range(dim):
        lv = left[i] if i < len(left) else 0.0
        rv = right[i] if i < len(right) else 0.0
        total += (lv - rv) ** 2
    return math.sqrt(total)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    """Generate evenly spaced numbers over a range."""
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]
