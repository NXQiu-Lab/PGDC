"""Reference structure selection logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from atom_dataset_index.domain import DatasetBundle, StructureRecord
from atom_dataset_index.metric_core.descriptors import DescriptorAdapter, SOAPDescriptorAdapter
from atom_dataset_index.metric_core.statistics import quantiles

if TYPE_CHECKING:
    import numpy as np


class ReferenceStructureSelector:
    """Choose a reference structure for distortion metrics."""

    def select(
        self,
        dataset: DatasetBundle,
        strategy: str = "min_energy",
    ) -> StructureRecord | None:
        if not dataset.records:
            return None
        if strategy == "first":
            return dataset.records[0]
        if strategy == "min_energy":
            with_energy = [record for record in dataset.records if record.energy is not None]
            if with_energy:
                return min(with_energy, key=lambda record: float(record.energy))
            return dataset.records[0]
        raise ValueError(f"Unsupported reference selection strategy: {strategy}")


class ReferenceSelectionResult:
    """Unified reference selection payload reused by multiple metrics."""

    def __init__(
        self,
        descriptor_backend: str,
        all_reference_sample_ids: tuple[int, ...],
        sample_descriptors: dict[int, tuple[float, ...]],
        reference_assignments: dict[int, int],
        reference_distances: dict[int, float],
        legacy_anchor_sample_ids: tuple[int, ...] = (),
        reference_sample_id: int | None = None,
        stable_reference_sample_ids: tuple[int, ...] = (),
        diversity_reference_sample_ids: tuple[int, ...] = (),
    ) -> None:
        self.descriptor_backend = descriptor_backend
        self.all_reference_sample_ids = all_reference_sample_ids
        self.sample_descriptors = sample_descriptors
        self.reference_assignments = reference_assignments
        self.reference_distances = reference_distances
        self.legacy_anchor_sample_ids = legacy_anchor_sample_ids
        self.reference_sample_id = reference_sample_id
        self.stable_reference_sample_ids = stable_reference_sample_ids
        self.diversity_reference_sample_ids = diversity_reference_sample_ids

    # Backward compatibility aliases
    @property
    def prototype_sample_ids(self) -> tuple[int, ...]:
        return self.all_reference_sample_ids

    @property
    def prototype_assignments(self) -> dict[int, int]:
        return self.reference_assignments

    @property
    def prototype_distances(self) -> dict[int, float]:
        return self.reference_distances

    @property
    def anchor_sample_ids(self) -> tuple[int, ...]:
        return self.legacy_anchor_sample_ids


class PrototypeStructureSelector:
    """Select unified reference structures (stability + diversity) for pool-aware profiling."""

    def __init__(self, adapter: DescriptorAdapter | None = None) -> None:
        self.adapter = adapter or SOAPDescriptorAdapter()

    def select(
        self,
        dataset: DatasetBundle,
        *,
        strategy: str = "low_energy_force_filtered_medoids",
        precomputed_descriptors: dict[int, tuple[float, ...]] | None = None,
    ) -> ReferenceSelectionResult:
        if not dataset.records:
            return ReferenceSelectionResult(
                descriptor_backend=self._backend_name(),
                all_reference_sample_ids=(),
                sample_descriptors={},
                reference_assignments={},
                reference_distances={},
            )
        if strategy != "low_energy_force_filtered_medoids":
            raise ValueError(f"Unsupported prototype selection strategy: {strategy}")

        return self.select_references(dataset, precomputed_descriptors=precomputed_descriptors)

    def select_references(
        self,
        dataset: DatasetBundle,
        precomputed_descriptors: dict[int, tuple[float, ...]] | None = None,
        min_stable_refs: int = 3,
        max_stable_refs: int = 8,
        min_diversity_refs: int = 2,
        max_diversity_refs: int = 5,
    ) -> ReferenceSelectionResult:
        """Select stability and diversity references for a pool."""
        sample_ids = [record.sample_id for record in dataset.records]
        sample_id_to_index = {sid: i for i, sid in enumerate(sample_ids)}

        if precomputed_descriptors is not None:
            descriptors_list = [precomputed_descriptors.get(sid, ()) for sid in sample_ids]
            sample_descriptors = {sid: precomputed_descriptors[sid] for sid in sample_ids if sid in precomputed_descriptors}
        else:
            descriptors_list = self.adapter.encode_batch(list(dataset.records))
            sample_descriptors = {sample_ids[i]: descriptors_list[i] for i in range(len(sample_ids))}

        import numpy as np

        max_dim = max((len(d) for d in descriptors_list), default=0)
        descriptor_matrix = np.zeros((len(descriptors_list), max_dim), dtype=np.float64)
        for i, d in enumerate(descriptors_list):
            descriptor_matrix[i, : len(d)] = d

        energy_per_atom = {
            record.sample_id: float(record.energy) / max(record.natoms, 1)
            for record in dataset.records
            if record.energy is not None
        }
        force_mean_norm = {
            record.sample_id: self._force_mean_norm(record)
            for record in dataset.records
        }

        # Primary reference: min energy
        reference_sample_id: int | None = None
        if energy_per_atom:
            reference_sample_id = min(energy_per_atom, key=energy_per_atom.get)  # type: ignore[arg-type]

        # Stability references
        stable_refs = self._select_stable_references(
            sample_ids,
            energy_per_atom,
            force_mean_norm,
            min_refs=min_stable_refs,
            max_refs=max_stable_refs,
        )

        # Candidate filtering for diversity references (same as old logic)
        if energy_per_atom:
            minimum_energy = min(energy_per_atom.values())
            delta_energy = {
                sample_id: value - minimum_energy
                for sample_id, value in energy_per_atom.items()
            }
        else:
            delta_energy = {record.sample_id: 0.0 for record in dataset.records}

        candidate_ids = self._select_candidates(
            tuple(record.sample_id for record in dataset.records),
            delta_energy,
            force_mean_norm,
        )

        # Phase stratification for medoid allocation
        phase_labels = self._cluster_phases(dataset)
        n_phases = int(phase_labels.max()) + 1 if phase_labels is not None else 1

        prototype_count = self._prototype_count(len(dataset.records))

        candidate_indices = [sample_id_to_index[sid] for sid in candidate_ids]
        candidate_matrix = descriptor_matrix[candidate_indices]

        if phase_labels is not None and n_phases > 1:
            diversity_refs = self._select_medoids_stratified(
                candidate_ids, candidate_matrix, phase_labels, candidate_indices, prototype_count
            )
        else:
            diversity_refs = self._select_medoids(candidate_ids, candidate_matrix, prototype_count)

        if not diversity_refs:
            diversity_refs = (dataset.records[0].sample_id,)

        # Clip diversity refs to [min_diversity_refs, max_diversity_refs]
        diversity_refs = diversity_refs[:max_diversity_refs]
        if len(diversity_refs) < min_diversity_refs:
            # Pad with additional medoids from full dataset if needed
            all_ids = tuple(record.sample_id for record in dataset.records)
            all_matrix = descriptor_matrix
            extra = self._select_medoids(all_ids, all_matrix, min_diversity_refs)
            combined = list(diversity_refs) + [e for e in extra if e not in diversity_refs]
            diversity_refs = tuple(combined[:max_diversity_refs])

        # Deduplicate diversity refs against stable refs
        filtered_diversity: list[int] = []
        stable_set = set(stable_refs)
        for sid in diversity_refs:
            if sid not in stable_set:
                filtered_diversity.append(sid)
        # If too few after dedup, pick more medoids
        if len(filtered_diversity) < min_diversity_refs:
            all_ids = tuple(record.sample_id for record in dataset.records)
            all_matrix = descriptor_matrix
            extra = self._select_medoids(all_ids, all_matrix, min_diversity_refs + len(stable_set))
            for sid in extra:
                if sid not in stable_set and sid not in filtered_diversity:
                    filtered_diversity.append(sid)
                if len(filtered_diversity) >= min_diversity_refs:
                    break
        diversity_refs = tuple(filtered_diversity[:max_diversity_refs])

        # Build unified reference set
        all_refs = tuple(dict.fromkeys((reference_sample_id,) + stable_refs + diversity_refs))
        # Remove None if present
        all_refs = tuple(sid for sid in all_refs if sid is not None)

        all_ref_indices = [sample_id_to_index[sid] for sid in all_refs]
        assignments, distances = self._assign_prototypes(
            descriptor_matrix, all_ref_indices, sample_ids
        )

        return ReferenceSelectionResult(
            descriptor_backend=self._backend_name(),
            all_reference_sample_ids=all_refs,
            sample_descriptors=sample_descriptors,
            reference_assignments=assignments,
            reference_distances=distances,
            legacy_anchor_sample_ids=stable_refs,
            reference_sample_id=reference_sample_id,
            stable_reference_sample_ids=stable_refs,
            diversity_reference_sample_ids=diversity_refs,
        )

    @staticmethod
    def _compute_pareto_frontier(energy: "np.ndarray", force: "np.ndarray") -> "np.ndarray":
        """O(n log n) Pareto frontier (lower-left is better)."""
        import numpy as np

        n = len(energy)
        order = np.lexsort((force, energy))
        sorted_e = energy[order]
        sorted_f = force[order]

        is_pareto = np.zeros(n, dtype=bool)
        min_force_so_far = np.inf
        for i in range(n):
            if sorted_f[i] <= min_force_so_far:
                is_pareto[order[i]] = True
                min_force_so_far = sorted_f[i]
        return is_pareto

    def _select_stable_references(
        self,
        sample_ids: list[int],
        energy_per_atom: dict[int, float],
        force_mean_norm: dict[int, float],
        min_refs: int,
        max_refs: int,
    ) -> tuple[int, ...]:
        """Select stability references from the Pareto frontier (energy, force)."""
        import numpy as np

        e = np.array([energy_per_atom.get(sid, 0.0) for sid in sample_ids])
        f = np.array([force_mean_norm.get(sid, 0.0) for sid in sample_ids])
        pareto_mask = self._compute_pareto_frontier(e, f)
        pareto_indices = np.where(pareto_mask)[0]
        sorted_pareto = sorted(pareto_indices, key=lambda i: e[i])

        n_pareto = len(sorted_pareto)
        M = min(max_refs, max(min_refs, n_pareto))
        if len(sorted_pareto) >= M:
            chosen = sorted_pareto[:M]
        else:
            non_pareto = [i for i in range(len(sample_ids)) if not pareto_mask[i]]
            non_pareto_sorted = sorted(non_pareto, key=lambda i: e[i])
            chosen = sorted_pareto + non_pareto_sorted[: M - len(sorted_pareto)]
        return tuple(sample_ids[i] for i in chosen)

    def _cluster_phases(self, dataset: DatasetBundle) -> "np.ndarray | None":
        """Cluster records by composition and cell volume. Returns phase labels or None if sklearn unavailable."""
        import numpy as np
        from ase import Atoms

        try:
            from sklearn.cluster import KMeans
        except Exception:
            return None

        if not dataset.records:
            return None

        # Dynamic element extraction
        present_elements = sorted({z for r in dataset.records for z in r.atomic_numbers})

        features = []
        for r in dataset.records:
            counts = {z: 0 for z in present_elements}
            for z in r.atomic_numbers:
                counts[z] = counts.get(z, 0) + 1
            total = max(sum(counts.values()), 1)
            frac = np.array([counts[z] / total for z in present_elements], dtype=float)
            if r.cell is not None:
                atoms = Atoms(
                    numbers=list(r.atomic_numbers),
                    positions=list(r.positions),
                    cell=list(r.cell),
                    pbc=list(r.pbc),
                )
                vol = atoms.get_volume()
            else:
                vol = 0.0
            features.append(np.concatenate([frac, [vol]]))
        features = np.array(features)

        vol_mean = features[:, -1].mean()
        vol_std = features[:, -1].std() + 1e-6
        features[:, -1] = (features[:, -1] - vol_mean) / vol_std

        n_clusters = min(12, max(4, len(dataset.records) // 3000), len(dataset.records))
        if n_clusters <= 1:
            return None
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(features)
        return labels

    def _select_candidates(
        self,
        sample_ids: tuple[int, ...],
        delta_energy: dict[int, float],
        force_mean_norm: dict[int, float],
    ) -> tuple[int, ...]:
        if not sample_ids:
            return ()
        energy_cutoff = quantiles(tuple(delta_energy.get(sample_id, 0.0) for sample_id in sample_ids), (0.25,))["0.25"]
        force_cutoff = quantiles(tuple(force_mean_norm.get(sample_id, 0.0) for sample_id in sample_ids), (0.5,))["0.5"]
        candidates = tuple(
            sample_id
            for sample_id in sample_ids
            if delta_energy.get(sample_id, 0.0) <= energy_cutoff
            and force_mean_norm.get(sample_id, 0.0) <= force_cutoff
        )
        if candidates:
            return candidates
        fallback_energy_cutoff = quantiles(
            tuple(delta_energy.get(sample_id, 0.0) for sample_id in sample_ids),
            (0.5,),
        )["0.5"]
        return tuple(
            sample_id for sample_id in sample_ids if delta_energy.get(sample_id, 0.0) <= fallback_energy_cutoff
        ) or sample_ids

    def _select_medoids_stratified(
        self,
        candidate_ids: tuple[int, ...],
        candidate_matrix: "np.ndarray",
        phase_labels: "np.ndarray",
        candidate_indices: list[int],
        prototype_count: int,
    ) -> tuple[int, ...]:
        """Allocate prototypes across phases and run per-phase medoid selection."""
        import numpy as np

        n_phases = int(phase_labels.max()) + 1
        phase_counts = {
            p: int((phase_labels[candidate_indices] == p).sum())
            for p in range(n_phases)
        }
        active_phases = [p for p, cnt in phase_counts.items() if cnt > 0]
        n_active = len(active_phases)

        if n_active == 0:
            return ()

        # Allocate quotas: at least 1 per active phase, then proportional
        quotas: dict[int, int] = {p: 1 for p in active_phases}
        remaining = prototype_count - sum(quotas.values())

        if remaining > 0:
            total_candidates = sum(phase_counts[p] for p in active_phases)
            if total_candidates > 0:
                fractions = {
                    p: remaining * phase_counts[p] / total_candidates for p in active_phases
                }
                floors = {p: int(fractions[p]) for p in active_phases}
                for p in active_phases:
                    quotas[p] += floors[p]
                remaining2 = remaining - sum(floors.values())
                sorted_by_frac = sorted(active_phases, key=lambda p: fractions[p] - floors[p], reverse=True)
                for i in range(remaining2):
                    quotas[sorted_by_frac[i]] += 1
        elif remaining < 0:
            while sum(quotas.values()) > prototype_count:
                reducible = [p for p in active_phases if quotas[p] > 1]
                if not reducible:
                    break
                p = min(reducible, key=lambda p: phase_counts[p])
                quotas[p] -= 1

        all_prototypes: list[int] = []
        for p in active_phases:
            q = quotas[p]
            local_indices = [i for i, idx in enumerate(candidate_indices) if phase_labels[idx] == p]
            if not local_indices:
                continue
            phase_matrix = candidate_matrix[local_indices]
            phase_ids = tuple(candidate_ids[i] for i in local_indices)
            selected = self._select_medoids(phase_ids, phase_matrix, q)
            all_prototypes.extend(selected)

        return tuple(all_prototypes)

    def _select_medoids(
        self,
        candidate_ids: tuple[int, ...],
        candidate_matrix: "np.ndarray",
        prototype_count: int,
    ) -> tuple[int, ...]:
        if not candidate_ids:
            return ()
        if len(candidate_ids) <= prototype_count:
            return candidate_ids

        import numpy as np

        full_dist = self._distance_matrix(candidate_matrix, candidate_matrix)
        mean_dists = (full_dist.sum(axis=1) - np.diag(full_dist)) / (full_dist.shape[1] - 1)
        first_idx = int(np.argmin(mean_dists))
        selected_indices = [first_idx]
        remaining = set(range(len(candidate_ids)))
        remaining.discard(first_idx)

        while remaining and len(selected_indices) < prototype_count:
            rem_list = list(remaining)
            rem_matrix = candidate_matrix[rem_list]
            sel_matrix = candidate_matrix[selected_indices]
            sub_dist = self._distance_matrix(rem_matrix, sel_matrix)
            min_to_selected = sub_dist.min(axis=1)
            next_rel_idx = int(np.argmax(min_to_selected))
            next_idx = rem_list[next_rel_idx]
            selected_indices.append(next_idx)
            remaining.discard(next_idx)

        return tuple(candidate_ids[i] for i in selected_indices)

    def _assign_prototypes(
        self,
        descriptor_matrix: "np.ndarray",
        prototype_indices: list[int],
        sample_ids: list[int],
    ) -> tuple[dict[int, int], dict[int, float]]:
        import numpy as np

        prot_matrix = descriptor_matrix[prototype_indices]
        dists = self._distance_matrix(descriptor_matrix, prot_matrix)
        closest_rel = np.argmin(dists, axis=1)
        closest_indices = [prototype_indices[i] for i in closest_rel]
        assignments = {
            sample_ids[i]: sample_ids[closest_indices[i]] for i in range(len(sample_ids))
        }
        distances = {
            sample_ids[i]: float(dists[i, closest_rel[i]]) for i in range(len(sample_ids))
        }
        return assignments, distances

    def _distance_matrix(
        self,
        left: "np.ndarray",
        right: "np.ndarray",
    ) -> "np.ndarray":
        try:
            from scipy.spatial.distance import cdist
            return cdist(left, right, metric="euclidean")
        except Exception:
            import numpy as np

            result = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
            for i in range(left.shape[0]):
                result[i, :] = np.linalg.norm(left[i, :] - right, axis=1)
            return result

    @staticmethod
    def _prototype_count(record_count: int) -> int:
        return max(5, int(record_count ** 0.25))

    @staticmethod
    def _force_mean_norm(record: StructureRecord) -> float:
        if not record.forces:
            return 0.0
        norms = [
            math.sqrt(sum(component * component for component in vector))
            for vector in record.forces
        ]
        return sum(norms) / len(norms)

    def _backend_name(self) -> str:
        return getattr(self.adapter, "backend_name", self.adapter.__class__.__name__.lower())
