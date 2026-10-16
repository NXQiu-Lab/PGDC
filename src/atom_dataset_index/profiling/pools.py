"""Pool building logic for pool-aware dataset profiling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from ase.data import chemical_symbols

from atom_dataset_index.domain import DatasetBundle, StructureRecord


@dataclass(slots=True)
class DatasetPool:
    """A flattened pool ready for independent metric evaluation."""

    pool_id: str
    records: tuple[StructureRecord, ...]
    pool_type: Literal["element", "subpool", "merged", "misc"]
    parent_pool_id: str | None = None


def _normalized_formula(atomic_numbers: tuple[int, ...]) -> str:
    """Return a canonical formula string, e.g. Cs8Sn8I24."""
    counts = Counter(atomic_numbers)
    parts = []
    for z in sorted(counts.keys()):
        n = counts[z]
        symbol = chemical_symbols[z] if 0 <= z < len(chemical_symbols) else f"E{z}"
        parts.append(f"{symbol}{n}")
    return "".join(parts)


class PoolBuilder:
    """Build evaluation pools via two-level stratification with small-pool merging."""

    def __init__(
        self,
        pool_fraction: float = 0.005,
        subpool_fraction: float = 0.002,
        min_abs_pool: int = 15,
        max_abs_pool: int = 500,
        min_abs_subpool: int = 10,
        max_abs_subpool: int = 200,
        jaccard_threshold: float = 0.5,
    ) -> None:
        self.pool_fraction = pool_fraction
        self.subpool_fraction = subpool_fraction
        self.min_abs_pool = min_abs_pool
        self.max_abs_pool = max_abs_pool
        self.min_abs_subpool = min_abs_subpool
        self.max_abs_subpool = max_abs_subpool
        self.jaccard_threshold = jaccard_threshold

    def build_pools(self, dataset: DatasetBundle) -> list[DatasetPool]:
        """Build final flat pools from a dataset."""
        if not dataset.records:
            return []

        total_records = len(dataset.records)
        min_pool_size = self._compute_min_pool_size(total_records)
        min_subpool_size = self._compute_min_subpool_size(total_records)

        # Step 1: element-composition pools
        element_pools = self._split_by_composition(dataset)

        # Step 2: classify into large / medium / small
        large: list[DatasetPool] = []
        medium: list[DatasetPool] = []
        small: list[DatasetPool] = []
        for pool in element_pools:
            size = len(pool.records)
            if size >= min_pool_size:
                large.append(pool)
            elif size >= self.min_abs_pool:
                medium.append(pool)
            else:
                small.append(pool)

        # Step 3: merge small pools
        merged = self._merge_small_pools(small, element_pools)

        # Step 4: cluster large pools into subpools
        final_pools: list[DatasetPool] = []
        for pool in large:
            subpools, unclassified = self._cluster_subpools(
                pool, min_subpool_size=min_subpool_size
            )
            final_pools.extend(subpools)
            if unclassified:
                final_pools.append(
                    DatasetPool(
                        pool_id=f"{pool.pool_id}::unclassified",
                        records=tuple(unclassified),
                        pool_type="subpool",
                        parent_pool_id=pool.pool_id,
                    )
                )

        for pool in medium + merged:
            final_pools.append(pool)

        return final_pools

    @staticmethod
    def _compute_min_pool_size(
        total_records: int,
        fraction: float = 0.005,
        min_abs: int = 30,
        max_abs: int = 500,
    ) -> int:
        return max(min_abs, min(max_abs, int(total_records * fraction)))

    @staticmethod
    def _compute_min_subpool_size(
        total_records: int,
        fraction: float = 0.002,
        min_abs: int = 20,
        max_abs: int = 200,
    ) -> int:
        return max(min_abs, min(max_abs, int(total_records * fraction)))

    def _split_by_composition(self, dataset: DatasetBundle) -> list[DatasetPool]:
        buckets: dict[str, list[StructureRecord]] = {}
        for record in dataset.records:
            formula = _normalized_formula(record.atomic_numbers)
            buckets.setdefault(formula, []).append(record)
        pools = []
        for formula, records in buckets.items():
            pools.append(
                DatasetPool(
                    pool_id=formula,
                    records=tuple(records),
                    pool_type="element",
                )
            )
        # deterministic order
        pools.sort(key=lambda p: (len(p.records), p.pool_id), reverse=True)
        return pools

    def _merge_small_pools(
        self,
        small_pools: list[DatasetPool],
        all_element_pools: list[DatasetPool],
    ) -> list[DatasetPool]:
        if not small_pools:
            return []

        # Build element vectors over the union of all elements in *all* pools
        all_elements = sorted(
            {
                z
                for pool in all_element_pools
                for record in pool.records
                for z in record.atomic_numbers
            }
        )

        def pool_element_vector(pool: DatasetPool) -> list[float]:
            total_atoms = sum(record.natoms for record in pool.records)
            counts: dict[int, float] = {}
            for record in pool.records:
                for z in record.atomic_numbers:
                    counts[z] = counts.get(z, 0.0) + 1.0
            if total_atoms == 0:
                return [0.0] * len(all_elements)
            return [counts.get(z, 0.0) / total_atoms for z in all_elements]

        def jaccard(left_set: set[int], right_set: set[int]) -> float:
            inter = len(left_set & right_set)
            union = len(left_set | right_set)
            return inter / union if union > 0 else 0.0

        def euclidean(left: list[float], right: list[float]) -> float:
            return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5

        pool_sets = {
            pool.pool_id: {z for record in pool.records for z in record.atomic_numbers}
            for pool in small_pools
        }
        pool_vectors = {pool.pool_id: pool_element_vector(pool) for pool in small_pools}

        # Inverted index: element -> list of pool ids
        element_to_pools: dict[int, list[str]] = {}
        for pool in small_pools:
            for z in pool_sets[pool.pool_id]:
                element_to_pools.setdefault(z, []).append(pool.pool_id)

        merged_out: list[DatasetPool] = []
        unmerged_ids = {p.pool_id for p in small_pools}
        pool_lookup = {p.pool_id: p for p in small_pools}

        # Greedy nearest-neighbor merging
        while unmerged_ids:
            # Pick the smallest remaining pool
            current_id = min(
                unmerged_ids,
                key=lambda pid: len(pool_lookup[pid].records),
            )
            unmerged_ids.remove(current_id)
            current_set = pool_sets[current_id]
            current_vec = pool_vectors[current_id]

            # Candidate neighbors via inverted index
            candidates: set[str] = set()
            for z in current_set:
                candidates.update(element_to_pools.get(z, []))
            candidates &= unmerged_ids

            best_neighbor: str | None = None
            best_distance = float("inf")
            for cid in candidates:
                if jaccard(current_set, pool_sets[cid]) < self.jaccard_threshold:
                    continue
                dist = euclidean(current_vec, pool_vectors[cid])
                if dist < best_distance:
                    best_distance = dist
                    best_neighbor = cid

            if best_neighbor is not None:
                # Merge current into neighbor
                neighbor_pool = pool_lookup[best_neighbor]
                current_pool = pool_lookup[current_id]
                new_records = neighbor_pool.records + current_pool.records
                new_id = f"merged::{min(neighbor_pool.pool_id, current_pool.pool_id)}"
                merged_pool = DatasetPool(
                    pool_id=new_id,
                    records=new_records,
                    pool_type="merged",
                )
                # Replace neighbor in lookup
                pool_lookup[new_id] = merged_pool
                del pool_lookup[best_neighbor]
                del pool_lookup[current_id]
                # Update tracking structures
                unmerged_ids.remove(best_neighbor)
                pool_sets[new_id] = {z for record in new_records for z in record.atomic_numbers}
                pool_vectors[new_id] = pool_element_vector(merged_pool)
                for z in pool_sets[new_id]:
                    lst = element_to_pools.setdefault(z, [])
                    # Replace old ids with new id in inverted index
                    if best_neighbor in lst:
                        lst[lst.index(best_neighbor)] = new_id
                    if current_id in lst:
                        lst[lst.index(current_id)] = new_id
                    if new_id not in lst:
                        lst.append(new_id)
            else:
                # Cannot merge; keep as standalone (will go to misc later)
                merged_out.append(
                    DatasetPool(
                        pool_id=current_id,
                        records=pool_lookup[current_id].records,
                        pool_type="misc",
                    )
                )
                del pool_lookup[current_id]

        # Add any remaining merged pools from lookup
        for pid, pool in pool_lookup.items():
            if pid not in {p.pool_id for p in merged_out}:
                merged_out.append(pool)

        # Group any remaining standalone misc pools into a single misc_pool
        misc_pools = [p for p in merged_out if p.pool_type == "misc"]
        non_misc = [p for p in merged_out if p.pool_type != "misc"]
        if len(misc_pools) > 1:
            all_misc_records = tuple(r for p in misc_pools for r in p.records)
            non_misc.append(
                DatasetPool(
                    pool_id="misc_pool",
                    records=all_misc_records,
                    pool_type="misc",
                )
            )
        elif len(misc_pools) == 1:
            non_misc.append(misc_pools[0])

        return non_misc

    def _cluster_subpools(
        self,
        pool: DatasetPool,
        min_subpool_size: int,
    ) -> tuple[list[DatasetPool], list[StructureRecord]]:
        """Run KMeans clustering inside a large element pool.

        Returns subpools that meet min_subpool_size and unclassified records.
        """
        import numpy as np
        from ase import Atoms

        try:
            from sklearn.cluster import KMeans
        except Exception:
            # sklearn unavailable: treat entire pool as one subpool
            return (
                [
                    DatasetPool(
                        pool_id=pool.pool_id,
                        records=pool.records,
                        pool_type="subpool",
                        parent_pool_id=pool.pool_id,
                    )
                ],
                [],
            )

        records = list(pool.records)
        if not records:
            return [], []

        # Dynamic element extraction from this pool
        present_elements = sorted({z for r in records for z in r.atomic_numbers})

        features = []
        for r in records:
            counts = {z: 0 for z in present_elements}
            for z in r.atomic_numbers:
                counts[z] = counts.get(z, 0) + 1
            total = max(r.natoms, 1)
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

        n_clusters = min(12, max(4, len(records) // 3000), len(records))
        if n_clusters <= 1:
            return (
                [
                    DatasetPool(
                        pool_id=pool.pool_id,
                        records=pool.records,
                        pool_type="subpool",
                        parent_pool_id=pool.pool_id,
                    )
                ],
                [],
            )

        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(features)
        label_counts = Counter(int(label) for label in labels)

        subpools: list[DatasetPool] = []
        unclassified: list[StructureRecord] = []
        for label in sorted(label_counts.keys()):
            members = [records[i] for i in range(len(records)) if labels[i] == label]
            if len(members) >= min_subpool_size:
                subpools.append(
                    DatasetPool(
                        pool_id=f"{pool.pool_id}::sub{label}",
                        records=tuple(members),
                        pool_type="subpool",
                        parent_pool_id=pool.pool_id,
                    )
                )
            else:
                unclassified.extend(members)

        return subpools, unclassified
