#!/usr/bin/env python3
"""Core-hour timing tracker for the MLIP-4 AL baseline."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class TimingTracker:
    """Track wall time and core-hours for each pipeline stage."""

    def __init__(self, output_file: str | Path):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._active: dict[str, dict[str, Any]] = {}
        # Load existing records for resumable runs
        if self.output_file.exists():
            try:
                with open(self.output_file, "r") as f:
                    data = json.load(f)
                self._records = data.get("records", [])
            except (json.JSONDecodeError, OSError):
                self._records = []

    def start(self, stage: str, n_cores: int = 1, iteration: int | None = None, **metadata):
        """Start timing a stage."""
        if stage in self._active:
            raise RuntimeError(f"Stage {stage} already active")
        self._active[stage] = {
            "stage": stage,
            "n_cores": n_cores,
            "iteration": iteration,
            "start_time": time.perf_counter(),
            "metadata": metadata,
        }

    def stop(self, stage: str, **metadata):
        """Stop timing a stage and append a record."""
        if stage not in self._active:
            raise RuntimeError(f"Stage {stage} not active")
        rec = self._active.pop(stage)
        wall = time.perf_counter() - rec["start_time"]
        n_cores = rec["n_cores"]
        core_hours = wall * n_cores / 3600.0
        record = {
            "stage": rec["stage"],
            "iteration": rec["iteration"],
            "n_cores": n_cores,
            "wall_time_seconds": wall,
            "core_hours": core_hours,
            "metadata": {**rec["metadata"], **metadata},
        }
        self._records.append(record)
        return record

    def record(
        self,
        stage: str,
        n_cores: int,
        wall_time_seconds: float,
        iteration: int | None = None,
        **metadata,
    ) -> dict[str, Any]:
        """Manually append a completed timing record (e.g. for batch jobs)."""
        record = {
            "stage": stage,
            "iteration": iteration,
            "n_cores": n_cores,
            "wall_time_seconds": wall_time_seconds,
            "core_hours": wall_time_seconds * n_cores / 3600.0,
            "metadata": metadata,
        }
        self._records.append(record)
        return record

    def summary(self) -> dict[str, float]:
        """Return total core-hours per stage."""
        totals: dict[str, float] = defaultdict(float)
        for rec in self._records:
            totals[rec["stage"]] += rec["core_hours"]
        return dict(totals)

    def total_core_hours(self) -> float:
        return sum(r["core_hours"] for r in self._records)

    def save(self):
        """Write records and summary to JSON."""
        data = {
            "records": self._records,
            "summary": self.summary(),
            "total_core_hours": self.total_core_hours(),
        }
        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
