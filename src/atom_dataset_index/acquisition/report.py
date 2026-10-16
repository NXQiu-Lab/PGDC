"""Report generation for acquisition mode results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AcquisitionRanking:
    """Ranking information for a single candidate structure.

    Attributes:
        poscar_path: Path to the POSCAR file for this candidate.
        sample_id: Sample ID assigned during reading.
        total_value: Combined acquisition value (0-1).
        uncertainty: Model uncertainty proxy (0-1).
        coverage_gap: Descriptor space coverage gap (0-1).
        trajectory_diversity: Estimated trajectory diversity (0-1).
    """

    poscar_path: str
    sample_id: int
    total_value: float
    uncertainty: float
    coverage_gap: float
    trajectory_diversity: float

    @property
    def recommendation(self) -> str:
        """Generate recommendation label based on total value.

        Returns:
            "HIGH" for values >= 0.7, "MEDIUM" for >= 0.4, "LOW" otherwise.
        """
        if self.total_value >= 0.7:
            return "HIGH"
        if self.total_value >= 0.4:
            return "MEDIUM"
        return "LOW"


@dataclass(slots=True)
class AcquisitionReport:
    """Complete acquisition evaluation report.

    Attributes:
        rankings: List of ranked candidates (sorted by total_value descending).
        summary: Dictionary of summary statistics.
        original_dataset_name: Name of the original dataset.
        original_sample_count: Number of samples in original dataset.
        candidate_count: Total number of candidates submitted (including rejected).
        incompatible_candidates: Candidates rejected by element compatibility check.
            Each entry is a dict with keys ``poscar_path``, ``sample_id``,
            ``elements`` (sorted list of atomic numbers), and ``reason``.
    """

    rankings: list[AcquisitionRanking]
    summary: dict[str, Any]
    original_dataset_name: str
    original_sample_count: int
    candidate_count: int
    incompatible_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "run_info": {
                "original_dataset": self.original_dataset_name,
                "original_sample_count": self.original_sample_count,
                "candidate_count": self.candidate_count,
                "incompatible_count": len(self.incompatible_candidates),
                "scored_count": self.candidate_count - len(self.incompatible_candidates),
            },
            "summary": self.summary,
            "rankings": [
                {
                    "rank": i + 1,
                    "poscar_path": r.poscar_path,
                    "sample_id": r.sample_id,
                    "total_value": round(r.total_value, 4),
                    "uncertainty": round(r.uncertainty, 4),
                    "coverage_gap": round(r.coverage_gap, 4),
                    "trajectory_diversity": round(r.trajectory_diversity, 4),
                    "recommendation": r.recommendation,
                }
                for i, r in enumerate(self.rankings)
            ],
            "incompatible_candidates": self.incompatible_candidates,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            JSON formatted string.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def export(self, path: str | Path) -> None:
        """Export report to JSON file.

        Args:
            path: Path to write the JSON file.
        """
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        n_incompatible = len(self.incompatible_candidates)
        n_scored = self.candidate_count - n_incompatible

        print(f"\n{'='*60}")
        print("Acquisition Mode Report")
        print(f"{'='*60}")
        print(f"Original Dataset: {self.original_dataset_name}")
        print(f"Original Samples: {self.original_sample_count}")
        print(f"Candidates Submitted: {self.candidate_count}")
        print(f"  Passed element check: {n_scored}")
        print(f"  Rejected (incompatible): {n_incompatible}")
        print(f"Top Recommendations: {len(self.rankings)}")
        print()

        print("Summary Statistics:")
        for key, value in self.summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

        print("Top Candidates (ranked by acquisition value):")
        print(f"{'Rank':<6}{'Path':<30}{'Value':<10}{'Recommendation':<15}")
        print("-" * 60)
        for i, ranking in enumerate(self.rankings[:10], 1):
            path_display = ranking.poscar_path
            if len(path_display) > 28:
                path_display = "..." + path_display[-25:]
            print(
                f"{i:<6}{path_display:<30}"
                f"{ranking.total_value:<10.4f}{ranking.recommendation:<15}"
            )

        if self.incompatible_candidates:
            print()
            print("Rejected Candidates (element incompatibility):")
            print(f"{'Path':<30}{'Elements':<20}{'Reason'}")
            print("-" * 60)
            for item in self.incompatible_candidates:
                path_display = item["poscar_path"]
                if len(path_display) > 28:
                    path_display = "..." + path_display[-25:]
                elems = str(item["elements"])
                reason_short = item["reason"][:40] + "..." if len(item["reason"]) > 40 else item["reason"]
                print(f"{path_display:<30}{elems:<20}{reason_short}")

        print(f"{'='*60}\n")


class ReportExporter:
    """Exporter for acquisition reports in various formats."""

    @staticmethod
    def export_json(report: AcquisitionReport, path: str | Path) -> None:
        """Export report to JSON file.

        Args:
            report: The report to export.
            path: Output file path.
        """
        Path(path).write_text(report.to_json(), encoding="utf-8")

    @staticmethod
    def export_markdown(report: AcquisitionReport, path: str | Path) -> None:
        """Export report to Markdown file.

        Args:
            report: The report to export.
            path: Output file path.
        """
        lines = [
            "# Acquisition Mode Report\n",
            "## Run Information\n",
            f"- **Original Dataset**: {report.original_dataset_name}\n",
            f"- **Original Samples**: {report.original_sample_count}\n",
            f"- **Candidates Evaluated**: {report.candidate_count}\n",
            f"- **Top Recommendations Shown**: {len(report.rankings)}\n",
            "\n## Summary Statistics\n",
        ]

        for key, value in report.summary.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}\n")
            else:
                lines.append(f"- **{key}**: {value}\n")

        lines.extend([
            "\n## Ranked Candidates\n",
            "| Rank | POSCAR Path | Total Value | Uncertainty | Coverage Gap | Diversity | Recommendation |\n",
            "|------|-------------|-------------|-------------|--------------|-----------|----------------|\n",
        ])

        for i, r in enumerate(report.rankings, 1):
            lines.append(
                f"| {i} | `{r.poscar_path}` | {r.total_value:.4f} | "
                f"{r.uncertainty:.4f} | {r.coverage_gap:.4f} | "
                f"{r.trajectory_diversity:.4f} | **{r.recommendation}** |\n"
            )

        Path(path).write_text("".join(lines), encoding="utf-8")
