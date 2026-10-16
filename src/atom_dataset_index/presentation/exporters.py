"""Presentation-layer exporters."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atom_dataset_index.domain import DatasetProfile
from atom_dataset_index.io import DatasetWriter
from atom_dataset_index.metric_core.results import MetricResult


@dataclass(slots=True)
class RunManifest:
    """Manifest written by task entrypoints."""

    run_id: str
    task_name: str
    project_version: str
    generated_at: str
    command: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class PresentationExporter:
    """Export raw figures metadata, tables, and run manifests."""

    def export_manifest(self, path: str, manifest: RunManifest) -> str:
        DatasetWriter.write_json(path, asdict(manifest))
        return path

    def export_profile_summary(
        self,
        path: str | Path,
        profile: DatasetProfile,
        dataset_name: str,
    ) -> Path:
        summary = profile.summary
        provenance = profile.provenance
        metric_results = profile.metric_results
        lines = [
            f"# 数据集质量画像总结：{dataset_name}",
            "",
            "## 1. 数据集级多维得分卡（Pool-Aware v1.3.0）",
            "",
            f"| 维度 | 得分 | 说明 |",
            f"|------|------|------|",
            f"| **Stability Score** | {summary.get('stability_score', 0.0):.2f} | 高 = 接近稳定态 |",
            f"| **Compositional Diversity Score** | {summary.get('compositional_diversity_score', 0.0):.2f} | 高 = 成分/有效相丰富 |",
            f"| **Structural Diversity Score** | {summary.get('structural_diversity_score', 0.0):.2f} | 高 = 描述符空间结构簇分布分散 |",
            f"| **Uniformity Score** | {summary.get('uniformity_score', 0.0):.2f} | 高 = 各池分布均衡 |",
            f"| **Coverage Score** | {summary.get('coverage_score', 0.0):.2f} | 高 = 规模-均匀性综合优秀 |",
            f"| **Redundancy Score** | {summary.get('redundancy_score', 0.0):.2f} | 高 = 局部冗余严重（负面指标） |",

            "",
            self._scorecard_interpretation(summary, metric_results),
            "",
        ]
        lines.extend(self._scaling_notice_lines(metric_results))
        lines.extend([
            "## 2. 数据准入与分池概览",
            f"- **样本总数**：{summary.get('record_count', 0)}",
            f"- **有效池数（phase_count_effective）**：{summary.get('phase_count_effective', 0)}",
            f"- **元素组合池数（pool_count）**：{summary.get('pool_count', 0)}",
            f"- **成功指标数**：{summary.get('metric_count', 0)}",
            f"- **跳过指标数**：{summary.get('skipped_metric_count', 0)}",
            f"- **失败指标数**：{summary.get('failed_metric_count', 0)}",
            f"- **描述符后端**：{provenance.get('descriptor_backend', 'unknown')}",
            f"- **原型策略**：{provenance.get('prototype_strategy', 'unknown')}",
            f"- **原型样本**：{provenance.get('prototype_sample_ids', '') or '无'}",
            "",
            "## 3. 核心指标摘要",
        ])
        lines.extend(self._metric_summary_lines(metric_results))
        lines.extend(
            [
                "",
                "## 4. 结果解读",
                self._interpretation(metric_results),
                "",
                "## 5. 使用建议",
                "- 结合 `stable_distance vs delta_energy_per_atom` 判断结构偏离是否和能量抬升一致。",
                "- 结合 `stable_distance vs force_mean_norm` 判断结构偏离是否对应更强的非平衡程度。",
                "- 若 `pool_count = 1`，应谨慎使用“多盆地覆盖”类结论，更适合解读为单 basin 近邻采样。",
                "- 若描述符后端不是 `dscribe_soap`，结果更适合用于流程验证，不建议直接作为最终论文结论。",
            ]
        )
        return DatasetWriter.write_text(path, "\n".join(lines) + "\n")

    def export_profile_summary_json(
        self,
        path: str | Path,
        profile: DatasetProfile,
    ) -> Path:
        metric_results = profile.metric_results
        payload = {
            "dataset_name": profile.dataset_name,
            "summary": profile.summary,
            "provenance": {
                "descriptor_backend": profile.provenance.get("descriptor_backend", ""),
                "prototype_strategy": profile.provenance.get("prototype_strategy", ""),
                "prototype_sample_ids": profile.provenance.get("prototype_sample_ids", ""),
            },
            "headline_metrics": {
                metric_id: metric_result.aggregate.values
                for metric_id, metric_result in metric_results.items()
                if isinstance(metric_result, MetricResult)
            },
        }
        return DatasetWriter.write_json(path, payload)

    def export_profile_figures(
        self,
        directory: str | Path,
        profile: DatasetProfile,
    ) -> dict[str, str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return {}

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated: dict[str, str] = {}

        energy = self._sample_values(profile.metric_results.get("energy.coverage"), "delta_energy_per_atom")
        force = self._sample_values(profile.metric_results.get("force.statistics"), "force_mean_norm")
        distortion = self._sample_values(
            profile.metric_results.get("distortion.prototype_relative"),
            "stable_distance",
        )
        prototypes = self._sample_values(profile.metric_results.get("prototype.coverage"), "prototype_id")

        if distortion and energy and len(distortion) == len(energy):
            figure_path = output_dir / "stable_distance_vs_energy.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.scatter(distortion, energy, s=18, alpha=0.8)
            ax.set_title("Stable Distance vs Relative Energy")
            ax.set_xlabel("Stable Distance")
            ax.set_ylabel("Delta Energy Per Atom")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_path, dpi=160)
            plt.close(fig)
            generated["stable_distance_vs_energy"] = str(figure_path)

        if distortion and force and len(distortion) == len(force):
            figure_path = output_dir / "stable_distance_vs_force.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.scatter(distortion, force, s=18, alpha=0.8, color="#c44e52")
            ax.set_title("Stable Distance vs Mean Force Norm")
            ax.set_xlabel("Stable Distance")
            ax.set_ylabel("Mean Force Norm")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_path, dpi=160)
            plt.close(fig)
            generated["stable_distance_vs_force"] = str(figure_path)

        if prototypes:
            figure_path = output_dir / "prototype_occupancy_hist.png"
            prototype_counts: dict[int, int] = {}
            for value in prototypes:
                prototype_id = int(round(value))
                prototype_counts[prototype_id] = prototype_counts.get(prototype_id, 0) + 1
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ordered_ids = sorted(prototype_counts)
            ax.bar([str(item) for item in ordered_ids], [prototype_counts[item] for item in ordered_ids], color="#4c72b0")
            ax.set_title("Prototype Occupancy")
            ax.set_xlabel("Prototype ID")
            ax.set_ylabel("Sample Count")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_path, dpi=160)
            plt.close(fig)
            generated["prototype_occupancy_hist"] = str(figure_path)

        if energy:
            figure_path = output_dir / "energy_distribution.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.hist(energy, bins=min(20, max(5, len(energy) // 4)), color="#55a868", alpha=0.85)
            ax.set_title("Relative Energy Distribution")
            ax.set_xlabel("Delta Energy Per Atom")
            ax.set_ylabel("Sample Count")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_path, dpi=160)
            plt.close(fig)
            generated["energy_distribution"] = str(figure_path)

        return generated

    @staticmethod
    def _scaling_notice_lines(metric_results: dict[str, object]) -> list[str]:
        """Generate a prominent notice when auto-scaling was applied."""
        notices: list[str] = []
        for metric_id in ("energy.coverage", "force.statistics", "distortion.prototype_relative"):
            metric_result = metric_results.get(metric_id)
            if not isinstance(metric_result, MetricResult):
                continue
            scaling = metric_result.aggregate.diagnostics.get("auto_scaling")
            if not isinstance(scaling, dict) or not scaling.get("applied"):
                continue
            notices.append(
                f"> ⚠️ **{metric_id}**：检测到数值超出合理物理范围，已执行自适应线性缩放。"
            )
            orig = scaling.get('original_value')
            thresh = scaling.get('threshold')
            sf = scaling.get('scale_factor')
            orig_str = f"{orig:.4f}" if isinstance(orig, (int, float)) else str(orig)
            thresh_str = f"{thresh:.4f}" if isinstance(thresh, (int, float)) else str(thresh)
            sf_str = f"{sf:.6f}" if isinstance(sf, (int, float)) else str(sf)
            notices.append(
                f">   原始 `{scaling.get('metric')}` = `{orig_str}`，"
                f"阈值 = `{thresh_str}`，"
                f"缩放因子 = `{sf_str}`。"
            )
        if notices:
            return ["## 1.5 数据自适应缩放声明", ""] + notices + [""]
        return []

    @staticmethod
    def _sample_values(metric_result: object, field_name: str) -> list[float]:
        if not isinstance(metric_result, MetricResult):
            return []
        return [
            float(sample_record.values[field_name])
            for sample_record in metric_result.sample_records
            if field_name in sample_record.values
        ]

    @staticmethod
    def _metric_summary_lines(metric_results: dict[str, object]) -> list[str]:
        lines: list[str] = []
        for metric_id in (
            "energy.coverage",
            "force.statistics",
            "distortion.prototype_relative",
            "prototype.coverage",
            "diversity.structural",
            "diversity.redundancy",
        ):
            metric_result = metric_results.get(metric_id)
            if not isinstance(metric_result, MetricResult):
                continue
            lines.append(f"### `{metric_id}`")
            for key, value in metric_result.aggregate.values.items():
                lines.append(f"- `{key}`: {value:.6f}")
            diagnostics = metric_result.aggregate.diagnostics
            for key, value in diagnostics.items():
                lines.append(f"- `{key}`: {value}")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return lines

    @staticmethod
    def _scorecard_interpretation(summary: dict[str, Any], metric_results: dict[str, object]) -> str:
        stability = summary.get("stability_score", 0.0)
        compositional = summary.get("compositional_diversity_score", 0.0)
        structural = summary.get("structural_diversity_score", 0.0)
        uniformity = summary.get("uniformity_score", 0.0)
        coverage = summary.get("coverage_score", 0.0)
        redundancy = summary.get("redundancy_score", 0.0)
        pool_count = summary.get("pool_count", 0)

        stability_text = (
            "数据集整体接近稳定态" if stability >= 70 else
            "数据集包含一定量偏离稳定态的构型" if stability >= 40 else
            "数据集整体结构偏离稳定态的程度较高"
        )
        compositional_text = (
            "成分/有效相覆盖有限" if compositional < 40 else
            "成分/有效相覆盖较好" if compositional < 70 else
            "成分/有效相覆盖非常丰富"
        )
        structural_text = (
            "描述符空间结构簇集中" if structural < 40 else
            "描述符空间结构簇分布较分散" if structural < 70 else
            "描述符空间结构簇分布非常分散"
        )
        uniformity_text = (
            "各池分布均匀" if uniformity >= 80 else
            "各池分布存在一定倾斜" if uniformity >= 50 else
            "各池分布极不均匀"
        )
        coverage_text = (
            "规模-均匀性综合覆盖优秀" if coverage >= 70 else
            "规模-均匀性综合覆盖尚可" if coverage >= 40 else
            "规模-均匀性综合覆盖偏弱，小数据集需警惕虚假均匀高分"
        )
        redundancy_text = (
            "描述符空间局部堆积明显，冗余较高" if redundancy >= 70 else
            "存在一定局部堆积" if redundancy >= 40 else
            "样本在描述符空间中分布较稀疏、冗余低"
        )
        pool_text = f"由 {pool_count} 个有效池聚合而成" if pool_count > 1 else "单一池全局评价"
        return f"> **总体印象**：{pool_text}；{stability_text}；{compositional_text}；{structural_text}；{uniformity_text}；{coverage_text}；{redundancy_text}。"

    @staticmethod
    def _interpretation(metric_results: dict[str, object]) -> str:
        energy_result = metric_results.get("energy.coverage")
        force_result = metric_results.get("force.statistics")
        distortion_result = metric_results.get("distortion.prototype_relative")
        prototype_result = metric_results.get("prototype.coverage")
        if not all(
            isinstance(item, MetricResult)
            for item in (energy_result, force_result, distortion_result, prototype_result)
        ):
            return "当前结果不完整，建议先检查标签完整性、适用性检查和被跳过的指标。"

        energy_range = energy_result.aggregate.values.get("energy_per_atom_range", 0.0)
        force_mean = force_result.aggregate.values.get("force_mean", 0.0)
        stable_distance_mean = distortion_result.aggregate.values.get("stable_distance_mean", 0.0)
        coverage_distance_mean = distortion_result.aggregate.values.get("coverage_distance_mean", 0.0)
        prototype_count = int(round(prototype_result.aggregate.values.get("prototype_count", 0.0)))

        energy_text = "能量覆盖较窄" if energy_range < 0.02 else "能量覆盖较宽"
        force_text = "整体非平衡程度较低" if force_mean < 1.0 else "存在较明显的非平衡态覆盖"
        distortion_text = "样本主要集中在稳定参考结构附近" if stable_distance_mean < 3.0 else "样本相对稳定参考结构存在明显偏离"
        coverage_text = "采样覆盖较为充分" if coverage_distance_mean < 3.0 else "采样覆盖存在一定稀疏区域"
        prototype_text = "当前更接近单原型分布" if prototype_count <= 1 else "当前已表现出多原型覆盖特征"
        return (
            f"- {energy_text}；{force_text}；{distortion_text}；{coverage_text}；{prototype_text}。"
        )
