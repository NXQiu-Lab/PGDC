"""Adaptive linear scaling for metric values exceeding physical bounds.

When a dataset uses non-standard units or exhibits extreme out-of-range
values for a particular metric, auto-scaling brings the aggregate into the
expected physical range via a uniform linear factor.  All sample-level
values, moments and relative proportions are preserved; only the absolute
scale changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScalingInfo:
    """Diagnostics record produced when auto-scaling is applied."""

    applied: bool
    original_value: float
    threshold: float
    scale_factor: float
    scaled_value: float
    metric_name: str
    metric_id: str


def auto_scale_linear(
    values: list[float],
    observed_metric: float,
    threshold: float,
    metric_name: str,
    metric_id: str,
) -> tuple[list[float], ScalingInfo | None]:
    """Linearly scale *values* if *observed_metric* exceeds *threshold*.

    Scaling formula: ``v' = v * (threshold / observed_metric)``

    Because every element is multiplied by the same constant, the shape of
    the distribution (histogram, quantiles, variance ratios, etc.) is
    preserved exactly.  Only the absolute scale changes.

    Returns ``(scaled_values, scaling_info)`` when scaling is applied,
    or ``(original_values, None)`` when the metric is already within
    bounds.
    """
    if observed_metric <= threshold or observed_metric <= 0:
        return values, None

    scale_factor = threshold / observed_metric
    scaled_values = [v * scale_factor for v in values]

    info = ScalingInfo(
        applied=True,
        original_value=observed_metric,
        threshold=threshold,
        scale_factor=scale_factor,
        scaled_value=observed_metric * scale_factor,
        metric_name=metric_name,
        metric_id=metric_id,
    )
    logger.warning(
        "%s: %s (%.4f) exceeds threshold (%.4f). "
        "Auto-scaling by factor %.6f.",
        metric_id,
        metric_name,
        observed_metric,
        threshold,
        scale_factor,
    )
    return scaled_values, info
