"""Basic statistical helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def quantiles(values: Sequence[float], qs: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {str(q): 0.0 for q in qs}
    ordered = sorted(values)
    result: dict[str, float] = {}
    for q in qs:
        if q <= 0:
            result[str(q)] = ordered[0]
            continue
        if q >= 1:
            result[str(q)] = ordered[-1]
            continue
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            result[str(q)] = ordered[lower]
        else:
            weight = index - lower
            result[str(q)] = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return result


def entropy(values: Sequence[float], bins: int = 10) -> float:
    if not values or bins <= 0:
        return 0.0
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return 0.0
    width = (maximum - minimum) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - minimum) / width), bins - 1)
        counts[index] += 1
    total = len(values)
    probs = [count / total for count in counts if count]
    return -sum(prob * math.log(prob) for prob in probs)
