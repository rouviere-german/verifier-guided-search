"""Classification metrics for the value model, and bootstrap confidence intervals.

Written here rather than imported so inference and reporting stay free of scikit-learn, and so each
one can be checked against a definition instead of against another library:

* :func:`roc_auc` is the Mann-Whitney statistic - the probability that a random positive scores
  above a random negative, counting ties as half - computed from midranks, which handles ties
  exactly. ``tests/unit/test_value_metrics.py`` compares it with the brute-force pair count.
* :func:`expected_calibration_error` is the usual equal-width binned ECE: the average, weighted by
  bin population, of ``|mean predicted - observed rate|``. It is reported alongside AUC because a
  value model that ranks well but is badly calibrated distorts PUCT, whose Q values are
  probabilities.
* :func:`bootstrap_ci` resamples *tasks*, not individual evaluations, because the evaluations within
  one task are anything but independent.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "Interval",
    "bootstrap_ci",
    "brier_score",
    "expected_calibration_error",
    "roc_auc",
]


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a percentile confidence interval."""

    estimate: float
    low: float
    high: float
    samples: int

    def as_dict(self) -> dict[str, float]:
        return {
            "estimate": round(self.estimate, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "samples": float(self.samples),
        }


def _check(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("at least one observation is required")


def _midranks(scores: Sequence[float]) -> list[float]:
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve. Raises when only one class is present (AUC is undefined)."""
    _check(labels, scores)
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs at least one positive and one negative label")
    ranks = _midranks(scores)
    rank_sum = math.fsum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], *, bins: int = 10
) -> float:
    """Population-weighted mean gap between predicted probability and observed rate."""
    _check(labels, scores)
    if bins < 1:
        raise ValueError("bins must be positive")
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for label, score in zip(labels, scores, strict=True):
        clamped = min(max(score, 0.0), 1.0)
        index = min(int(clamped * bins), bins - 1)
        buckets[index].append((label, clamped))
    total = len(labels)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        observed = math.fsum(float(label) for label, _ in bucket) / len(bucket)
        predicted = math.fsum(score for _, score in bucket) / len(bucket)
        error += len(bucket) / total * abs(observed - predicted)
    return error


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean squared error of the predicted probabilities."""
    _check(labels, scores)
    return math.fsum(
        (score - label) ** 2 for label, score in zip(labels, scores, strict=True)
    ) / len(labels)


def bootstrap_ci[T](
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over ``units``, resampled with replacement.

    ``units`` are whatever is independent in the experiment - tasks, not evaluations.
    """
    if not units:
        raise ValueError("at least one unit is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    estimate = statistic(units)
    if len(units) == 1 or resamples < 1:
        return Interval(estimate=estimate, low=estimate, high=estimate, samples=len(units))
    # Bootstrap resampling, not cryptography.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    count = len(units)
    draws = sorted(
        statistic([units[rng.randrange(count)] for _ in range(count)]) for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = min(max(math.floor(tail * resamples), 0), resamples - 1)
    high_index = min(max(math.ceil((1.0 - tail) * resamples) - 1, 0), resamples - 1)
    return Interval(estimate=estimate, low=draws[low_index], high=draws[high_index], samples=count)
