"""Metrics checked against their definitions, not against another library.

AUC is compared with a brute-force count over every positive/negative pair (ties counting a half),
which is the definition. ECE and Brier are compared with hand-computed values. The bootstrap is
checked for determinism, coverage of the point estimate and the degenerate single-unit case.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from verifier_guided_search.value.metrics import (
    Interval,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    roc_auc,
)

pytestmark = pytest.mark.unit


def brute_force_auc(labels: list[int], scores: list[float]) -> float:
    """The Mann-Whitney definition, written out: P(positive > negative) + 0.5 P(equal)."""
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    total = 0.0
    for p in positives:
        for n in negatives:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(positives) * len(negatives))


labelled = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=1),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=2,
    max_size=40,
).filter(lambda rows: {y for y, _ in rows} == {0, 1})

#: Deliberately coarse scores, so ties are common and the midrank path is exercised.
tied = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=1),
        st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
    ),
    min_size=2,
    max_size=40,
).filter(lambda rows: {y for y, _ in rows} == {0, 1})


@given(rows=labelled)
def test_auc_matches_the_brute_force_pair_count(rows: list[tuple[int, float]]) -> None:
    labels = [y for y, _ in rows]
    scores = [s for _, s in rows]
    assert roc_auc(labels, scores) == pytest.approx(brute_force_auc(labels, scores))


@given(rows=tied)
def test_auc_handles_ties_as_half_credit(rows: list[tuple[int, float]]) -> None:
    labels = [y for y, _ in rows]
    scores = [s for _, s in rows]
    assert roc_auc(labels, scores) == pytest.approx(brute_force_auc(labels, scores))


def test_auc_of_a_perfect_and_an_inverted_ranking() -> None:
    labels = [0, 0, 1, 1]
    assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5


@given(rows=labelled)
def test_auc_is_invariant_under_a_monotone_rescaling(rows: list[tuple[int, float]]) -> None:
    """AUC depends only on the order of the scores, so any order-preserving map leaves it alone.

    ``assume`` guards the one way the property can fail for reasons that are not about AUC: with
    subnormal inputs, ``3 * s + 1`` can round two distinct scores to the same float, which creates a
    tie that was not in the original data. Hypothesis finds that case immediately.
    """
    labels = [y for y, _ in rows]
    scores = [s for _, s in rows]
    rescaled = [3.0 * s + 1.0 for s in scores]
    assume(len(set(rescaled)) == len(set(scores)))
    assert roc_auc(labels, rescaled) == pytest.approx(roc_auc(labels, scores))


@given(rows=labelled)
def test_auc_of_the_flipped_labels_is_its_complement(rows: list[tuple[int, float]]) -> None:
    labels = [y for y, _ in rows]
    scores = [s for _, s in rows]
    assert roc_auc([1 - y for y in labels], scores) == pytest.approx(1.0 - roc_auc(labels, scores))


def test_auc_is_undefined_with_a_single_class() -> None:
    with pytest.raises(ValueError, match="at least one positive and one negative"):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="at least one positive and one negative"):
        roc_auc([0, 0], [0.1, 0.2])


def test_the_metric_helpers_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        roc_auc([1, 0], [0.5])
    with pytest.raises(ValueError, match="at least one observation"):
        roc_auc([], [])
    with pytest.raises(ValueError, match="bins must be positive"):
        expected_calibration_error([1, 0], [0.9, 0.1], bins=0)


def test_a_perfectly_calibrated_predictor_has_zero_calibration_error() -> None:
    # Two bins, each with the observed rate exactly equal to the predicted probability.
    labels = [1, 0, 0, 0, 1, 1, 1, 0]
    scores = [0.25] * 4 + [0.75] * 4
    assert expected_calibration_error(labels, scores, bins=4) == pytest.approx(0.0)


def test_a_confidently_wrong_predictor_has_the_maximum_calibration_error() -> None:
    assert expected_calibration_error([1, 1], [0.0, 0.0]) == pytest.approx(1.0)
    assert expected_calibration_error([0, 0], [1.0, 1.0]) == pytest.approx(1.0)


def test_calibration_error_is_population_weighted() -> None:
    """Three rows in a perfect bin and one in a wrong bin: the error is weighted 1/4, not 1/2."""
    labels = [1, 1, 1, 0]
    scores = [1.0, 1.0, 1.0, 0.9]
    assert expected_calibration_error(labels, scores, bins=10) == pytest.approx(0.9 / 4)


@given(rows=labelled)
def test_calibration_error_is_a_fraction(rows: list[tuple[int, float]]) -> None:
    labels = [y for y, _ in rows]
    scores = [s for _, s in rows]
    assert 0.0 <= expected_calibration_error(labels, scores) <= 1.0


def test_scores_outside_the_unit_interval_are_clamped() -> None:
    assert expected_calibration_error([1, 0], [5.0, -5.0]) == pytest.approx(0.0)


def test_brier_score_is_the_mean_squared_error() -> None:
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0
    assert brier_score([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)


def test_the_bootstrap_brackets_the_point_estimate_and_is_reproducible() -> None:
    units = list(range(40))

    def statistic(sample: Sequence[int]) -> float:
        return sum(1 for value in sample if value % 4 == 0) / len(sample)

    first = bootstrap_ci(units, statistic, resamples=300, seed=11)
    again = bootstrap_ci(units, statistic, resamples=300, seed=11)
    assert first == again
    assert first.estimate == pytest.approx(0.25)
    assert first.low <= first.estimate <= first.high
    assert first.samples == 40
    assert first.low < first.high, "40 units should give a non-degenerate interval"


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    units = list(range(30))

    def statistic(sample: Sequence[int]) -> float:
        return sum(sample) / len(sample)

    narrow = bootstrap_ci(units, statistic, resamples=400, confidence=0.5, seed=3)
    wide = bootstrap_ci(units, statistic, resamples=400, confidence=0.99, seed=3)
    assert (wide.high - wide.low) >= (narrow.high - narrow.low)


def test_a_single_unit_gives_a_degenerate_interval() -> None:
    interval = bootstrap_ci([7], lambda sample: float(sample[0]), resamples=100)
    assert interval == Interval(estimate=7.0, low=7.0, high=7.0, samples=1)


def test_zero_resamples_skips_the_bootstrap_but_keeps_the_estimate() -> None:
    interval = bootstrap_ci([1, 2, 3], lambda s: sum(s) / len(s), resamples=0)
    assert interval.estimate == pytest.approx(2.0)
    assert interval.low == interval.high == pytest.approx(2.0)


def test_the_bootstrap_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        bootstrap_ci([], lambda s: 0.0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1], lambda s: 0.0, confidence=1.5)


def test_the_interval_serialises_with_rounded_values() -> None:
    payload = Interval(estimate=1 / 3, low=0.1, high=0.9, samples=12).as_dict()
    assert payload == {
        "estimate": 0.333333,
        "ci_low": 0.1,
        "ci_high": 0.9,
        "samples": 12.0,
    }


def test_midranks_are_used_so_a_tie_heavy_input_is_not_order_dependent() -> None:
    """Shuffling rows must not change AUC; a naive rank implementation would drift here."""
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.5] * 6
    assert roc_auc(labels, scores) == 0.5
    assert roc_auc(list(reversed(labels)), scores) == 0.5
    assert not math.isnan(roc_auc(labels, scores))
