"""PUCT selection: the invariants that make the rule the rule, not just some arithmetic."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from verifier_guided_search.errors import SearchError
from verifier_guided_search.search.puct import (
    DEFAULT_C_PUCT,
    ChildStats,
    exploration_term,
    puct_score,
    rank_priors,
    select_index,
)

pytestmark = pytest.mark.unit

priors = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
values = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
visits = st.integers(min_value=0, max_value=200)
c_pucts = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)

children = st.builds(
    ChildStats, prior=priors, visits=visits, total_value=values, fallback_value=values
)


def test_q_falls_back_to_the_parent_estimate_when_a_child_is_unvisited() -> None:
    unvisited = ChildStats(prior=0.5, visits=0, total_value=0.0, fallback_value=0.3)
    assert unvisited.q == 0.3
    visited = ChildStats(prior=0.5, visits=4, total_value=2.0, fallback_value=0.3)
    assert visited.q == 0.5


def test_the_formula_is_the_alphazero_one() -> None:
    stats = ChildStats(prior=0.25, visits=3, total_value=1.5, fallback_value=0.0)
    expected = 0.5 + 1.4 * 0.25 * math.sqrt(16) / 4
    assert puct_score(stats, 16, 1.4) == pytest.approx(expected)


@given(stats=children, parent=visits, c=c_pucts)
def test_the_exploration_term_is_never_negative(stats: ChildStats, parent: int, c: float) -> None:
    assert exploration_term(stats, parent, c) >= 0.0


@given(prior_low=priors, prior_high=priors, parent=visits, c=c_pucts)
def test_the_score_is_monotone_in_the_prior(
    prior_low: float, prior_high: float, parent: int, c: float
) -> None:
    low, high = sorted((prior_low, prior_high))
    left = ChildStats(prior=low, visits=2, total_value=1.0)
    right = ChildStats(prior=high, visits=2, total_value=1.0)
    assert puct_score(left, parent, c) <= puct_score(right, parent, c) + 1e-12


@given(a=visits, b=visits, parent=visits, c=c_pucts, prior=priors)
def test_the_exploration_term_decreases_as_a_child_is_visited(
    a: int, b: int, parent: int, c: float, prior: float
) -> None:
    low, high = sorted((a, b))
    less = ChildStats(prior=prior, visits=low, total_value=0.0)
    more = ChildStats(prior=prior, visits=high, total_value=0.0)
    assert exploration_term(less, parent, c) >= exploration_term(more, parent, c) - 1e-12


@given(options=st.lists(children, min_size=1, max_size=8), parent=visits)
def test_zero_c_puct_is_pure_argmax_over_q(options: list[ChildStats], parent: int) -> None:
    chosen = select_index(options, parent, 0.0)
    best = max(item.q for item in options)
    assert options[chosen].q == pytest.approx(best)


@given(options=st.lists(children, min_size=1, max_size=8), parent=visits, c=c_pucts)
def test_selection_picks_a_maximiser_and_breaks_ties_to_the_lowest_index(
    options: list[ChildStats], parent: int, c: float
) -> None:
    chosen = select_index(options, parent, c)
    scores = [puct_score(item, parent, c) for item in options]
    assert scores[chosen] == pytest.approx(max(scores))
    first = min(i for i, s in enumerate(scores) if s == max(scores))
    assert chosen == first


@given(options=st.lists(children, min_size=1, max_size=6), parent=visits, c=c_pucts)
def test_selection_is_deterministic(options: list[ChildStats], parent: int, c: float) -> None:
    assert select_index(options, parent, c) == select_index(options, parent, c)


def test_selection_refuses_an_empty_child_list() -> None:
    with pytest.raises(SearchError, match="empty child list"):
        select_index([], 1, DEFAULT_C_PUCT)


def test_negative_inputs_are_refused() -> None:
    with pytest.raises(SearchError, match="visits"):
        ChildStats(prior=0.5, visits=-1, total_value=0.0)
    with pytest.raises(SearchError, match="prior"):
        ChildStats(prior=-0.1, visits=0, total_value=0.0)
    with pytest.raises(SearchError, match="c_puct"):
        exploration_term(ChildStats(prior=0.5, visits=0, total_value=0.0), 1, -0.1)
    with pytest.raises(SearchError, match="parent_visits"):
        exploration_term(ChildStats(prior=0.5, visits=0, total_value=0.0), -1, 1.0)


@given(count=st.integers(min_value=1, max_value=16), temperature=st.floats(0.1, 4.0))
def test_rank_priors_are_a_decreasing_probability_distribution(
    count: int, temperature: float
) -> None:
    out = rank_priors(count, temperature=temperature)
    assert len(out) == count
    assert math.fsum(out) == pytest.approx(1.0)
    assert all(value > 0 for value in out)
    assert list(out) == sorted(out, reverse=True)


def test_rank_priors_flatten_as_the_temperature_rises() -> None:
    sharp = rank_priors(4, temperature=0.3)
    flat = rank_priors(4, temperature=5.0)
    assert sharp[0] > flat[0]
    assert sharp[-1] < flat[-1]


def test_rank_priors_reject_degenerate_arguments() -> None:
    with pytest.raises(SearchError, match="count"):
        rank_priors(0)
    with pytest.raises(SearchError, match="temperature"):
        rank_priors(3, temperature=0.0)


def test_an_unvisited_high_prior_child_is_preferred_to_a_visited_bad_one() -> None:
    """The behaviour that makes exploration work: one bad rollout must not close a branch."""
    explored = ChildStats(prior=0.4, visits=6, total_value=0.0, fallback_value=0.2)
    fresh = ChildStats(prior=0.4, visits=0, total_value=0.0, fallback_value=0.2)
    assert select_index([explored, fresh], 6, DEFAULT_C_PUCT) == 1


def test_a_strong_q_eventually_beats_exploration() -> None:
    """With a small c_puct the accumulated value wins; with a large one the prior still does.

    The crossover is what ``c_puct`` buys, and it is why the ablations sweep it.
    """
    good = ChildStats(prior=0.1, visits=10, total_value=9.0, fallback_value=0.0)
    unexplored = ChildStats(prior=0.9, visits=0, total_value=0.0, fallback_value=0.0)
    assert select_index([good, unexplored], 10, 0.1) == 0
    assert select_index([good, unexplored], 10, 0.5) == 1
