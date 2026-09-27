"""PUCT selection, as pure functions over child statistics.

The rule is the one from AlphaGo Zero / AlphaZero (Silver et al., 2017), which is Rosin's
predictor-UCB (2011) with a prior from a policy:

    score(a) = Q(a) + c_puct * P(a) * sqrt(N(parent)) / (1 + N(a))

``Q(a)`` is the mean value backed up through ``a``; a child that has never been visited uses its
``fallback_value``, the value estimate its parent had, so an unvisited sibling is neither optimistic
nor pessimistic by default. ``P(a)`` comes from the generator's rank of the candidate, and
``c_puct`` is the exploration/exploitation knob the ablations sweep.

Keeping this separate from the tree means the invariants are testable directly: monotone in the
prior, monotone down in visits at fixed Q, and pure argmax over Q when ``c_puct`` is zero. Ties
break towards the lowest index, so selection is deterministic without an RNG.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from verifier_guided_search.errors import SearchError

__all__ = ["DEFAULT_C_PUCT", "ChildStats", "exploration_term", "puct_score", "select_index"]

DEFAULT_C_PUCT = 1.4


@dataclass(frozen=True, slots=True)
class ChildStats:
    """What PUCT needs to know about one child edge."""

    prior: float
    visits: int
    total_value: float
    fallback_value: float = 0.0

    def __post_init__(self) -> None:
        if self.visits < 0:
            raise SearchError("visits must not be negative")
        if self.prior < 0:
            raise SearchError("prior must not be negative")

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else self.fallback_value


def exploration_term(stats: ChildStats, parent_visits: int, c_puct: float) -> float:
    """``c_puct * P * sqrt(N_parent) / (1 + N_child)``, never negative."""
    if c_puct < 0:
        raise SearchError("c_puct must not be negative")
    if parent_visits < 0:
        raise SearchError("parent_visits must not be negative")
    return c_puct * stats.prior * math.sqrt(parent_visits) / (1 + stats.visits)


def puct_score(stats: ChildStats, parent_visits: int, c_puct: float) -> float:
    return stats.q + exploration_term(stats, parent_visits, c_puct)


def select_index(children: Sequence[ChildStats], parent_visits: int, c_puct: float) -> int:
    """Index of the highest-scoring child; ties go to the lowest index."""
    if not children:
        raise SearchError("cannot select from an empty child list")
    best_index = 0
    best_score = puct_score(children[0], parent_visits, c_puct)
    for index in range(1, len(children)):
        score = puct_score(children[index], parent_visits, c_puct)
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def rank_priors(count: int, *, temperature: float = 1.0) -> tuple[float, ...]:
    """Priors for ``count`` ranked candidates: ``exp(-rank / temperature)``, normalised.

    Rank rather than raw generator score, so the prior does not depend on the scale of whatever the
    generator scores candidates with - which for a model is an arbitrary log-probability and for the
    mutation generator is a noisy heuristic.
    """
    if count <= 0:
        raise SearchError("count must be positive")
    if temperature <= 0:
        raise SearchError("temperature must be positive")
    weights = [math.exp(-rank / temperature) for rank in range(count)]
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)
