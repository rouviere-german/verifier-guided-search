"""The budget accountant's one invariant, checked over random interleavings.

The spec's property is "the budget accountant never lets any strategy exceed its budget". Because
tokens and sandbox seconds are only known *after* the work, that has to hold across reserve, settle
and release in any order, which is what :func:`test_the_invariant_holds_over_random_operations`
does: it drives a random program of operations and asserts, after every single step, that committed
spend plus outstanding reservations is within every limit.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from verifier_guided_search.errors import BudgetError, BudgetExhausted
from verifier_guided_search.search.budget import (
    DIMENSIONS,
    BudgetAccountant,
    BudgetSpec,
    BudgetUsage,
    Reservation,
)

pytestmark = pytest.mark.unit

_SPEC = BudgetSpec.of(candidates=8, tokens=20_000, sandbox_seconds=12.0, usd=0.5)


def _usage(**kwargs: int) -> BudgetUsage:
    return BudgetUsage(**kwargs)


usages = st.builds(
    BudgetUsage,
    candidates=st.integers(min_value=0, max_value=6),
    generator_tokens=st.integers(min_value=0, max_value=8_000),
    sandbox_micros=st.integers(min_value=0, max_value=5_000_000),
    usd_micros=st.integers(min_value=0, max_value=200_000),
)


class BudgetMachine(RuleBasedStateMachine):
    """Reserve, settle and release in random order; the invariant is checked after every rule."""

    def __init__(self) -> None:
        super().__init__()
        self.accountant = BudgetAccountant(_SPEC, sandbox_usd_per_second=0.001)
        self.open: list[Reservation] = []

    @initialize()
    def start(self) -> None:
        assert self.accountant.invariant_holds()

    @rule(request=usages)
    def reserve(self, request: BudgetUsage) -> None:
        refusal: BudgetExhausted | None = None
        try:
            self.open.append(self.accountant.reserve(request))
        except BudgetExhausted as exhausted:
            refusal = exhausted
        if refusal is not None:
            assert refusal.dimension in DIMENSIONS
            assert refusal.attempted > refusal.limit

    @precondition(lambda self: bool(self.open))
    @rule(index=st.integers(min_value=0, max_value=32), fraction=st.floats(0.0, 1.0))
    def settle(self, index: int, fraction: float) -> None:
        reservation = self.open.pop(index % len(self.open))
        reserved = reservation.reserved
        reservation.settle(
            candidates=int(reserved.candidates * fraction),
            generator_tokens=int(reserved.generator_tokens * fraction),
            sandbox_micros=int(reserved.sandbox_micros * fraction),
            usd_micros=int(reserved.usd_micros * fraction),
        )

    @precondition(lambda self: bool(self.open))
    @rule(index=st.integers(min_value=0, max_value=32))
    def release(self, index: int) -> None:
        self.open.pop(index % len(self.open)).release()

    @invariant()
    def never_over_budget(self) -> None:
        assert self.accountant.invariant_holds()
        total = self.accountant.committed.plus(self.accountant.outstanding)
        assert total.exceeds(_SPEC) is None
        for name in DIMENSIONS:
            assert getattr(self.accountant.remaining, name) >= 0


def test_the_invariant_holds_over_random_operations() -> None:
    """Drive the state machine explicitly.

    ``BudgetMachine.TestCase`` would also run under pytest, but a generated class cannot be cited by
    name from CLAIMS.md or docs/THREAT_MODEL.md - and ``scripts/check_baseline.py`` refuses a
    reference it cannot resolve to a real test function, which is the point of that check.
    """
    run_state_machine_as_test(BudgetMachine)  # type: ignore[no-untyped-call]


@given(requests=st.lists(usages, min_size=1, max_size=20))
def test_committed_spend_never_exceeds_the_limits(requests: list[BudgetUsage]) -> None:
    """Settle everything at the worst case: the accountant must still refuse the overflow."""
    accountant = BudgetAccountant(_SPEC)
    for request in requests:
        try:
            reservation = accountant.reserve(request)
        except BudgetExhausted:
            continue
        reservation.settle()
        assert accountant.invariant_holds()
    assert accountant.committed.exceeds(_SPEC) is None


def test_a_refused_reservation_changes_nothing() -> None:
    accountant = BudgetAccountant(_SPEC)
    before = (accountant.committed, accountant.outstanding)
    with pytest.raises(BudgetExhausted) as info:
        accountant.reserve(_usage(candidates=_SPEC.max_candidates + 1))
    assert info.value.dimension == "candidates"
    assert (accountant.committed, accountant.outstanding) == before
    assert accountant.refusals == ("candidates",)


def test_outstanding_reservations_count_against_the_limit() -> None:
    """Two half-budget reservations may be held; a third must be refused before it spends."""
    accountant = BudgetAccountant(_SPEC)
    first = accountant.reserve(_usage(candidates=4))
    accountant.reserve(_usage(candidates=4))
    with pytest.raises(BudgetExhausted):
        accountant.reserve(_usage(candidates=1))
    first.release()
    accountant.reserve(_usage(candidates=1)).settle()


def test_settling_more_than_was_reserved_is_refused() -> None:
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve(_usage(candidates=1, generator_tokens=100))
    with pytest.raises(BudgetError, match="settled more than was reserved"):
        reservation.settle(generator_tokens=101)
    # The reservation is still open, so the caller can settle honestly instead.
    assert not reservation.closed
    reservation.settle(generator_tokens=100)
    assert accountant.committed.generator_tokens == 100


def test_a_reservation_is_single_use() -> None:
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve(_usage(candidates=1))
    reservation.settle()
    with pytest.raises(BudgetError, match="already settled"):
        reservation.settle()
    with pytest.raises(BudgetError, match="already settled"):
        reservation.release()


def test_the_context_manager_releases_an_unsettled_reservation() -> None:
    accountant = BudgetAccountant(_SPEC)
    with pytest.raises(RuntimeError, match="boom"), accountant.reserve(_usage(candidates=1)):
        raise RuntimeError("boom")
    assert accountant.outstanding == BudgetUsage()
    assert accountant.committed == BudgetUsage()


def test_negative_usage_is_rejected() -> None:
    with pytest.raises(BudgetError, match="must not be negative"):
        BudgetUsage(candidates=-1)
    with pytest.raises(BudgetError, match="must not be negative"):
        BudgetAccountant(_SPEC, sandbox_usd_per_second=-1.0)


def test_generation_reserves_the_worst_case_and_settles_the_actual() -> None:
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve_generation(
        model="claude-sonnet-5", input_tokens=500, max_output_tokens=1_000
    )
    assert reservation.reserved.generator_tokens == 1_500
    assert reservation.reserved.usd_micros > 0
    actual = accountant.settle_generation(
        reservation, model="claude-sonnet-5", input_tokens=500, output_tokens=40
    )
    assert actual.generator_tokens == 540
    assert actual.usd_micros <= reservation.reserved.usd_micros
    assert accountant.invariant_holds()


def test_a_free_model_costs_no_usd_but_still_costs_tokens() -> None:
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve_generation(
        model="fake-mutation", input_tokens=200, max_output_tokens=300
    )
    assert reservation.reserved.usd_micros == 0
    assert reservation.reserved.generator_tokens == 500
    reservation.settle()


def test_evaluation_reserves_the_wall_timeout_and_settles_the_real_duration() -> None:
    accountant = BudgetAccountant(_SPEC, sandbox_usd_per_second=0.01)
    reservation = accountant.reserve_evaluation(timeout_seconds=1.5)
    assert reservation.reserved.candidates == 1
    assert reservation.reserved.sandbox_micros == 1_500_000
    actual = accountant.settle_evaluation(reservation, seconds=0.2)
    assert actual.candidates == 1
    assert actual.sandbox_micros == 200_000
    assert accountant.committed.sandbox_seconds == pytest.approx(0.2)


def test_settling_more_generator_tokens_than_estimated_is_clamped_not_refused() -> None:
    """A live prompt is routinely bigger than chars/4, and that must not crash the search.

    ``reserve_generation`` charges an *estimate* of the input tokens (the offline generator's
    ``estimate_input_tokens`` is characters / 4) while a real provider reports the true count,
    summed over input, cache-creation and cache-read tokens. The first live call whose prompt beat
    the estimate raised ``BudgetError`` out of ``settle_generation``, and ``run_search`` caught only
    ``BudgetExhausted`` - so the search died with a traceback instead of degrading. Sandbox seconds
    were already clamped for exactly this reason; tokens were not.
    """
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve_generation(
        model="claude-sonnet-5", input_tokens=100, max_output_tokens=200
    )
    assert reservation.reserved.generator_tokens == 300
    actual = accountant.settle_generation(
        reservation, model="claude-sonnet-5", input_tokens=400, output_tokens=200
    )
    assert actual.generator_tokens == 300, "the settle is clamped to the reservation"
    assert accountant.invariant_holds()
    # Clamping hides an underestimate, so the shortfall is counted and reported.
    assert accountant.generator_token_overshoot == 300


def test_an_honest_generation_records_no_token_overshoot() -> None:
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve_generation(
        model="claude-sonnet-5", input_tokens=100, max_output_tokens=200
    )
    accountant.settle_generation(
        reservation, model="claude-sonnet-5", input_tokens=100, output_tokens=50
    )
    assert accountant.generator_token_overshoot == 0


def test_settling_longer_than_the_reserved_timeout_is_clamped_not_refused() -> None:
    """A watchdog can overshoot by microseconds; that must not crash a finished search."""
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve_evaluation(timeout_seconds=0.5)
    accountant.settle_evaluation(reservation, seconds=9.0)
    assert accountant.committed.sandbox_micros == 500_000
    assert accountant.invariant_holds()


def test_a_cache_hit_settles_at_zero_sandbox_time_but_still_costs_a_candidate() -> None:
    accountant = BudgetAccountant(_SPEC)
    reservation = accountant.reserve_evaluation(timeout_seconds=1.5)
    accountant.settle_evaluation(reservation, seconds=0.0)
    assert accountant.committed.candidates == 1
    assert accountant.committed.sandbox_micros == 0


def test_usage_arithmetic_round_trips() -> None:
    left = _usage(candidates=2, generator_tokens=30, sandbox_micros=40, usd_micros=5)
    right = _usage(candidates=1, generator_tokens=10, sandbox_micros=15, usd_micros=2)
    assert left.plus(right).minus(right) == left
    assert left.as_dict() == {
        "candidates": 2,
        "generator_tokens": 30,
        "sandbox_micros": 40,
        "usd_micros": 5,
    }
    assert left.exceeds(_SPEC) is None
    assert _usage(candidates=99).exceeds(_SPEC) == "candidates"


def test_scaled_candidates_changes_only_the_candidate_cap() -> None:
    scaled = _SPEC.scaled_candidates(3)
    assert scaled.max_candidates == 3
    assert scaled.max_generator_tokens == _SPEC.max_generator_tokens
    assert scaled.max_sandbox_micros == _SPEC.max_sandbox_micros


def test_snapshot_reports_committed_spend_only() -> None:
    accountant = BudgetAccountant(_SPEC)
    accountant.reserve(_usage(candidates=2))  # left outstanding on purpose
    reservation = accountant.reserve(_usage(candidates=1))
    reservation.settle()
    assert accountant.snapshot()["candidates"] == 1
