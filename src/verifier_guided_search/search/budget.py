"""The budget accountant: reserve first, settle after, never exceed a limit.

Compute-matched comparison is the point of the evaluation, so "how much did this strategy spend" has
to be exact rather than approximate. Two dimensions can only be measured *after* the work happens -
the tokens a model actually returns, the seconds a sandbox execution actually takes - so charging
them afterwards would let a single step overshoot the budget, which is precisely the loophole that
makes a strategy comparison meaningless.

The accountant therefore works in two phases:

* :meth:`BudgetAccountant.reserve` takes the worst case for the step that is about to happen (the
  requested ``max_tokens``, the sandbox's wall timeout) and refuses it if committed spend plus all
  outstanding reservations plus this request would pass any limit;
* :meth:`Reservation.settle` replaces the reservation with what was really spent, which must be at
  most what was reserved.

The invariant that follows - ``committed + outstanding <= limits`` at every point, for every
dimension - is what ``tests/unit/test_budget_accountant.py`` checks with hypothesis over random
interleavings of reservations, settlements and releases.

Dimensions
----------
``candidates``
    Candidate programs handed to the verifier. Charged whether or not the verdict cache answers,
    because it is the compute-matched axis of the experiments: the cache is a wall-clock
    optimisation, not a way to buy extra evaluations.
``generator_tokens``
    Input plus output tokens. The offline mutation generator reports a deterministic estimate for
    the prompt it *would* have sent, so offline and live runs are charged on the same axis.
``sandbox_micros``
    Microseconds of actual sandbox execution; a cache hit settles at zero.
``usd_micros``
    Derived: token cost from the model's price plus sandbox seconds at a configured rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.agentic.pricing import cost_usd_micros
from verifier_guided_search.errors import BudgetError, BudgetExhausted

__all__ = [
    "DIMENSIONS",
    "BudgetAccountant",
    "BudgetSpec",
    "BudgetUsage",
    "Reservation",
]

DIMENSIONS = ("candidates", "generator_tokens", "sandbox_micros", "usd_micros")


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """A point in budget space: how much of each dimension something costs or has cost."""

    candidates: int = 0
    generator_tokens: int = 0
    sandbox_micros: int = 0
    usd_micros: int = 0

    def __post_init__(self) -> None:
        for name in DIMENSIONS:
            if getattr(self, name) < 0:
                raise BudgetError(f"{name} must not be negative")

    def plus(self, other: BudgetUsage) -> BudgetUsage:
        return BudgetUsage(
            candidates=self.candidates + other.candidates,
            generator_tokens=self.generator_tokens + other.generator_tokens,
            sandbox_micros=self.sandbox_micros + other.sandbox_micros,
            usd_micros=self.usd_micros + other.usd_micros,
        )

    def minus(self, other: BudgetUsage) -> BudgetUsage:
        return BudgetUsage(
            candidates=self.candidates - other.candidates,
            generator_tokens=self.generator_tokens - other.generator_tokens,
            sandbox_micros=self.sandbox_micros - other.sandbox_micros,
            usd_micros=self.usd_micros - other.usd_micros,
        )

    def exceeds(self, limits: BudgetSpec) -> str | None:
        """The first dimension over its limit, or None."""
        for name in DIMENSIONS:
            if getattr(self, name) > getattr(limits, f"max_{name}"):
                return name
        return None

    @property
    def sandbox_seconds(self) -> float:
        return self.sandbox_micros / 1_000_000

    @property
    def usd(self) -> float:
        return self.usd_micros / 1_000_000

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in DIMENSIONS}


class BudgetSpec(BaseModel):
    """One budget level. Every strategy in a comparison runs against the same instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_candidates: int = Field(ge=0, le=100_000)
    max_generator_tokens: int = Field(ge=0, le=100_000_000)
    max_sandbox_micros: int = Field(ge=0, le=10_000_000_000)
    max_usd_micros: int = Field(ge=0, le=1_000_000_000)

    @classmethod
    def of(
        cls,
        *,
        candidates: int,
        tokens: int,
        sandbox_seconds: float,
        usd: float,
    ) -> BudgetSpec:
        return cls(
            max_candidates=candidates,
            max_generator_tokens=tokens,
            max_sandbox_micros=round(sandbox_seconds * 1_000_000),
            max_usd_micros=round(usd * 1_000_000),
        )

    @property
    def limits(self) -> BudgetUsage:
        return BudgetUsage(
            candidates=self.max_candidates,
            generator_tokens=self.max_generator_tokens,
            sandbox_micros=self.max_sandbox_micros,
            usd_micros=self.max_usd_micros,
        )

    def scaled_candidates(self, candidates: int) -> BudgetSpec:
        """The same budget with a different candidate cap; used to build sweep levels."""
        return self.model_copy(update={"max_candidates": candidates})


class Reservation:
    """A pending charge. Exactly one of :meth:`settle` or :meth:`release` may be called."""

    __slots__ = ("_accountant", "_closed", "_reserved")

    def __init__(self, accountant: BudgetAccountant, reserved: BudgetUsage) -> None:
        self._accountant = accountant
        self._reserved = reserved
        self._closed = False

    @property
    def reserved(self) -> BudgetUsage:
        return self._reserved

    @property
    def closed(self) -> bool:
        return self._closed

    def settle(
        self,
        *,
        candidates: int | None = None,
        generator_tokens: int | None = None,
        sandbox_micros: int | None = None,
        usd_micros: int | None = None,
    ) -> BudgetUsage:
        """Commit the actual spend. Omitted dimensions settle at what was reserved."""
        if self._closed:
            raise BudgetError("reservation is already settled or released")
        actual = BudgetUsage(
            candidates=self._reserved.candidates if candidates is None else candidates,
            generator_tokens=(
                self._reserved.generator_tokens if generator_tokens is None else generator_tokens
            ),
            sandbox_micros=(
                self._reserved.sandbox_micros if sandbox_micros is None else sandbox_micros
            ),
            usd_micros=self._reserved.usd_micros if usd_micros is None else usd_micros,
        )
        over = [
            name for name in DIMENSIONS if getattr(actual, name) > getattr(self._reserved, name)
        ]
        if over:
            raise BudgetError(
                f"settled more than was reserved for: {', '.join(over)}; "
                "reserve the worst case before spending"
            )
        self._closed = True
        self._accountant._commit(self._reserved, actual)
        return actual

    def release(self) -> None:
        """Give the whole reservation back: the step did not happen."""
        if self._closed:
            raise BudgetError("reservation is already settled or released")
        self._closed = True
        self._accountant._commit(self._reserved, BudgetUsage())

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._closed:
            self.release()


class BudgetAccountant:
    """Enforces one :class:`BudgetSpec`. Single-threaded by design; one search owns one instance."""

    def __init__(self, spec: BudgetSpec, *, sandbox_usd_per_second: float = 0.0) -> None:
        if sandbox_usd_per_second < 0:
            raise BudgetError("sandbox_usd_per_second must not be negative")
        self._spec = spec
        self._sandbox_usd_per_second = sandbox_usd_per_second
        self._committed = BudgetUsage()
        self._outstanding = BudgetUsage()
        self._refusals: list[str] = []

    @property
    def spec(self) -> BudgetSpec:
        return self._spec

    @property
    def committed(self) -> BudgetUsage:
        return self._committed

    @property
    def outstanding(self) -> BudgetUsage:
        return self._outstanding

    @property
    def refusals(self) -> tuple[str, ...]:
        """Dimensions that have refused a reservation, in order."""
        return tuple(self._refusals)

    @property
    def remaining(self) -> BudgetUsage:
        return self._spec.limits.minus(self._committed.plus(self._outstanding))

    def invariant_holds(self) -> bool:
        """``committed <= limits`` and ``committed + outstanding <= limits``, in every dimension."""
        total = self._committed.plus(self._outstanding)
        return self._committed.exceeds(self._spec) is None and total.exceeds(self._spec) is None

    def reserve(self, request: BudgetUsage) -> Reservation:
        """Hold ``request``, or raise :class:`BudgetExhausted` without changing anything."""
        projected = self._committed.plus(self._outstanding).plus(request)
        dimension = projected.exceeds(self._spec)
        if dimension is not None:
            self._refusals.append(dimension)
            raise BudgetExhausted(
                dimension, getattr(self._spec, f"max_{dimension}"), getattr(projected, dimension)
            )
        self._outstanding = self._outstanding.plus(request)
        return Reservation(self, request)

    def reserve_generation(
        self, *, model: str, input_tokens: int, max_output_tokens: int
    ) -> Reservation:
        """Reserve one generator call at its worst case: estimated input plus full output."""
        tokens = input_tokens + max_output_tokens
        usd = cost_usd_micros(model, input_tokens, max_output_tokens)
        return self.reserve(BudgetUsage(generator_tokens=tokens, usd_micros=usd))

    def reserve_evaluation(self, *, timeout_seconds: float) -> Reservation:
        """Reserve one candidate evaluation at its worst case: the sandbox's own wall timeout."""
        micros = math.ceil(timeout_seconds * 1_000_000)
        usd = self.sandbox_usd_micros(micros)
        return self.reserve(BudgetUsage(candidates=1, sandbox_micros=micros, usd_micros=usd))

    def sandbox_usd_micros(self, micros: int) -> int:
        return math.ceil(micros / 1_000_000 * self._sandbox_usd_per_second * 1_000_000)

    def settle_generation(
        self, reservation: Reservation, *, model: str, input_tokens: int, output_tokens: int
    ) -> BudgetUsage:
        return reservation.settle(
            generator_tokens=input_tokens + output_tokens,
            usd_micros=min(
                cost_usd_micros(model, input_tokens, output_tokens),
                reservation.reserved.usd_micros,
            ),
        )

    def settle_evaluation(self, reservation: Reservation, *, seconds: float) -> BudgetUsage:
        micros = min(math.ceil(max(0.0, seconds) * 1_000_000), reservation.reserved.sandbox_micros)
        return reservation.settle(
            candidates=1,
            sandbox_micros=micros,
            usd_micros=min(self.sandbox_usd_micros(micros), reservation.reserved.usd_micros),
        )

    def _commit(self, reserved: BudgetUsage, actual: BudgetUsage) -> None:
        self._outstanding = self._outstanding.minus(reserved)
        self._committed = self._committed.plus(actual)

    def snapshot(self) -> dict[str, int]:
        return replace(self._committed).as_dict()
