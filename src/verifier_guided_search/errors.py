"""Exception hierarchy for the parts of this repo that are not the shared agentic kit."""

from __future__ import annotations

__all__ = [
    "BudgetError",
    "BudgetExhausted",
    "PatchError",
    "SearchError",
    "ValueModelError",
    "VerifierGuidedSearchError",
]


class VerifierGuidedSearchError(Exception):
    """Base class for every error this project raises itself."""


class BudgetError(VerifierGuidedSearchError):
    """The budget accountant was used incorrectly (settled twice, over-settled, negative)."""


class BudgetExhausted(VerifierGuidedSearchError):
    """A reservation was refused because it would take a dimension past its limit.

    Raised *before* anything is spent, so the accountant's totals never exceed the budget.
    """

    def __init__(self, dimension: str, limit: int, attempted: int) -> None:
        super().__init__(f"{dimension} budget exhausted: {attempted} > {limit}")
        self.dimension = dimension
        self.limit = limit
        self.attempted = attempted


class SearchError(VerifierGuidedSearchError):
    """The search could not proceed (no candidates, an unknown strategy, a broken generator)."""


class ValueModelError(VerifierGuidedSearchError):
    """A value model could not be trained, exported, restored or verified."""


class PatchError(VerifierGuidedSearchError):
    """A patch could not be proposed or applied (path outside the workspace, stale diff)."""
