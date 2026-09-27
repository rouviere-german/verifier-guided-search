"""Ablations: what each knob of the MCTS configuration is actually worth.

The sweep answers "does search beat sampling at the same budget". The ablations answer "which part
of the search is doing the work", which is the more falsifiable question and the one that can
embarrass the design. Three knobs, all on MCTS, all at a single budget level so the numbers are
comparable:

``c_puct``
    The exploration weight. At ``0`` selection is pure argmax over the value estimate, so PUCT
    degenerates into a greedy descent over the value model and the tree stops widening. Large values
    ignore the value model and spread visits by prior alone.

    **Only the ``0`` versus ``> 0`` contrast is identifiable with the value model on.** The PUCT
    score is ``Q + c * P * sqrt(N) / (1 + n)``, so when ``Q`` is equal across siblings the argmax
    reduces to ``P / (1 + n)`` and every positive ``c`` produces the *same* ordering. This value
    model is saturated - it scores almost every candidate ~0.0001 - so ``Q`` is effectively constant
    and the positive rows are mechanically identical, which the results file shows directly: the
    ``0.7``, ``1.4`` and ``2.8`` rows agree to sixteen significant figures in
    ``mean_sandbox_seconds``, i.e. they ran the same programs in the same order. That is a stronger
    statement than "exploration does not help", and it is the reason the grid is also swept with
    ``use_value_model=False`` (knob ``c_puct_value_model_off``), where ``Q`` is the graded visible
    pass fraction, siblings differ, and the exploration constant really does change the argmax.
``children_per_expansion`` (k)
    How many candidates an expansion asks for. Small k means more expansions per candidate, which
    costs tokens; large k means a wide, shallow tree. With the candidate budget fixed, k trades tree
    breadth against depth.
``use_value_model``
    On/off. Off, a leaf's value is its visible pass fraction; on, it is the model's probability that
    the candidate passes the hidden tests. This is the ablation that says whether the learned part
    earns its place, and it is run against a model trained on a disjoint set of tasks
    (:mod:`verifier_guided_search.experiments.train`), so a gain here is not the model recognising
    its own training data.

Every variant runs on the same tasks with the same budget and the same generator seed, so the only
difference between two rows is the knob. Confidence intervals are bootstrap over tasks, and the
report prints them next to the estimate because with 60 tasks most of these gaps are not significant
on their own - which is itself worth saying out loud rather than hiding behind a point estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.experiments.sweep import (
    LevelStats,
    SweepConfig,
    SweepDeps,
    aggregate,
    run_sweep,
)
from verifier_guided_search.tasks.suite import TaskSuite

__all__ = [
    "C_PUCT_OFF_KNOB",
    "DEFAULT_C_PUCT_GRID",
    "DEFAULT_K_GRID",
    "AblationResults",
    "AblationRow",
    "run_ablations",
]

DEFAULT_C_PUCT_GRID: Final[tuple[float, ...]] = (0.0, 0.7, 1.4, 2.8)
DEFAULT_K_GRID: Final[tuple[int, ...]] = (2, 4, 8)
_ABLATED_STRATEGY: Final = "mcts"
#: Knob name for the c_puct grid run without the value model, where the constant is identifiable.
C_PUCT_OFF_KNOB: Final = "c_puct_value_model_off"


class AblationRow(BaseModel):
    """One variant: which knob, which value, and how the strategy did with it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    knob: str = Field(pattern=r"^[a-z_]+$")
    value: str
    stats: LevelStats

    @property
    def label(self) -> str:
        return f"{self.knob}={self.value}"


class AblationResults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    strategy: str = _ABLATED_STRATEGY
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    candidates_budget: int = Field(ge=1)
    value_model_version: str | None = None
    rows: tuple[AblationRow, ...] = Field(min_length=1)

    def by_knob(self, knob: str) -> tuple[AblationRow, ...]:
        return tuple(row for row in self.rows if row.knob == knob)

    @property
    def knobs(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for row in self.rows:
            seen.setdefault(row.knob, None)
        return tuple(seen)


def _variant(
    suite: TaskSuite,
    base: SweepConfig,
    deps: SweepDeps,
    *,
    knob: str,
    value: str,
    overrides: dict[str, object],
    budget: int,
) -> AblationRow:
    config = base.model_copy(update={"strategies": (_ABLATED_STRATEGY,), **overrides})
    variant_deps = SweepDeps(
        sandbox=deps.sandbox,
        generator=deps.generator,
        # The value-model ablation is the only one that touches this.
        value_model=None if overrides.get("use_value_model") is False else deps.value_model,
        traces=None,  # ablation runs are not training data
        metrics=deps.metrics,
        audit=deps.audit,
    )
    results = run_sweep(suite, config, variant_deps)
    return AblationRow(knob=knob, value=value, stats=results.level(_ABLATED_STRATEGY, budget))


def run_ablations(
    suite: TaskSuite,
    base: SweepConfig,
    deps: SweepDeps,
    *,
    budget: int | None = None,
    c_puct_grid: Sequence[float] = DEFAULT_C_PUCT_GRID,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    include_value_model: bool = True,
) -> AblationResults:
    """Sweep each knob in turn, holding everything else at ``base``.

    ``budget`` must be one of ``base.budget_levels`` (the largest by default), because a level is
    read off a run rather than run separately.
    """
    level = base.budget_levels[-1] if budget is None else budget
    if level not in base.budget_levels:
        raise ValueError(f"budget {level} is not one of the configured levels {base.budget_levels}")
    rows: list[AblationRow] = []
    for value in c_puct_grid:
        rows.append(
            _variant(
                suite,
                base,
                deps,
                knob="c_puct",
                value=f"{value:g}",
                overrides={"c_puct": value},
                budget=level,
            )
        )
    for value in c_puct_grid:
        # The identifiable version of the same sweep: with the value model off, Q is the graded
        # visible pass fraction, so siblings differ and c_puct changes which child is selected.
        rows.append(
            _variant(
                suite,
                base,
                deps,
                knob=C_PUCT_OFF_KNOB,
                value=f"{value:g}",
                overrides={"c_puct": value, "use_value_model": False},
                budget=level,
            )
        )
    for k in k_grid:
        rows.append(
            _variant(
                suite,
                base,
                deps,
                knob="children_per_expansion",
                value=str(k),
                overrides={"children_per_expansion": k},
                budget=level,
            )
        )
    if include_value_model:
        for flag in (True, False):
            rows.append(
                _variant(
                    suite,
                    base,
                    deps,
                    knob="use_value_model",
                    value="on" if flag else "off",
                    overrides={"use_value_model": flag},
                    budget=level,
                )
            )
    return AblationResults(
        suite_sha256=suite.sha256,
        task_count=len(suite),
        candidates_budget=level,
        value_model_version=deps.value_model.version if deps.value_model is not None else None,
        rows=tuple(rows),
    )


def best_value(rows: Sequence[AblationRow]) -> AblationRow:
    """The variant with the highest hidden solve rate; ties go to the first in grid order."""
    if not rows:
        raise ValueError("no ablation rows")
    return max(rows, key=lambda row: (row.stats.hidden_solve_rate, -row.stats.mean_candidates_used))


def aggregate_at(rows: Sequence[AblationRow], knob: str) -> tuple[float, float]:
    """(lowest, highest) hidden solve rate observed for ``knob``. The spread is the knob's worth."""
    selected = [row.stats.hidden_solve_rate for row in rows if row.knob == knob]
    if not selected:
        raise ValueError(f"no rows for knob {knob!r}")
    return min(selected), max(selected)


# Re-exported so the report module does not need to import sweep just for this.
aggregate_level = aggregate
