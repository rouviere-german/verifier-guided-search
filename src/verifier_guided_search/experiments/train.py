"""Collecting search traces and training the value model on them, without training on the answer.

Two leakage problems have to be solved at once, and they pull in opposite directions.

**The label must not be available to the search that produced the row.** It is not: a trace row's
features come from :class:`~verifier_guided_search.sandbox.protocol.VisibleFeedback` at the moment
the
candidate was scored, and the label (did it pass the hidden tests) is attached afterwards from the
full verdict. The feature function has no parameter through which a hidden outcome could arrive.

**The model must not be evaluated on tasks it was trained on.** Two mutants of the same reference
solution share most of their feature vector, so a row-level split would put near-duplicates on both
sides and report an AUC that means nothing. So the split is at the level of *tasks*, twice:

1. :func:`split_suite` divides the benchmark into a trace split and an evaluation split. Traces are
   collected only on the trace split, and the reported sweep runs only on the evaluation split. The
   value model therefore never saw any task it is later used on.
2. Inside the trace split, :func:`~verifier_guided_search.search.trace.split_by_task` holds out
   further tasks to measure AUD and ECE on, so the model card's numbers are out-of-sample too.

Trace collection runs with ``use_value_model=False``: the node values that drive the search are the
visible pass fractions, so the training data is not shaped by an earlier version of the model. That
keeps the "value model on/off" ablation meaningful instead of self-confirming.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.errors import ValueModelError
from verifier_guided_search.experiments.sweep import SweepConfig, SweepDeps, run_sweep
from verifier_guided_search.search.trace import TraceStore, TrainingSet, split_by_task
from verifier_guided_search.tasks.suite import TaskSuite
from verifier_guided_search.value.metrics import (
    brier_score,
    expected_calibration_error,
    roc_auc,
)
from verifier_guided_search.value.model import GradientBoostedValueModel
from verifier_guided_search.value.registry import ValueModelRegistry

__all__ = [
    "DEFAULT_TRACE_FRACTION",
    "TARGET_AUC",
    "TrainedValueModel",
    "ValueModelMetrics",
    "collect_traces",
    "split_suite",
    "train_from_traces",
    "train_value_model",
]

#: Share of tasks reserved for trace collection; the rest is what the sweep reports on.
DEFAULT_TRACE_FRACTION: Final = 0.4
#: The spec's acceptance bar for held-out AUC.
TARGET_AUC: Final = 0.75
_MIN_ROWS: Final = 40


def split_suite(
    suite: TaskSuite, *, trace_fraction: float = DEFAULT_TRACE_FRACTION, seed: int = 0
) -> tuple[TaskSuite, TaskSuite]:
    """Split the benchmark into (trace tasks, evaluation tasks), deterministically by task hash.

    Ranking by ``sha256(seed:task_id)`` rather than by index keeps families mixed across both sides,
    so neither split is accidentally all string tasks.
    """
    if not 0 < trace_fraction < 1:
        raise ValueError("trace_fraction must be between 0 and 1")
    tasks = list(suite)
    if len(tasks) < 2:
        raise ValueError("splitting a suite needs at least two tasks")
    ranked = sorted(tasks, key=lambda task: sha256_hex(f"{seed}:{task.task_id}"))
    cut = max(1, min(len(ranked) - 1, round(len(ranked) * trace_fraction)))
    trace_ids = {task.task_id for task in ranked[:cut]}
    trace = [task for task in tasks if task.task_id in trace_ids]
    evaluation = [task for task in tasks if task.task_id not in trace_ids]
    return TaskSuite(trace), TaskSuite(evaluation)


def collect_traces(suite: TaskSuite, config: SweepConfig, deps: SweepDeps) -> TraceStore:
    """Run the strategies over ``suite`` with the value model off, and return the trace rows.

    ``deps.traces`` is replaced by a fresh store, and ``deps.value_model`` is ignored, so a caller
    cannot accidentally collect training data that an earlier model already shaped.
    """
    store = TraceStore()
    collect_config = config.model_copy(update={"use_value_model": False})
    collect_deps = SweepDeps(
        sandbox=deps.sandbox,
        generator=deps.generator,
        value_model=None,
        traces=store,
        metrics=deps.metrics,
        audit=deps.audit,
        progress=deps.progress,
    )
    run_sweep(suite, collect_config, collect_deps)
    return store


class ValueModelMetrics(BaseModel):
    """Held-out classification metrics, plus the sizes they were measured on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auc: float = Field(ge=0, le=1)
    ece: float = Field(ge=0, le=1)
    brier: float = Field(ge=0, le=1)
    train_rows: int = Field(ge=1)
    holdout_rows: int = Field(ge=1)
    train_tasks: int = Field(ge=1)
    holdout_tasks: int = Field(ge=1)
    train_positive_rate: float = Field(ge=0, le=1)
    holdout_positive_rate: float = Field(ge=0, le=1)

    @property
    def meets_target(self) -> bool:
        return self.auc >= TARGET_AUC

    def as_flat(self) -> dict[str, float]:
        """A flat mapping for the value-model card and the registry index."""
        return {
            "holdout_auc": round(self.auc, 6),
            "holdout_ece": round(self.ece, 6),
            "holdout_brier": round(self.brier, 6),
            "train_rows": float(self.train_rows),
            "holdout_rows": float(self.holdout_rows),
            "train_positive_rate": round(self.train_positive_rate, 6),
            "holdout_positive_rate": round(self.holdout_positive_rate, 6),
        }


@dataclass(frozen=True, slots=True)
class TrainedValueModel:
    """A fitted model and the out-of-sample numbers it was accepted on."""

    model: GradientBoostedValueModel
    metrics: ValueModelMetrics

    @property
    def version(self) -> str:
        return self.model.version

    def save(self, registry: ValueModelRegistry, *, notes: str = "") -> str:
        return registry.save(self.model, metrics=self.metrics.as_flat(), notes=notes)


def _score(model: GradientBoostedValueModel, data: TrainingSet) -> list[float]:
    return [float(value) for value in model.predict_proba(data.features)]


def train_from_traces(
    store: TraceStore,
    *,
    seed: int = 0,
    holdout_fraction: float = 0.3,
    n_estimators: int = 80,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    notes: str = "",
) -> TrainedValueModel:
    """Fit on the trace split's training tasks and measure AUC, ECE and Brier on its held-out ones.

    Raises :class:`ValueModelError` with an actionable message when the traces cannot support a
    model (too few rows, one class only, a holdout with a single class).
    """
    data = store.training_set()
    if len(data) < _MIN_ROWS:
        raise ValueError(
            f"only {len(data)} trace rows; collect at least {_MIN_ROWS} before training "
            "(widen the budget or add tasks)"
        )
    train, holdout = split_by_task(data, holdout_fraction=holdout_fraction, seed=seed)
    if not train.both_classes_present():
        raise ValueError("the training split has a single class; widen the budget or add tasks")
    if not holdout.both_classes_present():
        raise ValueError("the holdout split has a single class; AUC would be undefined")
    model = GradientBoostedValueModel.train(
        train.features,
        train.labels,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        feature_names=train.feature_names,
        notes=notes,
    )
    scores = _score(model, holdout)
    metrics = ValueModelMetrics(
        auc=roc_auc(holdout.labels, scores),
        ece=expected_calibration_error(holdout.labels, scores),
        brier=brier_score(holdout.labels, scores),
        train_rows=len(train),
        holdout_rows=len(holdout),
        train_tasks=len(train.tasks),
        holdout_tasks=len(holdout.tasks),
        train_positive_rate=train.positive_rate,
        holdout_positive_rate=holdout.positive_rate,
    )
    # The metrics travel inside the card, so a saved model always carries what it was accepted on.
    carded = GradientBoostedValueModel(model.card.model_copy(update={"metrics": metrics.as_flat()}))
    return TrainedValueModel(model=carded, metrics=metrics)


def train_value_model(
    suite: TaskSuite,
    config: SweepConfig,
    deps: SweepDeps,
    *,
    registry: ValueModelRegistry | None = None,
    holdout_fraction: float = 0.3,
    notes: str = "",
) -> TrainedValueModel:
    """Collect traces on ``suite`` and train, optionally publishing to ``registry``."""
    store = collect_traces(suite, config, deps)
    trained = train_from_traces(
        store, seed=config.seed, holdout_fraction=holdout_fraction, notes=notes
    )
    if registry is not None:
        trained.save(registry, notes=notes)
    return trained


def load_or_train(
    registry: ValueModelRegistry,
    suite: TaskSuite,
    config: SweepConfig,
    deps: SweepDeps,
    *,
    version: str | None = None,
) -> GradientBoostedValueModel:
    """The current registry model if it verifies, otherwise train one and publish it."""
    wanted = version if version is not None else registry.current_version
    if wanted is not None and registry.verify(wanted):
        return registry.load(wanted)
    if version is not None:
        raise ValueModelError(f"value model {version[:12]} is missing or failed its hash check")
    return train_value_model(suite, config, deps, registry=registry).model


def feature_importance(model: GradientBoostedValueModel) -> list[tuple[str, float]]:
    """How often each feature is split on, normalised. Enough for a model card, not an
    explanation."""
    counts = dict.fromkeys(model.feature_names, 0.0)
    total = 0.0
    for tree in model.card.trees:
        for column in tree.feature:
            if column < 0:
                continue
            counts[model.feature_names[column]] += 1.0
            total += 1.0
    if total == 0:  # pragma: no cover - a model with no internal nodes
        return [(name, 0.0) for name in model.feature_names]
    ranked: Sequence[tuple[str, float]] = sorted(
        ((name, value / total) for name, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return list(ranked)


def value_model_paths(root: Path) -> ValueModelRegistry:
    """The registry under ``root``; kept here so callers do not hard-code the layout."""
    return ValueModelRegistry(root / "value_models")
