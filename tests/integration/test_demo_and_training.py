"""The offline demo and the training pipeline, end to end, with no key and no network.

The demo is the repo's "does the whole thing actually work" test: benchmark, traces, a trained value
model in a content-addressed registry, a budget-matched comparison, the escape suite, and a patch
that is refused until a second principal approves the exact diff. It is run once here and asserted
on as data rather than by scraping stdout.

The training tests pin the property the value model's numbers depend on: the tasks it is trained on
and the tasks it is reported on are disjoint, at both levels of the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier_guided_search.agentic.audit import verify_file
from verifier_guided_search.demo import main as demo_main
from verifier_guided_search.demo import run_demo
from verifier_guided_search.experiments.sweep import SweepConfig, SweepDeps, default_generator
from verifier_guided_search.experiments.train import (
    TARGET_AUC,
    collect_traces,
    feature_importance,
    load_or_train,
    split_suite,
    train_from_traces,
    train_value_model,
)
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.tasks.suite import TaskSuite, build_suite
from verifier_guided_search.value.features import FEATURE_NAMES
from verifier_guided_search.value.registry import ValueModelRegistry

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

_SEED = 20260927


# ---------------------------------------------------------------------------- splitting


def test_the_trace_and_evaluation_splits_are_disjoint(small_suite: TaskSuite) -> None:
    trace, evaluation = split_suite(small_suite, trace_fraction=0.4, seed=_SEED)
    trace_ids = {task.task_id for task in trace}
    eval_ids = {task.task_id for task in evaluation}
    assert trace_ids & eval_ids == set()
    assert trace_ids | eval_ids == {task.task_id for task in small_suite}
    assert len(trace) >= 1
    assert len(evaluation) >= 1


def test_the_split_is_deterministic_and_seed_dependent(small_suite: TaskSuite) -> None:
    first = {task.task_id for task in split_suite(small_suite, seed=1)[0]}
    again = {task.task_id for task in split_suite(small_suite, seed=1)[0]}
    other = {task.task_id for task in split_suite(small_suite, seed=2)[0]}
    assert first == again
    assert first != other


def test_the_split_mixes_families_rather_than_cutting_by_index(small_suite: TaskSuite) -> None:
    """Ranking by task hash keeps both sides varied; slicing by index would not."""
    trace, evaluation = split_suite(small_suite, trace_fraction=0.4, seed=_SEED)
    assert len({task.family for task in trace}) >= 3
    assert len({task.family for task in evaluation}) >= 3


def test_the_split_validates_its_arguments(small_suite: TaskSuite) -> None:
    for fraction in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="between 0 and 1"):
            split_suite(small_suite, trace_fraction=fraction)
    from verifier_guided_search.tasks.families import FAMILIES

    one_task = build_suite(instances_per_family=1, families=FAMILIES[:1])
    with pytest.raises(ValueError, match="at least two tasks"):
        split_suite(one_task)


# ---------------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def trained(
    shared_sandbox: CachingSandbox, small_suite: TaskSuite
) -> tuple[object, object, object]:
    """Collect traces on the trace split and train once; several tests read the result.

    Uses the session sandbox, so candidates other tests already ran cost a cache lookup.
    """
    sandbox = shared_sandbox
    trace_suite, eval_suite = split_suite(small_suite, seed=_SEED)
    config = SweepConfig(
        seed=_SEED,
        instances_per_family=1,
        strategies=("greedy", "mcts"),
        budget_levels=(1, 2, 4, 8, 16),
        bootstrap_resamples=0,
    )
    generator = default_generator(config)
    store = collect_traces(trace_suite, config, SweepDeps(sandbox=sandbox, generator=generator))
    model = train_from_traces(store, seed=_SEED, notes="integration test")
    return store, model, eval_suite


def test_traces_are_collected_with_the_value_model_off(
    trained: tuple[object, object, object],
) -> None:
    """Training data must not be shaped by an earlier version of the model."""
    store, _, _ = trained
    assert len(store) >= 40  # type: ignore[arg-type]
    rows = store.rows  # type: ignore[attr-defined]
    assert {row.strategy for row in rows} == {"greedy", "mcts"}
    assert all(row.feature_names == FEATURE_NAMES for row in rows)
    assert {row.label for row in rows} == {0, 1}, "traces must contain both classes"


def test_the_model_reaches_the_target_auc_on_held_out_tasks(
    trained: tuple[object, object, object],
) -> None:
    _, model, _ = trained
    metrics = model.metrics  # type: ignore[attr-defined]
    assert metrics.auc >= TARGET_AUC, f"held-out AUC {metrics.auc:.3f} below {TARGET_AUC}"
    assert metrics.meets_target
    assert 0.0 <= metrics.ece <= 1.0
    assert 0.0 <= metrics.brier <= 1.0
    assert metrics.train_tasks >= 1
    assert metrics.holdout_tasks >= 1
    assert metrics.train_rows + metrics.holdout_rows >= 40


def test_the_card_carries_the_metrics_it_was_accepted_on(
    trained: tuple[object, object, object],
) -> None:
    _, model, _ = trained
    card = model.model.card  # type: ignore[attr-defined]
    assert card.metrics["holdout_auc"] == pytest.approx(model.metrics.auc, abs=1e-6)  # type: ignore[attr-defined]
    assert card.feature_names == FEATURE_NAMES
    assert card.notes == "integration test"


def test_feature_importance_sums_to_one_and_names_real_features(
    trained: tuple[object, object, object],
) -> None:
    _, model, _ = trained
    ranked = feature_importance(model.model)  # type: ignore[attr-defined]
    assert {name for name, _ in ranked} == set(FEATURE_NAMES)
    assert sum(share for _, share in ranked) == pytest.approx(1.0)
    assert ranked == sorted(ranked, key=lambda item: (-item[1], item[0]))


def test_publishing_and_reloading_the_model_round_trips(
    trained: tuple[object, object, object], tmp_path: Path
) -> None:
    _, model, _ = trained
    registry = ValueModelRegistry(tmp_path / "value_models")
    version = model.save(registry, notes="published by a test")  # type: ignore[attr-defined]
    assert version == model.version  # type: ignore[attr-defined]
    assert registry.verify(version)
    reloaded = registry.load(version)
    features = [[0.3] * len(FEATURE_NAMES)]
    assert (
        reloaded.predict_proba(features).tolist()
        == model.model.predict_proba(  # type: ignore[attr-defined]
            features
        ).tolist()
    )
    assert registry.index().entries[0].metrics["holdout_auc"] > 0


def test_too_few_traces_are_refused_with_an_actionable_message() -> None:
    from verifier_guided_search.search.trace import TraceStore

    with pytest.raises(ValueError, match="collect at least"):
        train_from_traces(TraceStore(), seed=_SEED)


def test_a_single_class_holdout_is_refused(trained: tuple[object, object, object]) -> None:
    """AUC would be undefined, so the pipeline says so instead of reporting a number."""
    from verifier_guided_search.search.trace import TraceStore

    store, _, _ = trained
    single = TraceStore()
    for row in store.rows:  # type: ignore[attr-defined]
        single.append(row.model_copy(update={"hidden_all_passed": False}))
    with pytest.raises(ValueError, match="single class"):
        train_from_traces(single, seed=_SEED)


def test_load_or_train_reuses_a_verified_model(tmp_path: Path, tiny_suite: TaskSuite) -> None:
    from verifier_guided_search.sandbox.local import LocalSandbox

    registry = ValueModelRegistry(tmp_path / "value_models")
    config = SweepConfig(
        seed=_SEED,
        instances_per_family=1,
        strategies=("greedy",),
        budget_levels=(1, 2),
        bootstrap_resamples=0,
    )
    deps = SweepDeps(
        sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox")),
        generator=default_generator(config),
    )
    from verifier_guided_search.value.model import constant_value_model

    version = registry.save(constant_value_model(0.5))
    loaded = load_or_train(registry, tiny_suite, config, deps)
    assert loaded.version == version, "an existing verified model must be reused, not retrained"


def test_load_or_train_refuses_a_pinned_version_that_is_missing(
    tmp_path: Path, tiny_suite: TaskSuite
) -> None:
    from verifier_guided_search.errors import ValueModelError
    from verifier_guided_search.sandbox.local import LocalSandbox

    registry = ValueModelRegistry(tmp_path / "value_models")
    config = SweepConfig(seed=_SEED, instances_per_family=1, budget_levels=(1,))
    deps = SweepDeps(
        sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox")),
        generator=default_generator(config),
    )
    with pytest.raises(ValueModelError, match="missing or failed its hash check"):
        load_or_train(registry, tiny_suite, config, deps, version="b" * 64)


def test_train_value_model_publishes_to_the_registry(
    tmp_path: Path, small_suite: TaskSuite
) -> None:
    from verifier_guided_search.sandbox.local import LocalSandbox

    registry = ValueModelRegistry(tmp_path / "value_models")
    trace_suite, _ = split_suite(small_suite, seed=_SEED)
    config = SweepConfig(
        seed=_SEED,
        instances_per_family=1,
        strategies=("greedy", "mcts"),
        budget_levels=(1, 2, 4, 8, 16),
        bootstrap_resamples=0,
    )
    deps = SweepDeps(
        sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox")),
        generator=default_generator(config),
    )
    result = train_value_model(trace_suite, config, deps, registry=registry, notes="published")
    assert registry.current_version == result.version
    assert registry.verify()


# ---------------------------------------------------------------------------- the demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory, shared_sandbox: CachingSandbox) -> object:
    """One full demo run over all 15 families, shared by every assertion about it."""
    return run_demo(
        tmp_path_factory.mktemp("demo"),
        instances_per_family=1,
        quiet=True,
        sandbox=shared_sandbox,
    )


def test_the_demo_runs_end_to_end_offline(demo: object) -> None:
    results = demo.results  # type: ignore[attr-defined]
    assert results.task_count >= 5
    assert results.family_count >= 5
    assert set(results.config.strategies) == {"greedy", "best_of_n", "beam", "mcts"}
    assert results.total_evaluations > 0
    assert len(results.levels) == len(results.config.strategies) * len(results.config.budget_levels)


def test_the_demo_trains_and_publishes_a_value_model(demo: object) -> None:
    trained = demo.trained  # type: ignore[attr-defined]
    assert trained is not None, "the demo did not train a value model"
    assert demo.results.value_model_version == trained.version  # type: ignore[attr-defined]
    assert 0.0 <= trained.metrics.auc <= 1.0


def test_the_demo_blocks_the_whole_escape_suite(demo: object) -> None:
    assert demo.escape_blocked == 1.0  # type: ignore[attr-defined]
    assert demo.escape_attempts >= 12  # type: ignore[attr-defined]


def test_the_demo_patch_is_refused_before_it_is_approved(demo: object) -> None:
    patch = demo.patch  # type: ignore[attr-defined]
    assert patch is not None, "the demo found no winning candidate to apply"
    assert demo.approval_request_id is not None, "the apply was not refused first"  # type: ignore[attr-defined]
    assert demo.applied is True  # type: ignore[attr-defined]
    assert len(patch.diff_sha256) == 64
    assert patch.target == "solution.py"


def test_the_demos_audit_chain_verifies(demo: object) -> None:
    assert demo.audit_ok is True  # type: ignore[attr-defined]
    assert demo.audit_entries > demo.results.total_evaluations  # type: ignore[attr-defined]
    assert len(demo.audit_head) == 64  # type: ignore[attr-defined]


def test_the_demo_summary_is_json_serialisable(demo: object) -> None:
    payload = demo.summary()  # type: ignore[attr-defined]
    text = json.dumps(payload, sort_keys=True)
    assert '"escape_suite"' in text
    assert payload["escape_suite"]["blocked_fraction"] == 1.0
    assert payload["audit"]["verified"] is True
    assert payload["patch"]["applied"] is True


def test_no_approve_leaves_the_patch_pending(
    tmp_path: Path, shared_sandbox: CachingSandbox
) -> None:
    """What a real deployment looks like: the diff waits for a human."""
    result = run_demo(
        tmp_path / "pending",
        instances_per_family=1,
        approve=False,
        quiet=True,
        sandbox=shared_sandbox,
    )
    assert result.patch is not None
    assert result.approval_request_id is not None
    assert result.applied is False
    workspace = tmp_path / "pending" / "workspace" / "solution.py"
    assert workspace.read_text(encoding="utf-8") != result.patch.after
    assert verify_file(tmp_path / "pending" / "audit.jsonl").ok


def test_the_demo_writes_its_artifacts_where_it_says(demo: object) -> None:
    root = demo.workdir  # type: ignore[attr-defined]
    assert (root / "audit.jsonl").is_file()
    assert (root / "approvals.json").is_file()
    assert (root / "value_models" / "index.json").is_file()
    assert (root / "workspace" / "solution.py").is_file()


def test_the_demo_cli_prints_a_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four families keep this cheap; the full-suite run is the module fixture above."""
    code = demo_main(
        [
            "--workdir",
            str(tmp_path / "cli"),
            "--instances-per-family",
            "1",
            "--families",
            "4",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tasks"] >= 2
    assert payload["escape_suite"]["blocked_fraction"] == 1.0
