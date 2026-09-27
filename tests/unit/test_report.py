"""The Markdown renderers: every number in the docs comes from here, so the tables are pinned.

The renderers exist so no figure in the README or docs/EXPERIMENTS.md is retyped by hand. These
tests check the parts that a reader would be misled by if they drifted: the confidence interval is
printed next to every rate, the compute-to-target cell says between which two levels the crossing
happened rather than implying the exact number was measured, and a value model that misses its
target is labelled as missing it.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.experiments.ablations import AblationResults, AblationRow, best_value
from verifier_guided_search.experiments.report import (
    render_ablations,
    render_compute_to_target,
    render_escape_suite,
    render_experiments_document,
    render_solve_rate_table,
    render_summary_lines,
    render_value_model_card,
)
from verifier_guided_search.experiments.sweep import LevelStats, SweepConfig, SweepResults
from verifier_guided_search.experiments.train import TrainedValueModel, ValueModelMetrics
from verifier_guided_search.sandbox.escape import ESCAPE_SUITE, EscapeOutcome, escape_job
from verifier_guided_search.sandbox.protocol import (
    GroupResult,
    TestResult,
    Verdict,
)
from verifier_guided_search.value.model import constant_value_model

pytestmark = pytest.mark.unit

_SUITE_HASH = "a" * 64


def level(
    strategy: str,
    candidates: int,
    *,
    hidden: float,
    visible: float | None = None,
    false_pass: float = 0.1,
) -> LevelStats:
    return LevelStats(
        strategy=strategy,
        candidates_budget=candidates,
        tasks=18,
        hidden_solved=round(hidden * 18),
        visible_solved=round((visible if visible is not None else hidden) * 18),
        false_passes=round(false_pass * 18),
        hidden_solve_rate=hidden,
        hidden_ci_low=max(0.0, hidden - 0.2),
        hidden_ci_high=min(1.0, hidden + 0.2),
        visible_solve_rate=visible if visible is not None else hidden,
        false_pass_rate=false_pass,
        mean_candidates_used=candidates * 0.7,
        mean_sandbox_seconds=candidates * 0.2,
        mean_generator_tokens=candidates * 100.0,
    )


@pytest.fixture
def results() -> SweepResults:
    config = SweepConfig(
        seed=20260927,
        instances_per_family=2,
        strategies=("greedy", "mcts"),
        budget_levels=(1, 4, 16),
    )
    levels = (
        level("greedy", 1, hidden=0.0),
        level("greedy", 4, hidden=0.2),
        level("greedy", 16, hidden=0.35),
        level("mcts", 1, hidden=0.0),
        level("mcts", 4, hidden=0.3),
        level("mcts", 16, hidden=0.65, visible=0.8, false_pass=0.22),
    )
    return SweepResults(
        config=config,
        suite_sha256=_SUITE_HASH,
        task_count=18,
        family_count=11,
        generator="mutation",
        generator_model="fake-mutation",
        value_model_version="b" * 64,
        levels=levels,
        compute_to_target={"greedy": None, "mcts": 16},
        total_evaluations=630,
        total_sandbox_seconds=231.4,
        wall_seconds=96.2,
        cache_hit_ratio=0.572,
    )


@pytest.fixture
def trained() -> TrainedValueModel:
    return TrainedValueModel(
        model=constant_value_model(0.4),
        metrics=ValueModelMetrics(
            auc=0.952,
            ece=0.081,
            brier=0.07,
            train_rows=120,
            holdout_rows=49,
            train_tasks=8,
            holdout_tasks=4,
            train_positive_rate=0.3,
            holdout_positive_rate=0.28,
        ),
    )


def test_the_solve_rate_table_has_one_row_per_strategy_and_level(results: SweepResults) -> None:
    table = render_solve_rate_table(results)
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == 1 + len(results.levels)  # header plus the data rows
    assert "hidden solve rate" in rows[0]
    assert "verifier false-pass rate" in rows[0]
    assert "| mcts | 16 | 65.0% |" in table


def test_every_rate_is_printed_with_its_confidence_interval(results: SweepResults) -> None:
    """A table of point estimates alone invites a claim the data does not support."""
    table = render_solve_rate_table(results)
    for stats in results.levels:
        interval = f"[{100 * stats.hidden_ci_low:.1f}%, {100 * stats.hidden_ci_high:.1f}%]"
        assert interval in table, f"{stats.label} has no interval"


def test_the_compute_to_target_table_says_between_which_levels_the_crossing_was(
    results: SweepResults,
) -> None:
    table = render_compute_to_target(results)
    assert "16 (crossed between 4 and 16)" in table
    assert "not reached at 16" in table
    assert "60.0% hidden solve rate" in table


def test_a_crossing_at_the_first_level_reports_a_lower_bound_of_zero() -> None:
    config = SweepConfig(
        seed=1, instances_per_family=1, strategies=("mcts",), budget_levels=(1, 2, 4, 8, 16)
    )
    results = SweepResults(
        config=config,
        suite_sha256=_SUITE_HASH,
        task_count=4,
        family_count=4,
        generator="mutation",
        generator_model="fake-mutation",
        levels=(level("mcts", 1, hidden=0.9),),
        compute_to_target={"mcts": 1},
        total_evaluations=4,
        total_sandbox_seconds=1.0,
        wall_seconds=1.0,
        cache_hit_ratio=0.0,
    )
    assert "1 (crossed between 0 and 1)" in render_compute_to_target(results)


def test_the_provenance_block_pins_the_seed_the_hashes_and_the_versions(
    results: SweepResults,
) -> None:
    lines = "\n".join(render_summary_lines(results))
    assert "seed: 20260927" in lines
    assert _SUITE_HASH[:16] in lines
    assert ("b" * 64)[:16] in lines
    assert "tasks: 18 from 11 families" in lines
    assert "630 evaluations" in lines
    assert "57.2%" in lines


def test_a_run_without_a_value_model_says_so(results: SweepResults) -> None:
    without = results.model_copy(update={"value_model_version": None})
    assert "`n/a`" in "\n".join(render_summary_lines(without))


def test_the_value_model_card_reports_the_target_verdict(trained: TrainedValueModel) -> None:
    card = render_value_model_card(trained)
    assert "0.952" in card
    assert "meets the 0.75 target" in card
    assert "ECE: 0.081" in card
    assert "49 rows from 4 held-out tasks" in card
    assert "split is by task, not by row" in card


def test_a_value_model_below_the_target_is_labelled_as_missing_it(
    trained: TrainedValueModel,
) -> None:
    weak = TrainedValueModel(
        model=trained.model, metrics=trained.metrics.model_copy(update={"auc": 0.61})
    )
    card = render_value_model_card(weak)
    assert "MISSES the 0.75 target" in card
    assert not weak.metrics.meets_target


def test_the_value_model_card_lists_the_features_it_splits_on(
    trained: TrainedValueModel,
) -> None:
    card = render_value_model_card(trained, top_features=3)
    assert "| feature | share of splits |" in card
    rows = [line for line in card.splitlines() if line.startswith("| ") and "%" in line]
    assert len(rows) == 3


def test_the_ablation_tables_are_grouped_by_knob() -> None:
    rows = (
        AblationRow(knob="c_puct", value="0", stats=level("mcts", 8, hidden=0.45)),
        AblationRow(knob="c_puct", value="1.4", stats=level("mcts", 8, hidden=0.27)),
        AblationRow(knob="use_value_model", value="on", stats=level("mcts", 8, hidden=0.27)),
        AblationRow(knob="use_value_model", value="off", stats=level("mcts", 8, hidden=0.27)),
    )
    ablations = AblationResults(
        suite_sha256=_SUITE_HASH, task_count=11, candidates_budget=8, rows=rows
    )
    assert ablations.knobs == ("c_puct", "use_value_model")
    text = render_ablations(ablations)
    assert "**c_puct**" in text
    assert "**use_value_model**" in text
    assert "budget 8 candidates, 11 tasks" in text
    assert "| 0 | 45.0% |" in text
    assert best_value(ablations.by_knob("c_puct")).value == "0"


def test_the_escape_table_reports_what_stopped_each_attempt() -> None:
    outcomes = [
        EscapeOutcome(
            attempt=attempt,
            verdict=Verdict(
                program_sha256="c" * 64,
                suite_sha256=escape_job(attempt).suite_sha256,
                status="blocked",
                visible=GroupResult(
                    results=(
                        TestResult(name="escaped", status="error", error_class="SandboxViolation"),
                    )
                ),
            ),
        )
        for attempt in ESCAPE_SUITE[:3]
    ]
    table = render_escape_suite(outcomes)
    assert "| attempt | category | stopped by | verdict status | blocked |" in table
    assert "blocked: 100.0% of 3 attempts" in table
    assert "refused by the in-process guard" in table
    assert " NO |" not in table


def test_an_escape_that_succeeded_is_marked_clearly() -> None:
    attempt = ESCAPE_SUITE[0]
    escaped = EscapeOutcome(
        attempt=attempt,
        verdict=Verdict(
            program_sha256="c" * 64,
            suite_sha256=escape_job(attempt).suite_sha256,
            status="ok",
            visible=GroupResult(results=(TestResult(name="escaped", status="pass"),)),
        ),
    )
    table = render_escape_suite([escaped])
    assert "| NO |" in table
    assert "blocked: 0.0%" in table


def test_the_whole_document_is_assembled_with_every_section(
    results: SweepResults, trained: TrainedValueModel
) -> None:
    rows = (AblationRow(knob="c_puct", value="1.4", stats=level("mcts", 8, hidden=0.3)),)
    document = render_experiments_document(
        results,
        ablations=AblationResults(
            suite_sha256=_SUITE_HASH, task_count=11, candidates_budget=8, rows=rows
        ),
        trained=trained,
        escape=[],
        command="uv run python bench/bench_search.py",
        machine="Windows-10, 6 CPUs",
    )
    for heading in (
        "# Experiments",
        "## Run provenance",
        "## Hidden-test solve rate against budget",
        "## Compute to reach the target solve rate",
        "## Value model",
        "## Ablations",
    ):
        assert heading in document
    assert "Do not edit by hand" in document
    assert "uv run python bench/bench_search.py" in document
    assert "Windows-10, 6 CPUs" in document
    assert document.endswith("\n")


def test_the_document_omits_sections_it_has_no_data_for(results: SweepResults) -> None:
    document = render_experiments_document(results)
    assert "## Value model" not in document
    assert "## Ablations" not in document
    assert "## Sandbox escape suite" not in document
    assert "## Hidden-test solve rate against budget" in document


def test_an_empty_escape_run_is_omitted_rather_than_reported_as_zero(
    results: SweepResults,
) -> None:
    """Reporting "0% blocked" for a suite that never ran would be worse than saying nothing."""
    document = render_experiments_document(results, escape=[])
    assert "## Sandbox escape suite" not in document
