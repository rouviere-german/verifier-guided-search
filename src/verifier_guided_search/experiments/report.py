"""Rendering experiment results as Markdown, so no number in the docs is retyped by hand.

Every table the README and docs/EXPERIMENTS.md show is produced here from a
:class:`~verifier_guided_search.experiments.sweep.SweepResults`,
:class:`~verifier_guided_search.experiments.ablations.AblationResults` or
:class:`~verifier_guided_search.experiments.train.TrainedValueModel` object. The bench scripts write
both the JSON and the Markdown, and the Markdown is what gets pasted, so a stale README section
shows up as a diff rather than as a quiet inconsistency.

The renderers deliberately print confidence intervals next to every rate and say when a comparison
is inside them. A table that only shows point estimates invites a claim the data does not support.
"""

from __future__ import annotations

from collections.abc import Sequence

from verifier_guided_search.experiments.ablations import AblationResults
from verifier_guided_search.experiments.sweep import LevelStats, SweepResults
from verifier_guided_search.experiments.train import TrainedValueModel, feature_importance
from verifier_guided_search.sandbox.escape import EscapeOutcome, blocked_fraction

__all__ = [
    "render_ablations",
    "render_escape_suite",
    "render_experiments_document",
    "render_solve_rate_table",
    "render_summary_lines",
    "render_value_model_card",
]

_NA = "n/a"


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _ci(stats: LevelStats) -> str:
    return f"[{_pct(stats.hidden_ci_low)}, {_pct(stats.hidden_ci_high)}]"


def render_solve_rate_table(results: SweepResults) -> str:
    """Hidden solve rate against budget, one row per strategy, one column group per level."""
    lines = [
        "| strategy | budget (candidates) | hidden solve rate | 95% CI (bootstrap over tasks) "
        "| visible solve rate | verifier false-pass rate | mean candidates used "
        "| mean sandbox s |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for strategy in results.config.strategies:
        for stats in sorted(
            results.for_strategy(strategy), key=lambda item: item.candidates_budget
        ):
            lines.append(
                f"| {stats.strategy} | {stats.candidates_budget} "
                f"| {_pct(stats.hidden_solve_rate)} | {_ci(stats)} "
                f"| {_pct(stats.visible_solve_rate)} | {_pct(stats.false_pass_rate)} "
                f"| {stats.mean_candidates_used:.2f} | {stats.mean_sandbox_seconds:.2f} |"
            )
    return "\n".join(lines)


def render_compute_to_target(results: SweepResults) -> str:
    """Compute needed to reach the target hidden solve rate, per strategy."""
    target = _pct(results.config.target_solve_rate)
    lines = [
        f"| strategy | candidates to reach {target} hidden solve rate |",
        "|---|---:|",
    ]
    levels = results.config.budget_levels
    for strategy in results.config.strategies:
        value = results.compute_to_target.get(strategy)
        if value is None:
            cell = f"not reached at {levels[-1]}"
        else:
            index = levels.index(value)
            lower = levels[index - 1] if index else 0
            cell = f"{value} (crossed between {lower} and {value})"
        lines.append(f"| {strategy} | {cell} |")
    return "\n".join(lines)


def render_ablations(results: AblationResults) -> str:
    """One table per knob, at a single budget level."""
    blocks: list[str] = []
    for knob in results.knobs:
        rows = results.by_knob(knob)
        block = [
            f"**{knob}** (mcts, budget {results.candidates_budget} candidates, "
            f"{results.task_count} tasks)",
            "",
            "| value | hidden solve rate | 95% CI | false-pass rate | mean candidates used |",
            "|---|---:|---|---:|---:|",
        ]
        for row in rows:
            block.append(
                f"| {row.value} | {_pct(row.stats.hidden_solve_rate)} | {_ci(row.stats)} "
                f"| {_pct(row.stats.false_pass_rate)} | {row.stats.mean_candidates_used:.2f} |"
            )
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def render_value_model_card(trained: TrainedValueModel, *, top_features: int = 6) -> str:
    """The value model's own section: what it predicts, how well, and on what."""
    metrics = trained.metrics
    verdict = "meets" if metrics.meets_target else "MISSES"
    lines = [
        f"- version (content hash): `{trained.version}`",
        "- predicts: probability that a candidate passes every hidden test, from visible "
        "feedback and tree position only",
        f"- held-out AUC: **{metrics.auc:.3f}** ({verdict} the 0.75 target)",
        f"- held-out ECE: {metrics.ece:.3f} (10 equal-width bins)",
        f"- held-out Brier: {metrics.brier:.3f}",
        f"- trained on {metrics.train_rows} rows from {metrics.train_tasks} tasks "
        f"(positive rate {_pct(metrics.train_positive_rate)})",
        f"- evaluated on {metrics.holdout_rows} rows from {metrics.holdout_tasks} held-out tasks "
        f"(positive rate {_pct(metrics.holdout_positive_rate)})",
        "- split is by task, not by row: two mutants of one reference share most of their features",
        "",
        "| feature | share of splits |",
        "|---|---:|",
    ]
    for name, share in feature_importance(trained.model)[:top_features]:
        lines.append(f"| {name} | {_pct(share)} |")
    return "\n".join(lines)


def render_escape_suite(outcomes: Sequence[EscapeOutcome]) -> str:
    """The escape suite as a table: what was attempted, what stopped it, what the verdict was."""
    lines = [
        "| attempt | category | stopped by | verdict status | blocked |",
        "|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        lines.append(
            f"| {outcome.attempt.name} | {outcome.attempt.category} "
            f"| {outcome.attempt.mechanism} | {outcome.verdict.status} "
            f"| {'yes' if outcome.blocked else 'NO'} |"
        )
    lines.append("")
    lines.append(
        f"blocked: {_pct(blocked_fraction(outcomes))} of {len(outcomes)} attempts "
        f"({sum(1 for o in outcomes if o.guard_refused)} refused by the in-process guard)"
    )
    return "\n".join(lines)


def render_summary_lines(results: SweepResults) -> list[str]:
    """The provenance block: seeds, hashes and versions that pin this run."""
    value_model = results.value_model_version or _NA
    return [
        f"- tasks: {results.task_count} from {results.family_count} families, "
        f"suite hash `{results.suite_sha256[:16]}`",
        f"- seed: {results.config.seed}; generator: {results.generator} "
        f"({results.generator_model}, noise {results.config.generator_noise:g}, "
        f"seed {results.config.generator_seed})",
        f"- value model: `{value_model if value_model == _NA else value_model[:16]}`, "
        f"c_puct {results.config.c_puct:g}, k {results.config.children_per_expansion}, "
        f"beam width {results.config.beam_width}",
        f"- budget levels (candidates): {', '.join(str(x) for x in results.config.budget_levels)}; "
        f"sandbox timeout {results.config.timeout_seconds:g}s per candidate",
        f"- total: {results.total_evaluations} evaluations, "
        f"{results.total_sandbox_seconds:.1f}s of sandbox time, "
        f"{results.wall_seconds:.1f}s wall clock, "
        f"verdict cache hit ratio {_pct(results.cache_hit_ratio)}",
    ]


def render_experiments_document(
    results: SweepResults,
    *,
    ablations: AblationResults | None = None,
    trained: TrainedValueModel | None = None,
    escape: Sequence[EscapeOutcome] | None = None,
    command: str = "uv run python bench/bench_search.py",
    machine: str = "",
) -> str:
    """The whole of docs/EXPERIMENTS.md. Generated; edit the script, not the file."""
    parts = [
        "# Experiments",
        "",
        "Generated by `" + command + "`. Do not edit by hand: re-run the script.",
        "",
        "## Run provenance",
        "",
        *render_summary_lines(results),
    ]
    if machine:
        parts.append(f"- machine: {machine}")
    parts += [
        "",
        "## Hidden-test solve rate against budget",
        "",
        "Each strategy is run once per task at the largest budget; smaller budgets are the",
        "prefix of that run, which is what a smaller budget would actually have produced",
        "(`tests/integration/test_budget_prefix.py`).",
        "",
        render_solve_rate_table(results),
        "",
        "## Compute to reach the target solve rate",
        "",
        render_compute_to_target(results),
    ]
    if trained is not None:
        parts += ["", "## Value model", "", render_value_model_card(trained)]
    if ablations is not None:
        parts += ["", "## Ablations", "", render_ablations(ablations)]
    # An empty sequence means "not run", the same as None: render_escape_suite refuses to summarise
    # zero attempts rather than report a meaningless 0%.
    if escape:
        parts += ["", "## Sandbox escape suite", "", render_escape_suite(list(escape))]
    return "\n".join(parts) + "\n"
