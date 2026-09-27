"""The offline demo: search four ways under one budget, then ask to apply the winner.

``uv run python tasks.py demo`` runs this. It needs no API key, no Docker and no network: the
generator is the seeded mutation generator, the verifier is a local subprocess sandbox, and the
budget, audit chain and approval gate are the real ones.

What it does, in order:

1. build a small benchmark (one instance per family) and split it into trace tasks and evaluation
   tasks, so the value model is never used on a task it was trained on;
2. collect traces on the trace tasks with the value model switched off, train the value model, and
   publish it to a content-addressed registry;
3. run greedy, best-of-N, beam and MCTS on the evaluation tasks at matched budgets, printing the
   budget-versus-solve-rate table;
4. run the sandbox escape suite and print how many attempts were blocked;
5. pick the strategy that solved the most tasks, take one task it solved, and propose applying that
   program to a scratch workspace. The apply is refused until a *different* principal approves the
   exact diff hash; the demo then plays the reviewer, approves, and applies.

Step 5 is the part worth watching: the first apply raises ``ApprovalRequiredError`` and the audit
log records the refusal. ``--no-approve`` stops there, which is what a real deployment looks like.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from verifier_guided_search.agentic.approval import ApprovalStore
from verifier_guided_search.agentic.audit import AuditLog, verify_file
from verifier_guided_search.agentic.errors import ApprovalRequiredError
from verifier_guided_search.experiments.ablations import run_ablations
from verifier_guided_search.experiments.report import (
    render_escape_suite,
    render_solve_rate_table,
    render_summary_lines,
)
from verifier_guided_search.experiments.sweep import (
    SweepConfig,
    SweepDeps,
    SweepResults,
    TaskStrategyRun,
    default_generator,
    run_matrix,
    summarise,
)
from verifier_guided_search.experiments.train import (
    TrainedValueModel,
    collect_traces,
    split_suite,
    train_from_traces,
)
from verifier_guided_search.metrics import create_search_metrics
from verifier_guided_search.patch import PatchApplier, ProposedPatch
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.sandbox.escape import ESCAPE_SUITE, blocked_fraction, evaluate_attempt
from verifier_guided_search.sandbox.local import LocalSandbox
from verifier_guided_search.tasks.families import FAMILIES
from verifier_guided_search.tasks.suite import build_suite
from verifier_guided_search.value.registry import ValueModelRegistry

__all__ = ["DemoResult", "main", "run_demo"]

_DEMO_SEED: Final = 20260927
_DEMO_LEVELS: Final = (1, 2, 4, 8)
_WORKSPACE_TARGET: Final = "solution.py"
_REVIEWER: Final = "demo-reviewer"
_SEARCH_PRINCIPAL: Final = "demo-search"


@dataclass(frozen=True, slots=True)
class DemoResult:
    """What the demo produced, as data, so a test can assert on it instead of on stdout."""

    results: SweepResults
    trained: TrainedValueModel | None
    escape_blocked: float
    escape_attempts: int
    patch: ProposedPatch | None
    approval_request_id: str | None
    applied: bool
    audit_entries: int
    audit_head: str
    audit_ok: bool
    workdir: Path

    def summary(self) -> dict[str, Any]:
        best = max(self.results.levels, key=lambda item: (item.hidden_solve_rate, item.strategy))
        return {
            "tasks": self.results.task_count,
            "families": self.results.family_count,
            "suite_sha256": self.results.suite_sha256,
            "strategies": list(self.results.config.strategies),
            "budget_levels": list(self.results.config.budget_levels),
            "best": {
                "strategy": best.strategy,
                "budget": best.candidates_budget,
                "hidden_solve_rate": round(best.hidden_solve_rate, 4),
                "false_pass_rate": round(best.false_pass_rate, 4),
            },
            "value_model_version": self.results.value_model_version,
            "value_model_holdout_auc": (
                round(self.trained.metrics.auc, 4) if self.trained is not None else None
            ),
            "escape_suite": {
                "attempts": self.escape_attempts,
                "blocked_fraction": self.escape_blocked,
            },
            "patch": (
                None
                if self.patch is None
                else {
                    "target": self.patch.target,
                    "diff_sha256": self.patch.diff_sha256,
                    "approval_request_id": self.approval_request_id,
                    "applied": self.applied,
                }
            ),
            "audit": {
                "entries": self.audit_entries,
                "head": self.audit_head,
                "verified": self.audit_ok,
            },
            "evaluations": self.results.total_evaluations,
            "sandbox_seconds": self.results.total_sandbox_seconds,
            "cache_hit_ratio": self.results.cache_hit_ratio,
        }


def _pick_winner(results: SweepResults, runs: Sequence[TaskStrategyRun]) -> TaskStrategyRun | None:
    """A run whose answer passes the hidden tests, from the best strategy at the top level."""
    top = results.config.budget_levels[-1]
    ranked = sorted(
        (results.level(strategy, top) for strategy in results.config.strategies),
        key=lambda stats: (-stats.hidden_solve_rate, stats.mean_candidates_used, stats.strategy),
    )
    for stats in ranked:
        for run in runs:
            if run.strategy != stats.strategy:
                continue
            answer = run.answer_at(top)
            if answer is not None and answer.solved_hidden:
                return run
    return None


def run_demo(
    workdir: Path,
    *,
    instances_per_family: int = 1,
    families: int | None = None,
    approve: bool = True,
    quiet: bool = False,
    with_ablations: bool = False,
    sandbox: CachingSandbox | None = None,
) -> DemoResult:
    """Run the whole demo inside ``workdir``. Deterministic given the seed.

    ``families`` limits how many of the 15 task families take part, and ``sandbox`` lets a caller
    supply a warm verdict cache. Both exist so the test suite can run the demo several times (with
    and without approval, through the CLI) without paying for every sandbox execution again.
    """

    def say(text: str = "") -> None:
        if not quiet:
            print(text, flush=True)  # noqa: T201 - the demo's output is the point

    workdir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(workdir / "audit.jsonl")
    registry = ValueModelRegistry(workdir / "value_models")
    metrics = create_search_metrics()
    selected = FAMILIES if families is None else FAMILIES[:families]
    suite = build_suite(
        instances_per_family=instances_per_family, seed=_DEMO_SEED, families=selected
    )
    trace_suite, eval_suite = split_suite(suite, seed=_DEMO_SEED)
    say(
        f"benchmark: {len(suite)} tasks in {len(suite.families)} families "
        f"(suite {suite.sha256[:16]})"
    )
    say(f"  {len(trace_suite)} tasks for traces, {len(eval_suite)} held out for the comparison")

    config = SweepConfig(
        seed=_DEMO_SEED,
        instances_per_family=instances_per_family,
        budget_levels=_DEMO_LEVELS,
        bootstrap_resamples=200,
    )
    generator = default_generator(config)
    verifier = (
        sandbox
        if sandbox is not None
        else CachingSandbox(LocalSandbox(workdir / "sandbox"), metrics=metrics)
    )

    say()
    say("[1/4] collecting traces with the value model off ...")
    deps = SweepDeps(sandbox=verifier, generator=generator, metrics=metrics, audit=audit)
    store = collect_traces(trace_suite, config, deps)
    trained: TrainedValueModel | None = None
    try:
        trained = train_from_traces(
            store, seed=_DEMO_SEED, notes="offline demo, mutation generator"
        )
    except ValueError as exc:
        say(f"      not enough usable traces to train ({exc}); running without a value model")
    if trained is not None:
        version = trained.save(registry, notes="offline demo")
        say(
            f"      {len(store)} trace rows -> value model {version[:16]} "
            f"holdout AUC {trained.metrics.auc:.3f}, ECE {trained.metrics.ece:.3f}"
        )
        audit.append(
            _SEARCH_PRINCIPAL,
            "value_model.published",
            {"version": version, "auc": round(trained.metrics.auc, 4)},
        )

    say()
    say("[2/4] budget-matched strategy comparison on the held-out tasks ...")
    sweep_deps = SweepDeps(
        sandbox=verifier,
        generator=generator,
        value_model=trained.model if trained is not None else None,
        metrics=metrics,
        audit=audit,
    )
    started = time.perf_counter()
    runs = run_matrix(eval_suite, config, sweep_deps)
    results = summarise(
        eval_suite, config, sweep_deps, runs, wall_seconds=time.perf_counter() - started
    )
    say()
    say(render_solve_rate_table(results))
    say()
    for line in render_summary_lines(results):
        say(line)
    if with_ablations:
        say()
        say("      ablations ...")
        run_ablations(eval_suite, config, sweep_deps, budget=_DEMO_LEVELS[-1])

    say()
    say("[3/4] sandbox escape suite ...")
    outcomes = [evaluate_attempt(verifier.execute, attempt) for attempt in ESCAPE_SUITE]
    blocked = blocked_fraction(outcomes)
    say(render_escape_suite(outcomes))
    audit.append(
        _SEARCH_PRINCIPAL,
        "escape_suite.run",
        {"attempts": len(outcomes), "blocked_fraction": blocked},
    )

    say()
    say("[4/4] applying the winning patch, behind the approval gate ...")
    winner = _pick_winner(results, runs)
    patch: ProposedPatch | None = None
    request_id: str | None = None
    applied = False
    if winner is None:
        say("      no strategy solved a task at this budget; nothing to apply")
    else:
        workspace = workdir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        task = eval_suite.get(winner.task_id)
        (workspace / _WORKSPACE_TARGET).write_text(task.start_program, encoding="utf-8")
        applier = PatchApplier(
            workspace,
            store=ApprovalStore(workdir / "approvals.json"),
            principal=_SEARCH_PRINCIPAL,
            audit=audit,
        )
        answer = winner.answer_at(config.budget_levels[-1])
        # _pick_winner only returns a strategy that answered at the top budget level.
        assert answer is not None  # noqa: S101  # nosec B101
        patch = applier.propose(
            target=_WORKSPACE_TARGET,
            after=answer.program,
            task=task,
            verdict=answer.verdict,
            strategy=winner.strategy,
        )
        say(f"      candidate from {winner.strategy} on {task.task_id}")
        say(f"      {patch.evidence.render()}")
        say(f"      diff hash {patch.diff_sha256}")
        try:
            applier.apply(patch)
            say("      UNEXPECTED: applied without an approval")  # pragma: no cover
        except ApprovalRequiredError as required:
            request_id = required.request_id
            say(f"      refused: approval {request_id[:12]} required, bound to the diff hash")
        if approve and request_id is not None:
            applier.store.decide(
                request_id, approver=_REVIEWER, approve=True, reason="offline demo"
            )
            outcome = applier.apply(patch)
            applied = True
            say(
                f"      approved by {_REVIEWER} -> wrote {outcome.bytes_written} bytes to "
                f"{_WORKSPACE_TARGET}"
            )
        elif request_id is not None:
            say("      --no-approve: stopping with the patch pending, as a real deployment would")

    verification = verify_file(audit.path)
    say()
    say(
        f"audit: {verification.entries} entries, head {verification.head[:16]}, "
        f"chain {'verified' if verification.ok else 'BROKEN'}"
    )
    return DemoResult(
        results=results,
        trained=trained,
        escape_blocked=blocked,
        escape_attempts=len(outcomes),
        patch=patch,
        approval_request_id=request_id,
        applied=applied,
        audit_entries=verification.entries,
        audit_head=verification.head,
        audit_ok=verification.ok,
        workdir=workdir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``tasks.py demo``."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="demo", description="offline verifier-guided search demo")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--instances-per-family", type=int, default=1)
    parser.add_argument(
        "--families",
        type=int,
        default=None,
        help="use only the first N task families (the default is all 15)",
    )
    parser.add_argument(
        "--no-approve",
        action="store_true",
        help="stop with the patch pending instead of approving it",
    )
    parser.add_argument("--json", action="store_true", help="print only the JSON summary")
    args = parser.parse_args(list(argv) if argv is not None else None)

    def run(target: Path) -> int:
        result = run_demo(
            target,
            instances_per_family=args.instances_per_family,
            families=args.families,
            approve=not args.no_approve,
            quiet=bool(args.json),
        )
        print(json.dumps(result.summary(), indent=2, sort_keys=True))  # noqa: T201
        return 0

    if args.workdir is not None:
        return run(args.workdir)
    with tempfile.TemporaryDirectory(prefix="vgs-demo-") as tmp:
        return run(Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
