#!/usr/bin/env python3
"""The repo's evaluation: budget curves, compute-to-target, false-pass rate, value model, escapes.

Usage:
  uv run python bench/bench_search.py            # run, write bench/results/search.json +
  EXPERIMENTS.md uv run python bench/bench_search.py --check    # also exit 1 if
  bench/thresholds_search.json fails
  uv run python bench/bench_search.py --quick    # the CI-sized configuration (fewer tasks)

Everything runs offline on one CPU: the generator is the seeded mutation generator and the verifier
is a local subprocess sandbox. Nothing here touches an API.

What it measures, in the order the spec lists it:

1. hidden-test solve rate against budget for greedy, best-of-N, beam and MCTS, with percentile
   bootstrap 95% CIs over tasks;
2. candidates needed to reach the target hidden solve rate, per strategy;
3. verifier false-pass rate (visible pass, hidden fail) per strategy;
4. value model held-out AUC and ECE, trained on a *disjoint* set of tasks;
5. ablations over ``c_puct``, ``k`` and value model on/off - plus the value-model on/off
   comparison repeated for MCTS on the *whole reported split at the reported budget*, because the
   ablations run on a 12-task subset at budget 8 and the README's claim that the model does not help
   is about the headline configuration;
6. the sandbox escape suite;
7. what the reported tasks *start* from - how many are already correct (a free win for every
   strategy) and how many already satisfy their visible suite with a wrong program (a false pass at
   depth 0, for every strategy, at every budget). Both shares are measured on the reported split
   rather than on some other suite, written into the results JSON, and bounded by thresholds.

Thresholds (``bench/thresholds_search.json``) are floors and ceilings, not exact values: a floor on
the best strategy's solve rate, a floor on AUC, a ceiling on the false-pass rate, an exact 1.0 on
the escape suite, and a generous wall-clock ceiling. They are set well below/above what this machine
measures, because a shared CI runner is slower and the bootstrap is stochastic. They catch "the
search stopped working", not small drifts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from verifier_guided_search.experiments.ablations import (
    AblationResults,
    run_ablations,
)
from verifier_guided_search.experiments.report import (
    render_experiments_document,
)
from verifier_guided_search.experiments.sweep import (
    SweepConfig,
    SweepDeps,
    SweepResults,
    default_generator,
    run_sweep,
)
from verifier_guided_search.experiments.train import (
    TrainedValueModel,
    collect_traces,
    split_suite,
    train_from_traces,
)
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.sandbox.escape import (
    ESCAPE_SUITE,
    EscapeOutcome,
    blocked_fraction,
    evaluate_attempt,
)
from verifier_guided_search.sandbox.local import LocalSandbox
from verifier_guided_search.tasks.suite import TaskSuite, build_suite
from verifier_guided_search.tasks.validate import SuiteReport, classify_suite

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "search.json"
THRESHOLDS = ROOT / "bench" / "thresholds_search.json"
EXPERIMENTS = ROOT / "docs" / "EXPERIMENTS.md"

SEED: Final = 20260927
#: Tasks used to collect value-model training traces. Disjoint from the reported tasks.
TRACE_STRATEGIES: Final = ("greedy", "mcts")
FULL_INSTANCES: Final = 2
QUICK_INSTANCES: Final = 1
FULL_LEVELS: Final = (1, 2, 4, 8, 16)
QUICK_LEVELS: Final = (1, 2, 4, 8)
#: Ablations run at their own top level, on a subset, so nine variants stay inside a few minutes.
ABLATION_LEVELS: Final = (1, 2, 4, 8)
ABLATION_TASKS: Final = 12


@dataclass(frozen=True, slots=True)
class BenchOutput:
    results: SweepResults
    #: MCTS over the same tasks and budgets with the value model off. The ablation table answers the
    #: same question on a subset at a lower budget; this answers it where the headline lives.
    value_model_off: SweepResults
    ablations: AblationResults
    trained: TrainedValueModel
    escapes: tuple[EscapeOutcome, ...]
    starts: SuiteReport
    wall_seconds: float

    def metrics(self) -> dict[str, float]:
        """The flat numbers thresholds are checked against."""
        top = self.results.config.budget_levels[-1]
        levels = [self.results.level(s, top) for s in self.results.config.strategies]
        best = max(levels, key=lambda item: item.hidden_solve_rate)
        return {
            "best_strategy_hidden_solve_rate": round(best.hidden_solve_rate, 4),
            "mcts_hidden_solve_rate": round(self.results.level("mcts", top).hidden_solve_rate, 4),
            "mcts_hidden_solve_rate_value_model_off": round(
                self.value_model_off.level("mcts", top).hidden_solve_rate, 4
            ),
            "max_false_pass_rate": round(max(item.false_pass_rate for item in levels), 4),
            "value_model_holdout_auc": round(self.trained.metrics.auc, 4),
            "value_model_holdout_ece": round(self.trained.metrics.ece, 4),
            "escape_blocked_fraction": round(blocked_fraction(list(self.escapes)), 4),
            # Measured on the reported split, so the README's caveats are about these tasks.
            "eval_trivial_share": round(self.starts.trivial_share, 4),
            "eval_root_false_pass_share": round(self.starts.root_false_pass_share, 4),
            "wall_seconds": round(self.wall_seconds, 1),
            "sandbox_seconds": round(self.results.total_sandbox_seconds, 1),
            "evaluations": float(self.results.total_evaluations),
            "cache_hit_ratio": round(self.results.cache_hit_ratio, 4),
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "machine": machine(),
            "metrics": self.metrics(),
            "sweep": self.results.model_dump(mode="json"),
            "sweep_value_model_off": self.value_model_off.model_dump(mode="json"),
            "ablations": self.ablations.model_dump(mode="json"),
            "value_model": {
                "version": self.trained.version,
                **self.trained.metrics.model_dump(mode="json"),
            },
            "eval_split_starts": self.starts.as_dict(),
            "escape_suite": [item.as_row() for item in self.escapes],
        }


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def machine_line() -> str:
    info = machine()
    return f"{info['platform']}, {info['cpu_count']} CPUs, Python {info['python']}, CPU-only"


def run(*, quick: bool, workdir: Path, verbose: bool = True) -> BenchOutput:
    """Run the whole evaluation in ``workdir``."""

    def say(text: str) -> None:
        if verbose:
            print(text, flush=True)

    instances = QUICK_INSTANCES if quick else FULL_INSTANCES
    levels = QUICK_LEVELS if quick else FULL_LEVELS
    started = time.perf_counter()
    suite = build_suite(instances_per_family=instances, seed=SEED)
    trace_suite, eval_suite = split_suite(suite, seed=SEED)
    say(
        f"suite {suite.sha256[:16]}: {len(suite)} tasks -> "
        f"{len(trace_suite)} for traces, {len(eval_suite)} reported"
    )

    sandbox = CachingSandbox(LocalSandbox(workdir / "sandbox"))
    trace_config = SweepConfig(
        seed=SEED,
        instances_per_family=instances,
        strategies=TRACE_STRATEGIES,
        budget_levels=levels,
        bootstrap_resamples=0,
    )
    generator = default_generator(trace_config)
    say("collecting traces (value model off) ...")
    store = collect_traces(
        trace_suite, trace_config, SweepDeps(sandbox=sandbox, generator=generator)
    )
    trained = train_from_traces(
        store, seed=SEED, notes=f"mutation generator, {len(trace_suite)} trace tasks"
    )
    say(
        f"  {len(store)} rows -> value model {trained.version[:16]}, "
        f"holdout AUC {trained.metrics.auc:.3f}, ECE {trained.metrics.ece:.3f}"
    )

    config = SweepConfig(seed=SEED, instances_per_family=instances, budget_levels=levels)
    deps = SweepDeps(
        sandbox=sandbox,
        generator=generator,
        value_model=trained.model,
        progress=lambda text: say(f"  {text}"),
    )
    say("budget sweep ...")
    results = run_sweep(eval_suite, config, deps)

    say("the same sweep for mcts with the value model off ...")
    off_config = config.model_copy(update={"strategies": ("mcts",), "use_value_model": False})
    value_model_off = run_sweep(
        eval_suite,
        off_config,
        SweepDeps(sandbox=sandbox, generator=generator, progress=lambda text: say(f"  {text}")),
    )
    say(
        f"  mcts at {levels[-1]}: "
        f"{value_model_off.level('mcts', levels[-1]).hidden_solve_rate:.3f} without the model "
        f"vs {results.level('mcts', levels[-1]).hidden_solve_rate:.3f} with it"
    )

    say("ablations ...")
    ablation_suite = eval_suite.subset(per_family=1)
    if len(ablation_suite) > ABLATION_TASKS:
        ablation_suite = TaskSuite(ablation_suite.tasks[:ABLATION_TASKS])
    ablation_config = config.model_copy(update={"budget_levels": ABLATION_LEVELS})
    ablations = run_ablations(ablation_suite, ablation_config, deps, budget=ABLATION_LEVELS[-1])

    say("classifying what the reported tasks start from ...")
    starts = classify_suite(eval_suite, sandbox.execute, timeout_seconds=config.timeout_seconds)
    say(f"  {starts.render()}")

    say("escape suite ...")
    escapes = tuple(evaluate_attempt(sandbox.execute, attempt) for attempt in ESCAPE_SUITE)

    return BenchOutput(
        results=results,
        value_model_off=value_model_off,
        ablations=ablations,
        trained=trained,
        escapes=escapes,
        starts=starts,
        wall_seconds=time.perf_counter() - started,
    )


def check_thresholds(metrics: dict[str, float]) -> list[str]:
    """Compare the measured numbers with the floors and ceilings. Returns the failures."""
    spec: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, bounds in spec.items():
        if name not in metrics:
            problems.append(f"{name}: metric missing from this run")
            continue
        value = metrics[name]
        low, high = bounds.get("min"), bounds.get("max")
        if low is not None and value < low:
            problems.append(f"{name} = {value} is below the floor {low}")
        if high is not None and value > high:
            problems.append(f"{name} = {value} is above the ceiling {high}")
    return problems


def document(output: BenchOutput, command: str) -> str:
    return render_experiments_document(
        output.results,
        value_model_off=output.value_model_off,
        ablations=output.ablations,
        trained=output.trained,
        escape=list(output.escapes),
        starts=output.starts,
        command=command,
        machine=machine_line(),
    )


def print_tables(output: BenchOutput, command: str) -> None:
    print()
    print(document(output, command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verifier-guided search evaluation")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds_search.json")
    parser.add_argument("--quick", action="store_true", help="the smaller CI configuration")
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="also write docs/EXPERIMENTS.md from this run",
    )
    args = parser.parse_args(argv)

    command = "uv run python bench/bench_search.py" + (" --quick" if args.quick else "")
    with tempfile.TemporaryDirectory(prefix="vgs-bench-") as tmp:
        output = run(quick=args.quick, workdir=Path(tmp))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(output.as_json(), indent=2) + "\n", encoding="utf-8")
    print_tables(output, command)
    if args.write_docs:
        EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS.write_text(document(output, command), encoding="utf-8")
        print(f"wrote {EXPERIMENTS.relative_to(ROOT)}")

    print()
    print("| metric | value |")
    print("|---|---:|")
    for name, value in output.metrics().items():
        print(f"| {name} | {value} |")

    if args.check:
        problems = check_thresholds(output.metrics())
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all evaluation metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
