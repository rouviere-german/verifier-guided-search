#!/usr/bin/env python3
"""Calibration of the benchmark's two dials, measured instead of asserted.

Usage:
  uv run python bench/bench_calibration.py          # run, write bench/results/calibration.json
  uv run python bench/bench_calibration.py --check   # also enforce the calibration thresholds

Two numbers in this repo are choices, and both were being justified by prose:

``DEFAULT_NOISE`` (``verifier_guided_search.search.generator``)
    How noisy the offline generator's revert signal is. At zero noise it is an oracle and every
    strategy solves every task in ``d`` steps, so the comparison measures nothing; at large noise it
    is a uniform sampler and search cannot help, so the comparison measures nothing again. The
    default has to sit where the strategies *separate*.
``DEFAULT_CORRUPTION_DEPTH`` (``verifier_guided_search.tasks.suite``)
    How many edits away from the reference each task starts. Depth 1 is solvable by a single
    proposal, so every strategy including best-of-N gets it; depth 3 flattens everything at the
    bottom of the range.

This script sweeps both and reports the *spread* between strategies - the maximum hidden solve rate
minus the minimum, at a fixed budget. A dial setting that collapses the spread makes the whole
evaluation uninformative, whichever direction it collapses in. It runs a deliberately small
configuration - six families, one instance each, all four strategies, budget 16, and the value model
off so the generator dial is the only thing moving - because it is answering "is this dial in the
informative region", not "what is the solve rate": that is ``bench/bench_search.py``.

Everything is offline: the seeded mutation generator and the local subprocess sandbox.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from verifier_guided_search.experiments.sweep import (
    SweepConfig,
    SweepDeps,
    default_generator,
    run_sweep,
)
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.sandbox.local import LocalSandbox
from verifier_guided_search.search.generator import DEFAULT_NOISE
from verifier_guided_search.tasks.families import FAMILIES
from verifier_guided_search.tasks.suite import DEFAULT_CORRUPTION_DEPTH, build_suite

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "calibration.json"
THRESHOLDS = ROOT / "bench" / "thresholds_calibration.json"

SEED: Final = 20260927
#: Six families, one instance each: enough for a spread to exist, small enough to sweep six cells.
FAMILIES_USED: Final = 6
STRATEGIES: Final = ("greedy", "best_of_n", "beam", "mcts")
LEVELS: Final = (1, 2, 4, 8, 16)
NOISE_GRID: Final = (0.0, 0.25, DEFAULT_NOISE, 1.5)
DEPTH_GRID: Final = (1, DEFAULT_CORRUPTION_DEPTH, 3)


@dataclass(frozen=True, slots=True)
class Cell:
    """One (noise, depth) setting, and what the strategies did with it."""

    noise: float
    depth: int
    rates: dict[str, float]

    @property
    def spread(self) -> float:
        """Best strategy minus worst. Zero means the dial has made the comparison meaningless."""
        return max(self.rates.values()) - min(self.rates.values())

    @property
    def mean_rate(self) -> float:
        return sum(self.rates.values()) / len(self.rates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "noise": self.noise,
            "corruption_depth": self.depth,
            "hidden_solve_rate": {name: round(rate, 4) for name, rate in self.rates.items()},
            "spread": round(self.spread, 4),
            "mean_hidden_solve_rate": round(self.mean_rate, 4),
        }


def machine() -> dict[str, object]:
    return {"platform": platform.platform(), "python": sys.version.split()[0]}


def run_cell(*, noise: float, depth: int, sandbox: CachingSandbox) -> Cell:
    """Run every strategy on one dial setting and read the top budget level."""
    suite = build_suite(
        instances_per_family=1,
        seed=SEED,
        corruption_depth=depth,
        families=FAMILIES[:FAMILIES_USED],
    )
    config = SweepConfig(
        seed=SEED,
        instances_per_family=1,
        strategies=STRATEGIES,
        budget_levels=LEVELS,
        generator_noise=noise,
        bootstrap_resamples=0,
        use_value_model=False,
    )
    results = run_sweep(
        suite, config, SweepDeps(sandbox=sandbox, generator=default_generator(config))
    )
    top = LEVELS[-1]
    return Cell(
        noise=noise,
        depth=depth,
        rates={name: results.level(name, top).hidden_solve_rate for name in STRATEGIES},
    )


def run(*, workdir: Path, verbose: bool = True) -> dict[str, Any]:
    """Sweep noise at the default depth, then depth at the default noise."""

    def say(text: str) -> None:
        if verbose:
            print(text, flush=True)

    started = time.perf_counter()
    sandbox = CachingSandbox(LocalSandbox(workdir / "sandbox"))
    cells: dict[tuple[float, int], Cell] = {}

    def cell(noise: float, depth: int) -> Cell:
        key = (noise, depth)
        if key not in cells:
            cells[key] = run_cell(noise=noise, depth=depth, sandbox=sandbox)
            say(
                f"  noise {noise:g}, depth {depth}: spread {cells[key].spread:.3f}, "
                f"mean {cells[key].mean_rate:.3f}"
            )
        return cells[key]

    say(f"noise sweep at depth {DEFAULT_CORRUPTION_DEPTH} ...")
    noise_cells = [cell(noise, DEFAULT_CORRUPTION_DEPTH) for noise in NOISE_GRID]
    say(f"corruption-depth sweep at noise {DEFAULT_NOISE:g} ...")
    depth_cells = [cell(DEFAULT_NOISE, depth) for depth in DEPTH_GRID]

    chosen = cell(DEFAULT_NOISE, DEFAULT_CORRUPTION_DEPTH)
    return {
        "machine": machine(),
        "seed": SEED,
        "families": FAMILIES_USED,
        "strategies": list(STRATEGIES),
        "budget_levels": list(LEVELS),
        "defaults": {"noise": DEFAULT_NOISE, "corruption_depth": DEFAULT_CORRUPTION_DEPTH},
        "noise_sweep": [item.as_dict() for item in noise_cells],
        "depth_sweep": [item.as_dict() for item in depth_cells],
        "metrics": {
            "spread_at_default": round(chosen.spread, 4),
            "spread_at_zero_noise": round(cell(0.0, DEFAULT_CORRUPTION_DEPTH).spread, 4),
            "mean_rate_at_zero_noise": round(cell(0.0, DEFAULT_CORRUPTION_DEPTH).mean_rate, 4),
            "mean_rate_at_high_noise": round(
                cell(NOISE_GRID[-1], DEFAULT_CORRUPTION_DEPTH).mean_rate, 4
            ),
            "mean_rate_at_depth_1": round(cell(DEFAULT_NOISE, 1).mean_rate, 4),
            "mean_rate_at_depth_3": round(cell(DEFAULT_NOISE, 3).mean_rate, 4),
            "spread_at_depth_1": round(cell(DEFAULT_NOISE, 1).spread, 4),
            "spread_at_depth_3": round(cell(DEFAULT_NOISE, 3).spread, 4),
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
    }


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


def render(payload: dict[str, Any]) -> str:
    """The two tables, for pasting into an ADR."""
    width = "---:|" * (len(STRATEGIES) + 1)
    lines = ["| noise | " + " | ".join(STRATEGIES) + " | spread |", "|---:|" + width]
    for row in payload["noise_sweep"]:
        rates = row["hidden_solve_rate"]
        cells = " | ".join(f"{100 * rates[name]:.1f}%" for name in STRATEGIES)
        lines.append(f"| {row['noise']:g} | {cells} | {row['spread']:.3f} |")
    lines += [
        "",
        "| corruption depth | " + " | ".join(STRATEGIES) + " | spread |",
        "|---:|" + width,
    ]
    for row in payload["depth_sweep"]:
        rates = row["hidden_solve_rate"]
        cells = " | ".join(f"{100 * rates[name]:.1f}%" for name in STRATEGIES)
        lines.append(f"| {row['corruption_depth']} | {cells} | {row['spread']:.3f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="benchmark dial calibration")
    parser.add_argument(
        "--check", action="store_true", help="enforce bench/thresholds_calibration.json"
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="vgs-calibration-") as tmp:
        payload = run(workdir=Path(tmp))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(render(payload))
    print()
    print("| metric | value |")
    print("|---|---:|")
    for name, value in payload["metrics"].items():
        print(f"| {name} | {value} |")

    if args.check:
        problems = check_thresholds(payload["metrics"])
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all calibration metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
