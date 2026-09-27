"""The calibration bench's configuration and its thresholds, without running the sweep.

Running it costs a minute and a half of sandbox time, which belongs in `uv run python
bench/bench_calibration.py --check`, not in the test suite. What is worth pinning cheaply is that
the script is set up to answer the question the two defaults are justified by: the grids have to
contain the defaults and both extremes, and the thresholds have to encode the claim - a floor on the
spread at the default, ceilings on what happens when either dial leaves the informative region. A
threshold file that asserted nothing would let the provenance rot again in a new way.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verifier_guided_search.search.generator import DEFAULT_NOISE
from verifier_guided_search.search.strategies import STRATEGY_NAMES
from verifier_guided_search.tasks.suite import DEFAULT_CORRUPTION_DEPTH

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def bench(repo_root: Path) -> ModuleType:
    path = repo_root / "bench" / "bench_calibration.py"
    spec = importlib.util.spec_from_file_location("bench_calibration_for_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_both_grids_contain_the_default_and_both_extremes(bench: ModuleType) -> None:
    assert DEFAULT_NOISE in bench.NOISE_GRID
    assert min(bench.NOISE_GRID) == 0.0, "the oracle end has to be in the sweep"
    assert max(bench.NOISE_GRID) > DEFAULT_NOISE, "so does the uniform-sampler end"
    assert DEFAULT_CORRUPTION_DEPTH in bench.DEPTH_GRID
    assert min(bench.DEPTH_GRID) == 1
    assert max(bench.DEPTH_GRID) > DEFAULT_CORRUPTION_DEPTH


def test_the_sweep_compares_every_strategy_with_the_value_model_out_of_the_way(
    bench: ModuleType,
) -> None:
    """The dial is the generator's, so the value model is off and all four strategies are present.

    A spread measured over a subset of the strategies would not be the spread the defaults are
    justified by.
    """
    assert set(bench.STRATEGIES) == set(STRATEGY_NAMES)
    assert bench.LEVELS[-1] == 16


def test_the_thresholds_describe_the_regime_the_defaults_claim(repo_root: Path) -> None:
    spec = json.loads(
        (repo_root / "bench" / "thresholds_calibration.json").read_text(encoding="utf-8")
    )
    # The default has to separate the strategies ...
    assert spec["spread_at_default"]["min"] > 0
    # ... and each way out of the informative region has to be bounded on the other side.
    assert spec["mean_rate_at_zero_noise"]["min"] > 0.5, "an oracle solves almost everything"
    assert spec["mean_rate_at_high_noise"]["max"] < 0.5, "a uniform sampler solves little"
    assert spec["mean_rate_at_depth_1"]["min"] > 0.5, "one edit needs no search"
    assert spec["spread_at_depth_1"]["max"] < spec["spread_at_default"]["min"] * 3
    assert spec["mean_rate_at_depth_3"]["max"] < 0.5, "three edits flatten everything"


def test_the_committed_artifact_matches_the_script_and_the_defaults(
    repo_root: Path, bench: ModuleType
) -> None:
    """The results file has to be the one this script would write, for these defaults."""
    payload = json.loads(
        (repo_root / "bench" / "results" / "calibration.json").read_text(encoding="utf-8")
    )
    assert payload["defaults"] == {
        "noise": DEFAULT_NOISE,
        "corruption_depth": DEFAULT_CORRUPTION_DEPTH,
    }
    assert payload["strategies"] == list(bench.STRATEGIES)
    assert [row["noise"] for row in payload["noise_sweep"]] == list(bench.NOISE_GRID)
    assert [row["corruption_depth"] for row in payload["depth_sweep"]] == list(bench.DEPTH_GRID)
    problems = bench.check_thresholds(payload["metrics"])
    assert problems == [], problems
