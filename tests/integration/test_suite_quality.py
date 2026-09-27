"""Is the benchmark actually measuring search? Checked by running every starting program.

A corruption that happens to preserve behaviour on a task's inputs makes that task a free win:
every strategy solves it with its first evaluation. A benchmark full of those would report solve
rates that say nothing about search. So the share is measured, bounded, and reported next to the
results rather than assumed away - see :mod:`verifier_guided_search.tasks.validate`, the README
Results section and docs/EXPERIMENTS.md.

The suite classified here is **the one the reported numbers come from**: the evaluation split of the
seed and instance count that ``bench/bench_search.py`` uses, imported from that file so the two
cannot drift apart. It used to be a different 15-task suite built with a different seed, which meant
the bound was being enforced on tasks nobody reports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verifier_guided_search.experiments.train import split_suite
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.tasks.suite import TaskSuite, build_suite
from verifier_guided_search.tasks.validate import (
    MAX_TRIVIAL_SHARE,
    SuiteReport,
    classify_suite,
)

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]


def _load_bench(repo_root: Path) -> ModuleType:
    """Import ``bench/bench_search.py`` for its constants, so this test cannot drift from it."""
    path = repo_root / "bench" / "bench_search.py"
    spec = importlib.util.spec_from_file_location("bench_search_for_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench(repo_root: Path) -> ModuleType:
    return _load_bench(repo_root)


@pytest.fixture(scope="module")
def reported_suite(bench: ModuleType) -> TaskSuite:
    """The split the README's solve rates are measured on."""
    suite = build_suite(instances_per_family=bench.FULL_INSTANCES, seed=bench.SEED)
    _traces, evaluation = split_suite(suite, seed=bench.SEED)
    return evaluation


@pytest.fixture(scope="module")
def report(shared_sandbox: CachingSandbox, reported_suite: TaskSuite) -> SuiteReport:
    return classify_suite(reported_suite, shared_sandbox.execute, timeout_seconds=1.5)


def test_few_reported_tasks_start_already_correct(report: SuiteReport) -> None:
    """The bound that keeps the benchmark honest, on the tasks the rates are reported for."""
    assert report.trivial_share <= MAX_TRIVIAL_SHARE, report.render()


def test_the_bench_threshold_enforces_the_same_bound(repo_root: Path) -> None:
    """The bench checks the same number, so a CI run fails the same way this test does."""
    spec = json.loads((repo_root / "bench" / "thresholds_search.json").read_text(encoding="utf-8"))
    assert spec["eval_trivial_share"]["max"] == MAX_TRIVIAL_SHARE
    assert "eval_root_false_pass_share" in spec


def test_most_tasks_start_broken(report: SuiteReport) -> None:
    assert report.broken_share >= 0.6, report.render()


def test_the_classification_covers_every_task(
    report: SuiteReport, reported_suite: TaskSuite
) -> None:
    starts = report.starts
    assert len(starts) == len(reported_suite)
    assert {item.task_id for item in starts} == {task.task_id for task in reported_suite}
    assert sum(
        (report.trivial_share, report.root_false_pass_share, report.broken_share)
    ) == pytest.approx(1.0)


def test_a_trivial_task_really_passes_everything(report: SuiteReport) -> None:
    for item in report.of_kind("trivial"):
        assert item.status == "ok"
        assert item.visible_passed == item.visible_total
        assert item.hidden_passed == item.hidden_total


def test_a_root_false_pass_satisfies_the_verifier_while_being_wrong(report: SuiteReport) -> None:
    """These are not defects: they are the verifier-overfitting case, for free.

    A task in this class contributes a false pass to *every* strategy at *every* budget with no
    search at all, which is most of why the reported false-pass rate is flat in the budget.
    """
    for item in report.of_kind("root_false_pass"):
        assert item.visible_passed == item.visible_total
        assert item.hidden_passed < item.hidden_total


def test_the_report_serialises_for_a_results_file(report: SuiteReport) -> None:
    payload = report.as_dict()
    assert set(payload) == {
        "tasks",
        "trivial",
        "trivial_share",
        "root_false_pass",
        "root_false_pass_share",
        "broken",
        "trivial_task_ids",
        "root_false_pass_task_ids",
    }
    assert isinstance(payload["trivial_task_ids"], list)
    assert isinstance(payload["root_false_pass_task_ids"], list)
    assert "free win" in report.render()


def test_an_empty_report_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        SuiteReport(starts=())
