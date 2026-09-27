"""Is the benchmark actually measuring search? Checked by running every starting program.

A corruption that happens to preserve behaviour on a task's inputs makes that task a free win: every
strategy solves it with its first evaluation. A benchmark full of those would report high solve
rates that say nothing about search. So the share is measured, bounded, and reported next to the
results rather than being assumed away - see
:mod:`verifier_guided_search.tasks.validate` and the README Results section.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.tasks.suite import TaskSuite
from verifier_guided_search.tasks.validate import MAX_TRIVIAL_SHARE, classify_suite

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]


@pytest.fixture(scope="module")
def report(shared_sandbox: CachingSandbox, small_suite: TaskSuite) -> object:
    return classify_suite(small_suite, shared_sandbox.execute, timeout_seconds=1.0)


def test_few_tasks_start_already_correct(report: object) -> None:
    """The bound that keeps the benchmark honest: most tasks must really be broken."""
    assert report.trivial_share <= MAX_TRIVIAL_SHARE, report.render()  # type: ignore[attr-defined]


def test_most_tasks_start_broken(report: object) -> None:
    assert report.broken_share >= 0.6, report.render()  # type: ignore[attr-defined]


def test_the_classification_covers_every_task(report: object, small_suite: TaskSuite) -> None:
    starts = report.starts  # type: ignore[attr-defined]
    assert len(starts) == len(small_suite)
    assert {item.task_id for item in starts} == {task.task_id for task in small_suite}
    assert sum(
        (report.trivial_share, report.root_false_pass_share, report.broken_share)  # type: ignore[attr-defined]
    ) == pytest.approx(1.0)


def test_a_trivial_task_really_passes_everything(report: object) -> None:
    for item in report.of_kind("trivial"):  # type: ignore[attr-defined]
        assert item.status == "ok"
        assert item.visible_passed == item.visible_total
        assert item.hidden_passed == item.hidden_total


def test_a_root_false_pass_satisfies_the_verifier_while_being_wrong(report: object) -> None:
    """These are not defects: they are the verifier-overfitting case, for free."""
    for item in report.of_kind("root_false_pass"):  # type: ignore[attr-defined]
        assert item.visible_passed == item.visible_total
        assert item.hidden_passed < item.hidden_total


def test_the_report_serialises_for_a_results_file(report: object) -> None:
    payload = report.as_dict()  # type: ignore[attr-defined]
    assert set(payload) == {
        "tasks",
        "trivial",
        "trivial_share",
        "root_false_pass",
        "root_false_pass_share",
        "broken",
        "trivial_task_ids",
    }
    assert isinstance(payload["trivial_task_ids"], list)
    assert "free win" in report.render()  # type: ignore[attr-defined]


def test_an_empty_report_is_refused() -> None:
    from verifier_guided_search.tasks.validate import SuiteReport

    with pytest.raises(ValueError, match="at least one task"):
        SuiteReport(starts=())
