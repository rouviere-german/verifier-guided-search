"""Classifying what each task's starting program actually does, and reporting the awkward cases.

:mod:`verifier_guided_search.tasks.suite` builds tasks without executing any candidate: a corrupted
program can loop forever, and only the sandbox has a watchdog. The price of that safety is that the
suite cannot know, at build time, whether a corruption really broke anything. Two cases come out of
that, and both are reported rather than hidden:

``trivial``
    The starting program passes every visible *and* every hidden test. The mutation was
    behaviour-preserving on all of this task's inputs, so every strategy solves the task with its
    first evaluation. That is a constant added to every strategy's solve rate - it does not change
    which strategy wins, but it does inflate the absolute numbers, so the share is reported next to
    them (see the README Results section and docs/EXPERIMENTS.md).
``root_false_pass``
    The starting program passes every visible test but fails a hidden one. This is not a defect: it
    is precisely the verifier-overfitting case the benchmark exists to measure, arriving for free. A
    strategy stops as soon as the visible tests are satisfied, so it commits to a wrong program and
    the false-pass rate counts it.
``broken``
    The starting program fails a visible test. The intended case.

:func:`classify_suite` needs a verifier, so it is called from the bench, the demo and a test - never
from :func:`~verifier_guided_search.tasks.suite.build_suite`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from verifier_guided_search.sandbox.protocol import ExecJob, ExecLimits, Verdict
from verifier_guided_search.tasks.models import Task
from verifier_guided_search.tasks.suite import TaskSuite

__all__ = [
    "MAX_TRIVIAL_SHARE",
    "StartKind",
    "SuiteReport",
    "TaskStart",
    "classify_suite",
    "classify_task",
]

StartKind = Literal["trivial", "root_false_pass", "broken"]
ExecuteFn = Callable[[ExecJob], Verdict]
#: A benchmark where more than this share of tasks starts already correct would be measuring
#: nothing.
#: Enforced by ``tests/integration/test_suite_quality.py::test_few_tasks_start_already_correct``.
MAX_TRIVIAL_SHARE: Final = 0.15


@dataclass(frozen=True, slots=True)
class TaskStart:
    """What one task's starting program does."""

    task_id: str
    family: str
    kind: StartKind
    status: str
    visible_passed: int
    visible_total: int
    hidden_passed: int
    hidden_total: int

    @property
    def trivial(self) -> bool:
        return self.kind == "trivial"


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """The classification of a whole suite, with the shares a results file should carry."""

    starts: tuple[TaskStart, ...]

    def __post_init__(self) -> None:
        if not self.starts:
            raise ValueError("a suite report needs at least one task")

    def of_kind(self, kind: StartKind) -> tuple[TaskStart, ...]:
        return tuple(item for item in self.starts if item.kind == kind)

    @property
    def trivial_share(self) -> float:
        """Share of tasks every strategy solves with its first evaluation."""
        return len(self.of_kind("trivial")) / len(self.starts)

    @property
    def root_false_pass_share(self) -> float:
        """Share where the verifier is already satisfied by a wrong starting program."""
        return len(self.of_kind("root_false_pass")) / len(self.starts)

    @property
    def broken_share(self) -> float:
        return len(self.of_kind("broken")) / len(self.starts)

    def as_dict(self) -> dict[str, object]:
        return {
            "tasks": len(self.starts),
            "trivial": len(self.of_kind("trivial")),
            "trivial_share": round(self.trivial_share, 4),
            "root_false_pass": len(self.of_kind("root_false_pass")),
            "root_false_pass_share": round(self.root_false_pass_share, 4),
            "broken": len(self.of_kind("broken")),
            "trivial_task_ids": [item.task_id for item in self.of_kind("trivial")],
        }

    def render(self) -> str:
        return (
            f"{len(self.starts)} tasks: {len(self.of_kind('broken'))} start broken, "
            f"{len(self.of_kind('root_false_pass'))} already satisfy the visible tests while "
            f"failing a hidden one, {len(self.of_kind('trivial'))} start fully correct "
            f"({self.trivial_share:.1%} - a free win for every strategy)"
        )


def classify_task(task: Task, execute: ExecuteFn, *, timeout_seconds: float = 1.5) -> TaskStart:
    """Run one task's starting program and say which of the three cases it is."""
    verdict = execute(
        ExecJob(
            program=task.start_program,
            entry_point=task.entry_point,
            visible=task.visible.cases,
            hidden=task.hidden.cases,
            limits=ExecLimits(timeout_seconds=timeout_seconds),
        )
    )
    ran = verdict.status == "ok"
    kind: StartKind = "broken"
    if ran and verdict.visible.all_passed:
        kind = "trivial" if verdict.hidden.all_passed else "root_false_pass"
    return TaskStart(
        task_id=task.task_id,
        family=task.family,
        kind=kind,
        status=verdict.status,
        visible_passed=verdict.visible.passed,
        visible_total=verdict.visible.total,
        hidden_passed=verdict.hidden.passed,
        hidden_total=verdict.hidden.total,
    )


def classify_suite(
    suite: TaskSuite | Sequence[Task], execute: ExecuteFn, *, timeout_seconds: float = 1.5
) -> SuiteReport:
    """Classify every task's starting program. Costs one sandbox execution per task."""
    tasks = list(suite)
    return SuiteReport(
        starts=tuple(
            classify_task(task, execute, timeout_seconds=timeout_seconds) for task in tasks
        )
    )
