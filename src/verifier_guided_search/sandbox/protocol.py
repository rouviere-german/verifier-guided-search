"""The wire protocol between the search process and the runner, and the verdict types.

Two type distinctions carry weight here.

**Visible vs hidden.** :class:`Verdict` holds both groups; :class:`VisibleFeedback` holds only the
visible one. Everything downstream of the search - the generator, the value model features, the node
scores - takes ``VisibleFeedback``, so "search must not look at hidden tests" is enforced by the
signatures rather than by review discipline.

**Structured vs free text.** What a candidate program writes to stdout or puts in an exception
message is attacker-controlled once a real model writes the candidates. None of it reaches
``VisibleFeedback``: a test result carries a name, a status and an exception *class name* matching
``^[A-Za-z_][A-Za-z0-9_]{0,63}$``, nothing else. Free text from a run is kept on ``Verdict.detail``
for operators, redacted and truncated, and never sent to a model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.core.canonical import canonical_json, sha256_hex
from verifier_guided_search.tasks.models import TestCase

__all__ = [
    "ERROR_CLASS_PATTERN",
    "ExecJob",
    "ExecLimits",
    "GroupResult",
    "TestResult",
    "TestStatus",
    "Verdict",
    "VerdictStatus",
    "VisibleFeedback",
]

ERROR_CLASS_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
TestStatus = Literal["pass", "fail", "error"]
VerdictStatus = Literal[
    "ok",
    "syntax_error",
    "missing_entry_point",
    "load_error",
    "blocked",
    "timeout",
    "crashed",
]
#: Verdicts where no test ever ran, so a pass fraction of 0 is a fact rather than a measurement.
NO_RUN_STATUSES: frozenset[str] = frozenset(
    {"syntax_error", "missing_entry_point", "load_error", "blocked", "timeout", "crashed"}
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExecLimits(_Frozen):
    """Per-execution resource caps. Enforced in the child, and again by the container."""

    timeout_seconds: float = Field(default=1.5, gt=0, le=60)
    memory_bytes: int = Field(default=256 * 1024 * 1024, ge=16 * 1024 * 1024)
    max_processes: int = Field(default=1, ge=1, le=16)
    max_file_bytes: int = Field(default=1024 * 1024, ge=1024)
    max_output_bytes: int = Field(default=4096, ge=256, le=1024 * 1024)

    @property
    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class ExecJob(_Frozen):
    """One program plus the tests to run against it. Pure JSON; no Python objects cross over."""

    program: str = Field(min_length=1, max_length=64_000)
    entry_point: str = Field(pattern=r"^[a-z_][a-z0-9_]{0,63}$")
    visible: tuple[TestCase, ...] = Field(min_length=1)
    hidden: tuple[TestCase, ...] = ()
    limits: ExecLimits = ExecLimits()

    @property
    def program_sha256(self) -> str:
        return sha256_hex(self.program)

    @property
    def suite_sha256(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "entry_point": self.entry_point,
                    "visible": [case.model_dump(mode="json") for case in self.visible],
                    "hidden": [case.model_dump(mode="json") for case in self.hidden],
                }
            )
        )

    @property
    def cache_key(self) -> str:
        """Content address of the whole execution: program, tests and limits."""
        return sha256_hex(f"{self.program_sha256}:{self.suite_sha256}:{self.limits.sha256}")


class TestResult(_Frozen):
    # Not a pytest class: this is one benchmark test's outcome.
    __test__ = False

    name: str = Field(min_length=1, max_length=64)
    status: TestStatus
    error_class: str | None = Field(default=None, pattern=ERROR_CLASS_PATTERN)


class GroupResult(_Frozen):
    results: tuple[TestResult, ...] = ()

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.results if item.status == "pass")

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def pass_fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def failing_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.results if item.status != "pass")

    def error_class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.results:
            if item.error_class is not None:
                counts[item.error_class] = counts.get(item.error_class, 0) + 1
        return counts


class VisibleFeedback(_Frozen):
    """Everything the search is allowed to know about a program.

    Deliberately holds no hidden-test information and no free text from the run.
    """

    program_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: VerdictStatus
    visible: GroupResult
    violations: tuple[str, ...] = ()

    @property
    def solved(self) -> bool:
        return self.status == "ok" and self.visible.all_passed

    @property
    def pass_fraction(self) -> float:
        return self.visible.pass_fraction

    @property
    def digest(self) -> str:
        """Stable hash of the feedback; seeds the generator so expansion is reproducible."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class Verdict(_Frozen):
    """The full result, including hidden outcomes. Only the scorer may read ``hidden``."""

    program_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: VerdictStatus
    visible: GroupResult = GroupResult()
    hidden: GroupResult = GroupResult()
    violations: tuple[str, ...] = ()
    duration_seconds: float = Field(default=0.0, ge=0)
    exit_code: int = 0
    detail: str = Field(default="", max_length=2048)
    stdout_bytes: int = Field(default=0, ge=0)
    stderr_bytes: int = Field(default=0, ge=0)

    def visible_only(self) -> VisibleFeedback:
        """Project away everything the search must not see."""
        return VisibleFeedback(
            program_sha256=self.program_sha256,
            status=self.status,
            visible=self.visible,
            violations=self.violations,
        )

    @property
    def hidden_solved(self) -> bool:
        return self.status == "ok" and self.hidden.all_passed

    @property
    def false_pass(self) -> bool:
        """Passed every visible test but failed a hidden one: the verifier was fooled."""
        return self.status == "ok" and self.visible.all_passed and not self.hidden.all_passed
