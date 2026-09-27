"""Benchmark data types: a task, its visible/hidden test split and its starting program.

The visible/hidden split is the honesty mechanism of the whole repo. Search may look only at visible
outcomes; hidden outcomes exist to score the answer afterwards. Anything a strategy reads must come
from :class:`~verifier_guided_search.sandbox.protocol.VisibleFeedback`, never from a
:class:`~verifier_guided_search.sandbox.protocol.Verdict`, and the type split is what makes that
reviewable rather than a convention.

Test arguments and expected values are JSON data, so a task hashes canonically, travels to the
runner container as JSON and cannot smuggle Python objects into the sandbox.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from verifier_guided_search.core.canonical import canonical_json, sha256_hex

__all__ = ["Task", "TestCase", "TestGroup"]

_IDENTIFIER = r"^[a-z_][a-z0-9_]{0,63}$"


class TestCase(BaseModel):
    """One call of the entry point with positional JSON arguments and its expected result."""

    # Not a pytest class, despite the name: these are benchmark tests, not this repo's tests.
    __test__ = False
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.\[\]-]+$")
    args: tuple[JsonValue, ...] = ()
    expected: JsonValue = None


class TestGroup(BaseModel):
    """A named set of test cases. ``visible`` drives search; ``hidden`` only scores it."""

    __test__ = False
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^(visible|hidden)$")
    cases: tuple[TestCase, ...] = Field(min_length=1)

    @property
    def size(self) -> int:
        return len(self.cases)


class Task(BaseModel):
    """One benchmark task: a statement, a broken starting program and a reference solution.

    ``statement`` is untrusted text as far as prompts are concerned: it is generated here, but the
    generator wraps it in an untrusted-data envelope so the same code path holds for a task whose
    statement came from outside.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+/[0-9]{2}$")
    family: str = Field(pattern=_IDENTIFIER)
    entry_point: str = Field(pattern=_IDENTIFIER)
    statement: str = Field(min_length=1, max_length=4096)
    reference_solution: str = Field(min_length=1, max_length=8192)
    start_program: str = Field(min_length=1, max_length=8192)
    start_state: tuple[int, ...] = ()
    space_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visible: TestGroup
    hidden: TestGroup

    @property
    def suite_sha256(self) -> str:
        """Hash of everything a verdict depends on: entry point and both test groups."""
        return sha256_hex(
            canonical_json(
                {
                    "entry_point": self.entry_point,
                    "visible": [case.model_dump(mode="json") for case in self.visible.cases],
                    "hidden": [case.model_dump(mode="json") for case in self.hidden.cases],
                }
            )
        )

    @property
    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))
