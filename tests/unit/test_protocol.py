"""Grading lives in the parent, so it is a pure function and gets tested like one.

The child is sent each case's arguments and reports what the entry point returned; this is the half
that turns those outputs into pass/fail/error. It matters that it is unforgiving about rows that do
not line up with the cases: a child that reports the wrong number of rows, or renames one, has
stopped following the protocol, and the caller must turn that into ``crashed`` rather than into a
pass.
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from verifier_guided_search.sandbox.protocol import (
    MAX_OUTPUT_JSON_CHARS,
    CaseOutcome,
    grade_group,
)
from verifier_guided_search.tasks.models import TestCase

pytestmark = pytest.mark.unit


def case(name: str, expected: JsonValue) -> TestCase:
    return TestCase(name=name, args=(1,), expected=expected)


def returned(name: str, output_json: str) -> CaseOutcome:
    return CaseOutcome(name=name, outcome="returned", output_json=output_json)


def test_an_output_that_matches_its_expectation_passes() -> None:
    graded = grade_group(
        (case("v00", 2), case("v01", [1, 2])),
        (
            returned("v00", "2"),
            returned("v01", "[1, 2]"),
        ),
    )
    assert [row.status for row in graded.results] == ["pass", "pass"]
    assert graded.all_passed


def test_a_wrong_output_fails_and_a_raised_exception_errors() -> None:
    graded = grade_group(
        (case("v00", 2), case("v01", 3)),
        (
            returned("v00", "99"),
            CaseOutcome(name="v01", outcome="error", error_class="ValueError"),
        ),
    )
    assert [row.status for row in graded.results] == ["fail", "error"]
    assert graded.results[1].error_class == "ValueError"
    assert graded.failing_names == ("v00", "v01")


def test_a_tuple_expectation_compares_with_the_json_list_it_arrives_as() -> None:
    expectation = TestCase(name="v00", args=(1,), expected=[[1, 2], [3]])
    graded = grade_group((expectation,), (returned("v00", "[[1, 2], [3]]"),))
    assert graded.all_passed


def test_a_value_the_child_could_not_report_is_a_failure() -> None:
    for reason in ("type", "size", "depth", "non_finite"):
        graded = grade_group(
            (case("v00", 2),),
            (CaseOutcome(name="v00", outcome="unrepresentable", reason=reason),),
        )
        assert graded.results[0].status == "fail", reason
        assert graded.results[0].error_class is None


def test_output_that_is_not_json_is_a_failure_not_an_exception() -> None:
    graded = grade_group((case("v00", 2),), (returned("v00", "{ not json"),))
    assert graded.results[0].status == "fail"


def test_rows_that_do_not_line_up_with_the_cases_are_refused() -> None:
    with pytest.raises(ValueError, match="expected 2 case outcomes"):
        grade_group((case("v00", 2), case("v01", 2)), (returned("v00", "2"),))
    with pytest.raises(ValueError, match="does not match case"):
        grade_group((case("v00", 2),), (returned("somethingelse", "2"),))


def test_an_empty_group_is_graded_as_empty() -> None:
    graded = grade_group((), ())
    assert graded.total == 0
    assert not graded.all_passed


def test_a_reported_output_is_bounded() -> None:
    """A candidate does not get to make the parent parse an unbounded document."""
    with pytest.raises(ValueError, match="at most"):
        CaseOutcome(name="v00", outcome="returned", output_json="x" * (MAX_OUTPUT_JSON_CHARS + 1))
