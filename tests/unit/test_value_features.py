"""Features: the hidden tests must be unreachable from here, and that is checked from both sides.

The type signature already makes it impossible - :func:`node_features` takes a
:class:`VisibleFeedback`, which holds no hidden information - so the tests below pin the *behaviour*
that follows: permuting, replacing or emptying every hidden result leaves the vector untouched,
because it was never an input.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.protocol import (
    GroupResult,
    TestResult,
    Verdict,
    VisibleFeedback,
)
from verifier_guided_search.value.features import FEATURE_NAMES, FeatureContext, node_features

pytestmark = pytest.mark.unit

_PROGRAM_HASH = sha256_hex("candidate")
_SUITE_HASH = sha256_hex("suite")

statuses = st.sampled_from(["pass", "fail", "error"])
results = st.lists(
    st.builds(
        lambda index, status, klass: TestResult(
            name=f"t{index:02d}",
            status=status,
            error_class=klass if status == "error" else None,
        ),
        index=st.integers(min_value=0, max_value=20),
        status=statuses,
        klass=st.sampled_from(["ValueError", "IndexError", "TypeError"]),
    ),
    min_size=1,
    max_size=8,
)


def context(**overrides: float) -> FeatureContext:
    values: dict[str, float] = {
        "depth": 2,
        "edit_distance_from_root": 1,
        "program_chars": 300,
        "parent_pass_fraction": 0.5,
    }
    values.update(overrides)
    return FeatureContext(
        depth=int(values["depth"]),
        edit_distance_from_root=int(values["edit_distance_from_root"]),
        program_chars=int(values["program_chars"]),
        parent_pass_fraction=values["parent_pass_fraction"],
    )


def visible(rows: list[TestResult]) -> VisibleFeedback:
    return VisibleFeedback(
        program_sha256=_PROGRAM_HASH, status="ok", visible=GroupResult(results=tuple(rows))
    )


def test_the_vector_has_one_entry_per_named_feature() -> None:
    vector = node_features(visible([TestResult(name="t0", status="pass")]), context())
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in vector)


def test_the_documented_feature_order_is_what_is_produced() -> None:
    rows = [
        TestResult(name="t0", status="pass"),
        TestResult(name="t1", status="fail"),
        TestResult(name="t2", status="error", error_class="ValueError"),
        TestResult(name="t3", status="error", error_class="IndexError"),
    ]
    vector = node_features(visible(rows), context(depth=3, edit_distance_from_root=2))
    named = dict(zip(FEATURE_NAMES, vector, strict=True))
    assert named["visible_pass_fraction"] == pytest.approx(0.25)
    assert named["visible_failed"] == 3
    assert named["visible_total"] == 4
    assert named["status_ok"] == 1.0
    assert named["error_fraction"] == pytest.approx(0.5)
    assert named["fail_fraction"] == pytest.approx(0.25)
    assert named["distinct_error_classes"] == 2
    assert named["depth"] == 3
    assert named["edit_distance_from_root"] == 2
    assert named["program_chars_hundreds"] == pytest.approx(3.0)
    assert named["parent_pass_fraction"] == pytest.approx(0.5)
    assert named["pass_fraction_delta"] == pytest.approx(0.25 - 0.5)


@given(rows=results, hidden_rows=results)
def test_hidden_results_cannot_reach_the_feature_vector(
    rows: list[TestResult], hidden_rows: list[TestResult]
) -> None:
    """Build two verdicts that differ only in their hidden group; the features must be identical."""
    feedback = visible(rows)
    baseline = node_features(feedback, context())
    for hidden in (GroupResult(), GroupResult(results=tuple(hidden_rows))):
        verdict = Verdict(
            program_sha256=_PROGRAM_HASH,
            suite_sha256=_SUITE_HASH,
            status="ok",
            visible=GroupResult(results=tuple(rows)),
            hidden=hidden,
        )
        assert node_features(verdict.visible_only(), context()) == baseline


@given(rows=results)
def test_the_projection_to_visible_feedback_drops_the_hidden_group(
    rows: list[TestResult],
) -> None:
    verdict = Verdict(
        program_sha256=_PROGRAM_HASH,
        suite_sha256=_SUITE_HASH,
        status="ok",
        visible=GroupResult(results=tuple(rows)),
        hidden=GroupResult(results=tuple(rows)),
        detail="a candidate's stdout, which must never travel",
    )
    projected = verdict.visible_only()
    dumped = projected.model_dump(mode="json")
    assert "hidden" not in dumped
    assert "detail" not in dumped
    assert projected.visible == verdict.visible


@given(rows=results)
def test_features_are_deterministic(rows: list[TestResult]) -> None:
    feedback = visible(rows)
    assert node_features(feedback, context()) == node_features(feedback, context())


@given(rows=results)
def test_the_pass_fraction_delta_is_exactly_the_difference(rows: list[TestResult]) -> None:
    feedback = visible(rows)
    for parent in (0.0, 0.25, 1.0):
        named = dict(
            zip(
                FEATURE_NAMES,
                node_features(feedback, context(parent_pass_fraction=parent)),
                strict=True,
            )
        )
        assert named["pass_fraction_delta"] == pytest.approx(
            named["visible_pass_fraction"] - parent
        )


def test_an_empty_visible_group_produces_zeros_rather_than_dividing_by_zero() -> None:
    feedback = VisibleFeedback(
        program_sha256=_PROGRAM_HASH, status="blocked", visible=GroupResult()
    )
    named = dict(
        zip(FEATURE_NAMES, node_features(feedback, context(parent_pass_fraction=0.0)), strict=True)
    )
    assert named["visible_pass_fraction"] == 0.0
    assert named["visible_total"] == 0.0
    assert named["error_fraction"] == 0.0
    assert named["fail_fraction"] == 0.0
    assert named["status_ok"] == 0.0


def test_a_non_ok_status_clears_the_status_feature() -> None:
    for status in ("syntax_error", "timeout", "blocked", "crashed", "load_error"):
        feedback = VisibleFeedback(
            program_sha256=_PROGRAM_HASH,
            status=status,
            visible=GroupResult(),
        )
        named = dict(zip(FEATURE_NAMES, node_features(feedback, context()), strict=True))
        assert named["status_ok"] == 0.0


def test_distinct_error_classes_counts_classes_not_failures() -> None:
    same = [TestResult(name=f"t{i}", status="error", error_class="ValueError") for i in range(4)]
    varied = [
        TestResult(name="t0", status="error", error_class="ValueError"),
        TestResult(name="t1", status="error", error_class="IndexError"),
        TestResult(name="t2", status="error", error_class="KeyError"),
        TestResult(name="t3", status="error", error_class="TypeError"),
    ]
    index = FEATURE_NAMES.index("distinct_error_classes")
    assert node_features(visible(same), context())[index] == 1
    assert node_features(visible(varied), context())[index] == 4


def test_the_feedback_digest_changes_with_the_visible_outcome() -> None:
    first = visible([TestResult(name="t0", status="pass")])
    second = visible([TestResult(name="t0", status="fail")])
    assert first.digest != second.digest
    assert first.digest == visible([TestResult(name="t0", status="pass")]).digest
