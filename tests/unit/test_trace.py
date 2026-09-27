"""Search traces and the training set they become.

The point that matters is the split. Two mutants of one reference solution differ in a single
feature or two, so a row-level train/test split would put near-duplicates on both sides and report
an AUC that measures memorisation. :func:`split_by_task` splits by task, and
:func:`test_no_task_appears_on_both_sides_of_the_split` is the check that it actually does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.search.trace import (
    TraceRow,
    TraceStore,
    TrainingSet,
    split_by_task,
    to_training_set,
)
from verifier_guided_search.value.features import FEATURE_NAMES

pytestmark = pytest.mark.unit


def row(
    *,
    task: str = "family_a/00",
    program: str = "p0",
    hidden: bool = True,
    visible: bool = True,
    order: int = 0,
    strategy: str = "mcts",
) -> TraceRow:
    return TraceRow(
        task_id=task,
        family=task.split("/", maxsplit=1)[0],
        strategy=strategy,
        order=order,
        program_sha256=sha256_hex(program),
        features=tuple(float(index) for index in range(len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
        visible_all_passed=visible,
        hidden_all_passed=hidden,
        status="ok",
    )


def test_the_label_is_the_hidden_outcome() -> None:
    assert row(hidden=True).label == 1
    assert row(hidden=False).label == 0


def test_a_false_pass_is_a_visible_pass_with_a_hidden_failure() -> None:
    assert row(visible=True, hidden=False).false_pass is True
    assert row(visible=True, hidden=True).false_pass is False
    assert row(visible=False, hidden=False).false_pass is False


def test_the_training_set_deduplicates_by_task_and_program() -> None:
    """Four strategies evaluate overlapping candidates; one program is one training row."""
    rows = [
        row(program="p0", strategy="greedy"),
        row(program="p0", strategy="mcts"),
        row(program="p1", strategy="beam"),
        row(program="p0", task="family_b/00"),
    ]
    data = to_training_set(rows)
    assert len(data) == 3
    assert sorted(data.tasks) == ["family_a/00", "family_b/00"]


def test_the_training_set_reports_its_positive_rate_and_classes() -> None:
    data = to_training_set([row(program="p0", hidden=True), row(program="p1", hidden=False)])
    assert data.positive_rate == pytest.approx(0.5)
    assert data.both_classes_present()
    single = to_training_set([row(program="p0", hidden=False)])
    assert not single.both_classes_present()
    assert single.positive_rate == 0.0


def test_the_training_set_checks_its_own_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        TrainingSet([[0.0]], [1, 0], ["t/00"])


def test_select_keeps_only_the_named_tasks() -> None:
    data = to_training_set([row(task="a/00", program="p0"), row(task="b/00", program="p1")])
    picked = data.select(["a/00"])
    assert picked.task_ids == ["a/00"]
    assert len(picked) == 1
    assert picked.feature_names == FEATURE_NAMES


def many_rows(tasks: int = 10, per_task: int = 5) -> list[TraceRow]:
    return [
        row(
            task=f"family_{index}/00",
            program=f"p{index}_{repeat}",
            hidden=repeat % 2 == 0,
            order=repeat,
        )
        for index in range(tasks)
        for repeat in range(per_task)
    ]


def test_no_task_appears_on_both_sides_of_the_split() -> None:
    data = to_training_set(many_rows())
    train, holdout = split_by_task(data, holdout_fraction=0.3, seed=1)
    assert set(train.tasks) & set(holdout.tasks) == set()
    assert set(train.tasks) | set(holdout.tasks) == set(data.tasks)
    assert len(train) + len(holdout) == len(data)


def test_the_split_is_deterministic_in_the_seed() -> None:
    data = to_training_set(many_rows())
    first = split_by_task(data, seed=2)[1].tasks
    again = split_by_task(data, seed=2)[1].tasks
    other = split_by_task(data, seed=3)[1].tasks
    assert first == again
    assert first != other, "the split does not depend on the seed"


@given(fraction=st.floats(min_value=0.05, max_value=0.95))
def test_both_sides_of_the_split_are_always_non_empty(fraction: float) -> None:
    data = to_training_set(many_rows(tasks=6, per_task=2))
    train, holdout = split_by_task(data, holdout_fraction=fraction, seed=0)
    assert len(train.tasks) >= 1
    assert len(holdout.tasks) >= 1


def test_the_split_refuses_a_degenerate_fraction_or_too_few_tasks() -> None:
    data = to_training_set(many_rows())
    for fraction in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            split_by_task(data, holdout_fraction=fraction)
    single = to_training_set([row(task="only/00", program="p0")])
    with pytest.raises(ValueError, match="at least two tasks"):
        split_by_task(single)


def test_the_store_appends_rows_and_persists_them(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    store = TraceStore(path)
    store.append(row(program="p0"))
    store.append(row(program="p1", hidden=False))
    assert len(store) == 2
    reopened = TraceStore(path)
    assert len(reopened) == 2
    assert {item.program_sha256 for item in reopened.rows} == {
        sha256_hex("p0"),
        sha256_hex("p1"),
    }


def test_a_malformed_line_is_skipped_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    good = row(program="p0")
    path.write_text(
        "not json\n" + good.model_dump_json() + '\n{"task_id": "x"}\n\n', encoding="utf-8"
    )
    store = TraceStore(path)
    assert len(store) == 1
    assert store.skipped == 2


def test_an_in_memory_store_writes_nothing(tmp_path: Path) -> None:
    store = TraceStore()
    store.append(row())
    assert len(store) == 1
    assert not list(tmp_path.iterdir())


def test_the_store_produces_its_own_training_set(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.jsonl")
    for item in many_rows(tasks=4, per_task=3):
        store.append(item)
    data = store.training_set()
    assert len(data) == 12
    assert len(data.tasks) == 4
    assert data.feature_names == FEATURE_NAMES


def test_a_row_carries_its_own_feature_names_so_a_schema_change_is_visible() -> None:
    """A stored row names its features, so retraining on an old file cannot silently misalign."""
    old = row().model_copy(update={"feature_names": FEATURE_NAMES[:-1], "features": (1.0,)})
    data = to_training_set([old])
    assert data.feature_names == FEATURE_NAMES[:-1]
    assert data.feature_names != FEATURE_NAMES
