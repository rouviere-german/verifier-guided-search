"""The search tree: node identity, backup along the path, and how the answer is chosen."""

from __future__ import annotations

import pytest

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.protocol import (
    GroupResult,
    TestResult,
    VerdictStatus,
    VisibleFeedback,
)
from verifier_guided_search.search.node import SearchTree

pytestmark = pytest.mark.unit


def feedback(
    passed: int, total: int, *, status: VerdictStatus = "ok", program: str = "x"
) -> VisibleFeedback:
    results = tuple(
        TestResult(name=f"v{index:02d}", status="pass" if index < passed else "fail")
        for index in range(total)
    )
    return VisibleFeedback(
        program_sha256=sha256_hex(program), status=status, visible=GroupResult(results=results)
    )


def test_a_new_tree_has_one_node_and_the_root_is_it() -> None:
    tree = SearchTree("def f():\n    return 1\n", (3,))
    assert len(tree) == 1
    assert tree.nodes == (tree.root,)
    assert tree.root.depth == 0
    assert tree.root.state == (3,)
    assert tree.root.parent is None
    assert not tree.root.evaluated
    assert not tree.root.expanded
    assert tree.best() is tree.root, "an unevaluated tree answers with its root"


def test_children_get_fresh_indices_and_record_what_was_proposed() -> None:
    tree = SearchTree("root")
    first = tree.add_child(tree.root, program="a", state=(1,), prior=0.6)
    second = tree.add_child(tree.root, program="b", state=(2,), prior=0.4)
    assert (first.index, second.index) == (1, 2)
    assert first.depth == second.depth == 1
    assert tree.root.children == [first, second]
    assert tree.root.proposed == {(1,), (2,)}
    assert len(tree) == 3


def test_program_hashes_identify_nodes_by_content() -> None:
    tree = SearchTree("same")
    twin = tree.add_child(tree.root, program="same", state=(1,), prior=1.0)
    assert twin.program_sha256 == tree.root.program_sha256
    assert twin is not tree.root


def test_path_to_root_is_ordered_from_the_node_upwards() -> None:
    tree = SearchTree("root")
    child = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    grandchild = tree.add_child(child, program="b", state=(1, 2), prior=1.0)
    assert grandchild.path_to_root() == [grandchild, child, tree.root]
    assert grandchild.depth == 2


def test_backup_adds_one_visit_and_the_value_to_every_ancestor() -> None:
    tree = SearchTree("root")
    child = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    grandchild = tree.add_child(child, program="b", state=(1, 2), prior=1.0)
    grandchild.backup(0.5)
    grandchild.backup(0.25)
    for node in (grandchild, child, tree.root):
        assert node.visits == 2
        assert node.total_value == pytest.approx(0.75)
        assert node.mean_value == pytest.approx(0.375)
    # A sibling is untouched: backup follows the path, not the level.
    sibling = tree.add_child(child, program="c", state=(1, 3), prior=1.0)
    assert sibling.visits == 0


def test_mean_value_of_an_unvisited_node_is_its_own_estimate() -> None:
    tree = SearchTree("root")
    tree.root.value = 0.42
    assert tree.root.visits == 0
    assert tree.root.mean_value == pytest.approx(0.42)


def test_a_visible_solving_node_wins_over_a_higher_scored_unsolved_one() -> None:
    tree = SearchTree("root")
    partial = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    partial.feedback = feedback(3, 4, program="a")
    partial.value = 0.99
    solved = tree.add_child(tree.root, program="b", state=(2,), prior=0.1)
    solved.feedback = feedback(4, 4, program="b")
    solved.value = 0.01
    assert solved.solved
    assert not partial.solved
    assert tree.best() is solved


def test_without_a_solution_the_best_visible_pass_fraction_wins() -> None:
    tree = SearchTree("root")
    weak = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    weak.feedback = feedback(1, 4, program="a")
    strong = tree.add_child(tree.root, program="b", state=(2,), prior=1.0)
    strong.feedback = feedback(3, 4, program="b")
    assert tree.best() is strong


def test_ties_break_towards_the_shallower_then_the_earlier_node() -> None:
    tree = SearchTree("root")
    shallow = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    shallow.feedback = feedback(2, 4, program="a")
    deep = tree.add_child(shallow, program="b", state=(1, 2), prior=1.0)
    deep.feedback = feedback(2, 4, program="b")
    assert tree.best() is shallow

    twin = tree.add_child(tree.root, program="c", state=(3,), prior=1.0)
    twin.feedback = feedback(2, 4, program="c")
    assert tree.best() is shallow, "equal depth and pass fraction must pick the earlier index"


def test_a_crashed_node_is_not_solved_even_with_every_visible_test_passing() -> None:
    """``all_passed`` on an empty group is false, and a non-ok status is never a solution."""
    tree = SearchTree("root")
    node = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    node.feedback = VisibleFeedback(
        program_sha256=sha256_hex("a"), status="blocked", visible=GroupResult()
    )
    assert not node.solved
    assert node.pass_fraction == 0.0
    assert tree.best() is node, "the only evaluated node is still the answer"


def test_evaluated_nodes_lists_only_nodes_with_feedback() -> None:
    tree = SearchTree("root")
    scored = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    scored.feedback = feedback(1, 2, program="a")
    tree.add_child(tree.root, program="b", state=(2,), prior=1.0)
    assert tree.evaluated_nodes() == (scored,)


def test_an_unevaluated_node_reports_a_zero_pass_fraction() -> None:
    tree = SearchTree("root")
    child = tree.add_child(tree.root, program="a", state=(1,), prior=1.0)
    assert child.pass_fraction == 0.0
    assert not child.solved
