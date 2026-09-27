"""Properties of the edit space. If these hold, a state really is a program and back again."""

from __future__ import annotations

import ast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from verifier_guided_search.mutate import (
    DEFAULT_MAX_SITES,
    MutationOperator,
    MutationSpace,
    corrupt,
    hamming_distance,
)
from verifier_guided_search.tasks.families import FAMILIES

pytestmark = pytest.mark.unit

_SOURCE = """
def transform(values, limit):
    total = 0
    picked = []
    for index in range(len(values)):
        item = values[index]
        if item > limit:
            total = total + item - 1
        else:
            picked.append(min(item, limit))
    head = values[1:3]
    return [total, picked, head]
"""


@pytest.fixture(scope="module")
def space() -> MutationSpace:
    return MutationSpace(_SOURCE)


@pytest.fixture(scope="module")
def family_spaces() -> list[MutationSpace]:
    return [MutationSpace(f.render_source(f.parameters(0))) for f in FAMILIES]


def _states(space: MutationSpace) -> st.SearchStrategy[tuple[int, ...]]:
    indices = [site.index for site in space.sites]
    return st.lists(st.sampled_from(indices), unique=True, max_size=6).map(
        lambda picked: tuple(sorted(picked))
    )


def test_the_empty_state_renders_the_reference(space: MutationSpace) -> None:
    assert space.render(()) == space.reference
    assert ast.unparse(ast.parse(space.source)) == space.reference


def test_sites_are_enumerated_deterministically_and_capped(space: MutationSpace) -> None:
    again = MutationSpace(_SOURCE)
    assert [site.label for site in space.sites] == [site.label for site in again.sites]
    assert space.sha256 == again.sha256
    assert 0 < len(space) <= DEFAULT_MAX_SITES
    assert [site.index for site in space.sites] == list(range(len(space)))


def test_every_family_reference_has_sites_of_several_operators(
    family_spaces: list[MutationSpace],
) -> None:
    """A family whose reference offered only one kind of edit would make a degenerate search
    space."""
    for space in family_spaces:
        operators = {site.operator for site in space.sites}
        assert len(space) >= 4, f"only {len(space)} mutation sites"
        assert len(operators) >= 2, f"only operators {operators}"
    seen = {site.operator for space in family_spaces for site in space.sites}
    assert seen == set(MutationOperator), f"unused operators: {set(MutationOperator) - seen}"


@given(st.data())
def test_every_reachable_state_renders_to_parseable_python(
    space: MutationSpace, data: st.DataObject
) -> None:
    state = data.draw(_states(space))
    rendered = space.render(state)
    ast.parse(rendered)  # raises on failure, which is the assertion


@given(st.data())
def test_rendering_is_a_pure_function_of_the_state(
    space: MutationSpace, data: st.DataObject
) -> None:
    state = data.draw(_states(space))
    first = space.render(state)
    other = space.render(())
    assert space.render(state) == first
    assert other == space.reference, "rendering one state must not disturb the tree"


@given(st.data())
def test_a_nonempty_state_changes_the_program(space: MutationSpace, data: st.DataObject) -> None:
    """Every operator is behaviour-changing by construction, so the source must differ."""
    state = data.draw(_states(space).filter(bool))
    assert space.render(state) != space.reference


@given(st.data())
def test_flipping_a_site_twice_returns_to_the_same_state(
    space: MutationSpace, data: st.DataObject
) -> None:
    state = data.draw(_states(space))
    index = data.draw(st.sampled_from([site.index for site in space.sites]))
    once = space.flip(state, index)
    assert space.flip(once, index) == space.normalize(state)
    assert hamming_distance(state, once) == 1
    assert space.render(space.flip(once, index)) == space.render(state)


@given(st.data())
def test_neighbours_are_exactly_the_single_flips(space: MutationSpace, data: st.DataObject) -> None:
    state = data.draw(_states(space))
    neighbours = space.neighbours(state)
    assert len(neighbours) == len(space)
    assert len(set(neighbours)) == len(space), "two sites produced the same neighbour"
    for neighbour in neighbours:
        assert hamming_distance(state, neighbour) == 1


@given(st.data())
def test_normalize_is_idempotent_and_sorts(space: MutationSpace, data: st.DataObject) -> None:
    state = data.draw(_states(space))
    once = space.normalize(state)
    assert once == tuple(sorted(set(state)))
    assert space.normalize(once) == once
    assert space.normalize(list(reversed(once))) == once


def test_unknown_sites_are_refused(space: MutationSpace) -> None:
    with pytest.raises(KeyError, match="unknown mutation sites"):
        space.render([len(space) + 5])
    with pytest.raises(KeyError, match="unknown mutation site"):
        space.flip((), len(space) + 5)


def test_hamming_distance_is_a_metric_on_states(space: MutationSpace) -> None:
    a, b, c = (0,), (0, 1), (1, 2)
    assert hamming_distance(a, a) == 0
    assert hamming_distance(a, b) == hamming_distance(b, a)
    assert hamming_distance(a, c) <= hamming_distance(a, b) + hamming_distance(b, c)


@given(depth=st.integers(min_value=0, max_value=8), seed=st.integers(min_value=0, max_value=10_000))
def test_corrupt_is_deterministic_and_respects_the_depth(
    space: MutationSpace, depth: int, seed: int
) -> None:
    first = corrupt(space, depth=depth, seed=f"seed:{seed}")
    assert corrupt(space, depth=depth, seed=f"seed:{seed}") == first
    assert len(first) == min(depth, len(space))
    assert len(set(first)) == len(first)
    assert first == tuple(sorted(first))
    assert set(first) <= {site.index for site in space.sites}


def test_corrupt_rejects_a_negative_depth(space: MutationSpace) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        corrupt(space, depth=-1, seed="x")


def test_different_seeds_pick_different_corruptions(space: MutationSpace) -> None:
    picks = {corrupt(space, depth=2, seed=f"seed:{i}") for i in range(20)}
    assert len(picks) > 1, "corruption does not depend on the seed"


def test_a_docstring_is_not_treated_as_a_mutable_constant() -> None:
    """A docstring is an ast.Constant; mutating it would change nothing but the source hash."""
    documented = MutationSpace(
        'def f(x):\n    """Doc with the number 7 in it."""\n    return x + 3\n'
    )
    off_by_one = [s for s in documented.sites if s.operator is MutationOperator.OFF_BY_ONE]
    assert len(off_by_one) == 1, "only the literal 3 should be an off-by-one site"
    assert "Doc with the number 7" in documented.render([off_by_one[0].index])


def test_operator_swap_covers_comparisons_boolean_and_arithmetic() -> None:
    space = MutationSpace("def f(a, b):\n    return a < b and a - b == 0 or a % b > 1\n")
    swaps = [s for s in space.sites if s.operator is MutationOperator.OPERATOR_SWAP]
    assert len(swaps) >= 4
    for site in swaps:
        assert space.render([site.index]) != space.reference


def test_boundary_change_handles_both_open_and_closed_slices() -> None:
    space = MutationSpace("def f(xs):\n    return [xs[1:4], xs[2:]]\n")
    boundaries = [s for s in space.sites if s.operator is MutationOperator.BOUNDARY_CHANGE]
    assert {site.detail for site in boundaries} == {"upper", "lower"}
    for site in boundaries:
        ast.parse(space.render([site.index]))


def test_argument_swap_skips_commutative_operators() -> None:
    """Swapping the operands of ``a + b`` would be a no-op, so it must not be offered as a site."""
    space = MutationSpace("def f(a, b):\n    return a + b\n")
    assert not [s for s in space.sites if s.operator is MutationOperator.ARGUMENT_SWAP]
    minus = MutationSpace("def f(a, b):\n    return a - b\n")
    assert [s for s in minus.sites if s.operator is MutationOperator.ARGUMENT_SWAP]


def test_dropped_branch_removes_an_else_or_disables_the_test() -> None:
    with_else = MutationSpace(
        "def f(x):\n    if x:\n        return 1\n    else:\n        return 2\n"
    )
    site = next(s for s in with_else.sites if s.operator is MutationOperator.DROPPED_BRANCH)
    assert site.detail == "orelse"
    assert "else" not in with_else.render([site.index])

    without = MutationSpace("def f(x):\n    if x:\n        return 1\n    return 2\n")
    site = next(s for s in without.sites if s.operator is MutationOperator.DROPPED_BRANCH)
    assert site.detail == "test"
    assert "if False" in without.render([site.index])


def test_the_space_hash_changes_with_the_reference() -> None:
    other = MutationSpace(_SOURCE.replace("total = 0", "total = 1"))
    assert other.sha256 != MutationSpace(_SOURCE).sha256
