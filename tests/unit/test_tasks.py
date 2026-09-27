"""The benchmark: 15 families, a reference that passes its own tests, and a broken start program.

The load-bearing test here is
:func:`test_every_reference_solution_passes_its_own_visible_and_hidden_tests`. Expected outputs are
produced by running the rendered reference at build time, so if that were wrong every solve rate in
the repo would be measured against nonsense. It is checked in this process; a companion integration
test (``tests/integration/test_sandbox_local.py``) confirms the same references pass inside the real
sandbox, which is where candidate programs are actually judged.

The second load-bearing test is
:func:`test_hidden_tests_include_edge_cases_the_visible_ones_do_not`: the visible/hidden asymmetry
is what makes a verifier false pass possible at all. If the hidden group were a random resample of
the visible one, the reported false-pass rate would just be noise.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.mutate import MutationSpace, hamming_distance
from verifier_guided_search.tasks.families import FAMILIES, family_by_name
from verifier_guided_search.tasks.suite import (
    DEFAULT_CORRUPTION_DEPTH,
    SUITE_SAFE_BUILTINS,
    TaskSuite,
    build_suite,
    build_task,
    default_suite,
    load_reference,
)

pytestmark = pytest.mark.unit


def test_there_are_fifteen_families_with_unique_names_and_entry_points() -> None:
    assert len(FAMILIES) == 15
    names = [family.name for family in FAMILIES]
    assert len(set(names)) == 15
    assert len({family.entry_point for family in FAMILIES}) == 15
    for family in FAMILIES:
        assert family_by_name(family.name) is family


def test_an_unknown_family_name_is_refused() -> None:
    with pytest.raises(KeyError):
        family_by_name("no_such_family")


def test_every_reference_solution_passes_its_own_visible_and_hidden_tests(
    small_suite: TaskSuite,
) -> None:
    """The ground truth of the whole benchmark: the reference is correct by construction."""
    for task in small_suite:
        function = load_reference(task.reference_solution, task.entry_point)
        for group in (task.visible, task.hidden):
            for case in group.cases:
                actual = function(*case.args)
                normalised = list(actual) if isinstance(actual, tuple) else actual
                assert normalised == case.expected, f"{task.task_id} {group.name} {case.name}"


def test_the_start_program_is_never_the_reference(small_suite: TaskSuite) -> None:
    """A start program equal to its reference would contribute nothing to any solve rate.

    That the start program actually *fails* is asserted in
    ``tests/integration/test_sandbox_local.py::test_every_start_program_fails_a_visible_test``,
    which runs it in the sandbox. It cannot be checked here: a dropped branch or a shifted loop
    bound can turn a reference into an infinite loop, and only the sandbox has a watchdog.
    """
    for task in small_suite:
        assert task.start_program != task.reference_solution
        assert task.start_state, f"{task.task_id} has no corruption applied"


def test_hidden_tests_include_edge_cases_the_visible_ones_do_not(small_suite: TaskSuite) -> None:
    for task in small_suite:
        visible_args = {repr(case.args) for case in task.visible.cases}
        hidden_args = {repr(case.args) for case in task.hidden.cases}
        assert hidden_args - visible_args, f"{task.task_id} hidden tests add nothing new"
        assert task.hidden.size >= task.visible.size, f"{task.task_id} has a thin hidden group"


def test_both_groups_are_non_empty_and_named(small_suite: TaskSuite) -> None:
    for task in small_suite:
        assert task.visible.name == "visible"
        assert task.hidden.name == "hidden"
        assert task.visible.size >= 3
        assert task.hidden.size >= 3


def test_the_start_program_is_the_reference_with_the_corruption_applied(
    small_suite: TaskSuite,
) -> None:
    for task in small_suite:
        space = MutationSpace(task.reference_solution)
        assert space.sha256 == task.space_sha256
        assert space.render(task.start_state) == task.start_program
        assert hamming_distance(task.start_state, ()) == len(task.start_state)
        assert len(task.start_state) == min(DEFAULT_CORRUPTION_DEPTH, len(space))


def test_a_corruption_depth_above_one_makes_single_edit_repair_impossible(
    small_suite: TaskSuite,
) -> None:
    """The reason best-of-N over this edit space cannot win: the reference is 2 flips away."""
    for task in small_suite:
        assert len(task.start_state) >= 2, f"{task.task_id} is one edit from the reference"


def test_instances_of_one_family_are_different_functions() -> None:
    suite = build_suite(instances_per_family=4, families=FAMILIES[:1])
    sources = {task.reference_solution for task in suite}
    statements = {task.statement for task in suite}
    assert len(sources) > 1 or len(statements) > 1, "instances are renamed copies"
    assert len({task.task_id for task in suite}) == 4


def test_building_the_suite_is_deterministic_in_the_seed() -> None:
    first = build_suite(instances_per_family=1, seed=5)
    again = build_suite(instances_per_family=1, seed=5)
    other = build_suite(instances_per_family=1, seed=6)
    assert first.sha256 == again.sha256
    assert first.sha256 != other.sha256


def test_the_suite_hash_covers_every_task() -> None:
    small = build_suite(instances_per_family=1, families=FAMILIES[:2])
    larger = build_suite(instances_per_family=2, families=FAMILIES[:2])
    assert small.sha256 != larger.sha256
    assert len(small) == 2
    assert len(larger) == 4


def test_the_default_suite_has_one_hundred_and_fifty_tasks() -> None:
    suite = default_suite(1)
    assert len(suite) == 15
    assert len(suite.families) == 15
    # The documented default; built lazily and cached, so this is a metadata check only.
    from verifier_guided_search.tasks.suite import DEFAULT_INSTANCES_PER_FAMILY

    assert DEFAULT_INSTANCES_PER_FAMILY * len(FAMILIES) == 150


def test_the_default_suite_is_cached() -> None:
    assert default_suite(1) is default_suite(1)


def test_task_ids_are_family_slash_index(small_suite: TaskSuite) -> None:
    for task in small_suite:
        assert task.task_id == f"{task.family}/00"
    assert small_suite.get(small_suite.tasks[0].task_id) is small_suite.tasks[0]
    with pytest.raises(KeyError, match="unknown task"):
        small_suite.get("nope/00")


def test_the_suite_rejects_being_empty_or_holding_duplicates(small_suite: TaskSuite) -> None:
    with pytest.raises(ValueError, match="at least one task"):
        TaskSuite([])
    first = small_suite.tasks[0]
    with pytest.raises(ValueError, match="unique"):
        TaskSuite([first, first])


def test_subset_takes_the_first_instances_of_every_family() -> None:
    suite = build_suite(instances_per_family=3, families=FAMILIES[:2])
    subset = suite.subset(per_family=2)
    assert len(subset) == 4
    assert [task.task_id for task in subset] == [
        f"{FAMILIES[0].name}/00",
        f"{FAMILIES[0].name}/01",
        f"{FAMILIES[1].name}/00",
        f"{FAMILIES[1].name}/01",
    ]
    with pytest.raises(ValueError, match="at least 1"):
        suite.subset(per_family=0)


def test_the_suite_hash_of_a_task_covers_the_tests_and_the_entry_point(
    small_suite: TaskSuite,
) -> None:
    task = small_suite.tasks[0]
    renamed = task.model_copy(update={"entry_point": "other_name"})
    assert renamed.suite_sha256 != task.suite_sha256
    assert task.suite_sha256 == task.model_copy().suite_sha256


def test_references_only_need_the_safe_builtins(small_suite: TaskSuite) -> None:
    """A reference that needed an import would make the benchmark measure the sandbox's policy."""
    for task in small_suite:
        assert "import " not in task.reference_solution
        load_reference(task.reference_solution, task.entry_point)
    assert "open" not in SUITE_SAFE_BUILTINS
    assert "__import__" not in SUITE_SAFE_BUILTINS
    assert "eval" not in SUITE_SAFE_BUILTINS


def test_a_reference_without_its_entry_point_is_refused() -> None:
    with pytest.raises(TypeError, match="does not define"):
        load_reference("def other():\n    return 1\n", "solve")


def test_a_reference_returning_a_non_json_value_is_refused() -> None:
    family = FAMILIES[0]
    broken = family.__class__(
        name="broken",
        entry_point="broken",
        statement="Returns a set.",
        template="def broken(text):\n    return {1, 2}\n",
        parameters=lambda index: {},
        visible_inputs=lambda rng, params: [["ab"]],
        hidden_inputs=lambda rng, params: [["cd"]],
    )
    with pytest.raises(TypeError, match="is not JSON data"):
        build_task(broken, 0, seed=1, corruption_depth=1)


def test_every_task_statement_describes_its_parameters(small_suite: TaskSuite) -> None:
    """A statement with an unsubstituted placeholder would be unusable by a live generator."""
    for task in small_suite:
        assert "$" not in task.statement
        assert "$" not in task.reference_solution
        assert len(task.statement) > 40, f"{task.task_id} has a stub statement"


def test_a_task_serialises_to_json_and_hashes_stably(small_suite: TaskSuite) -> None:
    task = small_suite.tasks[0]
    payload = task.model_dump(mode="json")
    assert isinstance(payload["visible"]["cases"], list)
    assert task.sha256 == task.model_validate(payload).sha256
