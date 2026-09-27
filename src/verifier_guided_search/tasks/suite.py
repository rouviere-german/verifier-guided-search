"""Builds the benchmark: 15 families x N instances, each with a broken start program.

Expected test outputs come from running the rendered reference solution *in this process* at build
time. That is our own generated source, executed once per task in a namespace with no builtins
beyond the safe set the sandbox also allows, and a test asserts the reference passes all of its own
tests inside the real sandbox as well. Candidate programs are never run here.

The suite is content-addressed: :attr:`TaskSuite.sha256` covers every task, so a results file that
cites a suite hash pins exactly which benchmark produced it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import lru_cache
from typing import Any, Final, cast

from pydantic import JsonValue

from verifier_guided_search.core.canonical import canonical_json, sha256_hex
from verifier_guided_search.mutate import MutationSpace, corrupt
from verifier_guided_search.tasks.families import FAMILIES, ArgList, FamilySpec
from verifier_guided_search.tasks.models import Task, TestCase, TestGroup

__all__ = [
    "DEFAULT_CORRUPTION_DEPTH",
    "DEFAULT_INSTANCES_PER_FAMILY",
    "DEFAULT_SEED",
    "SUITE_SAFE_BUILTINS",
    "TaskSuite",
    "build_suite",
    "default_suite",
]

DEFAULT_SEED: Final = 20260927
DEFAULT_INSTANCES_PER_FAMILY: Final = 10
#: Edits applied to each reference solution to produce the starting program.
#:
#: This is the benchmark's difficulty dial, and the choice is measured rather than asserted:
#: ``bench/bench_calibration.py`` sweeps depth 1, 2 and 3 and writes
#: ``bench/results/calibration.json``. At depth 1 every strategy including best-of-N solves most
#: tasks and the spread between best and worst collapses; at depth 3 every strategy collapses to
#: the bottom of the range instead. Depth 2 is where the spread is widest, the only regime in which
#: the comparison says anything. It also means a task cannot be repaired by a single edit, so a
#: strategy that never leaves depth 1 (best-of-N over this edit space) is structurally unable to
#: solve it - see docs/adr/0003-mutation-operator-generator.md and the README Non-goals.
DEFAULT_CORRUPTION_DEPTH: Final = 2
_MAX_JSON_DEPTH: Final = 8

# Reference solutions need only these names. Building the suite with a restricted __builtins__
# keeps a typo in a template from reaching the filesystem or the network at build time.
SUITE_SAFE_BUILTINS: Final[Mapping[str, Any]] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _jsonable(value: object, depth: int = 0) -> JsonValue:
    """Convert a reference result to JSON data, rejecting anything outside the JSON model."""
    if depth > _MAX_JSON_DEPTH:
        raise TypeError("reference result nests too deeply to serialise")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    raise TypeError(f"reference result type {type(value).__name__} is not JSON data")


def load_reference(source: str, entry_point: str) -> Callable[..., object]:
    """Execute a rendered reference solution and return its entry point."""
    namespace: dict[str, Any] = {"__builtins__": dict(SUITE_SAFE_BUILTINS)}
    code = compile(source, "<reference>", "exec")
    exec(code, namespace)  # noqa: S102  # nosec B102
    function = namespace.get(entry_point)
    if not callable(function):
        raise TypeError(f"reference source does not define {entry_point}()")
    return cast("Callable[..., object]", function)


def _cases(
    function: Callable[..., object], inputs: Sequence[ArgList], prefix: str
) -> tuple[TestCase, ...]:
    """Build one group's cases by running the reference over each argument list.

    The cast is where :data:`~verifier_guided_search.tasks.families.JsonLike` literals become
    ``JsonValue``. Nothing is trusted by the cast itself: ``TestCase`` is a pydantic model with
    ``extra="forbid"``, so anything that is not JSON data is rejected on the next line.
    """
    out: list[TestCase] = []
    for index, args in enumerate(inputs):
        expected = _jsonable(function(*args))
        out.append(
            TestCase(
                name=f"{prefix}{index:02d}",
                args=cast("tuple[JsonValue, ...]", tuple(args)),
                expected=expected,
            )
        )
    return tuple(out)


def build_task(family: FamilySpec, index: int, *, seed: int, corruption_depth: int) -> Task:
    """Instantiate one task of ``family``: parameters, tests, reference and start program."""
    params = family.parameters(index)
    source = family.render_source(params)
    space = MutationSpace(source)
    function = load_reference(source, family.entry_point)
    # Benchmark fixture: reproducible task inputs, not cryptography.
    rng = random.Random(  # noqa: S311  # nosec B311
        sha256_hex(f"{seed}:{family.name}:{index}:inputs")
    )
    visible = _cases(function, family.visible_inputs(rng, params), "v")
    hidden = _cases(function, family.hidden_inputs(rng, params), "h")
    start_state = corrupt(space, depth=corruption_depth, seed=f"{seed}:{family.name}:{index}:start")
    return Task(
        task_id=f"{family.name}/{index:02d}",
        family=family.name,
        entry_point=family.entry_point,
        statement=family.render_statement(params),
        reference_solution=space.reference,
        start_program=space.render(start_state),
        start_state=start_state,
        space_sha256=space.sha256,
        visible=TestGroup(name="visible", cases=visible),
        hidden=TestGroup(name="hidden", cases=hidden),
    )


class TaskSuite:
    """An immutable, content-addressed collection of tasks."""

    def __init__(self, tasks: Sequence[Task]) -> None:
        if not tasks:
            raise ValueError("a task suite needs at least one task")
        self._tasks = tuple(tasks)
        self._by_id = {task.task_id: task for task in self._tasks}
        if len(self._by_id) != len(self._tasks):
            raise ValueError("task ids must be unique")

    @property
    def tasks(self) -> tuple[Task, ...]:
        return self._tasks

    @property
    def families(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for task in self._tasks:
            seen.setdefault(task.family, None)
        return tuple(seen)

    @property
    def sha256(self) -> str:
        return sha256_hex(canonical_json([task.sha256 for task in self._tasks]))

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def get(self, task_id: str) -> Task:
        task = self._by_id.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id!r}")
        return task

    def subset(self, *, per_family: int) -> TaskSuite:
        """The first ``per_family`` instances of every family, in suite order."""
        if per_family < 1:
            raise ValueError("per_family must be at least 1")
        counts: dict[str, int] = {}
        picked: list[Task] = []
        for task in self._tasks:
            seen = counts.get(task.family, 0)
            if seen < per_family:
                counts[task.family] = seen + 1
                picked.append(task)
        return TaskSuite(picked)


def build_suite(
    *,
    instances_per_family: int = DEFAULT_INSTANCES_PER_FAMILY,
    seed: int = DEFAULT_SEED,
    corruption_depth: int = DEFAULT_CORRUPTION_DEPTH,
    families: Sequence[FamilySpec] = FAMILIES,
) -> TaskSuite:
    """Build the benchmark. Deterministic in ``seed`` and the family definitions."""
    tasks = [
        build_task(family, index, seed=seed, corruption_depth=corruption_depth)
        for family in families
        for index in range(instances_per_family)
    ]
    return TaskSuite(tasks)


@lru_cache(maxsize=4)
def default_suite(
    instances_per_family: int = DEFAULT_INSTANCES_PER_FAMILY, seed: int = DEFAULT_SEED
) -> TaskSuite:
    """The default 150-task suite (cached, since building it re-executes every reference)."""
    return build_suite(instances_per_family=instances_per_family, seed=seed)
