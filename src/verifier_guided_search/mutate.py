"""A finite, enumerable edit space over one reference program.

Search needs a candidate generator. Offline there is no model, so the edit space itself is made
explicit: parsing a reference solution yields a deterministic list of *mutation sites*, each a
single local edit (off-by-one on a literal, a swapped comparison or arithmetic operator, two
swapped arguments, a shifted slice boundary, a branch that never runs). A program is then a subset
of applied sites - a bit vector - and :meth:`MutationSpace.render` turns any subset back into
source. The empty subset renders the reference.

This gives the offline generator a real search space with structure: a corrupted start program has
`d` sites applied, one edit away are `|sites|` neighbours, and exactly one of the `2**|sites|`
programs is the reference. Moving one bit is a meaningful step, so a strategy that spends its budget
well finds the reference sooner than one that does not.

What it is *not*: a model of how a language model writes code. The neighbourhood is local and
finite, every mutant is syntactically valid, and the reference is always reachable. Offline curves
therefore describe search over this space, not over a model's output distribution; the live
generator (``search/generator.py``) has no such guarantee. This limit is stated in the README
Non-goals and in docs/adr/0003-mutation-operator-generator.md.

Site enumeration walks the AST breadth-first (:func:`ast.walk`) and indexes nodes by position in
that walk, so a site is ``(node_index, operator, payload)`` and indices are stable as long as the
reference source is unchanged. Rendering re-parses the reference, applies the selected edits to the
fresh tree and unparses it, so no rendered program shares mutable state with another.
"""

from __future__ import annotations

import ast
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from verifier_guided_search.core.canonical import sha256_hex

__all__ = [
    "DEFAULT_MAX_SITES",
    "MutationOperator",
    "MutationSite",
    "MutationSpace",
    "corrupt",
    "hamming_distance",
]


class MutationOperator(StrEnum):
    """The five edit kinds. Each one is a single, local, behaviour-changing rewrite."""

    OFF_BY_ONE = "off_by_one"
    OPERATOR_SWAP = "operator_swap"
    ARGUMENT_SWAP = "argument_swap"
    BOUNDARY_CHANGE = "boundary_change"
    DROPPED_BRANCH = "dropped_branch"


@dataclass(frozen=True, slots=True)
class MutationSite:
    """One applicable edit: which AST node, which operator, and the operator's parameter."""

    index: int
    node_index: int
    operator: MutationOperator
    detail: str

    @property
    def label(self) -> str:
        return f"{self.index}:{self.operator.value}:{self.detail}"


# Comparison operators are swapped inside these pairs (both directions).
_COMPARE_SWAPS: Final[Mapping[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_BINOP_SWAPS: Final[Mapping[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.FloorDiv,
}
# Operands are swapped only where the operator is not commutative, so the edit always matters.
_NON_COMMUTATIVE: Final[tuple[type[ast.operator], ...]] = (
    ast.Sub,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
DEFAULT_MAX_SITES: Final = 24


def _walk_index(tree: ast.AST) -> list[ast.AST]:
    """Nodes in breadth-first order; the position in this list identifies a node."""
    return list(ast.walk(tree))


def _is_docstring_constant(node: ast.AST, parents: Mapping[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    return isinstance(parent, ast.Expr)


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


class MutationSpace:
    """The edit space of one reference program.

    ``sites`` is deterministic for a given source. ``render(bits)`` applies exactly the selected
    sites; ``render(())`` is the reference, normalised through :func:`ast.unparse`.
    """

    def __init__(self, reference_source: str, *, max_sites: int = DEFAULT_MAX_SITES) -> None:
        self._source = reference_source
        self._reference_tree = ast.parse(reference_source)
        self._sites = self._enumerate(self._reference_tree, max_sites)
        self._reference = ast.unparse(self._reference_tree)

    @property
    def source(self) -> str:
        """The reference source as originally supplied."""
        return self._source

    @property
    def reference(self) -> str:
        """The reference source normalised the same way every rendered program is."""
        return self._reference

    @property
    def sites(self) -> tuple[MutationSite, ...]:
        return self._sites

    def __len__(self) -> int:
        return len(self._sites)

    @property
    def sha256(self) -> str:
        """Identifies the space: the reference plus the ordered site labels."""
        return sha256_hex(self._reference + "\n" + "\n".join(s.label for s in self._sites))

    @staticmethod
    def _enumerate(tree: ast.Module, max_sites: int) -> tuple[MutationSite, ...]:
        nodes = _walk_index(tree)
        parents = _parent_map(tree)
        found: list[tuple[int, MutationOperator, str]] = []
        for node_index, node in enumerate(nodes):
            found.extend(
                (node_index, operator, detail)
                for operator, detail in MutationSpace._sites_of(node, parents)
            )
        found.sort(key=lambda item: (item[0], item[1].value, item[2]))
        return tuple(
            MutationSite(index=index, node_index=node_index, operator=operator, detail=detail)
            for index, (node_index, operator, detail) in enumerate(found[:max_sites])
        )

    @staticmethod
    def _sites_of(
        node: ast.AST, parents: Mapping[int, ast.AST]
    ) -> Iterator[tuple[MutationOperator, str]]:
        if isinstance(node, ast.Constant):
            value = node.value
            # bool is a subclass of int; flipping True/False is an operator swap, not an off-by-one.
            # Written as one condition so the isinstance narrows ``value`` for the arithmetic below.
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and not _is_docstring_constant(node, parents)
            ):
                delta = 1 if value <= 0 else -1
                yield MutationOperator.OFF_BY_ONE, f"{value:+d}{delta:+d}"
            return
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and type(node.ops[0]) in _COMPARE_SWAPS:
                yield MutationOperator.OPERATOR_SWAP, type(node.ops[0]).__name__
            return
        if isinstance(node, ast.BoolOp):
            yield MutationOperator.OPERATOR_SWAP, type(node.op).__name__
            return
        if isinstance(node, ast.BinOp):
            if type(node.op) in _BINOP_SWAPS:
                yield MutationOperator.OPERATOR_SWAP, type(node.op).__name__
            if isinstance(node.op, _NON_COMMUTATIVE):
                yield MutationOperator.ARGUMENT_SWAP, "operands"
            return
        if isinstance(node, ast.Call):
            positional = [a for a in node.args if not isinstance(a, ast.Starred)]
            if len(positional) >= 2:
                yield MutationOperator.ARGUMENT_SWAP, "args01"
            return
        if isinstance(node, ast.Slice):
            yield MutationOperator.BOUNDARY_CHANGE, "upper" if node.upper is not None else "lower"
            return
        if isinstance(node, ast.If):
            yield (
                MutationOperator.DROPPED_BRANCH,
                "orelse" if node.orelse else "test",
            )

    def render(self, applied: Iterable[int]) -> str:
        """Source with exactly the sites in ``applied`` mutated. Unknown indices raise."""
        selected = self.normalize(applied)
        tree = ast.parse(self._source)
        nodes = _walk_index(tree)
        by_index = {site.index: site for site in self._sites}
        for index in selected:
            site = by_index[index]
            _apply(nodes[site.node_index], site)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def normalize(self, applied: Iterable[int]) -> tuple[int, ...]:
        """Validate and canonicalise a state: sorted, de-duplicated site indices."""
        known = {site.index for site in self._sites}
        selected = sorted(set(applied))
        unknown = [index for index in selected if index not in known]
        if unknown:
            raise KeyError(f"unknown mutation sites: {unknown}")
        return tuple(selected)

    def neighbours(self, state: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Every state one single-site flip away from ``state``, in site order."""
        current = set(self.normalize(state))
        out: list[tuple[int, ...]] = []
        for site in self._sites:
            flipped = current ^ {site.index}
            out.append(tuple(sorted(flipped)))
        return tuple(out)

    def flip(self, state: Sequence[int], index: int) -> tuple[int, ...]:
        current = set(self.normalize(state))
        if index not in {site.index for site in self._sites}:
            raise KeyError(f"unknown mutation site: {index}")
        return tuple(sorted(current ^ {index}))


def _apply(node: ast.AST, site: MutationSite) -> None:  # noqa: PLR0912
    """Apply one site's edit to ``node`` in place. Raises if the node is not the expected kind."""
    operator = site.operator
    if operator is MutationOperator.OFF_BY_ONE:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, int):
            raise TypeError(f"site {site.index} expected an int constant")
        node.value = node.value + (1 if node.value <= 0 else -1)
        return
    if operator is MutationOperator.OPERATOR_SWAP:
        if isinstance(node, ast.Compare):
            node.ops = [_COMPARE_SWAPS[type(node.ops[0])]()]
            return
        if isinstance(node, ast.BoolOp):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            return
        if isinstance(node, ast.BinOp):
            node.op = _BINOP_SWAPS[type(node.op)]()
            return
        raise TypeError(f"site {site.index} expected a comparison, boolean or arithmetic operator")
    if operator is MutationOperator.ARGUMENT_SWAP:
        if isinstance(node, ast.BinOp):
            node.left, node.right = node.right, node.left
            return
        if isinstance(node, ast.Call):
            args = list(node.args)
            args[0], args[1] = args[1], args[0]
            node.args = args
            return
        raise TypeError(f"site {site.index} expected a call or arithmetic expression")
    if operator is MutationOperator.BOUNDARY_CHANGE:
        if not isinstance(node, ast.Slice):
            raise TypeError(f"site {site.index} expected a slice")
        if node.upper is not None:
            node.upper = ast.BinOp(left=node.upper, op=ast.Sub(), right=ast.Constant(1))
        else:
            node.lower = ast.Constant(1)
        return
    if not isinstance(node, ast.If):
        raise TypeError(f"site {site.index} expected an if statement")
    if node.orelse:
        node.orelse = []
    else:
        node.test = ast.Constant(False)


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Number of sites that differ between two states; the edit distance of the space."""
    return len(set(left) ^ set(right))


def corrupt(space: MutationSpace, *, depth: int, seed: str) -> tuple[int, ...]:
    """Pick ``depth`` distinct sites deterministically from ``seed``.

    Used to build each benchmark task's starting program, so every strategy and every generator
    begins from exactly the same broken source.
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    count = min(depth, len(space))
    # Benchmark fixture: reproducible corruption sites, not cryptography.
    rng = random.Random(sha256_hex(seed))  # noqa: S311  # nosec B311
    return tuple(sorted(rng.sample([site.index for site in space.sites], count)))
