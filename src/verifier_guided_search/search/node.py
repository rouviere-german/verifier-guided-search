"""The search tree.

A node is one candidate program plus what the verifier said about it - the *visible* part only.
Statistics are the usual MCTS triple (prior, visit count, accumulated value) and are also read by
beam and best-of-N, so one node type serves every strategy and the comparison stays apples to
apples.

``proposed`` holds the generator states this node has already handed out, so expanding the same node
twice yields new candidates instead of the same ones again; that is what lets best-of-N draw more
than ``k`` samples from the root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.protocol import VisibleFeedback

__all__ = ["SearchNode", "SearchTree"]


@dataclass(eq=False)
class SearchNode:
    """One candidate program and its search statistics."""

    index: int
    program: str
    state: tuple[int, ...]
    depth: int
    prior: float = 1.0
    parent: SearchNode | None = None
    children: list[SearchNode] = field(default_factory=list)
    feedback: VisibleFeedback | None = None
    value: float = 0.0
    visits: int = 0
    total_value: float = 0.0
    expansions: int = 0
    terminal: bool = False
    proposed: set[tuple[int, ...]] = field(default_factory=set)

    @property
    def program_sha256(self) -> str:
        return sha256_hex(self.program)

    @property
    def evaluated(self) -> bool:
        return self.feedback is not None

    @property
    def expanded(self) -> bool:
        return self.expansions > 0

    @property
    def pass_fraction(self) -> float:
        return self.feedback.pass_fraction if self.feedback is not None else 0.0

    @property
    def solved(self) -> bool:
        return self.feedback is not None and self.feedback.solved

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else self.value

    def path_to_root(self) -> list[SearchNode]:
        """This node first, the root last."""
        chain: list[SearchNode] = []
        node: SearchNode | None = self
        while node is not None:
            chain.append(node)
            node = node.parent
        return chain

    def backup(self, value: float) -> None:
        """Add one visit and ``value`` to this node and every ancestor."""
        for node in self.path_to_root():
            node.visits += 1
            node.total_value += value


class SearchTree:
    """Owns node identity and the root, so nodes never need a global counter."""

    def __init__(self, root_program: str, root_state: tuple[int, ...] = ()) -> None:
        self._ids = count()
        self.root = SearchNode(
            index=next(self._ids), program=root_program, state=root_state, depth=0, prior=1.0
        )
        self._nodes: list[SearchNode] = [self.root]

    @property
    def nodes(self) -> tuple[SearchNode, ...]:
        return tuple(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def add_child(
        self, parent: SearchNode, *, program: str, state: tuple[int, ...], prior: float
    ) -> SearchNode:
        child = SearchNode(
            index=next(self._ids),
            program=program,
            state=state,
            depth=parent.depth + 1,
            prior=prior,
            parent=parent,
        )
        parent.children.append(child)
        parent.proposed.add(state)
        self._nodes.append(child)
        return child

    def evaluated_nodes(self) -> tuple[SearchNode, ...]:
        return tuple(node for node in self._nodes if node.evaluated)

    def best(self) -> SearchNode:
        """The answer the search returns.

        A node that satisfied every visible test wins; otherwise the highest visible pass fraction,
        breaking ties towards the shallower and then the earlier node so the choice is
        deterministic.
        """
        evaluated = self.evaluated_nodes()
        if not evaluated:
            return self.root
        solved = [node for node in evaluated if node.solved]
        pool = solved or list(evaluated)
        return min(pool, key=lambda node: (-node.pass_fraction, node.depth, node.index))
