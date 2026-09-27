"""Candidate generators: the seeded mutation generator (offline) and the Claude one (live).

Both answer the same question - "given this program and this verifier report, propose ``k`` new
programs, best first" - and both report what the step cost in tokens, so a budget level means the
same thing either way.

:class:`MutationGenerator` walks the edit space from ``mutate.py``. For a node whose state has some
sites applied, flipping a site *off* moves towards the reference and flipping one *on* moves away.
It scores every available flip with a fixed advantage for reverting, adds Gaussian noise, and
returns the top ``k``. The noise level is the calibration knob: at zero noise the generator is an
oracle and every strategy solves in ``d`` steps; at large noise it is a uniform sampler and search
cannot help. ``DEFAULT_NOISE`` was chosen by the sweep in ``bench/bench_calibration.py`` so that the
strategies separate. Priors come from rank, not from the noisy score.

:class:`ClaudeGenerator` sends the same prompt to the API with structured output, no tools, and
``claude-sonnet-5`` for the generator role - candidate generation is the high-volume call in a
search, and the spec's architecture names Sonnet for it. Every returned program is re-parsed here
before it becomes a candidate, so a malformed response costs one expansion rather than poisoning
the tree.
"""

from __future__ import annotations

import ast
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from verifier_guided_search.agentic.errors import ProviderError
from verifier_guided_search.agentic.providers.base import LLMProvider, LLMRequest
from verifier_guided_search.agentic.providers.fake import estimate_tokens
from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.errors import SearchError
from verifier_guided_search.mutate import MutationSpace
from verifier_guided_search.search.node import SearchNode
from verifier_guided_search.search.prompting import (
    GENERATOR_ROLE,
    build_expansion_prompt,
    build_request,
)
from verifier_guided_search.search.puct import rank_priors
from verifier_guided_search.tasks.models import Task

__all__ = [
    "DEFAULT_NOISE",
    "GENERATOR_MODEL",
    "Candidate",
    "CandidateGenerator",
    "ClaudeGenerator",
    "Generation",
    "MutationGenerator",
]

#: Calibrated so greedy, best-of-N, beam and MCTS separate on the benchmark; see ADR 0003.
DEFAULT_NOISE = 0.55
DEFAULT_REVERT_ADVANTAGE = 1.0
DEFAULT_INTRODUCE_ADVANTAGE = 0.25
DEFAULT_PRIOR_TEMPERATURE = 1.5
GENERATOR_MODEL = "claude-sonnet-5"
_MAX_OUTPUT_TOKENS_PER_CANDIDATE = 700
_OFFLINE_MODEL = "fake-mutation"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposal: its source, the generator state behind it, and its rank-derived prior."""

    program: str
    state: tuple[int, ...]
    prior: float
    rank: int


@dataclass(frozen=True, slots=True)
class Generation:
    """The result of one expansion, with what it cost."""

    candidates: tuple[Candidate, ...]
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class CandidateGenerator(Protocol):
    """Proposes children of a node. Implementations are deterministic given their seed."""

    name: str
    model: str

    def estimate_input_tokens(self, task: Task, node: SearchNode, k: int) -> int: ...

    def max_output_tokens(self, k: int) -> int: ...

    def expand(self, task: Task, node: SearchNode, k: int) -> Generation: ...


def _prompt_tokens(task: Task, node: SearchNode, k: int) -> int:
    prompt = build_expansion_prompt(task, program=node.program, feedback=node.feedback, k=k)
    return estimate_tokens(prompt.system) + estimate_tokens(prompt.user)


class MutationGenerator:
    """Offline generator over the edit space of each task's reference solution."""

    name = "mutation"
    model = _OFFLINE_MODEL

    def __init__(
        self,
        *,
        seed: int = 7,
        noise: float = DEFAULT_NOISE,
        revert_advantage: float = DEFAULT_REVERT_ADVANTAGE,
        introduce_advantage: float = DEFAULT_INTRODUCE_ADVANTAGE,
        prior_temperature: float = DEFAULT_PRIOR_TEMPERATURE,
    ) -> None:
        if noise < 0:
            raise SearchError("noise must not be negative")
        self._seed = seed
        self._noise = noise
        self._revert = revert_advantage
        self._introduce = introduce_advantage
        self._temperature = prior_temperature
        self._spaces: dict[str, MutationSpace] = {}

    @property
    def noise(self) -> float:
        return self._noise

    def space(self, task: Task) -> MutationSpace:
        """The edit space for ``task``, built once and reused."""
        space = self._spaces.get(task.task_id)
        if space is None:
            space = MutationSpace(task.reference_solution)
            if space.sha256 != task.space_sha256:
                raise SearchError(
                    f"edit space for {task.task_id} does not match the one the suite recorded"
                )
            self._spaces[task.task_id] = space
        return space

    def estimate_input_tokens(self, task: Task, node: SearchNode, k: int) -> int:
        return _prompt_tokens(task, node, k)

    def max_output_tokens(self, k: int) -> int:
        return k * _MAX_OUTPUT_TOKENS_PER_CANDIDATE

    def _rng(self, task: Task, node: SearchNode) -> random.Random:
        digest = node.feedback.digest if node.feedback is not None else "unevaluated"
        material = f"{self._seed}:{task.task_id}:{node.state}:{node.expansions}:{digest}"
        # Simulation randomness, seeded for reproducibility; not a security decision.
        return random.Random(sha256_hex(material))  # noqa: S311  # nosec B311

    def expand(self, task: Task, node: SearchNode, k: int) -> Generation:
        if k < 1:
            raise SearchError("k must be at least 1")
        space = self.space(task)
        applied = set(node.state)
        rng = self._rng(task, node)
        scored: list[tuple[float, int, tuple[int, ...]]] = []
        for site in space.sites:
            state = space.flip(node.state, site.index)
            if state in node.proposed:
                continue
            advantage = self._revert if site.index in applied else self._introduce
            scored.append((advantage + rng.gauss(0.0, self._noise), site.index, state))
        # Sort by score, then site index, so equal scores never depend on dict or set ordering.
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = scored[:k]
        priors = rank_priors(len(chosen), temperature=self._temperature) if chosen else ()
        candidates = tuple(
            Candidate(program=space.render(state), state=state, prior=prior, rank=rank)
            for rank, ((_, _, state), prior) in enumerate(zip(chosen, priors, strict=True))
        )
        output_tokens = sum(estimate_tokens(candidate.program) for candidate in candidates)
        return Generation(
            candidates=candidates,
            model=self.model,
            input_tokens=self.estimate_input_tokens(task, node, k),
            output_tokens=output_tokens,
        )


class ClaudeGenerator:
    """Live generator: one structured, tool-free request per expansion."""

    name = "claude"

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str = GENERATOR_MODEL,
        prior_temperature: float = DEFAULT_PRIOR_TEMPERATURE,
        max_output_tokens_per_candidate: int = _MAX_OUTPUT_TOKENS_PER_CANDIDATE,
    ) -> None:
        self._provider = provider
        self.model = model
        self._temperature = prior_temperature
        self._per_candidate = max_output_tokens_per_candidate

    def estimate_input_tokens(self, task: Task, node: SearchNode, k: int) -> int:
        return _prompt_tokens(task, node, k)

    def max_output_tokens(self, k: int) -> int:
        return k * self._per_candidate

    def build_request(self, task: Task, node: SearchNode, k: int) -> LLMRequest:
        prompt = build_expansion_prompt(task, program=node.program, feedback=node.feedback, k=k)
        return build_request(prompt, max_tokens=self.max_output_tokens(k))

    def expand(self, task: Task, node: SearchNode, k: int) -> Generation:
        if k < 1:
            raise SearchError("k must be at least 1")
        request = self.build_request(task, node, k)
        if request.tools:
            raise SearchError("the generator must never be given tools")
        if request.role != GENERATOR_ROLE:
            raise SearchError("the generator request must use the generator role")
        response = self._provider.complete(request)
        programs = _read_programs(response.parsed, response.text, limit=k)
        if not programs:
            raise ProviderError("generator returned no usable candidate programs")
        priors = rank_priors(len(programs), temperature=self._temperature)
        candidates = tuple(
            Candidate(program=program, state=(), prior=prior, rank=rank)
            for rank, (program, prior) in enumerate(zip(programs, priors, strict=True))
        )
        return Generation(
            candidates=candidates,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def _read_programs(
    parsed: Mapping[str, object] | None, text: str, *, limit: int
) -> tuple[str, ...]:
    """Pull candidate programs out of a structured response; skip anything that will not parse."""
    payload: object = parsed
    if payload is None:
        payload = _loads(text)
    if not isinstance(payload, Mapping):
        return ()
    raw = payload.get("candidates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    programs: list[str] = []
    for item in raw[: max(limit, 0)]:
        if not isinstance(item, Mapping):
            continue
        program = item.get("program")
        if not isinstance(program, str) or not program.strip():
            continue
        try:
            ast.parse(program)
        except (SyntaxError, ValueError, RecursionError):
            continue
        programs.append(program)
    return tuple(programs)


def _loads(text: str) -> object:
    import json  # noqa: PLC0415 - only needed on the fallback path

    try:
        return json.loads(text)
    except ValueError:
        return None


def token_budget_for(generator: CandidateGenerator, *, expansions: int, k: int) -> int:
    """A generous token cap for ``expansions`` expansions, so tokens never bind before
    candidates."""
    per_call = generator.max_output_tokens(k) + 4_000
    return math.ceil(per_call * max(expansions, 1) * 1.5)
