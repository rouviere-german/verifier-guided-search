"""The live generator, driven offline: is it actually reachable, and does a search run on it?

``ClaudeGenerator`` was implemented and unit-tested but had no caller: the demo, the bench and the
search service all built ``MutationGenerator`` directly and the CLI had no flag, so a key bought
nothing. These tests cover the wiring rather than the prompt building: ``build_generator`` picks
it, ``run_search`` runs on it, and the whole path is exercised through a recorded cassette, so no
key and no network are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from verifier_guided_search.agentic.providers.base import LLMRequest, LLMResponse, Usage
from verifier_guided_search.agentic.providers.replay import CassetteEntry, ReplayProvider
from verifier_guided_search.core.config import Settings
from verifier_guided_search.errors import SearchError
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.search.budget import BudgetAccountant, BudgetSpec
from verifier_guided_search.search.generator import (
    GENERATOR_MODEL,
    ClaudeGenerator,
    MutationGenerator,
    build_generator,
)
from verifier_guided_search.search.prompting import GENERATOR_ROLE
from verifier_guided_search.search.strategies import (
    SearchConfig,
    SearchContext,
    run_search,
)
from verifier_guided_search.tasks.suite import TaskSuite

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

_K = 2


class _CapturingProvider:
    """Answers every generator request with fixed candidates, and remembers the request hash.

    This is what recording a cassette does, reduced to what the test needs: the hash cannot be
    computed ahead of time, because the prompt depends on the verifier feedback the root already
    collected.
    """

    name = "capturing"

    def __init__(self, *, programs: tuple[str, ...]) -> None:
        self._programs = programs
        self.entries: list[CassetteEntry] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = LLMResponse(
            model=GENERATOR_MODEL,
            stop_reason="end_turn",
            text=json.dumps({"candidates": [{"program": item} for item in self._programs]}),
            usage=Usage(input_tokens=900, output_tokens=300),
        )
        self.entries.append(
            CassetteEntry(request_sha256=request.sha256(), role=request.role, response=response)
        )
        return response


def _accountant() -> BudgetAccountant:
    return BudgetAccountant(
        BudgetSpec.of(candidates=4, tokens=100_000, sandbox_seconds=20.0, usd=1.0)
    )


def _write_cassette(path: Path, entries: list[CassetteEntry]) -> Path:
    path.write_text("".join(f"{entry.model_dump_json()}\n" for entry in entries), encoding="utf-8")
    return path


def _empty(tmp_path: Path) -> Path:
    path = tmp_path / "unused.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def test_build_generator_returns_the_offline_generator_by_name() -> None:
    generator = build_generator("mutation", seed=3, noise=0.25)
    assert isinstance(generator, MutationGenerator)
    assert generator.model == "fake-mutation"
    assert generator.noise == 0.25


def test_build_generator_returns_the_live_generator_over_any_provider(tmp_path: Path) -> None:
    generator = build_generator("claude", provider=ReplayProvider(_empty(tmp_path)))
    assert isinstance(generator, ClaudeGenerator)
    assert generator.model == GENERATOR_MODEL


def test_build_generator_refuses_an_unknown_name() -> None:
    with pytest.raises(SearchError, match="unknown generator"):
        build_generator("telepathy")


def test_build_generator_refuses_the_live_generator_without_a_key() -> None:
    """Fail at construction, not on the first expansion halfway through a sweep."""
    settings = Settings(_env_file=None, anthropic_api_key=None)
    with pytest.raises(SearchError, match="needs ANTHROPIC_API_KEY"):
        build_generator("claude", settings=settings)


def test_the_live_generator_uses_the_configured_generator_model() -> None:
    settings = Settings(
        _env_file=None,
        anthropic_api_key=SecretStr("sk-ant-not-used-offline"),
        generator_model="claude-haiku-4-5",
    )
    generator = build_generator("claude", settings=settings)
    assert isinstance(generator, ClaudeGenerator)
    assert generator.model == "claude-haiku-4-5"


def test_a_search_runs_end_to_end_on_the_live_generator_and_replays_identically(
    tmp_path: Path, tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """The claim this file exists for: a real ``run_search`` on the live generator's code path.

    The provider offers the reference solution first, so the search solves on its first expansion.
    The solve is not the interesting part - these two are: the generator's tokens are charged from
    the response's reported usage rather than from the offline characters/4 estimate, and the run
    replays byte-for-byte off a cassette, which is what makes an offline test of the live path
    possible at all.
    """
    task = tiny_suite.tasks[0]
    provider = _CapturingProvider(programs=(task.reference_solution, task.start_program))
    generator = build_generator("claude", provider=provider)
    assert isinstance(generator, ClaudeGenerator)

    live_accountant = _accountant()
    config = SearchConfig(strategy="beam", children_per_expansion=_K)
    live = run_search(
        task,
        SearchContext(
            sandbox=shared_sandbox,
            generator=generator,
            accountant=live_accountant,
            config=config,
        ),
    )
    assert live.stop_reason == "solved"
    assert live.answer is not None
    assert live.answer.solved_visible
    assert live.answer.solved_hidden, "the provider offered the reference solution"
    assert provider.entries, "the search never called the generator"
    assert all(entry.role == GENERATOR_ROLE for entry in provider.entries)
    # Charged from the response's usage (900 + 300 per call), not from an offline estimate.
    assert live_accountant.committed.generator_tokens == 1_200 * len(provider.entries)
    assert live_accountant.committed.usd_micros > 0, "a live model is not free"
    assert live_accountant.generator_token_overshoot == 0

    replay = ReplayProvider(_write_cassette(tmp_path / "generator.jsonl", provider.entries))
    replayed_accountant = _accountant()
    replayed = run_search(
        task,
        SearchContext(
            sandbox=shared_sandbox,
            generator=build_generator("claude", provider=replay),
            accountant=replayed_accountant,
            config=config,
        ),
    )
    assert replayed.answer is not None
    assert replayed.answer.program_sha256 == live.answer.program_sha256
    assert replayed.stop_reason == live.stop_reason
    assert (
        replayed_accountant.committed.generator_tokens == live_accountant.committed.generator_tokens
    )
    assert replay.remaining == 0, "the replay served every recorded response"
