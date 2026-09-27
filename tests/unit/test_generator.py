"""Candidate generators: the offline mutation one, and the live one against a scripted provider.

The mutation generator is what makes the offline curves mean anything, so the tests pin the three
properties the experiments rely on: it is deterministic given its seed, it never proposes a state
the node has already handed out, and its noise level really is the dial between "oracle" and "coin
flip".

The Claude generator is tested against :class:`FakeProvider` only. Nothing here touches the network
-
``tests/conftest.py`` blocks non-loopback sockets - and the request construction itself is covered
in
``tests/unit/test_anthropic_provider.py``.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.agentic.errors import ProviderError
from verifier_guided_search.agentic.providers.base import LLMResponse, Usage
from verifier_guided_search.agentic.providers.fake import FakeProvider, ScriptedPolicy
from verifier_guided_search.core.canonical import canonical_json
from verifier_guided_search.errors import SearchError
from verifier_guided_search.mutate import MutationSpace, hamming_distance
from verifier_guided_search.search.generator import (
    DEFAULT_NOISE,
    GENERATOR_MODEL,
    ClaudeGenerator,
    MutationGenerator,
    token_budget_for,
)
from verifier_guided_search.search.node import SearchTree
from verifier_guided_search.tasks.suite import TaskSuite

pytestmark = pytest.mark.unit


def tree_for(task: object) -> SearchTree:
    return SearchTree(task.start_program, tuple(task.start_state))  # type: ignore[attr-defined]


def test_the_generator_proposes_single_flip_neighbours(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    generator = MutationGenerator(seed=1)
    tree = tree_for(task)
    generation = generator.expand(task, tree.root, 4)
    assert len(generation.candidates) == 4
    assert generation.model == "fake-mutation"
    for candidate in generation.candidates:
        assert hamming_distance(candidate.state, task.start_state) == 1
        assert candidate.program != task.start_program


def test_candidates_come_back_ranked_with_decreasing_priors(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    generation = MutationGenerator(seed=2).expand(task, tree_for(task).root, 5)
    priors = [candidate.prior for candidate in generation.candidates]
    ranks = [candidate.rank for candidate in generation.candidates]
    assert ranks == [0, 1, 2, 3, 4]
    assert priors == sorted(priors, reverse=True)
    assert sum(priors) == pytest.approx(1.0)


def test_expansion_is_deterministic_given_the_seed(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    first = MutationGenerator(seed=3).expand(task, tree_for(task).root, 4)
    again = MutationGenerator(seed=3).expand(task, tree_for(task).root, 4)
    other = MutationGenerator(seed=4).expand(task, tree_for(task).root, 4)
    assert [c.state for c in first.candidates] == [c.state for c in again.candidates]
    assert [c.state for c in first.candidates] != [c.state for c in other.candidates]


def test_re_expanding_a_node_yields_states_it_has_not_proposed(tiny_suite: TaskSuite) -> None:
    """What lets best-of-N draw more than k samples from one node without repeating itself."""
    task = tiny_suite.tasks[0]
    generator = MutationGenerator(seed=5)
    tree = tree_for(task)
    seen: set[tuple[int, ...]] = set()
    for _ in range(3):
        generation = generator.expand(task, tree.root, 3)
        states = {candidate.state for candidate in generation.candidates}
        assert not (states & seen), "a state was proposed twice"
        seen |= states
        for candidate in generation.candidates:
            tree.add_child(
                tree.root,
                program=candidate.program,
                state=candidate.state,
                prior=candidate.prior,
            )
        tree.root.expansions += 1


def test_the_generator_runs_out_when_every_neighbour_has_been_proposed(
    tiny_suite: TaskSuite,
) -> None:
    task = tiny_suite.tasks[0]
    generator = MutationGenerator(seed=6)
    space = generator.space(task)
    tree = tree_for(task)
    tree.root.proposed = set(space.neighbours(task.start_state))
    generation = generator.expand(task, tree.root, 4)
    assert generation.candidates == ()


def test_zero_noise_makes_the_generator_an_oracle(tiny_suite: TaskSuite) -> None:
    """Without noise the top-ranked flip always reverts a real corruption."""
    task = tiny_suite.tasks[0]
    generation = MutationGenerator(seed=7, noise=0.0).expand(task, tree_for(task).root, 1)
    best = generation.candidates[0]
    assert len(best.state) < len(task.start_state), "the best flip did not revert a corruption"
    assert set(best.state) < set(task.start_state)


def test_large_noise_destroys_the_revert_signal(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    reverts = 0
    trials = 30
    for seed in range(trials):
        generation = MutationGenerator(seed=seed, noise=50.0).expand(task, tree_for(task).root, 1)
        if len(generation.candidates[0].state) < len(task.start_state):
            reverts += 1
    # With d corruptions among |sites| flips, an uninformed pick reverts about d/|sites| of the
    # time.
    space = MutationGenerator().space(task)
    chance = len(task.start_state) / len(space)
    assert reverts / trials < chance * 4, "noise did not remove the signal"


def test_the_calibrated_default_sits_between_the_extremes(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    reverts = sum(
        1
        for seed in range(40)
        if len(
            MutationGenerator(seed=seed, noise=DEFAULT_NOISE)
            .expand(task, tree_for(task).root, 1)
            .candidates[0]
            .state
        )
        < len(task.start_state)
    )
    assert 0 < reverts < 40, f"the default noise is degenerate ({reverts}/40 reverts)"


def test_the_generator_refuses_a_non_positive_k(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    with pytest.raises(SearchError, match="k must be at least 1"):
        MutationGenerator().expand(task, tree_for(task).root, 0)


def test_negative_noise_is_refused() -> None:
    with pytest.raises(SearchError, match="noise must not be negative"):
        MutationGenerator(noise=-0.1)


def test_a_task_whose_space_hash_moved_is_refused(tiny_suite: TaskSuite) -> None:
    """Protects against a stale results file citing a suite whose references changed."""
    task = tiny_suite.tasks[0]
    tampered = task.model_copy(update={"space_sha256": "0" * 64})
    with pytest.raises(SearchError, match="does not match the one the suite recorded"):
        MutationGenerator().space(tampered)


def test_the_generator_reports_tokens_on_the_same_axis_as_a_live_one(
    tiny_suite: TaskSuite,
) -> None:
    task = tiny_suite.tasks[0]
    generator = MutationGenerator()
    generation = generator.expand(task, tree_for(task).root, 3)
    assert generation.input_tokens > 0
    assert generation.output_tokens > 0
    assert generation.tokens == generation.input_tokens + generation.output_tokens
    assert generation.input_tokens == generator.estimate_input_tokens(task, tree_for(task).root, 3)
    assert generator.max_output_tokens(3) > generation.output_tokens


def test_the_token_budget_helper_scales_with_expansions_and_k() -> None:
    generator = MutationGenerator()
    small = token_budget_for(generator, expansions=2, k=2)
    large = token_budget_for(generator, expansions=8, k=4)
    assert large > small > 0


def _scripted(programs: list[str]) -> FakeProvider:
    payload = canonical_json({"candidates": [{"program": program} for program in programs]})
    return FakeProvider(
        ScriptedPolicy([LLMResponse(model="claude-sonnet-5", stop_reason="end_turn", text=payload)])
    )


def test_the_live_generator_parses_structured_candidates(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    space = MutationSpace(task.reference_solution)
    programs = [space.reference, space.render([space.sites[0].index])]
    generation = ClaudeGenerator(_scripted(programs)).expand(task, tree_for(task).root, 2)
    assert [c.program for c in generation.candidates] == programs
    assert generation.model == "claude-sonnet-5"
    assert [c.rank for c in generation.candidates] == [0, 1]


def test_the_live_generator_drops_candidates_that_do_not_parse(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    good = "def solve():\n    return 1\n"
    generation = ClaudeGenerator(_scripted(["def broken(:\n    pass\n", good, "   "])).expand(
        task, tree_for(task).root, 3
    )
    assert [c.program for c in generation.candidates] == [good]


def test_the_live_generator_raises_when_nothing_usable_comes_back(
    tiny_suite: TaskSuite,
) -> None:
    task = tiny_suite.tasks[0]
    with pytest.raises(ProviderError, match="no usable candidate programs"):
        ClaudeGenerator(_scripted(["def broken(:\n"])).expand(task, tree_for(task).root, 1)


def test_the_live_generator_falls_back_to_parsing_the_text_body(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    provider = FakeProvider(
        ScriptedPolicy(
            [
                LLMResponse(
                    model="claude-sonnet-5",
                    stop_reason="end_turn",
                    text='{"candidates": [{"program": "def solve():\\n    return 2\\n"}]}',
                )
            ]
        )
    )
    generation = ClaudeGenerator(provider).expand(task, tree_for(task).root, 1)
    assert generation.candidates[0].program == "def solve():\n    return 2\n"


def test_a_response_with_no_candidates_key_is_rejected(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    provider = FakeProvider(
        ScriptedPolicy(
            [LLMResponse(model="claude-sonnet-5", stop_reason="end_turn", text='{"other": []}')]
        )
    )
    with pytest.raises(ProviderError, match="no usable candidate programs"):
        ClaudeGenerator(provider).expand(task, tree_for(task).root, 1)


def test_the_live_generator_never_sends_tools(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    generator = ClaudeGenerator(_scripted(["def solve():\n    return 1\n"]))
    request = generator.build_request(task, tree_for(task).root, 2)
    assert request.tools == ()
    assert request.output_schema is not None


def test_the_live_generator_reports_the_usage_the_provider_returned(
    tiny_suite: TaskSuite,
) -> None:
    task = tiny_suite.tasks[0]
    provider = FakeProvider(
        ScriptedPolicy(
            [
                LLMResponse(
                    model="claude-sonnet-5",
                    stop_reason="end_turn",
                    text='{"candidates": [{"program": "def solve():\\n    return 1\\n"}]}',
                    usage=Usage(input_tokens=321, output_tokens=45),
                )
            ]
        )
    )
    generation = ClaudeGenerator(provider).expand(task, tree_for(task).root, 1)
    assert (generation.input_tokens, generation.output_tokens) == (321, 45)


def test_the_live_generator_refuses_a_non_positive_k(tiny_suite: TaskSuite) -> None:
    task = tiny_suite.tasks[0]
    with pytest.raises(SearchError, match="k must be at least 1"):
        ClaudeGenerator(_scripted(["def solve():\n    return 1\n"])).expand(
            task, tree_for(task).root, 0
        )


def test_the_default_live_model_is_sonnet() -> None:
    """The spec names Sonnet for the high-volume generator role; Opus is the project default."""
    assert GENERATOR_MODEL == "claude-sonnet-5"
    assert ClaudeGenerator(_scripted([])).model == "claude-sonnet-5"
