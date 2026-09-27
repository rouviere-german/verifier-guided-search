"""Builds the expansion prompt, and decides what is allowed into it.

Both generators use this module. The live one sends the prompt; the offline one only measures it, so
that a token budget means the same thing in both modes and offline budget curves sit on the same
axis as live ones.

Three rules about what goes in, all of them load-bearing:

1. **Everything that is not our instruction is untrusted data.** The task statement, the current
   program (model-written in the live case) and the failing test cases are each wrapped in an
   ``<untrusted-data>`` envelope with the kit's standard policy sentence in the system prompt.
2. **No free text from a sandbox run reaches the prompt.** A candidate's stdout, stderr and
   exception messages never leave :class:`Verdict.detail`. The verifier report in the prompt is
   built from :class:`VisibleFeedback`, so it holds test names, statuses and exception class names
   and nothing else.
3. **The generator gets no tools.** The request carries ``tools=()``. There is nothing for an
   injected "call this tool" instruction to reach, which is why a gullible model is contained by
   construction rather than by resisting the injection.
"""

from __future__ import annotations

from dataclasses import dataclass

from verifier_guided_search.agentic.envelope import (
    UNTRUSTED_DATA_POLICY,
    UntrustedData,
    wrap_user_content,
)
from verifier_guided_search.agentic.providers.base import ChatMessage, LLMRequest, TextBlock
from verifier_guided_search.core.canonical import canonical_json
from verifier_guided_search.sandbox.protocol import VisibleFeedback
from verifier_guided_search.tasks.models import Task

__all__ = [
    "CANDIDATE_SCHEMA",
    "GENERATOR_ROLE",
    "SYSTEM_PROMPT",
    "ExpansionPrompt",
    "build_expansion_prompt",
    "build_request",
]

GENERATOR_ROLE = "generator"
_MAX_FAILING_CASES = 4

SYSTEM_PROMPT = (
    "You repair small Python functions so that they pass a hidden test suite. You are given the "
    "task description, the current implementation and a structured report of which visible tests "
    "fail. Propose distinct, complete replacements for the function, best first. Each proposal "
    "must define the same entry point and must be valid Python. You have no tools and cannot run "
    "code; the caller runs every proposal in a sandbox.\n\n" + UNTRUSTED_DATA_POLICY
)

#: JSON Schema for the structured response. Constraints the strict-output subset rejects (such as
#: minItems) are stated in the descriptions instead; the caller re-validates every field anyway.
CANDIDATE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "description": "Between 1 and 8 proposals, best first.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["program"],
                "properties": {
                    "program": {
                        "type": "string",
                        "description": "The complete replacement module source.",
                    }
                },
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class ExpansionPrompt:
    """The rendered prompt for one expansion, plus the envelopes it is made of."""

    system: str
    user: str
    envelopes: tuple[UntrustedData, ...]

    @property
    def chars(self) -> int:
        return len(self.system) + len(self.user)


def _verifier_report(feedback: VisibleFeedback | None) -> str:
    """Structured only: names, statuses and exception class names."""
    if feedback is None:
        return canonical_json({"status": "not_yet_run"})
    return canonical_json(
        {
            "status": feedback.status,
            "visible_passed": feedback.visible.passed,
            "visible_total": feedback.visible.total,
            "failing": [
                {"name": item.name, "status": item.status, "error_class": item.error_class}
                for item in feedback.visible.results
                if item.status != "pass"
            ],
        }
    )


def build_expansion_prompt(
    task: Task, *, program: str, feedback: VisibleFeedback | None, k: int
) -> ExpansionPrompt:
    """Render the prompt for asking a generator for ``k`` children of ``program``."""
    failing = set(feedback.visible.failing_names) if feedback is not None else set()
    cases = [case for case in task.visible.cases if case.name in failing][:_MAX_FAILING_CASES]
    envelopes = (
        wrap_user_content("task_statement", task.statement),
        wrap_user_content("current_program", program),
        UntrustedData(source="tool:verifier_report", content=_verifier_report(feedback)),
        UntrustedData(
            source="user:failing_visible_tests",
            content=canonical_json([case.model_dump(mode="json") for case in cases]),
        ),
    )
    instruction = (
        f"Return exactly {k} candidate replacements for the function `{task.entry_point}`, "
        "ordered best first. Do not repeat the current implementation."
    )
    body = "\n\n".join(envelope.render() for envelope in envelopes)
    return ExpansionPrompt(
        system=SYSTEM_PROMPT, user=f"{instruction}\n\n{body}", envelopes=envelopes
    )


def build_request(prompt: ExpansionPrompt, *, max_tokens: int) -> LLMRequest:
    """A tool-free, structured-output request. ``tools`` is empty on purpose."""
    return LLMRequest(
        system=prompt.system,
        messages=(ChatMessage(role="user", content=(TextBlock(text=prompt.user),)),),
        tools=(),
        max_tokens=max_tokens,
        output_schema=CANDIDATE_SCHEMA,
        role=GENERATOR_ROLE,
    )
