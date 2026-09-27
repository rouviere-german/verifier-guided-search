"""Applying a winning program to a workspace, only after a human approved that exact diff.

The search itself touches nothing outside its temporary directories. Turning a winning candidate
into a real edit is a separate, explicitly gated step, and this module is the whole gate.

What an approval is bound to
----------------------------
A :class:`ProposedPatch` is content-addressed by :attr:`ProposedPatch.diff_sha256`, the hash of the
target path, the SHA-256 of the file before and after, the unified diff, *and* the verdict summary
that justified the change (visible and hidden pass counts, the verdict status, the task and suite
hashes). Change any of those and the hash changes, so:

* an approval granted for one diff cannot release a different one;
* an approval cannot be reused after the file on disk moved on, because ``before_sha256`` no longer
  matches and :meth:`PatchApplier.apply` refuses before consulting the gate;
* the reviewer sees the same bytes the applier will write, and the same verdicts the search saw.

The gate itself is the kit's :class:`~verifier_guided_search.agentic.approval.ApprovalGate`, reached
through a one-tool registry (:func:`patch_tool`). That is deliberate rather than convenient: it
means patch approval inherits the properties already tested for tool approvals - single use, a TTL,
separation of duties (the principal that requested cannot approve), a refusal if the stored
arguments do not hash to the record, and a canonical-args round trip so nothing can be smuggled past
the hash.

Path confinement
----------------
``target`` is always a relative POSIX path resolved under the workspace root. A path that escapes -
absolute, ``..``, a drive letter, or a symlink whose real location is outside the root - raises
:class:`~verifier_guided_search.errors.PatchError` before anything is read or written. The write is
atomic, and every decision (proposed, refused, applied) goes to the audit chain.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from verifier_guided_search.agentic.approval import ApprovalGate, ApprovalStore
from verifier_guided_search.agentic.audit import AuditLog
from verifier_guided_search.agentic.errors import ApprovalRequiredError
from verifier_guided_search.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
    ValidatedCall,
)
from verifier_guided_search.core.canonical import canonical_json, sha256_hex
from verifier_guided_search.core.fileio import atomic_write_text
from verifier_guided_search.errors import PatchError
from verifier_guided_search.sandbox.protocol import Verdict
from verifier_guided_search.tasks.models import Task

__all__ = [
    "PATCH_TOOL_NAME",
    "ApplyPatchArgs",
    "PatchApplier",
    "PatchOutcome",
    "ProposedPatch",
    "VerdictSummary",
    "patch_registry",
    "patch_tool",
    "propose_patch",
    "unified_diff",
]

PATCH_TOOL_NAME: Final = "apply_patch"
_MAX_DIFF_CHARS: Final = 200_000


class VerdictSummary(BaseModel):
    """The evidence a reviewer is shown with the diff. Hidden counts are included on purpose.

    A reviewer is not the search: at approval time the hidden tests have already been run and
    withholding them would make the review worse, not more honest. What matters is that the search
    never saw them, which the type split in
    :mod:`verifier_guided_search.sandbox.protocol` enforces.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str
    visible_passed: int = Field(ge=0)
    visible_total: int = Field(ge=0)
    hidden_passed: int = Field(ge=0)
    hidden_total: int = Field(ge=0)
    strategy: str
    false_pass: bool

    @classmethod
    def of(cls, task: Task, verdict: Verdict, *, strategy: str) -> VerdictSummary:
        return cls(
            task_id=task.task_id,
            suite_sha256=verdict.suite_sha256,
            status=verdict.status,
            visible_passed=verdict.visible.passed,
            visible_total=verdict.visible.total,
            hidden_passed=verdict.hidden.passed,
            hidden_total=verdict.hidden.total,
            strategy=strategy,
            false_pass=verdict.false_pass,
        )

    def render(self) -> str:
        return (
            f"task {self.task_id} via {self.strategy}: status={self.status} "
            f"visible {self.visible_passed}/{self.visible_total} "
            f"hidden {self.hidden_passed}/{self.hidden_total}"
            + (" [VERIFIER FALSE PASS]" if self.false_pass else "")
        )


class ProposedPatch(BaseModel):
    """A concrete edit waiting for approval: where, from what, to what, and on what evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str = Field(min_length=1, max_length=512)
    before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    before: str = Field(max_length=64_000)
    after: str = Field(min_length=1, max_length=64_000)
    diff: str = Field(max_length=_MAX_DIFF_CHARS)
    evidence: VerdictSummary

    @property
    def diff_sha256(self) -> str:
        """The value an approval is bound to: everything a reviewer was shown, hashed."""
        return sha256_hex(
            canonical_json(
                {
                    "v": 1,
                    "target": self.target,
                    "before_sha256": self.before_sha256,
                    "after_sha256": self.after_sha256,
                    "diff": self.diff,
                    "evidence": self.evidence.model_dump(mode="json"),
                }
            )
        )

    @property
    def is_noop(self) -> bool:
        return self.before_sha256 == self.after_sha256

    def render(self) -> str:
        """What a human reads before deciding."""
        return (
            f"target:   {self.target}\n"
            f"evidence: {self.evidence.render()}\n"
            f"diff sha: {self.diff_sha256}\n\n{self.diff}"
        )


class ApplyPatchArgs(BaseModel):
    """The approved call's arguments. These, canonicalised, are what the approval hash covers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str = Field(min_length=1, max_length=512)
    diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PatchOutcome:
    """The result of a successful apply."""

    patch: ProposedPatch
    path: Path
    approval_hash: str
    bytes_written: int


def _relative(target: str) -> PurePosixPath:
    """Validate ``target`` as a workspace-relative POSIX path in canonical form, or raise.

    Canonical form is required rather than merely safe form, because the target *string* is part of
    :attr:`ProposedPatch.diff_sha256`. If ``./solution.py`` and ``solution.py`` were both accepted,
    one file would have two approval identities, and an approval for one would not be recognised as
    covering the other - confusing at best, and a way to accumulate stale approvals at worst.
    """
    if "\\" in target:
        raise PatchError(f"target must use forward slashes: {target!r}")
    if "\x00" in target:
        raise PatchError("target must not contain NUL")
    candidate = PurePosixPath(target)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise PatchError(f"target must be relative to the workspace: {target!r}")
    parts = candidate.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise PatchError(f"target must not contain '.' or '..' segments: {target!r}")
    if any(":" in part for part in parts):
        # "C:/Windows/x.py" parses as a relative POSIX path with a "C:" segment; on Windows that
        # segment is a drive reference, so it is refused rather than resolved.
        raise PatchError(f"target must not contain a drive or ':' segment: {target!r}")
    if candidate.as_posix() != target:
        raise PatchError(
            f"target must already be in canonical form (got {target!r}, "
            f"canonical form is {candidate.as_posix()!r})"
        )
    return candidate


def unified_diff(before: str, after: str, target: str) -> str:
    """A unified diff of the two sources, labelled with the target path."""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{target}",
        tofile=f"b/{target}",
        n=3,
    )
    text = "".join(lines)
    if len(text) > _MAX_DIFF_CHARS:
        raise PatchError("the diff is too large to review; narrow the change")
    return text


def propose_patch(
    *,
    target: str,
    before: str,
    after: str,
    task: Task,
    verdict: Verdict,
    strategy: str,
) -> ProposedPatch:
    """Build the reviewable patch. Pure: reads nothing and writes nothing."""
    _relative(target)
    return ProposedPatch(
        target=target,
        before_sha256=sha256_hex(before),
        after_sha256=sha256_hex(after),
        before=before,
        after=after,
        diff=unified_diff(before, after, target),
        evidence=VerdictSummary.of(task, verdict, strategy=strategy),
    )


def patch_tool(applier: PatchApplier) -> ToolDefinition[ApplyPatchArgs]:
    """The single side-effecting tool in this repo, so approval flows through the kit's gate."""
    return ToolDefinition(
        name=PATCH_TOOL_NAME,
        description=(
            "Write an approved candidate program into the workspace. The arguments name the file "
            "and the hash of the reviewed diff; the patch body is held by the applier, never by "
            "the caller."
        ),
        args_model=ApplyPatchArgs,
        capabilities=frozenset({Capability.WRITE_DATA}),
        handler=applier.handle_call,
        # Explicit, although WRITE_DATA already forces it: this is the repo's one write gate.
        requires_approval=True,
    )


def patch_registry(applier: PatchApplier) -> ToolRegistry:
    registry = ToolRegistry({Capability.WRITE_DATA})
    registry.register(patch_tool(applier))
    return registry


class PatchApplier:
    """Applies approved patches inside one workspace root, and nothing else.

    ``pending`` holds the patch bodies keyed by their diff hash. The tool call carries only the
    hash, so even a caller that could forge a call cannot supply its own file contents: the bytes
    written are the bytes that were reviewed.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        store: ApprovalStore,
        principal: str = "search",
        audit: AuditLog | None = None,
    ) -> None:
        self._root = workspace_root
        self._store = store
        self._principal = principal
        self._audit = audit
        self._gate = ApprovalGate(store, principal=principal)
        self._pending: dict[str, ProposedPatch] = {}
        self._registry = patch_registry(self)

    @property
    def workspace_root(self) -> Path:
        return self._root

    @property
    def principal(self) -> str:
        return self._principal

    @property
    def store(self) -> ApprovalStore:
        """The approval store, so a reviewer tool (or the demo) can decide requests."""
        return self._store

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    # ------------------------------------------------------------------ paths
    def resolve(self, target: str) -> Path:
        """The absolute path for ``target``, proven to stay inside the workspace root.

        Both the root and the candidate are resolved, so a symlink pointing out of the workspace is
        rejected too, not just a textual ``..``.
        """
        relative = _relative(target)
        root = self._root.resolve()
        candidate = (root / Path(*relative.parts)).resolve()
        if candidate != root and root not in candidate.parents:
            raise PatchError(f"target escapes the workspace: {target!r}")
        if candidate == root:
            raise PatchError("target must be a file inside the workspace, not the root")
        return candidate

    def read_current(self, target: str) -> str:
        path = self.resolve(target)
        if not path.exists():
            return ""
        if not path.is_file():
            raise PatchError(f"target is not a regular file: {target!r}")
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ propose / approve
    def propose(
        self, *, target: str, after: str, task: Task, verdict: Verdict, strategy: str
    ) -> ProposedPatch:
        """Diff the winning program against what is on disk and register it for approval."""
        before = self.read_current(target)
        patch = propose_patch(
            target=target,
            before=before,
            after=after,
            task=task,
            verdict=verdict,
            strategy=strategy,
        )
        self._pending[patch.diff_sha256] = patch
        self._log(
            "patch.proposed",
            {
                "target": patch.target,
                "diff_sha256": patch.diff_sha256,
                "before_sha256": patch.before_sha256,
                "after_sha256": patch.after_sha256,
                "evidence": patch.evidence.model_dump(mode="json"),
            },
        )
        return patch

    def call_for(self, patch: ProposedPatch) -> ValidatedCall:
        """The validated tool call this patch would run, without running or authorizing it."""
        from verifier_guided_search.agentic.providers.base import ToolUseBlock  # noqa: PLC0415

        return self._registry.validate_call(
            ToolUseBlock(
                id=f"patch-{patch.diff_sha256[:16]}",
                name=PATCH_TOOL_NAME,
                arguments={"target": patch.target, "diff_sha256": patch.diff_sha256},
            )
        )

    def request_approval(self, patch: ProposedPatch) -> str:
        """Open (or reuse) the approval request for this patch and return its id."""
        record = self._store.request(self.call_for(patch), requested_by=self._principal)
        return record.id

    # ------------------------------------------------------------------ apply
    def apply(self, patch: ProposedPatch) -> PatchOutcome:
        """Apply ``patch`` if an approval for exactly this diff has been granted.

        Raises :class:`ApprovalRequiredError` when no approval is available (and files the request),
        and :class:`PatchError` when the workspace moved on since the diff was produced.
        """
        if patch.diff_sha256 not in self._pending:
            raise PatchError("this patch was not proposed by this applier")
        if patch.is_noop:
            raise PatchError("the winning program is identical to the file on disk")
        # Checked before the gate, so a stale approval is never even consulted.
        self._require_fresh(patch)
        call = self.call_for(patch)
        try:
            authorized = self._gate.authorize(call)
        except ApprovalRequiredError as required:
            self._log(
                "patch.refused",
                {
                    "target": patch.target,
                    "diff_sha256": patch.diff_sha256,
                    "reason": "approval_required",
                    "request_id": required.request_id,
                },
            )
            raise
        call.run(authorized_hash=authorized)
        path = self.resolve(patch.target)
        written = len(patch.after.encode("utf-8"))
        self._log(
            "patch.applied",
            {
                "target": patch.target,
                "diff_sha256": patch.diff_sha256,
                "after_sha256": patch.after_sha256,
                "approval_hash": authorized,
                "bytes": written,
            },
        )
        return PatchOutcome(patch=patch, path=path, approval_hash=authorized, bytes_written=written)

    def handle_call(self, args: ApplyPatchArgs) -> JsonValue:
        """The tool handler. Reached only through an authorized :class:`ValidatedCall`."""
        patch = self._pending.get(args.diff_sha256)
        if patch is None:
            raise PatchError(f"no proposed patch with diff hash {args.diff_sha256[:12]}")
        if patch.target != args.target:
            raise PatchError("the approved target does not match the proposed patch")
        self._require_fresh(patch)
        path = self.resolve(patch.target)
        atomic_write_text(path, patch.after)
        return {
            "target": patch.target,
            "after_sha256": patch.after_sha256,
            "bytes": len(patch.after),
        }

    def _require_fresh(self, patch: ProposedPatch) -> None:
        current = sha256_hex(self.read_current(patch.target))
        if current != patch.before_sha256:
            raise PatchError(
                f"{patch.target} changed since the diff was reviewed "
                f"({current[:12]} on disk, {patch.before_sha256[:12]} reviewed); "
                "re-propose and re-approve"
            )

    def _log(self, action: str, data: dict[str, object]) -> None:
        if self._audit is None:
            return
        # The audit model takes JSON data; model dumps and primitives are already JSON.
        self._audit.append(self._principal, action, data)  # type: ignore[arg-type]
