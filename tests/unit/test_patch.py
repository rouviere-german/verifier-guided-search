"""The approval gate on a patch: bound to the diff, single use, and confined to the workspace.

Four things have to hold, and each one is a way this could go wrong in a real system:

* nothing is written until a *different* principal approves;
* the approval is bound to the exact diff, so it does not travel to another change;
* the approval is stale the moment the file on disk moves, so a review of old bytes cannot release
  new ones;
* the target can never leave the workspace, whatever the path looks like.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verifier_guided_search.agentic.approval import ApprovalStatus, ApprovalStore
from verifier_guided_search.agentic.audit import AuditLog, verify_file
from verifier_guided_search.agentic.errors import ApprovalError, ApprovalRequiredError
from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.errors import PatchError
from verifier_guided_search.patch import (
    PATCH_TOOL_NAME,
    ApplyPatchArgs,
    PatchApplier,
    ProposedPatch,
    VerdictSummary,
    propose_patch,
    unified_diff,
)
from verifier_guided_search.sandbox.protocol import GroupResult, TestResult, Verdict
from verifier_guided_search.tasks.models import Task
from verifier_guided_search.tasks.suite import TaskSuite

pytestmark = pytest.mark.unit

_TARGET = "pkg/solution.py"
_SEARCH = "search"
_REVIEWER = "reviewer"


def make_verdict(task: Task, *, hidden_passed: bool = True) -> Verdict:
    return Verdict(
        program_sha256=sha256_hex("candidate"),
        suite_sha256=task.suite_sha256,
        status="ok",
        visible=GroupResult(results=(TestResult(name="v00", status="pass"),)),
        hidden=GroupResult(
            results=(TestResult(name="h00", status="pass" if hidden_passed else "fail"),)
        ),
    )


@pytest.fixture
def workspace(tmp_path: Path, tiny_suite: TaskSuite) -> tuple[Path, PatchApplier, Task]:
    task = tiny_suite.tasks[0]
    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / _TARGET).write_text(task.start_program, encoding="utf-8")
    applier = PatchApplier(
        root,
        store=ApprovalStore(tmp_path / "approvals.json"),
        principal=_SEARCH,
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    return root, applier, task


def propose(applier: PatchApplier, task: Task) -> ProposedPatch:
    return applier.propose(
        target=_TARGET,
        after=task.reference_solution,
        task=task,
        verdict=make_verdict(task),
        strategy="mcts",
    )


def approve(applier: PatchApplier, patch: ProposedPatch) -> str:
    request_id = applier.request_approval(patch)
    applier.store.decide(request_id, approver=_REVIEWER, approve=True)
    return request_id


def test_nothing_is_written_until_an_approval_exists(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    root, applier, task = workspace
    before = (root / _TARGET).read_text(encoding="utf-8")
    patch = propose(applier, task)
    with pytest.raises(ApprovalRequiredError) as info:
        applier.apply(patch)
    assert (root / _TARGET).read_text(encoding="utf-8") == before
    assert info.value.tool == PATCH_TOOL_NAME
    record = applier.store.get(info.value.request_id)
    assert record.status is ApprovalStatus.PENDING


def test_an_approved_patch_writes_exactly_the_reviewed_bytes(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    root, applier, task = workspace
    patch = propose(applier, task)
    approve(applier, patch)
    outcome = applier.apply(patch)
    assert (root / _TARGET).read_text(encoding="utf-8") == task.reference_solution
    assert outcome.path == (root / _TARGET).resolve()
    assert outcome.bytes_written == len(patch.after.encode("utf-8"))
    assert sha256_hex((root / _TARGET).read_text(encoding="utf-8")) == patch.after_sha256


def test_the_requester_cannot_approve_its_own_patch(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    patch = propose(applier, task)
    request_id = applier.request_approval(patch)
    with pytest.raises(ApprovalError, match="separation of duties"):
        applier.store.decide(request_id, approver=_SEARCH, approve=True)


def test_an_approval_is_single_use(workspace: tuple[Path, PatchApplier, Task]) -> None:
    root, applier, task = workspace
    patch = propose(applier, task)
    approve(applier, patch)
    applier.apply(patch)
    # The file has moved on, so the second attempt is refused as stale before the gate is consulted.
    with pytest.raises(PatchError, match="changed since the diff was reviewed"):
        applier.apply(patch)
    (root / _TARGET).write_text(task.start_program, encoding="utf-8")
    with pytest.raises(ApprovalRequiredError):
        applier.apply(patch)


def test_an_approval_does_not_release_a_different_diff(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    approved = propose(applier, task)
    approve(applier, approved)
    other = applier.propose(
        target=_TARGET,
        after=task.reference_solution + "\n# a different change\n",
        task=task,
        verdict=make_verdict(task),
        strategy="mcts",
    )
    assert other.diff_sha256 != approved.diff_sha256
    with pytest.raises(ApprovalRequiredError):
        applier.apply(other)


def test_a_patch_is_stale_once_the_file_changes(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    root, applier, task = workspace
    patch = propose(applier, task)
    approve(applier, patch)
    (root / _TARGET).write_text(
        task.start_program + "\n# edited by someone else\n", encoding="utf-8"
    )
    with pytest.raises(PatchError, match="changed since the diff was reviewed"):
        applier.apply(patch)


def test_the_diff_hash_covers_the_evidence_as_well_as_the_bytes(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    """A reviewer approves a change *and* the verdicts that justified it."""
    _, applier, task = workspace
    honest = propose(applier, task)
    misleading = propose_patch(
        target=_TARGET,
        before=honest.before,
        after=honest.after,
        task=task,
        verdict=make_verdict(task, hidden_passed=False),
        strategy="mcts",
    )
    assert misleading.after_sha256 == honest.after_sha256
    assert misleading.diff == honest.diff
    assert misleading.diff_sha256 != honest.diff_sha256
    assert misleading.evidence.false_pass is True


def test_a_no_op_patch_is_refused(workspace: tuple[Path, PatchApplier, Task]) -> None:
    _, applier, task = workspace
    patch = applier.propose(
        target=_TARGET,
        after=task.start_program,
        task=task,
        verdict=make_verdict(task),
        strategy="greedy",
    )
    assert patch.is_noop
    with pytest.raises(PatchError, match="identical to the file on disk"):
        applier.apply(patch)


def test_a_patch_this_applier_did_not_propose_is_refused(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    foreign = propose_patch(
        target=_TARGET,
        before="def solve():\n    return 0\n",
        after="def solve():\n    return 1\n",
        task=task,
        verdict=make_verdict(task),
        strategy="beam",
    )
    with pytest.raises(PatchError, match="not proposed by this applier"):
        applier.apply(foreign)


@pytest.mark.parametrize(
    "target",
    [
        "../outside.py",
        "pkg/../../outside.py",
        "/etc/passwd",
        "C:/Windows/system32/x.py",
        "pkg\\solution.py",
        "./solution.py",
        "",
    ],
)
def test_a_target_outside_the_workspace_is_refused(
    workspace: tuple[Path, PatchApplier, Task], target: str
) -> None:
    _, applier, _task = workspace
    with pytest.raises(PatchError):
        applier.resolve(target)


def test_the_workspace_root_itself_is_not_a_valid_target(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, _task = workspace
    with pytest.raises(PatchError, match=r"must be relative|not the root|must not contain"):
        applier.resolve(".")


def test_a_symlink_pointing_out_of_the_workspace_is_refused(
    workspace: tuple[Path, PatchApplier, Task], tmp_path: Path
) -> None:
    root, applier, _task = workspace
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    link = root / "pkg" / "link.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - needs privileges on Windows
        pytest.skip("this host does not allow creating symlinks")
    with pytest.raises(PatchError, match="escapes the workspace"):
        applier.resolve("pkg/link.py")


def test_a_directory_target_is_refused(workspace: tuple[Path, PatchApplier, Task]) -> None:
    _, applier, _task = workspace
    with pytest.raises(PatchError, match="not a regular file"):
        applier.read_current("pkg")


def test_a_new_file_diffs_against_nothing(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    patch = applier.propose(
        target="pkg/brand_new.py",
        after=task.reference_solution,
        task=task,
        verdict=make_verdict(task),
        strategy="greedy",
    )
    assert patch.before == ""
    assert patch.before_sha256 == sha256_hex("")
    approve(applier, patch)
    applier.apply(patch)
    assert (applier.workspace_root / "pkg/brand_new.py").is_file()


def test_the_handler_refuses_arguments_that_name_another_patch(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    patch = propose(applier, task)
    with pytest.raises(PatchError, match="no proposed patch with diff hash"):
        applier.handle_call(ApplyPatchArgs(target=_TARGET, diff_sha256="0" * 64))
    with pytest.raises(PatchError, match="approved target does not match"):
        applier.handle_call(ApplyPatchArgs(target="pkg/other.py", diff_sha256=patch.diff_sha256))


def test_the_patch_tool_declares_write_data_and_requires_approval(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, _task = workspace
    tool = applier.registry.get(PATCH_TOOL_NAME)
    assert tool.needs_approval is True
    assert {capability.value for capability in tool.capabilities} == {"write_data"}
    assert applier.registry.names == (PATCH_TOOL_NAME,)


def test_the_tool_call_carries_only_the_target_and_the_diff_hash(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    """The bytes written are the reviewed bytes: the call cannot supply its own content."""
    _, applier, task = workspace
    patch = propose(applier, task)
    call = applier.call_for(patch)
    assert set(call.canonical_args) == {"target", "diff_sha256"}
    assert call.canonical_args["diff_sha256"] == patch.diff_sha256
    assert "after" not in call.canonical_args


def test_every_decision_reaches_the_audit_chain(
    workspace: tuple[Path, PatchApplier, Task], tmp_path: Path
) -> None:
    _, applier, task = workspace
    patch = propose(applier, task)
    with pytest.raises(ApprovalRequiredError):
        applier.apply(patch)
    approve(applier, patch)
    applier.apply(patch)
    result = verify_file(tmp_path / "audit.jsonl")
    assert result.ok
    actions = [
        line.split('"action":"')[1].split('"')[0]
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert actions == ["patch.proposed", "patch.refused", "patch.applied"]


def test_the_rendered_patch_shows_the_diff_the_evidence_and_the_hash(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    patch = propose(applier, task)
    rendered = patch.render()
    assert _TARGET in rendered
    assert patch.diff_sha256 in rendered
    assert "hidden 1/1" in rendered
    assert "---" in rendered
    assert "+++" in rendered


def test_a_false_pass_is_flagged_in_the_evidence_a_reviewer_reads(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _root, _applier, task = workspace
    summary = VerdictSummary.of(task, make_verdict(task, hidden_passed=False), strategy="beam")
    assert summary.false_pass is True
    assert "VERIFIER FALSE PASS" in summary.render()


def test_unified_diff_refuses_an_unreasonably_large_change() -> None:
    huge = "\n".join(f"line {index}" for index in range(60_000))
    with pytest.raises(PatchError, match="too large to review"):
        unified_diff("", huge, _TARGET)


def test_pending_patches_are_listed_by_their_diff_hash(
    workspace: tuple[Path, PatchApplier, Task],
) -> None:
    _, applier, task = workspace
    assert not applier.pending, "a fresh applier has nothing pending"
    patch = propose(applier, task)
    assert list(applier.pending) == [patch.diff_sha256]
    assert applier.principal == _SEARCH
