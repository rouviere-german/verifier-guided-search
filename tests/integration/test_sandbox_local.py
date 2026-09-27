"""The real sandbox: a fresh interpreter per program, and a verdict for every way it can go wrong.

These tests spawn subprocesses, so they carry the ``sandbox`` marker and are kept few. What they
cover is the parent/child contract: every failure mode produces a *verdict* rather than an
exception, and a result that does not prove it came from this execution is refused rather than
trusted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.sandbox.local import (
    HARNESS_NAME,
    LocalSandbox,
    SandboxError,
    child_environment,
)
from verifier_guided_search.sandbox.protocol import (
    NO_RUN_STATUSES,
    ExecJob,
    ExecLimits,
    Verdict,
)
from verifier_guided_search.tasks.models import TestCase
from verifier_guided_search.tasks.suite import TaskSuite

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]


def job_for(program: str, *, timeout: float = 2.0, expected: object = 2) -> ExecJob:
    return ExecJob(
        program=program,
        entry_point="solve",
        visible=(TestCase(name="v00", args=(1,), expected=expected),),
        hidden=(TestCase(name="h00", args=(5,), expected=6),),
        limits=ExecLimits(timeout_seconds=timeout),
    )


_WORKING = "def solve(a):\n    return a + 1\n"


def test_a_correct_program_passes_both_groups(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(job_for(_WORKING))
    assert verdict.status == "ok"
    assert verdict.visible.all_passed
    assert verdict.hidden.all_passed
    assert verdict.hidden_solved
    assert not verdict.false_pass
    assert verdict.program_sha256 == sha256_hex(_WORKING)
    assert verdict.duration_seconds > 0
    assert verdict.exit_code == 0


def test_a_program_that_fools_the_visible_tests_is_reported_as_a_false_pass(
    sandbox: LocalSandbox,
) -> None:
    """The measurement the whole benchmark exists for."""
    overfitted = "def solve(a):\n    if a == 1:\n        return 2\n    return 0\n"
    verdict = sandbox.execute(job_for(overfitted))
    assert verdict.status == "ok"
    assert verdict.visible.all_passed
    assert not verdict.hidden.all_passed
    assert verdict.false_pass


def test_a_wrong_answer_is_a_failing_test_not_an_error(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(job_for("def solve(a):\n    return a\n"))
    assert verdict.status == "ok"
    assert verdict.visible.failing_names == ("v00",)
    assert verdict.visible.results[0].status == "fail"
    assert verdict.visible.results[0].error_class is None


def test_an_exception_is_reported_as_its_class_name_only(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(
        job_for("def solve(a):\n    raise ValueError('secret detail ' + str(a))\n")
    )
    assert verdict.status == "ok"
    assert verdict.visible.results[0].status == "error"
    assert verdict.visible.results[0].error_class == "ValueError"
    assert verdict.visible.error_class_counts() == {"ValueError": 1}
    # The message stays out of anything the search can read.
    assert "secret detail" not in verdict.visible_only().model_dump_json()


def test_a_syntax_error_is_a_verdict(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(job_for("def solve(:\n    return 1\n"))
    assert verdict.status == "syntax_error"
    assert verdict.status in NO_RUN_STATUSES
    assert verdict.visible.total == 0
    assert verdict.visible_only().pass_fraction == 0.0


def test_a_missing_entry_point_is_a_verdict(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(job_for("def other(a):\n    return a\n"))
    assert verdict.status == "missing_entry_point"


def test_a_module_level_failure_is_a_load_error(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(job_for("raise RuntimeError('at import time')\n"))
    assert verdict.status == "load_error"
    assert "RuntimeError" in verdict.detail


def test_an_endless_loop_is_killed_and_reported_as_a_timeout(sandbox: LocalSandbox) -> None:
    verdict = sandbox.execute(
        job_for("def solve(a):\n    while True:\n        pass\n", timeout=0.5)
    )
    assert verdict.status == "timeout"
    assert not verdict.visible.all_passed


def test_stdout_is_counted_but_never_forwarded(sandbox: LocalSandbox) -> None:
    noisy = "print('SANDBOX-STDOUT-MARKER' * 200)\n\n\ndef solve(a):\n    return a + 1\n"
    verdict = sandbox.execute(job_for(noisy))
    assert verdict.status == "ok"
    assert verdict.stdout_bytes > 0
    assert "SANDBOX-STDOUT-MARKER" not in verdict.detail
    assert "SANDBOX-STDOUT-MARKER" not in verdict.visible_only().model_dump_json()


def test_a_program_cannot_overwrite_its_own_result_file(sandbox: LocalSandbox) -> None:
    """The result path is outside the jail, so a candidate cannot rewrite the verdict about itself.

    The handle was opened before the guard went up and is never exposed to the candidate, and the
    guard refuses a fresh ``open`` for writing anywhere outside the jail. So the attempt is a
    ``blocked`` verdict, not a forged pass.
    """
    forging = """
import os


def solve(a):
    control = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
    for root, _dirs, files in os.walk(control):
        for name in files:
            if name == 'result.json':
                handle = open(os.path.join(root, name), 'w')
                handle.write('{"status": "ok"}')
                handle.close()
    return a + 1
"""
    verdict = sandbox.execute(job_for(forging))
    assert verdict.status == "blocked"
    assert verdict.violations, "the guard recorded no violation"
    assert not verdict.visible.all_passed


def test_a_forged_result_file_is_refused_by_the_parent(sandbox: LocalSandbox) -> None:
    """Drive ``_read_result`` directly with a file a hostile child could have written."""
    sandbox.prepare()
    job = job_for(_WORKING)
    passing: dict[str, object] = {
        "nonce": "the-right-nonce",
        "program_sha256": job.program_sha256,
        "status": "ok",
        "visible": [{"name": "v00", "status": "pass", "error_class": None}],
        "hidden": [{"name": "h00", "status": "pass", "error_class": None}],
    }
    path = sandbox.workroot / "forged.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def read(payload: Mapping[str, object], nonce: str) -> Verdict:
        path.write_text(json.dumps(dict(payload)), encoding="utf-8")
        return sandbox._read_result(
            job, path, nonce, duration=0.1, exit_code=0, stdout=b"", stderr=b""
        )

    honest = read(passing, "the-right-nonce")
    assert honest.status == "ok"
    assert honest.visible.all_passed

    replayed = read(passing, "a-different-nonce")
    assert replayed.status == "crashed"
    assert not replayed.visible.all_passed

    other_program = read({**passing, "program_sha256": "0" * 64}, "the-right-nonce")
    assert other_program.status == "crashed"

    made_up_status = read({**passing, "status": "definitely_fine"}, "the-right-nonce")
    assert made_up_status.status == "crashed"

    malformed_rows = read(
        {**passing, "visible": [{"name": "v00", "status": "great"}]}, "the-right-nonce"
    )
    assert malformed_rows.status == "crashed"

    path.write_text("{ not json", encoding="utf-8")
    unparseable = sandbox._read_result(
        job, path, "the-right-nonce", duration=0.1, exit_code=0, stdout=b"", stderr=b""
    )
    assert unparseable.status == "crashed"


def test_the_parent_rejects_a_result_whose_nonce_does_not_match(sandbox: LocalSandbox) -> None:
    job = job_for(_WORKING)
    bad = LocalSandbox._reject({"nonce": "wrong", "status": "ok"}, job, "right")
    assert bad == "result nonce does not match this execution"
    wrong_program = LocalSandbox._reject(
        {"nonce": "right", "program_sha256": "0" * 64, "status": "ok"}, job, "right"
    )
    assert wrong_program == "result names a different program than the one submitted"
    unknown = LocalSandbox._reject(
        {"nonce": "right", "program_sha256": job.program_sha256, "status": "sneaky"},
        job,
        "right",
    )
    assert unknown is not None
    assert "unknown verdict status" in unknown
    assert LocalSandbox._reject([], job, "right") is not None
    assert (
        LocalSandbox._reject(
            {"nonce": "right", "program_sha256": job.program_sha256, "status": "ok"},
            job,
            "right",
        )
        is None
    )


def test_a_missing_result_file_becomes_crashed(sandbox: LocalSandbox) -> None:
    """``os._exit`` skips the writer, so nothing is on disk: the parent must not invent a pass."""
    verdict = sandbox.execute(job_for("import os\nos._exit(0)\n"))
    assert verdict.status == "crashed"
    assert not verdict.visible.all_passed


def test_an_early_verdict_exits_cleanly(sandbox: LocalSandbox) -> None:
    """Every path that writes a verdict must also disarm the watchdog before returning.

    A timer left armed calls ``os._exit(3)``. With a long timeout the process would normally be gone
    first, so the observable signature is the exit code: 0 for a reported verdict, 3 only for a real
    timeout.
    """
    for program, expected in (
        ("def solve(:\n    pass\n", "syntax_error"),
        ("raise RuntimeError('boom')\n", "load_error"),
        ("def other(a):\n    return a\n", "missing_entry_point"),
        ("import socket\n\n\ndef solve(a):\n    socket.socket()\n    return a + 1\n", "blocked"),
        (_WORKING, "ok"),
    ):
        verdict = sandbox.execute(job_for(program, timeout=30.0))
        assert verdict.status == expected, program
        assert verdict.exit_code == 0, f"{expected} exited {verdict.exit_code} (3 = watchdog)"


def test_the_child_environment_carries_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-travel")
    monkeypatch.setenv("APP_API_KEY_SHA256", "a" * 64)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    env = child_environment()
    assert "ANTHROPIC_API_KEY" not in env
    assert "APP_API_KEY_SHA256" not in env
    assert "HTTP_PROXY" not in env
    assert "PYTHONPATH" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_each_execution_gets_a_fresh_directory_and_cleans_up(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path / "work")
    sandbox.execute(job_for(_WORKING))
    sandbox.execute(job_for(_WORKING))
    control = tmp_path / "work" / "control"
    jails = tmp_path / "work" / "jail"
    assert [path.name for path in control.iterdir()] == [HARNESS_NAME]
    assert list(jails.iterdir()) == []


def test_artifacts_can_be_kept_for_debugging(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path / "work", keep_artifacts=True)
    sandbox.execute(job_for(_WORKING))
    directories = [p for p in (tmp_path / "work" / "control").iterdir() if p.is_dir()]
    assert len(directories) == 1
    payload = json.loads((directories[0] / "job.json").read_text(encoding="utf-8"))
    assert payload["program_sha256"] == sha256_hex(_WORKING)
    assert "nonce" in payload
    assert (directories[0] / "result.json").is_file()


def test_prepare_is_idempotent_and_ping_reports_readiness(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path / "work")
    sandbox.prepare()
    sandbox.prepare()
    assert sandbox.harness_path.is_file()
    assert sandbox.workroot == tmp_path / "work"
    assert sandbox.ping() is True


def test_an_unusable_interpreter_raises_a_sandbox_error(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path / "work", python_executable=str(tmp_path / "no-such-python"))
    assert sandbox.ping() is False
    with pytest.raises(SandboxError, match="could not start the sandbox interpreter"):
        sandbox.execute(job_for(_WORKING))


def test_every_reference_solution_passes_inside_the_real_sandbox(
    shared_sandbox: CachingSandbox, small_suite: TaskSuite
) -> None:
    """The in-process check in test_tasks.py, repeated where candidates are actually judged."""
    for task in small_suite:
        job = ExecJob(
            program=task.reference_solution,
            entry_point=task.entry_point,
            visible=task.visible.cases,
            hidden=task.hidden.cases,
        )
        verdict = shared_sandbox.execute(job)
        assert verdict.status == "ok", f"{task.task_id}: {verdict.detail}"
        assert verdict.visible.all_passed, f"{task.task_id} failed a visible test"
        assert verdict.hidden.all_passed, f"{task.task_id} failed a hidden test"


def test_no_start_program_is_the_reference(small_suite: TaskSuite) -> None:
    """Every task starts from an edited program. What those edits *do* is classified separately.

    A corruption can be behaviour-preserving on a task's inputs, which makes the task a free win for
    every strategy. That case is measured rather than assumed away:
    ``tests/integration/test_suite_quality.py`` bounds how many tasks it may happen to, and the
    bench reports the share next to the solve rates.
    """
    for task in small_suite:
        assert task.start_program != task.reference_solution
