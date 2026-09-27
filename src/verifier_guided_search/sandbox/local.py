"""Parent side of the sandbox: spawn one fresh interpreter per program and read back a verdict.

Layout of a single execution::

    <workroot>/control/harness.py        the child harness, copied once
    <workroot>/control/<exec>/job.json   written by the parent, read-only to the child
    <workroot>/control/<exec>/result.json  opened by the child before its guard goes up
    <workroot>/jail/<exec>/              the child's cwd and its only writable directory

``control/`` sits outside the jail, so a candidate program cannot open the job or the result
file through the filesystem. Inside the container ``workroot`` is a tmpfs mount, which is why the
jail is a directory rather than a chroot.

That is a filesystem control, and it is **not** what keeps the hidden tests secret: the child reads
the job into its own address space, and a candidate can walk ``sys._getframe`` into the harness's
locals without touching the filesystem at all. So the job carries every case's *arguments* and no
expected value, the child reports what the entry point returned, and ``grade_group`` compares
outputs with expectations here, in a process the candidate never runs in.

Every execution gets a fresh directory and a fresh process; nothing is reused between programs
except the copied harness. The environment handed to the child is an allowlist, so a secret in the
parent's environment is not visible to untrusted code even before the guard exists.

A verdict is accepted only when the result file parses, echoes the execution nonce and names the
program that was actually sent. Anything else - a killed process, a truncated file, a forged result
-
becomes ``crashed`` or ``timeout``, never a passing run.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil

# Spawning the sandboxed interpreter is this module's entire purpose.
import subprocess  # nosec B404
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from verifier_guided_search.core.fileio import atomic_write_text
from verifier_guided_search.core.redaction import redact_text
from verifier_guided_search.sandbox import child as child_module
from verifier_guided_search.sandbox.protocol import (
    ERROR_CLASS_PATTERN,
    CaseOutcome,
    ExecJob,
    GroupResult,
    Verdict,
    VerdictStatus,
    grade_group,
)
from verifier_guided_search.tasks.models import TestCase

__all__ = ["HARNESS_NAME", "LocalSandbox", "SandboxError", "child_environment"]

HARNESS_NAME: Final = "harness.py"
_CHILD_FLAGS: Final = ("-I", "-S", "-B")
#: Wall-clock grace on top of the child's own watchdog, to cover interpreter startup.
_KILL_GRACE_SECONDS: Final = 5.0
_MAX_DETAIL: Final = 2048
_VALID_STATUSES: Final = frozenset[str](
    {
        "ok",
        "syntax_error",
        "missing_entry_point",
        "load_error",
        "blocked",
        "timeout",
        "crashed",
    }
)
#: Environment variables the child keeps. Everything else, secrets included, is dropped.
_ENV_ALLOWLIST: Final = (
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "PATHEXT",
    "LANG",
    "LC_ALL",
)
_VIOLATION_MAX = 16
#: Cap on the result file the parent will read. The child writes a few kilobytes; anything larger is
#: a child that has stopped following the protocol, and it is refused rather than parsed.
_MAX_RESULT_BYTES: Final = 1024 * 1024


class SandboxError(RuntimeError):
    """The sandbox could not be prepared or invoked at all (not a failing program)."""


def child_environment() -> dict[str, str]:
    """A minimal environment for the child: no API keys, no proxy settings, no PYTHON* vars."""
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _clean(text: str, limit: int = _MAX_DETAIL) -> str:
    return redact_text(text)[:limit]


def _graded(raw: object, cases: Sequence[TestCase]) -> GroupResult:
    """Validate the child's per-case rows and grade them against ``cases``.

    Anything malformed - not a list, not objects, a row that will not validate, a row count or a
    name that does not line up with the cases - makes the whole verdict unusable. The caller turns
    that into ``crashed``, never into a pass.
    """
    if not isinstance(raw, list):
        raise TypeError("case outcome group is not a list")
    if not raw:
        # No run happened: a syntax error, a guard refusal, a timeout or a kill. An empty group has
        # a pass fraction of 0 and ``all_passed`` False, so nothing here can become a pass.
        return GroupResult()
    rows: list[CaseOutcome] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("case outcome is not an object")
        rows.append(CaseOutcome.model_validate(item))
    return grade_group(cases, rows)


def _read_result_file(path: Path) -> object:
    """Parse the result file, refusing one larger than the protocol allows.

    The child writes a few kilobytes. A child that writes a gigabyte has stopped following the
    protocol, and the parent declines to load it into memory to find out.
    """
    if path.stat().st_size > _MAX_RESULT_BYTES:
        raise ValueError("result file is larger than the protocol allows")
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _case_arguments(case: TestCase) -> dict[str, Any]:
    """The part of a test case the child is allowed to see: its name and its arguments.

    Spelled out field by field rather than dumping the model minus one key, so a new field on
    ``TestCase`` cannot start travelling into the sandbox by accident.
    """
    return {"name": case.name, "args": list(case.args)}


class LocalSandbox:
    """Runs programs in fresh child interpreters on this machine.

    In production the same code runs *inside* the runner container and is reached over the internal
    network (:class:`~verifier_guided_search.sandbox.client.HttpSandbox`). Locally it is called
    directly, which is what makes the offline demo, the tests and the benches work without Docker.
    """

    def __init__(
        self,
        workroot: Path,
        *,
        python_executable: str | None = None,
        kill_grace_seconds: float = _KILL_GRACE_SECONDS,
        keep_artifacts: bool = False,
    ) -> None:
        self._workroot = workroot
        self._python = python_executable or sys.executable
        self._grace = kill_grace_seconds
        self._keep = keep_artifacts
        self._control = workroot / "control"
        self._jails = workroot / "jail"
        self._harness = self._control / HARNESS_NAME
        self._prepared = False

    @property
    def workroot(self) -> Path:
        return self._workroot

    @property
    def harness_path(self) -> Path:
        return self._harness

    def prepare(self) -> None:
        """Create the directory layout and copy the harness. Idempotent."""
        if self._prepared:
            return
        self._control.mkdir(parents=True, exist_ok=True)
        self._jails.mkdir(parents=True, exist_ok=True)
        source = Path(child_module.__file__).read_text(encoding="utf-8")
        atomic_write_text(self._harness, source)
        self._prepared = True

    def ping(self) -> bool:
        """Readiness probe: the harness is in place and the interpreter starts."""
        try:
            self.prepare()
            # Fixed argv built from our own paths, no shell.
            completed = subprocess.run(  # noqa: S603  # nosec B603
                [self._python, *_CHILD_FLAGS, "-c", "pass"],
                check=False,
                capture_output=True,
                timeout=30,
                env=child_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0 and self._harness.is_file()

    def execute(self, job: ExecJob) -> Verdict:
        """Run one program against one task's tests and return the verdict."""
        self.prepare()
        exec_id = uuid.uuid4().hex
        control = self._control / exec_id
        jail = self._jails / exec_id
        control.mkdir(parents=True, exist_ok=False)
        jail.mkdir(parents=True, exist_ok=False)
        nonce = secrets.token_hex(16)
        job_path = control / "job.json"
        result_path = control / "result.json"
        payload: dict[str, Any] = {
            "nonce": nonce,
            "program": job.program,
            "program_sha256": job.program_sha256,
            "entry_point": job.entry_point,
            # Arguments only. The expected values stay in this process; see the module docstring.
            "visible": [_case_arguments(case) for case in job.visible],
            "hidden": [_case_arguments(case) for case in job.hidden],
            "limits": job.limits.model_dump(mode="json"),
            "jail": str(jail.resolve()),
        }
        atomic_write_text(job_path, json.dumps(payload, sort_keys=True))
        try:
            return self._invoke(job, jail, job_path, result_path, nonce)
        finally:
            if not self._keep:
                shutil.rmtree(control, ignore_errors=True)
                shutil.rmtree(jail, ignore_errors=True)

    def _invoke(
        self, job: ExecJob, jail: Path, job_path: Path, result_path: Path, nonce: str
    ) -> Verdict:
        command = [
            self._python,
            *_CHILD_FLAGS,
            str(self._harness.resolve()),
            str(job_path.resolve()),
            str(result_path.resolve()),
        ]
        timeout = job.limits.timeout_seconds + self._grace
        started = time.perf_counter()
        try:
            # Fixed argv built from our own paths, no shell; the program itself only ever
            # reaches the child as data in job.json.
            completed = subprocess.run(  # noqa: S603  # nosec B603
                command,
                check=False,
                capture_output=True,
                cwd=str(jail),
                env=child_environment(),
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                text=False,
            )
        except subprocess.TimeoutExpired as expired:
            duration = time.perf_counter() - started
            return self._verdict(
                job,
                status="timeout",
                duration=duration,
                exit_code=-1,
                detail="parent kill after the child watchdog did not fire",
                stdout=expired.stdout or b"",
                stderr=expired.stderr or b"",
            )
        except OSError as exc:
            raise SandboxError(f"could not start the sandbox interpreter: {exc}") from exc
        duration = time.perf_counter() - started
        return self._read_result(
            job,
            result_path,
            nonce,
            duration=duration,
            exit_code=completed.returncode,
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )

    def _read_result(
        self,
        job: ExecJob,
        result_path: Path,
        nonce: str,
        *,
        duration: float,
        exit_code: int,
        stdout: bytes,
        stderr: bytes,
    ) -> Verdict:
        try:
            raw = _read_result_file(result_path)
        except (OSError, ValueError) as exc:
            return self._verdict(
                job,
                status="crashed",
                duration=duration,
                exit_code=exit_code,
                detail=f"no usable result file ({type(exc).__name__}); "
                f"stderr: {stderr.decode('utf-8', 'replace')}",
                stdout=stdout,
                stderr=stderr,
            )
        problem = self._reject(raw, job, nonce)
        if problem is not None:
            return self._verdict(
                job,
                status="crashed",
                duration=duration,
                exit_code=exit_code,
                detail=problem,
                stdout=stdout,
                stderr=stderr,
            )
        # _reject returned above for every non-dict payload.
        assert isinstance(raw, dict)  # noqa: S101  # nosec B101
        status: VerdictStatus = raw["status"]
        try:
            visible = _graded(raw.get("visible", []), job.visible)
            hidden = _graded(raw.get("hidden", []), job.hidden)
        except (ValueError, TypeError) as exc:
            return self._verdict(
                job,
                status="crashed",
                duration=duration,
                exit_code=exit_code,
                detail=f"malformed case outcomes: {type(exc).__name__}",
                stdout=stdout,
                stderr=stderr,
            )
        violations = tuple(str(item)[:64] for item in raw.get("violations", [])[:_VIOLATION_MAX])
        return self._verdict(
            job,
            status=status,
            duration=duration,
            exit_code=exit_code,
            detail=str(raw.get("detail", "")),
            stdout=stdout,
            stderr=stderr,
            visible=visible,
            hidden=hidden,
            violations=violations,
        )

    @staticmethod
    def _reject(raw: object, job: ExecJob, nonce: str) -> str | None:
        """Return why a result must not be trusted, or None when it may be."""
        if not isinstance(raw, dict):
            return "result file is not a JSON object"
        if raw.get("nonce") != nonce:
            return "result nonce does not match this execution"
        if raw.get("program_sha256") != job.program_sha256:
            return "result names a different program than the one submitted"
        if raw.get("status") not in _VALID_STATUSES:
            return f"unknown verdict status {raw.get('status')!r}"
        return None

    def _verdict(
        self,
        job: ExecJob,
        *,
        status: VerdictStatus,
        duration: float,
        exit_code: int,
        detail: str,
        stdout: bytes,
        stderr: bytes,
        visible: GroupResult | None = None,
        hidden: GroupResult | None = None,
        violations: tuple[str, ...] = (),
    ) -> Verdict:
        return Verdict(
            program_sha256=job.program_sha256,
            suite_sha256=job.suite_sha256,
            status=status,
            visible=visible if visible is not None else GroupResult(),
            hidden=hidden if hidden is not None else GroupResult(),
            violations=violations,
            duration_seconds=round(duration, 6),
            exit_code=exit_code,
            detail=_clean(detail),
            stdout_bytes=len(stdout),
            stderr_bytes=len(stderr),
        )


# Re-exported so callers validating error-class names use the same pattern the child produces.
ERROR_CLASS = ERROR_CLASS_PATTERN
