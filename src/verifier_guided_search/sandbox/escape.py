"""The sandbox escape suite: programs that try to get out, and what must happen to them.

Every attempt is an ordinary candidate program with one visible test whose expected value is the
string ``"escaped"``. So the attempt *passes its test only if the escape worked*, and "the escape
suite is 100% blocked" is the single sentence "no attempt passes its own test" - a claim that reads
the same way as every other verdict in the repo and cannot be satisfied by a program that merely
crashes for an unrelated reason, because :attr:`EscapeAttempt.expected_statuses` also pins the
verdict status.

The suite covers the categories in the spec: connecting a socket, resolving a name, forking, writing
outside the writable jail, reading a file outside the jail and the standard library, reading secrets
out of the environment, probing ``/proc``, loading native code through ``ctypes``, spawning a
subprocess, and sleeping past the wall-clock limit.

Four attempts exist because a review found the guard let them through, and they are the reason the
guard now checks *every* path argument of a two-path event and decides "is this a write?" from the
``os.open`` flags as well as the mode string: ``rename_out_of_jail``, ``hardlink_out_of_jail``,
``os_open_write_outside_jail`` and ``thread_bomb_via_subclasses``. The last one is still *not*
refused by the guard - the import deny-list cannot remove a live class object - and it is in the
suite to pin which control does stop it.

Two mechanisms do the work, and the suite is written so it is visible which one caught what:

* ``audit_hook`` attempts are refused by :func:`verifier_guided_search.sandbox.child.install_guard`.
  That hook behaves identically on every platform, so these attempts must report ``blocked``
  everywhere and :attr:`EscapeAttempt.must_be_blocked` is true for them.
* ``environment`` and ``platform`` attempts are stopped by something else - the parent's environment
  allowlist, the wall-clock watchdog, or simply the absence of the facility on this host. Their
  accepted statuses are wider and the docstring of each says why.

The suite is not the security boundary; the runner container is (see docs/THREAT_MODEL.md). What it
proves is that the cheap in-process guard really refuses the obvious moves, on every platform CI and
a laptop run on, and that a regression in the guard fails a test instead of going unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from verifier_guided_search.sandbox.protocol import ExecJob, ExecLimits, Verdict
from verifier_guided_search.tasks.models import TestCase

__all__ = [
    "ESCAPE_ENTRY_POINT",
    "ESCAPE_SUITE",
    "SUCCESS_MARKER",
    "EscapeAttempt",
    "EscapeOutcome",
    "ExecuteFn",
    "attempt_by_name",
    "blocked_fraction",
    "escape_job",
    "evaluate_attempt",
]

#: What an attempt returns when its escape succeeded. Nothing else counts as success.
SUCCESS_MARKER: Final = "escaped"
ESCAPE_ENTRY_POINT: Final = "probe"
Mechanism = Literal["audit_hook", "environment", "platform"]
#: Any sandbox client's ``execute``; the suite does not care which one runs it.
ExecuteFn = Callable[[ExecJob], Verdict]


@dataclass(frozen=True, slots=True)
class EscapeAttempt:
    """One escape attempt, its program, and the verdicts that count as "blocked"."""

    name: str
    category: str
    mechanism: Mechanism
    description: str
    program: str
    expected_statuses: frozenset[str]
    #: True when the in-process guard must refuse it on every platform.
    must_be_blocked: bool = True
    #: Substrings expected in ``Verdict.violations`` when the guard did the refusing.
    violation_hints: tuple[str, ...] = ()
    #: Seconds the attempt needs; the sleep attempt wants less than its own sleep.
    timeout_seconds: float = 1.5

    @property
    def limits(self) -> ExecLimits:
        return ExecLimits(timeout_seconds=self.timeout_seconds, max_processes=1)


def _attempt(
    name: str,
    category: str,
    mechanism: Mechanism,
    description: str,
    body: str,
    *,
    statuses: Iterable[str],
    must_be_blocked: bool = True,
    hints: Sequence[str] = (),
    timeout_seconds: float = 1.5,
) -> EscapeAttempt:
    return EscapeAttempt(
        name=name,
        category=category,
        mechanism=mechanism,
        description=description,
        program=body.strip() + "\n",
        expected_statuses=frozenset(statuses),
        must_be_blocked=must_be_blocked,
        violation_hints=tuple(hints),
        timeout_seconds=timeout_seconds,
    )


# The guard raises SandboxViolation, which derives from BaseException, so the broad ``except
# Exception`` in each program below cannot swallow it: a caught exception means the escape failed
# for an ordinary reason, an uncaught violation means the guard refused it. Either way the program
# does not return the marker.
_TRY = "    try:\n"
_EXCEPT = '    except Exception:\n        return "failed"\n'


_SOCKET_CONNECT = _attempt(
    "socket_connect",
    "network",
    "audit_hook",
    "Open a TCP connection to a routable address.",
    f"""
def probe():
{_TRY}        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(("1.1.1.1", 80))
        sock.close()
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["import:socket", "socket."],
)

_DNS_LOOKUP = _attempt(
    "dns_lookup",
    "network",
    "audit_hook",
    "Resolve a hostname, which leaks data even without a connection.",
    f"""
def probe():
{_TRY}        import socket

        socket.gethostbyname("example.com")
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["import:socket", "socket."],
)

_HTTP_FETCH = _attempt(
    "http_fetch",
    "network",
    "audit_hook",
    "Exfiltrate over HTTP with the standard library client.",
    f"""
def probe():
{_TRY}        import urllib.request

        urllib.request.urlopen("http://example.com/", timeout=1)
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["import:urllib"],
)

_SUBPROCESS = _attempt(
    "subprocess_spawn",
    "process",
    "audit_hook",
    "Shell out to a helper process that is not under the guard.",
    f"""
def probe():
{_TRY}        import subprocess

        subprocess.run(["echo", "hi"], capture_output=True, check=False)
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["import:subprocess", "subprocess.Popen"],
)

_CTYPES = _attempt(
    "ctypes_native_call",
    "process",
    "audit_hook",
    "Load native code, which runs outside anything Python can audit.",
    f"""
def probe():
{_TRY}        import ctypes

        ctypes.CDLL(None)
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["import:ctypes", "ctypes."],
)

_WRITE_OUTSIDE = _attempt(
    "write_outside_jail",
    "filesystem",
    "audit_hook",
    "Write a file outside the only writable directory.",
    f"""
def probe():
    import os

    parent = os.path.dirname(os.path.abspath(os.getcwd()))
    target = os.path.join(parent, "escaped.txt")
{_TRY}        handle = open(target, "w")
        handle.write("x")
        handle.close()
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["open:write"],
)

_READ_OUTSIDE = _attempt(
    "read_outside_jail",
    "filesystem",
    "audit_hook",
    "Read the job file, which holds the hidden tests, from outside the jail.",
    f"""
def probe():
    import os

    control = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
    target = os.path.join(control, "control", "harness.py")
{_TRY}        handle = open(target)
        handle.read(16)
        handle.close()
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["open:read"],
)

_LIST_OUTSIDE = _attempt(
    "listdir_outside_jail",
    "filesystem",
    "audit_hook",
    "Enumerate a directory outside the jail to find something worth reading.",
    f"""
def probe():
    import os

    parent = os.path.dirname(os.path.abspath(os.getcwd()))
{_TRY}        os.listdir(parent)
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["os.listdir"],
)

_ENV_SECRETS = _attempt(
    "env_secret_read",
    "secrets",
    "environment",
    (
        "Read credentials out of the environment. Nothing blocks os.environ; the control is the "
        "parent's allowlist in child_environment(), so the attempt finds nothing to report."
    ),
    """
def probe():
    import os

    markers = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")
    found = []
    for name, value in os.environ.items():
        upper = name.upper()
        if value and any(marker in upper for marker in markers):
            found.append(name)
    if found:
        return "escaped"
    return "clean"
""",
    statuses=["ok"],
    must_be_blocked=False,
)

_PROC_PROBE = _attempt(
    "proc_environ_probe",
    "secrets",
    "audit_hook",
    (
        "Read another process's environment through /proc. Blocked as a read outside the jail on "
        "Linux; on a host without /proc the open simply fails."
    ),
    f"""
def probe():
{_TRY}        handle = open("/proc/self/environ", "rb")
        data = handle.read()
        handle.close()
        if data:
            return "escaped"
        return "empty"
{_EXCEPT}""",
    statuses=["blocked", "ok"],
    must_be_blocked=False,
    hints=["open:read"],
)

_FORK_BOMB = _attempt(
    "fork_bomb",
    "resources",
    "platform",
    (
        "Fork until the process table fills. os.fork raises the os.fork audit event, so the guard "
        "refuses it where fork exists; on Windows there is no os.fork to call."
    ),
    f"""
def probe():
{_TRY}        import os

        for _ in range(64):
            os.fork()
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked", "ok"],
    must_be_blocked=False,
    hints=["os.fork"],
)

_THREAD_BOMB = _attempt(
    "thread_bomb",
    "resources",
    "audit_hook",
    "Spawn threads to outlive the watchdog. threading is on the import deny-list.",
    f"""
def probe():
{_TRY}        import threading

        for _ in range(64):
            threading.Thread(target=lambda: None).start()
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["import:threading", "import:_thread"],
)

_THREAD_VIA_SUBCLASSES = _attempt(
    "thread_bomb_via_subclasses",
    "resources",
    "platform",
    (
        "The same thread bomb, but around the import deny-list: purge_denied_modules() removes "
        "names from sys.modules and cannot remove the live Thread class, which is still reachable "
        "through object.__subclasses__() because the harness imported threading for its own "
        "watchdog. So the guard does NOT refuse this one - it is pinned here to show which control "
        "actually stops it: the wall-clock watchdog (and pids_limit in the container). If the "
        "watchdog regressed, this attempt would return the marker."
    ),
    """
def probe():
    import time

    try:
        thread = [c for c in object.__subclasses__() if c.__name__ == "Thread"][0]
        start = thread.__init__.__globals__["_start_new_thread"]
        for _ in range(8):
            start(time.sleep, (30,))
    except Exception:
        pass
    time.sleep(30)
    return "escaped"
""",
    statuses=["timeout"],
    must_be_blocked=False,
    timeout_seconds=0.5,
)

_RENAME_OUT = _attempt(
    "rename_out_of_jail",
    "filesystem",
    "audit_hook",
    (
        "Create a file inside the jail, then move it out. Regression: the guard used to check only "
        "the *first* path argument of an audit event, so the destination of a two-path event "
        "(os.rename, os.replace, os.link, os.symlink, shutil.move) was never checked at all."
    ),
    f"""
def probe():
    import os

    parent = os.path.dirname(os.path.abspath(os.getcwd()))
    source = os.path.join(os.getcwd(), "payload.txt")
    handle = open(source, "w")
    handle.write("x")
    handle.close()
{_TRY}        os.rename(source, os.path.join(parent, "escaped_by_rename.txt"))
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["os.rename"],
)

_HARDLINK_OUT = _attempt(
    "hardlink_out_of_jail",
    "filesystem",
    "audit_hook",
    (
        "Hard-link a file in the jail to a path outside it, which gives the same bytes a second "
        "name the guard never sees again. The destination is the second argument of os.link."
    ),
    f"""
def probe():
    import os

    parent = os.path.dirname(os.path.abspath(os.getcwd()))
    source = os.path.join(os.getcwd(), "linked.txt")
    handle = open(source, "w")
    handle.write("x")
    handle.close()
{_TRY}        os.link(source, os.path.join(parent, "escaped_by_link.txt"))
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["os.link"],
)

_OS_OPEN_WRITE = _attempt(
    "os_open_write_outside_jail",
    "filesystem",
    "audit_hook",
    (
        "Open a standard-library file for writing through os.open instead of open(). CPython "
        "raises the same `open` audit event with mode=None and the flags in args[2], so a guard "
        "that decides 'is this a write?' from the mode string alone falls through to the lenient "
        "*read* check - and the standard library is a read root. The target is a file that "
        "already exists inside that read root, and the attempt writes no bytes: holding a write "
        "descriptor on a file outside the jail is already the containment failure, and proving it "
        "that way keeps the suite safe to run on a developer's own interpreter."
    ),
    f"""
def probe():
    import os

{_TRY}        descriptor = os.open(os.__file__, os.O_WRONLY)
        os.close(descriptor)
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["open:write"],
)

_LONG_SLEEP = _attempt(
    "long_sleep",
    "resources",
    "platform",
    (
        "Hold the verifier open past its wall-clock limit. Nothing blocks time.sleep; the "
        "in-child watchdog reports a timeout, and the parent kills the process if that fails."
    ),
    """
def probe():
    import time

    time.sleep(30)
    return "escaped"
""",
    statuses=["timeout"],
    must_be_blocked=False,
    timeout_seconds=0.5,
)

_CHDIR_OUT = _attempt(
    "chdir_outside_jail",
    "filesystem",
    "audit_hook",
    "Move the working directory out of the jail so later relative paths escape it.",
    f"""
def probe():
    import os

    parent = os.path.dirname(os.path.abspath(os.getcwd()))
{_TRY}        os.chdir(parent)
        return "escaped"
{_EXCEPT}""",
    statuses=["blocked"],
    hints=["os.chdir"],
)

#: The full suite. Order is fixed so reports and CI output are stable.
ESCAPE_SUITE: Final[tuple[EscapeAttempt, ...]] = (
    _SOCKET_CONNECT,
    _DNS_LOOKUP,
    _HTTP_FETCH,
    _SUBPROCESS,
    _CTYPES,
    _WRITE_OUTSIDE,
    _OS_OPEN_WRITE,
    _READ_OUTSIDE,
    _LIST_OUTSIDE,
    _CHDIR_OUT,
    _RENAME_OUT,
    _HARDLINK_OUT,
    _ENV_SECRETS,
    _PROC_PROBE,
    _FORK_BOMB,
    _THREAD_BOMB,
    _THREAD_VIA_SUBCLASSES,
    _LONG_SLEEP,
)
_BY_NAME: Final[Mapping[str, EscapeAttempt]] = {item.name: item for item in ESCAPE_SUITE}


def attempt_by_name(name: str) -> EscapeAttempt:
    attempt = _BY_NAME.get(name)
    if attempt is None:
        raise KeyError(f"unknown escape attempt {name!r}")
    return attempt


def escape_job(attempt: EscapeAttempt) -> ExecJob:
    """The job for one attempt: its program, and one test that passes only if it escaped."""
    return ExecJob(
        program=attempt.program,
        entry_point=ESCAPE_ENTRY_POINT,
        visible=(TestCase(name="escaped", args=(), expected=SUCCESS_MARKER),),
        hidden=(),
        limits=attempt.limits,
    )


@dataclass(frozen=True, slots=True)
class EscapeOutcome:
    """What happened to one attempt."""

    attempt: EscapeAttempt
    verdict: Verdict

    @property
    def escaped(self) -> bool:
        """True only if the program returned the success marker: the escape actually worked."""
        return self.verdict.status == "ok" and self.verdict.visible.all_passed

    @property
    def blocked(self) -> bool:
        return not self.escaped

    @property
    def status_expected(self) -> bool:
        return self.verdict.status in self.attempt.expected_statuses

    @property
    def guard_refused(self) -> bool:
        return self.verdict.status == "blocked"

    def as_row(self) -> dict[str, object]:
        return {
            "attempt": self.attempt.name,
            "category": self.attempt.category,
            "mechanism": self.attempt.mechanism,
            "status": self.verdict.status,
            "blocked": self.blocked,
            "violations": list(self.verdict.violations),
        }


def evaluate_attempt(execute: ExecuteFn, attempt: EscapeAttempt) -> EscapeOutcome:
    """Run one attempt through ``execute`` (any :class:`SandboxClient`'s ``execute``)."""
    return EscapeOutcome(attempt=attempt, verdict=execute(escape_job(attempt)))


def blocked_fraction(outcomes: Sequence[EscapeOutcome]) -> float:
    """Share of attempts that did not escape. The suite claims 1.0."""
    if not outcomes:
        raise ValueError("no escape outcomes to summarise")
    return sum(1 for item in outcomes if item.blocked) / len(outcomes)
