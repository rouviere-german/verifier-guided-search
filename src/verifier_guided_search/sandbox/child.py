"""The child harness: runs one untrusted program against one task's tests, then exits.

This file is copied verbatim into a scratch directory and executed by a bare interpreter as
``python -I -S -B harness.py <job> <result>``. ``-I`` drops the environment and the script
directory from ``sys.path``; ``-S`` skips ``site``, so site-packages is not importable at all and a
candidate program is limited to the standard library. Nothing here may import this package: it has
to stand alone, and it stays standard-library only.

Order of operations, which is the whole point of the file:

1. read the job, and open the result file **before** any guard is installed, keeping the handle.
   The result path lives outside the jail, so once the guard is up no new handle to it can be
   obtained;
2. start a watchdog timer and apply POSIX rlimits (CPU, address space, processes, file size), in
   that order, so a program that never returns is reported as ``timeout`` with a real result rather
   than only killed by the parent;
3. compile the program - a syntax error is a verdict, not a crash;
4. install the audit hook (see :func:`install_guard`), which is the point of no return: audit hooks
   cannot be removed;
5. execute the program, run the visible tests and then the hidden ones, and write the result
   through the handle from step 1.

The guard is defence in depth, not the security boundary. It blocks the obvious escapes - sockets,
name lookups, subprocesses, ``ctypes``, writes or reads outside the jail, a deny-list of modules -
and it raises :class:`SandboxViolation`, which derives from ``BaseException`` so a candidate's
``except Exception`` cannot swallow it; every violation is also recorded before the raise, so a
swallowed one still shows up in the verdict. But the harness and the candidate share one
interpreter, so a determined program can still interfere with its own bookkeeping. The container
(read-only rootfs, non-root, ``cap_drop: ALL``, no egress, pids and memory limits) is the boundary;
this file makes the common cases cheap to block and easy to test. See docs/THREAT_MODEL.md.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

MAX_DETAIL = 400
MAX_NORMALIZE_DEPTH = 12
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_TIMEOUT = 3

#: Modules a candidate program has no business importing. Enforced on the ``import`` audit event,
#: which CPython raises only for a module that is *not* already in ``sys.modules``. The harness
#: itself needs ``threading`` for its watchdog, so :func:`purge_denied_modules` drops every denied
#: name from ``sys.modules`` just before the guard goes up; the already-running timer keeps working
#: because it holds its own references, and a later import of the same name fires the event.
DENIED_MODULES = frozenset(
    {
        "_multiprocessing",
        "_socket",
        "_ssl",
        "_thread",
        "asyncio",
        "ctypes",
        "ftplib",
        "http",
        "imaplib",
        "multiprocessing",
        "pty",
        "shutil",
        "signal",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "subprocess",
        "threading",
        "urllib",
        "webbrowser",
        "xmlrpc",
    }
)
#: Audit events that are refused outright, whatever their arguments.
BLOCKED_EVENTS = frozenset(
    {
        "builtins.input",
        "builtins.input/result",
        "ctypes.addressof",
        "ctypes.call_function",
        "ctypes.dlopen",
        "ctypes.dlsym",
        "ctypes.dlsym/handle",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.putenv",
        "os.spawn",
        "os.startfile",
        "os.system",
        "os.unsetenv",
        "pty.spawn",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.sendto",
        "subprocess.Popen",
        "webbrowser.open",
    }
)
#: Audit events whose first argument is a path that must stay inside the jail.
JAILED_PATH_EVENTS = frozenset(
    {
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.link",
        "os.listdir",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.scandir",
        "os.symlink",
        "os.truncate",
        "os.utime",
        "shutil.copyfile",
        "shutil.copymode",
        "shutil.copystat",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
        "shutil.unpack_archive",
    }
)
_WRITE_MODE_CHARS = "wax+"

#: Populated by the guard before it raises, so a swallowed violation still reaches the verdict.
VIOLATIONS: list[str] = []


class SandboxViolation(BaseException):
    """A guard refusal. Derived from BaseException so ``except Exception`` cannot hide it."""


def _record(reason: str) -> SandboxViolation:
    if reason not in VIOLATIONS:
        VIOLATIONS.append(reason)
    return SandboxViolation(reason)


def _forms(path: str) -> tuple[str, ...]:
    """The lexical and the link-resolved form of ``path``, normalised for case.

    Both are needed because the two can disagree on a prefix. On Windows, resolving a *file* under
    a reparse point (a junction such as ``AppData\\Roaming`` under a packaged application)
    rewrites the prefix, while resolving the *directory* above it may not - so a standard-library
    file can resolve to a path that does not start with the standard-library root. Comparing only
    resolved
    forms then refuses a legitimate ``import``, which turns an ordinary candidate into a ``blocked``
    verdict and silently corrupts every solve rate measured on that host.
    """
    lexical = os.path.normcase(os.path.abspath(path))
    try:
        resolved = os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):  # pragma: no cover - realpath is lenient
        resolved = lexical
    return (lexical,) if lexical == resolved else (lexical, resolved)


def read_roots() -> tuple[str, ...]:
    """Directories a candidate may read from: the standard library and its extension modules.

    Each anchor contributes both of its :func:`_forms`, *and* the directory of a resolved file
    inside it. The last part matters: under a reparse point the platform can resolve a file to a
    prefix that resolving the directory never produces, so anchoring on ``os.py`` as well as on its
    parent is what makes both forms of every standard-library path recognisable.
    """
    anchors = {sys.prefix, sys.base_prefix, os.path.dirname(os.__file__)}
    roots: set[str] = set()
    for anchor in anchors:
        if anchor:
            roots.update(_forms(anchor))
    module = getattr(os, "__file__", None)
    if module:
        roots.update(_forms(os.path.dirname(os.path.realpath(module))))
    return tuple(sorted(roots))


def _inside(candidate: str, roots: Sequence[str]) -> bool:
    """Is ``candidate`` (already normalised) inside any of ``roots``? Roots are normalised here.

    Normalising the roots in one place means a caller cannot break the check by passing a path that
    merely differs in case or separators - on Windows that would otherwise refuse everything.
    """
    for root in roots:
        normalised = os.path.normcase(root)
        if candidate == normalised or candidate.startswith(normalised + os.sep):
            return True
    return False


def _under(path: object, roots: Sequence[str], *, strict: bool = False) -> bool:
    """Is ``path`` inside one of ``roots``?

    ``strict`` compares only the fully resolved form, so a symlink pointing out of the root is
    refused. It is used for the writable jail, where a false negative is a containment failure.
    Reads use the lenient comparison, where a false positive would only ever grant access to a
    standard library file the candidate may already read, and a false negative breaks ``import``.
    """
    if path is None or isinstance(path, int):
        return True  # an already-open descriptor; the open that created it was checked
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    try:
        decoded = os.fsdecode(path)
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    try:
        forms = _forms(decoded)
    except (OSError, TypeError, ValueError):
        return False
    if strict:
        return _inside(forms[-1], roots)
    return any(_inside(form, roots) for form in forms)


def purge_denied_modules() -> tuple[str, ...]:
    """Drop denied modules from ``sys.modules`` so re-importing one raises the audit event."""
    dropped: list[str] = []
    for name in sorted(sys.modules):
        if name.split(".")[0] in DENIED_MODULES:
            del sys.modules[name]
            dropped.append(name)
    return tuple(dropped)


def install_guard(jail: str, allowed_read_roots: Sequence[str]) -> None:
    """Install the audit hook. Irreversible; call it last, immediately before executing."""
    purge_denied_modules()
    # Writes are checked against the fully resolved jail only, so a link out of it is refused.
    write_roots: tuple[str, ...] = (os.path.normcase(os.path.realpath(jail)),)
    read_only_roots: tuple[str, ...] = (*_forms(jail), *allowed_read_roots)

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event in BLOCKED_EVENTS:
            raise _record(event)
        if event == "open":
            path, mode = args[0], args[1]
            writing = isinstance(mode, str) and any(char in mode for char in _WRITE_MODE_CHARS)
            if writing:
                if not _under(path, write_roots, strict=True):
                    raise _record("open:write")
                return
            if not _under(path, read_only_roots):
                raise _record("open:read")
            return
        if event in JAILED_PATH_EVENTS:
            if not _under(args[0] if args else None, write_roots, strict=True):
                raise _record(event)
            return
        if event == "import":
            root = str(args[0]).split(".")[0]
            if root in DENIED_MODULES:
                raise _record(f"import:{root}")

    sys.addaudithook(hook)


if sys.platform == "win32":

    def apply_limits(limits: dict[str, Any]) -> tuple[str, ...]:
        """Windows has no ``setrlimit``; the watchdog and the container carry these limits."""
        del limits
        return ("rlimits_unavailable",)

else:
    import resource

    def apply_limits(limits: dict[str, Any]) -> tuple[str, ...]:
        """CPU seconds, address space, process count and file size, via POSIX rlimits."""
        cpu = max(1, int(float(limits["timeout_seconds"]) + 1))
        memory = int(limits["memory_bytes"])
        processes = int(limits["max_processes"])
        file_bytes = int(limits["max_file_bytes"])
        pairs = (
            ("cpu", resource.RLIMIT_CPU, (cpu, cpu + 1)),
            ("as", resource.RLIMIT_AS, (memory, memory)),
            ("nproc", resource.RLIMIT_NPROC, (processes, processes)),
            ("fsize", resource.RLIMIT_FSIZE, (file_bytes, file_bytes)),
        )
        notes: list[str] = []
        for name, which, values in pairs:
            try:
                resource.setrlimit(which, values)
            except (ValueError, OSError):  # pragma: no cover - depends on the host's own limits
                notes.append(f"rlimit_refused:{name}")
        return tuple(notes)


def _normalize(value: object, depth: int = 0) -> object:
    """Tuples become lists so a candidate result compares equal to JSON expectations."""
    if depth > MAX_NORMALIZE_DEPTH:
        return "<too-deep>"
    if isinstance(value, (list, tuple)):
        return [_normalize(item, depth + 1) for item in value]
    return value


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_DETAIL]


def _safe_class_name(exc: BaseException) -> str:
    name = "".join(char for char in type(exc).__name__ if char.isalnum() or char == "_")[:64]
    return name if name and not name[0].isdigit() else "Exception"


def run_group(
    function: Callable[..., object], cases: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run one test group: one entry per case - pass, fail, or error with the exception class."""
    results: list[dict[str, Any]] = []
    for case in cases:
        name = str(case["name"])
        args = list(case.get("args") or [])
        try:
            actual = _normalize(function(*args))
        except SandboxViolation:
            raise
        except BaseException as exc:  # noqa: BLE001 - any failure of untrusted code is a data point
            results.append({"name": name, "status": "error", "error_class": _safe_class_name(exc)})
            continue
        passed = actual == _normalize(case.get("expected"))
        results.append({"name": name, "status": "pass" if passed else "fail", "error_class": None})
    return results


class ResultWriter:
    """Writes the result exactly once, through a handle opened before the guard went up."""

    def __init__(self, handle: Any, nonce: str, program_sha256: str) -> None:
        self._handle = handle
        self._nonce = nonce
        self._program_sha256 = program_sha256
        self._lock = threading.Lock()
        self._written = False

    def write(self, payload: dict[str, Any]) -> bool:
        with self._lock:
            if self._written:
                return False
            self._written = True
            body = dict(payload)
            body["nonce"] = self._nonce
            body["program_sha256"] = self._program_sha256
            body["violations"] = list(VIOLATIONS)
            self._handle.write(json.dumps(body, sort_keys=True))
            self._handle.flush()
            return True


def _finish(
    writer: ResultWriter,
    handle: Any,
    payload: dict[str, Any],
    watchdog: threading.Timer | None = None,
) -> int:
    """Write the verdict, stop the watchdog, close the handle.

    Cancelling the watchdog here rather than at one happy-path call site is what makes every
    terminal path safe: a syntax error, a blocked run or a missing entry point all finish through
    this function, and a timer left armed would later call ``os._exit`` on a process that had
    already reported its verdict.
    """
    payload.setdefault("visible", [])
    payload.setdefault("hidden", [])
    writer.write(payload)
    if watchdog is not None:
        watchdog.cancel()
    handle.close()
    return EXIT_OK


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: harness.py <job.json> <result.json>\n")
        return EXIT_USAGE
    job_path, result_path = argv[0], argv[1]
    with open(job_path, encoding="utf-8") as handle:
        job: dict[str, Any] = json.load(handle)
    limits: dict[str, Any] = job["limits"]
    # Opened before the guard: afterwards this path is outside every writable root.
    result_handle = open(result_path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
    writer = ResultWriter(result_handle, str(job["nonce"]), str(job["program_sha256"]))

    def on_timeout() -> None:
        if writer.write(
            {"status": "timeout", "visible": [], "hidden": [], "detail": "watchdog expired"}
        ):
            result_handle.close()
        os._exit(EXIT_TIMEOUT)

    # The watchdog thread starts before RLIMIT_NPROC is lowered, which on Linux also caps threads.
    watchdog = threading.Timer(float(limits["timeout_seconds"]), on_timeout)
    watchdog.daemon = True
    watchdog.start()

    def finish(payload: dict[str, Any]) -> int:
        return _finish(writer, result_handle, payload, watchdog)

    notes = list(apply_limits(limits))

    try:
        code = compile(str(job["program"]), "<candidate>", "exec")
    except (SyntaxError, ValueError) as exc:
        return finish({"status": "syntax_error", "detail": _detail(exc)})

    started = time.perf_counter()
    install_guard(str(job["jail"]), read_roots())
    namespace: dict[str, Any] = {"__name__": "candidate"}
    try:
        # Executing the candidate under the rlimits and audit guard installed above is
        # this harness's whole purpose. The container is the real boundary; see
        # docs/THREAT_MODEL.md.
        exec(code, namespace)  # nosec B102
    except SandboxViolation as exc:
        return finish({"status": "blocked", "detail": _detail(exc)})
    except BaseException as exc:  # noqa: BLE001 - a module-level failure is a verdict, not a crash
        return finish({"status": "load_error", "detail": _detail(exc)})

    function = namespace.get(str(job["entry_point"]))
    if not callable(function):
        return finish({"status": "missing_entry_point", "detail": "not defined"})

    try:
        visible = run_group(function, job.get("visible") or [])
        hidden = run_group(function, job.get("hidden") or [])
    except SandboxViolation as exc:
        return finish({"status": "blocked", "detail": _detail(exc)})

    return finish(
        {
            "status": "blocked" if VIOLATIONS else "ok",
            "visible": visible,
            "hidden": hidden,
            "detail": " ".join(notes),
            "child_seconds": round(time.perf_counter() - started, 6),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
