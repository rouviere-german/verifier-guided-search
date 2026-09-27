"""The child harness, exercised in this process, where its guard can be inspected.

The harness normally runs in a subprocess, so an integration test can only see its verdicts. These
tests import it directly and check the pieces the verdict depends on: the deny-list is complete for
the categories it claims to cover, the jail check resolves paths rather than comparing strings, the
result writer writes once, and a candidate's result is normalised so a tuple compares equal to the
JSON the benchmark stores.

:func:`install_guard` is irreversible - an audit hook cannot be removed - so nothing here installs
it. The hook's *behaviour* is what the escape suite measures, in a subprocess where its
irreversibility
does not matter (``tests/integration/test_sandbox_escape.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from verifier_guided_search.sandbox.child import (
    BLOCKED_EVENTS,
    DENIED_MODULES,
    EXIT_OK,
    EXIT_USAGE,
    JAILED_PATH_EVENTS,
    MAX_NORMALIZE_DEPTH,
    ResultWriter,
    SandboxViolation,
    _forms,
    _normalize,
    _safe_class_name,
    _under,
    apply_limits,
    main,
    purge_denied_modules,
    read_roots,
    run_group,
)
from verifier_guided_search.sandbox.protocol import ERROR_CLASS_PATTERN

pytestmark = pytest.mark.unit


def test_a_violation_cannot_be_swallowed_by_except_exception() -> None:
    """The reason the guard derives from BaseException: untrusted code must not hide a refusal."""
    assert issubclass(SandboxViolation, BaseException)
    assert not issubclass(SandboxViolation, Exception)

    def guard() -> None:
        raise SandboxViolation("blocked")

    caught = False
    try:
        try:
            guard()
        except Exception:  # noqa: BLE001 - a candidate's broad handler is the point of the test
            caught = True
    except SandboxViolation:
        pass
    assert not caught, "a candidate's except Exception swallowed the violation"


def test_the_deny_list_covers_every_way_out_the_suite_claims_to_block() -> None:
    for module in ("socket", "subprocess", "ctypes", "urllib", "threading", "multiprocessing"):
        assert module in DENIED_MODULES, module
    for event in ("os.fork", "os.system", "subprocess.Popen", "socket.connect", "ctypes.dlopen"):
        assert event in BLOCKED_EVENTS, event
    for event in ("os.chdir", "os.listdir", "os.remove", "os.rename"):
        assert event in JAILED_PATH_EVENTS, event


def test_the_read_roots_are_the_standard_library_and_nothing_else() -> None:
    roots = read_roots()
    assert roots
    assert all(os.path.isabs(root) for root in roots)
    assert not any(root in {os.sep, ""} for root in roots), (
        "the whole filesystem is not a read root"
    )
    assert not _under("/etc/passwd", roots)
    assert not _under(str(Path(__file__)), roots), "the test tree is not a read root"


def test_a_standard_library_module_is_readable_whichever_way_the_path_resolves() -> None:
    r"""Regression: a stdlib import must not be refused because of how its path resolves.

    On a host where the interpreter lives under a junction (a Windows packaged application puts
    ``AppData\Roaming`` behind a reparse point), resolving a stdlib *file* rewrites its prefix while
    resolving the *directory* above it does not. Comparing only resolved paths refused every
    not-yet-imported ``import``, turning ordinary candidates into ``blocked`` verdicts and silently
    corrupting every solve rate measured on that host.
    """
    roots = read_roots()
    stdlib = os.path.dirname(os.__file__)
    for module in ("json/__init__.py", "textwrap.py", "statistics.py"):
        candidate = os.path.join(stdlib, *module.split("/"))
        if not os.path.exists(candidate):  # pragma: no cover - depends on the build
            continue
        assert _under(candidate, roots), f"{candidate} is not readable under {roots}"
        assert _under(os.path.realpath(candidate), roots)


def test_both_path_forms_are_compared() -> None:
    forms = _forms(os.path.dirname(os.__file__))
    assert forms
    assert all(os.path.isabs(form) for form in forms)
    assert all(form == os.path.normcase(form) for form in forms)


def test_a_strict_check_refuses_a_link_that_leaves_the_root(tmp_path: Path) -> None:
    """Writes use the strict comparison, so a symlink out of the jail is not a way out."""
    jail = tmp_path / "jail"
    jail.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = jail / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - needs privileges on Windows
        pytest.skip("this host does not allow creating symlinks")
    roots = (os.path.normcase(os.path.realpath(str(jail))),)
    assert _under(str(link), roots) is True, "the lenient form sees it inside the jail"
    assert _under(str(link), roots, strict=True) is False, "the strict form must refuse it"


def test_the_jail_check_resolves_paths_instead_of_comparing_strings(tmp_path: Path) -> None:
    jail = str((tmp_path / "jail").resolve())
    os.makedirs(jail, exist_ok=True)
    assert _under(os.path.join(jail, "file.txt"), [jail])
    assert _under(os.path.join(jail, "sub", "file.txt"), [jail])
    # A textual prefix that is a sibling directory, not a child.
    assert not _under(jail + "-sneaky", [jail])
    assert not _under(os.path.join(jail, "..", "escape.txt"), [jail])
    assert not _under(str(tmp_path), [jail])


def test_an_already_open_descriptor_is_allowed_because_its_open_was_checked() -> None:
    assert _under(3, ["/jail"]) is True
    assert _under(None, ["/jail"]) is True


def test_a_path_of_an_unusable_type_is_refused() -> None:
    assert _under(object(), ["/jail"]) is False
    assert _under(b"\xff\xfe/nope", ["/jail"]) is False


def test_purging_denied_modules_reports_what_it_dropped() -> None:
    """The harness needs ``threading`` for its watchdog, so denied names leave ``sys.modules``.

    Dropping them is what makes a later ``import threading`` raise the ``import`` audit event, since
    CPython raises it only for a module that is not already imported.

    The exact module objects are put back afterwards. Re-importing instead would give *new* module
    objects for ``threading``, ``asyncio`` and friends while the rest of this process still holds
    the old ones, which breaks pytest itself - so the restore matters as much as the purge.
    """
    import threading  # noqa: F401 - imported so it is in sys.modules

    saved = {
        name: module for name, module in sys.modules.items() if name.split(".")[0] in DENIED_MODULES
    }
    assert "threading" in saved, "the fixture needs threading to be imported"
    try:
        dropped = purge_denied_modules()
        assert "threading" in dropped
        assert "threading" not in sys.modules
        assert set(dropped) == set(saved), "purge and restore must cover the same names"
        assert not any(name.split(".")[0] in DENIED_MODULES for name in sys.modules)
    finally:
        sys.modules.update(saved)
    assert sys.modules["threading"] is saved["threading"]


def test_apply_limits_reports_what_the_platform_could_not_do() -> None:
    notes = apply_limits(
        {
            "timeout_seconds": 1.5,
            "memory_bytes": 256 * 1024 * 1024,
            "max_processes": 1,
            "max_file_bytes": 1024 * 1024,
        }
    )
    assert isinstance(notes, tuple)
    if sys.platform == "win32":
        assert notes == ("rlimits_unavailable",), "Windows has no setrlimit; it must say so"
    else:  # pragma: no cover - only one branch runs per platform
        assert all(note.startswith("rlimit_refused:") for note in notes)


def test_tuples_are_normalised_to_lists_so_a_result_compares_with_json() -> None:
    assert _normalize((1, 2, 3)) == [1, 2, 3]
    assert _normalize([(1, 2), (3,)]) == [[1, 2], [3]]
    assert _normalize("text") == "text"
    assert _normalize(None) is None
    assert _normalize(1.5) == 1.5


def test_a_deeply_nested_result_is_truncated_rather_than_recursing_forever() -> None:
    nested: object = 1
    for _ in range(MAX_NORMALIZE_DEPTH + 5):
        nested = [nested]
    assert "<too-deep>" in repr(_normalize(nested))


def test_run_group_classifies_pass_fail_and_error() -> None:
    def function(value: int) -> int:
        if value < 0:
            raise ValueError("negative")
        return value * 2

    rows = run_group(
        function,
        [
            {"name": "t0", "args": [2], "expected": 4},
            {"name": "t1", "args": [3], "expected": 99},
            {"name": "t2", "args": [-1], "expected": 0},
        ],
    )
    assert [row["status"] for row in rows] == ["pass", "fail", "error"]
    assert rows[0]["error_class"] is None
    assert rows[2]["error_class"] == "ValueError"


def test_run_group_lets_a_violation_through_instead_of_recording_it_as_an_error() -> None:
    def escaping(value: int) -> int:
        raise SandboxViolation("open:write")

    with pytest.raises(SandboxViolation):
        run_group(escaping, [{"name": "t0", "args": [1], "expected": 1}])


def test_run_group_handles_a_case_with_no_arguments() -> None:
    rows = run_group(lambda: 7, [{"name": "t0", "expected": 7}])
    assert rows[0]["status"] == "pass"


def test_an_exception_class_name_is_always_a_safe_identifier() -> None:
    import re

    class OddError(Exception):
        pass

    OddError.__name__ = "1bad-name!"
    assert re.fullmatch(ERROR_CLASS_PATTERN, _safe_class_name(OddError()))

    class EmptyError(Exception):
        pass

    EmptyError.__name__ = "!!!"
    assert _safe_class_name(EmptyError()) == "Exception"

    class LongError(Exception):
        pass

    LongError.__name__ = "A" * 200
    assert len(_safe_class_name(LongError())) <= 64


def test_the_result_writer_writes_exactly_once(tmp_path: Path) -> None:
    """A watchdog firing while the main path is writing must not produce two results."""
    path = tmp_path / "result.json"
    with path.open("w", encoding="utf-8") as handle:
        writer = ResultWriter(handle, "the-nonce", "a" * 64)
        assert writer.write({"status": "ok", "visible": [], "hidden": []}) is True
        assert writer.write({"status": "timeout", "visible": [], "hidden": []}) is False
    payload = path.read_text(encoding="utf-8")
    assert '"status":"ok"' in payload.replace(" ", "")
    assert "timeout" not in payload


def test_the_result_writer_stamps_the_nonce_and_the_program_hash(tmp_path: Path) -> None:
    import json

    path = tmp_path / "result.json"
    with path.open("w", encoding="utf-8") as handle:
        ResultWriter(handle, "the-nonce", "b" * 64).write({"status": "ok"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nonce"] == "the-nonce"
    assert payload["program_sha256"] == "b" * 64
    assert payload["violations"] == []


def test_the_harness_reports_a_usage_error_for_the_wrong_argument_count() -> None:
    assert main([]) == EXIT_USAGE
    assert main(["only-one"]) == EXIT_USAGE
    assert main(["a", "b", "c"]) == EXIT_USAGE


def test_the_harness_writes_a_syntax_error_verdict_without_installing_the_guard(
    tmp_path: Path,
) -> None:
    """A syntax error is a verdict. Checked here because it happens *before* the guard goes up."""
    import json

    job = tmp_path / "job.json"
    result = tmp_path / "result.json"
    jail = tmp_path / "jail"
    jail.mkdir()
    job.write_text(
        json.dumps(
            {
                "nonce": "n",
                "program": "def solve(:\n    pass\n",
                "program_sha256": "c" * 64,
                "entry_point": "solve",
                "visible": [],
                "hidden": [],
                "jail": str(jail),
                "limits": {
                    "timeout_seconds": 5.0,
                    "memory_bytes": 256 * 1024 * 1024,
                    "max_processes": 1,
                    "max_file_bytes": 1024 * 1024,
                    "max_output_bytes": 4096,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main([str(job), str(result)]) == EXIT_OK
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "syntax_error"
    assert payload["nonce"] == "n"
    assert payload["program_sha256"] == "c" * 64


def test_an_early_verdict_cancels_the_watchdog(tmp_path: Path) -> None:
    """Regression: a verdict written early must disarm the timer, or it later calls ``os._exit``.

    The syntax-error, blocked, load-error and missing-entry-point paths all used to return without
    cancelling the watchdog. In a subprocess that is nearly harmless because the process exits
    anyway, but it is a latent way to turn a reported verdict into an exit-code-3 kill - and it
    killed the
    *test* process outright once ``main`` was exercised in-process.

    Only the syntax-error path can be checked here: every later path installs the audit hook first,
    and an audit hook cannot be removed. The others are covered by
    ``tests/integration/test_sandbox_local.py::test_an_early_verdict_exits_cleanly``.
    """
    import json
    import threading

    before = threading.active_count()
    job = tmp_path / "job.json"
    result = tmp_path / "result.json"
    jail = tmp_path / "jail"
    jail.mkdir()
    job.write_text(
        json.dumps(
            {
                "nonce": "n",
                "program": "def solve(:\n    pass\n",
                "program_sha256": "d" * 64,
                "entry_point": "solve",
                "visible": [{"name": "v00", "args": [], "expected": 1}],
                "hidden": [],
                "jail": str(jail),
                # Long enough that an armed timer would outlive the whole test session.
                "limits": {
                    "timeout_seconds": 30.0,
                    "memory_bytes": 256 * 1024 * 1024,
                    "max_processes": 1,
                    "max_file_bytes": 1024 * 1024,
                    "max_output_bytes": 4096,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main([str(job), str(result)]) == EXIT_OK
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "syntax_error"
    # A live timer keeps a thread alive and fires os._exit(3) thirty seconds later.
    assert threading.active_count() <= before, "the watchdog was left armed"


def test_the_harness_is_standard_library_only(repo_root: Path) -> None:
    """It is copied into a scratch directory and run with ``-I -S``, so it cannot import this
    package."""
    import ast

    source = (repo_root / "src" / "verifier_guided_search" / "sandbox" / "child.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    allowed = {
        "json",
        "os",
        "sys",
        "threading",
        "time",
        "collections",
        "typing",
        "resource",
        "__future__",
    }
    assert imported <= allowed, f"the harness imports {imported - allowed}"
