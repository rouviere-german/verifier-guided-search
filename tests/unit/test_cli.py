"""The command line: every subcommand reachable, and the cheap ones genuinely cheap.

The import-cost test is the one with teeth. ``--help`` and ``tasks`` must not drag in uvicorn,
scikit-learn or the Anthropic SDK, because a CLI that takes a second to print its own usage is a CLI
nobody runs, and a module that imports the SDK at the top would break every offline code path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verifier_guided_search.cli import main

pytestmark = pytest.mark.unit


def test_no_arguments_is_an_error_not_a_crash() -> None:
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "serve",
        "serve-runner",
        "generate-api-key",
        "audit",
        "tasks",
        "escape-suite",
        "search",
        "demo",
    ):
        assert command in out


def test_tasks_lists_the_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tasks", "--instances-per-family", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tasks"] == 15
    assert len(payload["families"]) == 15
    assert len(payload["suite_sha256"]) == 64
    assert len(payload["task_ids"]) == 15


def test_tasks_prints_one_task_without_its_reference_solution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A task's reference is the answer; the CLI shows the statement and the broken program."""
    assert main(["tasks", "run_length_encode/00", "--instances-per-family", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_id"] == "run_length_encode/00"
    assert payload["entry_point"] == "run_length_encode"
    assert "start_program" in payload
    assert "reference_solution" not in payload
    assert payload["visible_tests"] >= 3
    assert payload["hidden_tests"] >= 3


def test_an_unknown_task_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tasks", "nope/00", "--instances-per-family", "1"]) == 2
    assert "unknown task" in capsys.readouterr().err


def test_generate_api_key_prints_a_key_and_only_stores_its_digest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from verifier_guided_search.core.security import hash_api_key

    assert main(["generate-api-key"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert hash_api_key(payload["api_key"]) == payload["api_key_sha256"]
    assert "Store only api_key_sha256" in captured.err


def test_audit_verify_reports_a_good_and_a_tampered_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from verifier_guided_search.agentic.audit import AuditLog

    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("cli-test", "example", {"index": 1})
    assert main(["audit", "verify", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["entries"] == 1

    assert main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False

    path.write_text(
        path.read_text(encoding="utf-8").replace("example", "exampleX"), encoding="utf-8"
    )
    assert main(["audit", "verify", str(path)]) == 1


@pytest.mark.sandbox
def test_escape_suite_runs_and_reports_zero_escapes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["escape-suite", "--workdir", str(tmp_path / "escape")]) == 0
    out = capsys.readouterr().out
    assert "blocked: 100.0%" in out
    assert "| attempt | category |" in out
    assert " NO |" not in out


def test_the_cheap_subcommands_do_not_import_the_heavy_dependencies() -> None:
    """``tasks`` must not pull in uvicorn, scikit-learn or the Anthropic SDK."""
    code = (
        "import sys\n"
        "from verifier_guided_search.cli import main\n"
        "main(['tasks', '--instances-per-family', '1'])\n"
        "heavy = [n for n in ('uvicorn', 'sklearn', 'anthropic', 'fastapi') if n in sys.modules]\n"
        "sys.stderr.write(','.join(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=180
    )
    assert result.stderr.strip() == "", f"these were imported: {result.stderr}"


def test_the_module_entry_point_is_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "verifier_guided_search", "tasks", "--instances-per-family", "1"],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    assert json.loads(result.stdout)["tasks"] == 15


def test_the_console_scripts_point_at_the_cli(repo_root: Path) -> None:
    import tomllib

    with (repo_root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    scripts = data["project"]["scripts"]
    assert scripts["verifier-guided-search"] == "verifier_guided_search.cli:main"
    assert scripts["vgs"] == "verifier_guided_search.cli:main"


@pytest.mark.sandbox
def test_search_runs_one_budgeted_search_and_prints_what_it_committed_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``vgs search`` is the entry point the live generator was missing. Offline: mutation."""
    assert (
        main(
            [
                "search",
                "run_length_encode/00",
                "--strategy",
                "beam",
                "--candidates",
                "4",
                "--workdir",
                str(tmp_path / "search"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_id"] == "run_length_encode/00"
    assert payload["generator"] == "mutation"
    assert payload["model"] == "fake-mutation"
    assert payload["evaluations"] <= 4
    assert payload["usage"]["candidates"] == payload["evaluations"]
    assert set(payload) >= {"solved_visible", "solved_hidden", "false_pass", "stop_reason"}


def test_search_refuses_an_unknown_task_or_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "nope/00"]) == 2
    assert "unknown task" in capsys.readouterr().err
    assert main(["search", "run_length_encode/00", "--strategy", "telepathy"]) == 2
    assert "unknown strategy" in capsys.readouterr().err


def test_search_refuses_the_live_generator_without_a_key(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing key is a clear refusal before any work, not a failure mid-search."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["search", "run_length_encode/00", "--generator", "claude"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
