"""The CI escape check: an unreachable runner must never be reported as "nothing escaped".

This module is what `.github/workflows/ci.yml#sandbox-escape` runs against the containerised runner.
Docker is not started here; the runner is a mock transport. What is tested is the exit-code
contract, because that is the whole interface CI has to it - and because the dangerous failure
mode is a green job caused by a runner that never answered.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from verifier_guided_search.api.escape_check import (
    EXIT_ESCAPED,
    EXIT_OK,
    EXIT_UNREACHABLE,
    main,
    report,
    run_against,
)
from verifier_guided_search.core.config import Settings
from verifier_guided_search.core.security import hash_api_key
from verifier_guided_search.sandbox.client import HttpSandbox
from verifier_guided_search.sandbox.escape import (
    ESCAPE_SUITE,
    SUCCESS_MARKER,
    EscapeOutcome,
    attempt_by_name,
    escape_job,
)
from verifier_guided_search.sandbox.protocol import (
    ExecJob,
    GroupResult,
    TestResult,
    Verdict,
    VerdictStatus,
)

pytestmark = pytest.mark.unit

_KEY = "escape-check-key-" + "z" * 24


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "runner_url": "http://runner:8001",
        "runner_api_key": _KEY,
        "api_key_sha256": [hash_api_key(_KEY)],
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def verdict(job: ExecJob, status: VerdictStatus, *, passed: bool) -> Verdict:
    return Verdict(
        program_sha256=job.program_sha256,
        suite_sha256=job.suite_sha256,
        status=status,
        visible=GroupResult(
            results=(TestResult(name="escaped", status="pass" if passed else "fail"),)
        ),
        violations=("import:socket",) if status == "blocked" else (),
    )


def blocking_runner(request: httpx.Request) -> httpx.Response:
    """A runner that blocks everything, answering each attempt with a status the suite expects."""
    if request.url.path == "/healthz":
        return httpx.Response(200, json={"status": "ok"})
    import json

    job = ExecJob.model_validate(json.loads(request.content))
    attempt = next(item for item in ESCAPE_SUITE if item.program == job.program)
    status: VerdictStatus = sorted(attempt.expected_statuses)[0]  # type: ignore[assignment]
    return httpx.Response(200, json=verdict(job, status, passed=False).model_dump(mode="json"))


def client_for(handler: Any) -> HttpSandbox:
    return HttpSandbox(
        "http://runner:8001",
        api_key=_KEY,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_a_fully_blocking_runner_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "verifier_guided_search.api.escape_check.HttpSandbox",
        lambda *args, **kwargs: client_for(blocking_runner),
    )
    assert main(settings()) == EXIT_OK
    out = capsys.readouterr().out
    assert f"all {len(ESCAPE_SUITE)} attempts blocked" in out
    assert "| attempt | category |" in out


def test_one_escape_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure the job exists to catch."""
    leaky_name = "socket_connect"

    def leaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        import json

        job = ExecJob.model_validate(json.loads(request.content))
        escaped = job.program == attempt_by_name(leaky_name).program
        status: VerdictStatus = "ok" if escaped else "blocked"
        return httpx.Response(
            200, json=verdict(job, status, passed=escaped).model_dump(mode="json")
        )

    monkeypatch.setattr(
        "verifier_guided_search.api.escape_check.HttpSandbox",
        lambda *args, **kwargs: client_for(leaky),
    )
    assert main(settings()) == EXIT_ESCAPED
    captured = capsys.readouterr()
    assert f"ESCAPED: {leaky_name}" in captured.err
    assert " NO |" in captured.out


def test_an_unexpected_status_exits_one_even_though_nothing_escaped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A guard that stopped enforcing would leave attempts crashing for unrelated reasons."""

    def crashing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        import json

        job = ExecJob.model_validate(json.loads(request.content))
        return httpx.Response(
            200, json=verdict(job, "syntax_error", passed=False).model_dump(mode="json")
        )

    monkeypatch.setattr(
        "verifier_guided_search.api.escape_check.HttpSandbox",
        lambda *args, **kwargs: client_for(crashing),
    )
    assert main(settings()) == EXIT_ESCAPED
    assert "UNEXPECTED:" in capsys.readouterr().err


def test_an_unreachable_runner_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dangerous failure mode: a green job because the runner never answered."""

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(
        "verifier_guided_search.api.escape_check.HttpSandbox",
        lambda *args, **kwargs: client_for(dead),
    )
    assert main(settings()) == EXIT_UNREACHABLE
    assert "not reachable" in capsys.readouterr().err


def test_a_runner_that_dies_mid_suite_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = {"calls": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        state["calls"] += 1
        if state["calls"] > 2:
            return httpx.Response(503, json={})
        import json

        job = ExecJob.model_validate(json.loads(request.content))
        blocked = verdict(job, "blocked", passed=False)
        return httpx.Response(200, json=blocked.model_dump(mode="json"))

    monkeypatch.setattr(
        "verifier_guided_search.api.escape_check.HttpSandbox",
        lambda *args, **kwargs: client_for(flaky),
    )
    assert main(settings()) == EXIT_UNREACHABLE
    assert "unavailable mid-suite" in capsys.readouterr().err


def test_no_runner_url_is_an_infrastructure_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(Settings(_env_file=None)) == EXIT_UNREACHABLE
    assert "APP_RUNNER_URL is not set" in capsys.readouterr().err


def test_the_whole_suite_is_sent_in_its_fixed_order(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def recording(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        import json

        job = ExecJob.model_validate(json.loads(request.content))
        seen.append(next(a.name for a in ESCAPE_SUITE if a.program == job.program))
        blocked = verdict(job, "blocked", passed=False)
        return httpx.Response(200, json=blocked.model_dump(mode="json"))

    outcomes = run_against(client_for(recording))
    assert seen == [attempt.name for attempt in ESCAPE_SUITE]
    assert len(outcomes) == len(ESCAPE_SUITE)


def test_the_marker_is_what_decides_an_escape() -> None:
    """A passing test means the escape worked; nothing else counts."""
    attempt = attempt_by_name("dns_lookup")
    job = escape_job(attempt)
    assert job.visible[0].expected == SUCCESS_MARKER
    escaped = EscapeOutcome(attempt=attempt, verdict=verdict(job, "ok", passed=True))
    blocked = EscapeOutcome(attempt=attempt, verdict=verdict(job, "blocked", passed=False))
    assert escaped.escaped
    assert blocked.blocked
    assert report([blocked]) == EXIT_OK
    assert report([escaped]) == EXIT_ESCAPED
