"""The HTTP client for the runner container: what it accepts back, and what it refuses.

Driven through an ``httpx.MockTransport``, so nothing leaves the process. The interesting cases are
all refusals: a runner that is unreachable, one that answers with an error, one that answers with
something that is not a verdict, and - the one that matters most - one that answers about a
*different program* than the one submitted. That last check is what stops a confused or malicious
runner from having a cached pass attributed to the wrong candidate.
"""

from __future__ import annotations

import httpx
import pytest

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.client import HttpSandbox, SandboxUnavailable
from verifier_guided_search.sandbox.protocol import (
    ExecJob,
    GroupResult,
    TestResult,
    Verdict,
)
from verifier_guided_search.tasks.models import TestCase

pytestmark = pytest.mark.unit

_PROGRAM = "def solve(a):\n    return a + 1\n"
_BASE = "http://runner:8001"


def job() -> ExecJob:
    return ExecJob(
        program=_PROGRAM,
        entry_point="solve",
        visible=(TestCase(name="v00", args=(1,), expected=2),),
        hidden=(TestCase(name="h00", args=(5,), expected=6),),
    )


def verdict_for(target: ExecJob, *, program_sha256: str | None = None) -> Verdict:
    return Verdict(
        program_sha256=program_sha256 or target.program_sha256,
        suite_sha256=target.suite_sha256,
        status="ok",
        visible=GroupResult(results=(TestResult(name="v00", status="pass"),)),
        hidden=GroupResult(results=(TestResult(name="h00", status="pass"),)),
        duration_seconds=0.07,
    )


def client_for(handler: object) -> HttpSandbox:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return HttpSandbox(_BASE, api_key="runner-key", client=httpx.Client(transport=transport))


def test_a_verdict_comes_back_and_the_key_is_sent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=verdict_for(job()).model_dump(mode="json"))

    with client_for(handler) as sandbox:
        result = sandbox.execute(job())
    assert result.status == "ok"
    assert result.visible.all_passed
    assert seen[0].url.path == "/v1/execute"
    assert seen[0].headers["X-API-Key"] == "runner-key"
    assert seen[0].method == "POST"


def test_the_program_and_its_tests_are_sent_as_json() -> None:
    import json

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=verdict_for(job()).model_dump(mode="json"))

    with client_for(handler) as sandbox:
        sandbox.execute(job())
    assert captured["program"] == _PROGRAM
    assert captured["entry_point"] == "solve"
    assert isinstance(captured["visible"], list)
    assert isinstance(captured["limits"], dict)


def test_a_verdict_about_another_program_is_refused() -> None:
    """The check that stops a pass being attributed to the wrong candidate."""

    def handler(request: httpx.Request) -> httpx.Response:
        wrong = verdict_for(job(), program_sha256=sha256_hex("some other program"))
        return httpx.Response(200, json=wrong.model_dump(mode="json"))

    with (
        client_for(handler) as sandbox,
        pytest.raises(SandboxUnavailable, match="answered about a different program"),
    ):
        sandbox.execute(job())


def test_an_unreachable_runner_raises_rather_than_reporting_a_failure() -> None:
    """A dead verifier must not look like "the program did not pass"."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with client_for(handler) as sandbox, pytest.raises(SandboxUnavailable, match="unreachable"):
        sandbox.execute(job())


def test_a_timeout_is_reported_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with client_for(handler) as sandbox, pytest.raises(SandboxUnavailable, match="unreachable"):
        sandbox.execute(job())


@pytest.mark.parametrize("status", [401, 422, 500, 503])
def test_a_non_200_answer_is_refused(status: int) -> None:
    with (
        client_for(lambda request: httpx.Response(status, json={})) as sandbox,
        pytest.raises(SandboxUnavailable, match=f"HTTP {status}"),
    ):
        sandbox.execute(job())


def test_an_answer_that_is_not_a_verdict_is_refused() -> None:
    payloads: list[object] = [{"status": "ok"}, {"nonsense": True}, []]
    for payload in payloads:
        with (
            client_for(lambda request, body=payload: httpx.Response(200, json=body)) as sandbox,
            pytest.raises(SandboxUnavailable, match="unusable verdict"),
        ):
            sandbox.execute(job())


def test_a_non_json_answer_is_refused() -> None:
    with (
        client_for(
            lambda request: httpx.Response(
                200, content=b"not json", headers={"content-type": "application/json"}
            )
        ) as sandbox,
        pytest.raises(SandboxUnavailable, match="unusable verdict"),
    ):
        sandbox.execute(job())


def test_the_readiness_probe_reports_reachability_without_raising() -> None:
    with client_for(lambda request: httpx.Response(200, json={"status": "ok"})) as sandbox:
        assert sandbox.ping() is True
    with client_for(lambda request: httpx.Response(503, json={})) as sandbox:
        assert sandbox.ping() is False

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with client_for(dead) as sandbox:
        assert sandbox.ping() is False


def test_the_probe_hits_healthz() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    with client_for(handler) as sandbox:
        sandbox.ping()
    assert seen == ["/healthz"]


def test_a_trailing_slash_in_the_base_url_does_not_double_up() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=verdict_for(job()).model_dump(mode="json"))

    sandbox = HttpSandbox(_BASE + "/", client=httpx.Client(transport=httpx.MockTransport(handler)))
    sandbox.execute(job())
    sandbox.close()
    assert seen == [f"{_BASE}/v1/execute"]


def test_no_key_means_no_auth_header() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=verdict_for(job()).model_dump(mode="json"))

    sandbox = HttpSandbox(_BASE, client=httpx.Client(transport=httpx.MockTransport(handler)))
    sandbox.execute(job())
    sandbox.close()
    assert "x-api-key" not in seen[0]


def test_a_client_the_caller_supplied_is_not_closed_by_the_sandbox() -> None:
    """The service shares one connection pool, so the sandbox must not close what it did not
    open."""
    supplied = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"}))
    )
    with HttpSandbox(_BASE, client=supplied) as sandbox:
        sandbox.ping()
    assert not supplied.is_closed
    supplied.close()
