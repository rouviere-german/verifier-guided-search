"""The two HTTP surfaces, driven through the real app: auth, clamping, and what each will not do.

The runner is the only process that executes candidate code, so the tests that matter are the ones
about what a *caller* can make it do: it cannot raise the resource ceiling, it cannot reach the
route without a key, and it cannot get free text from a run back. The search service is tested for
the mirror-image property: it will not accept a program or a test suite, only the name of a
benchmark task, and its readiness probe fails when the verifier or the value model is not usable.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, cast

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from verifier_guided_search.api.runner import clamp_limits, create_runner_app
from verifier_guided_search.api.service import (
    SearchRequest,
    build_sandbox,
    create_search_app,
    exec_limits,
)
from verifier_guided_search.core.concurrency import ConcurrencyLimiter
from verifier_guided_search.core.config import Settings
from verifier_guided_search.core.security import hash_api_key
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.sandbox.local import LocalSandbox
from verifier_guided_search.sandbox.protocol import ExecJob, ExecLimits
from verifier_guided_search.tasks.models import TestCase
from verifier_guided_search.tasks.suite import TaskSuite
from verifier_guided_search.value.model import constant_value_model
from verifier_guided_search.value.registry import ValueModelRegistry

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

API_KEY = "runner-test-key-" + "y" * 24
AUTH = {"X-API-Key": API_KEY}


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "api_key_sha256": [hash_api_key(API_KEY)],
        "environment": "test",
        "rate_limit_per_second": 1000.0,
        "rate_limit_burst": 1000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


_WORKING = "def solve(a):\n    return a + 1\n"


def job_payload(program: str = _WORKING, **limit_overrides: Any) -> dict[str, Any]:
    return ExecJob(
        program=program,
        entry_point="solve",
        visible=(TestCase(name="v00", args=(1,), expected=2),),
        hidden=(TestCase(name="h00", args=(5,), expected=6),),
        limits=ExecLimits(**limit_overrides) if limit_overrides else ExecLimits(),
    ).model_dump(mode="json")


# ------------------------------------------------------------------------------- runner


@pytest.fixture
def runner_client(tmp_path: Path) -> TestClient:
    app = create_runner_app(settings(), workroot=tmp_path / "sandbox")
    return TestClient(app, raise_server_exceptions=False)


def test_the_runner_executes_a_candidate_and_returns_a_verdict(runner_client: TestClient) -> None:
    response = runner_client.post("/v1/execute", json=job_payload(), headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["visible"]["results"][0]["status"] == "pass"
    assert body["hidden"]["results"][0]["status"] == "pass"


def test_the_execute_route_needs_a_key(runner_client: TestClient) -> None:
    assert runner_client.post("/v1/execute", json=job_payload()).status_code == 401
    assert (
        runner_client.post(
            "/v1/execute", json=job_payload(), headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )


def test_a_caller_cannot_raise_the_resource_ceiling(runner_client: TestClient) -> None:
    """A compromised search service must not be able to ask for a 60-second timeout."""
    greedy = job_payload("def solve(a):\n    return a + 1\n", timeout_seconds=60.0)
    greedy["limits"]["memory_bytes"] = 8 * 1024 * 1024 * 1024
    response = runner_client.post("/v1/execute", json=greedy, headers=AUTH)
    assert response.status_code == 200
    # The ceiling is the service's, so an over-long program is still killed at 1.5s.
    slow = job_payload("def solve(a):\n    while True:\n        pass\n", timeout_seconds=60.0)
    verdict = runner_client.post("/v1/execute", json=slow, headers=AUTH).json()
    assert verdict["status"] == "timeout"
    assert verdict["duration_seconds"] < 30


def test_clamping_only_ever_lowers_a_limit() -> None:
    config = settings(sandbox_timeout_seconds=1.0, sandbox_memory_bytes=64 * 1024 * 1024)
    asked = ExecLimits(timeout_seconds=30.0, memory_bytes=1024 * 1024 * 1024, max_processes=16)
    clamped = clamp_limits(asked, config)
    assert clamped.timeout_seconds == pytest.approx(1.0)
    assert clamped.memory_bytes == 64 * 1024 * 1024
    assert clamped.max_processes == 1

    modest = ExecLimits(timeout_seconds=0.25, memory_bytes=32 * 1024 * 1024)
    kept = clamp_limits(modest, config)
    assert kept.timeout_seconds == pytest.approx(0.25), "a smaller request must be honoured"
    assert kept.memory_bytes == 32 * 1024 * 1024


def test_the_runner_reports_a_blocked_escape_as_a_verdict(runner_client: TestClient) -> None:
    escaping = "import socket\n\n\ndef solve(a):\n    socket.socket()\n    return a + 1\n"
    verdict = runner_client.post("/v1/execute", json=job_payload(escaping), headers=AUTH).json()
    assert verdict["status"] == "blocked"
    assert verdict["violations"]


def test_the_runner_never_returns_free_text_from_a_run(runner_client: TestClient) -> None:
    noisy = (
        "print('RUNNER-STDOUT-MARKER')\n\n\n"
        "def solve(a):\n    raise ValueError('RUNNER-MESSAGE-MARKER')\n"
    )
    body = runner_client.post("/v1/execute", json=job_payload(noisy), headers=AUTH).text
    assert "RUNNER-STDOUT-MARKER" not in body
    assert "RUNNER-MESSAGE-MARKER" not in body
    assert "ValueError" in body, "the exception class is structured data and may travel"


def test_the_runner_rejects_a_malformed_job_without_echoing_it(runner_client: TestClient) -> None:
    bad = runner_client.post(
        "/v1/execute", json={"program": "", "entry_point": "solve", "visible": []}, headers=AUTH
    )
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "validation_error"
    extra = runner_client.post("/v1/execute", json={**job_payload(), "sneaky": "x"}, headers=AUTH)
    assert extra.status_code == 422


def test_the_runner_reports_ready_only_when_its_sandbox_works(tmp_path: Path) -> None:
    good = create_runner_app(settings(), workroot=tmp_path / "ok")
    with TestClient(good) as client:
        assert client.get("/healthz").status_code == 200
        body = client.get("/readyz").json()
        assert body == {"status": "ready", "checks": {"sandbox": "ok"}}

    broken = create_runner_app(
        settings(),
        sandbox=LocalSandbox(tmp_path / "bad", python_executable=str(tmp_path / "missing")),
    )
    with TestClient(broken) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"] == {"sandbox": "fail"}


class _BlockingSandbox(LocalSandbox):
    """A sandbox that parks inside ``execute`` until it is released.

    Stands in for the real blocking work - ``subprocess.run`` up to the wall timeout plus a grace -
    without needing a program that actually takes that long.
    """

    def __init__(self, workroot: Path, started: threading.Event, release: threading.Event) -> None:
        super().__init__(workroot)
        self._started = started
        self._release = release

    #: How long the fake blocking work lasts if nobody releases it. Long enough that a handler
    #: running on the event loop is unmistakable in the timings below, short enough that a
    #: regression fails in seconds.
    BLOCK_SECONDS = 5.0

    def execute(self, job: ExecJob) -> Any:
        self._started.set()
        self._release.wait(timeout=self.BLOCK_SECONDS)
        return super().execute(job)


def test_the_health_probes_answer_while_an_execution_is_in_flight(tmp_path: Path) -> None:
    """The route must not run on the event loop, or one execution takes the probes down with it.

    Both handlers do fully synchronous work. Declared ``async def`` they ran it on the loop, so
    ``/healthz``, ``/readyz``, ``/metrics`` and the rate-limit middleware were stalled for the whole
    duration of every execution and a single caller could wedge the process.

    Written against an ASGI transport on one event loop rather than through ``TestClient``, because
    that is exactly the condition being tested. The assertion is a timing one and has to be: with a
    ``def`` handler Starlette runs the work in its threadpool and the probe comes back immediately,
    while an ``async def`` handler holds the loop and the probe cannot be served until the work
    finishes - so what separates the two is how long the probe took, not whether it answered.
    """
    started = threading.Event()
    release = threading.Event()
    app = create_runner_app(
        settings(), sandbox=_BlockingSandbox(tmp_path / "sandbox", started, release)
    )

    async def scenario() -> tuple[int, int, float]:
        statuses: dict[str, int] = {}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:

            async def execute() -> None:
                response = await client.post("/v1/execute", json=job_payload(), headers=AUTH)
                statuses["execute"] = response.status_code

            async with anyio.create_task_group() as group:
                started_at = time.perf_counter()
                group.start_soon(execute)
                with anyio.fail_after(_BlockingSandbox.BLOCK_SECONDS * 6):
                    while not started.is_set():
                        await anyio.sleep(0.01)
                    probe = await client.get("/healthz")
                answered_after = time.perf_counter() - started_at
                statuses["healthz"] = probe.status_code
                release.set()
        return statuses["healthz"], statuses["execute"], answered_after

    health_status, execute_status, answered_after = anyio.run(scenario)
    assert health_status == 200
    assert execute_status == 200
    assert answered_after < _BlockingSandbox.BLOCK_SECONDS / 2, (
        f"/healthz took {answered_after:.2f}s, so the execution was holding the event loop"
    )


def test_a_second_concurrent_execution_is_refused_rather_than_queued(tmp_path: Path) -> None:
    """The threadpool is finite, so the route bounds itself and refuses the overflow quickly."""
    started = threading.Event()
    release = threading.Event()
    app = create_runner_app(
        settings(runner_max_concurrent_executions=1),
        sandbox=_BlockingSandbox(tmp_path / "sandbox", started, release),
    )

    async def scenario() -> tuple[int, int]:
        statuses: dict[str, int] = {}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:

            async def execute(label: str) -> None:
                response = await client.post("/v1/execute", json=job_payload(), headers=AUTH)
                statuses[label] = response.status_code

            async with anyio.create_task_group() as group:
                group.start_soon(execute, "first")
                with anyio.fail_after(_BlockingSandbox.BLOCK_SECONDS * 6):
                    while not started.is_set():
                        await anyio.sleep(0.01)
                    await execute("second")
                release.set()
        return statuses["first"], statuses["second"]

    first, second = anyio.run(scenario)
    assert first == 200
    assert second == 503
    limiter = cast(ConcurrencyLimiter, app.state.limiter)
    assert limiter.limit == 1
    assert limiter.refusals == 1


def test_the_runner_exposes_only_its_own_routes(runner_client: TestClient) -> None:
    """No /v1/search on the runner: it must not be able to start a search of its own."""
    app = cast("FastAPI", runner_client.app)
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/v1/execute" in paths
    assert "/v1/search" not in paths
    assert runner_client.post("/v1/search", json={}, headers=AUTH).status_code == 404


# ------------------------------------------------------------------------------- search service


@pytest.fixture
def search_client(tmp_path: Path, tiny_suite: TaskSuite) -> TestClient:
    registry = ValueModelRegistry(tmp_path / "var" / "value_models")
    registry.save(constant_value_model(0.5))
    app = create_search_app(
        settings(data_dir=tmp_path / "var", search_max_candidates=4),
        suite=tiny_suite,
        sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox")),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_a_search_request_returns_what_the_search_committed_to(
    search_client: TestClient, tiny_suite: TaskSuite
) -> None:
    task = tiny_suite.tasks[0]
    response = search_client.post(
        "/v1/search", json={"task_id": task.task_id, "strategy": "greedy"}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task.task_id
    assert body["family"] == task.family
    assert body["strategy"] == "greedy"
    assert body["evaluations"] >= 1
    # A candidate that timed out or failed to compile ran no tests, so a total of 0 is a fact
    # rather than a measurement; anything else must be the task's own group size.
    assert body["visible_total"] in {0, task.visible.size}
    assert body["hidden_total"] in {0, task.hidden.size}
    assert body["usage"]["candidates"] == body["evaluations"]
    assert len(body["program_sha256"]) == 64


def test_the_search_route_needs_a_key(search_client: TestClient, tiny_suite: TaskSuite) -> None:
    body = {"task_id": tiny_suite.tasks[0].task_id}
    assert search_client.post("/v1/search", json=body).status_code == 401


def test_a_request_cannot_supply_a_program_or_its_own_tests(
    search_client: TestClient, tiny_suite: TaskSuite
) -> None:
    """The tests a candidate is judged against are always the benchmark's."""
    assert set(SearchRequest.model_fields) == {
        "task_id",
        "strategy",
        "max_candidates",
        "children_per_expansion",
        "c_puct",
        "use_value_model",
    }
    smuggled = {
        "task_id": tiny_suite.tasks[0].task_id,
        "program": "import os\nos.system('id')\n",
        "visible": [],
    }
    response = search_client.post("/v1/search", json=smuggled, headers=AUTH)
    assert response.status_code == 422


def test_an_unknown_task_or_strategy_is_refused(search_client: TestClient) -> None:
    missing = search_client.post("/v1/search", json={"task_id": "nope/00"}, headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "unknown_task"
    malformed = search_client.post("/v1/search", json={"task_id": "NotATask"}, headers=AUTH)
    assert malformed.status_code == 422


def test_the_candidate_budget_is_clamped_to_the_services_ceiling(
    search_client: TestClient, tiny_suite: TaskSuite
) -> None:
    body = search_client.post(
        "/v1/search",
        json={"task_id": tiny_suite.tasks[0].task_id, "max_candidates": 200, "strategy": "mcts"},
        headers=AUTH,
    ).json()
    assert body["evaluations"] <= 4, "the request exceeded search_max_candidates"
    assert body["usage"]["candidates"] <= 4


def test_readiness_requires_the_runner_and_a_verifiable_value_model(tmp_path: Path) -> None:
    root = tmp_path / "var"
    registry = ValueModelRegistry(root / "value_models")
    version = registry.save(constant_value_model(0.5))
    config = settings(data_dir=root)

    app = create_search_app(
        config, suite=None, sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox"))
    )
    with TestClient(app) as client:
        body = client.get("/readyz").json()
    assert body == {"status": "ready", "checks": {"runner": "ok", "value_model": "ok"}}

    # A tampered model file fails its content hash, so the service reports not-ready.
    path = registry.path_for(version)
    path.write_text(
        path.read_text(encoding="utf-8").replace('"notes":"', '"notes":"edited'), encoding="utf-8"
    )
    tampered = create_search_app(
        config, suite=None, sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox"))
    )
    with TestClient(tampered) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["value_model"] == "fail"


def test_an_unreachable_runner_makes_the_service_not_ready(tmp_path: Path) -> None:
    """A search against a dead verifier would return "solved nothing", which is worse than 503."""
    dead = CachingSandbox(
        LocalSandbox(tmp_path / "sandbox", python_executable=str(tmp_path / "missing"))
    )
    app = create_search_app(settings(data_dir=tmp_path / "var"), suite=None, sandbox=dead)
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["runner"] == "fail"


def test_a_service_with_no_value_model_is_ready_and_says_so(tmp_path: Path) -> None:
    app = create_search_app(
        settings(data_dir=tmp_path / "empty"),
        suite=None,
        sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox")),
    )
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
    assert app.state.value_model is None


def test_a_pinned_but_missing_value_model_version_is_not_ready(tmp_path: Path) -> None:
    app = create_search_app(
        settings(data_dir=tmp_path / "var", value_model_version="a" * 64),
        suite=None,
        sandbox=CachingSandbox(LocalSandbox(tmp_path / "sandbox")),
    )
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["value_model"] == "fail"
    assert app.state.value_model is None, "a missing pinned model must not silently fall back"


def test_the_verifier_client_is_the_runner_when_a_url_is_configured() -> None:
    from verifier_guided_search.sandbox.client import HttpSandbox

    remote = build_sandbox(settings(runner_url="http://runner:8001"))
    assert isinstance(remote, CachingSandbox)
    assert isinstance(remote.inner, HttpSandbox)
    local = build_sandbox(settings())
    assert isinstance(local, CachingSandbox)
    assert isinstance(local.inner, LocalSandbox)


def test_a_runner_url_with_a_path_is_refused() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="runner_url"):
        settings(runner_url="http://runner:8001/v1/execute")
    with pytest.raises(pydantic.ValidationError, match="runner_url"):
        settings(runner_url="not-a-url")


def test_the_exec_limits_come_from_the_settings() -> None:
    limits = exec_limits(settings(sandbox_timeout_seconds=0.75, sandbox_memory_bytes=2**25))
    assert limits.timeout_seconds == pytest.approx(0.75)
    assert limits.memory_bytes == 2**25


def test_the_search_response_reports_hidden_outcomes_of_the_chosen_candidate_only(
    search_client: TestClient, tiny_suite: TaskSuite
) -> None:
    """Grading the search is fine; the search itself never saw these numbers."""
    body = search_client.post(
        "/v1/search",
        json={"task_id": tiny_suite.tasks[0].task_id, "strategy": "best_of_n"},
        headers=AUTH,
    ).json()
    assert body["hidden_passed"] <= body["hidden_total"]
    # "Solved" needs tests to have actually run and all of them to have passed. An empty group is
    # not a pass: that is the ``NO_RUN_STATUSES`` case, and treating 0/0 as success would turn every
    # timeout into a solve.
    assert body["solved_hidden"] == (
        body["hidden_total"] > 0 and body["hidden_passed"] == body["hidden_total"]
    )
    assert body["solved_visible"] == (
        body["visible_total"] > 0 and body["visible_passed"] == body["visible_total"]
    )
    assert body["false_pass"] == (body["solved_visible"] and not body["solved_hidden"])
