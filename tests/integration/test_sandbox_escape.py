"""The escape suite, run for real: every attempt must fail to escape.

Each attempt is a candidate program whose single visible test passes *only if the escape worked*, so
"100% blocked" is one assertion: no attempt passes its own test. The suite is also checked from the
other side - the attempt that should have been refused by the in-process guard really reports
``blocked`` and names a violation - so a guard that silently stopped enforcing would fail here
rather than quietly turn the suite into a set of programs that crash for unrelated reasons.

What this does *not* prove: that the container is escape-proof. The guard is defence in depth and
the
container is the boundary (docs/THREAT_MODEL.md). The compose-level suite runs on a Linux runner in
CI (`.github/workflows/ci.yml#sandbox-escape`).
"""

from __future__ import annotations

import pytest

from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.sandbox.escape import (
    ESCAPE_SUITE,
    SUCCESS_MARKER,
    EscapeOutcome,
    attempt_by_name,
    blocked_fraction,
    escape_job,
    evaluate_attempt,
)
from verifier_guided_search.sandbox.local import LocalSandbox

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> list[EscapeOutcome]:
    sandbox = LocalSandbox(tmp_path_factory.mktemp("escape"))
    return [evaluate_attempt(sandbox.execute, attempt) for attempt in ESCAPE_SUITE]


def test_the_whole_suite_is_blocked(outcomes: list[EscapeOutcome]) -> None:
    """The spec's claim: the sandbox escape suite is 100% blocked."""
    escaped = [item.attempt.name for item in outcomes if item.escaped]
    assert escaped == [], f"these attempts escaped: {escaped}"
    assert blocked_fraction(outcomes) == 1.0


def test_every_attempt_reported_a_status_the_suite_expected(outcomes: list[EscapeOutcome]) -> None:
    """Guards against an attempt "passing" because it broke for an unrelated reason."""
    for item in outcomes:
        assert item.status_expected, (
            f"{item.attempt.name} reported {item.verdict.status}, "
            f"expected one of {sorted(item.attempt.expected_statuses)}"
        )


def test_the_guard_refuses_every_attempt_that_is_its_job(outcomes: list[EscapeOutcome]) -> None:
    for item in outcomes:
        attempt = item.attempt
        if not attempt.must_be_blocked:
            continue
        assert item.guard_refused, (
            f"{attempt.name} was not refused by the guard (status {item.verdict.status})"
        )
        violations = set(item.verdict.violations)
        assert violations, f"{attempt.name} was blocked without recording a violation"
        assert any(
            any(hint in violation for violation in violations) for hint in attempt.violation_hints
        ), f"{attempt.name} recorded {violations}, expected one of {attempt.violation_hints}"


def test_the_suite_covers_every_category_the_spec_names(outcomes: list[EscapeOutcome]) -> None:
    categories = {item.attempt.category for item in outcomes}
    assert categories == {"network", "process", "filesystem", "secrets", "resources"}
    assert len(ESCAPE_SUITE) >= 12


def test_network_and_process_attempts_are_all_refused_by_the_guard(
    outcomes: list[EscapeOutcome],
) -> None:
    for item in outcomes:
        if item.attempt.category in {"network", "process"}:
            assert item.verdict.status == "blocked"


def test_a_long_running_attempt_is_killed_rather_than_blocked(
    outcomes: list[EscapeOutcome],
) -> None:
    sleep = next(item for item in outcomes if item.attempt.name == "long_sleep")
    assert sleep.verdict.status == "timeout"
    assert sleep.blocked


def test_the_environment_attempt_finds_no_credentials(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set a credential in the parent, then prove the child cannot see it.

    Nothing blocks ``os.environ``; the control is the parent's allowlist in ``child_environment()``.
    Without this test the attempt would pass trivially in an environment that happens to be clean.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-reach-the-child")
    monkeypatch.setenv("SOME_DEPLOY_TOKEN", "t0ken-should-never-reach-the-child")
    sandbox = LocalSandbox(tmp_path_factory.mktemp("escape-env"))
    outcome = evaluate_attempt(sandbox.execute, attempt_by_name("env_secret_read"))
    assert outcome.blocked
    assert outcome.verdict.status == "ok", "the probe should have run and found nothing"
    assert not outcome.verdict.visible.all_passed


def test_an_attempt_that_did_escape_would_be_detected(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A negative control: the harness really does report a pass when the marker comes back.

    Without this, "everything is blocked" could just mean the runner never reports a pass at all.
    """
    sandbox = LocalSandbox(tmp_path_factory.mktemp("escape-control"))
    honest = attempt_by_name("socket_connect")
    faked = type(honest)(
        name="control",
        category="network",
        mechanism="platform",
        description="Returns the success marker without doing anything forbidden.",
        program=f'def probe():\n    return "{SUCCESS_MARKER}"\n',
        expected_statuses=frozenset({"ok"}),
        must_be_blocked=False,
    )
    outcome = evaluate_attempt(sandbox.execute, faked)
    assert outcome.escaped, "a returned marker must be reported as an escape"
    assert not outcome.blocked
    assert blocked_fraction([outcome]) == 0.0


def test_the_job_for_an_attempt_expects_the_marker_and_nothing_else() -> None:
    job = escape_job(attempt_by_name("dns_lookup"))
    assert len(job.visible) == 1
    assert job.visible[0].expected == SUCCESS_MARKER
    assert job.hidden == ()
    assert job.entry_point == "probe"


def test_attempt_names_are_unique_and_addressable() -> None:
    names = [attempt.name for attempt in ESCAPE_SUITE]
    assert len(set(names)) == len(names)
    for name in names:
        assert attempt_by_name(name).name == name
    with pytest.raises(KeyError, match="unknown escape attempt"):
        attempt_by_name("no_such_attempt")


def test_the_suite_rows_serialise_for_a_report(outcomes: list[EscapeOutcome]) -> None:
    rows = [item.as_row() for item in outcomes]
    assert all(
        set(row) == {"attempt", "category", "mechanism", "status", "blocked", "violations"}
        for row in rows
    )
    assert all(row["blocked"] is True for row in rows)


def test_blocked_fraction_refuses_an_empty_run() -> None:
    with pytest.raises(ValueError, match="no escape outcomes"):
        blocked_fraction([])


def test_the_suite_runs_through_the_caching_sandbox_too(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The escape suite must hold for the client the search actually uses, not just the raw one."""
    sandbox = CachingSandbox(LocalSandbox(tmp_path_factory.mktemp("escape-cached")))
    attempt = attempt_by_name("subprocess_spawn")
    first = evaluate_attempt(sandbox.execute, attempt)
    second = evaluate_attempt(sandbox.execute, attempt)
    assert first.blocked
    assert second.blocked
    assert sandbox.record.cache_hits == 1
    assert second.verdict.status == "blocked"
