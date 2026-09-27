"""Run the sandbox escape suite against a *remote* runner, and exit non-zero if anything escaped.

The ordinary test suite runs the same attempts against the same child harness in a local subprocess
(``tests/integration/test_sandbox_escape.py``). That measures the in-process guard. This module
measures the **boundary**: it drives the runner as the compose file declares it - internal network,
no egress, read-only rootfs, tmpfs jail, dropped capabilities, pids and memory limits - over HTTP,
from a container on that same network.

It is a module rather than a shell script in the workflow because the suite, the success marker and
the per-attempt expected statuses live in :mod:`verifier_guided_search.sandbox.escape`; a script
would duplicate them and drift.

Run by ``.github/workflows/ci.yml#sandbox-escape``:

    APP_RUNNER_URL=http://runner:8001 APP_RUNNER_API_KEY=... \\
      python -m verifier_guided_search.api.escape_check

Exit codes: 0 when every attempt was blocked with an expected status, 1 when any attempt escaped or
reported something the suite did not expect, 2 when the runner could not be reached at all (an
infrastructure failure, which must not be mistaken for a pass).
"""

from __future__ import annotations

import sys
from typing import Final

from verifier_guided_search.core.config import Settings, get_settings
from verifier_guided_search.experiments.report import render_escape_suite
from verifier_guided_search.sandbox.client import HttpSandbox, SandboxUnavailable
from verifier_guided_search.sandbox.escape import (
    ESCAPE_SUITE,
    EscapeOutcome,
    blocked_fraction,
    evaluate_attempt,
)

__all__ = ["EXIT_ESCAPED", "EXIT_OK", "EXIT_UNREACHABLE", "main", "run_against"]

EXIT_OK: Final = 0
EXIT_ESCAPED: Final = 1
EXIT_UNREACHABLE: Final = 2


def run_against(sandbox: HttpSandbox) -> list[EscapeOutcome]:
    """Run every attempt through ``sandbox``, in the suite's fixed order."""
    return [evaluate_attempt(sandbox.execute, attempt) for attempt in ESCAPE_SUITE]


def report(outcomes: list[EscapeOutcome]) -> int:
    """Print the table and return the exit code. Any escape or surprise is a failure."""
    print(render_escape_suite(outcomes))  # noqa: T201 - this module's output is its purpose
    escaped = [item.attempt.name for item in outcomes if item.escaped]
    unexpected = [
        f"{item.attempt.name} reported {item.verdict.status}, "
        f"expected one of {sorted(item.attempt.expected_statuses)}"
        for item in outcomes
        if not item.status_expected
    ]
    for name in escaped:
        print(f"ESCAPED: {name}", file=sys.stderr)  # noqa: T201
    for note in unexpected:
        print(f"UNEXPECTED: {note}", file=sys.stderr)  # noqa: T201
    if escaped or unexpected:
        return EXIT_ESCAPED
    print(f"all {len(outcomes)} attempts blocked ({blocked_fraction(outcomes):.0%})")  # noqa: T201
    return EXIT_OK


def main(settings: Settings | None = None) -> int:
    cfg = settings if settings is not None else get_settings()
    if not cfg.runner_url:
        print("APP_RUNNER_URL is not set; this check needs a remote runner", file=sys.stderr)  # noqa: T201
        return EXIT_UNREACHABLE
    key = cfg.runner_api_key.get_secret_value() if cfg.runner_api_key else None
    with HttpSandbox(cfg.runner_url, api_key=key, timeout=cfg.runner_timeout_seconds) as sandbox:
        if not sandbox.ping():
            # Distinguished from a pass on purpose: "we could not reach the runner" must never be
            # reported as "nothing escaped".
            print(f"runner at {cfg.runner_url} is not reachable", file=sys.stderr)  # noqa: T201
            return EXIT_UNREACHABLE
        try:
            outcomes = run_against(sandbox)
        except SandboxUnavailable as exc:
            print(f"runner became unavailable mid-suite: {exc}", file=sys.stderr)  # noqa: T201
            return EXIT_UNREACHABLE
    return report(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
