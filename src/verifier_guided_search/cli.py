"""Command line entry point: ``verifier-guided-search`` / ``vgs``, or ``python -m
verifier_guided_search``.

Subcommands:

``demo``             the offline end-to-end demo (no API key, no Docker, no network)
``serve``            the search service
``serve-runner``     the sandbox runner service (what the runner container runs)
``tasks``            list the benchmark, or print one task
``escape-suite``     run the sandbox escape suite and report what was blocked
``audit verify``     verify a hash-chained audit log
``generate-api-key`` print a new API key and its SHA-256 digest

Heavy imports (uvicorn, the app factories, scikit-learn) happen inside the subcommand that needs
them, so ``--help`` and ``tasks`` stay instant and the offline paths never import the Anthropic SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


def _serve_search(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from verifier_guided_search.api.service import create_search_app  # noqa: PLC0415
    from verifier_guided_search.core.config import get_settings  # noqa: PLC0415
    from verifier_guided_search.core.logging import configure_logging  # noqa: PLC0415
    from verifier_guided_search.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_search_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _serve_runner(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415

    from verifier_guided_search.api.runner import create_runner_app  # noqa: PLC0415
    from verifier_guided_search.core.config import get_settings  # noqa: PLC0415
    from verifier_guided_search.core.logging import configure_logging  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_runner_app(settings)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _tasks(task_id: str | None, instances: int) -> int:
    from verifier_guided_search.tasks.suite import build_suite  # noqa: PLC0415

    suite = build_suite(instances_per_family=instances)
    if task_id is None:
        print(
            json.dumps(
                {
                    "suite_sha256": suite.sha256,
                    "tasks": len(suite),
                    "families": list(suite.families),
                    "task_ids": [task.task_id for task in suite],
                },
                indent=2,
            )
        )
        return 0
    try:
        task = suite.get(task_id)
    except KeyError:
        print(f"unknown task {task_id!r}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "family": task.family,
                "entry_point": task.entry_point,
                "statement": task.statement,
                "start_program": task.start_program,
                "visible_tests": task.visible.size,
                "hidden_tests": task.hidden.size,
                "suite_sha256": task.suite_sha256,
            },
            indent=2,
        )
    )
    return 0


def _escape_suite(workdir: Path | None) -> int:
    import tempfile  # noqa: PLC0415

    from verifier_guided_search.experiments.report import render_escape_suite  # noqa: PLC0415
    from verifier_guided_search.sandbox.escape import (  # noqa: PLC0415
        ESCAPE_SUITE,
        blocked_fraction,
        evaluate_attempt,
    )
    from verifier_guided_search.sandbox.local import LocalSandbox  # noqa: PLC0415

    def run(root: Path) -> int:
        sandbox = LocalSandbox(root)
        outcomes = [evaluate_attempt(sandbox.execute, attempt) for attempt in ESCAPE_SUITE]
        print(render_escape_suite(outcomes))
        unexpected = [item for item in outcomes if not item.status_expected]
        for item in unexpected:
            print(
                f"UNEXPECTED STATUS: {item.attempt.name} reported {item.verdict.status}, "
                f"expected one of {sorted(item.attempt.expected_statuses)}",
                file=sys.stderr,
            )
        return 0 if blocked_fraction(outcomes) == 1.0 and not unexpected else 1

    if workdir is not None:
        return run(workdir)
    with tempfile.TemporaryDirectory(prefix="vgs-escape-") as tmp:
        return run(Path(tmp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verifier-guided-search")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the search service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    runner = sub.add_parser("serve-runner", help="run the sandbox runner service")
    runner.add_argument("--host", default=None)
    runner.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    listing = sub.add_parser("tasks", help="list the benchmark, or print one task")
    listing.add_argument("task_id", nargs="?", default=None)
    listing.add_argument("--instances-per-family", type=int, default=10)

    escape = sub.add_parser("escape-suite", help="run the sandbox escape suite")
    escape.add_argument("--workdir", type=Path, default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=None)
    demo.add_argument("--instances-per-family", type=int, default=1)
    demo.add_argument("--no-approve", action="store_true")
    demo.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve_search(args.host, args.port)
    if args.command == "serve-runner":
        return _serve_runner(args.host, args.port)
    if args.command == "generate-api-key":
        from verifier_guided_search.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "audit":
        from verifier_guided_search.agentic.audit import main as audit_main  # noqa: PLC0415

        forwarded = ["verify", str(args.path)]
        if args.expect_head:
            forwarded += ["--expect-head", args.expect_head]
        return audit_main(forwarded)
    if args.command == "tasks":
        return _tasks(args.task_id, args.instances_per_family)
    if args.command == "escape-suite":
        return _escape_suite(args.workdir)

    from verifier_guided_search.demo import main as demo_main  # noqa: PLC0415

    forwarded = ["--instances-per-family", str(args.instances_per_family)]
    if args.workdir is not None:
        forwarded += ["--workdir", str(args.workdir)]
    if args.no_approve:
        forwarded.append("--no-approve")
    if args.json:
        forwarded.append("--json")
    return demo_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
