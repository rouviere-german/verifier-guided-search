"""Command line entry point: ``verifier-guided-search`` / ``vgs``, or ``python -m
verifier_guided_search``.

Subcommands:

``demo``             the offline end-to-end demo (no API key, no Docker, no network)
``search``           one budgeted search over one benchmark task, offline or with the live generator
``serve``            the search service
``serve-runner``     the sandbox runner service (what the runner container runs)
``tasks``            list the benchmark, or print one task
``escape-suite``     run the sandbox escape suite and report what was blocked
``audit verify``     verify a hash-chained audit log
``generate-api-key`` print a new API key and its SHA-256 digest

Heavy imports (uvicorn, the app factories, scikit-learn) happen inside the subcommand that needs
them, so ``--help`` and ``tasks`` stay instant and the offline paths never import the Anthropic SDK.

``search`` exists because ``ClaudeGenerator`` had no caller outside its own unit tests: the demo,
the bench and the service all hard-wired the offline mutation generator, so a key bought nothing.
``vgs search TASK --generator claude`` is the live path; ``--generator mutation`` (the default) is
the same code path offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verifier_guided_search.search.generator import GENERATOR_NAMES

__all__ = ["build_parser", "main"]


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


def _search(
    task_id: str,
    *,
    strategy: str,
    candidates: int,
    children: int,
    generator_name: str,
    use_value_model: bool,
    workdir: Path | None,
) -> int:
    """Run one budgeted search and print what it committed to, as JSON."""
    import tempfile  # noqa: PLC0415

    from verifier_guided_search.errors import SearchError  # noqa: PLC0415
    from verifier_guided_search.sandbox.client import CachingSandbox  # noqa: PLC0415
    from verifier_guided_search.sandbox.local import LocalSandbox  # noqa: PLC0415
    from verifier_guided_search.sandbox.protocol import ExecLimits  # noqa: PLC0415
    from verifier_guided_search.search.budget import (  # noqa: PLC0415
        BudgetAccountant,
        BudgetSpec,
    )
    from verifier_guided_search.search.generator import (  # noqa: PLC0415
        build_generator,
        token_budget_for,
    )
    from verifier_guided_search.search.strategies import (  # noqa: PLC0415
        STRATEGIES,
        SearchConfig,
        SearchContext,
        run_search,
    )
    from verifier_guided_search.tasks.suite import build_suite  # noqa: PLC0415

    if strategy not in STRATEGIES:
        print(
            f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}", file=sys.stderr
        )
        return 2
    suite = build_suite(instances_per_family=10)
    try:
        task = suite.get(task_id)
    except KeyError:
        print(f"unknown task {task_id!r}; list them with 'vgs tasks'", file=sys.stderr)
        return 2
    try:
        generator = build_generator(generator_name)
    except SearchError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    def run(root: Path) -> int:
        limits = ExecLimits()
        accountant = BudgetAccountant(
            BudgetSpec.of(
                candidates=candidates,
                tokens=token_budget_for(generator, expansions=candidates, k=children),
                sandbox_seconds=candidates * limits.timeout_seconds * 4.0,
                usd=1.0,
            )
        )
        outcome = run_search(
            task,
            SearchContext(
                sandbox=CachingSandbox(LocalSandbox(root)),
                generator=generator,
                accountant=accountant,
                config=SearchConfig(
                    strategy=strategy,
                    children_per_expansion=children,
                    use_value_model=use_value_model,
                    limits=limits,
                ),
                actor=f"cli:{generator_name}",
            ),
        )
        answer = outcome.answer
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "strategy": strategy,
                    "generator": generator_name,
                    "model": generator.model,
                    "stop_reason": outcome.stop_reason,
                    "evaluations": len(outcome.evaluations),
                    "nodes": outcome.nodes,
                    "expansions": outcome.expansions,
                    "solved_visible": bool(answer is not None and answer.solved_visible),
                    # Reported, never seen by the search: the hidden group grades the answer it
                    # had already committed to.
                    "solved_hidden": bool(answer is not None and answer.solved_hidden),
                    "false_pass": bool(answer is not None and answer.false_pass),
                    "usage": accountant.committed.as_dict(),
                    "generator_token_overshoot": accountant.generator_token_overshoot,
                },
                indent=2,
            )
        )
        return 0

    if workdir is not None:
        return run(workdir)
    with tempfile.TemporaryDirectory(prefix="vgs-search-") as tmp:
        return run(Path(tmp))


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


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, in one place so ``main`` stays a dispatch table."""
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

    one = sub.add_parser("search", help="run one budgeted search over one benchmark task")
    one.add_argument("task_id")
    one.add_argument("--strategy", default="mcts")
    one.add_argument("--candidates", type=int, default=16, help="the candidate budget")
    one.add_argument("--children", type=int, default=4, help="children per expansion (k)")
    one.add_argument(
        "--generator",
        choices=GENERATOR_NAMES,
        default="mutation",
        help="mutation is offline; claude needs ANTHROPIC_API_KEY",
    )
    one.add_argument("--no-value-model", action="store_true")
    one.add_argument("--workdir", type=Path, default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=None)
    demo.add_argument("--instances-per-family", type=int, default=1)
    demo.add_argument("--no-approve", action="store_true")
    demo.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if args.command == "search":
        return _search(
            args.task_id,
            strategy=args.strategy,
            candidates=args.candidates,
            children=args.children,
            generator_name=args.generator,
            use_value_model=not args.no_value_model,
            workdir=args.workdir,
        )

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
