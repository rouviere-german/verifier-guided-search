"""Verifier-guided tree search over candidate programs, under an explicit compute budget.

Sub-packages, roughly in dependency order:

- ``core``: settings, logging with PII redaction, tracing, metrics, hardened FastAPI app factory.
- ``agentic``: LLM provider protocol (Fake/Replay/Anthropic), untrusted-data envelopes, tool
  registry, approval gates, budget ledger and a hash-chained audit log.
- ``tasks``: the parametric benchmark - 15 families, each rendered with a reference solution, a
  visible test group used during search and a hidden group used only for scoring.
- ``sandbox``: the execution boundary. A child harness runs one untrusted program under rlimits,
  an audit-hook guard and a wall-clock watchdog; verdicts are cached by program hash.
- ``search``: the budget accountant, the candidate generators and the four strategies (greedy,
  best-of-N, beam, and MCTS with PUCT selection).
- ``value``: features that see visible feedback only, the gradient-boosting value model and its
  content-addressed registry.
- ``experiments``: budget sweeps, ablations and the report writer behind ``bench/``.
- ``api``: the search service and the runner service that executes candidates.

Nothing outside ``sandbox`` ever executes a candidate program, and nothing outside ``tasks`` and
``experiments`` may read a hidden test result - see docs/THREAT_MODEL.md.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
