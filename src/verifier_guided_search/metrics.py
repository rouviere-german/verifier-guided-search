"""Prometheus metrics for the search, on a per-instance registry (never the global default).

Names follow the spec's list: nodes expanded, sandbox execution seconds, verdicts by result, budget
exhaustions and the verdict cache hit ratio, plus the counters the ratio is derived from so a scrape
can recompute it over any window.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["SearchMetrics", "create_search_metrics"]


@dataclass(frozen=True, slots=True)
class SearchMetrics:
    registry: CollectorRegistry
    nodes_expanded: Counter
    programs_evaluated_total: Counter
    sandbox_exec_seconds: Histogram
    verifier_verdicts_total: Counter
    budget_exhausted_total: Counter
    verdict_cache_hits_total: Counter
    verdict_cache_misses_total: Counter
    verdict_cache_hit_ratio: Gauge
    searches_total: Counter


def create_search_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "vgs"
) -> SearchMetrics:
    reg = registry if registry is not None else CollectorRegistry()
    return SearchMetrics(
        registry=reg,
        nodes_expanded=Counter(
            "nodes_expanded",
            "Search nodes expanded, by strategy.",
            ["strategy"],
            namespace=namespace,
            registry=reg,
        ),
        programs_evaluated_total=Counter(
            "programs_evaluated_total",
            "Candidate programs sent to the verifier, by strategy.",
            ["strategy"],
            namespace=namespace,
            registry=reg,
        ),
        sandbox_exec_seconds=Histogram(
            "sandbox_exec_seconds",
            "Wall-clock seconds per sandbox execution (cache hits excluded).",
            namespace=namespace,
            registry=reg,
            buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
        ),
        verifier_verdicts_total=Counter(
            "verifier_verdicts_total",
            "Verdicts returned by the verifier, by result status.",
            ["result"],
            namespace=namespace,
            registry=reg,
        ),
        budget_exhausted_total=Counter(
            "budget_exhausted_total",
            "Searches stopped because a budget dimension was exhausted, by dimension.",
            ["dimension"],
            namespace=namespace,
            registry=reg,
        ),
        verdict_cache_hits_total=Counter(
            "verdict_cache_hits_total",
            "Verifier lookups answered from the content-addressed verdict cache.",
            namespace=namespace,
            registry=reg,
        ),
        verdict_cache_misses_total=Counter(
            "verdict_cache_misses_total",
            "Verifier lookups that required a sandbox execution.",
            namespace=namespace,
            registry=reg,
        ),
        verdict_cache_hit_ratio=Gauge(
            "verdict_cache_hit_ratio",
            "Cache hits divided by lookups for the current process.",
            namespace=namespace,
            registry=reg,
        ),
        searches_total=Counter(
            "searches_total",
            "Completed searches, by strategy and outcome.",
            ["strategy", "outcome"],
            namespace=namespace,
            registry=reg,
        ),
    )
