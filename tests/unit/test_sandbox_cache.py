"""The content-addressed verdict cache, and the one thing it must never do: answer the wrong job.

The key is ``sha256(program || tests || limits)``, so the interesting tests are the near misses: the
same program against different tests, the same tests with a different timeout, and a persisted line
whose stored verdict no longer re-derives its key. All three must miss or be dropped, because a
wrong hit rewrites every result downstream of it silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.cache import CacheEntry, VerdictCache
from verifier_guided_search.sandbox.client import CachingSandbox, SandboxClient
from verifier_guided_search.sandbox.protocol import (
    ExecJob,
    ExecLimits,
    GroupResult,
    TestResult,
    Verdict,
)
from verifier_guided_search.tasks.models import TestCase

pytestmark = pytest.mark.unit

_PROGRAM = "def solve(a):\n    return a\n"


def job(program: str = _PROGRAM, *, expected: object = 1, timeout: float = 1.5) -> ExecJob:
    return ExecJob(
        program=program,
        entry_point="solve",
        visible=(TestCase(name="v00", args=(1,), expected=expected),),
        hidden=(TestCase(name="h00", args=(2,), expected=2),),
        limits=ExecLimits(timeout_seconds=timeout),
    )


def verdict_for(target: ExecJob, *, status: str = "ok", seconds: float = 0.05) -> Verdict:
    return Verdict(
        program_sha256=target.program_sha256,
        suite_sha256=target.suite_sha256,
        status=status,
        visible=GroupResult(results=(TestResult(name="v00", status="pass"),)),
        hidden=GroupResult(results=(TestResult(name="h00", status="pass"),)),
        duration_seconds=seconds,
    )


class CountingSandbox:
    """A stand-in verifier that records how often it was actually asked to run something."""

    def __init__(self) -> None:
        self.calls = 0

    def ping(self) -> bool:
        return True

    def execute(self, target: ExecJob) -> Verdict:
        self.calls += 1
        return verdict_for(target)


def test_a_byte_identical_job_hits(tmp_path: Path) -> None:
    cache = VerdictCache()
    first = job()
    assert cache.get(first) is None
    cache.put(first, verdict_for(first))
    assert cache.get(job()) is not None
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_ratio == pytest.approx(0.5)
    assert job() in cache


def test_a_different_program_misses() -> None:
    cache = VerdictCache()
    first = job()
    cache.put(first, verdict_for(first))
    assert cache.get(job(program=_PROGRAM + "\n# a comment\n")) is None


def test_the_same_program_against_different_tests_misses() -> None:
    """The near miss that matters most: a cached pass must not travel to another task."""
    cache = VerdictCache()
    first = job()
    cache.put(first, verdict_for(first))
    assert cache.get(job(expected=999)) is None


def test_the_same_job_under_different_limits_misses() -> None:
    cache = VerdictCache()
    first = job()
    cache.put(first, verdict_for(first))
    assert cache.get(job(timeout=0.5)) is None


def test_storing_a_verdict_for_another_job_is_refused() -> None:
    cache = VerdictCache()
    with pytest.raises(ValueError, match="does not belong to this job"):
        cache.put(job(), verdict_for(job(expected=2)))


def test_putting_the_same_key_twice_is_a_no_op() -> None:
    cache = VerdictCache()
    first = job()
    cache.put(first, verdict_for(first, seconds=0.1))
    cache.put(first, verdict_for(first, seconds=0.9))
    kept = cache.get(first)
    assert kept is not None
    assert kept.duration_seconds == pytest.approx(0.1)
    assert len(cache) == 1


def test_verdicts_persist_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    first = job()
    VerdictCache(path).put(first, verdict_for(first))
    reopened = VerdictCache(path)
    assert len(reopened) == 1
    assert reopened.get(first) is not None
    assert list(reopened.keys()) == [first.cache_key]


def test_a_line_whose_verdict_does_not_rederive_its_key_is_dropped(tmp_path: Path) -> None:
    """A poisoned cache would silently rewrite results, so an inconsistent line is refused."""
    path = tmp_path / "verdicts.jsonl"
    first = job()
    entry = CacheEntry(
        key=first.cache_key,
        program_sha256=first.program_sha256,
        suite_sha256=first.suite_sha256,
        limits_sha256=first.limits.sha256,
        verdict=verdict_for(first),
    )
    assert entry.consistent()
    payload = json.loads(entry.model_dump_json())
    payload["key"] = sha256_hex("not the content address")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    cache = VerdictCache(path)
    assert len(cache) == 0
    assert cache.stats.rejected == 1


def test_a_malformed_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    first = job()
    good = CacheEntry(
        key=first.cache_key,
        program_sha256=first.program_sha256,
        suite_sha256=first.suite_sha256,
        limits_sha256=first.limits.sha256,
        verdict=verdict_for(first),
    )
    path.write_text(
        "not json at all\n\n" + good.model_dump_json() + "\n",
        encoding="utf-8",
    )
    cache = VerdictCache(path)
    assert len(cache) == 1
    assert cache.stats.rejected == 1


def test_an_entry_whose_verdict_names_another_program_is_inconsistent() -> None:
    first = job()
    other = job(program=_PROGRAM + "# other\n")
    entry = CacheEntry(
        key=first.cache_key,
        program_sha256=first.program_sha256,
        suite_sha256=first.suite_sha256,
        limits_sha256=first.limits.sha256,
        verdict=verdict_for(other).model_copy(update={"suite_sha256": first.suite_sha256}),
    )
    assert not entry.consistent()


def test_the_caching_sandbox_runs_once_per_distinct_job() -> None:
    inner = CountingSandbox()
    sandbox: SandboxClient = CachingSandbox(inner)
    sandbox.execute(job())
    sandbox.execute(job())
    sandbox.execute(job(expected=2))
    assert inner.calls == 2
    record = sandbox.record  # type: ignore[attr-defined]
    assert (record.executions, record.cache_hits) == (2, 1)
    assert record.lookups == 3
    assert record.cache_hit_ratio == pytest.approx(1 / 3)


def test_the_caching_sandbox_sums_only_real_execution_time() -> None:
    inner = CountingSandbox()
    sandbox = CachingSandbox(inner)
    sandbox.execute(job())
    before = sandbox.record.seconds
    sandbox.execute(job())  # a hit
    assert sandbox.record.seconds == pytest.approx(before)
    assert before == pytest.approx(0.05)


def test_the_caching_sandbox_forwards_the_readiness_probe() -> None:
    assert CachingSandbox(CountingSandbox()).ping() is True


def test_metrics_record_hits_misses_and_the_ratio() -> None:
    from verifier_guided_search.metrics import create_search_metrics

    metrics = create_search_metrics()
    sandbox = CachingSandbox(CountingSandbox(), metrics=metrics)
    sandbox.execute(job())
    sandbox.execute(job())

    def value(name: str, labels: dict[str, str] | None = None) -> float:
        got = metrics.registry.get_sample_value(name, labels or {})
        return float(got) if got is not None else 0.0

    assert value("vgs_verdict_cache_hits_total") == 1.0
    assert value("vgs_verdict_cache_misses_total") == 1.0
    assert value("vgs_verdict_cache_hit_ratio") == pytest.approx(0.5)
    assert value("vgs_verifier_verdicts_total", {"result": "ok"}) == 2.0


def test_an_empty_cache_reports_a_zero_hit_ratio() -> None:
    cache = VerdictCache()
    assert cache.stats.hit_ratio == 0.0
    assert cache.stats.lookups == 0
    assert CachingSandbox(CountingSandbox()).record.cache_hit_ratio == 0.0


def test_the_cache_key_is_stable_across_processes() -> None:
    """Documented so a persisted cache file stays usable: the key is pure content, no salt."""
    first = job()
    assert first.cache_key == sha256_hex(
        f"{first.program_sha256}:{first.suite_sha256}:{first.limits.sha256}"
    )
    assert len(first.cache_key) == 64
