"""Content-addressed verdict cache.

The key is the SHA-256 of ``program || test suite || limits`` (:attr:`ExecJob.cache_key`), so a
cached verdict can only be reused for a byte-identical program run against identical tests under
identical limits. Search revisits the same program constantly - four strategies exploring one task's
edit neighbourhood propose overlapping candidates, and a budget sweep replays prefixes - so the hit
ratio is high and executions, not cache lookups, are what the budget pays for.

Persistence is an append-only JSONL file. Each line stores the key and the verdict; on load a line
whose verdict does not re-derive its key is dropped rather than trusted, because a poisoned cache
would silently rewrite every result downstream of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.sandbox.protocol import ExecJob, Verdict

__all__ = ["CacheEntry", "CacheStats", "VerdictCache"]


class CacheEntry(BaseModel):
    """One persisted verdict, tagged with the key it was stored under."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: Verdict

    def consistent(self) -> bool:
        """True when the stored key really is the content address of the stored parts."""
        derived = sha256_hex(f"{self.program_sha256}:{self.suite_sha256}:{self.limits_sha256}")
        return (
            derived == self.key
            and self.verdict.program_sha256 == self.program_sha256
            and self.verdict.suite_sha256 == self.suite_sha256
        )


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    stored: int
    rejected: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class VerdictCache:
    """In-memory map with optional append-only JSONL persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._entries: dict[str, Verdict] = {}
        self._hits = 0
        self._misses = 0
        self._rejected = 0
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = CacheEntry.model_validate_json(line)
            except ValidationError:
                self._rejected += 1
                continue
            if not entry.consistent():
                self._rejected += 1
                continue
            self._entries[entry.key] = entry.verdict

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, job: object) -> bool:
        return isinstance(job, ExecJob) and job.cache_key in self._entries

    def keys(self) -> Iterator[str]:
        return iter(tuple(self._entries))

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits, misses=self._misses, stored=len(self._entries), rejected=self._rejected
        )

    def get(self, job: ExecJob) -> Verdict | None:
        verdict = self._entries.get(job.cache_key)
        if verdict is None:
            self._misses += 1
            return None
        self._hits += 1
        return verdict

    def put(self, job: ExecJob, verdict: Verdict) -> None:
        """Store a verdict. A verdict for a different program or suite is refused."""
        if verdict.program_sha256 != job.program_sha256 or verdict.suite_sha256 != job.suite_sha256:
            raise ValueError("verdict does not belong to this job")
        key = job.cache_key
        if key in self._entries:
            return
        self._entries[key] = verdict
        if self._path is None:
            return
        entry = CacheEntry(
            key=key,
            program_sha256=job.program_sha256,
            suite_sha256=job.suite_sha256,
            limits_sha256=job.limits.sha256,
            verdict=verdict,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(entry.model_dump_json() + "\n")
