"""A content-addressed registry for value models: JSON on disk, hash-verified on load.

A model's version *is* the SHA-256 of its card, so the file name and the content cannot drift apart.
:meth:`ValueModelRegistry.load` re-derives the version from the bytes it read and refuses a file
whose content no longer matches its name - which is the whole integrity story, since the registry is
also what ``/readyz`` checks before the service reports itself ready to serve searches.

The index records which versions exist and which one is current, written atomically so a crash
mid-publish leaves the previous pointer intact. No pickle, no server, no network: a directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verifier_guided_search.core.fileio import atomic_write_text, file_lock
from verifier_guided_search.errors import ValueModelError
from verifier_guided_search.value.model import GradientBoostedValueModel, ValueModelCard

__all__ = ["RegistryEntry", "RegistryIndex", "ValueModelRegistry"]

_INDEX_NAME = "index.json"


class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    trained_rows: int = Field(ge=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=1024)


class RegistryIndex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    current: str | None = None
    entries: tuple[RegistryEntry, ...] = ()


class ValueModelRegistry:
    """Versions live at ``<root>/<version>.json``; the pointer lives in ``index.json``."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, version: str) -> Path:
        if len(version) != 64 or any(char not in "0123456789abcdef" for char in version):
            raise ValueModelError(f"not a version hash: {version!r}")
        return self._root / f"{version}.json"

    def index(self) -> RegistryIndex:
        path = self._root / _INDEX_NAME
        if not path.is_file():
            return RegistryIndex()
        try:
            return RegistryIndex.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise ValueModelError(f"registry index is unusable: {type(exc).__name__}") from exc

    def versions(self) -> tuple[str, ...]:
        return tuple(entry.version for entry in self.index().entries)

    @property
    def current_version(self) -> str | None:
        return self.index().current

    def save(
        self,
        model: GradientBoostedValueModel,
        *,
        metrics: Mapping[str, float] | None = None,
        notes: str = "",
        make_current: bool = True,
    ) -> str:
        """Write the card and update the index. Saving the same model twice is a no-op."""
        version = model.version
        payload = model.to_json()
        self._root.mkdir(parents=True, exist_ok=True)
        index_path = self._root / _INDEX_NAME
        with file_lock(index_path):
            atomic_write_text(self.path_for(version), payload + "\n")
            index = self.index()
            entries = list(index.entries)
            if all(entry.version != version for entry in entries):
                entries.append(
                    RegistryEntry(
                        version=version,
                        created_at=datetime.now(UTC).isoformat(),
                        trained_rows=model.card.trained_rows,
                        metrics=dict(metrics or model.card.metrics),
                        notes=notes or model.card.notes,
                    )
                )
            updated = RegistryIndex(
                current=version if make_current else index.current, entries=tuple(entries)
            )
            atomic_write_text(
                index_path,
                json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            )
        return version

    def load(self, version: str | None = None) -> GradientBoostedValueModel:
        """Load a version (the current one by default) and verify its content hash."""
        wanted = version if version is not None else self.current_version
        if wanted is None:
            raise ValueModelError("the registry has no current value model")
        path = self.path_for(wanted)
        if not path.is_file():
            raise ValueModelError(f"value model {wanted[:12]} is not in the registry")
        try:
            card = ValueModelCard.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise ValueModelError(
                f"value model {wanted[:12]} could not be read: {type(exc).__name__}"
            ) from exc
        if card.version != wanted:
            raise ValueModelError(
                f"value model {wanted[:12]} failed its content hash check; the file was modified"
            )
        return GradientBoostedValueModel(card)

    def verify(self, version: str | None = None) -> bool:
        """True when the requested version loads and matches its hash. Used by ``/readyz``."""
        try:
            self.load(version)
        except ValueModelError:
            return False
        return True
