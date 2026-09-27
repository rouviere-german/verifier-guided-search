"""Search traces: the rows a search leaves behind, and the training set they become.

One row per evaluated candidate. The features are computed at evaluation time from visible feedback
only; the label - did this program pass every hidden test - is attached by the scorer afterwards.
That ordering is why traces are both the audit record of a run and legitimate training data: the
label was never available to the search that produced the row.

Rows are grouped by task when they are turned into a training set, so the held-out split is *by
task*. Splitting by row would put near-identical neighbours of one program on both sides and inflate
AUC.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.value.features import FEATURE_NAMES

__all__ = ["TraceRow", "TraceStore", "TrainingSet", "split_by_task", "to_training_set"]


class TraceRow(BaseModel):
    """One evaluated candidate: its features, its hidden-test label and its provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    task_id: str
    family: str
    strategy: str
    order: int = Field(ge=0)
    program_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    features: tuple[float, ...] = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    visible_all_passed: bool
    hidden_all_passed: bool
    status: str
    cache_hit: bool = False

    @property
    def label(self) -> int:
        return 1 if self.hidden_all_passed else 0

    @property
    def false_pass(self) -> bool:
        return self.visible_all_passed and not self.hidden_all_passed


class TrainingSet:
    """Feature rows, labels and the task each row came from."""

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        task_ids: Sequence[str],
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        if not (len(features) == len(labels) == len(task_ids)):
            raise ValueError("features, labels and task_ids must have the same length")
        self.features = [list(row) for row in features]
        self.labels = list(labels)
        self.task_ids = list(task_ids)
        self.feature_names = tuple(feature_names)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def positive_rate(self) -> float:
        return sum(self.labels) / len(self.labels) if self.labels else 0.0

    @property
    def tasks(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for task_id in self.task_ids:
            seen.setdefault(task_id, None)
        return tuple(seen)

    def select(self, task_ids: Iterable[str]) -> TrainingSet:
        wanted = set(task_ids)
        keep = [index for index, task_id in enumerate(self.task_ids) if task_id in wanted]
        return TrainingSet(
            [self.features[index] for index in keep],
            [self.labels[index] for index in keep],
            [self.task_ids[index] for index in keep],
            feature_names=self.feature_names,
        )

    def both_classes_present(self) -> bool:
        return len(set(self.labels)) == 2


def to_training_set(rows: Iterable[TraceRow]) -> TrainingSet:
    """Deduplicate by (task, program) - the same program may be evaluated by every strategy."""
    seen: set[tuple[str, str]] = set()
    features: list[list[float]] = []
    labels: list[int] = []
    task_ids: list[str] = []
    names: tuple[str, ...] = FEATURE_NAMES
    for row in rows:
        key = (row.task_id, row.program_sha256)
        if key in seen:
            continue
        seen.add(key)
        features.append(list(row.features))
        labels.append(row.label)
        task_ids.append(row.task_id)
        names = row.feature_names
    return TrainingSet(features, labels, task_ids, feature_names=names)


def split_by_task(
    data: TrainingSet, *, holdout_fraction: float = 0.3, seed: int = 0
) -> tuple[TrainingSet, TrainingSet]:
    """Deterministic task-level split: a task's hash decides which side it lands on."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    tasks = sorted(data.tasks)
    if len(tasks) < 2:
        raise ValueError("splitting by task needs at least two tasks")
    ranked = sorted(tasks, key=lambda task_id: sha256_hex(f"{seed}:{task_id}"))
    cut = max(1, min(len(ranked) - 1, round(len(ranked) * holdout_fraction)))
    holdout = set(ranked[:cut])
    train = [task_id for task_id in tasks if task_id not in holdout]
    return data.select(train), data.select(holdout)


class TraceStore:
    """Append-only JSONL of trace rows. Malformed lines are skipped, not trusted."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._rows: list[TraceRow] = []
        self._skipped = 0
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                self._rows.append(TraceRow.model_validate_json(line))
            except ValidationError:
                self._skipped += 1

    @property
    def rows(self) -> tuple[TraceRow, ...]:
        return tuple(self._rows)

    @property
    def skipped(self) -> int:
        return self._skipped

    def __len__(self) -> int:
        return len(self._rows)

    def append(self, row: TraceRow) -> None:
        self._rows.append(row)
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(row.model_dump_json() + "\n")

    def training_set(self) -> TrainingSet:
        return to_training_set(self._rows)
