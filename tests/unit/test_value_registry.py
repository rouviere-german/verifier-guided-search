"""The value-model registry: a directory whose file names are the content hashes of their contents.

The integrity claim is narrow and worth stating precisely. The registry detects a file that no
longer matches the version in its name, which covers accidental corruption and an edit by anyone who
did not also rename the file. It does **not** detect an attacker who rewrites a model and renames it
to its new hash, then updates the index - that needs a signature, and is recorded as residual risk
in docs/THREAT_MODEL.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verifier_guided_search.errors import ValueModelError
from verifier_guided_search.value.features import FEATURE_NAMES
from verifier_guided_search.value.model import GradientBoostedValueModel, constant_value_model
from verifier_guided_search.value.registry import RegistryIndex, ValueModelRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def registry(tmp_path: Path) -> ValueModelRegistry:
    return ValueModelRegistry(tmp_path / "value_models")


@pytest.fixture
def model() -> GradientBoostedValueModel:
    return constant_value_model(0.4)


def test_an_empty_registry_has_no_current_version(registry: ValueModelRegistry) -> None:
    assert registry.current_version is None
    assert registry.versions() == ()
    assert registry.index() == RegistryIndex()
    assert registry.verify() is False
    with pytest.raises(ValueModelError, match="no current value model"):
        registry.load()


def test_saving_publishes_the_version_and_points_the_index_at_it(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    version = registry.save(model, metrics={"holdout_auc": 0.8}, notes="unit test")
    assert version == model.version
    assert registry.current_version == version
    assert registry.versions() == (version,)
    assert registry.path_for(version).is_file()
    entry = registry.index().entries[0]
    assert entry.metrics == {"holdout_auc": 0.8}
    assert entry.notes == "unit test"
    assert entry.created_at.endswith("+00:00")


def test_a_saved_model_loads_back_identically(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    registry.save(model)
    loaded = registry.load()
    assert loaded.version == model.version
    features = [[0.5] * len(FEATURE_NAMES)]
    assert np.array_equal(loaded.predict_proba(features), model.predict_proba(features))


def test_saving_the_same_model_twice_does_not_duplicate_the_entry(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    registry.save(model)
    registry.save(model)
    assert len(registry.index().entries) == 1


def test_a_second_model_becomes_current_and_the_first_stays_loadable(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    first = registry.save(model)
    second = registry.save(constant_value_model(0.7))
    assert registry.current_version == second
    assert set(registry.versions()) == {first, second}
    assert registry.load(first).version == first


def test_make_current_false_publishes_without_moving_the_pointer(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    first = registry.save(model)
    candidate = registry.save(constant_value_model(0.9), make_current=False)
    assert registry.current_version == first
    assert candidate in registry.versions()
    assert registry.verify(candidate) is True


def test_a_tampered_file_fails_its_content_hash_check(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    version = registry.save(model)
    path = registry.path_for(version)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["notes"] = "edited after publication"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert registry.verify(version) is False
    with pytest.raises(ValueModelError, match="failed its content hash check"):
        registry.load(version)


def test_a_truncated_file_is_reported_rather_than_crashing(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    version = registry.save(model)
    registry.path_for(version).write_text('{"kind": "gbdt-b', encoding="utf-8")
    assert registry.verify(version) is False
    with pytest.raises(ValueModelError, match="could not be read"):
        registry.load(version)


def test_a_missing_version_is_reported_clearly(registry: ValueModelRegistry) -> None:
    missing = "a" * 64
    with pytest.raises(ValueModelError, match="is not in the registry"):
        registry.load(missing)
    assert registry.verify(missing) is False


def test_a_version_that_is_not_a_hash_is_refused(registry: ValueModelRegistry) -> None:
    for bad in ("not-a-hash", "A" * 64, "0" * 63, "../../etc/passwd"):
        with pytest.raises(ValueModelError, match="not a version hash"):
            registry.path_for(bad)


def test_an_unreadable_index_is_reported_not_ignored(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    registry.save(model)
    (registry.root / "index.json").write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(ValueModelError, match="index is unusable"):
        registry.index()


def test_the_index_is_written_atomically_and_stays_human_readable(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    registry.save(model)
    text = (registry.root / "index.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '"current"' in text
    # Indented and key-sorted, so a reviewer can read a diff of the pointer.
    assert "\n  " in text
    assert not list(registry.root.glob("*.tmp"))


def test_verify_is_what_readyz_calls_and_it_never_raises(
    registry: ValueModelRegistry, model: GradientBoostedValueModel
) -> None:
    version = registry.save(model)
    assert registry.verify() is True
    assert registry.verify(version) is True
    registry.path_for(version).unlink()
    assert registry.verify() is False
