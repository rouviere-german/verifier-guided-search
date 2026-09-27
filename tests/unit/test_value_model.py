"""The value model: the JSON export must be the fitted estimator, to floating-point precision.

The export is the whole reason this repo can ship a gradient-boosted model without a pickle, so the
central test compares :meth:`GradientBoostedValueModel.predict_proba` against
``GradientBoostingClassifier.predict_proba`` on held-out rows. If that agreement ever breaks - a new
scikit-learn leaf encoding, a changed decision-function decomposition - the export raises instead of
silently predicting something else, and :func:`test_a_changed_decomposition_is_refused` pins that
too.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from verifier_guided_search.errors import ValueModelError
from verifier_guided_search.value.features import FEATURE_NAMES
from verifier_guided_search.value.model import (
    MODEL_KIND,
    GradientBoostedValueModel,
    TreeCard,
    ValueModelCard,
    constant_value_model,
)

pytestmark = pytest.mark.unit

_N_FEATURES = len(FEATURE_NAMES)


def make_data(rows: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A learnable but noisy problem over the real feature width."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, _N_FEATURES))
    signal = 1.4 * matrix[:, 0] - 0.9 * matrix[:, 1] + 0.5 * matrix[:, 6]
    labels = (signal + rng.normal(scale=0.6, size=rows) > 0).astype(np.int64)
    if len(set(labels.tolist())) == 1:  # pragma: no cover - guards a pathological seed
        labels[0] = 1 - labels[0]
    return matrix, labels


@pytest.fixture(scope="module")
def fitted() -> tuple[GradientBoostedValueModel, object, np.ndarray]:
    from sklearn.ensemble import GradientBoostingClassifier

    matrix, labels = make_data()
    train, test = matrix[:180], matrix[180:]
    model = GradientBoostedValueModel.train(train, labels[:180], seed=7, n_estimators=40)
    estimator = GradientBoostingClassifier(
        n_estimators=40, max_depth=3, learning_rate=0.1, min_samples_leaf=8, random_state=7
    )
    estimator.fit(train, labels[:180])
    return model, estimator, test


def test_the_export_matches_scikit_learn_on_held_out_rows(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, estimator, test = fitted
    expected = np.asarray(estimator.predict_proba(test))[:, 1]  # type: ignore[attr-defined]
    assert np.max(np.abs(model.predict_proba(test) - expected)) < 1e-9


def test_the_decision_function_matches_scikit_learn(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, estimator, test = fitted
    expected = np.asarray(estimator.decision_function(test)).reshape(-1)  # type: ignore[attr-defined]
    assert np.max(np.abs(model.decision_function(test) - expected)) < 1e-9


def test_a_round_trip_through_json_preserves_predictions_and_version(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, test = fitted
    restored = GradientBoostedValueModel.from_json(model.to_json())
    assert restored.version == model.version
    assert np.array_equal(restored.predict_proba(test), model.predict_proba(test))


def test_the_version_is_the_content_hash_of_the_card(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, _ = fitted
    tweaked = GradientBoostedValueModel(model.card.model_copy(update={"notes": "different"}))
    assert tweaked.version != model.version
    payload = json.loads(model.to_json())
    assert payload["kind"] == MODEL_KIND
    assert payload["feature_names"] == list(FEATURE_NAMES)


def test_score_one_agrees_with_the_batch_path(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, test = fitted
    batch = model.predict_proba(test)
    for index in range(min(8, len(test))):
        assert model.score_one(test[index].tolist()) == pytest.approx(float(batch[index]))


def test_probabilities_stay_in_the_unit_interval_for_extreme_inputs(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, _ = fitted
    extreme = np.array([[1e6] * _N_FEATURES, [-1e6] * _N_FEATURES, [0.0] * _N_FEATURES])
    out = model.predict_proba(extreme)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_a_wrong_feature_width_is_refused(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, _ = fitted
    with pytest.raises(ValueModelError, match="expected feature vectors of length"):
        model.predict_proba([[0.0] * (_N_FEATURES - 1)])


def test_training_needs_both_classes() -> None:
    matrix, _ = make_data(rows=60)
    with pytest.raises(ValueModelError, match="needs both classes"):
        GradientBoostedValueModel.train(matrix, np.zeros(60, dtype=np.int64))


def test_training_checks_the_shapes() -> None:
    matrix, labels = make_data(rows=60)
    with pytest.raises(ValueModelError, match="same number of rows"):
        GradientBoostedValueModel.train(matrix, labels[:10])
    with pytest.raises(ValueModelError, match="does not match the feature names"):
        GradientBoostedValueModel.train(matrix, labels, feature_names=FEATURE_NAMES[:-1])


def test_an_unsupported_kind_is_refused(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, _ = fitted
    card = model.card.model_copy(update={"kind": "some-other-kind"})
    with pytest.raises(ValueModelError, match="unsupported value model kind"):
        GradientBoostedValueModel(card)


def _card(tree: TreeCard) -> ValueModelCard:
    return ValueModelCard(
        feature_names=FEATURE_NAMES,
        init_raw=0.0,
        learning_rate=0.1,
        trees=(tree,),
        trained_rows=1,
        positive_rate=0.5,
        seed=0,
    )


def test_a_structurally_broken_tree_is_refused_on_load() -> None:
    inconsistent = TreeCard(
        feature=(0, -1), threshold=(0.0, 0.0), left=(1, -1), right=(1, -1), value=(0.0,)
    )
    with pytest.raises(ValueModelError, match="inconsistent lengths"):
        GradientBoostedValueModel(_card(inconsistent))

    leaf_with_children = TreeCard(
        feature=(-1,), threshold=(0.0,), left=(0,), right=(-1,), value=(0.0,)
    )
    with pytest.raises(ValueModelError, match="leaf with children"):
        GradientBoostedValueModel(_card(leaf_with_children))

    out_of_range = TreeCard(
        feature=(99, -1, -1),
        threshold=(0.0, 0.0, 0.0),
        left=(1, -1, -1),
        right=(2, -1, -1),
        value=(0.0, 0.1, 0.2),
    )
    with pytest.raises(ValueModelError, match="out of range"):
        GradientBoostedValueModel(_card(out_of_range))

    dangling = TreeCard(
        feature=(0, -1, -1),
        threshold=(0.0, 0.0, 0.0),
        left=(1, -1, -1),
        right=(9, -1, -1),
        value=(0.0, 0.1, 0.2),
    )
    with pytest.raises(ValueModelError, match="points outside the tree"):
        GradientBoostedValueModel(_card(dangling))


def test_a_changed_decomposition_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed ``decision_function`` decomposition must fail training, not skew scores."""
    from sklearn.ensemble import GradientBoostingClassifier

    matrix, labels = make_data(rows=120, seed=3)
    original = GradientBoostingClassifier.decision_function

    def skewed(self: object, X: np.ndarray) -> np.ndarray:  # noqa: N803 - sklearn's signature
        out = np.asarray(original(self, X), dtype=np.float64).reshape(-1)
        return out + np.arange(out.shape[0], dtype=np.float64)  # not a constant offset

    monkeypatch.setattr(GradientBoostingClassifier, "decision_function", skewed)
    with pytest.raises(ValueModelError, match="not init \\+ learning_rate"):
        GradientBoostedValueModel.train(matrix, labels, seed=3, n_estimators=10)


def test_the_constant_baseline_predicts_exactly_its_probability() -> None:
    model = constant_value_model(0.3)
    out = model.predict_proba([[0.0] * _N_FEATURES, [5.0] * _N_FEATURES])
    assert np.allclose(out, 0.3)
    assert model.score_one([1.0] * _N_FEATURES) == pytest.approx(0.3)


def test_the_constant_baseline_rejects_degenerate_probabilities() -> None:
    for value in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueModelError, match="strictly between 0 and 1"):
            constant_value_model(value)


def test_a_one_dimensional_input_is_treated_as_a_single_row(
    fitted: tuple[GradientBoostedValueModel, object, np.ndarray],
) -> None:
    model, _, test = fitted
    assert model.predict_proba(test[0]).shape == (1,)


def test_the_card_carries_the_metrics_it_was_accepted_on() -> None:
    matrix, labels = make_data(rows=120, seed=5)
    model = GradientBoostedValueModel.train(
        matrix, labels, seed=5, n_estimators=10, metrics={"holdout_auc": 0.81}, notes="unit test"
    )
    assert model.card.metrics == {"holdout_auc": 0.81}
    assert model.card.notes == "unit test"
    assert model.card.trained_rows == 120
    assert model.card.positive_rate == pytest.approx(float(labels.mean()))
