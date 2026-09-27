"""The learned value model: trained with scikit-learn, shipped and evaluated as JSON.

The model predicts, from a candidate's *visible* outcome and its position in the tree, the
probability that the candidate passes the *hidden* tests. MCTS uses that probability as the value of
a leaf, and beam uses it to rank a frontier, so a better value model turns into better use of the
same budget.

**Why JSON and not a pickle.** Loading a pickled estimator executes whatever the file says, and this
repo runs untrusted programs for a living; a model artifact is exactly the kind of file an attacker
would like us to deserialise. So training stays in scikit-learn, and the fitted ensemble is then
exported to plain arrays - per tree: feature index, threshold, children, leaf values - and scored by
:meth:`GradientBoostedValueModel.predict_proba`, about 20 lines of numpy. Inference has no
scikit-learn import at all. ``tests/unit/test_value_model.py`` asserts the exported model matches
``GradientBoostingClassifier.predict_proba`` to 1e-9, which is what makes the export trustworthy
rather than merely pickle-free. See docs/adr/0002-json-exported-value-model.md.

**The raw-score offset.** A gradient-boosted classifier's decision function is
``init + learning_rate * sum(tree(x))``. Rather than reaching into a private attribute for ``init``,
:meth:`train` measures it: it evaluates ``decision_function`` on the training rows, subtracts the
tree sum, and requires the remainder to be constant to 1e-9. If a future scikit-learn changes that
decomposition the export fails loudly instead of predicting something subtly different.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.core.canonical import canonical_json, sha256_hex
from verifier_guided_search.errors import ValueModelError
from verifier_guided_search.value.features import FEATURE_NAMES

__all__ = [
    "MODEL_KIND",
    "GradientBoostedValueModel",
    "TreeCard",
    "ValueModelCard",
    "constant_value_model",
]

MODEL_KIND: Final = "gbdt-binary-logodds"
_EXPORT_TOLERANCE: Final = 1e-9
_LEAF: Final = -1


class TreeCard(BaseModel):
    """One regression tree as flat arrays.

    The card's own convention, normalised on export: a leaf has ``feature == -1`` and ``left ==
    right == -1``. scikit-learn marks leaves differently (feature ``-2``, children ``-1``), so
    :func:`_export_trees` rewrites them rather than storing the library's sentinels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: tuple[int, ...] = Field(min_length=1)
    threshold: tuple[float, ...] = Field(min_length=1)
    left: tuple[int, ...] = Field(min_length=1)
    right: tuple[int, ...] = Field(min_length=1)
    value: tuple[float, ...] = Field(min_length=1)

    def validate_shape(self, n_features: int) -> None:
        sizes = {
            len(self.feature),
            len(self.threshold),
            len(self.left),
            len(self.right),
            len(self.value),
        }
        if len(sizes) != 1:
            raise ValueModelError("tree arrays have inconsistent lengths")
        size = len(self.feature)
        for index in range(size):
            column = self.feature[index]
            if column == _LEAF:
                if self.left[index] != _LEAF or self.right[index] != _LEAF:
                    raise ValueModelError(f"tree node {index} is a leaf with children")
                continue
            if not 0 <= column < n_features:
                raise ValueModelError(f"tree node {index} uses feature {column} out of range")
            for child in (self.left[index], self.right[index]):
                if not 0 <= child < size:
                    raise ValueModelError(f"tree node {index} points outside the tree")


class ValueModelCard(BaseModel):
    """The serialisable model: hyper-parameters, the trees, and the metrics it was accepted on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    kind: str = Field(default=MODEL_KIND, pattern=r"^[a-z0-9-]{1,48}$")
    feature_names: tuple[str, ...] = Field(min_length=1)
    init_raw: float
    learning_rate: float = Field(gt=0, le=10)
    trees: tuple[TreeCard, ...] = Field(min_length=1)
    trained_rows: int = Field(ge=1)
    positive_rate: float = Field(ge=0, le=1)
    seed: int
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=1024)

    @property
    def version(self) -> str:
        """Content address of the card: identity and integrity in one value."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class GradientBoostedValueModel:
    """Scores feature vectors from a :class:`ValueModelCard`. No scikit-learn at inference time."""

    def __init__(self, card: ValueModelCard) -> None:
        if card.kind != MODEL_KIND:
            raise ValueModelError(f"unsupported value model kind {card.kind!r}")
        for tree in card.trees:
            tree.validate_shape(len(card.feature_names))
        self._card = card
        self._n_features = len(card.feature_names)
        self._feature = [np.asarray(tree.feature, dtype=np.int64) for tree in card.trees]
        self._threshold = [np.asarray(tree.threshold, dtype=np.float64) for tree in card.trees]
        self._left = [np.asarray(tree.left, dtype=np.int64) for tree in card.trees]
        self._right = [np.asarray(tree.right, dtype=np.int64) for tree in card.trees]
        self._value = [np.asarray(tree.value, dtype=np.float64) for tree in card.trees]

    @property
    def card(self) -> ValueModelCard:
        return self._card

    @property
    def version(self) -> str:
        return self._card.version

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._card.feature_names

    def _as_matrix(
        self, features: Sequence[Sequence[float]] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self._n_features:
            raise ValueModelError(
                f"expected feature vectors of length {self._n_features}, got shape {matrix.shape}"
            )
        return matrix

    def decision_function(
        self, features: Sequence[Sequence[float]] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Raw log-odds: ``init + learning_rate * sum over trees``."""
        matrix = self._as_matrix(features)
        raw = np.full(matrix.shape[0], self._card.init_raw, dtype=np.float64)
        for index in range(len(self._feature)):
            raw += self._card.learning_rate * self._tree_predict(index, matrix)
        return raw

    def _tree_predict(self, tree_index: int, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        feature = self._feature[tree_index]
        threshold = self._threshold[tree_index]
        left = self._left[tree_index]
        right = self._right[tree_index]
        value = self._value[tree_index]
        nodes = np.zeros(matrix.shape[0], dtype=np.int64)
        active = feature[nodes] != _LEAF
        # Each iteration descends one level; the loop ends when every row has reached a leaf.
        while bool(active.any()):
            rows = np.nonzero(active)[0]
            current = nodes[rows]
            go_left = matrix[rows, feature[current]] <= threshold[current]
            nodes[rows] = np.where(go_left, left[current], right[current])
            active = feature[nodes] != _LEAF
        return value[nodes]

    def predict_proba(
        self, features: Sequence[Sequence[float]] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Probability of the positive class (the candidate passes every hidden test)."""
        raw = self.decision_function(features)
        # Numerically stable logistic: exp is only ever applied to non-positive values.
        out = np.empty_like(raw)
        positive = raw >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
        exp_raw = np.exp(raw[~positive])
        out[~positive] = exp_raw / (1.0 + exp_raw)
        return out

    def score_one(self, features: Sequence[float]) -> float:
        return float(self.predict_proba([list(features)])[0])

    def to_json(self) -> str:
        return canonical_json(self._card.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: str) -> GradientBoostedValueModel:
        return cls(ValueModelCard.model_validate_json(payload))

    @classmethod
    def train(
        cls,
        features: Sequence[Sequence[float]] | NDArray[np.float64],
        labels: Sequence[int] | NDArray[np.int64],
        *,
        seed: int = 0,
        n_estimators: int = 80,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 8,
        feature_names: Sequence[str] = FEATURE_NAMES,
        metrics: Mapping[str, float] | None = None,
        notes: str = "",
    ) -> GradientBoostedValueModel:
        """Fit a scikit-learn classifier and export it. Requires both classes to be present."""
        from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415 - training only

        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] != target.shape[0]:
            raise ValueModelError("features and labels must have the same number of rows")
        if matrix.shape[1] != len(feature_names):
            raise ValueModelError("feature matrix does not match the feature names")
        if set(np.unique(target).tolist()) != {0, 1}:
            raise ValueModelError(
                "training needs both classes; collect more traces or widen the budget"
            )
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        estimator.fit(matrix, target)
        trees = _export_trees(estimator)
        init_raw = _measure_init_raw(
            estimator, matrix, trees, learning_rate, feature_names=tuple(feature_names)
        )
        card = ValueModelCard(
            feature_names=tuple(feature_names),
            init_raw=init_raw,
            learning_rate=learning_rate,
            trees=trees,
            trained_rows=int(matrix.shape[0]),
            positive_rate=float(target.mean()),
            seed=seed,
            metrics=dict(metrics or {}),
            notes=notes,
        )
        model = cls(card)
        _verify_export(model, estimator, matrix)
        return model


def _export_trees(estimator: Any) -> tuple[TreeCard, ...]:
    """Flatten every stage's regression tree, normalising scikit-learn's leaf sentinels."""
    cards: list[TreeCard] = []
    for stage in estimator.estimators_:
        tree = stage[0].tree_
        left_raw = [int(item) for item in tree.children_left]
        right_raw = [int(item) for item in tree.children_right]
        feature_raw = [int(item) for item in tree.feature]
        threshold_raw = [float(item) for item in tree.threshold]
        values = [float(item) for item in tree.value.reshape(-1)]
        leaf = [index for index, child in enumerate(left_raw) if child < 0]
        leaves = set(leaf)
        cards.append(
            TreeCard(
                feature=tuple(
                    _LEAF if index in leaves else feature_raw[index]
                    for index in range(len(feature_raw))
                ),
                threshold=tuple(
                    0.0 if index in leaves else threshold_raw[index]
                    for index in range(len(threshold_raw))
                ),
                left=tuple(
                    _LEAF if index in leaves else left_raw[index] for index in range(len(left_raw))
                ),
                right=tuple(
                    _LEAF if index in leaves else right_raw[index]
                    for index in range(len(right_raw))
                ),
                value=tuple(values),
            )
        )
    if not cards:
        raise ValueModelError("the fitted estimator has no trees")
    return tuple(cards)


def _measure_init_raw(
    estimator: Any,
    matrix: NDArray[np.float64],
    trees: Sequence[TreeCard],
    learning_rate: float,
    *,
    feature_names: tuple[str, ...],
) -> float:
    """Recover the constant initial raw score without touching scikit-learn internals."""
    probe = GradientBoostedValueModel(
        ValueModelCard(
            feature_names=feature_names,
            init_raw=0.0,
            learning_rate=learning_rate,
            trees=tuple(trees),
            trained_rows=int(matrix.shape[0]),
            positive_rate=0.0,
            seed=0,
        )
    )
    reference = np.asarray(estimator.decision_function(matrix), dtype=np.float64).reshape(-1)
    residual = reference - probe.decision_function(matrix)
    spread = float(residual.max() - residual.min())
    if spread > _EXPORT_TOLERANCE:
        raise ValueModelError(
            "the estimator's decision function is not init + learning_rate * sum(trees) "
            f"(residual spread {spread:.3g}); the JSON export would not be faithful"
        )
    return float(residual.mean())


def _verify_export(
    model: GradientBoostedValueModel, estimator: Any, matrix: NDArray[np.float64]
) -> None:
    expected = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)[:, 1]
    largest = float(np.max(np.abs(model.predict_proba(matrix) - expected)))
    if largest > _EXPORT_TOLERANCE:
        raise ValueModelError(
            f"exported model disagrees with the fitted estimator by {largest:.3g}"
        )


def constant_value_model(
    probability: float, *, feature_names: Sequence[str] = FEATURE_NAMES
) -> GradientBoostedValueModel:
    """A single-leaf model that always predicts ``probability``. Useful as a calibration
    baseline."""
    if not 0.0 < probability < 1.0:
        raise ValueModelError("probability must be strictly between 0 and 1")
    odds = float(np.log(probability / (1.0 - probability)))
    card = ValueModelCard(
        feature_names=tuple(feature_names),
        init_raw=odds,
        learning_rate=1.0,
        trees=(TreeCard(feature=(-1,), threshold=(-2.0,), left=(-1,), right=(-1,), value=(0.0,)),),
        trained_rows=1,
        positive_rate=probability,
        seed=0,
        notes="constant baseline",
    )
    return GradientBoostedValueModel(card)
