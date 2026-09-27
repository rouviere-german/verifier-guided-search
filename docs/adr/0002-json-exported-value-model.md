# 2. Export the value model to JSON instead of pickling the estimator

Date: 2026-09-27

## Status

Accepted.

## Context

The search needs a learned value model: given a candidate's visible feedback and its position in the
tree, predict the probability that it passes the hidden tests. Gradient-boosted trees are the right
tool - the features are few, tabular and heterogeneous, the dataset is hundreds of rows, and training
has to finish in seconds on a CPU. scikit-learn's `GradientBoostingClassifier` is the obvious
implementation.

The obvious way to *ship* it is `pickle` (or `joblib`, which is pickle). That is what almost every
tutorial does.

It is also the one thing this repo cannot do. Loading a pickle executes whatever the file says. This
service exists to run untrusted programs; a model artifact sitting on disk next to a sandbox is
exactly the file an attacker would like us to deserialise, and "we load a pickle from our own
registry" is only as strong as write access to that registry. A pickle is also opaque: it cannot be
diffed, reviewed, or content-addressed in a way a human can check.

## Decision

Train with scikit-learn, then **export the fitted ensemble to flat JSON arrays** and score it with
about twenty lines of numpy. Inference has no scikit-learn import at all.

A `ValueModelCard` holds, per stage tree, the feature index, threshold, left and right child indices
and leaf values, plus the initial raw score and the learning rate. `predict_proba` walks the trees
with numpy and applies a numerically stable logistic.

Two details make this trustworthy rather than merely pickle-free:

**The export is verified against the estimator.** `train` compares
`GradientBoostedValueModel.predict_proba` with `GradientBoostingClassifier.predict_proba` on the
training rows and raises if they differ by more than 1e-9. A test repeats the comparison on held-out
rows.

**The raw-score offset is measured, not read from a private attribute.** A GBDT's decision function is
`init + learning_rate * sum(trees)`. Rather than reaching into a private attribute for `init`, `train`
evaluates `decision_function`, subtracts the tree sum, and requires the remainder to be *constant* to
1e-9. If a future scikit-learn changes that decomposition the export fails loudly instead of
predicting something subtly wrong. `test_a_changed_decomposition_is_refused` monkeypatches
`decision_function` to prove the guard fires.

## Consequences

**Good.** No arbitrary-code-execution path through a model file. The artifact is a JSON document whose
SHA-256 *is* its version, so the registry can refuse a file that no longer matches its own name and
`/readyz` can check that before the service reports ready. A reviewer can read a diff of a model. The
inference path has one dependency (numpy) instead of the whole of scikit-learn, which also keeps the
runtime image smaller.

**Bad.** The export only covers a binary `GradientBoostingClassifier`. Switching to a random forest,
LightGBM, or multi-class would need new export code, and the `kind` field exists so that an
unsupported card is refused rather than mis-scored. There is a real coupling to scikit-learn's tree
internals (`tree_.children_left`, `tree_.feature`, `tree_.value`), which the leaf-sentinel
normalisation and the verification step contain but do not remove.

**Cost.** Roughly 150 lines of export and inference code that a `joblib.dump` would have replaced with
one. That is the price of not having a deserialisation sink in a repo built around running untrusted
code, and it is worth paying.

## Alternatives considered

**ONNX.** Solves the same problem properly and is the right answer at a larger scale. Rejected here:
`skl2onnx` plus `onnxruntime` is a large dependency pair for a twelve-feature model, and the artifact
becomes a binary blob again - reviewable by tooling, not by a person.

**`safetensors`.** Designed for tensors, not for tree structure; we would end up inventing the same
flat-array layout and storing it in a less readable container.

**Keep the pickle but sign it.** Moves the problem to key management and still executes code on load
if the signature check is ever skipped or misconfigured. A format that *cannot* execute code is
stronger than a format that usually is not allowed to.

**Reimplement the training too.** Then there is no reference implementation to verify the export
against, which is exactly the check that makes this design defensible.
