# 4. Keep the value model although it does not improve the search, and report why

Date: 2026-09-27

## Status

Accepted, with a documented negative result.

## Context

PUCT needs a value for each node: `score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))`. The
spec calls for a learned value model - a gradient-boosted classifier over visible feedback and tree
position, predicting hidden-test success, with a held-out AUC of at least 0.75 - and for a
value-model-on/off ablation.

The model was built, trained on traces from a disjoint set of tasks, and measured. It reaches a
**held-out AUC of 0.952** and an ECE of 0.081: comfortably past the target, and by the usual standards
a good model.

Then the ablation was run. At budget 8 the hidden solve rate is identical with the model and without
it. At budget 16, MCTS scores **61.1% with the model and 66.7% without it**. The model does not help;
if anything it hurts slightly.

## Decision

**Keep the model, report the negative result prominently, and do not paper over it.** Specifically:

- the default stays `use_value_model=True`, because the spec's architecture calls for it and because
  the ablation's sample (11-18 tasks) cannot settle a difference this small;
- the README Results section states the on/off numbers and says plainly that the model does not help;
- the cause is diagnosed rather than left as noise;
- `c_puct` stays at the literature value of 1.4 even though the ablation prefers 0.0, because
  retuning it on the evaluation tasks would be selecting on the test set.

## The cause

Visible in the raw scores. Across one run the model's outputs are approximately:

```
0.0001372  0.0001372  0.0001372  0.0001372  0.0001372  0.0001262  ...  0.8968
```

It assigns about 1e-4 to essentially every candidate and about 0.9 to a solving one. That is an
excellent *ranking* - hence the AUC - and a useless *search signal*, because PUCT does not need to know
which candidate is the solution. It needs to order the candidates that are **not** solutions, to decide
where to descend next. The model collapses all of them onto the same value.

The alternative signal, the visible pass fraction, is coarse (0, 1/3, 2/3, 1 on a three-test group) but
*graded*: it distinguishes a candidate that passes two tests from one that passes none. For guiding a
descent that is worth more than a sharp, saturated probability.

Stated generally: **a value model selected for discrimination is not automatically a good search
heuristic.** AUC measures the separation of two classes; PUCT needs resolution *within* the negative
class. Those are different objectives, and optimising the first does not deliver the second.

The same observation explains the `c_puct` result. With a value signal this flat, exploration has
nothing to trade off against, so pure exploitation of the graded pass fraction does better.

## Consequences

**Good.** The repo reports a measured negative result about its own centrepiece, with a mechanism
rather than a shrug. That is more informative than a small positive number would have been, and the
diagnosis is reusable: anyone wiring a learned value function into a tree search should check the
*distribution* of its outputs over non-terminal nodes, not just its AUC.

**Bad.** The learned component currently earns its place in the architecture and not in the results. A
reader could reasonably conclude the value model should be deleted. It is kept because it is what the
spec asks for, because it makes the registry, the disjoint-task training split and the on/off ablation
real, and because the fix is known rather than hypothetical.

**Not done** (recorded in the README Non-goals):

- **blend the signals**: `value = a * model_probability + (1 - a) * visible_pass_fraction`, which keeps
  the model's endgame sharpness and restores mid-search resolution;
- **calibrate**: isotonic regression or Platt scaling on the held-out split, then re-run the ablation.
  An ECE of 0.081 is not bad overall but says nothing about resolution inside the negative class;
- **train a ranker instead of a classifier**, with a pairwise loss over sibling candidates, which is
  the objective the search actually wants;
- **more data**: 169 trace rows over 12 tasks is small, and a model this saturated is partly a symptom
  of learning from few positives.

Each of those changes the reported numbers, so each needs the full bench re-run rather than a
plausible-sounding paragraph.

## Alternatives considered

**Turn the model off by default and report the better number.** Rejected. 66.7% against 61.1% on 18
tasks is inside the confidence intervals, so picking the default that produces the higher headline is
selecting on the test set - the same mistake as retuning `c_puct`.

**Quietly blend the signals and re-run until the model helps.** Rejected for the same reason. A blend
is a defensible design, but arriving at it by searching for a configuration that makes the learned
component look good is how meaningless results get produced. If it is implemented later it should be
justified by the saturation diagnosis above and measured once.

**Drop the value model and the registry entirely.** It would simplify the repo, and it would also
remove the content-addressed artifact integrity, the disjoint-task training split and the on/off
ablation - three things the repo demonstrates properly. The negative result is a finding, not a
failure.
