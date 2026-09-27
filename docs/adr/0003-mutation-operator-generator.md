# 3. Make the offline candidate generator an explicit edit space, and state what that costs

Date: 2026-09-27

## Status

Accepted.

## Context

The repo's claim is about *search*: that spending a fixed inference budget on a tree search over
candidate programs beats spending it on simpler strategies. Testing that claim needs a candidate
generator, and the generator must work offline - the whole suite, demo and benchmark run with no API
key on a CPU-only machine.

A generator has to produce candidates that are (a) syntactically valid, (b) related to the current
program, and (c) *sometimes* right, with a controllable success rate. Without (c) there is nothing to
search: at a 100% per-step success rate every strategy solves everything in `d` steps, and at 0% none
of them ever solve anything. Either way the comparison is empty.

## Decision

Parse each task's reference solution and **enumerate its edit sites**: every integer literal that can
be shifted by one, every comparison, boolean or arithmetic operator that can be swapped, every
non-commutative operand pair or two-argument call that can be transposed, every slice boundary that
can move, every `if` whose branch can be dropped. A program is then a *subset of applied sites* - a
bit vector - and rendering re-parses the reference, applies the selected edits and unparses.

A task's starting program is the reference with `d` sites applied. The generator scores every
single-site flip with a fixed advantage for reverting an applied site over introducing a new one, adds
Gaussian noise, and returns the top `k`. Priors come from rank, not from the noisy score, so the prior
does not depend on the scale of whatever the generator scores with.

Two parameters set the regime, and both are chosen by measurement. `bench/bench_calibration.py`
sweeps them and writes `bench/results/calibration.json`; the numbers below are from that file (six
families, one instance each, all four strategies, budget 16, value model off, so the dial is the only
thing moving). "Spread" is the best strategy's hidden solve rate minus the worst one's: it is what
collapses when a dial leaves the informative region, in either direction.

| noise | greedy | best_of_n | beam | mcts | spread |
|---:|---:|---:|---:|---:|---:|
| 0 | 100.0% | 33.3% | 83.3% | 83.3% | 0.667 |
| 0.25 | 83.3% | 33.3% | 83.3% | 83.3% | 0.500 |
| **0.55** | 33.3% | 33.3% | 66.7% | 66.7% | **0.333** |
| 1.5 | 16.7% | 33.3% | 33.3% | 33.3% | 0.167 |

| corruption depth | greedy | best_of_n | beam | mcts | spread |
|---:|---:|---:|---:|---:|---:|
| 1 | 66.7% | 83.3% | 83.3% | 83.3% | 0.167 |
| **2** | 33.3% | 33.3% | 66.7% | 66.7% | **0.333** |
| 3 | 33.3% | 16.7% | 33.3% | 33.3% | 0.167 |

- **noise = 0.55.** At 0 the generator is an oracle - greedy solves everything, and *greedy beats the
  searches*, because with a perfect signal backtracking is wasted budget. At 1.5 every strategy is
  down at 29% on average and indistinguishable. At 0.55 the two search strategies are 33 points
  above the two baselines. Unit tests pin the two ends as properties of the generator itself
  (`test_zero_noise_makes_the_generator_an_oracle`,
  `test_large_noise_destroys_the_revert_signal`) and that the default sits strictly between them.
- **corruption depth = 2.** At depth 1 best-of-N does as well as the searches (83.3% each): a
  single-edit bug needs no search. At depth 3 everything sits at 29% on average. Depth 2 is the one
  of the three where the strategies separate, and `--check` puts a floor on that spread so the choice
  cannot quietly stop holding.

## Consequences

**Good.** The search space has real structure: a corrupted program has `d` sites applied, one edit away
are `|sites|` neighbours, and exactly one of `2**|sites|` programs is the reference. Moving one bit is a
meaningful step, so a strategy that spends its budget well finds the reference sooner than one that
does not - which is precisely the thing under test. Everything is deterministic given a seed, every
mutant is valid Python, and the whole benchmark builds in about 30 ms.

**Bad, and this is the important part.** The offline curves describe search over an *edit space*, not
over a language model's output distribution. Specifically:

1. The neighbourhood is **local and finite**, and the reference is **always reachable**. A model's
   output distribution has neither property.
2. **best-of-N is under-represented.** In this framing "one sample" is one edit from the current
   program, so the best-of-N strategy - which only ever expands the root - is confined to depth 1 and
   *structurally cannot* repair a two-edit bug. Its 11.1% in the results table is therefore a property
   of the framing, not a measurement of best-of-N over whole programs. A test asserts the structural
   fact (`test_best_of_n_never_leaves_depth_one`) so the reason is visible rather than inferred.
3. **Depth 2 was chosen because it separates the strategies.** That is a legitimate way to pick an
   operating point for a comparison, and it is also the kind of choice that can be used to manufacture
   a result. It is recorded here, in the constant's own docstring, and in the README Non-goals.

The honest summary: the offline benchmark shows that *on a search problem with this structure*, depth
beats breadth and backtracking beats committing. It does not produce a number that transfers to a real
coding agent. The live `ClaudeGenerator` shares the prompt, the budget accounting and the verifier, so
the machinery transfers even though the numbers do not.

## Alternatives considered

**Model whole-program resampling instead.** Give each candidate an independent per-site decision, so a
sample is a fresh program rather than a local edit. Rejected after working through the arithmetic: with
a principled threshold (the midpoint of the two advantages) a sample gets each of about 24 sites right
with probability about 0.75, so a fully correct program arrives with probability 0.75 to the power 24,
roughly 0.001 - and best-of-N is hopeless for a different and less interesting reason. Any threshold
that makes it competitive is a free parameter tuned to produce a desired answer, which is worse than
the current framing's stated limitation.

**Ship cassettes of real model output.** Genuinely better data, and the recording needs a key and
money. Left as future work; the provider abstraction and the `ReplayProvider` are already in place for
it.

**Use a public benchmark (HumanEval, MBPP).** Right for a paper, wrong here: the tasks are fixed, so
there is no visible/hidden split to construct, and contamination makes the absolute numbers
meaningless. The parametric families exist so the hidden group can hold edge cases the visible group
does not - which is what makes the verifier false-pass rate measurable at all.
