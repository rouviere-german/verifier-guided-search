# verifier-guided-search

Test-time compute for coding agents: tree search over candidate programs, guided by sandboxed test
runs and a learned value model, compared against simpler strategies at an identical compute budget.

Everything runs offline on one CPU. The test suite, the demo and the benchmarks need no API key, no
network and no Docker.

## Problem

Single-sample code generation is unreliable, so systems spend more inference compute and keep the
best attempt. Three things usually go wrong when they do:

**Naive best-of-N spends its budget evenly on hopeless branches.** Every sample starts from scratch,
so nothing learned from one failure informs the next.

**Search overfits the tests it can see.** A candidate that satisfies the visible tests is accepted,
and nobody finds out that it was wrong until later. On this benchmark that happens to **22% of
tasks** at every budget level and for every strategy - a number that only exists because the
benchmark keeps a second, hidden test group the search never sees.

**Comparisons are not compute-matched.** "Our search beats sampling" means nothing unless both ran
under the same budget, and a budget is only meaningful if something enforces it. Here a budget
accountant reserves the worst case before each step and settles the actual cost afterwards, so a
single step can never overshoot ([property test](tests/unit/test_budget_accountant.py)).

And underneath all of it: **the candidates are untrusted code that has to be executed**. This repo
treats that as the primary engineering problem, not a detail.

## Architecture

```mermaid
flowchart TB
    subgraph bench["Benchmark (150 tasks, 15 parametric families)"]
        ref["Reference solution<br/>(template + instance parameters)"] --> space["MutationSpace<br/>enumerated edit sites"]
        space --> start["Start program<br/>(2 edits applied)"]
        ref --> vis["Visible tests<br/>random draws"]
        ref --> hid["Hidden tests<br/>edge cases + draws"]
    end

    subgraph search["Search service (holds the credentials, runs no candidate code)"]
        strat{{"Strategy<br/>greedy · best-of-N · beam · MCTS"}}
        strat --> gen["Generator<br/>seeded mutations (offline)<br/>claude-sonnet-5 (live)"]
        strat --> puct["PUCT selection<br/>Q + c·P·√N/(1+n)"]
        puct --> value["Value model<br/>GBDT, exported as JSON"]
        strat --> acct[("Budget accountant<br/>reserve → settle")]
        strat --> trace[("Search traces")]
    end

    subgraph runner["Runner container — network internal: true, no egress"]
        api["POST /v1/execute"] --> child["Fresh interpreter per program<br/>python -I -S"]
        child --> guard["setrlimit · watchdog<br/>sys.addaudithook guard"]
        child --> jail[("tmpfs jail<br/>the only writable path")]
    end

    start --> strat
    vis --> strat
    strat -- "ExecJob (JSON)" --> api
    api -- "Verdict" --> cache[("Verdict cache<br/>keyed by sha256(program‖tests‖limits)")]
    cache --> strat
    hid -. "scoring only, never to the search" .-> score["Hidden-test scorer"]
    strat --> score
    trace --> train["Train value model<br/>on disjoint tasks"]
    train --> registry[("Content-addressed<br/>model registry")]
    registry --> value
    score --> patch["Winning patch"]
    patch --> gate{{"Approval gate<br/>bound to the diff hash"}}
    gate -- "a different principal approves" --> ws[("Workspace")]
    gate --> audit[("Hash-chained audit")]
```

Three design decisions carry most of the weight.

**The visible/hidden split is enforced by types, not discipline.** A sandbox run produces a
`Verdict`, which holds both test groups. Everything downstream of the search - the generator, the
value-model features, the node scores - takes a `VisibleFeedback`, which is a projection with the
hidden group and all free text removed. "The search cannot see the hidden tests" is therefore a
property of the signatures ([`sandbox/protocol.py`](src/verifier_guided_search/sandbox/protocol.py)),
and [a property test](tests/unit/test_value_features.py) confirms that permuting every hidden result
leaves the feature vector byte-identical.

That covers the search. The *candidate* is a second reader, and types cannot help there: the harness
and the candidate share one interpreter, so a program can walk `sys._getframe` into the harness's
locals and read whatever the harness was given. A review found exactly that - a candidate that
harvested the job's expected values and returned them from a dict passed every hidden test with no
algorithm at all, and no guard bypass was needed. So the expectations no longer enter the sandbox:
the child is sent each case's *arguments*, reports what the entry point returned, and the parent
compares outputs with expectations
([`grade_group`](src/verifier_guided_search/sandbox/protocol.py),
[regression test](tests/integration/test_sandbox_local.py)). The child holds nothing worth stealing,
which also takes most of the value out of forging a verdict.

**A smaller budget is a prefix of a larger one.** Every strategy is a generator that yields one
evaluation at a time and never inspects the remaining budget, so a run capped at `B` candidates
produces exactly the first `B` evaluations of a longer run. The sweep therefore runs each
(task, strategy) pair once at the top level and reads the five levels off as prefixes: five times
less compute, and every level compared on identical work.
[`tests/integration/test_budget_prefix.py`](tests/integration/test_budget_prefix.py) re-runs the
search with a real smaller accountant and asserts the evaluations match, for all four strategies.

**The model artifact is JSON, not a pickle.** This repo executes untrusted programs for a living; a
pickled estimator is exactly the file an attacker would like it to deserialise. Training uses
scikit-learn, then the fitted ensemble is exported to flat arrays and scored by ~20 lines of numpy.
[A test asserts the export matches `GradientBoostingClassifier.predict_proba` to 1e-9](tests/unit/test_value_model.py),
which is what makes the export trustworthy rather than merely pickle-free.

## Quickstart

```bash
# Python 3.12 and uv. Keep the environment off a small disk if you need to:
export UV_PROJECT_ENVIRONMENT=~/.venvs/verifier-guided-search
uv sync

uv run python tasks.py demo        # the whole thing, offline, ~90s
uv run python tasks.py test        # 812 passing, 2 environment-skipped, coverage floor 85%
uv run python tasks.py bench       # the micro-benchmarks and the evaluation below, ~3 minutes
uv run python tasks.py calibrate   # why the two difficulty dials are where they are, ~90s
uv run python tasks.py lint typecheck audit baseline
```

The demo builds the benchmark, collects traces, trains and publishes a value model, runs all four
strategies at matched budgets, runs the sandbox escape suite, and then proposes applying the winning
program to a scratch workspace. That last step is the one worth watching: the apply is **refused**
until a different principal approves the exact diff hash, and the refusal is recorded in the audit
chain. `--no-approve` stops there, which is what a real deployment looks like.

Useful one-liners:

```bash
uv run vgs tasks                        # list the benchmark and its content hash
uv run vgs tasks run_length_encode/00   # one task: statement, broken program, test counts
uv run vgs search run_length_encode/00  # one budgeted search; --generator claude for the live one
uv run vgs escape-suite                 # run the sandbox escape suite on its own
uv run vgs audit verify var/audit.jsonl
uv run vgs serve                        # the search service
uv run vgs serve-runner                 # the sandbox runner (what the runner container runs)
```

## Results

Produced by `uv run python bench/bench_search.py --check --write-docs` on
Windows-10-10.0.19045-SP0, 6 CPUs, Python 3.12.14, CPU-only, in **147 seconds** (232s of sandbox time
across 630 evaluations; the difference is the verdict cache). Full tables, ablations, the start
classification and provenance: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); raw JSON:
`bench/results/search.json`.

The benchmark is 30 tasks from 15 families. 12 are spent collecting value-model training traces and
the remaining **18 are what these numbers report**, so the value model has never seen a task it is
scored on. Confidence intervals are percentile bootstrap over tasks, 1000 resamples.

### Hidden-test solve rate against budget

| strategy | 1 | 2 | 4 | 8 | 16 candidates | 95% CI at 16 |
|---|---:|---:|---:|---:|---:|---|
| greedy | 0.0% | 0.0% | 22.2% | 33.3% | 33.3% | [11.1%, 55.6%] |
| best_of_n | 0.0% | 0.0% | 5.6% | 11.1% | 11.1% | [0.0%, 27.8%] |
| beam | 0.0% | 0.0% | 5.6% | 44.4% | **72.2%** | [50.0%, 88.9%] |
| mcts | 0.0% | 0.0% | 5.6% | 27.8% | **61.1%** | [38.9%, 83.3%] |

| strategy | candidates to reach a 60% hidden solve rate |
|---|---|
| greedy | not reached at 16 |
| best_of_n | not reached at 16 |
| beam | 16 (crossed between 8 and 16) |
| mcts | 16 (crossed between 8 and 16) |

Beam and MCTS reach the 60% target; greedy plateaus at 33% from budget 8 onwards and best-of-N never
gets going. Two mechanisms explain that, and both are asserted as tests rather than argued:

- **best-of-N cannot repair a two-edit bug.** It only ever expands the root, so every candidate it
  evaluates is one edit from the start program, while every task needs two
  ([`test_best_of_n_never_leaves_depth_one`](tests/integration/test_strategies.py)). This is a
  property of searching an *edit space*, not a measurement of best-of-N over whole programs - see
  Non-goals.
- **greedy cannot backtrack.** It commits to the top-ranked edit at each step, so a wrong turn costs
  it the rest of its budget. The extra budget between 8 and 16 buys it nothing.

### Verifier false-pass rate

**22.2% of tasks** at budget 4 and above, for all four strategies: the search satisfied every visible
test and the hidden tests then failed. This is the number the two-group benchmark exists to produce.
Visible solve rates run 16.7-22.2 points above hidden ones at budget 4 and above, and 5.6-11.1 points
below it - a sentence rendered from the results file rather than typed
([`render_verifier_gap`](src/verifier_guided_search/experiments/report.py)), because a hand-written
range is exactly the kind of number that drifts away from the table above it.

It does not shrink as the budget grows, and the reason is not that search keeps producing new wrong
answers. The set of tasks whose visible suite a wrong program can satisfy is fixed and small. Beam,
MCTS and best-of-N are fooled on the *same four tasks* - two instances of `balanced_brackets` and two
of `normalize_spacing`, 2 of the 11 reported families - while finding 16, 14 and 5 visible solves
respectively; greedy is fooled on four as well, three of them the same. And one of those tasks,
`balanced_brackets/01`, starts out already passing every visible test and failing a hidden one, so it
is a false pass for every strategy at every budget with no search at all. The per-strategy task ids
are printed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), so this is checkable rather than asserted.

### Value model

- held-out AUC **0.952**, ECE 0.081, trained on 57 rows from 8 trace tasks and evaluated on 24
  rows from 4 more - 12 tasks in total, all disjoint from the 18 reported, and the holdout is
  split by task rather than by row so two mutants of one reference cannot straddle it
- comfortably above the 0.75 target
- **and it does not improve the search.** The on/off ablation is flat at budget 8, and repeating the
  whole reported sweep for MCTS with the model off gives **66.7% against 61.1% with it** at budget 16
  - one task's difference, well inside the intervals, but certainly not an improvement. That
  comparison is a measured row in `bench/results/search.json` (`sweep_value_model_off`), not an
  aside: the ablation table runs on 12 tasks at budget 8, and the claim is about the headline.

That negative result is the most interesting thing the ablations found, and the cause is visible in
the scores: the model assigns ~0.0001 to almost every candidate and ~0.9 to a solving one. It ranks
superbly (hence the AUC) but is *saturated*, so it cannot order the unsolved candidates - which is
the only thing PUCT actually needs it for. The coarse-but-graded visible pass fraction (0, ⅓, ⅔, …)
is a better search signal than a sharp-but-collapsed probability. A calibrated or blended value
signal is listed under Non-goals.

Relatedly, the `c_puct` ablation favours **0.0** (45.5% versus 27.3% at every positive value, on 11
tasks, intervals heavily overlapping) - that is, no exploration at all. What that ablation can and
cannot show is worth being precise about, because **only the `0` versus `> 0` contrast is
identifiable here, and that is arithmetic rather than a finding.** The score is `Q + c·P·√N/(1+n)`: if
`Q` is tied across the siblings being compared, the argmax reduces to `P/(1+n)` and the constant
cancels. Ties are the normal case here from both directions - the value model is saturated, and with
it off `Q` is a pass fraction with four or five possible values. The results file shows the
consequence: with the model on, the `c_puct` 0.7 and 1.4 rows agree to sixteen significant figures in
both `mean_sandbox_seconds` and `mean_generator_tokens`, i.e. they ran the same programs in the same
order. The grid is therefore also swept with the value model off (`c_puct_value_model_off`), where the
three positive values do spend differently - and still land on the same solve counts. The default is
left at the literature value of 1.4 rather than retuned, because tuning it on the tasks these results
are reported on would be selecting on the test set.

### Sandbox escape suite

**100% of 18 attempts blocked** - socket connect, DNS lookup, HTTP fetch, subprocess spawn,
`ctypes` load, writes and reads and `listdir` and `chdir` outside the jail, renaming and hard-linking
a file out of the jail, taking an `os.open` write handle on a standard-library file, `/proc` probing,
environment secret reads, a fork bomb, two thread bombs and a long sleep. 14 were refused by the
in-process audit guard; the rest by the environment allowlist or the wall-clock watchdog. Each attempt
is a program whose single test passes *only if the escape succeeded*, so "blocked" means "no attempt
returned its success marker" and not merely "nothing crashed".

Four of the 18 are there because an adversarial review escaped that way, and they now pin the fixes.
The guard checked only the *first* path argument of an audit event, so the destination of `os.rename`,
`os.replace` and `os.link` was never checked and a candidate could move or hard-link a file from its
jail to any absolute path; and it decided "is this a write?" from the mode string alone, which
`os.open` does not supply, so an `os.open` with write flags fell through to the lenient *read* check -
which accepts the standard library. (That attempt writes no bytes: it opens an existing standard-
library file for writing and closes it, because holding the descriptor is already the containment
failure and the suite has to be safe to run on a developer's own interpreter.) The fourth is `thread_bomb_via_subclasses`, which is **not** refused by the guard: the
import deny-list removes names from `sys.modules` and cannot remove the live `Thread` class that
`object.__subclasses__()` still reaches. It is in the suite to pin which control does stop it - the
wall-clock watchdog, and `pids_limit` in the container.

### Micro-benchmarks

`uv run python bench/bench_baseline.py --check`, same machine:

| benchmark | runs | p50 (µs) | p95 (µs) |
|---|---:|---:|---:|
| rate_limiter_check | 20000 | 1.8 | 2.0 |
| llm_request_sha256_10_messages | 2000 | 97.5 | 125.3 |
| verdict_cache_hit | 5000 | 130.7 | 164.9 |
| redact_text_1kb | 2000 | 339.4 | 438.5 |
| mutation_render_3_sites | 500 | 495.9 | 679.6 |
| mutation_space_build | 200 | 696.2 | 871.5 |
| audit_append_fsync | 200 | 7962.7 | 9021.3 |
| audit_verify_1000_entries | 10 | 34621.6 | 40644.5 |

A sandbox execution costs ~130 **ms** - a fresh interpreter per program - so a cache hit at 131 µs is
a thousand times cheaper, and the cache answered **57.2%** of lookups in the bench run. (The hit is
not free: the key is `sha256(program ‖ tests ‖ limits)` and the test suite is re-hashed per lookup.
Memoising that is an obvious optimisation and is not done, because 131 µs against 130 ms is not where
the time goes.) The cache is a wall-clock optimisation only - the budget charges a candidate whether
or not the cache answers, so it cannot buy extra evaluations.

## Security

The threat this repo is built around is that **candidate programs are untrusted code**. The
[threat model](docs/THREAT_MODEL.md) is specific to that; the controls are:

| control | where | evidence |
|---|---|---|
| Candidates run only in the runner container: internal network with no egress, read-only rootfs, tmpfs jail, non-root UID, `cap_drop: ALL`, pids/memory/CPU caps | `docker-compose.yml`, `api/runner.py` | [`test_containers.py`](tests/integration/test_containers.py) |
| No Docker socket mounted anywhere, no privileged service | compose + Dockerfile | [`test_no_service_mounts_the_docker_socket`](tests/integration/test_containers.py) |
| Fresh interpreter per program, `-I -S`, rlimits, wall-clock watchdog, `sys.addaudithook` guard raising a `BaseException` a candidate cannot swallow | `sandbox/child.py` | [escape suite](tests/integration/test_sandbox_escape.py), [harness tests](tests/unit/test_sandbox_child.py) |
| Resource limits are a ceiling the caller can only ask for *less* than | `api/runner.py` | [`test_a_caller_cannot_raise_the_resource_ceiling`](tests/integration/test_api.py) |
| No secrets in the runner: its environment is an allowlist, and the image and compose service hold no API key | `sandbox/local.py`, compose | [`test_the_child_environment_carries_no_secrets`](tests/integration/test_sandbox_local.py) |
| A patch is applied only after a *different* principal approves the exact diff hash; single use, expiring, refused if the file moved on | `patch.py` | [`test_patch.py`](tests/unit/test_patch.py) |
| Token, sandbox-second, candidate and USD budgets that cannot be exceeded | `search/budget.py` | [stateful property test](tests/unit/test_budget_accountant.py) |
| Task statements, programs and test output are untrusted data in prompts; the generator gets no tools | `search/prompting.py` | [`test_prompting.py`](tests/unit/test_prompting.py) |
| No free text from a run reaches a prompt - only test names, statuses and exception class names | `sandbox/protocol.py` | [`test_no_free_text_from_a_run_reaches_the_prompt`](tests/unit/test_prompting.py) |
| Hash-chained audit of every executed program hash, verdict, refusal and approval | `agentic/audit.py` | [`test_every_executed_program_reaches_the_audit_chain`](tests/integration/test_strategies.py) |
| Value models verified against their own content hash before use; `/readyz` refuses an edited artifact | `value/registry.py` | [`test_value_registry.py`](tests/unit/test_value_registry.py) |
| No pickle anywhere; the model ships as JSON | `value/model.py` | [`test_no_pickle.py`](tests/unit/test_no_pickle.py) |
| API-key auth (digests only), rate limiting, body-size limits, PII redaction, non-leaking errors | `core/` | [`test_app.py`](tests/integration/test_app.py) |
| Neither blocking route runs on the event loop, and each bounds its own concurrency, so one in-flight search cannot take `/healthz` down with it | `core/concurrency.py`, `api/` | [`test_the_health_probes_answer_while_an_execution_is_in_flight`](tests/integration/test_api.py) |

Every claim in this file maps to a test, bench script or CI job in [CLAIMS.md](CLAIMS.md), and
`scripts/check_baseline.py` fails if any of those references goes stale - and, since a docstring
citing a file is also a claim, if any repo-relative path mentioned in `src/`, `bench/`, `scripts/` or
`docs/` does not resolve.

**The in-process guard is defence in depth, not the boundary.** The harness and the candidate share
one interpreter, so a determined program can interfere with its own bookkeeping. The container is the
boundary. The escape suite measures the guard because a cheap layer that demonstrably refuses the
obvious moves is worth having and worth regression-testing; it is not a claim that the guard is
sufficient.

## Non-goals

Honest limits, in rough order of how much they constrain what the numbers mean.

- **The offline curves describe search over an edit space, not over a model's output
  distribution.** The offline generator enumerates single-edit neighbours of each reference solution
  and scores them with a noisy revert/introduce signal. Every mutant is syntactically valid, the
  neighbourhood is finite and local, and the reference is always reachable - none of which is true of
  a language model. The live `claude-sonnet-5` generator has no such guarantees.
- **No numbers are reported for the live generator.** It is wired and reachable - `vgs search
  <task> --generator claude`, or `APP_GENERATOR=claude` for the search service - and the whole path
  is tested offline by recording a run and replaying it from a cassette
  ([`test_live_generator.py`](tests/integration/test_live_generator.py)). But no `ANTHROPIC_API_KEY`
  was available here, so nothing in Results comes from it, no cassettes of real Claude responses are
  committed, and `tasks.py record` is still a stub rather than a recording workflow.
- **best-of-N is measured over that same edit space, which under-represents it.** Here "one sample"
  is one edit from the current program, so best-of-N is confined to depth 1 and structurally cannot
  repair a two-edit bug. Real best-of-N samples whole programs. The result in the table is correct
  for the edit-space framing the spec sets out and should not be read as a general finding.
- **Benchmark difficulty is a dial that was chosen by measurement.** Corruption depth 2 is where the
  strategies separate; at depth 1 best-of-N matches the searches and at depth 3 everything flattens
  at the bottom. Same for the generator's noise level. Both sweeps are a script with a committed
  artifact - `uv run python bench/bench_calibration.py --check`, `bench/results/calibration.json` -
  and the numbers are in [ADR 0003](docs/adr/0003-mutation-operator-generator.md), not in prose.
- **0 of the 18 reported tasks start behaviour-preserving**, so no reported solve rate contains a
  free win. 2 of the 30 tasks in the whole suite do, and both landed in the 12-task trace split -
  worth stating that way round, because "2 of 30" invites discounting 11.1% from every rate, which is
  exactly best-of-N's headline number. The classification is produced by the bench on the reported
  split and printed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
  ([`tasks/validate.py`](src/verifier_guided_search/tasks/validate.py),
  [`test_suite_quality.py`](tests/integration/test_suite_quality.py)).
- **1 of the 18 reported tasks already fools its own visible suite at depth 0.**
  `balanced_brackets/01` starts passing every visible test and failing a hidden one, so it
  contributes a false pass to every strategy at every budget with no search at all. That is not a
  defect - it is the verifier-overfitting case arriving for free - but it is most of why the
  false-pass rate looks constant, and it is reported rather than left to be inferred.
- **18 tasks is a small sample.** Most gaps between adjacent strategies sit inside their confidence
  intervals. The beam-versus-best-of-N gap does not; the beam-versus-MCTS gap does. The full
  150-task suite exists (`vgs tasks`) but takes far longer than the "minutes on a CPU" this repo
  commits to, so the reported configuration is the small one.
- **The value model does not help the search**, as measured (66.7% without it against 61.1% with it,
  MCTS at budget 16 on the reported split). A calibrated or blended value signal (model probability
  mixed with visible pass fraction), and a value model trained on more than 57 rows, are the obvious
  next steps and are not implemented. [ADR 0004](docs/adr/0004-value-model-does-not-help-search.md)
  says why the default is nevertheless left on.
- **Tokens are pro-rated, not measured per candidate.** A generator call produces `k` candidates, so
  a prefix's token count is estimated by the share of evaluations it contains. The headline compute
  axis is therefore verifier calls; token and sandbox-second spend are reported alongside.
- **Not implemented:** the spec's `EXPERIMENTS.md`-generating script covers the sweep, ablations,
  value model and escape suite, but there is no OpenTelemetry span per expansion (only the HTTP
  spans the baseline provides), and no distributed or parallel search. Executions are serial by
  design on a 2 GB CPU-only host.
- **The container controls are verified by reading the declarations**, not by starting Docker. A
  separate CI job (`.github/workflows/ci.yml#sandbox-escape`) runs the escape suite inside compose on
  a Linux runner.
- **Registry integrity is a content hash, not a signature.** It detects corruption and any edit that
  did not also rename the file; it does not stop someone who can rewrite both the artifact and the
  index.
- **The coverage figure does not cover the child harness.** `sandbox/child.py` is copied out and run
  by a separate interpreter as `python -I -S -B`, and `-S` skips `site`, so coverage's subprocess
  startup hook never runs - instrumenting it would mean handing the sandboxed interpreter its
  environment and its site-packages back, which is the hardening the file exists for. The escape
  suite and the sandbox tests exercise it hard and contribute nothing to the number, so the coverage
  table's figure for that one file (the lowest in the repo) understates it while the repo-wide figure
  slightly overstates the rest. The controls in that file are measured by the escape suite instead,
  which is a behavioural check rather than a line count.

## Licence

MIT. See [LICENSE](LICENSE).
