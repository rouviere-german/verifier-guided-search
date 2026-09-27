# Security policy

## Scope and status

This is a portfolio project. It is **not** a hosted service, it has no users, and it is not intended
for production use as-is. Treat it as a worked example of how to build a system that executes
untrusted code safely, not as a supported product.

That said, the security controls are real and are meant to hold. If one of them does not, the finding
is welcome and the repository should be fixed.

## What this project does that deserves care

It **executes untrusted Python programs on purpose**. That is the point of the system: a search
generates candidate programs and a verifier runs them. Everything in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) follows from that, and two properties matter most:

- Candidate programs run only in the **runner** service: a container on a Compose network declared
  `internal: true` (no egress), with a read-only root filesystem, a size-bounded tmpfs jail, a
  non-root UID, `cap_drop: ALL`, `no-new-privileges`, and pids, memory and CPU limits. It holds no
  credentials.
- The in-process guard in `src/verifier_guided_search/sandbox/child.py` (rlimits, a wall-clock
  watchdog, and a `sys.addaudithook` deny-list) is **defence in depth, not the boundary**. The harness
  and the candidate share one interpreter. A guard bypass on its own is a real finding, but it is not
  a full compromise; a container escape would be.

## Running it yourself

The test suite, the offline demo and the benchmarks execute candidate programs in **local
subprocesses, without a container**. That is deliberate - it is what makes the whole thing runnable on
a laptop with no Docker - and it means the only isolation in that mode is the in-process guard and the
OS. The programs those paths run are the benchmark's own generated mutations and the escape suite's
own attempts, all of which ship in this repository, so the risk is bounded by what you can read.

If you point the search at a live model (`ANTHROPIC_API_KEY` plus `ClaudeGenerator`), the candidates
become model output, and you should **run the verifier in the container** (`docker compose up`) rather
than locally.

Also worth knowing:

- Never commit a `.env`. Only `.env.example` is tracked, and `.gitignore` covers `.env`, caches and
  generated artifacts.
- API keys are configured as SHA-256 digests (`APP_API_KEY_SHA256`), so the process never holds a raw
  key. Generate one with `uv run vgs generate-api-key`.
- The default `APP_HOST` is `127.0.0.1`. The compose stack publishes the search service on loopback
  only, and the runner on nothing at all.

## Reporting a vulnerability

Open a **GitHub security advisory** on this repository (Security → Advisories → Report a
vulnerability), which keeps the report private until there is a fix. If that is not available to you,
open a normal issue that says only that you have a security report and asks for a contact - please do
not put the details in a public issue.

Useful things to include:

- what the control was supposed to prevent, and what it let through;
- whether the finding is in the in-process guard, the container declaration, the approval gate, the
  budget accountant, or the HTTP surface;
- a program, request or compose change that demonstrates it - an escape-suite attempt in the style of
  `src/verifier_guided_search/sandbox/escape.py` is ideal, because it drops straight into the test
  suite as a regression test.

Since this is a personal project, expect a best-effort reply within a week or two rather than a
committed SLA. A fix will land with a test that would have caught it.

## Out of scope

- A container-runtime escape, a malicious base image, or a compromised dependency (scikit-learn,
  pydantic, FastAPI).
- An attacker who can already write to this repository, the value-model registry, or the workspace a
  patch is applied to.
- Denial of service against your own machine by running the benchmark - it is CPU-bound by design.
- The absence of a signature on value-model artifacts. The registry verifies a content hash, which
  detects corruption and any edit that did not also rename the file; it does not stop someone who can
  rewrite both the artifact and the index. This is recorded as residual risk in the threat model
  rather than treated as a bug.
- Findings that require an `ANTHROPIC_API_KEY` to be leaked by something outside this repository.

## Verifying the controls yourself

```bash
uv run python tasks.py test        # includes the escape suite and the container declaration checks
uv run vgs escape-suite            # the escape suite on its own, with a table of what stopped what
uv run python tasks.py audit       # bandit + pip-audit
uv run python tasks.py baseline    # every claim's evidence reference still resolves
```

[CLAIMS.md](CLAIMS.md) maps every security claim in the README to the test that proves it.
