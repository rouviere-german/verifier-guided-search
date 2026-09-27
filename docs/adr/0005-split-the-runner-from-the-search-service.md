# 5. Put the verifier in its own process on a network with no egress

Date: 2026-09-27

## Status

Accepted.

## Context

The system executes candidate programs. Those programs are, in the offline case, mutations of our own
reference solutions, and in the live case whatever a language model emitted - which means, in the
limit, whatever a prompt-injected language model emitted. They have to run, and they have to run
often: a budget sweep is hundreds of executions.

The search service, meanwhile, is the process that holds the API key digests, the Anthropic key, and
the ability to write to a workspace. Running candidate programs in that process means a sandbox escape
hands an attacker all of it at once.

## Decision

**Two processes, two networks, one image.**

`verifier-guided-search serve` is the search service. It has an ingress port and the credentials, and
it never executes a candidate. `verifier-guided-search serve-runner` is the runner. It exposes exactly
one working route, `POST /v1/execute`, and:

- sits on a compose network declared `internal: true`, so the daemon attaches no gateway and there is
  no route to the internet, the host, or any other network;
- publishes **no host port**, so nothing outside the compose network can reach it;
- has a read-only root filesystem with a size-bounded tmpfs at `/sandbox`, the only writable path;
- runs as a non-root UID with `cap_drop: ALL` and `no-new-privileges`, under `pids_limit`, `mem_limit`
  and `cpus`;
- is given **no credentials at all** beyond the digest of the key for its own route - no Anthropic key,
  nothing else;
- clamps the resource limits a caller asks for *down* to its own configured ceiling, never up.

Inside it, each program gets a fresh interpreter (`python -I -S`, so the environment and site-packages
are gone), POSIX rlimits, a wall-clock watchdog, and a `sys.addaudithook` guard.

One image builds both roles, so the harness that executes candidates in production is byte-identical to
the one the tests and the offline demo exercise.

## Consequences

**Good.** A sandbox escape gets an attacker a container with no network, no writable filesystem outside
a small tmpfs, no capabilities, no secrets, and a pids limit. The blast radius is one execution. The
escape suite can be run against the same code path locally, without Docker, which is why it is part of
the ordinary test run rather than a Linux-only CI job. And because the split is *declared* in compose,
it is checkable as data: `tests/integration/test_containers.py` asserts the internal network, the
absent ports, the hardening flags, the absent Docker socket and the absent credentials, all without
starting a container.

**Bad.** Two services to run and one more hop per execution. The HTTP client has to treat "the runner is
unreachable" as distinct from "the program failed" - conflating them would turn an outage into a
confident report of zero solves - so `/readyz` fails when the runner is unreachable and the client
raises `SandboxUnavailable` rather than returning a verdict.

**Deliberately not done.** The runner does not mount the Docker socket and does not spawn a container
per execution. That is the other common design, and it trades a process boundary for a socket that
grants root-equivalent control of the host. A test asserts the socket is absent in every form it can be
written - a bind mount, a long-form mount, the `/run/docker.sock` path, and the `DOCKER_HOST` variable -
because the spec asks for it and because a comment claiming its absence is worth nothing.

## Alternatives considered

**One process, guard only.** Simplest, and the guard shares an interpreter with the candidate, so
`ctypes`, a native extension or a bug in the hook is a full compromise of the process holding the keys.
Rejected; the guard is kept as a cheap first layer, not as the boundary.

**gVisor or Firecracker per execution.** Stronger isolation. Rejected for this repo: neither runs on the
CPU-only Windows host the offline path has to work on, and the point of the offline path is that the
identical harness runs everywhere.

**A container per execution via the Docker socket.** Mounting the socket to get isolation is a net loss:
it grants root-equivalent control of the host to the process most likely to be attacked.

**Run untrusted code in a subprocess with seccomp filters applied from Python.** There is no maintained,
cross-platform way to do this from CPython. The container's default seccomp profile provides it at the
right layer instead.
