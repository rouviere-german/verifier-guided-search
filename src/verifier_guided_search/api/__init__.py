"""The two HTTP surfaces: the runner that executes untrusted code, and the search service.

They are separate applications on purpose, and the separation is the security boundary the whole
repo is built around:

* :mod:`verifier_guided_search.api.runner` is the only process that ever executes a candidate
  program. It runs in a container on a compose network declared ``internal: true``, with a read-only
  root filesystem, a tmpfs workdir, a non-root UID, ``cap_drop: ALL`` and pids, memory and CPU caps.
  It exposes exactly one route that does work: ``POST /v1/execute``.
* :mod:`verifier_guided_search.api.service` is the search service. It talks to the runner over the
  internal network and never runs candidate code itself. It is the only one with an ingress port.

Neither mounts the Docker socket; ``tests/integration/test_containers.py`` asserts that from the
compose file and the Dockerfile rather than trusting the comment.
"""

from __future__ import annotations

from verifier_guided_search.api.runner import create_runner_app
from verifier_guided_search.api.service import create_search_app

__all__ = ["create_runner_app", "create_search_app"]
