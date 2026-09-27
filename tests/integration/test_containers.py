"""The compose stack and Dockerfile, read as data: the runner must have no way out.

Docker is never started here (a separate serial step verifies containers). What is checked is the
*declaration*, because that is where these controls live and where a regression would land: a
mounted Docker socket, a network that forgot ``internal: true``, a writable root filesystem, a
dropped
``cap_drop``, a published port on the runner, or a secret handed to the process that runs untrusted
code.

The spec asks specifically that a test inspect the compose config for a Docker socket mount. That is
:func:`test_no_service_mounts_the_docker_socket`, and it looks for the socket in every form it can
be written - a bind under ``volumes``, a long-form mount, the ``/run/docker.sock`` path, and the
``DOCKER_HOST`` variable - rather than grepping for one string.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.integration

_SOCKET_PATHS = ("/var/run/docker.sock", "/run/docker.sock", "docker.sock")
_RUNNER = "runner"
_APP = "app"


@pytest.fixture(scope="module")
def compose(repo_root: Path) -> dict[str, Any]:
    loaded = yaml.safe_load((repo_root / "docker-compose.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, Any]:
    found = compose.get("services")
    assert isinstance(found, dict), "services is not a mapping"
    assert found, "docker-compose.yml declares no services"
    return found


@pytest.fixture(scope="module")
def dockerfile(repo_root: Path) -> str:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    # Join line continuations so a multi-line instruction matches as one.
    return re.sub(r"\\\n[ \t]*", " ", text)


def _mount_strings(service: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for entry in service.get("volumes", []) or []:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            out.append(f"{entry.get('source', '')}:{entry.get('target', '')}")
    return out


def test_the_stack_declares_both_an_app_and_a_runner(services: dict[str, Any]) -> None:
    assert _APP in services, "the search service is missing"
    assert _RUNNER in services, "the sandbox runner is missing"


def test_no_service_mounts_the_docker_socket(services: dict[str, Any]) -> None:
    """The spec's explicit requirement. Checked in every form the mount can be written."""
    for name, service in services.items():
        for mount in _mount_strings(service):
            for socket in _SOCKET_PATHS:
                assert socket not in mount, f"service {name} mounts {socket}"
        environment = service.get("environment", {}) or {}
        names = (
            environment
            if isinstance(environment, dict)
            else dict(item.split("=", 1) for item in environment if "=" in item)
        )
        assert "DOCKER_HOST" not in names, f"service {name} is given DOCKER_HOST"
        assert not service.get("privileged"), f"service {name} is privileged"


def test_the_whole_file_never_mentions_the_docker_socket(repo_root: Path) -> None:
    """A belt-and-braces textual check, so a socket added under a key this test does not know about
    (``configs``, an extension field, a future compose key) still fails."""
    text = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")
    lowered = text.lower()
    for socket in _SOCKET_PATHS:
        for line in lowered.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # a comment explaining the absence is fine
            assert socket not in stripped, f"docker-compose.yml references {socket}"


def test_the_runner_network_has_no_egress(
    compose: dict[str, Any], services: dict[str, Any]
) -> None:
    networks = compose.get("networks", {}) or {}
    runner_networks = services[_RUNNER].get("networks", []) or []
    assert runner_networks, "the runner is on the default bridge, which has egress"
    for name in runner_networks:
        declared = networks.get(name) or {}
        assert declared.get("internal") is True, f"network {name} is not internal: true"


def test_the_runner_publishes_no_ports(services: dict[str, Any]) -> None:
    assert not services[_RUNNER].get("ports"), "the runner must not be reachable from the host"
    assert not services[_RUNNER].get("expose") or services[_RUNNER].get("expose") == [
        services[_RUNNER].get("expose", [None])[0]
    ]


def test_the_app_is_published_only_on_loopback(services: dict[str, Any]) -> None:
    for mapping in services[_APP].get("ports", []) or []:
        assert str(mapping).startswith("127.0.0.1:"), (
            f"port {mapping} is published on all interfaces"
        )


@pytest.mark.parametrize("service", [_APP, _RUNNER])
def test_every_service_is_hardened(services: dict[str, Any], service: str) -> None:
    declared = services[service]
    assert declared.get("read_only") is True, f"{service} has a writable root filesystem"
    assert declared.get("cap_drop") == ["ALL"], f"{service} does not drop all capabilities"
    assert "no-new-privileges:true" in (declared.get("security_opt") or []), service
    assert declared.get("pids_limit"), f"{service} has no pids limit"
    assert declared.get("mem_limit"), f"{service} has no memory limit"
    assert declared.get("cpus"), f"{service} has no CPU limit"
    assert declared.get("user"), f"{service} does not pin a non-root UID"
    assert not str(declared["user"]).startswith(("root", "0:")), service


def test_the_runner_gets_a_writable_tmpfs_for_its_jail_and_nothing_else(
    services: dict[str, Any],
) -> None:
    tmpfs = services[_RUNNER].get("tmpfs") or []
    assert tmpfs, "the runner needs a tmpfs to run programs in"
    assert any(str(entry).startswith("/sandbox") for entry in tmpfs), tmpfs
    assert all("size=" in str(entry) for entry in tmpfs), (
        "an unbounded tmpfs is a memory limit hole"
    )
    assert not _mount_strings(services[_RUNNER]), "the runner must not bind-mount host paths"


def test_the_runner_holds_no_model_credentials(services: dict[str, Any]) -> None:
    """The process that runs untrusted code must not be the process holding the API key."""
    environment = services[_RUNNER].get("environment", {}) or {}
    names = (
        set(environment)
        if isinstance(environment, dict)
        else {item.split("=", 1)[0] for item in environment}
    )
    assert "ANTHROPIC_API_KEY" not in names
    assert not any("ANTHROPIC" in name for name in names)


def test_the_app_reaches_the_runner_over_the_internal_network(services: dict[str, Any]) -> None:
    environment = services[_APP].get("environment", {}) or {}
    assert isinstance(environment, dict)
    runner_url = str(environment.get("APP_RUNNER_URL", ""))
    assert runner_url, "the app does not know where the runner is"
    assert _RUNNER in runner_url, f"APP_RUNNER_URL does not point at the runner: {runner_url}"
    assert "localhost" not in runner_url, runner_url
    assert "127.0.0.1" not in runner_url, runner_url


def test_the_app_is_on_both_networks_and_the_runner_only_on_the_internal_one(
    services: dict[str, Any],
) -> None:
    app_networks = set(services[_APP].get("networks", []) or [])
    runner_networks = set(services[_RUNNER].get("networks", []) or [])
    assert runner_networks <= app_networks, "the app cannot reach the runner"
    assert len(app_networks) >= 2, "the app needs an ingress network as well as the internal one"


def test_the_image_is_multi_stage_pinned_and_non_root(dockerfile: str) -> None:
    froms = re.findall(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)", dockerfile, re.M | re.I)
    assert len(froms) >= 2, "a multi-stage build is required"
    args = dict(re.findall(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", dockerfile, re.M))
    stages = set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", dockerfile, re.M | re.I))
    assert stages, "no named build stage"
    for image in froms:
        resolved = re.sub(
            r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: args.get(m.group(1), m.group(0)), image
        )
        # A FROM that names an earlier stage is not a registry reference, so it has no digest.
        if resolved in stages:
            continue
        assert "@sha256:" in resolved, f"base image {image} is not pinned by digest"
    assert any(
        image.startswith("python:3.12-slim")
        for image in (
            re.sub(
                r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?",
                lambda m: args.get(m.group(1), m.group(0)),
                candidate,
            )
            for candidate in froms
        )
    ), "the runtime must be built on python:3.12-slim"
    users = re.findall(r"^USER\s+(\S+)", dockerfile, re.M)
    assert users, "the Dockerfile never sets USER"
    assert users[-1].split(":", maxsplit=1)[0] not in {"root", "0"}, users[-1]


def test_the_dockerfile_never_mounts_or_installs_a_docker_client(dockerfile: str) -> None:
    lowered = dockerfile.lower()
    for socket in _SOCKET_PATHS:
        assert socket not in lowered
    assert "docker-ce" not in lowered
    assert "docker.io" not in lowered


def test_the_image_ships_no_secrets(dockerfile: str) -> None:
    for secret in ("ANTHROPIC_API_KEY", "APP_API_KEY_SHA256", "APP_RUNNER_API_KEY"):
        assert not re.search(rf"^(ENV|ARG)\s+{secret}\s*=\s*\S", dockerfile, re.M), secret


def test_the_healthcheck_uses_the_python_probe(dockerfile: str) -> None:
    assert re.search(r"^HEALTHCHECK\s.*python", dockerfile, re.M)
    assert "verifier_guided_search.core.healthcheck" in dockerfile


def test_the_runner_command_is_the_runner_entry_point(services: dict[str, Any]) -> None:
    command = services[_RUNNER].get("command")
    assert command, "the runner does not say which service to start"
    assert (
        "serve-runner" in " ".join(command)
        if isinstance(command, list)
        else "serve-runner" in command
    )
