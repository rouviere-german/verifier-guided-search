# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="verifier-guided-search" \
      org.opencontainers.image.licenses="MIT"
# The account, and the runner's work root. APP_RUNNER_WORKROOT defaults to /sandbox and the runner
# creates its control and jail directories under it, but only root can create a directory at /, so
# without this line the image depends on something else supplying the mount point: a standalone
# `docker run <image> serve-runner` answers /healthz and fails /readyz forever.
#
# Mode 1777 is the same mode compose declares for the tmpfs, and it has to be, twice over: a tmpfs
# mounted here inherits this directory's mode but is owned by root, so 0700 would leave the mount
# unwritable by uid 10001, and with no mount at all this is an ordinary directory the runner writes
# to. Sticky, so nothing can remove another execution's jail; the container runs one unprivileged
# uid, and the guard and the rlimits are what constrain a candidate, not these bits.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app \
    && install --directory --owner=10001 --group=10001 --mode=1777 /sandbox
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-m", "verifier_guided_search.core.healthcheck"]
ENTRYPOINT ["python", "-m", "verifier_guided_search"]
CMD ["serve"]
