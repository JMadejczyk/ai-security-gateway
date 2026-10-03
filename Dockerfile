# AI Control Layer gateway. Two listeners from one process (`python -m gateway`):
#   ACL_AGENT_HOST:8080     agent API    (/v1, /mcp)
#   ACL_OPERATOR_HOST:9090  operator API (/auth/demo-token, /admin/*, /metrics, /healthz)
# Both default to 127.0.0.1; docker-compose.yml binds the operator listener to the gateway's
# static IP on the `ops` network only, so it is unreachable from `edge` and the upstream networks.
# policy.yaml and feeds/ are bind-mounted read-only at runtime; secrets come from the environment.
FROM ghcr.io/astral-sh/uv:0.12.22 AS uv

FROM python:3.12.15-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

FROM python:3.12.15-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ACL_POLICY_PATH=/app/policy.yaml \
    ACL_AGENT_HOST=127.0.0.1 \
    ACL_AGENT_PORT=8080 \
    ACL_OPERATOR_HOST=127.0.0.1 \
    ACL_OPERATOR_PORT=9090
RUN groupadd --system --gid 10001 acl \
    && useradd --system --uid 10001 --gid acl --no-create-home --shell /usr/sbin/nologin acl
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY gateway ./gateway
RUN python -m compileall -q gateway
USER acl
EXPOSE 8080 9090
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; e = os.environ; u = f\"http://{e['ACL_OPERATOR_HOST']}:{e['ACL_OPERATOR_PORT']}/healthz\"; urllib.request.urlopen(u, timeout=2)"]
CMD ["python", "-m", "gateway"]
