# Demo agent container. It sits on the `edge` network only: the gateway is the one service it can
# resolve or reach. It holds no signing key and no upstream credentials; tokens are issued to it
# by an operator via the gateway's operator API. The real demo agent replaces the idle command.
FROM python:3.12.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt && rm /tmp/requirements.txt \
    && groupadd --system --gid 10001 agent \
    && useradd --system --uid 10001 --gid agent --no-create-home --shell /usr/sbin/nologin agent
WORKDIR /app
USER agent
CMD ["sleep", "infinity"]
