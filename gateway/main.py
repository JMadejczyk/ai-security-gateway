"""The two HTTP listeners (SPEC "No bypassing the gateway").

- Agent API (``edge`` network): ``/v1/chat/completions``, ``/v1/models``, ``DELETE /v1/session``.
  Errors use the OpenAI shape ``{"error": {"message", "type", "code"}}`` with the reason code
  as ``code`` and never any payload or upstream data.
- Operator API (``ops`` network only): ``/healthz``, ``/metrics``, ``/auth/demo-token``,
  ``/admin/reload``.

Both apps share one `GatewayContainer`; each lifespan enters its ``running()`` context.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http import HTTPStatus
from typing import Any, Final

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from gateway.adapters.llm import ChatCompletionRequest
from gateway.container import GatewayContainer
from gateway.core.types import Action, Channel, Decision
from gateway.errors import RejectionError, RequestTooLargeError
from gateway.identity import DemoTokenRequest, IssuedToken
from gateway.pipeline import CallRequest
from gateway.proxies.llm import sse_events
from gateway.telemetry import REGISTRY, ReloadResult, set_tainted_sessions

ADMIN_ROLE: Final = "admin"

_ERROR_TYPES: Final = {
    HTTPStatus.BAD_REQUEST: "invalid_request_error",
    HTTPStatus.UNAUTHORIZED: "authentication_error",
    HTTPStatus.FORBIDDEN: "permission_error",
    HTTPStatus.NOT_FOUND: "not_found_error",
    HTTPStatus.METHOD_NOT_ALLOWED: "invalid_request_error",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "invalid_request_error",
    HTTPStatus.TOO_MANY_REQUESTS: "rate_limit_error",
    HTTPStatus.BAD_GATEWAY: "upstream_error",
}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    """An OpenAI-style error; ``code`` is the structured reason code."""
    kind = _ERROR_TYPES.get(HTTPStatus(status), "api_error")
    headers = {"www-authenticate": "Bearer"} if status == HTTPStatus.UNAUTHORIZED else None
    body = {"error": {"message": message, "type": kind, "code": code}}
    return JSONResponse(body, status_code=status, headers=headers)


def bearer_token(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


async def read_capped(request: Request, limit: int) -> bytes:
    """The body, but never more than ``limit + 1`` bytes: enough to tell it is too large."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise RequestTooLargeError(limit)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            break
    return bytes(body)


def _lifespan(
    container: GatewayContainer,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        async with container.running():
            yield

    return lifespan


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RejectionError)
    async def on_rejection(_request: Request, exc: RejectionError) -> Response:
        return error_response(exc.status_code, exc.reason_code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(_request: Request, exc: StarletteHTTPException) -> Response:
        phrase = HTTPStatus(exc.status_code).phrase
        return error_response(exc.status_code, phrase.lower().replace(" ", "_"), phrase)

    @app.exception_handler(Exception)
    async def on_crash(_request: Request, _exc: Exception) -> Response:
        return error_response(500, "internal_error", "internal gateway error")


# ------------------------------------------------------------------------------ agent API


def create_agent_app(container: GatewayContainer) -> FastAPI:
    app = FastAPI(
        title="AI Control Layer: agent API",
        lifespan=_lifespan(container),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    _install_error_handlers(app)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        snapshot = container.policy_store.current  # the one snapshot this call reads
        body = await read_capped(request, snapshot.policy.limits.max_request_bytes)
        call = CallRequest(channel=Channel.LLM, token=bearer_token(request), body=body)
        outcome = await container.pipeline.handle(call, snapshot)
        if not outcome.released:
            return error_response(outcome.status_code, outcome.reason_code, outcome.message)
        result: dict[str, Any] = outcome.result
        sent = ChatCompletionRequest.model_validate(outcome.request)
        if sent.stream:
            usage = sent.stream_options is not None and sent.stream_options.include_usage
            events: Iterator[bytes] = sse_events(result, include_usage=usage)
            return StreamingResponse(
                events, media_type="text/event-stream", headers={"cache-control": "no-cache"}
            )
        return JSONResponse(result)

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        snapshot = container.policy_store.current
        evaluator = container.pipeline.evaluator
        now = container.clock()
        async with container.gate.admit(bearer_token(request), snapshot) as (claims, ctx):
            cards = await container.llm.list_models(snapshot)
            principal = claims.principal_context()
            allowed = [
                card
                for card in cards
                if evaluator.decide(
                    snapshot,
                    principal,
                    ctx,
                    channel=Channel.LLM,
                    action=Action.GENERATE,
                    resource=f"model:{card['id']}",
                    now=now,
                ).decision
                is not Decision.BLOCK
            ]
        return JSONResponse({"object": "list", "data": allowed})

    @app.delete("/v1/session")
    async def end_session(request: Request) -> Response:
        snapshot = container.policy_store.current
        async with container.gate.admit(bearer_token(request), snapshot) as (claims, _ctx):
            await container.sessions.end(claims.session_id)
        set_tainted_sessions(await container.sessions.tainted_count())
        return JSONResponse({"session_id": claims.session_id, "ended": True})

    return app


# --------------------------------------------------------------------------- operator API


def create_operator_app(container: GatewayContainer) -> FastAPI:
    app = FastAPI(
        title="AI Control Layer: operator API",
        lifespan=_lifespan(container),
        docs_url=None,
        redoc_url=None,
    )
    _install_error_handlers(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "policy_revision": container.policy_store.current.revision}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    issuer = container.issuer
    if issuer is not None:  # ACL_DEMO_TOKENS=0: the route does not exist (404)

        @app.post("/auth/demo-token")
        async def demo_token(body: DemoTokenRequest) -> IssuedToken:
            return issuer.issue(body, container.policy_store.current)

    @app.post("/admin/reload")
    async def reload_policy(request: Request) -> Response:
        claims = container.verifier.verify(bearer_token(request), container.policy_store.current)
        if ADMIN_ROLE not in claims.roles:
            raise RejectionError("admin_required", "this operation needs the admin role")
        outcome = await asyncio.to_thread(container.policy_store.reload)
        status = 422 if outcome.result is ReloadResult.INVALID else 200
        return JSONResponse(outcome.model_dump(mode="json"), status_code=status)

    return app
