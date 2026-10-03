"""HTTP client for the operator listener (``/admin/*``), used by every CLI subcommand.

The CLI never talks to Redis or the policy file: everything goes through the operator API,
so the same operator-token checks apply as for any other operator client.
"""

from collections.abc import Mapping
from http import HTTPStatus
from types import TracebackType
from typing import Final, Self, cast

import httpx
from pydantic import BaseModel

DEFAULT_URL: Final = "http://127.0.0.1:9090"
URL_ENV: Final = "ACL_OPERATOR_URL"
BEARER_ENV: Final = "ACL_OPERATOR_TOKEN"
TIMEOUT_S: Final = 10.0


class OperatorError(Exception):
    """The operator API refused or could not be reached; ``code`` is its reason code."""

    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status


class OperatorClient:
    """``async with OperatorClient(url, token) as client: await client.get("/admin/...")``."""

    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token
        self._http = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=TIMEOUT_S)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def get(self, path: str, **params: str | int) -> httpx.Response:
        """GET ``path``; raises `OperatorError` unless the answer is 2xx."""
        return await self._send("GET", path, params=params)

    async def post(self, path: str, json: object = None) -> httpx.Response:
        """POST ``path`` with a JSON body; raises `OperatorError` unless the answer is 2xx."""
        return await self._send("POST", path, json=json)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
        json: object = None,
    ) -> httpx.Response:
        if not self._token:
            raise OperatorError("token_missing", f"pass --token or set {BEARER_ENV}")
        headers = {"authorization": f"Bearer {self._token}"}
        try:
            response = await self._http.request(
                method, path, headers=headers, params=params, json=json
            )
        except httpx.HTTPError as exc:
            raise OperatorError("unreachable", type(exc).__name__) from None
        if not response.is_success:
            raise _refusal(response)
        return response


def parse[M: BaseModel](response: httpx.Response, model: type[M]) -> M:
    return model.model_validate_json(response.content)


def _refusal(response: httpx.Response) -> OperatorError:
    """The operator API's error body is ``{"error": {"code", "message"}}``; FastAPI's own
    validation errors are ``{"detail": [...]}``."""
    try:
        body: object = response.json()
    except ValueError:
        body = None
    error = cast("dict[str, object]", body).get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        details = cast("dict[str, object]", error)
        code, message = str(details.get("code", "error")), str(details.get("message", ""))
        return OperatorError(code, message, response.status_code)
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        return OperatorError("invalid_request", "the request was rejected", response.status_code)
    return OperatorError("http_error", f"HTTP {response.status_code}", response.status_code)
