"""Refusals the gateway turns into HTTP responses.

A `RejectionError` carries a structured ``reason_code`` (audited, returned to the agent as
``error.code``) and a human message that never contains payload data.
"""


class RejectionError(Exception):
    """A call refused before or instead of reaching the upstream."""

    status_code: int = 403  # subclasses and instances may narrow it

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = reason_code
        self.message = message or reason_code.replace("_", " ")
        super().__init__(f"{reason_code}: {self.message}")


class StartupError(Exception):
    """The process configuration is unusable: the gateway refuses to start (never degrades)."""


class InvalidRequestError(RejectionError):
    """The request body is not something the entry point accepts."""

    status_code = 400


class RequestTooLargeError(RejectionError):
    status_code = 413

    def __init__(self, limit: int) -> None:
        super().__init__("request_too_large", f"request body exceeds {limit} bytes")
