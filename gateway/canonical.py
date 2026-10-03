"""The one canonical JSON form operations are bound by.

Sorted keys, compact separators, UTF-8, no NaN or infinity. Sealing controls (`sql_guard`)
seal a payload by these bytes and approvals bind an operation by a keyed digest of them, so
"the same operation" means the same thing to both.
"""

import hashlib
import hmac
import json


def canonical_bytes(payload: object) -> bytes | None:
    """Sorted-key, compact, UTF-8 JSON without NaN; None if ``payload`` is not plain JSON."""
    try:
        text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError):
        return None
    return text.encode()


def canonical_digest(key: bytes, payload: object) -> str | None:
    """Keyed HMAC-SHA256 of the canonical bytes (a plain hash is guessable for low-entropy
    arguments); None if ``payload`` has no canonical form."""
    canonical = canonical_bytes(payload)
    if canonical is None:
        return None
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()
