"""`CallCounter` in Redis: ``loop_detect``'s sliding windows, shared by every gateway.

One sorted set per session and fingerprint (``acl:loop:<session>:<sha256 of the key>``); each
occurrence is a member scored with its time in microseconds. One Lua script per hit drops the
occurrences that left the window, adds this one, counts, and sets the key to expire with the
window, so a session's counts vanish on their own once it stops repeating itself.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Final, cast, override

from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.controls.loop_detect import CallCounter, CallCounterUnavailableError

KEY_PREFIX: Final = "acl:loop"
_MICROS: Final = 1_000_000

# KEYS[1]: the window. ARGV: now (us), window (us), unique member. Members at or before
# now - window have left it (as in InMemoryCallCounter).
_HIT: Final = """
local now, window = tonumber(ARGV[1]), tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
redis.call('ZADD', KEYS[1], now, ARGV[3])
redis.call('PEXPIRE', KEYS[1], math.ceil(window / 1000))
return redis.call('ZCARD', KEYS[1])
"""


class RedisCallCounter(CallCounter):
    """Sliding-window counts in Redis; any Redis or connection error fails closed."""

    def __init__(self, client: Redis) -> None:
        self._hit = client.register_script(_HIT)

    @override
    async def hit(self, session_id: str, key: str, now: datetime, window_s: float) -> int:
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        now_us = int(now.timestamp() * _MICROS)
        args = [now_us, int(window_s * _MICROS), f"{now_us}-{secrets.token_hex(4)}"]
        try:
            count = await self._hit(keys=[f"{KEY_PREFIX}:{session_id}:{digest}"], args=args)
        except (RedisError, OSError) as error:
            raise CallCounterUnavailableError from error
        return int(cast("int", count))
