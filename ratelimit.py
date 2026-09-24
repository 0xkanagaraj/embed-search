"""
ratelimit.py — minimal in-process sliding-window rate limiter.

Scope / limitations:
  This is intentionally dependency-free and keeps its state in a plain
  dict, so it's per-process. That's enough to stop naive brute-force
  scripts against a single-worker deployment (the default `python
  server.py` / single uvicorn worker described in the README). If you
  scale to multiple worker processes or containers, replace this with a
  shared store (Redis `INCR`+`EXPIRE`, or similar) so limits are
  enforced across all of them — otherwise each worker gets its own
  independent quota.
"""

import time
from collections import defaultdict, deque

# key → deque of timestamps (seconds) of recent hits
_HITS: dict[str, deque] = defaultdict(deque)


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded, retry after {retry_after:.0f}s")


def check(key: str, max_hits: int, window_seconds: float) -> None:
    """
    Record a hit for `key` and raise RateLimitExceeded if it has been hit
    more than `max_hits` times in the trailing `window_seconds`.
    """
    now = time.monotonic()
    dq = _HITS[key]
    cutoff = now - window_seconds
    while dq and dq[0] < cutoff:
        dq.popleft()

    if len(dq) >= max_hits:
        retry_after = window_seconds - (now - dq[0])
        raise RateLimitExceeded(max(retry_after, 1.0))

    dq.append(now)


def reset(key: str) -> None:
    """Clear a key's history (e.g. call after a successful login)."""
    _HITS.pop(key, None)
