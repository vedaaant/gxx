"""Per-device-token sliding-window rate limiter (in-process).

Sized to survive a full demo session; keys never touch the client, so this is the
main guard against a shared public relay being drained. Not distributed — fine for
a single relay instance at hackathon scale.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int = 120, window_secs: float = 60.0, clock=time.monotonic):
        self.max_requests = max_requests
        self.window = window_secs
        self._clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, token: str) -> bool:
        now = self._clock()
        with self._lock:
            q = self._hits[token]
            cutoff = now - self.window
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True

    def remaining(self, token: str) -> int:
        now = self._clock()
        with self._lock:
            q = self._hits[token]
            cutoff = now - self.window
            while q and q[0] < cutoff:
                q.popleft()
            return max(0, self.max_requests - len(q))
