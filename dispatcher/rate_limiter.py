"""
DISPATCH Enterprise Rate Limiter — Phase 2 Blast-Radius Control.

Provides sliding-window rate limiting per agent_id and API key.
Integrates dynamically with the Trust Ledger:
  - Probation agents: strictly throttled (default 10 RPM)
  - Standard agents: standard threshold (default 60 RPM)
  - Trusted agents: expanded capacity (default 180 RPM)

In-memory sliding window with automatic stale key eviction;
Redis-compatible interface for distributed multi-instance deployment.
"""

import time
import threading
from collections import deque
from typing import Optional


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = {}
        self._last_cleanup = time.time()

    def _cleanup_stale(self, now: float) -> None:
        """Evict keys that haven't been active in 2x the window duration."""
        if now - self._last_cleanup < self.window_seconds:
            return
        self._last_cleanup = now
        cutoff = now - (self.window_seconds * 2)
        stale_keys = [
            k for k, q in self._windows.items()
            if not q or q[-1] < cutoff
        ]
        for k in stale_keys:
            self._windows.pop(k, None)

    def check(self, key: str, max_rpm: int) -> tuple[bool, int, int]:
        """Check whether a request is allowed under the rate limit.

        Args:
            key: Rate limit identifier (e.g., "agent:nightly-bot" or "key:sk-...")
            max_rpm: Maximum allowed requests per minute.

        Returns:
            Tuple of:
              - allowed (bool): True if request can proceed, False if throttled
              - remaining (int): Remaining requests available in the current window
              - retry_after (int): Seconds until the oldest request expires (if throttled)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            self._cleanup_stale(now)
            q = self._windows.setdefault(key, deque())

            # Drop timestamps outside the sliding window
            while q and q[0] <= cutoff:
                q.popleft()

            current_count = len(q)
            if current_count >= max_rpm:
                # Throttled: calculate time until earliest request expires
                oldest = q[0]
                retry_after = max(1, int(oldest + self.window_seconds - now))
                return False, 0, retry_after

            # Allowed: record request timestamp
            q.append(now)
            remaining = max(0, max_rpm - len(q))
            return True, remaining, 0

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limiter state (useful in tests)."""
        with self._lock:
            if key is not None:
                self._windows.pop(key, None)
            else:
                self._windows.clear()


# Default singleton instance
rate_limiter = SlidingWindowRateLimiter()
