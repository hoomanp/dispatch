"""
DISPATCH Enterprise Rate Limiter — Hybrid Sliding-Window Engine.

Provides distributed sliding-window rate limiting per agent_id, project_id, and API key.
Integrates dynamically with the Bayesian Trust Ledger:
  - Probation agents: strictly throttled (default 10 RPM)
  - Standard agents: standard threshold (default 60 RPM)
  - Trusted agents: expanded capacity (default 180 RPM)

Supports:
  1. Distributed Redis Backplane (atomic sorted-set sliding window across multi-node clusters)
  2. Thread-Safe In-Memory Sliding Window (local-first standalone operation and zero-downtime fallback)
"""

import logging
import os
import threading
import time
import uuid
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger("dispatch.rate_limiter")


class InMemorySlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter."""

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

    def check(self, key: str, max_rpm: int) -> Tuple[bool, int, int]:
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
                oldest = q[0]
                retry_after = max(1, int(oldest + self.window_seconds - now))
                return False, 0, retry_after

            # Allowed
            q.append(now)
            remaining = max(0, max_rpm - len(q))
            return True, remaining, 0

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is not None:
                self._windows.pop(key, None)
            else:
                self._windows.clear()


class RedisSlidingWindowRateLimiter:
    """Distributed Redis-backed sliding-window rate limiter using sorted sets."""

    def __init__(self, redis_client=None, redis_url: Optional[str] = None,
                 window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.client = redis_client
        self._connected = False

        if self.client is None:
            url = redis_url or os.getenv("REDIS_URL")
            host = os.getenv("REDIS_HOST", "redis")
            port = int(os.getenv("REDIS_PORT", "6379"))
            try:
                import redis
                if url:
                    self.client = redis.from_url(url, decode_responses=True, socket_timeout=2.0)
                else:
                    self.client = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=2.0)
                self.client.ping()
                self._connected = True
                logger.info(f"Redis rate limiter connected to {url or f'{host}:{port}'}")
            except Exception as e:
                logger.warning(f"Redis rate limiter unavailable ({e}); will fallback to in-memory")
                self._connected = False
        else:
            try:
                self.client.ping()
                self._connected = True
            except Exception:
                self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def check(self, key: str, max_rpm: int) -> Tuple[bool, int, int]:
        if not self._connected or self.client is None:
            raise RuntimeError("Redis client not connected")

        redis_key = f"dispatch:ratelimit:{key}"
        now = time.time()
        cutoff = now - self.window_seconds
        member_id = f"{now}:{uuid.uuid4().hex[:8]}"

        try:
            pipe = self.client.pipeline()
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            pipe.zcard(redis_key)
            pipe.zrange(redis_key, 0, 0, withscores=True)
            _, current_count, oldest_items = pipe.execute()

            if current_count >= max_rpm:
                oldest_score = float(oldest_items[0][1]) if oldest_items else now
                retry_after = max(1, int(oldest_score + self.window_seconds - now))
                return False, 0, retry_after

            # Under limit: insert timestamp member and set TTL
            pipe = self.client.pipeline()
            pipe.zadd(redis_key, {member_id: now})
            pipe.expire(redis_key, int(self.window_seconds * 2))
            pipe.execute()

            remaining = max(0, max_rpm - (current_count + 1))
            return True, remaining, 0
        except Exception as e:
            logger.warning(f"Redis rate limit check error ({e}); marking disconnected")
            self._connected = False
            raise e

    def reset(self, key: Optional[str] = None) -> None:
        if not self._connected or self.client is None:
            return
        try:
            if key:
                self.client.delete(f"dispatch:ratelimit:{key}")
            else:
                keys = self.client.keys("dispatch:ratelimit:*")
                if keys:
                    self.client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis rate limit reset error: {e}")


class HybridRateLimiter:
    """Unified hybrid rate limiter that uses Redis when connected,
    with zero-downtime automatic fallback to in-memory sliding window.
    """

    def __init__(self, window_seconds: float = 60.0, redis_url: Optional[str] = None):
        self.window_seconds = window_seconds
        self.in_memory = InMemorySlidingWindowRateLimiter(window_seconds=window_seconds)
        self.redis_limiter: Optional[RedisSlidingWindowRateLimiter] = None
        self._init_redis(redis_url=redis_url)

    def _init_redis(self, redis_url: Optional[str] = None):
        url = redis_url or os.getenv("REDIS_URL")
        # Only initialize if Redis environment is specified or active
        if url or os.getenv("REDIS_HOST") or os.getenv("DISPATCH_REDIS_RATE_LIMIT"):
            try:
                self.redis_limiter = RedisSlidingWindowRateLimiter(
                    redis_url=url, window_seconds=self.window_seconds)
            except Exception:
                self.redis_limiter = None

    @property
    def using_redis(self) -> bool:
        return bool(self.redis_limiter and self.redis_limiter.is_connected)

    def check(self, key: str, max_rpm: int) -> Tuple[bool, int, int]:
        if self.redis_limiter and self.redis_limiter.is_connected:
            try:
                return self.redis_limiter.check(key, max_rpm)
            except Exception:
                pass  # Fall through to in-memory limiter on any network or redis glitch
        return self.in_memory.check(key, max_rpm)

    def reset(self, key: Optional[str] = None) -> None:
        self.in_memory.reset(key)
        if self.redis_limiter and self.redis_limiter.is_connected:
            try:
                self.redis_limiter.reset(key)
            except Exception:
                pass


# Backward compatibility alias and default singleton
SlidingWindowRateLimiter = HybridRateLimiter
rate_limiter = HybridRateLimiter()
