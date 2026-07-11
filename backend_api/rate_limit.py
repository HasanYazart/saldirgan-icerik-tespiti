"""Tek süreçli kurulumlar için küçük bir kayan-pencere oran sınırlayıcı."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:
    Redis = None
    RedisError = Exception


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                raise HTTPException(
                    status_code=429,
                    detail="Çok fazla istek gönderildi. Lütfen daha sonra tekrar deneyin.",
                    headers={"Retry-After": str(retry_after)},
                )
            events.append(now)


class RedisRateLimiter:
    """Birden fazla worker/sunucu tarafından paylaşılan sabit-pencere limiter."""

    def __init__(self, redis_url: str, limit: int, window_seconds: int):
        if Redis is None:
            raise RuntimeError("REDIS_URL ayarlı ancak redis paketi kurulu değil.")
        self.client = Redis.from_url(redis_url, decode_responses=True)
        self.limit = limit
        self.window_seconds = window_seconds

    def check(self, key: str) -> None:
        bucket = int(time.time()) // self.window_seconds
        redis_key = f"moderation-rate:{bucket}:{key}"
        try:
            count = self.client.incr(redis_key)
            if count == 1:
                self.client.expire(redis_key, self.window_seconds + 2)
        except RedisError as error:
            raise HTTPException(status_code=503, detail="Oran sınırı servisi kullanılamıyor.") from error
        if count > self.limit:
            raise HTTPException(
                status_code=429,
                detail="Çok fazla istek gönderildi. Lütfen daha sonra tekrar deneyin.",
                headers={"Retry-After": str(self.window_seconds)},
            )


def build_rate_limiter(redis_url: str, limit: int, window_seconds: int):
    if redis_url:
        return RedisRateLimiter(redis_url, limit, window_seconds)
    return SlidingWindowRateLimiter(limit, window_seconds)
