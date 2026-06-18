"""In-memory TTL cache for prompt search results.

Deliberately minimal — a SQLite-backed cache will replace this when source
count grows past 3.  For now we want zero new dependencies and zero on-disk
state.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Any

_TTL_SECONDS = 6 * 3600  # 6 hours
_MAX_ENTRIES = 256

_lock = Lock()
_cache: "dict[str, tuple[float, Any]]" = {}


def get(key: str) -> Any | None:
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.time():
            _cache.pop(key, None)
            return None
        return value


def put(key: str, value: Any, *, ttl: int = _TTL_SECONDS) -> None:
    with _lock:
        if len(_cache) >= _MAX_ENTRIES:
            # Evict oldest expiry
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest_key, None)
        _cache[key] = (time.time() + ttl, value)


def clear() -> None:
    with _lock:
        _cache.clear()


def stats() -> dict[str, int]:
    with _lock:
        now = time.time()
        live = sum(1 for exp, _ in _cache.values() if exp >= now)
        return {"entries": len(_cache), "live": live, "max": _MAX_ENTRIES}
