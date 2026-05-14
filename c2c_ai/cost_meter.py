"""Cost meter — tracks token usage and enforces daily caps.

Persists usage to ``~/.c2c/usage.jsonl`` so the dashboard survives ComfyUI
restarts. One line per call; rotation is unnecessary at the expected volume.

Hard cap behaviour:
    * Soft warn at 80 % of cap → emits a warning event, allows call
    * At 100 % of cap → BLOCKS further cloud calls until midnight local time
    * Local-tier calls are recorded for analytics but never blocked
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .types import Tier

log = logging.getLogger("c2c_ai.cost")


def _config_dir() -> Path:
    base = os.environ.get("C2C_HOME") or str(Path.home() / ".c2c")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


USAGE_LOG = _config_dir() / "usage.jsonl"
CONFIG_FILE = _config_dir() / "cost_config.json"

DEFAULT_DAILY_CAP_USD = 1.00


@dataclass
class UsageEntry:
    ts: float
    feature: str
    backend_id: str
    tier: str            # cloud | local
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class CostConfig:
    daily_cap_usd: float = DEFAULT_DAILY_CAP_USD
    warn_threshold: float = 0.8        # fraction of cap
    enabled: bool = True

    @classmethod
    def load(cls) -> "CostConfig":
        if CONFIG_FILE.is_file():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception as exc:
                log.warning("cost config load failed: %s — using defaults", exc)
        return cls()

    def save(self) -> None:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


# ---------- core meter --------------------------------------------------------

class CostMeter:
    """Singleton — accessed via :func:`get_meter`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config = CostConfig.load()
        self._today_key = _today_key()
        self._today_cost: float = 0.0
        self._today_calls: int = 0
        self._recompute_today()

    # -- public surface --------------------------------------------------------

    @property
    def config(self) -> CostConfig:
        return self._config

    def set_daily_cap(self, usd: float) -> None:
        with self._lock:
            self._config.daily_cap_usd = max(0.0, float(usd))
            self._config.save()

    def can_spend(self, est_cost_usd: float) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. Local calls are always allowed."""
        with self._lock:
            self._roll_day_if_needed()
            cap = self._config.daily_cap_usd
            if cap <= 0:
                return True, "no cap configured"
            projected = self._today_cost + est_cost_usd
            if projected > cap:
                return False, (
                    f"daily cap reached: spent ${self._today_cost:.4f} "
                    f"of ${cap:.2f}, this call would add ${est_cost_usd:.4f}"
                )
            return True, "ok"

    def record(self, entry: UsageEntry) -> None:
        with self._lock:
            self._roll_day_if_needed()
            # Append JSONL — durable, append-only, easy to tail
            try:
                with open(USAGE_LOG, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(entry)) + "\n")
            except OSError as exc:
                log.warning("failed to append usage log: %s", exc)
            if entry.tier == Tier.CLOUD.value:
                self._today_cost += entry.cost_usd
            self._today_calls += 1

    def snapshot(self) -> dict:
        with self._lock:
            self._roll_day_if_needed()
            cap = self._config.daily_cap_usd
            used = self._today_cost
            pct = (used / cap) if cap > 0 else 0.0
            return {
                "today_cost_usd": round(used, 6),
                "today_calls": self._today_calls,
                "cap_usd": cap,
                "fraction_used": round(pct, 4),
                "over_warn": pct >= self._config.warn_threshold,
                "over_cap": cap > 0 and used >= cap,
                "day_key": self._today_key,
            }

    # -- internals -------------------------------------------------------------

    def _roll_day_if_needed(self) -> None:
        k = _today_key()
        if k != self._today_key:
            self._today_key = k
            self._today_cost = 0.0
            self._today_calls = 0
            self._recompute_today()

    def _recompute_today(self) -> None:
        # On startup we replay today's entries so cap survives restarts.
        if not USAGE_LOG.is_file():
            return
        try:
            with open(USAGE_LOG, "r", encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        e = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if _day_key_of(e.get("ts", 0.0)) != self._today_key:
                        continue
                    self._today_calls += 1
                    if e.get("tier") == Tier.CLOUD.value:
                        self._today_cost += float(e.get("cost_usd", 0.0))
        except OSError as exc:
            log.warning("failed to replay usage log: %s", exc)


def _today_key() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


def _day_key_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().strftime("%Y-%m-%d")


_METER: CostMeter | None = None


def get_meter() -> CostMeter:
    global _METER
    if _METER is None:
        _METER = CostMeter()
    return _METER


# ---------- helpers for backends ---------------------------------------------

def estimate_cost(backend_info, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from a backend's price card."""
    inp = (input_tokens / 1000.0) * backend_info.cost_per_1k_input
    out = (output_tokens / 1000.0) * backend_info.cost_per_1k_output
    return inp + out


def make_entry(*, feature: str, backend_info, model: str,
               input_tokens: int, output_tokens: int, latency_ms: float) -> UsageEntry:
    return UsageEntry(
        ts=time.time(),
        feature=feature,
        backend_id=backend_info.id,
        tier=backend_info.tier.value,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost(backend_info, input_tokens, output_tokens),
        latency_ms=latency_ms,
    )
