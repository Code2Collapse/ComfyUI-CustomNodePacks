"""Router — the only entry point features should use.

Public surface:
    ask(feature, messages, **kw) -> AskResponse
    stream(feature, messages, **kw) -> generator[str]
    get_router() -> Router        (for advanced use / settings UI)
    get_status() -> dict          (for status bar HUD)
    get_cost_today() -> dict      (for status bar HUD)

Responsibilities:
    1. Resolve the effective policy for the feature.
    2. Pick a healthy backend that satisfies required capabilities and policy.
    3. Redact messages if heading to a cloud backend.
    4. Enforce the cost cap (cloud only).
    5. Dispatch, record usage, surface latency.
    6. Fall back on transient failure.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from typing import Generator, Iterable

from . import policy as policy_mod
from . import redactor
from .backends.base import Backend
from .cost_meter import get_meter, make_entry
from .types import (
    AskRequest,
    AskResponse,
    BackendInfo,
    Capability,
    Message,
    Policy,
    Sensitivity,
    Tier,
)

log = logging.getLogger("c2c_ai.router")


class Router:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._order: list[str] = []                # registration order, used for tie-breaks
        self._lock = threading.RLock()
        self._probe_thread: threading.Thread | None = None
        self._probe_stop = threading.Event()
        self._meter = get_meter()

    # ============================================================= registration
    def register(self, backend: Backend, *, default: bool = False) -> None:
        with self._lock:
            if backend.info.id in self._backends:
                log.info("replacing backend %s", backend.info.id)
                self._order = [x for x in self._order if x != backend.info.id]
            self._backends[backend.info.id] = backend
            if default:
                self._order.insert(0, backend.info.id)
            else:
                self._order.append(backend.info.id)
            log.info("registered %s (%s)", backend.info.id, backend.info.tier.value)

    def unregister(self, backend_id: str) -> None:
        with self._lock:
            self._backends.pop(backend_id, None)
            self._order = [x for x in self._order if x != backend_id]

    def all_backends(self) -> list[Backend]:
        with self._lock:
            return [self._backends[b] for b in self._order if b in self._backends]

    def get(self, backend_id: str) -> Backend | None:
        return self._backends.get(backend_id)

    # =========================================================== probing
    def probe_all(self, parallel: bool = True) -> dict[str, dict]:
        """Refresh health for every backend. Returns ``{backend_id: health_dict}``."""
        backends = self.all_backends()
        if parallel and len(backends) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, len(backends))) as ex:
                ex.map(lambda b: b.probe(), backends)
        else:
            for b in backends:
                try:
                    b.probe()
                except Exception as exc:
                    log.warning("probe %s failed: %s", b.info.id, exc)
        return {b.info.id: _health_dict(b) for b in backends}

    def start_periodic_probe(self, interval_s: float = 60.0) -> None:
        if self._probe_thread and self._probe_thread.is_alive():
            return
        self._probe_stop.clear()

        def loop() -> None:
            # First probe runs immediately so the status bar isn't blank.
            try:
                self.probe_all()
            except Exception as exc:
                log.warning("initial probe failed: %s", exc)
            while not self._probe_stop.wait(interval_s):
                try:
                    self.probe_all()
                except Exception as exc:
                    log.warning("periodic probe failed: %s", exc)

        self._probe_thread = threading.Thread(target=loop, name="c2c-ai-probe", daemon=True)
        self._probe_thread.start()

    def stop_periodic_probe(self) -> None:
        self._probe_stop.set()

    # ============================================================ selection
    # Track D.1 — features for which we transparently fall back to the
    # deterministic rule-pack backend when no LLM is healthy. Anything
    # error-explanation-shaped should never raise "no backend available"
    # because we always have the offline rule pack to lean on.
    _DETERMINISTIC_FALLBACK_FEATURES = frozenset({
        "doctor.explain",
        "error.explain",
        "plain.english",
        "error_translator",
        "error_assistant",
    })

    def select(self, feature: str, required: set[Capability],
               policy_override: Policy | None = None) -> Backend:
        """Pick the best backend or raise."""
        eff_policy = policy_mod.resolve(feature, policy_override)
        candidates = self._filter(required, eff_policy)
        if not candidates:
            # Track D.1 — graceful fallback to the deterministic rule pack
            # for explanation features. Only kicks in when the requested
            # policy isn't already one that excludes deterministic (the
            # explicit *_ONLY policies are user choices we must honor).
            if (feature in self._DETERMINISTIC_FALLBACK_FEATURES
                    and eff_policy not in (Policy.CLOUD_ONLY, Policy.LOCAL_ONLY)):
                det = self._deterministic_backends(required)
                if det:
                    log.info("no LLM backend for feature=%s — falling back to %s",
                             feature, det[0].info.id)
                    return det[0]
            raise RuntimeError(
                f"no backend available for feature={feature} policy={eff_policy.value} "
                f"required={sorted(c.value for c in required)}. "
                "Open Settings → C2C → AI Backends to configure one."
            )
        # ``candidates`` is already ordered by policy preference; pick first healthy.
        healthy = [c for c in candidates if c.health.ok]
        if healthy:
            return healthy[0]
        # All unhealthy — return the most-recently-probed one so we still try (and surface the error)
        candidates.sort(key=lambda c: c.health.last_probe_at, reverse=True)
        return candidates[0]

    def _deterministic_backends(self, required: set[Capability]) -> list[Backend]:
        return [b for b in self.all_backends()
                if b.info.enabled
                and b.info.tier == Tier.DETERMINISTIC
                and required.issubset(b.info.capabilities)
                and b.health.ok]

    def _filter(self, required: set[Capability], pol: Policy) -> list[Backend]:
        backends = [b for b in self.all_backends() if b.info.enabled]
        # capability filter
        backends = [b for b in backends if required.issubset(b.info.capabilities)]
        # Bucket by tier
        cloud = [b for b in backends if b.info.tier == Tier.CLOUD]
        local = [b for b in backends if b.info.tier == Tier.LOCAL]
        deterministic = [b for b in backends if b.info.tier == Tier.DETERMINISTIC]
        if pol == Policy.DETERMINISTIC_ONLY:
            return deterministic
        if pol == Policy.CLOUD_ONLY:
            return cloud
        if pol == Policy.LOCAL_ONLY:
            return local
        if pol == Policy.PREFER_LOCAL:
            return local + cloud
        if pol == Policy.PREFER_CLOUD:
            return cloud + local
        # AUTO: cheapest first. Deterministic is intentionally NOT in this list
        # — it is reserved as an explicit-opt-in tier (DETERMINISTIC_ONLY) or
        # as a no-LLM-available fallback inside select(), so that AUTO doesn't
        # silently route every chat call to the rule pack just because it's
        # the cheapest backend.
        local_sorted = sorted(local, key=lambda b: b.info.cost_per_1k_input)
        cloud_sorted = sorted(cloud, key=lambda b: b.info.cost_per_1k_input)
        return local_sorted + cloud_sorted

    # ================================================================ ask
    def ask(self, feature: str, messages: list[Message], **kw) -> AskResponse:
        req = self._build_req(feature, messages, **kw)
        backend, redacted_req, redacted_flag = self._prepare(req)
        return self._dispatch(backend, redacted_req, redacted_flag, feature)

    def stream(self, feature: str, messages: list[Message], **kw) -> Generator[str, None, AskResponse]:
        req = self._build_req(feature, messages, **kw)
        backend, redacted_req, redacted_flag = self._prepare(req)

        t0 = time.monotonic()
        gen = backend.stream(redacted_req)
        resp: AskResponse | None = None
        try:
            while True:
                try:
                    chunk = next(gen)
                except StopIteration as stop:
                    resp = stop.value
                    break
                yield chunk
        finally:
            if resp is None:
                # generator exited without returning a response — synthesise one
                resp = AskResponse(
                    text="", backend_id=backend.info.id, model=backend.info.model,
                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                )
            resp.redacted = redacted_flag
            self._meter.record(make_entry(
                feature=feature, backend_info=backend.info, model=backend.info.model,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                latency_ms=resp.latency_ms,
            ))
        return resp

    # ----------------------------------------------------------- internals
    def _build_req(self, feature: str, messages: list[Message], **kw) -> AskRequest:
        return AskRequest(
            feature=feature,
            messages=messages,
            sensitivity=kw.get("sensitivity", Sensitivity.NORMAL),
            required=kw.get("required") or {Capability.CHAT},
            max_tokens=kw.get("max_tokens", 1024),
            temperature=kw.get("temperature", 0.4),
            policy_override=kw.get("policy_override"),
            json_schema=kw.get("json_schema"),
        )

    def _prepare(self, req: AskRequest) -> tuple[Backend, AskRequest, bool]:
        backend = self.select(req.feature, req.required, req.policy_override)
        redacted_flag = False
        redacted_msgs = req.messages

        if backend.info.tier == Tier.CLOUD:
            if req.sensitivity == Sensitivity.SENSITIVE:
                raise RuntimeError(
                    "feature marked SENSITIVE refuses cloud backend without explicit user opt-in. "
                    "Either select a local backend or pass sensitivity=NORMAL after confirming."
                )
            redacted_msgs, redacted_flag = redactor.redact_messages(req.messages, req.sensitivity)
            # cost gate (cheap pre-check: assume worst-case 2× current input tokens for output)
            est_in = sum(len(m.content) for m in redacted_msgs) // 4
            est_out = req.max_tokens
            est_cost = (est_in / 1000.0) * backend.info.cost_per_1k_input + \
                       (est_out / 1000.0) * backend.info.cost_per_1k_output
            ok, reason = self._meter.can_spend(est_cost)
            if not ok:
                # Try local fallback
                fallback = self._filter(req.required, Policy.LOCAL_ONLY)
                fallback = [b for b in fallback if b.health.ok]
                if fallback:
                    log.warning("cost cap blocked %s: %s — falling back to %s",
                                backend.info.id, reason, fallback[0].info.id)
                    backend = fallback[0]
                    redacted_msgs = req.messages
                    redacted_flag = False
                else:
                    raise RuntimeError(f"daily cost cap reached and no local fallback: {reason}")

        new_req = AskRequest(
            feature=req.feature,
            messages=redacted_msgs,
            sensitivity=req.sensitivity,
            required=req.required,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            policy_override=req.policy_override,
            json_schema=req.json_schema,
        )
        return backend, new_req, redacted_flag

    def _dispatch(self, backend: Backend, req: AskRequest,
                  redacted_flag: bool, feature: str) -> AskResponse:
        try:
            resp = backend.ask(req)
        except Exception as exc:
            log.warning("backend %s failed: %s — trying next", backend.info.id, exc)
            # one-shot fallback to any other healthy backend matching the policy
            try:
                pol = policy_mod.resolve(feature, req.policy_override)
                alts = [b for b in self._filter(req.required, pol)
                        if b is not backend and b.health.ok]
                if alts:
                    backend = alts[0]
                    resp = backend.ask(req)
                else:
                    raise
            except Exception:
                raise exc

        resp.redacted = redacted_flag
        self._meter.record(make_entry(
            feature=feature, backend_info=backend.info, model=backend.info.model,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
        ))
        return resp

    # =============================================================== status
    def status(self) -> dict:
        with self._lock:
            return {
                "backends": [
                    {
                        "id": b.info.id,
                        "tier": b.info.tier.value,
                        "display_name": b.info.display_name,
                        "model": b.info.model,
                        "enabled": b.info.enabled,
                        "capabilities": sorted(c.value for c in b.info.capabilities),
                        "cost_per_1k_input": b.info.cost_per_1k_input,
                        "cost_per_1k_output": b.info.cost_per_1k_output,
                        "health": _health_dict(b),
                    }
                    for b in self.all_backends()
                ],
                "policies": [
                    {
                        "feature": e.feature,
                        "default": e.default.value,
                        "override": e.override.value if e.override else None,
                        "effective": e.effective.value,
                    }
                    for e in policy_mod.listing()
                ],
                "cost": self._meter.snapshot(),
            }


# ============================================================ module singleton

_ROUTER: Router | None = None
_INIT_LOCK = threading.Lock()


def get_router() -> Router:
    global _ROUTER
    if _ROUTER is None:
        with _INIT_LOCK:
            if _ROUTER is None:
                _ROUTER = Router()
    return _ROUTER


def ask(feature: str, messages: list[Message], **kw) -> AskResponse:
    return get_router().ask(feature, messages, **kw)


def stream(feature: str, messages: list[Message], **kw) -> Generator[str, None, AskResponse]:
    return get_router().stream(feature, messages, **kw)


def get_status() -> dict:
    return get_router().status()


def get_cost_today() -> dict:
    return get_meter().snapshot()


# ============================================================ helpers

def _health_dict(b: Backend) -> dict:
    h = b.health
    return {
        "ok": h.ok,
        "last_rtt_ms": round(h.last_rtt_ms, 2),
        "last_error": h.last_error,
        "last_probe_at": h.last_probe_at,
    }
