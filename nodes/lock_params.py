"""Lock parameter values - the server side (V6-A, owner 2026-11-21; design docs/h2vault/V6_cnp_vault_enhancements.md).

The front end (js/c2c_lock_params.js) keeps a locked node's values encrypted in the workflow and puts the real values
only into the one request that queues a prompt, marking each such entry with `_meta.c2c_locked = [input names]`.
The server has to see the values to run them; this module keeps them out of what it stores or hands out:

  history          PromptQueue.task_done(..., process_item=) is chained: the stored prompt is redacted
  hidden PROMPT    DynamicPrompt.get_original_prompt returns a redacted copy (SaveImage's PNG metadata, metadata
                   nodes); execution reads `original_prompt` / get_node directly and is unchanged
  /queue, /api/jobs  get_current_queue / get_current_queue_volatile return redacted copies

Verified against core 0.39 (execution.py PromptQueue.task_done + hidden PROMPT, main.py remove_sensitive,
server.py queue routes). A prompt without the marker passes through untouched (same object).

License: Apache-2.0
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("c2c.lock_params")

LOCKED = "<locked>"
MARKER = "c2c_locked"
_CACHE_MAX = 4

_state: dict[str, Any] = {"patches": [], "cache": []}


def _is_link(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def _marked(entry) -> list:
    if not isinstance(entry, dict):
        return []
    meta = entry.get("_meta")
    names = meta.get(MARKER) if isinstance(meta, dict) else None
    return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []


def redact_prompt(prompt):
    """A copy of `prompt` whose marked inputs read "<locked>" (links stay); the same object when nothing is marked."""
    if not isinstance(prompt, dict) or not any(_marked(e) for e in prompt.values()):
        return prompt
    out = {}
    for nid, entry in prompt.items():
        names = _marked(entry)
        inputs = entry.get("inputs") if isinstance(entry, dict) else None
        if not names or not isinstance(inputs, dict):
            out[nid] = entry
            continue
        new_inputs = dict(inputs)
        for n in names:
            if n in new_inputs and not _is_link(new_inputs[n]):
                new_inputs[n] = LOCKED
        out[nid] = dict(entry, inputs=new_inputs)
    return out


def _redact_item(item):
    """A queue item (number, prompt_id, prompt, extra_data, outputs[, sensitive]) with its prompt redacted."""
    if isinstance(item, (list, tuple)) and len(item) > 2:
        red = redact_prompt(item[2])
        if red is not item[2]:
            seq = list(item)
            seq[2] = red
            return tuple(seq) if isinstance(item, tuple) else seq
    return item


_MISSING = object()


def _patch(obj, name, new):
    """Set obj.name and remember what obj ITSELF held (its own __dict__ entry, or nothing) so it can be undone."""
    own = obj.__dict__.get(name, _MISSING) if hasattr(obj, "__dict__") else _MISSING
    _state["patches"].append((obj, name, own))
    setattr(obj, name, new)


def _cached_redaction(prompt):
    for ref, red in _state["cache"]:
        if ref is prompt:
            return red
    red = redact_prompt(prompt)
    if red is not prompt:
        _state["cache"].append((prompt, red))
        del _state["cache"][:-_CACHE_MAX]
    return red


def _install_dynamic_prompt(cls) -> None:
    if cls is None:
        try:
            from comfy_execution.graph import DynamicPrompt as cls  # noqa: N813
        except Exception:  # noqa: BLE001 - no core here (tests, tools): nothing to patch
            return
    if getattr(cls, "__c2c_lock_params", False):
        return
    orig = cls.get_original_prompt

    def get_original_prompt(self, _orig=orig):
        p = _orig(self)
        try:
            return _cached_redaction(p)
        except Exception:  # noqa: BLE001 - never break execution
            log.warning("C2C lock parameters: PROMPT redaction failed", exc_info=True)
            return p

    _patch(cls, "get_original_prompt", get_original_prompt)
    _patch(cls, "__c2c_lock_params", True)


def install(server, dynamic_prompt_cls=None) -> bool:
    """Wrap the queue's history / views and the hidden PROMPT once. False (never an exception) without a queue."""
    q = getattr(server, "prompt_queue", None)
    if q is None or not hasattr(q, "task_done"):
        return False
    try:
        if not getattr(q, "__c2c_lock_params", False):
            orig_done = q.task_done

            def task_done(item_id, history_result, status, *a, _orig=orig_done, **k):
                core = k.pop("process_item", a[0] if a else None)
                rest = a[1:] if a else ()

                def process_item(item, _core=core):
                    item = _core(item) if _core is not None else item
                    try:
                        return _redact_item(item)
                    except Exception:  # noqa: BLE001
                        log.warning("C2C lock parameters: history redaction failed", exc_info=True)
                        return item
                return _orig(item_id, history_result, status, process_item, *rest, **k)

            _patch(q, "task_done", task_done)
            for name in ("get_current_queue", "get_current_queue_volatile"):
                if hasattr(q, name):
                    _patch(q, name, _wrap_view(getattr(q, name)))
            _patch(q, "__c2c_lock_params", True)
        _install_dynamic_prompt(dynamic_prompt_cls)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("C2C lock parameters: server redaction not installed: %s", exc)
        return False


def _wrap_view(orig):
    def view(*a, _orig=orig, **k):
        out = _orig(*a, **k)
        try:
            running, pending = out
            return [_redact_item(i) for i in running], [_redact_item(i) for i in pending]
        except Exception:  # noqa: BLE001
            log.warning("C2C lock parameters: queue view redaction failed", exc_info=True)
            return out
    return view


def _reset_for_tests() -> None:
    """Undo every patch (newest first) and clear the cache."""
    for obj, name, own in reversed(_state["patches"]):
        try:
            if own is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, own)
        except Exception:  # noqa: BLE001
            pass
    _state["patches"].clear()
    _state["cache"].clear()
