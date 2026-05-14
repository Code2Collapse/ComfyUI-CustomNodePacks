"""
token_counter.py — Phase 10: Prompt Token Counter.

Exact CLIP token count for a given string, using whatever CLIP tokenizer is
currently loaded in the running ComfyUI session. Falls back to a fast
word-heuristic when no CLIP model has been loaded yet.

Route: POST /mec/token_count
Body:  {"text": "..."}
Reply: {"success": true, "data": {"tokens": int, "exact": bool, "limit": 77,
                                  "method": "clip"|"heuristic"}}
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict

log = logging.getLogger("MEC.token_counter")

# CLIP-ViT-L/14 (SD 1.x, SDXL CLIP-L) hard limit.
_CLIP_LIMIT = 77
# SDXL CLIP-G can hit 75-real-tokens + 2 boundary, same effective UX limit.

_WORD_SPLIT = re.compile(r"\s+|[.,;:!?()\[\]{}\"'`<>/\\]+")


def _heuristic_count(text: str) -> int:
    parts = [p for p in _WORD_SPLIT.split(text or "") if p]
    # Empirical multiplier for CLIP BPE: ≈1.3 tokens per "word"
    return int(round(len(parts) * 1.3))


def _try_exact_count(text: str) -> int | None:
    """Best-effort exact tokenization using whatever CLIP tokenizer the
    running ComfyUI process has cached. We probe a few well-known locations
    without forcing a model load.
    """
    # ComfyUI exposes sd1_clip / sdxl_clip tokenizers under comfy.sd1_clip
    try:
        from comfy import sd1_clip  # type: ignore
        # The SDTokenizer class wraps the CLIP tokenizer; instantiate cheaply.
        tk = sd1_clip.SDTokenizer()
        out = tk.tokenize_with_weights(text or "", return_word_ids=False)
        # `out` is a dict { "l": [[ (token, weight), ... ]] } in 2025+ builds,
        # OR a flat list-of-lists in older builds. Normalize:
        if isinstance(out, dict):
            seqs = []
            for v in out.values():
                if isinstance(v, list):
                    seqs.extend(v)
        else:
            seqs = out if isinstance(out, list) else []
        if not seqs:
            return None
        first = seqs[0]
        if isinstance(first, list):
            # Subtract the leading start + trailing end + pads. The tokenizer
            # pads to multiples of 77, so we count non-pad tokens.
            count = 0
            for item in first:
                # item is (token_id, weight, ...) tuple
                if isinstance(item, (tuple, list)) and item:
                    tid = item[0]
                else:
                    tid = item
                # 49407 is the CLIP end-of-text token id; pad uses the same.
                if tid in (49406,):     # start
                    continue
                if tid in (49407,):     # end / pad
                    break
                count += 1
            return count
    except Exception as e:
        log.debug("[token_counter] sd1_clip path failed: %s", e)

    # Fallback: transformers CLIPTokenizer if present
    try:
        from transformers import CLIPTokenizer  # type: ignore
        tk = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        ids = tk(text or "", add_special_tokens=False)["input_ids"]
        return len(ids)
    except Exception as e:
        log.debug("[token_counter] transformers path failed: %s", e)

    return None


def count_tokens(text: str) -> Dict[str, Any]:
    exact = _try_exact_count(text)
    if exact is not None:
        return {
            "tokens": exact,
            "exact":  True,
            "limit":  _CLIP_LIMIT,
            "method": "clip",
        }
    return {
        "tokens": _heuristic_count(text),
        "exact":  False,
        "limit":  _CLIP_LIMIT,
        "method": "heuristic",
    }


def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[token_counter] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.post("/mec/token_count")
    async def _count(req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            body = {}
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            return web.json_response(
                {"success": False, "error": "missing_text"}, status=400)
        try:
            data = count_tokens(text)
        except Exception as e:
            log.exception("[token_counter] count failed")
            return web.json_response(
                {"success": False, "error": "count_failed", "message": str(e)},
                status=500)
        return web.json_response({"success": True, "data": data})

    log.info("[token_counter] Route registered: POST /mec/token_count")
