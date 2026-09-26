"""Magnific — the vendor's ComfyUI pack, ported into CNP.

Fifteen nodes over Magnific's hosted MCP surface: generate image / video /
music, the upscaler, voiceover, skin enhancer, background removal, retouch,
stock search, and the pickers that browse a Magnific library. Everything runs
on Magnific's servers against a signed-in account, so these nodes do nothing
useful offline — that is the nature of the service, not a fault in the port.

Ported rather than left in third_party/ because third_party/ is gitignored:
anything that lives there is simply absent on the Linux box. Four things
changed on the way in, each recorded at the site that changed:

  config.py         the version was read from a sibling pyproject.toml that
                    does not exist here, and the failure path returned
                    "0.0.0" — below every possible floor.
  update_check.py   a CDN manifest could refuse to let the nodes run. It now
                    warns instead. Source in this repository does not ask a
                    remote file for permission.
  js/magnific/      one directory deeper than upstream, so the ComfyUI imports
                    needed a third "../". A short count 404s the file, and a
                    404 in one pack module takes the pack's whole front-end
                    down with it.
  this file         upstream registered routes behind `except ImportError`.
                    PromptServer imports fine during a headless load and then
                    has no `.instance`, which raises AttributeError, which
                    nothing caught — and one unguarded import at pack level
                    removes every CNP node from /object_info, not just these.

Sign-in lives at ~/.magnific/comfyui_auth.json, the same path the vendor's own
pack uses, so a machine that has signed in once stays signed in across both.

Upstream: magnific-comfyui 0.7.0, https://www.magnific.com/plugins
Licence: declared UNLICENSED by the vendor (all rights reserved) — see
CREDITS.md. This is a private workspace port, not redistribution.
"""

from __future__ import annotations

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_log = logging.getLogger("MEC")


def _register_routes() -> None:
    """Mount /magnific/* on the ComfyUI server, if there is one.

    Every step is optional. Importing the pack must never depend on a running
    server: the node schemas are read during a headless load too, and these
    routes only matter to the sign-in panel in the browser.
    """
    try:
        from server import PromptServer  # type: ignore
    except Exception:  # no ComfyUI server in this process
        return
    instance = getattr(PromptServer, "instance", None)
    routes = getattr(instance, "routes", None) if instance is not None else None
    if routes is None:
        # Imported before the server was constructed. The nodes still load;
        # only the sign-in panel is unavailable, and it says so.
        _log.debug("[MEC] Magnific: no PromptServer yet, sign-in routes skipped")
        return
    try:
        from .routes import register
    except Exception as exc:  # aiohttp missing, or a syntax error in routes
        _log.warning("[MEC] Magnific sign-in routes unavailable: %s", exc)
        return
    try:
        register(routes)
    except Exception as exc:
        # A duplicate mount on a reload is the common case and is harmless.
        _log.debug("[MEC] Magnific routes not mounted: %s", exc)


_register_routes()
