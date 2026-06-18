"""C2C prompt library — multi-source prompt gallery cache + REST routes.

Exposes ``register_routes(server)`` for wiring into the top-level pack
__init__.py.  Endpoints surface under ``/c2c/prompts/*``.

Sources today: lexica (public search API, no auth).
Sources planned: civitai, openart (REST APIs); promptdexter, prompthero,
playgroundai, magespace (scraped on opt-in).
"""
from __future__ import annotations

from .api_routes import register_routes  # noqa: F401

__all__ = ["register_routes"]
