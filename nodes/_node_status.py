"""Per-node-type status for the front-end layout engine (A28, ORDERS 29; js/_c2c_node_layout.js).

An input declared with ``"c2c_show_if_status": "<key>"`` is shown only while ``GET /c2c/node_status?type=<NodeType>``
answers ``{"status": {"<key>": true}}``. A node module registers a provider with ``register_status``; the provider
must only LOOK (file checks through the node's own resolvers): it never downloads, never loads a model and never walks
a model tree, because the front end asks on page load and after every run.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

LOG = logging.getLogger("C2C.NodeStatus")

_PROVIDERS: Dict[str, Callable[[], dict]] = {}


def register_status(node_type: str, provider: Callable[[], dict]) -> None:
    _PROVIDERS[str(node_type)] = provider


def status_of(node_type: str) -> dict:
    """The provider's answer; KeyError for a type without one."""
    out = _PROVIDERS[str(node_type)]()
    return {str(k): v for k, v in dict(out).items()}


def register_routes(server) -> None:
    from aiohttp import web

    @server.routes.get("/c2c/node_status")
    async def _node_status(request: "web.Request") -> "web.Response":
        node_type = request.rel_url.query.get("type", "")
        if node_type not in _PROVIDERS:
            return web.json_response({"error": f"no status for node type {node_type!r}"}, status=404)
        try:
            return web.json_response({"type": node_type, "status": status_of(node_type)})
        except Exception as exc:  # noqa: BLE001 - the front end then shows the control (honest default)
            LOG.warning("node status %s failed: %s", node_type, exc)
            return web.json_response({"error": str(exc)}, status=500)
