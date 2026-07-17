"""RIB pre-flight validation — the fatality preventer.

Before a workflow leaves the building, ask the target backend for its
/object_info and verify every class_type in the submitted prompt exists
there. A miss fails LOCALLY with a plain-English error instead of burning
cloud GPU time on a guaranteed crash.
"""

from __future__ import annotations


def required_classes(prompt: dict) -> set[str]:
    return {
        node["class_type"]
        for node in prompt.values()
        if isinstance(node, dict) and node.get("class_type")
    }


def validate_backend_nodes(prompt: dict, adapter, backend_name: str | None = None):
    """Raises RuntimeError listing every missing node class. No-op if the
    backend cannot report its classes (None) — better to try than to block
    on a probe the gateway may not expose."""
    name = backend_name or getattr(adapter, "name", "backend")
    installed = adapter.installed_node_classes()
    if installed is None:
        return
    missing = sorted(required_classes(prompt) - installed)
    if missing:
        raise RuntimeError(
            f"Backend missing required nodes. Please update Git repo. "
            f"Backend '{name}' lacks {len(missing)} node class(es) used by this "
            f"workflow: {', '.join(missing)}. Add the pack(s) to "
            f"custom_nodes_manifest.txt and rebuild the backend Docker image."
        )
