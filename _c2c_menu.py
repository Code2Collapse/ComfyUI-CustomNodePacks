"""_c2c_menu.py - one menu root for every Code2Collapse pack.

Every node from every Code2Collapse ComfyUI pack lives under ONE root in the
Add Node menu and the node library:

    🐺 C2C / <pack> / <family>

The same way Pixaroma files everything under "👑 Pixaroma". Before this, the
packs spread over more than a dozen roots - CustomNodePacks alone used seven
(C2C, MEC, MaskEditControl, MaskEnhancedControl, utils, Code2Collapse,
ComfyUI-CustomNodePacks), WanAnimatePreprocess spelled its own name three ways,
and WanNodeExperiments filed 139 nodes under "WanVideoWrapper", inside Kijai's
menu and indistinguishable from his.

It is applied ONCE, at registration, from each pack's __init__.py - the node
classes keep the CATEGORY they were written with, and this maps them. That
keeps each family name next to its node in the source, and keeps the whole
tree decided in one place per pack. Node ids are not touched, so saved
workflows are unaffected; only where a node appears in the menu changes.

One identical copy ships in every pack (packs cannot import each other: a
cross-pack import fails on a standalone install). CustomNodePacks'
tests/test_menu.py holds the copies in step.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# The root. The wolf is the house mark (the "lone alpha wolf" identity the
# night palette was built for); change it here and in every copy together.
MENU_ROOT = "\U0001F43A C2C"


def menu_category(
    category: Any,
    label: str,
    strip: Iterable[str] = (),
    rename: Mapping[str, str] | None = None,
) -> str:
    """Map one category into the house tree.

    `strip`  - legacy root segments to drop from the front, repeatedly
               ("MEC/Mask" -> "Mask").
    `rename` - applied to what remains: a key matching the WHOLE remaining
               path wins; otherwise a key matching its first segment
               ("Masking" -> "Mask"). An empty value removes that segment.
    Already-mapped categories pass through unchanged, so applying it twice is
    harmless.
    """
    if isinstance(category, str) and category.startswith(MENU_ROOT):
        return category
    segs = [s.strip() for s in str(category or "").split("/") if s.strip()]
    strip_set = set(strip)
    while segs and segs[0] in strip_set:
        segs.pop(0)
    rest = "/".join(segs)
    if rename:
        if rest in rename:
            rest = rename[rest]
        elif segs and segs[0] in rename:
            head = rename[segs[0]]
            rest = "/".join(([head] if head else []) + segs[1:])
    return f"{MENU_ROOT}/{label}" + (f"/{rest}" if rest else "")


def rebrand_v1(
    mappings: Mapping[str, type],
    label: str,
    strip: Iterable[str] = (),
    rename: Mapping[str, str] | None = None,
) -> int:
    """Rewrite CATEGORY on every V1 node class in `mappings`. Returns how many
    changed. Never raises: a class that refuses the attribute keeps its old
    category, which is a menu placement, not a failure.

    A V3 node registered through a V1 mapping (NukeMax's ReLight 2D is one) is
    handed to rebrand_v3: ComfyUI builds its /object_info from the schema, so
    assigning CATEGORY would only mask the classproperty while the menu kept
    the old path."""
    strip = tuple(strip)
    changed = 0
    seen: set[int] = set()
    for cls in mappings.values():
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if _is_v3(cls):
            changed += rebrand_v3([cls], label, strip, rename)
            continue
        old = getattr(cls, "CATEGORY", None)
        new = menu_category(old, label, strip, rename)
        if new != old:
            try:
                cls.CATEGORY = new
                changed += 1
            except Exception:
                pass
    return changed


def _is_v3(cls: Any) -> bool:
    """A comfy_api io.ComfyNode: its node info comes from define_schema()."""
    return callable(getattr(cls, "GET_NODE_INFO_V1", None)) and callable(
        getattr(cls, "define_schema", None))


def rebrand_v3(
    classes: Iterable[type],
    label: str,
    strip: Iterable[str] = (),
    rename: Mapping[str, str] | None = None,
) -> int:
    """Rewrite the category of V3 (comfy_api io.ComfyNode) classes.

    A V3 node's category comes from define_schema(), which ComfyUI calls
    afresh for every /object_info request and caches into cls._CATEGORY - so
    the schema is wrapped (idempotently) and that cache is cleared.
    """
    strip = tuple(strip)
    changed = 0
    for cls in classes:
        if cls.__dict__.get("_c2c_menu_wrapped"):
            continue
        orig = getattr(cls, "define_schema", None)
        if orig is None:
            continue

        def define_schema(c, _orig=orig):
            schema = _orig()
            try:
                schema.category = menu_category(schema.category, label, strip, rename)
            except Exception:
                pass
            return schema

        try:
            cls.define_schema = classmethod(define_schema)
            cls._c2c_menu_wrapped = True
            if getattr(cls, "_CATEGORY", None) is not None:
                cls._CATEGORY = None
            changed += 1
        except Exception:
            pass
    return changed
