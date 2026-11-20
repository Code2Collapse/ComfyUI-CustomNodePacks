"""L9.172 (A25 spec item 52): named groups of semantic classes for the face / clothes parsers.

Shared by SemanticSegmentMEC (nodes/semantic_segment.py) and Segment (C2C) (nodes/segment_c2c.py). A class list is a
comma-separated string; "@name" expands a group (user groups first, then the built-ins of the parser family), and a
leading "-" removes a class or a group ("@head, -hair"). A list with no "@" and no leading "-" gives exactly the old
class set (strip + lower case).

Apache-2.0, C2C.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

MAX_DEPTH = 8

BUILTIN_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "face": {
        "eyes": ["l_eye", "r_eye"],
        "brows": ["l_brow", "r_brow"],
        "lips": ["u_lip", "l_lip"],
        "mouth_area": ["mouth", "u_lip", "l_lip"],
        "ears": ["l_ear", "r_ear", "ear_r"],
        "face_features": ["@brows", "@eyes", "nose", "@mouth_area"],
        "head": ["skin", "@face_features", "@ears", "hair"],
        "accessories": ["eye_g", "necklace", "hat", "ear_r"],
    },
    "clothes": {
        "shoes": ["left_shoe", "right_shoe"],
        "legs": ["left_leg", "right_leg"],
        "arms": ["left_arm", "right_arm"],
        "body_skin": ["face", "@arms", "@legs"],
        "garments": ["upper_clothes", "skirt", "pants", "dress", "belt", "scarf"],
        "accessories": ["hat", "sunglasses", "bag", "belt", "scarf"],
    },
}

_FORMAT = ('groups_json must be a JSON object of group name -> list of class names or "@group" names, e.g. '
           '{"my_face": ["@eyes", "nose", "@lips"]}.')


def family_of(name: str) -> str:
    """'face' for any face-parsing family / model name ('segformer_face', 'face parts'), else 'clothes'."""
    return "face" if "face" in str(name or "").lower() else "clothes"


def uses_groups(csv: str) -> bool:
    """True when the class list names a group ("@x") or excludes something ("-x")."""
    return any(t.strip().startswith(("@", "-")) for t in str(csv or "").split(","))


def parse_user_groups(user_groups_json: str) -> Dict[str, List[str]]:
    """The user's groups_json -> {lower-case name: [lower-case members]}; "" = no groups. Bad input -> ValueError."""
    if not user_groups_json or not str(user_groups_json).strip():
        return {}
    try:
        d = json.loads(user_groups_json)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"groups_json is not valid JSON ({exc}). {_FORMAT}") from exc
    if not isinstance(d, dict):
        raise ValueError(f"groups_json is a {type(d).__name__}, not an object. {_FORMAT}")
    out: Dict[str, List[str]] = {}
    for name, members in d.items():
        key = str(name).strip().lstrip("@").lower()
        if not key:
            raise ValueError(f"groups_json has a group with an empty name. {_FORMAT}")
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            raise ValueError(f"groups_json group {name!r} must be a list of strings. {_FORMAT}")
        out[key] = [m.strip().lower() for m in members if m.strip()]
    return out


def expand_classes(csv: str, valid_classes: Sequence[str], family: str,
                   user_groups_json: str = "") -> Tuple[List[str], List[str], List[str]]:
    """Expand a class list -> (included, excluded, used_groups), all lower case, order of first appearance.

    included = the classes asked for minus the excluded ones. "@name" expands recursively (user groups first, then
    BUILTIN_GROUPS[family]); a cycle, depth > MAX_DEPTH or an unknown group -> ValueError (the error lists the groups
    there are). Class names are NOT checked here: an unknown class comes back in `included` as-is so the caller keeps
    its own "no class matches" error. `valid_classes` (the parser's class names) only maps a class to its own spelling.
    """
    fam = family_of(family)
    user = parse_user_groups(user_groups_json)
    builtin = BUILTIN_GROUPS.get(fam, {})
    spelling = {str(c).lower(): str(c).lower() for c in (valid_classes or [])}
    used: List[str] = []

    def available() -> str:
        names = list(user) + [g for g in builtin if g not in user]
        return ", ".join("@" + g for g in names) or "(none)"

    def expand(token: str, stack: Tuple[str, ...]) -> List[str]:
        if not token.startswith("@"):
            return [spelling.get(token, token)]
        name = token[1:].strip()
        if name in stack:
            raise ValueError(f"Group @{name} contains itself ({' -> '.join('@' + s for s in stack + (name,))}); "
                             "remove the loop from groups_json.")
        if len(stack) >= MAX_DEPTH:
            raise ValueError(f"Groups nest deeper than {MAX_DEPTH} levels at @{name}; flatten groups_json.")
        members = user.get(name, builtin.get(name)) if name in user or name in builtin else None
        if members is None:
            raise ValueError(f"Unknown group @{name}. Groups for the {fam} parser: {available()}.")
        if name not in used:
            used.append(name)
        out: List[str] = []
        for m in members:
            out.extend(expand(m, stack + (name,)))
        return out

    inc: List[str] = []
    exc: List[str] = []
    for raw in str(csv or "").split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        target = inc
        if tok.startswith("-"):
            tok, target = tok[1:].strip(), exc
            if not tok:
                continue
        for c in expand(tok, ()):
            if c not in target:
                target.append(c)
    return [c for c in inc if c not in exc], exc, used
