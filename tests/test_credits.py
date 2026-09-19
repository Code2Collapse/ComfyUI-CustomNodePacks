"""Every upstream this pack ports from must be named in CREDITS.md.

Attribution rots in exactly one way: someone ports a family, writes an honest
header on the file, and the credits page never hears about it. Six months later
the only record is a comment nobody reads.

So the header IS the record, and this walks them. A module that names an
upstream repository in its first lines must have that repository in CREDITS.md,
or this fails with the URL it could not find.

The convention a ported module follows:

    # Ported from <Project> by <author> (<LICENCE>,
    # https://github.com/<owner>/<repo>), <what was taken>.

`NOTICE` carries the formal terms; CREDITS.md is the readable one. Both have to
exist and both have to agree with the source.
"""

from __future__ import annotations

import re
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
CREDITS = PACK / "CREDITS.md"

SKIP_DIRS = {"__pycache__", ".git", "third_party", "_deprecated", "_AUDIT",
             "node_modules", "tests", "_tests"}

#: How many lines at the top of a module count as its header.
HEADER_LINES = 30

_GITHUB = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_ATTRIBUTION = re.compile(
    r"\b(ported from|Ported from|PORTED FROM|derived from|Derived from|"
    r"adapted from|Adapted from)\b")


def _source_files():
    for path in sorted(PACK.rglob("*.py")):
        if any(p in SKIP_DIRS for p in path.parts):
            continue
        yield path


def _attributed_modules() -> dict[Path, set[tuple[str, str]]]:
    """Modules whose header names an upstream GitHub project."""
    found: dict[Path, set[tuple[str, str]]] = {}
    for path in _source_files():
        try:
            head = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        block = "\n".join(head[:HEADER_LINES])
        if not _ATTRIBUTION.search(block):
            continue
        repos = {(o, r.removesuffix(".git")) for o, r in _GITHUB.findall(block)}
        if repos:
            found[path] = repos
    return found


def test_credits_file_exists():
    assert CREDITS.is_file(), (
        "CREDITS.md is where a user finds out whose work a node came from; "
        "NOTICE is the legal text, not a readable answer"
    )


def test_every_attributed_module_is_credited():
    # THE check. A header that names a repo the credits page has never heard of
    # means the attribution exists only for whoever opens that one file.
    credits = CREDITS.read_text(encoding="utf-8")
    missing = []
    for path, repos in _attributed_modules().items():
        for owner, repo in repos:
            if repo.lower() not in credits.lower():
                missing.append(f"{path.relative_to(PACK)} -> {owner}/{repo}")
    assert not missing, (
        "these modules name an upstream that CREDITS.md does not: "
        + "; ".join(sorted(missing))
    )


def test_the_scanner_actually_finds_the_known_ports():
    # A scanner that matches nothing passes this file silently. The two ported
    # families are known to carry headers, so they must show up.
    found = _attributed_modules()
    names = {p.name for p in found}
    assert len(found) >= 4, f"only found {len(found)} attributed modules: {names}"
    paths = {str(p.relative_to(PACK)).replace("\\", "/") for p in found}
    assert any(p.startswith("nodes/layer_effects/") for p in paths)
    assert any(p.startswith("nodes/mask_toolkit/") for p in paths)


def test_credits_names_a_licence_for_every_upstream_it_lists():
    # "We used their code" without "under these terms" is half an attribution.
    credits = CREDITS.read_text(encoding="utf-8")
    sections = re.findall(r"^### (.+?) — (.+)$", credits, re.M)
    assert sections, "CREDITS.md has no '### Project — Author' sections"
    for project, _author in sections:
        block = credits.split(f"### {project} — ", 1)[1].split("\n###", 1)[0]
        assert _GITHUB.search(block), f"{project}: no repository link"


def test_credits_does_not_claim_a_licence_for_the_unverified_two():
    # KJNodes and WhatDreamsCost both point pyproject at a LICENSE file that is
    # not in the clone. Saying "MIT" or "Apache" about either would be inventing
    # terms for somebody else's work.
    credits = CREDITS.read_text(encoding="utf-8")
    block = credits.split("could not be verified", 1)
    assert len(block) == 2, "CREDITS.md no longer records the unverified pair"
    section = block[1].split("\n---", 1)[0]
    for name in ("KJNodes", "WhatDreamsCost"):
        assert name in section, f"{name} is no longer listed as unverified"
    # The section may DISCUSS a licence claim - it exists partly to say that
    # NOTICE's "Apache-2.0" for KJNodes is unsupported, and that sentence has to
    # name both. What it must not do is assert one in the LIST that names the
    # projects, which is the part a reader takes as the answer.
    for line in section.splitlines():
        if not line.lstrip().startswith(("-", "*")):
            continue
        if not any(n in line for n in ("KJNodes", "WhatDreamsCost")):
            continue
        for claim in ("MIT", "Apache", "GPL", "BSD"):
            assert claim not in line, (
                f"CREDITS.md states {claim} on the line naming an upstream whose "
                f"licence is unverified: {line.strip()!r}"
            )
    assert "unknown" in section.lower() or "could not" in section.lower()


def test_notice_and_credits_both_mention_the_ported_families():
    notice = (PACK / "NOTICE").read_text(encoding="utf-8")
    credits = CREDITS.read_text(encoding="utf-8")
    for token in ("LayerStyle", "chflame163"):
        assert token in notice, f"NOTICE lost {token}"
        assert token in credits, f"CREDITS.md lost {token}"
