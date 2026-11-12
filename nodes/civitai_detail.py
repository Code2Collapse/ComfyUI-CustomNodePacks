"""Civitai detail view data (L5.04): a model's versions, files, trigger words and example images, trimmed to what the
Model Browser shows, with the description sanitised.

Shapes follow developer.civitai.com/site/reference (models.md, model-versions.md; read 2026-11-12): the model-level
object carries `modelVersions[]`; `trainedWords` is documented on GET /api/v1/model-versions/{id}; file objects carry
`id, name, type, sizeKB, metadata{format,size,fp}, hashes{SHA256,...}, downloadUrl, primary`.

Early access / login: a version still in early access is marked and its files are NOT offered for download - this
view never works around Civitai's access rules.
"""
from __future__ import annotations

import html
import time
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse

ALLOWED_TAGS = {"p", "br", "b", "strong", "i", "em", "u", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "code",
                "pre", "blockquote", "a", "hr", "span", "div"}
DROP_WITH_CONTENT = {"script", "style", "iframe", "object", "embed", "svg", "math", "template", "noscript", "form",
                     "textarea", "select", "button", "input", "link", "meta", "head", "title"}
VOID = {"br", "hr"}
MAX_DESC_CHARS = 20000


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[str] = []
        self.skip = 0                                      # >0 inside a dropped element

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in DROP_WITH_CONTENT:
            if tag not in VOID:
                self.skip += 1
            return
        if self.skip or tag not in ALLOWED_TAGS:
            return
        if tag == "a":
            href = dict(attrs).get("href") or ""
            scheme = urlparse(href.strip()).scheme.lower()
            if scheme in ("http", "https"):
                self.out.append(f'<a href="{html.escape(href.strip(), quote=True)}" target="_blank" '
                                'rel="noopener noreferrer nofollow">')
            else:
                self.out.append("<a>")                     # javascript:, data:, relative: link text only
        else:
            self.out.append(f"<{tag}>")                    # every attribute dropped (on*, style, class, id ...)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if tag.lower() in VOID and not self.skip:
            self.out.append(f"<{tag.lower()}>")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in DROP_WITH_CONTENT:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag not in self.stack:
            return
        while self.stack:                                  # close anything left open inside it
            t = self.stack.pop()
            self.out.append(f"</{t}>")
            if t == tag:
                break

    def handle_data(self, data):
        if not self.skip:
            self.out.append(html.escape(data, quote=False))

    def result(self) -> str:
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")
        return "".join(self.out)


def sanitize_html(src) -> str:
    """Allow-listed HTML: a few formatting tags, http(s) links (new tab, no opener), no attributes, no scripts or
    styles. Text is escaped. Safe to put into innerHTML."""
    if not src:
        return ""
    p = _Sanitizer()
    p.feed(str(src)[: MAX_DESC_CHARS * 2])
    p.close()
    return p.result()[: MAX_DESC_CHARS * 2]


def _early_access(v: dict) -> str:
    """'' or a human line, e.g. 'Early access until 2026-12-01' (never bypassed)."""
    ends = v.get("earlyAccessEndsAt")
    if v.get("availability") == "EarlyAccess" or ends:
        when = ""
        if isinstance(ends, str):
            try:
                dt = datetime.fromisoformat(ends.replace("Z", "+00:00"))
                if dt.timestamp() < time.time():
                    return ""                              # early access is over
                when = f" until {dt.date().isoformat()}"
            except ValueError:
                pass
        return "Early access" + when
    return ""


def _file(f: dict, gated: bool) -> dict:
    md = f.get("metadata") or {}
    hashes = f.get("hashes") or {}
    sha = hashes.get("SHA256") or hashes.get("sha256") or ""
    return {"id": f.get("id"), "name": f.get("name") or "", "type": f.get("type") or "", "size_kb": f.get("sizeKB"),
            "format": md.get("format") or "", "precision": md.get("fp") or "", "pruned": md.get("size") or "",
            "sha256": sha, "primary": bool(f.get("primary")),
            "scan": {"pickle": f.get("pickleScanResult") or "", "virus": f.get("virusScanResult") or ""},
            "downloadable": not gated}


def _image(im: dict, allow_nsfw: bool) -> dict | None:
    level = im.get("nsfwLevel")
    nsfw = (isinstance(level, int) and level > 1) or (isinstance(level, str) and level.lower() not in ("none", "pg", "")) \
        or bool(im.get("nsfw"))
    if nsfw and not allow_nsfw:
        return None
    url = im.get("url") or ""
    if urlparse(url).scheme not in ("http", "https"):
        return None
    return {"url": url, "width": im.get("width"), "height": im.get("height"), "nsfw": bool(nsfw)}


def version_detail(v: dict, allow_nsfw: bool = False) -> dict:
    gate = _early_access(v)
    return {"id": v.get("id"), "name": v.get("name") or "", "base_model": v.get("baseModel") or "",
            "trained_words": [str(w) for w in (v.get("trainedWords") or []) if str(w).strip()][:50],
            "early_access": gate, "description": sanitize_html(v.get("description")),
            "files": [_file(f, bool(gate)) for f in (v.get("files") or [])],
            "images": [i for i in (_image(im, allow_nsfw) for im in (v.get("images") or [])[:12]) if i],
            "downloads": (v.get("stats") or {}).get("downloadCount")}


def model_detail(m: dict, allow_nsfw: bool = False) -> dict:
    creator = m.get("creator") or {}
    return {"id": m.get("id"), "name": m.get("name") or "", "type": m.get("type") or "",
            "nsfw": bool(m.get("nsfw")), "creator": creator.get("username") or "",
            "tags": [str(t) if not isinstance(t, dict) else str(t.get("name", "")) for t in (m.get("tags") or [])][:20],
            "description": sanitize_html(m.get("description")),
            "stats": {k: (m.get("stats") or {}).get(k) for k in ("downloadCount", "rating", "thumbsUpCount")},
            "versions": [{"id": v.get("id"), "name": v.get("name") or "", "base_model": v.get("baseModel") or "",
                          "early_access": _early_access(v)} for v in (m.get("modelVersions") or [])],
            "first_version": version_detail((m.get("modelVersions") or [{}])[0], allow_nsfw)
            if m.get("modelVersions") else None}
