"""Which folders the C2C loader's server-path mode may read (owner decision D0.35, 2026-10-31; ledger L9.89).

* The browser-facing routes (probe, preview, scan, still, frames plan) answer only inside ComfyUI's input, output and
  temp folders plus the configured media roots.
* A queued run reads any path - every saved Load Video Path workflow keeps loading - UNLESS ComfyUI listens on an
  address other machines can reach (--listen 0.0.0.0, a LAN IP ...): then runs are held to the same roots.

The extra roots come from the ``C2C_MEDIA_ROOTS`` environment variable (os.pathsep-separated) and from
``c2c_media_roots.txt`` (one folder per line, # comments) in the ComfyUI base folder. Deliberately NOT a front-end
setting or a file in the user folder: the web API can write both, so a remote client could widen its own access.
Symlinks are resolved before the check, so a link inside a root cannot point outside it.
"""

from __future__ import annotations

import ipaddress
import os

ENV = "C2C_MEDIA_ROOTS"
FILE_NAME = "c2c_media_roots.txt"


def _folder_paths():
    import folder_paths  # type: ignore[import-not-found]

    return folder_paths


def _real(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def configured_roots() -> list[str]:
    """The extra roots (env + file), as written."""
    out: list[str] = []
    for part in os.environ.get(ENV, "").split(os.pathsep):
        if part.strip():
            out.append(part.strip().strip('"'))
    try:
        base = getattr(_folder_paths(), "base_path", "") or ""
    except Exception:  # noqa: BLE001 - outside ComfyUI (tests, tools)
        base = ""
    if base:
        try:
            with open(os.path.join(base, FILE_NAME), encoding="utf-8") as f:
                for line in f:
                    line = line.split("#", 1)[0].strip().strip('"')
                    if line:
                        out.append(line)
        except OSError:
            pass
    return out


def media_roots() -> list[str]:
    """Every allowed root, resolved: input, output, temp, then the configured ones."""
    roots: list[str] = []
    try:
        fp = _folder_paths()
        for getter in ("get_input_directory", "get_output_directory", "get_temp_directory"):
            fn = getattr(fp, getter, None)
            if callable(fn):
                roots.append(fn())
    except Exception:  # noqa: BLE001
        pass
    roots.extend(configured_roots())
    seen: list[str] = []
    for r in roots:
        rr = _real(r)
        if rr not in seen:
            seen.append(rr)
    return seen


def is_inside_roots(path: str, roots: list[str] | None = None) -> bool:
    """True when ``path`` (symlinks resolved) lies inside one of the roots. A pattern (shot.####.exr) is judged by its
    folder."""
    target = _real(path)
    if not os.path.exists(target):
        target = _real(os.path.dirname(path) or ".")
    for root in roots if roots is not None else media_roots():
        try:
            if os.path.commonpath([root, target]) == root:
                return True
        except ValueError:                       # another drive on Windows
            continue
    return False


def outside_message(path: str) -> str:
    folder = os.path.dirname(os.path.abspath(path)) if not os.path.isdir(path) else os.path.abspath(path)
    return (f"{folder} is outside the C2C media roots - add it to the {ENV} environment variable or to "
            f"{FILE_NAME} in the ComfyUI folder (then restart ComfyUI).")


def _listen_addresses() -> list[str]:
    try:
        from comfy.cli_args import args  # type: ignore[import-not-found]

        raw = str(getattr(args, "listen", "127.0.0.1") or "127.0.0.1")
    except Exception:  # noqa: BLE001 - outside ComfyUI: local
        return ["127.0.0.1"]
    return [a.strip() for a in raw.split(",") if a.strip()]


def server_is_local() -> bool:
    """True when ComfyUI listens only on loopback addresses (the default 127.0.0.1)."""
    for addr in _listen_addresses():
        if addr.lower() == "localhost":
            continue
        try:
            if not ipaddress.ip_address(addr.strip("[]")).is_loopback:
                return False
        except ValueError:
            return False                         # a host name: assume reachable
    return True


def check_run_path(path: str) -> str | None:
    """None when a RUN may read ``path``; else the message (only when the server is reachable from other machines)."""
    if server_is_local() or is_inside_roots(path):
        return None
    return outside_message(path)


def check_route_path(path: str) -> str | None:
    """None when a ROUTE may read ``path``; else the message (always restricted to the roots)."""
    return None if is_inside_roots(path) else outside_message(path)


__all__ = ["ENV", "FILE_NAME", "check_route_path", "check_run_path", "configured_roots", "is_inside_roots",
           "media_roots", "outside_message", "server_is_local"]
