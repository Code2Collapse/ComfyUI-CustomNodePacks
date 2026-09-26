"""Update check + killswitch against the plugins distribution manifest — the
counterpart of the panel's useUpdateCheck for zip installs, which never
self-update. Two outcomes: `version` > installed → "update available" surfaced
as a toast; installed < `minVersion` (ops-managed floor) → node executions are
blocked with the download link. An unreachable/malformed manifest FAILS OPEN:
offline users and a down CDN must never lose the nodes.
"""

import json
import logging
import threading
import time
import urllib.request
from typing import Tuple

from . import config, tls_trust


class VersionBlockedError(RuntimeError):
    pass


def _parse_version(value: str) -> Tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else 0 for part in str(value).split("."))


_cache: dict = {"at": 0.0, "state": None}
_lock = threading.Lock()


def state(force: bool = False) -> dict:
    """{installed, latest, min_version, download_url, update_available, blocked}."""
    with _lock:
        fresh = time.time() - _cache["at"] < config.UPDATE_CHECK_TTL_SECONDS
        if _cache["state"] is not None and fresh and not force:
            return _cache["state"]

        installed = config.PLUGIN_VERSION
        result = {
            "installed": installed,
            "latest": None,
            "min_version": None,
            "download_url": None,
            "update_available": False,
            "blocked": False,
        }
        try:
            with urllib.request.urlopen(config.MANIFEST_URL, timeout=10, context=tls_trust.get_ssl_context()) as response:
                manifest = json.loads(response.read().decode())
            entry = manifest.get(config.MANIFEST_HOST_KEY)
            if isinstance(entry, dict):
                latest = str(entry.get("version") or "")
                min_version = str(entry.get("minVersion") or "0.0.0")
                downloads = entry.get("downloads") if isinstance(entry.get("downloads"), dict) else {}
                result.update(
                    latest=latest or None,
                    min_version=min_version,
                    download_url=downloads.get("zip"),
                    update_available=bool(latest)
                    and _parse_version(latest) > _parse_version(installed),
                    blocked=_parse_version(installed) < _parse_version(min_version),
                )
        except Exception:
            pass  # fail open

        _cache.update(at=time.time(), state=result)
        return result


_warned = False


def assert_not_blocked() -> None:
    """Called at the start of every executing node — cheap (cached <=1 fetch/h).

    PORT CHANGE, deliberate: upstream RAISED here, so a `minVersion` floor
    published to a CDN could stop the nodes running. That is reasonable for a
    plugin the vendor ships and can replace; it is not reasonable for source
    that now lives in this repository, where a remote file would be deciding
    whether the user's own checkout runs. A ported copy also cannot satisfy
    the floor by "downloading the latest zip" — the message told the user to
    replace a folder that no longer exists here.

    So this warns once and continues. The check still runs and `state()` still
    reports `blocked`, which the status route and the menu surface as "this
    version is retired" — the user finds out, and decides. If the service
    genuinely refuses an old client, the API says so on the next call, and
    that error is the honest one to show.
    """
    global _warned
    current = state()
    if not current["blocked"] or _warned:
        return
    _warned = True
    logging.getLogger("MEC").warning(
        "[MEC] Magnific reports version %s as retired (its floor is %s). The "
        "nodes still run; re-sync nodes/magnific/ from upstream if calls start "
        "failing. See %s",
        current["installed"], current["min_version"] or "?",
        current["download_url"] or config.DOWNLOAD_PAGE_URL,
    )
