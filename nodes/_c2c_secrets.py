"""C2C generic secrets vault (P0.1d).

A scoped secret store usable by any feature in the pack — distinct from
``c2c_ai/keychain.py`` which is dedicated to AI provider keys.

Backends, picked at import time in this priority order:
  1. ``keyring`` (Windows Credential Manager / macOS Keychain / libsecret).
     Each (scope, name) pair becomes service=``c2c-secrets:<scope>``,
     username=``<name>``.
  2. Encrypted file via ``cryptography.fernet``. The Fernet key is derived
     from a master passphrase using PBKDF2-HMAC-SHA256 (200000 iterations,
     16-byte random salt persisted in the file). The passphrase is read
     from env var ``C2C_SECRETS_PASSPHRASE``; if absent at first write a
     random 32-byte passphrase is generated and persisted to
     ``ComfyUI/user/c2c/secrets.passphrase`` (chmod 0600 on POSIX).
     File location: ``ComfyUI/user/c2c/secrets.enc.json``.
  3. **No silent plaintext fallback.** If neither backend is available,
     ``set`` raises ``RuntimeError``. ``get`` returns ``None``.

Public API
----------
    has(scope, name)          -> bool
    get(scope, name)          -> str | None         # local code only
    set_(scope, name, value)  -> None                # underscore avoids builtin
    delete(scope, name)       -> bool
    list_(scope)              -> list[str]           # names only, never values
    scopes()                  -> list[str]
    backend_name()            -> str                 # 'keyring' | 'enc-file' | 'none'

HTTP routes (registered via ``register_routes``):
    GET    /c2c/secrets/scopes
    GET    /c2c/secrets/list?scope=<s>
    GET    /c2c/secrets/has?scope=<s>&name=<n>
    POST   /c2c/secrets/set      {scope, name, value}
    POST   /c2c/secrets/delete   {scope, name}
    GET    /c2c/secrets/backend

The HTTP layer NEVER returns secret VALUES — only existence and metadata.
Reading a secret value is restricted to in-process Python callers.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets as _stdlib_secrets
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("c2c.secrets")

# Track which backend ended up active so callers/tests can introspect.
_BACKEND: str = "none"  # one of: 'keyring', 'enc-file', 'none'
_LOCK = threading.RLock()

# Reserved scope names we expose as defaults.
SCOPE_AI = "ai"           # mirrors c2c_ai/keychain (kept separate, do not collide)
SCOPE_GENERAL = "general"
SCOPE_INTEGRATIONS = "integrations"
SCOPE_TELEMETRY = "telemetry"

DEFAULT_SCOPES = (SCOPE_GENERAL, SCOPE_INTEGRATIONS, SCOPE_TELEMETRY)


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def _user_dir() -> Path:
    """Return ComfyUI/user/c2c — created if missing."""
    try:
        import folder_paths  # type: ignore
        base = Path(folder_paths.get_user_directory())
    except Exception:
        # Fallback: parent-of-parent of this file → repo → walk up to ComfyUI/user
        here = Path(__file__).resolve()
        # Find a 'ComfyUI' segment if possible
        for p in here.parents:
            cand = p / "user"
            if cand.is_dir() and (p / "main.py").exists():
                base = cand
                break
        else:
            base = here.parent / "_user"
    out = base / "c2c"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _enc_path() -> Path:
    return _user_dir() / "secrets.enc.json"


def _passphrase_path() -> Path:
    return _user_dir() / "secrets.passphrase"


# --------------------------------------------------------------------------
# Backend 1 — keyring
# --------------------------------------------------------------------------

def _try_keyring():
    """Return the keyring module if usable, else None."""
    try:
        import keyring  # type: ignore
        # Probe the backend; some environments install keyring but have no
        # working backend, in which case set/get throws.
        try:
            kr_backend = keyring.get_keyring()
            name = type(kr_backend).__name__
            if "Fail" in name or "Null" in name:
                log.info("keyring present but backend is %s; skipping", name)
                return None
        except Exception:
            return None
        return keyring
    except Exception:
        return None


_KR = _try_keyring()


def _kr_service(scope: str) -> str:
    return f"c2c-secrets:{scope}"


def _kr_index_service() -> str:
    """A separate service entry holds a JSON index of (scope -> [names]) so
    list_() can enumerate without needing a backend-specific iterator (which
    keyring deliberately doesn't expose)."""
    return "c2c-secrets:_index"


def _kr_index_load() -> dict:
    if not _KR:
        return {}
    try:
        raw = _KR.get_password(_kr_index_service(), "index")
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _kr_index_save(idx: dict) -> None:
    if not _KR:
        return
    try:
        _KR.set_password(_kr_index_service(), "index", json.dumps(idx))
    except Exception as exc:
        log.warning("keyring index save failed: %s", exc)


# --------------------------------------------------------------------------
# Backend 2 — encrypted file (Fernet / PBKDF2)
# --------------------------------------------------------------------------

def _try_cryptography():
    try:
        from cryptography.fernet import Fernet  # type: ignore  # noqa: F401
        from cryptography.hazmat.primitives import hashes  # type: ignore  # noqa: F401
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


_HAS_CRYPTO = _try_cryptography()


def _derive_key(passphrase: bytes, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes  # type: ignore
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase))


def _read_or_create_passphrase() -> bytes:
    env = os.environ.get("C2C_SECRETS_PASSPHRASE")
    if env:
        return env.encode("utf-8")
    p = _passphrase_path()
    if p.exists():
        return p.read_bytes().strip()
    # Generate, persist, restrict permissions.
    pp = _stdlib_secrets.token_urlsafe(32).encode("utf-8")
    p.write_bytes(pp)
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass
    return pp


def _enc_load() -> dict:
    """Returns {'salt': b64, 'data': {scope: {name: cipher_b64}}}."""
    p = _enc_path()
    if not p.exists():
        # Fresh structure with new salt
        salt = _stdlib_secrets.token_bytes(16)
        return {"salt": base64.b64encode(salt).decode("ascii"), "data": {}}
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception as exc:
        log.error("encrypted secrets file unreadable: %s", exc)
        # Refuse to clobber a damaged file silently.
        raise


def _enc_save(blob: dict) -> None:
    p = _enc_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def _enc_fernet(blob: dict):
    from cryptography.fernet import Fernet  # type: ignore
    salt = base64.b64decode(blob["salt"])
    key = _derive_key(_read_or_create_passphrase(), salt)
    return Fernet(key)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

if _KR is not None:
    _BACKEND = "keyring"
elif _HAS_CRYPTO:
    _BACKEND = "enc-file"
else:
    _BACKEND = "none"

log.info("c2c.secrets backend = %s", _BACKEND)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def backend_name() -> str:
    return _BACKEND


def _validate(scope: str, name: str) -> None:
    if not isinstance(scope, str) or not scope or "/" in scope or ":" in scope:
        raise ValueError(f"invalid scope: {scope!r}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid name: {name!r}")
    if len(name) > 256:
        raise ValueError("name too long")


def has(scope: str, name: str) -> bool:
    _validate(scope, name)
    with _LOCK:
        if _BACKEND == "keyring":
            try:
                return _KR.get_password(_kr_service(scope), name) is not None
            except Exception:
                return False
        elif _BACKEND == "enc-file":
            try:
                blob = _enc_load()
            except Exception:
                return False
            return name in blob.get("data", {}).get(scope, {})
        return False


def get(scope: str, name: str) -> Optional[str]:
    _validate(scope, name)
    with _LOCK:
        if _BACKEND == "keyring":
            try:
                return _KR.get_password(_kr_service(scope), name)
            except Exception as exc:
                log.warning("secrets.get(%s/%s) failed: %s", scope, name, exc)
                return None
        elif _BACKEND == "enc-file":
            try:
                blob = _enc_load()
                ct = blob.get("data", {}).get(scope, {}).get(name)
                if not ct:
                    return None
                f = _enc_fernet(blob)
                return f.decrypt(ct.encode("ascii")).decode("utf-8")
            except Exception as exc:
                log.warning("secrets.get(%s/%s) decrypt failed: %s", scope, name, exc)
                return None
        return None


def set_(scope: str, name: str, value: str) -> None:
    """Store / overwrite a secret. Trailing underscore avoids shadowing builtin set()."""
    _validate(scope, name)
    if not isinstance(value, str) or not value:
        raise ValueError("value must be a non-empty string")
    if len(value) > 64 * 1024:
        raise ValueError("value too large (max 64 KiB)")
    with _LOCK:
        if _BACKEND == "keyring":
            _KR.set_password(_kr_service(scope), name, value)
            idx = _kr_index_load()
            names = set(idx.get(scope, []))
            names.add(name)
            idx[scope] = sorted(names)
            _kr_index_save(idx)
            return
        if _BACKEND == "enc-file":
            blob = _enc_load()
            f = _enc_fernet(blob)
            ct = f.encrypt(value.encode("utf-8")).decode("ascii")
            blob.setdefault("data", {}).setdefault(scope, {})[name] = ct
            _enc_save(blob)
            return
        raise RuntimeError(
            "no secrets backend available — install 'keyring' or 'cryptography' "
            "(pip install keyring  OR  pip install cryptography). C2C will not "
            "store secrets in plaintext."
        )


def delete(scope: str, name: str) -> bool:
    _validate(scope, name)
    with _LOCK:
        if _BACKEND == "keyring":
            try:
                _KR.delete_password(_kr_service(scope), name)
                ok = True
            except Exception:
                ok = False
            idx = _kr_index_load()
            if scope in idx and name in idx[scope]:
                idx[scope] = [n for n in idx[scope] if n != name]
                if not idx[scope]:
                    idx.pop(scope, None)
                _kr_index_save(idx)
            return ok
        if _BACKEND == "enc-file":
            try:
                blob = _enc_load()
            except Exception:
                return False
            d = blob.get("data", {}).get(scope, {})
            if name in d:
                del d[name]
                if not d:
                    blob["data"].pop(scope, None)
                _enc_save(blob)
                return True
            return False
        return False


def list_(scope: str) -> list[str]:
    if not scope:
        raise ValueError("scope required")
    with _LOCK:
        if _BACKEND == "keyring":
            idx = _kr_index_load()
            return sorted(idx.get(scope, []))
        if _BACKEND == "enc-file":
            try:
                blob = _enc_load()
            except Exception:
                return []
            return sorted(blob.get("data", {}).get(scope, {}).keys())
        return []


def scopes() -> list[str]:
    with _LOCK:
        if _BACKEND == "keyring":
            idx = _kr_index_load()
            seen = set(idx.keys()) | set(DEFAULT_SCOPES)
        elif _BACKEND == "enc-file":
            try:
                blob = _enc_load()
                seen = set(blob.get("data", {}).keys()) | set(DEFAULT_SCOPES)
            except Exception:
                seen = set(DEFAULT_SCOPES)
        else:
            seen = set(DEFAULT_SCOPES)
    return sorted(seen)


# --------------------------------------------------------------------------
# HTTP routes
# --------------------------------------------------------------------------

_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    """Idempotent registration of /c2c/secrets/* routes on the PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
    except Exception as exc:
        log.warning("aiohttp not importable, skipping secrets routes: %s", exc)
        return

    routes = server.routes

    @routes.get("/c2c/secrets/backend")
    async def _backend(_request):
        return web.json_response({"backend": _BACKEND})

    @routes.get("/c2c/secrets/scopes")
    async def _scopes(_request):
        return web.json_response({"scopes": scopes()})

    @routes.get("/c2c/secrets/list")
    async def _list(request):
        scope = request.query.get("scope", "").strip()
        if not scope:
            return web.json_response({"error": "scope required"}, status=400)
        try:
            return web.json_response({"scope": scope, "names": list_(scope)})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/c2c/secrets/has")
    async def _has(request):
        scope = request.query.get("scope", "").strip()
        name = request.query.get("name", "").strip()
        try:
            return web.json_response({"scope": scope, "name": name, "has": has(scope, name)})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/c2c/secrets/set")
    async def _set(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        scope = (body.get("scope") or "").strip()
        name = (body.get("name") or "").strip()
        value = body.get("value")
        try:
            set_(scope, name, value)
            return web.json_response({"ok": True, "scope": scope, "name": name})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/c2c/secrets/delete")
    async def _del(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        scope = (body.get("scope") or "").strip()
        name = (body.get("name") or "").strip()
        try:
            ok = delete(scope, name)
            return web.json_response({"ok": ok, "scope": scope, "name": name})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True
    log.info("c2c.secrets routes registered (/c2c/secrets/*) backend=%s", _BACKEND)
