"""Gateway data directory: users, apps, write-once releases (private part encrypted), jobs, outputs, audit.

Layout (owner-only permissions, never under a served root)::

    secret.key                      32 random bytes (HMAC + encryption root; separate subkeys derived per use)
    users.json                      {name: {role, salt, hash, n, disabled, created}}
    revoked_tokens.json             [jti, ...]
    apps/<app_id>/app.json          {app_id, name, owner, acl, current, releases, created}
    apps/<app_id>/releases/<n>/manifest.json   public manifest (plaintext)
    apps/<app_id>/releases/<n>/private.bin     AES-256-GCM(private release JSON), AAD "<app_id>:<n>"
    jobs/<job_id>.json              job record (its "_private" part is never projected to a runner)
    outputs/<job_id>/               fetched, declared outputs only
    audit.jsonl                     one event per line; never input values, prompts or paths
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

ROLES = ("owner", "publisher", "runner")
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
USER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{22,64}$")


class StoreError(ValueError):
    """A request the store refuses (bad name, duplicate, missing release); the message is safe to show the owner."""


def _write_json(path: Path, obj, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=1, sort_keys=True).encode("utf-8")
    if exclusive:
        with open(path, "xb") as f:
            f.write(data)
        _private_mode(path)
        return
    tmp = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    _private_mode(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _private_mode(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:  # Windows ACLs: the deployment guide covers directory permissions
        pass


class Store:
    def __init__(self, root: str | os.PathLike, *, scrypt_n: int = 2 ** 15):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.scrypt_n = int(scrypt_n)
        self._lock = threading.RLock()
        kp = self.root / "secret.key"
        if not kp.exists():
            with open(kp, "xb") as f:
                f.write(os.urandom(32))
            _private_mode(kp)
        self._key = kp.read_bytes()
        if len(self._key) != 32:
            raise StoreError("secret.key must be exactly 32 bytes.")

    # ── keys ────────────────────────────────────────────────────────────────────────────────────────────────
    def subkey(self, purpose: str) -> bytes:
        return hmac.new(self._key, b"c2c-gateway:" + purpose.encode(), hashlib.sha256).digest()

    # ── users ───────────────────────────────────────────────────────────────────────────────────────────────
    def _users(self) -> dict:
        return _read_json(self.root / "users.json", {}) or {}

    def _hash(self, password: str, salt: bytes, n: int) -> bytes:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=8, p=1, maxmem=2 ** 27, dklen=32)

    def add_user(self, name: str, password: str, role: str) -> dict:
        if not USER_RE.match(name or ""):
            raise StoreError("User names use letters, digits, '.', '_' or '-' (at most 64).")
        if role not in ROLES:
            raise StoreError(f"Role must be one of {', '.join(ROLES)}.")
        if not isinstance(password, str) or len(password) < 10:
            raise StoreError("Passwords need at least 10 characters.")
        with self._lock:
            users = self._users()
            if name in users:
                raise StoreError("That user already exists.")
            salt = os.urandom(16)
            users[name] = {"role": role, "salt": salt.hex(), "n": self.scrypt_n,
                           "hash": self._hash(password, salt, self.scrypt_n).hex(), "disabled": False,
                           "created": time.time()}
            _write_json(self.root / "users.json", users)
        return self.get_user(name)

    def get_user(self, name: str) -> dict | None:
        u = self._users().get(name)
        if not u:
            return None
        return {"name": name, "role": u["role"], "disabled": bool(u.get("disabled"))}

    def list_users(self) -> list[dict]:
        return [self.get_user(n) for n in sorted(self._users())]

    def verify_password(self, name: str, password: str) -> dict | None:
        u = self._users().get(name or "")
        if not u:
            self._hash(password or "", b"\0" * 16, self.scrypt_n)       # same cost for unknown users
            return None
        got = self._hash(password or "", bytes.fromhex(u["salt"]), int(u.get("n", self.scrypt_n)))
        if not hmac.compare_digest(got, bytes.fromhex(u["hash"])) or u.get("disabled"):
            return None
        return self.get_user(name)

    def set_disabled(self, name: str, disabled: bool) -> None:
        with self._lock:
            users = self._users()
            if name not in users:
                raise StoreError("No such user.")
            users[name]["disabled"] = bool(disabled)
            _write_json(self.root / "users.json", users)

    # ── tokens ──────────────────────────────────────────────────────────────────────────────────────────────
    def revoked_tokens(self) -> set[str]:
        return set(_read_json(self.root / "revoked_tokens.json", []) or [])

    def revoke_token(self, jti: str) -> None:
        with self._lock:
            s = self.revoked_tokens()
            s.add(str(jti))
            _write_json(self.root / "revoked_tokens.json", sorted(s))

    # ── apps and releases ───────────────────────────────────────────────────────────────────────────────────
    def _app_dir(self, app_id: str) -> Path:
        if not isinstance(app_id, str) or not APP_ID_RE.match(app_id):
            raise StoreError("App ids use lower-case letters, digits and '-' (at most 63).")
        return self.root / "apps" / app_id

    def create_app(self, app_id: str, owner: str, name: str) -> dict:
        d = self._app_dir(app_id)
        with self._lock:
            if (d / "app.json").exists():
                raise StoreError("That app id is taken.")
            app = {"app_id": app_id, "name": str(name or app_id)[:120], "owner": owner, "acl": [], "current": None,
                   "releases": [], "created": time.time()}
            _write_json(d / "app.json", app, exclusive=True)
        return app

    def get_app(self, app_id: str) -> dict | None:
        try:
            d = self._app_dir(app_id)
        except StoreError:
            return None
        return _read_json(d / "app.json")

    def save_app(self, app: dict) -> None:
        with self._lock:
            _write_json(self._app_dir(app["app_id"]) / "app.json", app)

    def list_apps(self) -> list[dict]:
        base = self.root / "apps"
        if not base.is_dir():
            return []
        out = []
        for d in sorted(base.iterdir()):
            a = _read_json(d / "app.json")
            if a:
                out.append(a)
        return out

    def write_release(self, app_id: str, manifest: dict, private: dict) -> int:
        """Write-once: a release directory is created exclusively and never rewritten."""
        d = self._app_dir(app_id)
        with self._lock:
            app = self.get_app(app_id)
            if not app:
                raise StoreError("No such app.")
            n = (max(app["releases"]) + 1) if app["releases"] else 1
            rd = d / "releases" / str(n)
            rd.mkdir(parents=True, exist_ok=False)
            manifest = dict(manifest, release=n)
            _write_json(rd / "manifest.json", manifest, exclusive=True)
            with open(rd / "private.bin", "xb") as f:
                f.write(self._seal(json.dumps(private, sort_keys=True).encode("utf-8"), f"{app_id}:{n}"))
            _private_mode(rd / "private.bin")
            app["releases"].append(n)
            app["current"] = n
            self.save_app(app)
        return n

    def read_manifest(self, app_id: str, n: int) -> dict | None:
        return _read_json(self._app_dir(app_id) / "releases" / str(int(n)) / "manifest.json")

    def read_private(self, app_id: str, n: int) -> dict:
        blob = (self._app_dir(app_id) / "releases" / str(int(n)) / "private.bin").read_bytes()
        return json.loads(self._open(blob, f"{app_id}:{int(n)}"))

    def _seal(self, data: bytes, aad: str) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        return b"C2CGW1" + nonce + AESGCM(self.subkey("private-release")).encrypt(nonce, data, aad.encode())

    def _open(self, blob: bytes, aad: str) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if not blob.startswith(b"C2CGW1"):
            raise StoreError("Unknown private release format.")
        return AESGCM(self.subkey("private-release")).decrypt(blob[6:18], blob[18:], aad.encode())

    # ── jobs and outputs ────────────────────────────────────────────────────────────────────────────────────
    def job_path(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id or ""):
            raise StoreError("bad job id")
        return self.root / "jobs" / f"{job_id}.json"

    def save_job(self, job: dict) -> None:
        with self._lock:
            _write_json(self.job_path(job["job_id"]), job)

    def load_job(self, job_id: str) -> dict | None:
        try:
            return _read_json(self.job_path(job_id))
        except StoreError:
            return None

    def list_jobs(self) -> list[dict]:
        d = self.root / "jobs"
        if not d.is_dir():
            return []
        return [j for j in (_read_json(p) for p in sorted(d.glob("*.json"))) if j]

    def output_dir(self, job_id: str) -> Path:
        self.job_path(job_id)                                             # validates the id
        d = self.root / "outputs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── audit ───────────────────────────────────────────────────────────────────────────────────────────────
    def audit(self, event: str, **fields) -> None:
        rec = {"t": round(time.time(), 3), "event": event}
        rec.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(rec, sort_keys=True) + "\n"
        with self._lock, open(self.root / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(line)

    def rotate_audit(self, max_mb: float = 20.0, keep: int = 10) -> None:
        """Rename audit.jsonl to audit-<time>.jsonl once it passes ``max_mb``; keep the newest ``keep`` old files."""
        p = self.root / "audit.jsonl"
        with self._lock:
            if not p.exists() or p.stat().st_size < max_mb * 1024 * 1024:
                return
            os.replace(p, self.root / f"audit-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}.jsonl")
            for old in sorted(self.root.glob("audit-*.jsonl"))[:-keep]:
                old.unlink()

    def read_audit(self, limit: int = 200) -> list[dict]:
        p = self.root / "audit.jsonl"
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").splitlines()[-max(1, min(int(limit), 5000)):]
        return [json.loads(x) for x in lines if x.strip()]
