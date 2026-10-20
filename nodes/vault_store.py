"""Where the vault keeps its local secrets: the install key pair and the second-factor state (A17, D0.25).

Everything goes through the pack's generic secrets store (`_c2c_secrets`, scope "vault"): the OS credential store via
`keyring` when available, else an encrypted file in the ComfyUI user directory - never a workflow, never plaintext.
Tests replace BACKEND with a dict.
"""

from __future__ import annotations

from typing import Optional

SCOPE = "vault"
BACKEND = None          # tests: a dict {name: value}


def get(name: str) -> Optional[str]:
    if BACKEND is not None:
        return BACKEND.get(name)
    from . import _c2c_secrets

    return _c2c_secrets.get(SCOPE, name)


def put(name: str, value: str) -> None:
    if BACKEND is not None:
        BACKEND[name] = value
        return
    from . import _c2c_secrets

    _c2c_secrets.set_(SCOPE, name, value)


def delete(name: str) -> None:
    if BACKEND is not None:
        BACKEND.pop(name, None)
        return
    from . import _c2c_secrets

    _c2c_secrets.delete(SCOPE, name)


def backend_name() -> str:
    if BACKEND is not None:
        return "test"
    from . import _c2c_secrets

    return _c2c_secrets.backend_name()
