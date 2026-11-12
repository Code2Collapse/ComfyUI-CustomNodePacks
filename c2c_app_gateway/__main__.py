"""CLI: ``python -m c2c_app_gateway serve --config gateway.json`` / ``add-user`` / ``check``.

Passwords are read from the terminal (or stdin when piped), never from the command line (it would land in shell
history and process listings).
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys


def _cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _password() -> str:
    if sys.stdin.isatty():
        a, b = getpass.getpass("Password: "), getpass.getpass("Repeat: ")
        if a != b:
            sys.exit("Passwords differ.")
        return a
    return sys.stdin.readline().rstrip("\r\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="c2c_app_gateway", description="C2C Owner-Hosted App Gateway")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--config", required=True)
    u = sub.add_parser("add-user", help="add an owner / publisher / runner (password from the terminal or stdin)")
    u.add_argument("--config", required=True)
    u.add_argument("--name", required=True)
    u.add_argument("--role", required=True, choices=("owner", "publisher", "runner"))
    c = sub.add_parser("check", help="print the effective configuration warnings")
    c.add_argument("--config", required=True)
    a = ap.parse_args(argv)

    from .app import DEFAULTS, build
    from .store import Store, StoreError
    cfg = dict(DEFAULTS, **_cfg(a.config))
    if a.cmd == "add-user":
        try:
            Store(cfg["data_dir"], scrypt_n=cfg["scrypt_n"]).add_user(a.name, _password(), a.role)
        except StoreError as exc:
            sys.exit(str(exc))
        print(f"added {a.role} {a.name}")
        return 0
    warnings = []
    if cfg["bind"] not in ("127.0.0.1", "::1", "localhost"):
        warnings.append(f"bind {cfg['bind']}: the gateway is meant to sit behind a TLS reverse proxy on loopback.")
    if not cfg.get("public_origin"):
        warnings.append("public_origin is not set: Origin checks fall back to the request Host header.")
    po = cfg.get("public_origin") or ""
    if po and not po.startswith("https://") and "127.0.0.1" not in po and "localhost" not in po:
        warnings.append(f"public_origin {po} is not https: put the gateway behind a TLS reverse proxy.")
    if not cfg["secure_cookies"]:
        warnings.append("secure_cookies is false: only acceptable for local testing without TLS.")
    w = cfg["worker_url"]
    if not any(h in w for h in ("127.0.0.1", "[::1]", "localhost")):
        warnings.append(f"worker_url {w}: make sure the worker is reachable ONLY from the gateway (private network).")
    for x in warnings:
        print("WARNING:", x, file=sys.stderr)
    if a.cmd == "check":
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from aiohttp import web
    _, app = build(cfg)
    web.run_app(app, host=cfg["bind"], port=int(cfg["port"]), access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
