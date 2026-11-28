"""
Parameter Memory v2 — Compact persistent parameter history for ComfyUI.

Two components:
  1. ParameterHistoryMEC node: displays stored parameter history in the workflow
  2. Server route /mec/param_history: receives delta snapshots from JS, stores in SQLite

The JS extension (js/parameter_memory.js) handles:
  - Real-time widget change interception on ALL nodes
  - Pre-execution diff snapshots (only non-default values)
  - Hover tooltips (Alt+hover → see previous value & default)
  - Right-click context menus (history, diff, reset, presets)

v2 changes:
  - DB stores one row per node per run (JSON diff blob) instead of one row per param
  - Accepts delta-only payloads from JS (only changed-from-default params)
  - Auto-prunes old runs beyond MAX_RUNS
  - Cleaner grouped output formatting
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
try:                                                       # A28 layout declarations (display only)
    from ._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

logger = logging.getLogger("MEC")

# ── Config ────────────────────────────────────────────────────────────
MAX_RUNS = 30  # auto-prune: keep only this many distinct runs

# A22 L9.135: secrets never reach param_history.db. A widget whose NAME looks like a secret is dropped, and a string
# value that looks like a provider key (whatever its widget is called) is dropped too.
_SECRET_NAME = re.compile(r"api[_ -]?key|apikey|secret|passw(?:or)?d|passphrase|credential|private[_ -]?key|"
                          r"(?:^|_)token$", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"\b(?:sk-(?:ant-)?[A-Za-z0-9_\-]{20,}|hf_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|"
                           r"xox[abp]-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_\-]{35})")
_SCRUBBED_DB = False


def _strip_secrets(delta):
    """Return ``delta`` without secret-looking entries (see _SECRET_NAME / _SECRET_VALUE)."""
    if not isinstance(delta, dict):
        return {}
    out = {}
    for k, v in delta.items():
        if _SECRET_NAME.search(str(k)):
            continue
        if isinstance(v, str) and _SECRET_VALUE.search(v):
            continue
        out[k] = v
    return out


def _scrub_existing(conn):
    """Once per process: remove secret-looking entries that older versions may have stored."""
    global _SCRUBBED_DB
    if _SCRUBBED_DB:
        return
    _SCRUBBED_DB = True
    try:
        changed = []
        for row_id, dj in conn.execute("SELECT id, delta_json FROM param_diffs").fetchall():
            try:
                d = json.loads(dj)
            except Exception:
                continue
            clean = _strip_secrets(d)
            if clean != d:
                changed.append((json.dumps(clean), row_id))
        if changed:
            conn.executemany("UPDATE param_diffs SET delta_json = ? WHERE id = ?", changed)
            conn.commit()
            logger.info("[MEC] param_history: removed secret-looking values from %d stored row(s)", len(changed))
    except Exception as e:
        logger.warning(f"[MEC] param_history scrub error: {e}")

# ── DB path lives next to this file ──────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "param_history.db")
_DB_PATH = os.path.normpath(_DB_PATH)


def _get_db():
    """Get a SQLite connection with WAL mode and v2 schema."""
    conn = sqlite3.connect(_DB_PATH, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    # v2 schema: one row per node per run, delta JSON blob
    conn.execute("""
        CREATE TABLE IF NOT EXISTS param_diffs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT    NOT NULL,
            run_id     INTEGER NOT NULL,
            node_id    TEXT    NOT NULL,
            node_title TEXT    NOT NULL,
            node_class TEXT    NOT NULL,
            delta_json TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_diffs_run
        ON param_diffs(run_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_diffs_class
        ON param_diffs(node_class)
    """)
    conn.commit()
    _scrub_existing(conn)
    return conn


def _prune_old_runs(conn):
    """Keep only the last MAX_RUNS distinct runs."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM param_diffs ORDER BY run_id DESC"
        ).fetchall()
        if len(rows) > MAX_RUNS:
            cutoff = rows[MAX_RUNS - 1][0]
            conn.execute("DELETE FROM param_diffs WHERE run_id < ?", (cutoff,))
            conn.commit()
    except Exception as e:
        logger.warning(f"[MEC] prune error: {e}")


def _store_snapshot(data: dict):
    """Store a run snapshot (delta format) into the DB.

    data format from JS v2:
    {
        "node_id": {
            "title": "KSampler",
            "class": "KSampler",
            "run_id": 5,
            "ts": "2026-03-19 10:00:00",
            "delta": {"steps": 20, "cfg": 7.0}   // only non-default params
        },
        ...
    }
    """
    try:
        conn = _get_db()
        rows = []
        # One POST = one queued run. The browser's run counter restarts at 0 on every page reload, so its run_id
        # would collide with (and be pruned below) older runs; the server assigns a monotonic id instead.
        _mx = conn.execute("SELECT MAX(run_id) FROM param_diffs").fetchone()
        server_run_id = int((_mx[0] if _mx else 0) or 0) + 1
        for node_id, info in data.items():
            ts = info.get("ts", datetime.now().isoformat(sep=" ", timespec="seconds"))
            run_id = server_run_id
            title = info.get("title", "")
            cls = info.get("class", "")
            # v2: accept "delta" field; fall back to "values" for backward compat
            delta = _strip_secrets(info.get("delta", info.get("values", {})))
            if not delta:
                continue  # skip nodes with no changes from defaults
            rows.append((ts, run_id, str(node_id), title, cls, json.dumps(delta)))

        if rows:
            conn.executemany(
                "INSERT INTO param_diffs (ts, run_id, node_id, node_title, node_class, delta_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            _prune_old_runs(conn)
        conn.close()
    except Exception as e:
        logger.warning(f"[MEC] param_history DB write error: {e}")


def _query_history(node_class: str = "", last_n_runs: int = 10) -> list[dict]:
    """Query parameter history from the DB."""
    try:
        conn = _get_db()
        if node_class:
            rows = conn.execute(
                "SELECT ts, run_id, node_id, node_title, node_class, delta_json "
                "FROM param_diffs WHERE node_class = ? "
                "ORDER BY run_id DESC, id DESC LIMIT ?",
                (node_class, last_n_runs * 20),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, run_id, node_id, node_title, node_class, delta_json "
                "FROM param_diffs ORDER BY run_id DESC, id DESC LIMIT ?",
                (last_n_runs * 20,),
            ).fetchall()
        conn.close()

        results = []
        for r in rows:
            # Defensive: a corrupted delta_json row would otherwise abort the
            # entire history query. Skip the bad row and log it.
            try:
                delta = json.loads(r[5])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"[MEC] param_history: dropping row ts={r[0]} run_id={r[1]} "
                    f"node_id={r[2]} with malformed delta_json ({e})"
                )
                continue
            results.append({
                "ts": r[0], "run_id": r[1], "node_id": r[2],
                "node_title": r[3], "node_class": r[4],
                "delta": delta,
            })
        return results
    except Exception as e:
        logger.warning(f"[MEC] param_history DB query error: {e}")
        return []


def _diff_runs(run_a: int, run_b: int) -> list[dict]:
    """Compare two runs and return parameter differences."""
    try:
        conn = _get_db()

        def _get_run_deltas(rid):
            rows = conn.execute(
                "SELECT node_id, node_title, node_class, delta_json "
                "FROM param_diffs WHERE run_id = ?",
                (rid,),
            ).fetchall()
            nodes = {}
            for r in rows:
                try:
                    delta = json.loads(r[3])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"[MEC] _diff_runs: dropping node_id={r[0]} with "
                        f"malformed delta_json ({e})"
                    )
                    continue
                nodes[r[0]] = {
                    "node_id": r[0], "node_title": r[1],
                    "node_class": r[2], "delta": delta,
                }
            return nodes

        na = _get_run_deltas(run_a)
        nb = _get_run_deltas(run_b)
        conn.close()

        diffs = []
        all_node_ids = set(na.keys()) | set(nb.keys())
        for nid in sorted(all_node_ids):
            da = na.get(nid, {}).get("delta", {})
            db_ = nb.get(nid, {}).get("delta", {})
            info = na.get(nid) or nb.get(nid)
            all_params = set(da.keys()) | set(db_.keys())
            for param in sorted(all_params):
                va = da.get(param)
                vb = db_.get(param)
                if va != vb:
                    diffs.append({
                        "node_id":    info["node_id"],
                        "node_title": info["node_title"],
                        "node_class": info["node_class"],
                        "param_name": param,
                        "run_a":      va,
                        "run_b":      vb,
                    })
        return diffs
    except Exception as e:
        logger.warning(f"[MEC] param_history diff error: {e}")
        return []


# ── L9.174 (A25): workflow branch snapshots + selective restore plan ───────────────────────────────────────────────
# Named snapshots are COPIES of a run's deltas in their own table, so the MAX_RUNS pruning never removes them. The
# table is created on first use only (the history modes never write to the database).
SNAPSHOT_ACTIONS = ["none", "save", "compare", "restore_plan"]


def _snap_db():
    conn = sqlite3.connect(_DB_PATH, timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS param_snapshots (
            name      TEXT PRIMARY KEY,
            ts        TEXT NOT NULL,
            run_id    INTEGER NOT NULL,
            data_json TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _run_nodes(run_id: int) -> dict:
    """{node_id: {node_title, node_class, delta}} of one stored run."""
    conn = _get_db()
    try:
        rows = conn.execute("SELECT node_id, node_title, node_class, delta_json FROM param_diffs WHERE run_id = ? "
                            "ORDER BY id", (int(run_id),)).fetchall()
    finally:
        conn.close()
    out = {}
    for nid, title, cls, dj in rows:
        try:
            delta = _strip_secrets(json.loads(dj))
        except (json.JSONDecodeError, TypeError):
            continue
        out.setdefault(str(nid), {"node_title": title, "node_class": cls, "delta": {}})["delta"].update(delta)
    return out


def _latest_run_id():
    conn = _get_db()
    try:
        r = conn.execute("SELECT MAX(run_id) FROM param_diffs").fetchone()
    finally:
        conn.close()
    return None if r is None or r[0] is None else int(r[0])


def _snapshot_names() -> list:
    conn = _snap_db()
    try:
        return [r[0] for r in conn.execute("SELECT name FROM param_snapshots ORDER BY name").fetchall()]
    finally:
        conn.close()


def _load_snapshot(name: str) -> dict:
    conn = _snap_db()
    try:
        r = conn.execute("SELECT ts, run_id, data_json FROM param_snapshots WHERE name = ?", (name,)).fetchone()
    finally:
        conn.close()
    if r is None:
        raise ValueError(f"Parameter History: no snapshot named {name!r}. Saved snapshots: "
                         f"{', '.join(_snapshot_names()) or 'none'}.")
    return {"name": name, "ts": r[0], "run_id": r[1], "nodes": json.loads(r[2])}


def save_snapshot(name: str, run_id: int = 0) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Parameter History: snapshot_name is empty - give the snapshot a name.")
    rid = int(run_id) if run_id and int(run_id) > 0 else _latest_run_id()
    if rid is None:
        raise ValueError("Parameter History: no runs recorded yet - run the workflow once, then save a snapshot.")
    nodes = _run_nodes(rid)
    if not nodes:
        raise ValueError(f"Parameter History: run #{rid} has no stored parameters.")
    conn = _snap_db()
    try:
        existed = conn.execute("SELECT 1 FROM param_snapshots WHERE name = ?", (name,)).fetchone() is not None
        conn.execute("INSERT OR REPLACE INTO param_snapshots (name, ts, run_id, data_json) VALUES (?, ?, ?, ?)",
                     (name, datetime.now().isoformat(sep=" ", timespec="seconds"), rid, json.dumps(nodes)))
        conn.commit()
    finally:
        conn.close()
    return {"action": "save", "name": name, "run_id": rid, "nodes": len(nodes),
            "params": sum(len(n["delta"]) for n in nodes.values()), "overwritten": existed}


def _resolve(name: str) -> tuple:
    """A snapshot by name, or the latest run when the name is empty -> (label, nodes)."""
    name = (name or "").strip()
    if name:
        snap = _load_snapshot(name)
        return f"snapshot {name!r} (run #{snap['run_id']})", snap["nodes"]
    rid = _latest_run_id()
    if rid is None:
        raise ValueError("Parameter History: no runs recorded yet.")
    return f"latest run #{rid}", _run_nodes(rid)


def compare_snapshots(name: str, compare_to: str = "") -> dict:
    if not (name or "").strip():
        raise ValueError("Parameter History: compare needs snapshot_name (compare_to empty = the latest run).")
    la, na = _resolve(name)
    lb, nb = _resolve(compare_to)
    diffs = []
    for nid in sorted(set(na) | set(nb)):
        da, db_ = (na.get(nid) or {}).get("delta", {}), (nb.get(nid) or {}).get("delta", {})
        info = na.get(nid) or nb.get(nid)
        for param in sorted(set(da) | set(db_)):
            if da.get(param) != db_.get(param):
                diffs.append({"node_id": nid, "node_title": info["node_title"], "node_class": info["node_class"],
                              "param_name": param, "run_a": da.get(param), "run_b": db_.get(param)})
    return {"action": "compare", "a": la, "b": lb, "differences": diffs, "count": len(diffs)}


def restore_plan(name: str, restore_filter: str) -> dict:
    tokens = [t.strip() for t in (restore_filter or "").split(",") if t.strip()]
    if not tokens:
        raise ValueError("Parameter History: restore_plan needs restore_filter (node class / title / id, a "
                         "parameter name or Class.param, comma-separated) - restoring everything is not selective.")
    label, nodes = _resolve(name)
    low = [t.lower() for t in tokens]
    restore, matched, skipped = {}, 0, 0
    for nid, n in nodes.items():
        node_hit = any(t in (n["node_class"].lower(), (n["node_title"] or "").lower(), nid.lower()) for t in low)
        for param, val in n["delta"].items():
            hit = node_hit or any(t in (param.lower(), f"{n['node_class']}.{param}".lower()) for t in low)
            if hit:
                restore.setdefault(nid, {"node_class": n["node_class"], "node_title": n["node_title"],
                                         "params": {}})["params"][param] = val
                matched += 1
            else:
                skipped += 1
    return {"action": "restore_plan", "source": label, "filter": tokens, "restore": restore,
            "matched_params": matched, "skipped_params": skipped,
            "note": "the exact values to set on those nodes; every other setting stays as it is"}


# ══════════════════════════════════════════════════════════════════════
#  ComfyUI Node: Parameter History Viewer
# ══════════════════════════════════════════════════════════════════════

# A28 (docs/audit/A28/PARAM_CNP_B.md, ParameterHistoryMEC): each mode's own inputs (query(): diff mode reads run_a /
# run_b only, the history modes last_n_runs). Display only.
_HISTORY_LAYOUT = {
    "last_n_runs": {"c2c_show_if": {"mode": ["all_history", "node_class_filter"]}},
    "node_class_filter": {"c2c_show_if": {"mode": ["node_class_filter"]}},
    "run_a": {"c2c_show_if": {"mode": ["last_run_diff"]}},
    "run_b": {"c2c_show_if": {"mode": ["last_run_diff"]}},
    "snapshot_name": {"c2c_show_if": {"snapshot_action": ["save", "compare", "restore_plan"]}},
    "compare_to": {"c2c_show_if": {"snapshot_action": ["compare"]}},
    "restore_filter": {"c2c_show_if": {"snapshot_action": ["restore_plan"]}},
}


class ParameterHistoryMEC:
    """View and query the parameter history database.

    Connects to the SQLite DB written by the JS extension and provides
    structured output of parameter change history.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "mode": (["all_history", "last_run_diff", "node_class_filter"], {
                    "default": "all_history",
                    "tooltip": (
                        "all_history: show recent parameter changes across all nodes\n"
                        "last_run_diff: show what changed between the last two runs\n"
                        "node_class_filter: filter history to a specific node class"
                    ),
                }),
                "last_n_runs": ("INT", {
                    "default": 5, "min": 1, "max": 100, "step": 1,
                    "tooltip": "How many recent runs to include",
                }),
            },
            "optional": {
                "node_class_filter": ("STRING", {
                    "default": "",
                    "tooltip": "Node class name to filter by (e.g. 'KSampler')",
                }),
                "run_a": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "First run number for diff mode (0 = auto-detect last two)",
                }),
                "run_b": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Second run number for diff mode (0 = auto-detect last two)",
                }),
                # L9.174 (A25), appended: "none" = the behaviour of every saved workflow
                "snapshot_action": (SNAPSHOT_ACTIONS, {
                    "default": "none",
                    "tooltip": "Branch snapshots. save: store the parameters of run_b (0 = the latest run) under "
                               "snapshot_name; snapshots are kept when old runs are pruned. compare: differences "
                               "between snapshot_name and compare_to. restore_plan: only the parameters matching "
                               "restore_filter from snapshot_name (empty = latest run). Result in snapshot_json.",
                }),
                "snapshot_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the snapshot to save, compare or restore from (e.g. 'branch-A').",
                }),
                "compare_to": ("STRING", {
                    "default": "",
                    "tooltip": "compare: the second snapshot's name; empty = the latest run.",
                }),
                "restore_filter": ("STRING", {
                    "default": "",
                    "tooltip": "restore_plan: comma-separated node classes, node titles, node ids, parameter names "
                               "or Class.param (e.g. 'KSampler, cfg'). Only matching parameters are listed; the "
                               "output is the exact patch to apply, nothing else is touched.",
                }),
            },
        }, _HISTORY_LAYOUT)

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("history_report", "snapshot_json")
    OUTPUT_TOOLTIPS = ("Human-readable parameter history report for the chosen mode.",
                       "snapshot_action result as JSON: the saved snapshot, the differences, or the restore patch "
                       "{node_id: {params}}. Empty when snapshot_action is none.")
    OUTPUT_NODE = True
    FUNCTION = "query"
    CATEGORY = "C2C/Utils"
    DESCRIPTION = (
        "Query the parameter history database. Shows what parameters were changed, "
        "when, and what the previous values were. Supports run-to-run diffs."
    )

    @classmethod
    def IS_CHANGED(cls, mode="all_history", last_n_runs=5, node_class_filter="", run_a=0, run_b=0,
                   snapshot_action="none", snapshot_name="", compare_to="", restore_filter="", **_kw):
        """L9.196: the report reads a database the front end keeps writing; without this a re-queue showed the cached,
        stale report. The database's mtime and size are the fingerprint (no read, O(1))."""
        try:
            st = os.stat(_DB_PATH)
            db = f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            db = "missing"
        base = f"{db}|{mode}|{last_n_runs}|{node_class_filter}|{run_a}|{run_b}"
        if snapshot_action == "none":
            return base                                         # the saved-workflow fingerprint, unchanged
        return f"{base}|{snapshot_action}|{snapshot_name}|{compare_to}|{restore_filter}"

    def query(self, mode, last_n_runs, node_class_filter="", run_a=0, run_b=0, snapshot_action="none",
              snapshot_name="", compare_to="", restore_filter=""):
        if snapshot_action not in SNAPSHOT_ACTIONS:
            raise ValueError(f"Parameter History: snapshot_action must be one of {SNAPSHOT_ACTIONS}")
        if mode == "last_run_diff":
            report = self._diff_mode(run_a, run_b)
        elif mode == "node_class_filter":
            report = self._filter_mode(node_class_filter, last_n_runs)
        else:
            report = self._all_mode(last_n_runs)
        snap = ""
        if snapshot_action == "save":
            snap = json.dumps(save_snapshot(snapshot_name, run_b), indent=2, default=str)
        elif snapshot_action == "compare":
            snap = json.dumps(compare_snapshots(snapshot_name, compare_to), indent=2, default=str)
        elif snapshot_action == "restore_plan":
            snap = json.dumps(restore_plan(snapshot_name, restore_filter), indent=2, default=str)
        return (report[0], snap)

    def _all_mode(self, last_n_runs):
        rows = _query_history(last_n_runs=last_n_runs)
        if not rows:
            return ("No parameter history recorded yet. Run the workflow at least once.",)

        # Group by run_id
        runs: dict[int, list] = {}
        for r in rows:
            rid = r["run_id"]
            if rid not in runs:
                runs[rid] = []
            runs[rid].append(r)

        lines = ["═══ Parameter History (deltas from defaults) ═══", ""]
        for rid in sorted(runs.keys(), reverse=True):
            entries = runs[rid]
            ts = entries[0]["ts"] if entries else "?"
            lines.append(f"── Run #{rid} ({ts}) ──")
            for e in entries:
                delta = e["delta"]
                params = ", ".join(f"{k}={json.dumps(v)}" for k, v in delta.items())
                lines.append(f"  {e['node_title']} ({e['node_class']})")
                lines.append(f"    {params}")
            lines.append("")

        return ("\n".join(lines),)

    def _diff_mode(self, run_a, run_b):
        if run_a == 0 or run_b == 0:
            try:
                conn = _get_db()
                rids = conn.execute(
                    "SELECT DISTINCT run_id FROM param_diffs ORDER BY run_id DESC LIMIT 2"
                ).fetchall()
                conn.close()
                if len(rids) < 2:
                    return ("Need at least 2 runs for diff. Run the workflow more.",)
                run_b = rids[0][0]
                run_a = rids[1][0]
            except Exception:
                return ("Could not read run history from DB.",)

        diffs = _diff_runs(run_a, run_b)
        if not diffs:
            return (f"No parameter differences between Run #{run_a} and Run #{run_b}.",)

        lines = [f"═══ Diff: Run #{run_a} → Run #{run_b} ═══", ""]
        for d in diffs:
            lines.append(
                f"  {d['node_title']} ({d['node_class']}) → {d['param_name']}: "
                f"{d['run_a']} → {d['run_b']}"
            )
        return ("\n".join(lines),)

    def _filter_mode(self, node_class, last_n_runs):
        if not node_class:
            return ("Provide a node_class_filter value to filter history.",)

        rows = _query_history(node_class=node_class, last_n_runs=last_n_runs)
        if not rows:
            return (f"No history found for node class '{node_class}'.",)

        lines = [f"═══ History for {node_class} (deltas) ═══", ""]
        for r in rows:
            delta = r["delta"]
            params = ", ".join(f"{k}={json.dumps(v)}" for k, v in delta.items())
            lines.append(f"  Run #{r['run_id']} ({r['ts']})")
            lines.append(f"    {params}")
        return ("\n".join(lines),)


# ══════════════════════════════════════════════════════════════════════
#  Server route registration (called from __init__.py)
# ══════════════════════════════════════════════════════════════════════

def register_routes(server):
    """Register /mec/param_history POST endpoint on the ComfyUI server."""
    from aiohttp import web

    @server.routes.post("/mec/param_history")
    async def _handle_param_history(request):
        try:
            data = await request.json()
            _store_snapshot(data)
            return web.json_response({"status": "ok"})
        except Exception as e:
            logger.warning(f"[MEC] /mec/param_history error: {e}")
            return web.json_response({"status": "error", "msg": str(e)}, status=400)
