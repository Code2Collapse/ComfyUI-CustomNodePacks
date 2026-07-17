// rib_dashboard.js — RIB render-farm Master Dashboard (Tractor-style).
// ---------------------------------------------------------------------------
// Sidebar tab with four views over the /rib/* REST routes:
//   LIVE    — running jobs: progress bar + live preview thumbnail (backend ws)
//   QUEUE   — jobs held locally waiting for capacity, with Bump Priority
//   HISTORY — sortable table over the SQLite audit log
//   ADMIN   — every user's jobs with cancel/pause (server enforces roles;
//             non-admins get a 403 and a plain-English toast)
// Polls only while the tab is visible (2s live / 5s history). Plain DOM, no
// deps; renders whatever the server returns — role logic stays server-side.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TAB_ID = "rib-dashboard";
let _root = null;
let _view = "live";
let _timer = null;
let _actor = { name: "", role: "unset" };
let _histSort = { key: "submitted_at", dir: -1 };

const CSS = `
.rib-wrap { display:flex; flex-direction:column; height:100%; font: 12px ui-monospace,monospace; color: var(--fg-color,#ddd); }
.rib-tabs { display:flex; gap:2px; padding:4px; border-bottom:1px solid rgba(255,255,255,0.12); }
.rib-tabs button { flex:1; padding:5px 2px; background:rgba(255,255,255,0.06); border:0; border-radius:4px; color:inherit; cursor:pointer; font:inherit; }
.rib-tabs button.on { background:#3a5f8a; color:#fff; }
.rib-body { flex:1; overflow:auto; padding:6px; }
.rib-user { padding:4px 8px; font-size:11px; opacity:0.8; border-bottom:1px solid rgba(255,255,255,0.08); }
.rib-user .role { color:#8cff66; } .rib-user .role.unset { color:#ff8c66; }
.rib-card { border:1px solid rgba(255,255,255,0.12); border-radius:6px; padding:6px; margin-bottom:6px; background:rgba(0,0,0,0.25); }
.rib-card .top { display:flex; justify-content:space-between; align-items:center; gap:4px; }
.rib-card .meta { opacity:0.75; font-size:11px; margin:3px 0; }
.rib-bar { height:8px; background:rgba(255,255,255,0.1); border-radius:4px; overflow:hidden; margin:4px 0; }
.rib-bar > div { height:100%; background:#4da6ff; transition:width .4s; }
.rib-prev { max-width:100%; border-radius:4px; margin-top:4px; display:block; }
.rib-btn { background:rgba(255,255,255,0.1); border:0; border-radius:4px; color:inherit; padding:3px 8px; cursor:pointer; font:inherit; }
.rib-btn:hover { background:rgba(255,255,255,0.22); }
.rib-btn.danger:hover { background:#8a3a3a; }
.rib-table { width:100%; border-collapse:collapse; font-size:11px; }
.rib-table th { text-align:left; padding:3px 5px; cursor:pointer; border-bottom:1px solid rgba(255,255,255,0.2); position:sticky; top:0; background:var(--comfy-menu-bg,#202020); user-select:none; }
.rib-table td { padding:3px 5px; border-bottom:1px solid rgba(255,255,255,0.06); white-space:nowrap; }
.rib-status-complete { color:#8cff66; } .rib-status-failed { color:#ff6666; }
.rib-status-running { color:#4da6ff; } .rib-status-queued, .rib-status-paused { color:#ffc966; }
.rib-status-cancelled { color:#999; }
.rib-empty { opacity:0.6; padding:14px; text-align:center; }
`;

function esc(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function jget(path) {
    const r = await api.fetchApi(path);
    if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
    return r.json();
}

async function jpost(path, body) {
    const r = await api.fetchApi(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.ok === false) {
        toast(data.error || `${path} failed (HTTP ${r.status})`, "error");
        return false;
    }
    return true;
}

function toast(msg, severity = "info") {
    try {
        app.extensionManager?.toast?.add?.({ severity, summary: "RIB Farm", detail: msg, life: 4000 });
    } catch (_) { console.log("[RIB]", msg); }
}

function fmtTime(t) {
    return t ? new Date(t * 1000).toLocaleString() : "-";
}

function fmtDur(s) {
    if (s == null) return "-";
    s = Math.round(s);
    return s >= 3600 ? `${(s / 3600).toFixed(1)}h` : s >= 60 ? `${(s / 60).toFixed(1)}m` : `${s}s`;
}

// ── job card (live + admin views) ───────────────────────────────────
function jobCard(j, { withPreview } = {}) {
    const card = document.createElement("div");
    card.className = "rib-card";
    const pct = j.progress != null ? Math.round(j.progress * 100) : null;
    card.innerHTML = `
      <div class="top">
        <b>${esc(j.job_id)}</b>
        <span class="rib-status-${esc(j.status)}">${esc(j.status)}</span>
      </div>
      <div class="meta">${esc(j.user)} · ${esc(j.project_name || "no project")} ·
        ${esc(j.backend_name)} · ${esc(j.compute_profile)} · prio ${j.priority}</div>
      ${j.status === "running" || j.status === "dispatching"
        ? `<div class="rib-bar"><div style="width:${pct ?? 4}%"></div></div>
           <div class="meta">${pct != null ? pct + "%" : "progress unknown (install websocket-client for live %)"}</div>`
        : ""}
      ${j.error ? `<div class="meta" style="color:#ff8888">${esc(j.error)}</div>` : ""}`;
    const row = document.createElement("div");
    const canControl = _actor.role === "admin" || _actor.name === j.user;
    if (canControl && !["complete", "failed", "cancelled"].includes(j.status)) {
        const cancel = document.createElement("button");
        cancel.className = "rib-btn danger";
        cancel.textContent = "Cancel";
        cancel.onclick = () => jpost("/rib/cancel", { job_id: j.job_id }).then(refresh);
        row.appendChild(cancel);
        if (j.status === "queued") {
            const pause = document.createElement("button");
            pause.className = "rib-btn";
            pause.textContent = "Pause";
            pause.style.marginLeft = "4px";
            pause.onclick = () => jpost("/rib/pause", { job_id: j.job_id }).then(refresh);
            row.appendChild(pause);
        }
        if (j.status === "paused") {
            const res = document.createElement("button");
            res.className = "rib-btn";
            res.textContent = "Resume";
            res.style.marginLeft = "4px";
            res.onclick = () => jpost("/rib/resume", { job_id: j.job_id }).then(refresh);
            row.appendChild(res);
        }
    }
    card.appendChild(row);
    if (withPreview && (j.status === "running" || j.status === "dispatching")) {
        const img = document.createElement("img");
        img.className = "rib-prev";
        img.alt = "";
        jget(`/rib/preview/${encodeURIComponent(j.job_id)}`)
            .then((d) => { if (d.preview) img.src = d.preview; else img.remove(); })
            .catch(() => img.remove());
        card.appendChild(img);
    }
    return card;
}

// ── views ────────────────────────────────────────────────────────────
async function renderLive(body) {
    const snap = await jget("/rib/jobs");
    _actor = snap.actor || _actor;
    updateUserline();
    body.replaceChildren();
    const live = [...snap.active, ...snap.recent.slice(0, 8)];
    if (!live.length) {
        body.innerHTML = `<div class="rib-empty">No running jobs. Spool one with the
            <b>RIB Farm Submit</b> node.</div>`;
        return;
    }
    for (const j of live) body.appendChild(jobCard(j, { withPreview: true }));
}

async function renderQueue(body) {
    const snap = await jget("/rib/jobs");
    _actor = snap.actor || _actor;
    updateUserline();
    body.replaceChildren();
    if (!snap.pending.length) {
        body.innerHTML = `<div class="rib-empty">Local spool queue is empty — nothing is
            waiting for backend capacity.</div>`;
        return;
    }
    for (const j of snap.pending) {
        const card = jobCard(j);
        const bump = document.createElement("button");
        bump.className = "rib-btn";
        bump.textContent = "▲ Bump priority";
        bump.style.marginTop = "4px";
        bump.onclick = () => jpost("/rib/bump", { job_id: j.job_id, priority: j.priority + 1 }).then(refresh);
        card.appendChild(bump);
        body.appendChild(card);
    }
}

async function renderHistory(body) {
    const { rows } = await jget("/rib/history?limit=300");
    body.replaceChildren();
    if (!rows.length) {
        body.innerHTML = `<div class="rib-empty">Audit log is empty.</div>`;
        return;
    }
    const cols = [["job_id", "Job"], ["user", "User"], ["project_name", "Project"],
                  ["backend_name", "Backend"], ["priority", "Prio"], ["status", "Status"],
                  ["submitted_at", "Submitted"], ["duration_seconds", "Duration"]];
    rows.sort((a, b) => {
        const k = _histSort.key, va = a[k] ?? "", vb = b[k] ?? "";
        return (va > vb ? 1 : va < vb ? -1 : 0) * _histSort.dir;
    });
    const table = document.createElement("table");
    table.className = "rib-table";
    const thead = table.createTHead().insertRow();
    for (const [key, label] of cols) {
        const th = document.createElement("th");
        th.textContent = label + (_histSort.key === key ? (_histSort.dir > 0 ? " ▲" : " ▼") : "");
        th.onclick = () => {
            _histSort = { key, dir: _histSort.key === key ? -_histSort.dir : -1 };
            renderHistory(body);
        };
        thead.appendChild(th);
    }
    const tb = table.createTBody();
    for (const r of rows) {
        const tr = tb.insertRow();
        tr.innerHTML = `
          <td>${esc(r.job_id)}</td><td>${esc(r.user)}</td><td>${esc(r.project_name)}</td>
          <td>${esc(r.backend_name)}</td><td>${r.priority}</td>
          <td class="rib-status-${esc(r.status)}">${esc(r.status)}</td>
          <td>${fmtTime(r.submitted_at)}</td><td>${fmtDur(r.duration_seconds)}</td>`;
    }
    body.appendChild(table);
}

async function renderAdmin(body) {
    const snap = await jget("/rib/jobs");
    _actor = snap.actor || _actor;
    updateUserline();
    body.replaceChildren();
    if (_actor.role !== "admin") {
        body.innerHTML = `<div class="rib-empty">Admin view — your role is
            '${esc(_actor.role)}'. Ask an admin to promote you in
            renderfarm/config/users.json.</div>`;
        return;
    }
    const all = [...snap.active, ...snap.pending, ...snap.recent];
    if (!all.length) {
        body.innerHTML = `<div class="rib-empty">No jobs anywhere on the farm.</div>`;
        return;
    }
    for (const j of all) body.appendChild(jobCard(j));
    // cluster capacity footer
    try {
        const { backends } = await jget("/rib/cluster");
        const foot = document.createElement("div");
        foot.className = "rib-card";
        foot.innerHTML = "<b>Cluster</b>" + backends.map((b) => `
            <div class="meta">${esc(b.backend)} — ${b.disabled ? "disabled"
                : b.reachable ? `up · running ${b.running} · pending ${b.pending}`
                : `unreachable${b.error ? " · " + esc(b.error) : ""}`}</div>`).join("");
        body.appendChild(foot);
    } catch (_) { /* capacity is best-effort */ }
}

const VIEWS = { live: renderLive, queue: renderQueue, history: renderHistory, admin: renderAdmin };

// ── shell ────────────────────────────────────────────────────────────
function updateUserline() {
    const el = _root?.querySelector(".rib-user");
    if (el) el.innerHTML = `user: <b>${esc(_actor.name || "?")}</b> ·
        role: <span class="role ${esc(_actor.role)}">${esc(_actor.role)}</span>` +
        (_actor.error ? ` — <span style="color:#ff8c66">${esc(_actor.error)}</span>` : "");
}

async function refresh() {
    const body = _root?.querySelector(".rib-body");
    if (!body || !_root.isConnected) return;
    try {
        await VIEWS[_view](body);
    } catch (exc) {
        body.innerHTML = `<div class="rib-empty">Dashboard error: ${esc(exc.message)}<br>
            Is the RIB backend package loaded? Check the ComfyUI log for
            'render_farm'.</div>`;
    }
}

function startPolling() {
    stopPolling();
    const tick = () => {
        // The sidebar hands render(el) a DETACHED element and mounts it after
        // render returns — a not-yet/no-longer connected root must SKIP the
        // poll but keep the timer alive, otherwise the dashboard dies on its
        // very first tick and never paints.
        if (_root && _root.isConnected && !document.hidden) refresh();
        _timer = setTimeout(tick, _view === "history" ? 5000 : 2000);
    };
    tick();
}

function stopPolling() {
    if (_timer) { clearTimeout(_timer); _timer = null; }
}

function renderTab(el) {
    _root = document.createElement("div");
    _root.className = "rib-wrap";
    if (!document.getElementById("rib-dash-css")) {
        const style = document.createElement("style");
        style.id = "rib-dash-css";
        style.textContent = CSS;
        document.head.appendChild(style);
    }
    const tabs = document.createElement("div");
    tabs.className = "rib-tabs";
    for (const [id, label] of [["live", "Live"], ["queue", "Queue"], ["history", "History"], ["admin", "Admin"]]) {
        const b = document.createElement("button");
        b.textContent = label;
        b.dataset.view = id;
        b.className = id === _view ? "on" : "";
        b.onclick = () => {
            _view = id;
            tabs.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.view === id));
            refresh();
        };
        tabs.appendChild(b);
    }
    const userline = document.createElement("div");
    userline.className = "rib-user";
    const body = document.createElement("div");
    body.className = "rib-body";
    _root.append(tabs, userline, body);
    el.replaceChildren(_root);
    startPolling();
}

if (!(app.extensions || []).some((e) => e?.name === "C2C.RIBDashboard")) app.registerExtension({
    name: "C2C.RIBDashboard",
    async setup() {
        if (app.extensionManager?.registerSidebarTab) {
            app.extensionManager.registerSidebarTab({
                id: TAB_ID,
                icon: "pi pi-server",
                title: "RIB Farm",
                tooltip: "RIB render farm — jobs, queue, history, cluster",
                type: "custom",
                render: (el) => renderTab(el),
            });
        }
        // Submit-node progress bar (rides the same event style as other packs).
        api.addEventListener("rib.job.progress", (ev) => {
            const d = ev.detail || {};
            const node = app.graph?._nodes?.find((n) => String(n.id) === String(d.node));
            if (!node) return;
            node.__ribProg = { pct: +d.pct || 0, label: d.label || "", t: performance.now() };
            app.graph.setDirtyCanvas(true, false);
        });
    },
    // NOTE: beforeRegisterNodeDef fires BEFORE setup() in the extension
    // lifecycle — the draw hook must not depend on setup-time state.
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "RIB_Submit") return;
        const onDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDraw ? onDraw.apply(this, arguments) : undefined;
            const p = this.__ribProg;
            if (!p) return r;
            if (p.pct >= 1 && performance.now() - p.t > 2500) { this.__ribProg = null; return r; }
            const w = this.size[0] - 20;
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.fillRect(10, 4, w, 12);
            ctx.fillStyle = p.pct >= 1 ? "#8cff66" : "#4da6ff";
            ctx.fillRect(11, 5, Math.max(2, (w - 2) * Math.min(1, p.pct)), 10);
            ctx.fillStyle = "rgba(240,244,255,0.95)";
            ctx.font = "9px ui-monospace,monospace";
            ctx.textAlign = "left"; ctx.textBaseline = "middle";
            let txt = p.label;
            if (ctx.measureText(txt).width > w - 8) txt = txt.slice(0, 42) + "…";
            ctx.fillText(txt, 14, 10);
            ctx.restore();
            return r;
        };
    },
});
