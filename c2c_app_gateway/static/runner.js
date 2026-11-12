// C2C App Gateway runner pages. Everything shown comes from the public manifest and the job projection;
// this page never sees a workflow. Text is always set with textContent.
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  let me = null;

  async function api(path, opts = {}) {
    const o = { credentials: "same-origin", ...opts, headers: { ...(opts.headers || {}) } };
    if (o.method && o.method !== "GET" && me && me.csrf) o.headers["X-C2C-CSRF"] = me.csrf;
    const r = await fetch(path, o);
    let body = null;
    try { body = await r.json(); } catch (_) { body = null; }
    return { ok: r.ok, status: r.status, body };
  }

  async function whoami() {
    const r = await api("/api/v1/me");
    me = r.ok ? r.body : null;
    if (me) {
      $("who").textContent = me.user;
      $("logout").hidden = false;
      $("logout").onclick = async () => { await api("/api/v1/logout", { method: "POST" }); location.href = "/"; };
    }
    return me;
  }

  // ── index ────────────────────────────────────────────────────────────────────────────────────
  async function index() {
    if (!(await whoami())) {
      $("login").hidden = false;
      $("login-form").onsubmit = async (e) => {
        e.preventDefault();
        $("login-error").textContent = "";
        const r = await api("/api/v1/login", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user: $("user").value, password: $("password").value }) });
        if (!r.ok) { $("login-error").textContent = (r.body && r.body.message) || "Sign-in failed."; return; }
        const next = new URLSearchParams(location.search).get("next");
        location.href = next && /^\/app\/[a-z0-9-]+$/.test(next) ? next : "/";
      };
      return;
    }
    const r = await api("/api/v1/apps");
    $("apps").hidden = false;
    const list = (r.body && r.body.apps) || [];
    $("no-apps").hidden = list.length > 0;
    for (const a of list) {
      const li = document.createElement("li"), link = document.createElement("a"), d = document.createElement("span");
      link.href = "/app/" + encodeURIComponent(a.app_id);
      link.textContent = a.name;
      d.className = "desc";
      d.textContent = a.description || "";
      link.appendChild(d);
      li.appendChild(link);
      $("app-list").appendChild(li);
    }
  }

  // ── app ──────────────────────────────────────────────────────────────────────────────────────
  function field(inp) {
    const wrap = document.createElement("div"), id = "in_" + inp.id;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = inp.label + (inp.required ? " *" : "");
    wrap.appendChild(label);
    let el;
    switch (inp.type) {
      case "image":
        el = document.createElement("input"); el.type = "file"; el.accept = inp.accept.join(",");
        if (inp.required) el.required = true;
        break;
      case "text":
        el = document.createElement("textarea"); el.maxLength = inp.max_len; el.value = inp.default || "";
        break;
      case "number": case "int":
        el = document.createElement("input"); el.type = "number"; el.min = inp.min; el.max = inp.max;
        el.step = inp.step || (inp.type === "int" ? 1 : "any");
        if (inp.default !== undefined) el.value = inp.default;
        break;
      case "choice":
        el = document.createElement("select");
        for (const o of inp.options) { const op = document.createElement("option"); op.value = o; op.textContent = o; el.appendChild(op); }
        if (inp.default !== undefined) el.value = inp.default;
        break;
      case "seed":
        el = document.createElement("input"); el.type = "number"; el.min = 0; el.step = 1;
        el.placeholder = "random";
        if (inp.default !== undefined) el.value = inp.default;
        break;
      case "boolean":
        el = document.createElement("input"); el.type = "checkbox"; el.checked = !!inp.default;
        break;
    }
    el.id = id;
    el.dataset.input = inp.id;
    el.dataset.type = inp.type;
    wrap.appendChild(el);
    if (inp.help) { const h = document.createElement("p"); h.className = "help"; h.textContent = inp.help; wrap.appendChild(h); }
    return wrap;
  }

  function collect(form) {
    const fd = new FormData(), values = {};
    for (const el of form.querySelectorAll("[data-input]")) {
      const id = el.dataset.input, t = el.dataset.type;
      if (t === "image") { if (el.files[0]) fd.append(id, el.files[0], el.files[0].name); continue; }
      if (t === "boolean") { values[id] = el.checked; continue; }
      if (t === "number" || t === "int" || t === "seed") { if (el.value !== "") values[id] = Number(el.value); continue; }
      values[id] = el.value;
    }
    fd.append("values", JSON.stringify(values));
    return fd;
  }

  function showJob(view) {
    $("job").hidden = false;
    const s = $("job-state");
    s.textContent = view.state; s.dataset.state = view.state;
    const pct = Math.round((view.progress || 0) * 100);
    $("meter-fill").style.width = pct + "%";
    $("meter").setAttribute("aria-valuenow", String(pct));
    $("job-msg").textContent = view.error ? view.error.message : (view.state === "queued" ? "Waiting for the server…" : "");
    $("cancel").hidden = ["succeeded", "failed", "cancelled"].includes(view.state);
  }

  async function showOutputs(jobId) {
    const r = await api("/api/v1/jobs/" + encodeURIComponent(jobId) + "/outputs");
    const box = $("outputs");
    box.replaceChildren();
    for (const o of (r.body && r.body.outputs) || []) {
      const fig = document.createElement("figure");
      if (o.type === "image") { const im = document.createElement("img"); im.src = o.url; im.alt = o.id; fig.appendChild(im); }
      else if (o.type === "video") { const v = document.createElement("video"); v.src = o.url; v.controls = true; fig.appendChild(v); }
      const cap = document.createElement("figcaption"), a = document.createElement("a");
      a.href = o.url; a.download = ""; a.textContent = "Download " + o.id + (o.index ? " " + (o.index + 1) : "");
      cap.appendChild(a); fig.appendChild(cap); box.appendChild(fig);
    }
  }

  function follow(jobId) {
    const es = new EventSource("/api/v1/jobs/" + encodeURIComponent(jobId) + "/events");
    es.addEventListener("job", (e) => {
      const v = JSON.parse(e.data);
      showJob(v);
      if (["succeeded", "failed", "cancelled"].includes(v.state)) {
        es.close(); $("run").disabled = false;
        if (v.state === "succeeded") showOutputs(jobId);
      }
    });
    es.addEventListener("revoked", () => { es.close(); $("job-msg").textContent = "Your access to this app was removed."; });
    es.onerror = () => { /* the browser reconnects; terminal states close the stream */ };
    $("cancel").onclick = async () => {
      $("cancel").disabled = true;
      const r = await api("/api/v1/jobs/" + encodeURIComponent(jobId) + "/cancel", { method: "POST" });
      $("cancel").disabled = false;
      if (r.ok) showJob(r.body); else $("job-msg").textContent = (r.body && r.body.message) || "Could not cancel.";
    };
  }

  async function app() {
    if (!(await whoami())) { location.href = "/?next=" + encodeURIComponent(location.pathname); return; }
    const appId = decodeURIComponent(location.pathname.split("/").pop());
    const r = await api("/api/v1/apps/" + encodeURIComponent(appId) + "/manifest");
    if (!r.ok) { $("app-name").textContent = "App not found"; return; }
    const m = r.body;
    document.title = m.name;
    $("app-name").textContent = m.name;
    $("app-desc").textContent = m.description || "";
    $("app-release").textContent = "Release " + m.release;
    const fields = $("fields");
    for (const inp of m.inputs) fields.appendChild(field(inp));
    $("run-form").hidden = false;
    $("run-form").onsubmit = async (e) => {
      e.preventDefault();
      $("run-error").textContent = "";
      $("run").disabled = true;
      $("outputs").replaceChildren();
      const res = await api("/api/v1/apps/" + encodeURIComponent(appId) + "/runs", { method: "POST", body: collect(e.target) });
      if (!res.ok) { $("run").disabled = false; $("run-error").textContent = (res.body && res.body.message) || "The run was not accepted."; return; }
      showJob(res.body);
      follow(res.body.job_id);
    };
  }

  document.addEventListener("DOMContentLoaded", () => {
    (document.body.dataset.page === "app" ? app : index)().catch((err) => console.error(err));
  });
})();
