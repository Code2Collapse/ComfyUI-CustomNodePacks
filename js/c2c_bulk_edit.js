// c2c_bulk_edit.js — Bulk Value Editor for multi-selected nodes (C2C)
// ---------------------------------------------------------------------
// What it does:
//   • Select 2+ nodes of the SAME class (or compatible classes) →
//     press Ctrl+Shift+E (or right-click → "Bulk-edit widgets…") →
//     floating panel shows the INTERSECTION of widgets across the
//     selection. Edit a value here → applied to every selected node.
//   • Numeric, text, combo, boolean widgets all supported.
//   • "Apply only if all equal" toggle: shows a `…` placeholder if
//     selected nodes disagree, and you can choose to leave them
//     untouched or unify to a new value.
//   • Esc closes.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const ROOT_ID = "c2c-bulk-edit-root";
const SETTING_ID = "c2c.bulkEdit.enabled";

let _root = null, _body = null, _hdr = null;
let _targets = [];

function injectStyle() {
    if (document.getElementById("c2c-bulk-edit-style")) return;
    const s = document.createElement("style");
    s.id = "c2c-bulk-edit-style";
    s.textContent = `
#${ROOT_ID} {
    position: fixed; top: 80px; right: 24px;
    z-index: 100000;
    width: 360px; max-height: 70vh;
    background: rgba(22,24,30,0.96);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    box-shadow: 0 10px 36px rgba(0,0,0,0.55);
    backdrop-filter: blur(10px);
    color: #e7ecf3; font: 12px ui-sans-serif, system-ui, sans-serif;
    overflow: hidden; display: flex; flex-direction: column;
}
#${ROOT_ID} .hdr {
    padding: 9px 12px;
    background: rgba(91,141,239,0.12);
    border-bottom: 1px solid rgba(91,141,239,0.25);
    display: flex; align-items: center; gap: 8px;
}
#${ROOT_ID} .hdr .ttl { font-weight: 600; color:#9ec1ff; }
#${ROOT_ID} .hdr .closex {
    margin-left:auto; cursor:pointer; color:#9aa6b2;
    width:18px; height:18px; line-height:18px; text-align:center;
    border-radius:4px;
}
#${ROOT_ID} .hdr .closex:hover { color:#fff; background:rgba(255,255,255,0.07); }
#${ROOT_ID} .body {
    padding: 8px 10px 12px; overflow-y: auto;
}
#${ROOT_ID} .row {
    display: grid; grid-template-columns: 110px 1fr;
    gap: 6px; padding: 4px 0; align-items: center;
}
#${ROOT_ID} .row label {
    color: #cfd6e0; font-size: 11.5px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
#${ROOT_ID} .row input, #${ROOT_ID} .row select {
    width: 100%; box-sizing: border-box;
    background: rgba(0,0,0,0.30); color:#fff;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 5px; padding: 4px 7px;
    font: 12px ui-monospace, monospace;
}
#${ROOT_ID} .row.differs input,
#${ROOT_ID} .row.differs select {
    border-color: rgba(255,209,102,0.55);
}
#${ROOT_ID} .row .badge {
    font-size: 9.5px; color: #ffd166;
    margin-left: 4px;
}
#${ROOT_ID} .empty {
    color:#7d8896; text-align:center; padding: 16px 0;
}`;
    document.head.appendChild(s);
}

function selectedNodes() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return [];
    return Object.values(sel);
}

function commonWidgets(nodes) {
    if (!nodes.length) return [];
    // Build (name, type) tuples present on EVERY node.
    const base = nodes[0].widgets || [];
    const out = [];
    for (const w of base) {
        if (!w?.name) continue;
        let allMatch = true;
        let values = [w.value];
        for (let i = 1; i < nodes.length; i++) {
            const w2 = (nodes[i].widgets || []).find((x) => x.name === w.name && x.type === w.type);
            if (!w2) { allMatch = false; break; }
            values.push(w2.value);
        }
        if (!allMatch) continue;
        const differs = values.some((v) => v !== values[0]);
        out.push({ name: w.name, type: w.type, refWidget: w, values, differs });
    }
    return out;
}

function applyValue(name, type, raw) {
    let v = raw;
    if (type === "INT") v = parseInt(raw, 10);
    else if (type === "FLOAT" || type === "number" || type === "slider") v = parseFloat(raw);
    else if (type === "BOOLEAN" || type === "toggle") v = !!raw;
    if (Number.isNaN(v) && (type === "INT" || type === "FLOAT")) return;
    for (const node of _targets) {
        const w = (node.widgets || []).find((x) => x.name === name && x.type === type);
        if (!w) continue;
        w.value = v;
        if (typeof w.callback === "function") {
            try { w.callback(v, app.canvas, node); } catch { /* */ }
        }
    }
    app.canvas?.setDirty(true, true);
}

function open() {
    _targets = selectedNodes();
    if (_targets.length < 2) {
        try { app.extensionManager?.toast?.add({ severity: "info", summary: "Bulk Edit", detail: "Select 2+ nodes first.", life: 2500 }); } catch { /* */ }
        return;
    }
    injectStyle();
    if (!_root) {
        _root = document.createElement("div");
        _root.id = ROOT_ID;
        _root.innerHTML = `
            <div class="hdr">
                <div class="ttl">Bulk Edit</div>
                <div class="sub" style="opacity:.7"></div>
                <div class="closex" title="Close (Esc)">✕</div>
            </div>
            <div class="body"></div>`;
        document.body.appendChild(_root);
        _hdr  = _root.querySelector(".hdr");
        _body = _root.querySelector(".body");
        _hdr.querySelector(".closex").addEventListener("click", close);
        window.addEventListener("keydown", (ev) => {
            if (ev.key === "Escape" && _root && _root.style.display !== "none") {
                ev.preventDefault(); close();
            }
        });
    }
    _root.style.display = "flex";
    const widgets = commonWidgets(_targets);
    const types = [...new Set(_targets.map((n) => n.type))];
    _hdr.querySelector(".sub").textContent = `${_targets.length} nodes · ${types.length === 1 ? types[0] : "mixed"}`;
    if (!widgets.length) {
        _body.innerHTML = `<div class="empty">No common widgets across selection.</div>`;
        return;
    }
    _body.innerHTML = widgets.map((w) => {
        const id = `bulk-${w.name.replace(/\W/g, "_")}`;
        const v0 = w.values[0];
        const placeholder = w.differs ? "…" : "";
        const badge = w.differs ? `<span class="badge">differs</span>` : "";
        let inputHtml = "";
        if (Array.isArray(w.refWidget?.options?.values)) {
            const opts = w.refWidget.options.values.map((o) => {
                const s = String(o);
                return `<option value="${s.replace(/"/g, "&quot;")}" ${s === String(v0) ? "selected" : ""}>${s}</option>`;
            }).join("");
            inputHtml = `<select id="${id}" data-name="${w.name}" data-type="${w.type}">${opts}</select>`;
        } else if (w.type === "BOOLEAN" || w.type === "toggle") {
            inputHtml = `<input id="${id}" type="checkbox" data-name="${w.name}" data-type="${w.type}" ${v0 ? "checked" : ""} />`;
        } else if (w.type === "INT" || w.type === "FLOAT" || w.type === "number" || w.type === "slider") {
            const step = w.type === "INT" ? "1" : "any";
            inputHtml = `<input id="${id}" type="number" step="${step}" placeholder="${placeholder}" value="${w.differs ? "" : (v0 ?? "")}" data-name="${w.name}" data-type="${w.type}" />`;
        } else {
            inputHtml = `<input id="${id}" type="text" placeholder="${placeholder}" value="${w.differs ? "" : (v0 ?? "")}" data-name="${w.name}" data-type="${w.type}" />`;
        }
        return `<div class="row ${w.differs ? "differs" : ""}">
                    <label title="${w.name}">${w.name}${badge}</label>
                    ${inputHtml}
                </div>`;
    }).join("");
    _body.querySelectorAll("input, select").forEach((el) => {
        const handler = () => {
            const v = el.type === "checkbox" ? el.checked : el.value;
            applyValue(el.dataset.name, el.dataset.type, v);
        };
        el.addEventListener("change", handler);
        if (el.tagName === "INPUT" && el.type !== "checkbox") {
            el.addEventListener("input", () => { /* live-update on input for text/number */
                if (el.value !== "") handler();
            });
        }
    });
}

function close() {
    if (_root) _root.style.display = "none";
}

function isEditingField() {
    const ae = document.activeElement;
    if (!ae) return false;
    if (_root && _root.contains(ae)) return false;  // our own field
    const tag = ae.tagName?.toLowerCase();
    return tag === "input" || tag === "textarea" || ae.isContentEditable;
}

function onKey(ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    if (ev.key !== "e" && ev.key !== "E") return;
    if (isEditingField()) return;
    try {
        if (app.ui?.settings?.getSettingValue?.(SETTING_ID, true) === false) return;
    } catch { /* */ }
    ev.preventDefault();
    ev.stopPropagation();
    open();
}

app.registerExtension({
    name: "C2C.BulkEdit",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Bulk Value Editor (Ctrl+Shift+E)",
                type: "boolean", defaultValue: true,
                category: ["c2c", "Editing", "Bulk Edit"],
            });
        } catch { /* */ }
        window.addEventListener("keydown", onKey, true);
        // Add right-click "Bulk-edit widgets…" via canvas menu hook.
        const origGetMenu = window.LGraphCanvas?.prototype?.getCanvasMenuOptions;
        if (origGetMenu && !window.LGraphCanvas.prototype._c2c_bulk_patched) {
            window.LGraphCanvas.prototype.getCanvasMenuOptions = function () {
                const opts = origGetMenu.apply(this, arguments) || [];
                const sel = Object.values(this.selected_nodes || {});
                if (sel.length >= 2) {
                    opts.unshift({
                        content: `Bulk-edit ${sel.length} nodes…`,
                        callback: open,
                    });
                }
                return opts;
            };
            window.LGraphCanvas.prototype._c2c_bulk_patched = true;
        }
        console.log("[C2C.BulkEdit] ready (Ctrl+Shift+E).");
    },
});
