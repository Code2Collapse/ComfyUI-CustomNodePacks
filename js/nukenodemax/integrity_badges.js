// FILE: web/extensions/nukenodemax/integrity_badges.js
// FEATURE: W2 — Conflict / integrity warning badges
// INTEGRATES WITH: nodes/integrity_guard.py (event "nukenodemax.integrity")

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const STATE = { events: [], pipOk: true, drift: 0 };

function refresh() {
    if (!app.graph) return;
    app.graph.setDirtyCanvas(true, true);
}

function onIntegrity(ev) {
    const d = ev.detail || ev;
    if (!d) return;
    if (d.type === "report") {
        STATE.events = d.events || [];
        STATE.pipOk = d.pip_check ? !!d.pip_check.ok : true;
        STATE.drift = (d.checksum_drift || []).length;
        refresh();
    }
}

async function reinstall(pkg) {
    if (!confirm(`Force-reinstall package "${pkg}"? This runs pip install --force-reinstall.`)) return;
    const r = await fetch(`/nukenodemax/reinstall?package=${encodeURIComponent(pkg)}&confirm=yes`,
        { method: "POST" });
    const j = await r.json();
    alert(j.ok ? `Reinstalled ${pkg}.` : `Failed: ${j.error || j.stderr || "unknown"}`);
}

app.registerExtension({
    name: "nukenodemax.integrity_badges",
    setup() {
        api.addEventListener("nukenodemax.integrity", onIntegrity);

        // Top-bar global badge.
        const bar = document.createElement("div");
        bar.id = "mec-integrity-bar";
        Object.assign(bar.style, {
            position: "fixed", right: "8px", top: "8px", zIndex: 9999,
            padding: "4px 8px", borderRadius: "4px",
            background: "#222", color: "#fff", font: "11px monospace",
            cursor: "pointer", display: "none",
        });
        document.body.appendChild(bar);
        bar.addEventListener("click", async () => {
            const r = await fetch("/nukenodemax/integrity_report");
            const j = await r.json();
            const lines = (j.events || []).map(e => `[${e.severity}] ${e.kind}: ${e.message}`);
            alert(lines.length ? lines.join("\n") : "No integrity events.");
        });

        // Refresh hook on dirty canvas.
        const origDraw = LGraphCanvas.prototype.drawNodeShape;
        LGraphCanvas.prototype.drawNodeShape = function (node, ctx, size, fg, bg, selected, mouseOver) {
            origDraw.apply(this, arguments);
            // Per-file checksum drift badge:
            const drifted = STATE.events.find(
                e => e.kind === "checksum_drift" && e.file && node.type &&
                     e.file.includes(node.type.toLowerCase())
            );
            if (drifted) {
                ctx.save();
                ctx.fillStyle = "#c8a200";
                ctx.beginPath();
                ctx.arc(8, -8, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = "#000";
                ctx.font = "bold 10px monospace";
                ctx.textBaseline = "middle";
                ctx.textAlign = "center";
                ctx.fillText("!", 8, -8);
                ctx.restore();
            }
        };

        // Periodic refresh of the global bar.
        setInterval(() => {
            const issues = STATE.events.length;
            if (issues || !STATE.pipOk || STATE.drift) {
                bar.style.display = "block";
                bar.style.background = STATE.pipOk ? "#a06000" : "#a02020";
                bar.textContent = `⚠ Integrity: ${issues} event${issues === 1 ? "" : "s"}` +
                                  (STATE.pipOk ? "" : " · pip conflicts") +
                                  (STATE.drift ? ` · ${STATE.drift} file drift` : "");
            } else {
                bar.style.display = "none";
            }
        }, 2000);

        // Right-click menu: offer reinstall for a known-bad package.
        const origMenu = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const opts = origMenu ? origMenu.apply(this, arguments) : [];
            const conflict = STATE.events.find(e => e.kind === "dependency_conflict");
            if (conflict) {
                const m = conflict.message.match(/^([A-Za-z0-9_.\-]+)\s+/);
                if (m) {
                    opts.unshift({
                        content: `Force-reinstall ${m[1]}`,
                        callback: () => reinstall(m[1]),
                    });
                    opts.unshift(null);
                }
            }
            return opts;
        };
    },
});
