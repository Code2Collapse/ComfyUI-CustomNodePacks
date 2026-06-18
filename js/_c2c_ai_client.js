/**
 * _c2c_ai_client.js — Single SSE streaming client for /c2c/ai/stream.
 *
 * Replaces ~40 lines of duplicated reader/decoder/event-parser boilerplate
 * that was copy-pasted across:
 *   - c2c_ai_error_translator.js
 *   - c2c_ai_explainer.js
 *   - c2c_workflow_doctor.js
 *   - c2c_prompt_wizard.js
 *
 * Usage:
 *   import { streamAI } from "./_c2c_ai_client.js";
 *   const { ok, text } = await streamAI({
 *       feature: "node_explainer",
 *       sensitivity: "normal",
 *       max_tokens: 600,
 *       temperature: 0.3,
 *       messages: [{role:"system", content:"..."}, {role:"user", content:"..."}],
 *       onStatus: (s) => statusEl.textContent = s,     // "connecting"|"streaming"|"done"|"error"
 *       onChunk:  (chunk, total) => { body.textContent = total; },
 *       onError:  (err) => { body.textContent = "Error: " + err.message; },
 *       onDone:   (total) => { ...finalise... },
 *       signal:   abortController.signal,              // optional
 *   });
 *
 * Always calls `window.__C2C_AI_HUD__?.refresh?.()` after completion so the
 * status bar cost pill updates.
 *
 * License: Apache-2.0 (same as the rest of CustomNodePacks).
 */
import { reportFailure as __c2cReport } from "./_c2c_report.js";


const STREAM_URL = "/c2c/ai/stream";

/**
 * Stream a chat completion from /c2c/ai/stream and pipe deltas to callbacks.
 *
 * Returns { ok: boolean, text: string, status?: number } where `text` is the
 * concatenated chunk content. On HTTP/network failure, `onError` is invoked
 * once and the promise still resolves (it does NOT throw).
 */
export async function streamAI(opts) {
    const {
        feature, sensitivity = "normal",
        max_tokens = 600, temperature = 0.3,
        messages,
        onStatus, onChunk, onError, onDone,
        signal,
    } = opts || {};

    if (!feature || !Array.isArray(messages) || messages.length === 0) {
        const err = { kind: "bad-request", message: "feature + messages required" };
        onError?.(err);
        return { ok: false };
    }

    onStatus?.("connecting");
    let r;
    try {
        r = await fetch(STREAM_URL, {
            method: "POST",
            signal,
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            body: JSON.stringify({
                feature, sensitivity, max_tokens, temperature, messages,
            }),
        });
    } catch (e) {
        onError?.({ kind: "network", message: e?.message || String(e) });
        _hudRefresh();
        return { ok: false };
    }

    if (!r.ok || !r.body) {
        const j = await r.json().catch(() => ({}));
        onError?.({
            kind: "http",
            status: r.status,
            message: j.message || ("HTTP " + r.status),
        });
        _hudRefresh();
        return { ok: false, status: r.status };
    }

    onStatus?.("streaming");
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let total = "";
    let sawError = false;

    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });

            let idx;
            while ((idx = buf.indexOf("\n\n")) !== -1) {
                const frame = buf.slice(0, idx);
                buf = buf.slice(idx + 2);

                let event = "message";
                let data = "";
                for (const line of frame.split("\n")) {
                    if (line.startsWith("event: ")) event = line.slice(7).trim();
                    else if (line.startsWith("data: ")) data += line.slice(6);
                }
                if (!data) continue;

                if (event === "error") {
                    sawError = true;
                    onError?.({ kind: "stream", message: data });
                } else if (event === "done") {
                    onStatus?.("done");
                } else {
                    try {
                        const obj = JSON.parse(data);
                        if (obj && typeof obj.chunk === "string" && obj.chunk.length) {
                            total += obj.chunk;
                            onChunk?.(obj.chunk, total);
                        }
                    } catch (__c2cErr) { __c2cReport("_c2c_ai_client", __c2cErr); }
                }
            }
        }
    } catch (e) {
        onError?.({ kind: "stream-abort", message: e?.message || String(e) });
        _hudRefresh();
        return { ok: false, text: total };
    }

    _hudRefresh();
    onDone?.(total);
    return { ok: !sawError, text: total };
}

function _hudRefresh() {
    try { window.__C2C_AI_HUD__?.refresh?.(); } catch (__c2cErr) { __c2cReport("_c2c_ai_client", __c2cErr); }
}
