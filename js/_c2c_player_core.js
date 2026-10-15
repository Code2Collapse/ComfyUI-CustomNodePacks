/**
 * Pure logic of the C2C video player (no DOM, no ComfyUI imports), so it runs under Node in the tests.
 *
 * Frames are DOMAIN frames: the whole clip after force_rate, numbered from 0. The node's selection is [in, out].
 */

export const SPEEDS = [0.25, 0.5, 1, 2, 4];
export const LOOP_MODES = ["selection", "clip", "once", "pingpong"];
export const LOOP_LABELS = { selection: "loop sel", clip: "loop all", once: "once", pingpong: "ping-pong" };

/** hh:mm:ss:ff from a frame number (ff counts frames inside the second). */
export function timecode(frame, fps) {
    if (!(fps > 0)) return "";
    const f = Math.max(0, Math.round(frame));
    const whole = Math.max(1, Math.round(fps));
    const ff = f % whole;
    const total = Math.floor(f / whole);
    const s = total % 60, m = Math.floor(total / 60) % 60, h = Math.floor(total / 3600);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(h)}:${p(m)}:${p(s)}:${p(ff)}`;
}

/** Seconds as "4.04 s" below a minute, "1:02.5" above. */
export function formatSeconds(sec) {
    if (!(sec >= 0) || !Number.isFinite(sec)) return "";
    if (sec < 60) return `${sec.toFixed(sec < 10 ? 2 : 1)} s`;
    const m = Math.floor(sec / 60);
    const s = sec - m * 60;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

/** Selection from the node's widgets -> [in, out] in domain frames. ``count`` = frames that will load. */
export function selectionFromWidgets(skip, cap, nth, domainFrames) {
    const n = Math.max(1, domainFrames | 0);
    const start = Math.min(n - 1, Math.max(0, skip | 0));
    const every = Math.max(1, nth | 0);
    const available = Math.floor((n - 1 - start) / every) + 1;
    const count = (cap | 0) > 0 ? Math.min(cap | 0, available) : available;
    return { in: start, out: start + (count - 1) * every, count };
}

/** [in, out] -> widget values. Selecting the whole clip writes cap 0 ("all"). */
export function widgetsFromSelection(inF, outF, nth, domainFrames) {
    const n = Math.max(1, domainFrames | 0);
    const a = Math.max(0, Math.min(n - 1, Math.round(Math.min(inF, outF))));
    const b = Math.max(a, Math.min(n - 1, Math.round(Math.max(inF, outF))));
    const every = Math.max(1, nth | 0);
    const cap = (a === 0 && b === n - 1) ? 0 : Math.floor((b - a) / every) + 1;
    return { skip: a, cap };
}

/** Next playback speed in ``SPEEDS`` (dir +1 faster, -1 slower), clamped at the ends. */
export function nextSpeed(current, dir) {
    let i = SPEEDS.indexOf(current);
    if (i < 0) i = SPEEDS.indexOf(1);
    return SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, i + (dir > 0 ? 1 : -1)))];
}

export function nextLoopMode(mode) {
    const i = LOOP_MODES.indexOf(mode);
    return LOOP_MODES[(i + 1) % LOOP_MODES.length];
}

/** Shuttle (J / K / L, as in editing programs): L plays forward, again = faster; J the same backwards; K stops.
 *  ``rate`` is signed (negative = backwards), 0 = stopped. Returns the new signed rate. */
export function shuttle(rate, key) {
    const k = String(key).toLowerCase();
    if (k === "k") return 0;
    const sign = k === "l" ? 1 : k === "j" ? -1 : 0;
    if (!sign) return rate;
    if (Math.sign(rate) !== sign) return sign;           // start, or reverse direction at 1x
    return sign * Math.min(4, Math.abs(rate) * 2);       // 1 -> 2 -> 4
}

/**
 * What playback does when it reaches ``frame`` moving in direction ``dir`` (+1 / -1).
 * Bounds: the selection for "selection" and "pingpong", the whole clip for "clip" and "once".
 * Returns { frame, dir, stop }: the frame to show, the direction to continue in, and whether to stop.
 */
export function loopStep({ frame, dir, mode, inF, outF, frames }) {
    const last = Math.max(0, (frames | 0) - 1);
    const lo = (mode === "selection" || mode === "pingpong") ? Math.max(0, inF | 0) : 0;
    const hi = (mode === "selection" || mode === "pingpong") ? Math.min(last, Math.max(lo, outF | 0)) : last;
    const d = dir < 0 ? -1 : 1;
    if (frame >= lo && frame <= hi) return { frame, dir: d, stop: false };
    const past = d > 0 ? frame > hi : frame < lo;
    if (!past) return { frame: Math.max(lo, Math.min(hi, frame)), dir: d, stop: false };
    if (mode === "once") return { frame: d > 0 ? hi : lo, dir: d, stop: true };
    if (mode === "pingpong") return { frame: d > 0 ? hi : lo, dir: -d, stop: false };
    return { frame: d > 0 ? lo : hi, dir: d, stop: false };   // selection / clip: wrap around
}

/**
 * Ruler ticks for a scrubber ``widthPx`` wide over ``frames`` frames. unit "frames" or "time".
 * Major ticks carry a label and sit at least ~64 px apart; minor ticks split each major step in 2, 4 or 5.
 * Returns [{ frame, major, label }].
 */
export function rulerTicks(frames, fps, widthPx, unit = "frames") {
    const n = Math.max(1, frames | 0);
    const w = Math.max(1, widthPx);
    if (n < 2) return [{ frame: 0, major: true, label: unit === "time" ? "0 s" : "0" }];
    const pxPerFrame = w / (n - 1);
    const minMajorFrames = 64 / pxPerFrame;
    let step;
    if (unit === "time" && fps > 0) {
        const secSteps = [0.04, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
        const sec = secSteps.find((s) => s * fps >= minMajorFrames) ?? secSteps[secSteps.length - 1];
        step = sec * fps;
    } else {
        const base = [1, 2, 5];
        step = 1;
        for (let mag = 1; mag < 1e9 && step < minMajorFrames; mag *= 10) {
            for (const b of base) { step = b * mag; if (step >= minMajorFrames) break; }
        }
    }
    const minorDiv = pxPerFrame * step >= 160 ? 4 : 2;
    const out = [];
    const minor = step / minorDiv;
    for (let i = 0, f = 0; f <= n - 1 + 1e-6 && i < 10000; i += 1, f = i * minor) {
        const major = i % minorDiv === 0;
        const fr = Math.round(f * 1000) / 1000;
        let label = "";
        if (major) {
            if (unit === "time" && fps > 0) {
                const s = fr / fps;
                label = s >= 60 ? formatSeconds(s) : `${Math.round(s * 100) / 100} s`;
            } else {
                label = String(Math.round(fr));
            }
        }
        out.push({ frame: fr, major, label });
    }
    return out;
}

/** Parse what the user typed into the frame box: a frame number, or a timecode hh:mm:ss:ff / mm:ss:ff / ss:ff. */
export function parseFrameInput(text, fps) {
    const t = String(text ?? "").trim();
    if (!t) return null;
    if (/^\d+$/.test(t)) return parseInt(t, 10);
    if (/^[\d:]+$/.test(t) && fps > 0) {
        const parts = t.split(":").map((x) => parseInt(x || "0", 10));
        if (parts.some((x) => !Number.isFinite(x))) return null;
        const ff = parts.pop();
        let sec = 0;
        for (const p of parts) sec = sec * 60 + p;
        return Math.round(sec * Math.max(1, Math.round(fps)) + ff);
    }
    return null;
}

// ── crop tool (L2.53): rectangles are FRACTIONS of the frame {x, y, w, h}, as the node's crop widgets hold them ──

export const CROP_PRESETS = ["Free", "Original", "1:1", "4:5", "5:4", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2"];
export const FULL_RECT = Object.freeze({ x: 0, y: 0, w: 1, h: 1 });
const MIN_FRAC = 0.02;

/** Pixel aspect (width / height) a preset asks for; null = free. "Original" = the frame's own aspect. */
export function presetRatio(preset, frameW, frameH) {
    if (!preset || preset === "Free") return null;
    if (preset === "Original") return frameW > 0 && frameH > 0 ? frameW / frameH : null;
    const m = /^(\d+):(\d+)$/.exec(preset);
    return m ? Number(m[1]) / Number(m[2]) : null;
}

/** Python's round() (half to even), so the label shows the exact pixels the run crops. */
export function roundHalfEven(v) {
    const r = Math.round(v);
    return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** The crop in pixels exactly as nodes/c2c_video/handle.py C2CVideo.cropped() computes it. */
export function cropPixels(rect, W, H) {
    const fx = Math.min(Math.max(rect.x, 0), 1), fy = Math.min(Math.max(rect.y, 0), 1);
    const fw = Math.min(Math.max(rect.w, 0), 1 - fx), fh = Math.min(Math.max(rect.h, 0), 1 - fy);
    const px = roundHalfEven(fx * W), py = roundHalfEven(fy * H);
    const pw = Math.max(1, Math.min(W - px, roundHalfEven(fw * W)));
    const ph = Math.max(1, Math.min(H - py, roundHalfEven(fh * H)));
    const full = (px === 0 && py === 0 && pw === W && ph === H) || fw <= 0 || fh <= 0;
    return { x: full ? 0 : px, y: full ? 0 : py, w: full ? W : pw, h: full ? H : ph, full };
}

export function isFullRect(r) {
    return !r || (r.x <= 1e-6 && r.y <= 1e-6 && r.w >= 1 - 1e-6 && r.h >= 1 - 1e-6);
}

/** The largest rectangle of pixel aspect ``ratio`` centred on ``around`` (default the frame centre). */
export function fitAspect(ratio, frameW, frameH, around = null) {
    if (!(ratio > 0) || !(frameW > 0) || !(frameH > 0)) return { ...FULL_RECT };
    const frameRatio = frameW / frameH;
    // in fractions: w_frac * frameW / (h_frac * frameH) = ratio
    let w = 1, h = frameRatio / ratio;
    if (h > 1) { h = 1; w = ratio / frameRatio; }
    const cx = around ? around.x + around.w / 2 : 0.5, cy = around ? around.y + around.h / 2 : 0.5;
    const x = Math.min(Math.max(cx - w / 2, 0), 1 - w), y = Math.min(Math.max(cy - h / 2, 0), 1 - h);
    return { x, y, w, h };
}

/**
 * Drag a crop rectangle. ``part``: "move", or the edges it holds: "n", "s", "e", "w", "ne", "nw", "se", "sw".
 * (dx, dy) in frame fractions. ``ratio`` (pixel aspect) locks the shape; frameW / frameH convert it to fractions.
 * The result stays inside the frame and at least MIN_FRAC wide and high.
 */
export function dragRect(start, part, dx, dy, ratio = null, frameW = 1, frameH = 1) {
    let { x, y, w, h } = start;
    if (part === "move") {
        x = Math.min(Math.max(x + dx, 0), 1 - w);
        y = Math.min(Math.max(y + dy, 0), 1 - h);
        return { x, y, w, h };
    }
    let l = x, t = y, r = x + w, b = y + h;
    if (part.includes("w")) l = Math.min(Math.max(l + dx, 0), r - MIN_FRAC);
    if (part.includes("e")) r = Math.max(Math.min(r + dx, 1), l + MIN_FRAC);
    if (part.includes("n")) t = Math.min(Math.max(t + dy, 0), b - MIN_FRAC);
    if (part.includes("s")) b = Math.max(Math.min(b + dy, 1), t + MIN_FRAC);
    if (ratio > 0 && frameW > 0 && frameH > 0) {
        const k = ratio * frameH / frameW;               // w_frac / h_frac for this pixel aspect
        let nw = r - l, nh = b - t;
        const horizontal = part === "e" || part === "w";
        const vertical = part === "n" || part === "s";
        if (horizontal) nh = nw / k;
        else if (vertical) nw = nh * k;
        else if (nw / nh > k) nw = nh * k; else nh = nw / k;
        // anchor: the edges not being dragged stay put; single edges grow symmetrically on the other axis
        const cx = (l + r) / 2, cy = (t + b) / 2;
        if (horizontal) { t = cy - nh / 2; b = cy + nh / 2; }
        else if (vertical) { l = cx - nw / 2; r = cx + nw / 2; }
        else {
            if (part.includes("w")) l = r - nw; else r = l + nw;
            if (part.includes("n")) t = b - nh; else b = t + nh;
        }
        // keep it inside the frame without breaking the ratio: shrink about the anchor, then shift
        const over = Math.max((r - l) > 1 ? (r - l) : 0, (b - t) * k > 1 ? (b - t) * k : 0);
        if (over > 1) { const s = 1 / over; const cw = (r - l) * s, ch = (b - t) * s; r = l + cw; b = t + ch; }
        if (l < 0) { r -= l; l = 0; }
        if (t < 0) { b -= t; t = 0; }
        if (r > 1) { l -= r - 1; r = 1; }
        if (b > 1) { t -= b - 1; b = 1; }
        l = Math.max(0, l); t = Math.max(0, t);
    }
    return { x: l, y: t, w: r - l, h: b - t };
}
