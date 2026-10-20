/**
 * Pure logic of the Load Video (C2C) settings UX (L2.53, owner A14: "Widget UX" from ComfyUI-4K-Video-Save-Load).
 * No DOM, no ComfyUI imports, so it runs under Node in the tests (tests/test_loader_hints.py).
 *
 * The numbers shown are the SERVER's: the probe route builds the same handle a run builds and reports what loads
 * (frames, seconds, size, fps, decoder). Nothing here re-implements the selection maths, so the grey hints, the
 * player's "N will load" label and the run cannot disagree.
 *
 * Ideas PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 web/js/video4k.js (MIT, Copyright (c) 2026
 * Matthew Hallett): the grey "real value <-" beside a widget, the reset arrow, reset-all keeping the clip, the
 * details line ("loads 1.5s to 4.2s (2.7s, 65 frames)"), the frame-count grid stepping. Written anew for our probe.
 */

// Mirrors FORMAT_PRESETS in nodes/c2c_video/nodes_load.py (tests/test_loader_hints.py compares the two).
export const FORMAT_GRID = {
    "None": null,
    "AnimateDiff": null,
    "Mochi": [6, 1],
    "LTXV": [8, 1],
    "Hunyuan": [4, 1],
    "Cosmos": [8, 1],
    "Wan": [4, 1],
    "MiniMax H3": [17, 5],
};
export const FORMAT_RATE = {
    "AnimateDiff": 8, "Mochi": 24, "LTXV": 24, "Hunyuan": 24, "Cosmos": 24, "Wan": 16, "MiniMax H3": 24,
};

/** The settings "reset all to defaults" restores; the clip itself (video / directory) is kept. */
export const SETTINGS = [
    "force_rate", "custom_width", "custom_height", "frame_load_cap", "skip_first_frames", "select_every_nth",
    "format", "sequence_fps", "crop_x", "crop_y", "crop_w", "crop_h", "decoder", "seconds_cap", "resolution",
    "divisible_by", "audio_when_missing",
];

/** Widgets that get the reset arrow, and what they go back to. */
export const RESETTABLE = {
    force_rate: 0, custom_width: 0, custom_height: 0, frame_load_cap: 0, skip_first_frames: 0, select_every_nth: 1,
    seconds_cap: 0,
};

export function tenths(seconds) {
    return Math.floor(Number(seconds) * 10 + 1e-6) / 10;
}

/** Largest count on the grid (div, mod) that is <= n, or 0 when none is. */
export function gridFloor(n, grid) {
    if (!grid) return Math.max(0, Math.floor(n));
    const [d, m] = grid;
    return n < m ? 0 : Math.floor((n - m) / d) * d + m;
}

/** Smallest count on the grid that is >= n (n >= 1). */
export function gridCeil(n, grid) {
    if (!grid) return Math.max(0, Math.ceil(n));
    const [d, m] = grid;
    return n <= m ? m : Math.ceil((n - m) / d) * d + m;
}

/**
 * The value frame_load_cap should hold after the user changed it from ``prev`` to ``value`` under a format grid.
 * 0 stays 0 ("all"). An arrow step (exactly one grid step, or one frame) moves to the next legal count in that
 * direction; a typed value snaps to the nearest legal count (ties go down: never ask for frames that may not exist).
 */
export function snapCap(value, prev, grid) {
    const v = Math.max(0, Math.round(Number(value) || 0));
    if (!grid || v === 0) return v;
    const [d] = grid;
    const p = Math.max(0, Math.round(Number(prev) || 0));
    const delta = v - p;
    if (delta === d || delta === 1) return gridCeil(v, grid);
    if (delta === -d || delta === -1) return gridFloor(v, grid);
    const lo = gridFloor(v, grid), hi = gridCeil(v, grid);
    if (!lo) return hi;
    return (v - lo) <= (hi - v) ? lo : hi;
}

/**
 * Arrow step (widget option step2) for frame_load_cap: always 1. The frontend's INT handler rounds EVERY value, typed
 * ones too, to a multiple of step2 before our callback runs (frontend 1.53.10 onValueChange), so a grid step of 17
 * turned a typed 60 into 68 and then 73. With 1, snapCap sees a one-frame arrow step and moves to the next legal
 * count (H3: 0 -> 5 -> 22 -> 39), and a typed value snaps to the nearest legal count.
 */
export function capStep(_format) {
    return 1;
}

function sameNumber(a, b, tol = 1e-6) {
    return Math.abs(Number(a) - Number(b)) <= tol;
}

/**
 * Grey hints: for each widget whose own value is not what will load, what will load.
 * ``values``: the node's widget values by name. ``load``: the probe's ``load`` block (null until known).
 * Returns { name: { shown, differs, canReset } } for the widgets that have a hint or a reset arrow.
 */
export function hints(values, load) {
    const v = (k, d = 0) => (values[k] ?? d);
    const out = {};
    const add = (name, shown, differs) => {
        if (!(name in values)) return;
        const def = RESETTABLE[name];
        const canReset = name === "custom_width" || name === "custom_height"
            ? (Number(v("custom_width")) !== 0 || Number(v("custom_height")) !== 0 || (v("resolution", "custom") !== "custom"))
            : def !== undefined && !sameNumber(v(name, def), def);
        out[name] = { shown, differs: !!differs && shown !== null && shown !== undefined, canReset };
    };
    if (load) {
        add("frame_load_cap", load.frames, !sameNumber(load.frames, v("frame_load_cap")));
        const secs = tenths(load.duration);
        add("seconds_cap", secs.toFixed(1), !sameNumber(secs, v("seconds_cap"), 0.05));
        add("custom_width", load.width, !sameNumber(load.width, v("custom_width")));
        add("custom_height", load.height, !sameNumber(load.height, v("custom_height")));
        const fps = Math.round(Number(load.fps) * 100) / 100;
        add("force_rate", fps, !sameNumber(fps, v("force_rate"), 0.005));
    } else {
        for (const n of ["frame_load_cap", "seconds_cap", "custom_width", "custom_height", "force_rate"]) add(n, null, false);
    }
    add("skip_first_frames", null, false);
    add("select_every_nth", null, false);
    return out;
}

/** The size widgets after the user typed into one of them (4K pack): leave the preset, the other follows aspect. */
export function sizeEdit(edited, values) {
    if ((values.resolution ?? "custom") === "custom") return null;
    return { resolution: "custom", [edited === "custom_width" ? "custom_height" : "custom_width"]: 0 };
}

/** frames <-> seconds: typing one non-zero cap hands the other back to the clip (it then shows its real value). */
export function capEdit(edited, value) {
    if (!(Number(value) > 0)) return null;
    return edited === "frame_load_cap" ? { seconds_cap: 0 } : edited === "seconds_cap" ? { frame_load_cap: 0 } : null;
}

function fmtS(s) {
    return `${tenths(s).toFixed(1)}s`;
}

/** The file's details and the part that will load, from the probe payload. Returns lines of text. */
export function describe(probe) {
    if (!probe) return [];
    const lines = [];
    const fps = Number(probe.fps) || 0;
    const head = [probe.file || "", probe.width && probe.height ? `${probe.width}x${probe.height}` : "",
        fps ? `${Math.round(fps * 1000) / 1000} fps` : "", probe.duration ? fmtS(probe.duration) : "",
        probe.frame_count ? `${probe.frame_count} frames` : "", probe.audio ? "audio" : "no audio"];
    lines.push(head.filter(Boolean).join("  "));
    const L = probe.load;
    if (L) {
        let line = `loads ${fmtS(L.start_seconds)} to ${fmtS(L.end_seconds)} (${fmtS(L.duration)}, ${L.frames} frames)`;
        line += ` at ${L.width}x${L.height}, ${Math.round(Number(L.fps) * 100) / 100} fps`;
        lines.push(line);
        const dec = L.decoder;
        if (dec) lines.push(`decoder: ${dec.used}${dec.note ? ` (${dec.note})` : ""}`);
        if (L.audio === "silence" && !probe.audio) lines.push("no audio track: a silent one is output");
    }
    return lines;
}

/** A note when a format preset expects another frame rate than the one that loads (the rate is not forced). */
export function rateNote(format, load, nth = 1) {
    const want = FORMAT_RATE[format];
    if (!want || !load) return "";
    const rate = Number(load.fps) * Math.max(1, Number(nth) || 1);
    if (Math.abs(rate - want) < 0.01) return "";
    return `${format} expects ${want} fps; this loads ${Math.round(rate * 1000) / 1000} fps (set force_rate to ${want})`;
}
