// L7.65 P01 (= L7.54): the Image Mask Editor carries the former editors' controls under names prefixed with their mode
// (nodes/image_mask_editor/modes.py). An editor installed on it (points, spline / roto, video, paint canvas) looks its
// widgets up through this map, by the node's current mode; on every other node a name is itself.

export const IME = "ImageMaskEditorC2C";

const FLOW = ["pattern", "thickness", "amplitude", "frequency", "turbulence", "turbulence_scale", "edge_softness",
              "taper_start", "taper_end", "frames", "animation_speed", "flow_direction", "mod_decay", "seed",
              "use_embedded_editor"];

const MAP = {
    "points + box": {
        width: "points_width", height: "points_height", editor_data: "points_editor_data",
        default_radius: "points_default_radius", softness: "points_softness", normalize: "points_normalize",
        reference_image: "image",
    },
    spline: {
        closed: "spline_closed", samples_per_segment: "spline_samples_per_segment", feather_radius: "spline_feather_radius",
        invert: "spline_invert", smoothing: "spline_smoothing", centripetal_alpha: "spline_centripetal_alpha",
        width: "spline_width", height: "spline_height", mask_color: "spline_mask_color",
        mask_opacity: "spline_mask_opacity",
        ...Object.fromEntries(FLOW.map((n) => [n, `spline_flow_${n}`])),
    },
    roto: {
        frame_count: "roto_frame_count", width: "roto_width", height: "roto_height",
        samples_per_seg: "roto_samples_per_seg", feather_px: "roto_feather_px",
    },
    video: {
        session_id: "video_session_id", tween_mode: "video_tween_mode", feather: "video_feather",
        threshold: "video_threshold", outside_keys: "video_outside_keys",
    },
    paint: {
        brush_type: "canvas_brush_type", brush_color: "canvas_brush_color", brush_opacity: "canvas_brush_opacity",
        brush_hardness: "canvas_brush_hardness", brush_size: "canvas_brush_size", mask_hardness: "canvas_mask_hardness",
        mask_expansion: "canvas_mask_expansion", mask_blur_radius: "canvas_mask_blur_radius",
        mask_blur_strength: "canvas_mask_blur_strength", reference_image: "image",
    },
};

// Other nodes that host a former editor under other names. The spline tracker UI (js/spline_mask_tracker.js) writes its
// keyframes to "keyframes_json", which only the old standalone tracker had: on Spline Mask (track) and Mask Track
// (spline_track) the keyframes live in spline_data - before this map they were never saved there (L7.65 P01).
const TYPE_MAP = {
    SplineMaskMEC: { keyframes_json: "spline_data" },
    MaskTrackerMEC: {
        keyframes_json: "spline_data", closed: "spline_closed", samples_per_segment: "spline_samples_per_segment",
        tracking_weight: "spline_tracking_weight", klt_window: "spline_klt_window",
        feather_radius: "spline_feather_radius", stroke_width: "spline_stroke_width",
    },
};

export const isIME = (node) => node?.comfyClass === IME || node?.type === IME;

export function imeMode(node) {
    const w = node?.widgets?.find?.((x) => x && x.name === "mode");
    return String(w?.value ?? "paint");
}

/** The widget name a former editor's `name` has on this node (in `mode`, default: the node's current mode). */
export function widgetName(node, name, mode) {
    if (!isIME(node)) return TYPE_MAP[node?.comfyClass ?? node?.type]?.[name] ?? name;
    return MAP[mode ?? imeMode(node)]?.[name] ?? name;
}

/** node.widgets.find by the former editor's name. */
export function findWidget(node, name, mode) {
    const n = widgetName(node, name, mode);
    return node?.widgets?.find?.((w) => w && w.name === n) ?? null;
}

/** The input slot a former editor calls `name` (reference_image is the editor's `image`). */
export function inputName(node, name, mode) {
    return widgetName(node, name, mode);
}
