/**
 * _fi_source_name.js — how FolderIncrementer reads a source name out of ANY
 * loader, whatever node it is.
 *
 * The old rule looked for a fixed list of widget names and kept a value only
 * if it contained a dot. Folder loaders (Load Images From Dir, VHS Load Images
 * (Path), WAS Load Image Batch, KJ Load Images From Folder, EXR sequence
 * readers) hold a DIRECTORY - no dot - so they were skipped and the name came
 * from whichever other loader was nearby, or from nowhere.
 *
 * Now a value is judged by what it IS, not by the node or the widget:
 *   D:/shots/sh010_plate/            -> sh010_plate       (folder)
 *   D:/shots/sh010_plate/*.exr       -> sh010_plate       (glob: the folder)
 *   sh010_plate.1001.exr             -> sh010_plate.1001.exr  (Python strips the frame)
 *   D:/shots/sh010_plate/0001.exr    -> sh010_plate.exr   (bare frame number: the folder)
 *   D:/shots/sh010/exr/              -> sh010             (generic folder: its parent)
 *   clip.mov [input]                 -> clip.mov          (ComfyUI annotation dropped)
 *
 * Pure: no imports, no DOM. FolderIncrementer's Python mirrors these rules
 * (folder_incrementer._source_from_value) and tests/test_folder_incrementer_sources.py
 * runs one fixture table through BOTH, so they cannot drift apart.
 */

export const VIDEO_EXTS = ["mp4", "mov", "webm", "mkv", "avi", "m4v", "flv", "wmv", "mpeg", "mpg",
    "ts", "mts", "m2ts", "mxf", "r3d", "braw", "ari", "3gp", "ogv", "y4m", "dv", "vob", "f4v"];
export const IMAGE_EXTS = ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "tga", "exr",
    "dpx", "cin", "hdr", "heic", "avif", "jp2", "j2k", "jxl", "psd", "sgi", "iff", "pic", "pfm",
    "ppm", "dng", "cr2", "nef", "arw"];
export const AUDIO_EXTS = ["wav", "mp3", "aac", "flac", "ogg", "m4a", "opus", "aiff"];

/** Stills that a numbered frame sequence is made of. */
export const SEQ_EXTS = ["exr", "dpx", "cin", "tif", "tiff", "tga", "png", "jpg", "jpeg", "hdr",
    "jp2", "j2k", "sgi", "pic", "iff", "dng", "webp", "bmp"];

const ALL_EXTS = [...VIDEO_EXTS, ...IMAGE_EXTS, ...AUDIO_EXTS, "pdf", "zip"];
export const KNOWN_EXT_RE = new RegExp(`\\.(${ALL_EXTS.join("|")})$`, "i");
const SEQ_EXT_RE = new RegExp(`^\\.(${SEQ_EXTS.join("|")})$`, "i");

/** Explicit frame syntax at the end of a stem - never part of a real name:
 *  .1001 is NOT here (a bare number can be a shot number; Python decides by
 *  width), but ####, @@@@, %04d, $F4 and [1001-1100] always are. */
export const FRAME_SYNTAX_RE = /[._-]?(#{1,8}|@{1,8}|%0?\d*d|\$F\d*|\[\d+[-:]\d+\])$/i;

/** Widgets that name a FILE. Order = priority. */
export const FILE_WIDGETS = [
    "image", "video", "filename", "file", "audio", "url",
    "source", "file_path", "filepath", "image_path", "video_path", "exr_path", "exr",
    "uploaded_file", "media", "media_path", "clip", "plate", "movie",
];
/** Widgets that name a FOLDER (or a file/folder/pattern - judged by value). */
export const FOLDER_WIDGETS = [
    "directory", "folder", "dir", "folder_path", "dir_path", "input_folder", "input_dir",
    "image_folder", "images_folder", "images_path", "image_dir", "frames_path", "frames_dir",
    "sequence_path", "sequence_folder", "sequence", "exr_sequence", "exr_folder", "exr_dir",
    "load_path", "input_path", "path", "pattern",
];

/** Container folders that say nothing about the shot: take their parent. */
const GENERIC_DIRS = new Set([
    "exr", "exrs", "dpx", "png", "pngs", "jpg", "jpgs", "jpeg", "tif", "tiff", "tga", "cin",
    "frames", "frame", "images", "image", "imgs", "img", "seq", "seqs", "sequence", "sequences",
    "render", "renders", "plate", "plates", "input", "inputs", "output", "outputs", "src",
    "source", "sources", "media", "footage", "full", "fullres", "proxy", "proxies", "hires",
    "lores", "uhd", "hd", "sd", "linear", "acescg", "aces", "srgb", "rec709", "log", "raw",
    "beauty", "rgba", "rgb", "main", "data", "temp", "tmp", "cache", "comfyui", "video",
    "videos", "movies", "clips",
]);
const VERSION_DIR_RE = /^(v|ver|version|rev)\d+$/i;
const RES_DIR_RE = /^(\d{3,5}x\d{3,5}|\d{3,4}[pk]|[1-8]k)$/i;
const DRIVE_RE = /^[A-Za-z]:$/;

function extOf(base) {
    const m = String(base).match(KNOWN_EXT_RE);
    return m ? m[0] : "";
}

/** The last folder in `segs` that actually names something. */
export function meaningfulDir(segs) {
    const real = segs.filter((s) => s && !DRIVE_RE.test(s) && s !== "~" && s !== "." && s !== "..");
    for (let i = real.length - 1; i >= 0; i--) {
        const s = real[i];
        if (GENERIC_DIRS.has(s.toLowerCase()) || VERSION_DIR_RE.test(s) || RES_DIR_RE.test(s)) continue;
        return s;
    }
    return real.length ? real[real.length - 1] : "";
}

/** Does this string look like a file or folder path at all (for widgets whose
 *  NAME says nothing - a generic STRING "value", a custom loader's field)? */
export function looksLikePath(v) {
    // length first: never copy a multi-MB widget value (base64 image data) to trim it
    if (typeof v !== "string" || v.length > 4096) return false;
    const s = v.trim();
    if (!s || s.length > 2048 || /[\r\n]/.test(s)) return false;
    if (/^([A-Za-z]:[\\/]|\\\\|\/|~[\\/])/.test(s)) return true;           // absolute path
    const noAnn = s.replace(/\s*\[(input|output|temp)\]$/i, "");
    return KNOWN_EXT_RE.test(noAnn) && !/\s{2,}/.test(noAnn);              // a media file name
}

/**
 * Read one widget value. Returns { filename, kind } or null.
 *   filename - what FolderIncrementer writes to source_filename (+extension);
 *              a folder yields its name with no extension
 *   kind     - "video" | "image" | "audio" | "sequence" (numbered frames) | "folder"
 * `folderHint` = the widget is a folder widget, so a bare word ("sh010", the
 * value of VHS Load Images' directory combo) is a folder name, not noise.
 */
export function sourceFromValue(value, folderHint = false) {
    if (typeof value !== "string" || value.length > 4096) return null;
    let s = value.trim();
    if (!s || s.length > 2048 || /[\r\n]/.test(s)) return null;
    s = s.replace(/\s*\[(input|output|temp)\]$/i, "");
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) s = s.replace(/[?#].*$/, "");  // URL: drop the query
    s = s.replace(/\\/g, "/").replace(/\/+$/, "");
    const segs = s.split("/").filter(Boolean);
    if (!segs.length) return null;
    const base = segs[segs.length - 1];
    const parents = segs.slice(0, -1);
    const ext = extOf(base);

    // A glob / pattern: the folder is the name ("D:/sh010/*.exr", "*.png").
    if (/[*?]/.test(base) || /\{[^}]*\}/.test(base)) {
        const dir = meaningfulDir(parents);
        return dir ? { filename: dir + ext, kind: "sequence" } : null;
    }

    if (ext) {
        let stem = base.slice(0, -ext.length);
        const isSeqExt = SEQ_EXT_RE.test(ext);
        if (isSeqExt) {
            const bare = stem.replace(FRAME_SYNTAX_RE, "");
            // "0001.exr", "####.exr", ".exr": the frame IS the file name, so the
            // folder is the name of the sequence
            if (!bare || /^\d+$/.test(bare)) {
                const dir = meaningfulDir(parents);
                return dir ? { filename: dir + ext, kind: "sequence" } : null;
            }
            const numbered = bare !== stem || /[._-]\d{2,8}$/.test(stem);
            if (bare !== stem) stem = bare;          // explicit syntax always goes
            const kindOf = numbered ? "sequence" : "image";
            return { filename: stem + ext, kind: kindOf };
        }
        const lower = ext.slice(1).toLowerCase();
        const kind = VIDEO_EXTS.includes(lower) ? "video"
            : AUDIO_EXTS.includes(lower) ? "audio"
            : IMAGE_EXTS.includes(lower) ? "image" : "file";
        return { filename: base, kind };
    }

    // No extension: a folder - when the value is a path, or the widget says so.
    if (parents.length || folderHint || /^[A-Za-z]:/.test(s)) {
        const dir = meaningfulDir(segs);
        return dir ? { filename: dir, kind: "folder" } : null;
    }
    return null;
}

const SKIP_WIDGETS = new Set([
    "filename_prefix", "output_path", "save_path", "output_dir", "out_path", "base_path",
    "custom_name", "source_filename", "source_extension", "folder_name_override", "suffix",
    "text", "prompt", "positive", "negative", "string", "label", "title",
]);

/**
 * The source a node reads, judged from ALL its widgets: named file widgets
 * first, then folder widgets, then any other widget whose value is a path.
 * Returns { filename, kind, widget } or null.
 */
export function sourceFromWidgets(widgets) {
    if (!Array.isArray(widgets) || !widgets.length) return null;
    const byName = new Map();
    for (const w of widgets) if (w && typeof w.name === "string" && !byName.has(w.name)) byName.set(w.name, w);
    for (const name of FILE_WIDGETS) {
        const w = byName.get(name);
        const r = w && sourceFromValue(w.value, false);
        if (r) return { ...r, widget: name };
    }
    for (const name of FOLDER_WIDGETS) {
        const w = byName.get(name);
        const r = w && sourceFromValue(w.value, true);
        if (r) return { ...r, widget: name };
    }
    for (const w of widgets) {
        if (!w || SKIP_WIDGETS.has(w.name) || FILE_WIDGETS.includes(w.name) || FOLDER_WIDGETS.includes(w.name)) continue;
        if (!looksLikePath(w.value)) continue;
        const r = sourceFromValue(w.value, false);
        if (r) return { ...r, widget: w.name };
    }
    return null;
}

/** How a candidate ranks for a source_choice (higher = better). */
export function kindRank(kind, choice) {
    const R = {
        video: { video: 4, sequence: 3, folder: 3, image: 1, file: 1, audio: 0 },
        image: { image: 4, sequence: 3, folder: 3, video: 1, file: 1, audio: 0 },
        exr: { sequence: 4, folder: 3, image: 2, video: 1, file: 1, audio: 0 },
        auto: { video: 4, sequence: 4, folder: 4, image: 2, file: 1, audio: 0 },
    };
    return (R[choice] || R.auto)[kind] ?? 0;
}
