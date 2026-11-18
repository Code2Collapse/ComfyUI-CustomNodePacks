"""Face Landmarks (Facepoints) - 620 dense face points + the 68 / 98-point sets, for any model (L9.199).

Runs myned-ai/facepoints (HRNet-W48; code Apache-2.0, ported in nodes/facepoints_engine/; WEIGHTS CC BY-NC 4.0, not
shipped, downloaded only with allow_download). Every output is a standard format so any landmark consumer can use it:

  pose_keypoint   POSE_KEYPOINT, OpenPose JSON per frame: face_keypoints_2d = the 68 iBUG points + 2 pupils (70 x
                  [x, y, c] in pixels) - what DWPose / OpenPose face consumers, ControlNet face renderers and pose
                  editors read.
  landmark_image  the chosen set as white dots on black (OpenPose-face style control image).
  overlay         the input with the points drawn (620 coloured by kind; faint where the model is unsure).
  face_mask       the filled hull of the face points.
  landmarks_json  every set per frame and face: 620 points with their own error estimate, 68, 98, iris, box.

Faces are found by OpenCV's YuNet (MIT, 0.2 MB, downloaded with the same permission), an installed Ultralytics face
model, a face_mask, or the whole image (an image that is already a face crop). On video the box follows the face's own
points from the previous frame (track_video), as the model card recommends: the output moves with the box.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import folder_paths  # ComfyUI
except Exception:  # pragma: no cover - imported outside ComfyUI (tests)
    folder_paths = None

LOG = logging.getLogger("C2C.Facepoints")

REPO = "myned-ai/facepoints"
REVISION = "6d51ab48af940aca03819840c856984222002e3d"         # the commit the owner named (2026-11-12)
REPO_FILES = {"config.json": "model/config.json", "model.safetensors": "model/model.safetensors"}
WEIGHTS_SHA256 = "b37075caa3eaf0d00f23f32925b16ec3a8e2e45c42aa9c00956f94f2574e6526"   # the model card's value
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
FOLDER = "facepoints"
STD_CALIBRATION = 1.34          # model card: "multiply std by 1.34 to calibrate on average"
LOST_FACTOR = 4.0               # a face whose median std passes 4 x the threshold: no face in the box
CONFIDENT_PCT = 2.5             # model card NME ~3.3 % of eye distance ~ 1.3 % of face size: a typical good point

FINDER_YUNET = "yunet (finds faces)"
FINDER_WHOLE = "whole image (already a face crop)"
FINDER_YOLO = "yolo: "
SETS = ("68 + iris (OpenPose face)", "98 (WFLW)", "620 dense (GNM Head)")
# Colour contract (one boundary, never crossed twice):
#   controls are RGB hex "#RRGGBB" -> facepoints_controls.parse_hex -> (r, g, b) -> rgb_to_bgr -> (b, g, r)
#   -> OpenCV draws on BGR canvases (the frame is converted RGB -> BGR once, on the way in) -> the finished canvases
#   are converted BGR -> RGB once (run(): lm[:, :, ::-1], ov[:, :, ::-1]) -> IMAGE tensors in RGB 0..1.
# Visualisation defaults per GNM class (not model-required colours), high contrast on dark and light footage:
#   lmk #1E88E5 blue, skin #FF8C00 orange, lid #00C850 green, teeth_u #E63296 magenta, teeth_l #FF4DA6 pink,
#   iris #FFD700 yellow. Stored here as BGR (OpenCV order).
CLASS_BGR = {"lmk": (229, 136, 30), "skin": (0, 140, 255), "lid": (80, 200, 0), "teeth_u": (150, 50, 230),
             "teeth_l": (166, 77, 255), "iris": (0, 215, 255)}
# The published GNM release (myned-ai/facepoints config.json "classes", checked 2026-11-17 on the installed copy).
GNM_CLASS_COUNTS = {"lmk": 416, "skin": 150, "lid": 36, "teeth_u": 7, "teeth_l": 9, "iris": 2}
UNKNOWN_CLASS_BGR = (189, 189, 189)      # #BDBDBD: a class this node does not know is drawn grey and named in notes

if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=True)
    except Exception:  # pragma: no cover - older Comfy
        pass

_DETECTOR: Dict[Tuple[str, float], object] = {}
_YUNET: Dict[str, object] = {}


# ── model files ────────────────────────────────────────────────────────────────────────────────────────────────────

def _folders() -> List[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_folder_paths(FOLDER))
    except Exception:
        models_dir = getattr(folder_paths, "models_dir", None)
        return [os.path.join(models_dir, FOLDER)] if models_dir else []


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def find_release(folders: Optional[List[str]] = None) -> Optional[str]:
    """The first folder holding model.safetensors + config.json (also a model/ subfolder, the repo's old layout)."""
    for d in folders if folders is not None else _folders():
        for cand in (d, os.path.join(d, "model")):
            if all(os.path.isfile(os.path.join(cand, f)) for f in REPO_FILES):
                return cand
    return None


def ensure_release() -> str:
    """The installed release folder; with the node's permission, downloads it (pinned commit, SHA-256 checked)."""
    from ._download_policy import ModelNotDownloaded, downloads_allowed, note_found

    found = find_release()
    if found:
        note_found(found)
        return found
    folders = _folders()
    target = folders[0] if folders else os.path.abspath(FOLDER)
    if not downloads_allowed():
        raise ModelNotDownloaded(
            f"{REPO} (model.safetensors 416 MB + config.json; weights CC BY-NC 4.0, non-commercial)",
            target=target, source=f"https://huggingface.co/{REPO}", file_name="model.safetensors")
    from huggingface_hub import hf_hub_download

    os.makedirs(target, exist_ok=True)
    tmp = os.path.join(target, ".download")
    for name, repo_path in REPO_FILES.items():
        got = hf_hub_download(REPO, repo_path, revision=REVISION, local_dir=tmp)
        if name == "model.safetensors" and _sha256(got) != WEIGHTS_SHA256:
            os.remove(got)
            raise RuntimeError(f"{REPO} model.safetensors failed its SHA-256 check (expected {WEIGHTS_SHA256}); "
                               "the partial file was removed - run again to retry.")
        os.replace(got, os.path.join(target, name))
    shutil.rmtree(tmp, ignore_errors=True)
    return target


def _yunet_path() -> str:
    from ._download_policy import ModelNotDownloaded, downloads_allowed

    folders = _folders()
    for d in folders:
        p = os.path.join(d, YUNET_FILE)
        if os.path.isfile(p):
            return p
    target = folders[0] if folders else os.path.abspath(FOLDER)
    if not downloads_allowed():
        raise ModelNotDownloaded("YuNet face finder (OpenCV Zoo, MIT, 0.2 MB)", target=target, source=YUNET_URL,
                                 file_name=YUNET_FILE)
    import urllib.request

    os.makedirs(target, exist_ok=True)
    dst = os.path.join(target, YUNET_FILE)
    urllib.request.urlretrieve(YUNET_URL, dst + ".part")
    os.replace(dst + ".part", dst)
    return dst


def _detector(device: str):
    from .facepoints_engine import Detector

    folder = ensure_release()
    key = (folder, os.path.getmtime(os.path.join(folder, "model.safetensors")))
    det = _DETECTOR.get(key)
    if det is None:
        _DETECTOR.clear()
        det = Detector(folder, "cpu")
        _DETECTOR[key] = det
    return det.to(device)


# ── face finding ───────────────────────────────────────────────────────────────────────────────────────────────────

def _yolo_models() -> List[str]:
    out: List[str] = []
    if folder_paths is None:
        return out
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            for n in folder_paths.get_filename_list(key):
                if n.lower().endswith((".pt", ".onnx")) and FINDER_YOLO + n not in out:
                    out.append(FINDER_YOLO + n)
        except Exception:
            pass
    return out


def find_faces(bgr: np.ndarray, finder: str, confidence: float) -> List[Tuple[float, float, float, float, float]]:
    """(x, y, w, h, score) boxes, best first."""
    import cv2

    h, w = bgr.shape[:2]
    if finder == FINDER_WHOLE:
        return [(0.0, 0.0, float(w), float(h), 1.0)]
    if finder.startswith(FINDER_YOLO):
        from .mec_face_fixer import _detect_faces_yolo

        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        boxes = _detect_faces_yolo(rgb, finder[len(FINDER_YOLO):], confidence, 64)
        return sorted([(x0, y0, x1 - x0, y1 - y0, s) for x0, y0, x1, y1, s in boxes], key=lambda b: -b[4])
    path = _yunet_path()
    yn = _YUNET.get(path)
    if yn is None:
        yn = cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold=0.5)
        _YUNET[path] = yn
    yn.setScoreThreshold(float(confidence))
    yn.setInputSize((w, h))
    _, faces = yn.detect(bgr)
    if faces is None:
        return []
    return sorted([tuple(float(v) for v in f[:4]) + (float(f[-1]),) for f in faces], key=lambda b: -b[4])


def mask_box(mask: torch.Tensor, i: int) -> Optional[Tuple[float, float, float, float, float]]:
    m = mask if mask.ndim == 3 else mask[None]
    m = m[min(i, m.shape[0] - 1)]
    ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return None
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()) + 1, float(ys.min()), float(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0, 1.0)


# ── formats ────────────────────────────────────────────────────────────────────────────────────────────────────────

def face_size(points_xy: np.ndarray) -> float:
    """The face's size in px: the larger side of its skin points' extent (independent of the box it started from)."""
    p = np.asarray(points_xy, np.float64)
    return float(max(1e-6, (p.max(0) - p.min(0)).max()))


def confidence(std_px: np.ndarray, side: float, confident_pct: float) -> np.ndarray:
    """0..1 from the model's own error estimate: 0.5 at the 'confident' threshold (confident_pct % of the face)."""
    thr = max(1e-6, float(confident_pct) / 100.0 * float(side))
    s = np.asarray(std_px, np.float64) * STD_CALIBRATION
    return 1.0 / (1.0 + (s / thr) ** 2)


def openpose_face70(face: dict, classes: np.ndarray, conf_face: float, conf_iris: np.ndarray) -> List[float]:
    """68 iBUG points + the 2 pupils, as OpenPose orders them (68 = pupil of points 36-41, 69 = of 42-47)."""
    p68 = np.asarray(face["300w"], np.float64)
    iris = np.asarray(face["points"], np.float64)[classes == "iris"]
    flat: List[float] = []
    for x, y in p68:
        flat += [float(x), float(y), float(conf_face)]
    if len(iris) == 2:
        c36, c42 = p68[36:42].mean(0), p68[42:48].mean(0)
        first = 0 if np.linalg.norm(iris[0] - c36) + np.linalg.norm(iris[1] - c42) <= \
            np.linalg.norm(iris[1] - c36) + np.linalg.norm(iris[0] - c42) else 1
        for k in (first, 1 - first):
            flat += [float(iris[k][0]), float(iris[k][1]), float(conf_iris[k])]
    else:
        flat += [0.0] * 6
    return flat


def _set_points(face: dict, which: str) -> np.ndarray:
    if which == SETS[1]:
        return np.asarray(face["wflw"])
    if which == SETS[2]:
        return np.asarray(face["points"])
    return np.concatenate([np.asarray(face["300w"]), np.asarray(face["iris"])], 0)


def draw(canvas: np.ndarray, pts: np.ndarray, colour, radius: int, conf: Optional[np.ndarray] = None) -> None:
    import cv2

    for i, (x, y) in enumerate(pts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        c = colour if conf is None or conf[i] >= 0.5 else tuple(int(v * 0.45) for v in colour)
        cv2.circle(canvas, (int(round(x)), int(round(y))), int(radius), c, -1, lineType=cv2.LINE_AA)


def _image_stats(t: torch.Tensor) -> dict:
    """Size, value range and lit pixels of an output (IMAGE [B,H,W,3] or MASK [B,H,W]); 'lit' = brighter than 2 %."""
    t = t.detach().float().cpu()
    lit = (t.amax(-1) if t.ndim == 4 else t) > 0.02
    return {"shape": list(t.shape), "min": round(float(t.min()), 4) if t.numel() else None,
            "max": round(float(t.max()), 4) if t.numel() else None, "lit_pixels": int(lit.sum()),
            "lit_fraction": round(float(lit.float().mean()), 6) if t.numel() else 0.0}


def class_plan(classes: np.ndarray, kinds: dict) -> Tuple[List[Tuple[str, np.ndarray, tuple]], List[str]]:
    """The 620-point drawing plan by class NAME (never by index range): [(class, point mask, BGR colour)] in the
    palette's order, then any class the palette does not know (drawn UNKNOWN_CLASS_BGR). Returns (plan, warnings);
    a warning names unknown classes and per-class counts that differ from the published GNM release."""
    names = [str(v) for v in np.asarray(classes).tolist()]
    present = list(dict.fromkeys(names))
    plan, warn = [], []
    for k in list(kinds) + [k for k in present if k not in kinds]:
        sel = np.asarray(classes) == k
        if not sel.any():
            continue
        plan.append((k, sel, kinds.get(k, UNKNOWN_CLASS_BGR)))
    unknown = [k for k in present if k not in kinds]
    if unknown:
        warn.append(f"the model has point classes this node has no colour for: {', '.join(unknown)} - drawn grey "
                    f"(#BDBDBD)")
    counts = {k: names.count(k) for k in present}
    if counts != GNM_CLASS_COUNTS:
        warn.append(f"the model's point classes differ from the published GNM release: {counts} (expected "
                    f"{GNM_CLASS_COUNTS}); colours follow the class names")
    return plan, warn


# ── node ───────────────────────────────────────────────────────────────────────────────────────────────────────────

# A27 FACE-001: appended controls. Every default reproduces the earlier output exactly (tests compare bit for bit).
# Name prefixes are the groups: appearance_ / strength_ / resolution_ / screening_ / mask_ / temporal_.
SCHEMES = ("model card (default)", "custom")
LOW_CONF = ("dim (default)", "fade", "hide", "keep")
MASK_FACES_CHOICES = ("all faces", "largest face", "face index")   # = facepoints_controls.MASK_FACES
DEFAULT_HEX = {"lmk": "#1E88E5", "skin": "#FF8C00", "lid": "#00C850", "teeth_u": "#E63296", "teeth_l": "#FF4DA6",
               "iris": "#FFD700"}        # = CLASS_BGR as RGB hex
DEFAULT_SET_HEX = "#E63C3C"              # the 68 / 98 overlay colour (BGR 60, 60, 230)


# A28 (ORDERS 29): the appended controls live in the node's collapsible sections (js/_c2c_node_layout.js), one per
# name prefix, closed by default. They carry no core "advanced" flag: Nodes 2.0 would fold them a second time.
CONTROL_SECTIONS = {"appearance": "Appearance", "strength": "Strength", "resolution": "Resolution",
                    "screening": "Screening", "mask": "Mask", "temporal": "Temporal"}


def _controls_spec() -> dict:
    from .facepoints_controls import MASK_PREVIEWS, OUTPUT_SIZES, PROCESSING_SIZES
    c = {}
    # Appearance
    c["appearance_scheme"] = (list(SCHEMES), ({"default": SCHEMES[0], "tooltip": "Appearance. model card: the "
                              "original colours (white dots on black; 620 points coloured by kind). custom: the "
                              "colours below. Colours change the images only, never the POSE_KEYPOINT numbers."}))
    c["appearance_landmark_colour"] = ("STRING", ({"default": "#FFFFFF", "tooltip": "Appearance (custom): dot "
                                       "colour in landmark_image, #RRGGBB."}))
    c["appearance_background_colour"] = ("STRING", ({"default": "#000000", "tooltip": "Appearance (custom): "
                                         "landmark_image background, #RRGGBB."}))
    c["appearance_overlay_colour"] = ("STRING", ({"default": "", "tooltip": "Appearance (custom): one colour "
                                      "for every dot in overlay, #RRGGBB. Empty = the per-set / per-kind colours "
                                      "below."}))
    for k in ("lmk", "skin", "lid", "teeth_u", "teeth_l", "iris"):
        c[f"appearance_colour_{k}"] = ("STRING", ({"default": DEFAULT_HEX[k], "tooltip": f"Appearance (custom): "
                                       f"overlay colour of the 620-point kind '{k}', #RRGGBB."}))
    c["appearance_colour_68"] = ("STRING", ({"default": DEFAULT_SET_HEX, "tooltip": "Appearance (custom): "
                                 "overlay colour of the 68 + iris set, #RRGGBB."}))
    c["appearance_colour_98"] = ("STRING", ({"default": DEFAULT_SET_HEX, "tooltip": "Appearance (custom): "
                                 "overlay colour of the 98 set, #RRGGBB."}))
    c["appearance_kinds_on_landmark_image"] = ("BOOLEAN", ({"default": False, "tooltip": "Appearance (custom): "
                                               "landmark_image uses the overlay's per-set / per-kind colours "
                                               "instead of one dot colour."}))
    # Strength
    c["strength_overlay_opacity"] = ("FLOAT", ({"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Strength: how opaque the overlay's dots are over the image (real "
                                                "alpha blending). 1 = solid."}))
    c["strength_dot_intensity"] = ("FLOAT", ({"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Strength: landmark_image dot brightness, blended toward the "
                                              "background (a weaker control image). 1 = full colour."}))
    c["strength_low_confidence_opacity"] = ("FLOAT", ({"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Strength: how strong an unsure point is drawn - its colour "
                                                       "factor in 'dim' mode, its opacity in 'fade' mode."}))
    c["strength_render_threshold"] = ("FLOAT", ({"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Strength: a point whose confidence is below this is drawn as "
                                                 "unsure (see screening_low_confidence_points). Drawing only."}))
    c["strength_pose_confidence_scale"] = ("FLOAT", ({"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                                           "tooltip": "Strength: multiplies the confidence values in "
                                                      "pose_keypoint (clamped to 0..1). Coordinates are unchanged."}))
    # Resolution
    c["resolution_processing"] = (list(PROCESSING_SIZES), ({"default": "source", "tooltip": "Resolution: "
                                  "size the faces are FOUND and cropped at. source = the input; a number = the "
                                  "longer side at most that (downscale only, aspect kept). The network itself "
                                  "always sees the checkpoint's own input size. Points come back in source "
                                  "pixels."}))
    c["resolution_processing_max"] = ("INT", ({"default": 1024, "min": 64, "max": 16384, "step": 8,
                                      "tooltip": "Resolution: the longer side for resolution_processing = custom."}))
    c["resolution_output"] = (list(OUTPUT_SIZES), ({"default": "original", "tooltip": "Resolution: size of "
                              "landmark_image, overlay, face_mask - and the coordinates in pose_keypoint and JSON "
                              "follow it, so everything stays aligned. original = the input size."}))
    c["resolution_output_width"] = ("INT", ({"default": 1024, "min": 16, "max": 16384, "step": 8,
                                    "tooltip": "Resolution: width for resolution_output = custom."}))
    c["resolution_output_height"] = ("INT", ({"default": 1024, "min": 16, "max": 16384, "step": 8,
                                     "tooltip": "Resolution: height for resolution_output = custom."}))
    c["resolution_keep_aspect"] = ("BOOLEAN", ({"default": True, "tooltip": "Resolution: fit inside the "
                                   "output size keeping the aspect ratio (the output is the fitted size). Off: "
                                   "stretch to exactly that size."}))
    # Confidence / screening
    c["screening_face_confidence_min"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Screening: faces whose overall landmark confidence is below "
                                                     "this are left out of the images and pose_keypoint (the face "
                                                     "finder's own threshold is finder_confidence)."}))
    c["screening_point_confidence_min"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Screening: pose_keypoint points below this confidence get "
                                                      "c = 0 (consumers skip them); their x, y stay. JSON always "
                                                      "keeps every point and lists the screened ones."}))
    c["screening_low_confidence_points"] = (list(LOW_CONF), ({"default": LOW_CONF[0], "tooltip": "Screening: "
                                            "how unsure 620-set points (below strength_render_threshold) are DRAWN "
                                            "- dim (the original look, overlay only), fade (transparent), hide, or "
                                            "keep (like sure points). fade / hide apply to landmark_image too. Data "
                                            "is never removed by this."}))
    c["screening_keep_rejected_faces"] = ("BOOLEAN", ({"default": False, "tooltip": "Screening: list rejected "
                                          "faces (no face in the box, below the face confidence) in the JSON with "
                                          "the reason, for diagnosis. They never enter the images or pose."}))
    # Face mask
    c["mask_grow"] = ("INT", ({"default": 0, "min": 0, "max": 512, "tooltip": "Face mask: expand (dilate) by "
                      "this many px."}))
    c["mask_shrink"] = ("INT", ({"default": 0, "min": 0, "max": 512, "tooltip": "Face mask: contract (erode) "
                        "by this many px (after mask_grow)."}))
    c["mask_feather"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip": "Face "
                         "mask: Gaussian blur, sigma in px (soft on both sides of the edge)."}))
    c["mask_edge_softness"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip":
                               "Face mask: fade in from the edge over this many px (the outer edge stays put)."}))
    c["mask_threshold"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Face "
                           "mask: > 0 makes the mask hard at this level (applied last). 0 = off."}))
    c["mask_hull_smoothing"] = ("INT", ({"default": 0, "min": 0, "max": 6, "tooltip": "Face mask: rounds the "
                                "hull outline (corner-cutting passes). 0 = the original hull."}))
    c["mask_preview"] = (list(MASK_PREVIEWS), ({"default": "off", "tooltip": "Face mask: show the mask on the "
                         "overlay image (tint or outline). The MASK output is not changed by this."}))
    # Temporal stabilisation
    c["temporal_smoothing"] = ("BOOLEAN", ({"default": False, "tooltip": "Temporal: smooth landmark "
                               "trajectories over a video (adaptive: still points steady, real motion followed). "
                               "Independent of track_video. Raw points stay in the JSON."}))
    c["temporal_strength"] = ("FLOAT", ({"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip":
                              "Temporal: smoothing of still points. 0 = none; higher = steadier."}))
    c["temporal_confidence_aware"] = ("BOOLEAN", ({"default": True, "tooltip": "Temporal: points the model is "
                                      "less sure of than before move less toward their new position."}))
    c["temporal_max_offset"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5, "tooltip":
                                "Temporal: the most (px) smoothing may hold a point away from its detected "
                                "position in a frame. 0 = no limit."}))
    c["temporal_scene_cut"] = ("FLOAT", ({"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                               "Temporal: a frame that differs from the previous one by more than this (mean "
                               "absolute difference, 0..1) is a scene cut: smoothing and identities restart. 0 = "
                               "off."}))
    c["temporal_reset_on_loss"] = ("BOOLEAN", ({"default": True, "tooltip": "Temporal: when a face is not "
                                   "found in a frame, its smoothing restarts when it comes back."}))
    c["temporal_stable_order"] = ("BOOLEAN", ({"default": False, "tooltip": "Temporal: keep each face's "
                                  "position in pose_keypoint / JSON by identity across frames (matched by "
                                  "position). Off: left to right in every frame (the original order)."}))
    # work order 2026-11-14 (Face Landmarks): connection-line strength and mask face selection, appended
    c["strength_connection_lines"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Strength: draw the face contours (jaw, brows, nose, eyes, lips) as "
                                                 "lines between the points - 0 = off. For the 68 and 98 sets (the "
                                                 "620 dense points have no contour order). Drawing only."}))
    c["appearance_line_width"] = ("INT", ({"default": 1, "min": 1, "max": 16, "tooltip": "Appearance: "
                                  "connection line width in px (output pixels)."}))
    c["mask_faces"] = (list(MASK_FACES_CHOICES), ({"default": MASK_FACES_CHOICES[0], "tooltip": "Face mask: "
                       "which faces fill face_mask - all, the largest, or one by its position in pose_keypoint."}))
    c["mask_face_index"] = ("INT", ({"default": 0, "min": 0, "max": 15, "tooltip": "Face mask: the face for "
                            "mask_faces = face index (0 = first in pose_keypoint)."}))
    # L9.279 (owner 2026-11-17): which part of the head face_mask covers, invert, and a directional extend that
    # follows the head's tilt. Appended last; the defaults give today's mask exactly.
    from .facepoints_controls import MASK_PARTS, HEAD_REFINE
    c["mask_part"] = (list(MASK_PARTS), ({"default": MASK_PARTS[0], "tooltip": "Face mask: which part of the head "
                      "face_mask covers. Features (eyes, lids, brows, nose, lips, inner mouth) and areas (around the "
                      "mouth, cheeks, chin + jaw) come from the landmarks; forehead, ears, hair and head + hair are "
                      "ESTIMATED (no landmark covers them) and fitted to the picture by mask_head_refine. custom "
                      "list: several parts joined, typed in mask_parts_custom."}))
    c["mask_parts_custom"] = ("STRING", ({"default": "", "tooltip": "Face mask (mask_part = custom list): parts "
                              "joined into one mask, comma separated, e.g. 'eyes, lips' or 'head + hair (estimated), "
                              "whole face'.", "c2c_show_if": {"mask_part": ["custom list"]}}))
    c["mask_invert"] = ("BOOLEAN", ({"default": False, "tooltip": "Face mask: everything EXCEPT the chosen parts "
                        "(applied after grow / shrink / feather)."}))
    c["mask_head_refine"] = (list(HEAD_REFINE), ({"default": HEAD_REFINE[0], "tooltip": "Face mask (estimated "
                             "parts): image edges = fit the estimated forehead / ears / hair / head to the picture "
                             "(GrabCut on a small crop, cheap); off = the geometric estimate only."}))
    for side in ("up", "down", "left", "right"):
        c[f"mask_extend_{side}"] = ("FLOAT", ({"default": 0.0, "min": -100.0, "max": 300.0, "step": 1.0,
                                    "tooltip": f"Face mask: extend the mask {side} by this % of the face height "
                                               "(chin to brows); negative shrinks that side. 'up' is the head's up "
                                               "when mask_extend_follow_tilt is on."}))
    from .facepoints_controls import PARTS_ESTIMATED
    c["mask_head_refine"][1]["c2c_show_if"] = {"mask_part": list(PARTS_ESTIMATED) + ["custom list"]}
    _extending = [{f"mask_extend_{d}": {"truthy": True}} for d in ("up", "down", "left", "right")]
    c["mask_extend_follow_tilt"] = ("BOOLEAN", ({"default": True, "c2c_show_if_any": _extending,
                                    "tooltip": "Face mask: extend along the head's "
                                    "own axes (a tilted head is extended along its tilt); off = the picture's axes."}))
    c["mask_extend_falloff"] = ("FLOAT", ({"default": 0.0, "min": 0.0, "max": 300.0, "step": 1.0,
                                "c2c_show_if_any": _extending, "tooltip":
                                "Face mask: soft fall-off of the extended band, % of the face height (0 = hard)."}))
    from .facepoints_controls import ROBUST_MODES
    c["screening_robust_detection"] = (list(ROBUST_MODES), ({"default": ROBUST_MODES[0], "tooltip": "Screening: "
                                       "low light, colour casts and motion blur. auto: a frame where no face is "
                                       "found is analysed and retried on enhanced copies (light, colour, deblur, a "
                                       "1.5x look) and then with a nearby frame's face box; always: every frame is "
                                       "detected on the enhanced copy; off: as before. The copies are only shown to "
                                       "the detector - every output keeps the original footage."}))
    for name, spec in c.items():
        spec[1]["c2c_section"] = CONTROL_SECTIONS[name.split("_")[0]]
    return c


_DEFAULTS: Optional[dict] = None


def control_defaults() -> dict:
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = {k: v[1]["default"] for k, v in _controls_spec().items()}
    return _DEFAULTS


def _mask_shaped(c: dict) -> bool:
    return any(c[k] for k in ("mask_grow", "mask_shrink", "mask_feather", "mask_edge_softness", "mask_threshold"))


class FacepointsLandmarksC2C:
    DESCRIPTION = (
        "Dense face landmarks for any model: 620 points on Google's GNM Head mesh, each with its own error estimate, "
        "plus the standard 68-point (iBUG) and 98-point (WFLW) sets and both pupils, from myned-ai/facepoints. "
        "Outputs OpenPose POSE_KEYPOINT (face = 68 + 2 pupils, pixels), a white-dot control image, an overlay, a face "
        "mask and JSON with every set. Video: the box follows the face's own points frame to frame. Weights are "
        "CC BY-NC 4.0 (non-commercial use only) and are downloaded only with allow_download, into models/facepoints."
        "\n\nAdvanced controls (the node's collapsible sections):appearance_ (colours), strength_ (opacity, intensity, pose "
        "confidence), resolution_ (processing and output size; everything stays aligned), screening_ (face / point "
        "confidence, how unsure points are drawn, rejected faces in JSON), mask_ (grow, shrink, feather, softness, "
        "threshold, hull smoothing, preview), temporal_ (landmark smoothing, scene cuts, stable face order). The "
        "defaults give the original output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "face_mask": ("MASK", {"tooltip": "A mask over the face (per frame, or one for all): its box is used "
                                              "instead of the face finder. One face."}),
            "finder_confidence": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 0.99, "step": 0.01,
                                            "tooltip": "Minimum score for yunet / yolo faces."}),
            "confident_pct": ("FLOAT", {"default": CONFIDENT_PCT, "min": 0.1, "max": 10.0, "step": 0.1,
                                        "tooltip": "A point counts as sure when its calibrated error estimate is "
                                                   "below this % of the face size (2.5 = a little above a typical good point); its confidence c is 0.5 "
                                                   "exactly there. Unsure points are drawn faint in overlay."}),
        }
        optional.update(_controls_spec())                 # A27 FACE-001, appended after the original inputs
        return _with_layout({
            "required": {
                "image": ("IMAGE", {"tooltip": "One image or a video (frames). Any size."}),
                "face_finder": ([FINDER_YUNET, FINDER_WHOLE] + _yolo_models(), {
                    "default": FINDER_YUNET,
                    "tooltip": "How the face box is found. yunet: OpenCV's face finder (MIT, 0.2 MB, downloaded with "
                               "allow_download). whole image: the image is already a face crop. yolo: an installed "
                               "Ultralytics face model (models/ultralytics/bbox). A wired face_mask overrides this."}),
                "max_faces": ("INT", {"default": 1, "min": 1, "max": 16,
                                      "tooltip": "Faces per frame, left to right."}),
                "passes": ("INT", {"default": 3, "min": 1, "max": 5,
                                   "tooltip": "Network passes per face. Each pass re-frames the box on the previous "
                                              "points (the model's own procedure; 3 = the released default)."}),
                "track_video": ("BOOLEAN", {"default": True,
                                            "tooltip": "Video: start each frame's box from the face's points in the "
                                                       "previous frame (steadier, and the finder only looks for new "
                                                       "faces). Off: every frame is found on its own."}),
                "landmark_set": (list(SETS), {"default": SETS[0],
                                              "tooltip": "The points drawn into landmark_image. pose_keypoint is "
                                                         "always OpenPose's 68 + 2 pupils."}),
                "point_size": ("INT", {"default": 3, "min": 1, "max": 16,
                                       "tooltip": "Dot radius in pixels for landmark_image and overlay."}),
                "allow_download": ("BOOLEAN", {"default": False,
                                               "tooltip": "Allow downloading myned-ai/facepoints (416 MB, CC BY-NC "
                                                          "4.0 = non-commercial use only) and YuNet (0.2 MB, MIT) "
                                                          "into models/facepoints when they are missing."}),
            },
            "optional": optional,
        })

    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("pose_keypoint", "landmark_image", "overlay", "face_mask", "landmarks_json")
    OUTPUT_TOOLTIPS = (
        "OpenPose JSON per frame; face_keypoints_2d = 68 iBUG points + 2 pupils (70 x [x, y, c], pixels of the "
        "output size). strength_pose_confidence_scale / screening_point_confidence_min change c only.",
        "The chosen landmark set as dots (white on black by default) - a control image.",
        "The input with the points drawn (620: coloured by kind, unsure points dim / faded / hidden).",
        "Filled hull of the face points, 0..1 (grow / shrink / feather / softness / threshold when set).",
        "Every set per frame and face: 620 points with std, 68, 98, iris, box, confidence; raw points when "
        "smoothed; screened points and rejected faces when asked.",
    )
    FUNCTION = "run"
    CATEGORY = "C2C/Face"

    # ---- controls ----------------------------------------------------------------------------------------------
    @staticmethod
    def _controls(kw: dict) -> dict:
        d = dict(control_defaults())
        unknown = [k for k in kw if k not in d]
        if unknown:
            raise TypeError(f"Face Landmarks: unknown input(s) {', '.join(sorted(unknown))}")
        d.update({k: v for k, v in kw.items() if v is not None})
        for k in ("screening_low_confidence_points", "appearance_scheme"):
            allowed = LOW_CONF if k.startswith("screening") else SCHEMES
            if d[k] not in allowed:
                raise ValueError(f"{k}: unknown value {d[k]!r} (one of {', '.join(allowed)})")
        from .facepoints_controls import MASK_PARTS, HEAD_REFINE
        from .facepoints_controls import ROBUST_MODES
        for k, allowed in (("mask_part", MASK_PARTS), ("mask_head_refine", HEAD_REFINE),
                           ("screening_robust_detection", ROBUST_MODES)):
            if d[k] not in allowed:
                raise ValueError(f"{k}: unknown value {d[k]!r} (one of {', '.join(allowed)})")
        return d

    @staticmethod
    def _palette(c: dict) -> dict:
        """BGR colours for drawing; validated hex in custom mode, the original colours otherwise."""
        from .facepoints_controls import parse_hex, rgb_to_bgr
        if c["appearance_scheme"] == SCHEMES[0]:
            return {"lm": (255, 255, 255), "bg": (0, 0, 0), "overlay": None, "kinds": dict(CLASS_BGR),
                    "set68": (60, 60, 230), "set98": (60, 60, 230), "kinds_on_lm": False}

        def col(name):
            return rgb_to_bgr(parse_hex(c[name], name))
        ov = c["appearance_overlay_colour"]
        return {"lm": col("appearance_landmark_colour"), "bg": col("appearance_background_colour"),
                "overlay": col("appearance_overlay_colour") if str(ov or "").strip() else None,
                "kinds": {k: col(f"appearance_colour_{k}") for k in CLASS_BGR},
                "set68": col("appearance_colour_68"), "set98": col("appearance_colour_98"),
                "kinds_on_lm": bool(c["appearance_kinds_on_landmark_image"])}

    def run(self, image, face_finder=FINDER_YUNET, max_faces=1, passes=3, track_video=True,
            landmark_set=SETS[0], point_size=3, allow_download=False, face_mask=None, finder_confidence=0.6,
            confident_pct=CONFIDENT_PCT, **controls):
        import comfy.model_management as mm
        import comfy.utils

        from . import facepoints_controls as FC
        from ._download_policy import allow_downloads, model_status
        from .facepoints_engine import box_of_points, square_box

        c = self._controls(controls)
        defaults = control_defaults()
        changed = sorted(k for k, v in c.items() if v != defaults[k])
        pal = self._palette(c)
        low_mode = c["screening_low_confidence_points"].split(" ")[0]
        img = image if image.ndim == 4 else image[None]
        frames, H, W = img.shape[0], img.shape[1], img.shape[2]
        P = FC.processing_resample(W, H, c["resolution_processing"], c["resolution_processing_max"])
        O = FC.output_resample(W, H, c["resolution_output"], c["resolution_output_width"],
                               c["resolution_output_height"], bool(c["resolution_keep_aspect"]))
        Wo, Ho = (W, H) if O.identity else (O.width, O.height)
        back = P.inverse(W, H)                               # processing px -> source px
        device = str(mm.get_torch_device())
        pbar = comfy.utils.ProgressBar(frames)
        pose, lm_imgs, overlays, masks, js_frames = [], [], [], [], []
        status = ""
        tracks: List = []
        next_ident = 0
        largest_ident = None             # L9.252 F2: the identity "largest face" is locked to
        prev_bgr = None
        index_missing: List[int] = []
        smoothing = bool(c["temporal_smoothing"])
        shaped_mask = _mask_shaped(c)
        from . import _mask_directional as MD
        parts = FC.parse_parts(c["mask_part"], c["mask_parts_custom"])
        extend_any = any(float(c[f"mask_extend_{s}"]) for s in ("up", "down", "left", "right"))
        parts_mode = parts != [FC.PART_DEFAULT] or extend_any
        robust_mode = c["screening_robust_detection"]
        robust_frames: List[int] = []
        with allow_downloads(allow_download):
            det = _detector(device)
            status = model_status()
            try:
                classes = det.cls
                mesh = det.mesh
                plan620, class_warn = class_plan(classes, pal["kinds"])
                for w_ in class_warn:
                    LOG.warning("Face Landmarks: %s", w_)
                no_face_frames: List[int] = []
                slots: List[Optional[dict]] = []        # tracked faces, PROCESSING px (left-to-right order kept)
                for i in range(frames):
                    mm.throw_exception_if_processing_interrupted()
                    rgb = (img[i, :, :, :3].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    work = FC.resize_image(bgr, P)
                    starts: List[tuple] = []
                    finder_boxes, tracked = None, 0
                    box_source = "face_mask" if face_mask is not None else "finder"
                    if face_mask is not None:
                        mb = mask_box(face_mask, i)
                        if mb and not P.identity:
                            mb = (mb[0] * P.sx, mb[1] * P.sy, mb[2] * P.sx, mb[3] * P.sy, mb[4])
                        starts = [square_box(*mb[:4])] if mb else []
                        slots = []
                    else:
                        # L9.252 F1: on a scene cut the old shot's boxes must not seed this frame's search. Checked
                        # only where cuts are detected at all (smoothing / stable order), so defaults are unchanged.
                        thr_cut = float(c["temporal_scene_cut"])
                        if (slots and (smoothing or c["temporal_stable_order"]) and thr_cut > 0
                                and prev_bgr is not None and FC.frame_change(prev_bgr, bgr) > thr_cut):
                            slots = []
                        live = [s for s in slots if s is not None] if (track_video and i > 0) else []
                        starts = [box_of_points(s["points"][mesh]) for s in live]
                        tracked = len(starts)
                        if len(starts) < int(max_faces):
                            fbs = find_faces(work, face_finder, finder_confidence)
                            finder_boxes = len(fbs)
                            for fb in fbs:
                                cx, cy = fb[0] + fb[2] / 2, fb[1] + fb[3] / 2
                                if any(b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[2] for b in starts):
                                    continue        # already tracked
                                starts.append(square_box(*fb[:4]))
                                if len(starts) >= int(max_faces):
                                    break
                    def _evaluate(found_):
                        kept_, rejected_ = [], []
                        for f in found_:
                            if not P.identity:                  # back to source pixels
                                f = {**f, "proc_points": f["points"], "points": back.points(f["points"]),
                                     "std": back.length(f["std"]), "300w": back.points(f["300w"]),
                                     "wflw": back.points(f["wflw"]), "box": back.box(f["box"])}
                            side = face_size(f["points"][mesh])
                            f["face_size_px"] = side
                            med = float(np.median(f["std"][mesh])) * STD_CALIBRATION
                            f["median_std_px"] = med
                            f["iris"] = np.asarray(f["points"])[classes == "iris"]
                            if med > LOST_FACTOR * float(confident_pct) / 100.0 * side:
                                rejected_.append((f, "no face in the box (error estimate too large)"))
                                continue                # no face in this box (the card: "threshold it")
                            kept_.append(f)
                        return kept_, rejected_

                    def _boxes(w_):
                        out_ = []
                        for fb in find_faces(w_, face_finder, finder_confidence):
                            out_.append(square_box(*fb[:4]))
                            if len(out_) >= int(max_faces):
                                break
                        return out_

                    robust = {"rung": None}
                    det_in = work
                    if robust_mode == FC.ROBUST_MODES[2] and face_mask is None:     # always: detect on the copy
                        analysis = FC.analyse_frame(work)
                        name_, det_in = FC.enhancement_ladder(work, analysis)[-1]
                        robust = {"rung": f"always: {name_}", "analysis": analysis}
                        if not tracked:
                            starts = _boxes(det_in) or starts
                    found = det.detect_many([det_in] * len(starts), starts, int(passes)) if starts else []
                    kept, rejected = _evaluate(found)
                    if not kept and robust_mode == FC.ROBUST_MODES[0] and face_mask is None:
                        # L9.280: nothing found - analyse the frame and retry on enhanced DETECTION copies, then with
                        # the box of a nearby frame. Outputs keep the original footage.
                        analysis = FC.analyse_frame(work)
                        robust = {"rung": None, "analysis": analysis, "tried": []}
                        ladder = FC.enhancement_ladder(work, analysis)
                        for name_, w_e in ladder:
                            robust["tried"].append(name_)
                            st = _boxes(w_e)
                            if not st and face_finder != FINDER_WHOLE:         # small faces: a 1.5x look
                                import cv2
                                up = cv2.resize(w_e, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                                st = [(b[0] / 1.5, b[1] / 1.5, b[2] / 1.5) for b in _boxes(up)]
                                name_ = f"{name_} + 1.5x" if st else name_
                            if not st:
                                continue
                            k_, r_ = _evaluate(det.detect_many([w_e] * len(st), st, int(passes)))
                            robust["rejected"] = robust.get("rejected", 0) + len(r_)
                            if k_:
                                kept, robust["rung"], starts = k_, name_, st
                                break
                        if not kept and face_finder != FINDER_WHOLE:
                            for j in [i + d for d in (1, -1, 2, -2, 3, 4, 6, 8) if 0 <= i + d < frames]:
                                rgb_j = (img[j, :, :, :3].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
                                w_j = FC.resize_image(np.ascontiguousarray(rgb_j[:, :, ::-1]), P)
                                st = _boxes(w_j) or _boxes(ladder[0][1] if j == i else
                                                           FC.enhancement_ladder(w_j)[0][1])
                                if not st:
                                    continue
                                w_use = ladder[-1][1]
                                k_, r_ = _evaluate(det.detect_many([w_use] * len(st), st, int(passes)))
                                if k_:
                                    kept, robust["rung"], starts = k_, f"box from frame {j}", st
                                    break
                        if kept:
                            robust_frames.append(i)
                    kept.sort(key=lambda f: float(np.mean(f["points"][:, 0])))
                    # tracking uses the processing-space points (independent of screening and smoothing)
                    slots = [({**f, "points": f["proc_points"]} if "proc_points" in f else f) for f in kept]
                    shown = []
                    for f in kept:                              # confidence and face screening (source px)
                        side = f["face_size_px"]
                        f["conf620"] = confidence(f["std"], side, confident_pct)
                        f["conf_face"] = float(confidence(np.array([f["median_std_px"] / STD_CALIBRATION]), side,
                                                          confident_pct)[0])
                        if f["conf_face"] < float(c["screening_face_confidence_min"]):
                            rejected.append((f, f"face confidence {f['conf_face']:.3f} is below "
                                                f"screening_face_confidence_min"))
                            continue
                        shown.append(f)
                    cut = False
                    if smoothing or c["temporal_stable_order"]:
                        tracks, next_ident, cut = _temporal(FC, c, shown, tracks, next_ident, prev_bgr, bgr, mesh,
                                                            classes, smoothing)
                    prev_bgr = bgr
                    people, js_faces = [], []
                    lm = np.zeros((Ho, Wo, 3), np.uint8)
                    if pal["bg"] != (0, 0, 0):
                        lm[:] = pal["bg"]
                    ov = FC.resize_image(bgr, O).copy() if not O.identity else bgr.copy()
                    fm = np.zeros((Ho, Wo), np.uint8)
                    r = int(point_size)
                    lm_col = pal["lm"]
                    if float(c["strength_dot_intensity"]) < 1.0:
                        k = float(c["strength_dot_intensity"])
                        lm_col = tuple(int(round(b + (v - b) * k)) for v, b in zip(lm_col, pal["bg"]))
                    opacity = float(c["strength_overlay_opacity"])
                    rthr = float(c["strength_render_threshold"])
                    low_op = float(c["strength_low_confidence_opacity"])
                    line_k, line_w = float(c["strength_connection_lines"]), int(c["appearance_line_width"])
                    line_runs = (FC.LINES_68 if landmark_set == SETS[0] else FC.LINES_98 if landmark_set == SETS[1]
                                 else ())
                    outs = []
                    for f in shown:
                        g = f if O.identity else {**f, "points": O.points(f["points"]), "300w": O.points(f["300w"]),
                                                  "wflw": O.points(f["wflw"]), "iris": O.points(f["iris"]),
                                                  "box": O.box(f["box"]), "std": O.length(f["std"]),
                                                  "face_size_px": float(O.length(f["face_size_px"]))}
                        outs.append(g)
                        face70 = openpose_face70(g, classes, f["conf_face"], f["conf620"][classes == "iris"])
                        f["screened_pose"] = _screen_pose(face70, float(c["strength_pose_confidence_scale"]),
                                                          float(c["screening_point_confidence_min"]))
                        people.append({"pose_keypoints_2d": [], "face_keypoints_2d": face70,
                                       "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": []})
                        lm_pts = _set_points(g, landmark_set)
                        if line_k > 0 and line_runs:
                            FC.draw_lines(lm, lm_pts, line_runs, lm_col if not pal["kinds_on_lm"] else (
                                pal["overlay"] or (pal["set98"] if landmark_set == SETS[1] else pal["set68"])),
                                line_w, line_k)
                        lm_alpha = None
                        if landmark_set == SETS[2] and low_mode in ("fade", "hide"):
                            lm_alpha, _ = FC.point_alpha(f["conf620"], len(lm_pts), rthr, low_mode, low_op)
                        if pal["kinds_on_lm"] and landmark_set == SETS[2]:
                            for kk, sel, col in plan620:
                                FC.draw_points(lm, lm_pts[sel], pal["overlay"] or col, r,
                                               None if lm_alpha is None else lm_alpha[sel])
                        else:
                            col = lm_col
                            if pal["kinds_on_lm"]:
                                col = pal["overlay"] or (pal["set98"] if landmark_set == SETS[1] else pal["set68"])
                            FC.draw_points(lm, lm_pts, col, r, lm_alpha)
                    if cut:
                        largest_ident = None
                    idents = [gg.get("identity") for gg in outs]
                    if c["mask_faces"] == "largest face" and outs and all(x is not None for x in idents):
                        picked, largest_ident = FC.sticky_largest([gg["face_size_px"] for gg in outs], idents,
                                                                  largest_ident)
                    else:
                        picked = FC.mask_face_selection([gg["face_size_px"] for gg in outs], c["mask_faces"],
                                                        int(c["mask_face_index"]))
                    if c["mask_faces"] == "face index" and outs and not picked:
                        index_missing.append(i)          # L9.252 F3: said in the JSON, never a silent empty mask
                    if not parts_mode:
                        for k in picked:
                            FC.fill_hull(fm, outs[k]["points"][mesh], int(c["mask_hull_smoothing"]))
                    else:                                # L9.279: head parts, refine, directional extend
                        acc = np.zeros((Ho, Wo), np.float32)
                        for k in picked:
                            g = outs[k]
                            sm = int(c["mask_hull_smoothing"])
                            one = np.zeros((Ho, Wo), np.uint8)
                            for part in parts:
                                pm = FC.part_mask(part, (Ho, Wo), g["300w"], g["points"], classes, mesh, sm)
                                if part in FC.PARTS_ESTIMATED and c["mask_head_refine"] == FC.HEAD_REFINE[0]:
                                    sure = FC.part_mask("whole face", (Ho, Wo), g["300w"], g["points"], classes,
                                                        mesh, sm)
                                    if part in ("hair (estimated)", "head + hair (estimated)"):
                                        # the head may follow the picture past the estimate (long hair, volume)
                                        head = pm if part.startswith("head") else FC.part_mask(
                                            "head + hair (estimated)", (Ho, Wo), g["300w"], g["points"], classes,
                                            mesh, sm)
                                        fit = FC.refine_head(ov, head, sure)
                                        pm = fit if part.startswith("head") else (np.where(
                                            sure > 0, 0, fit).astype(np.uint8) & FC.above_eyes((Ho, Wo), g["300w"]))
                                    else:                # forehead / ears: the picture can only trim the estimate
                                        pm = FC.refine_head(ov, np.maximum(pm, sure), sure) & pm
                                one = np.maximum(one, pm)
                            fr = FC.face_frame(g["300w"])
                            if extend_any:
                                Hf = fr.H / 100.0
                                ext = MD.extend(one, up=c["mask_extend_up"] * Hf, down=c["mask_extend_down"] * Hf,
                                                left=c["mask_extend_left"] * Hf, right=c["mask_extend_right"] * Hf,
                                                angle_deg=fr.angle_deg if c["mask_extend_follow_tilt"] else 0.0,
                                                falloff_px=c["mask_extend_falloff"] * Hf)
                            else:
                                ext = one.astype(np.float32) / 255.0
                            acc = np.maximum(acc, ext)
                        fm = np.round(acc * 255.0).astype(np.uint8)
                    if shaped_mask:
                        mask_f = FC.shape_mask(fm, int(c["mask_grow"]), int(c["mask_shrink"]),
                                               float(c["mask_feather"]), float(c["mask_edge_softness"]),
                                               float(c["mask_threshold"]))
                    else:
                        mask_f = None
                    if c["mask_invert"]:
                        if mask_f is not None:
                            mask_f = (1.0 - mask_f).astype(np.float32)
                        else:
                            fm = (255 - fm).astype(np.uint8)
                    if c["mask_preview"] != "off":
                        FC.preview_mask(ov, mask_f if mask_f is not None else fm.astype(np.float32) / 255.0,
                                        c["mask_preview"])
                    drawn_per_class: Dict[str, int] = {}
                    finite_in_frame = 0
                    for f, g in zip(shown, outs):
                        gp = np.asarray(g["points"], np.float64)
                        inside = (np.isfinite(gp).all(1) & (gp[:, 0] >= 0) & (gp[:, 0] < Wo) & (gp[:, 1] >= 0)
                                  & (gp[:, 1] < Ho))
                        finite_in_frame += int(inside.sum())
                        if landmark_set == SETS[2]:
                            alpha, factor = FC.point_alpha(f["conf620"], len(g["points"]), rthr, low_mode, low_op,
                                                           opacity)
                            for kk, sel, col in plan620:
                                FC.draw_points(ov, g["points"][sel], pal["overlay"] or col, max(1, r - 1),
                                               alpha[sel], factor[sel])
                                drawn_per_class[kk] = drawn_per_class.get(kk, 0) + int(
                                    (inside[sel] & (np.asarray(alpha)[sel] > 0)).sum())
                        else:
                            sp = np.asarray(_set_points(g, landmark_set), np.float64)
                            key = "68+iris" if landmark_set == SETS[0] else "98"
                            drawn_per_class[key] = drawn_per_class.get(key, 0) + int(
                                (np.isfinite(sp).all(1) & (sp[:, 0] >= 0) & (sp[:, 0] < Wo) & (sp[:, 1] >= 0)
                                 & (sp[:, 1] < Ho)).sum()) if opacity > 0 else 0
                            col = pal["overlay"] or (pal["set98"] if landmark_set == SETS[1] else pal["set68"])
                            if line_k > 0 and line_runs:
                                FC.draw_lines(ov, _set_points(g, landmark_set), line_runs, col, line_w,
                                              line_k * opacity)
                            FC.draw_points(ov, _set_points(g, landmark_set), col, r, opacity)
                        js_faces.append(_face_json(f, g, O))
                    pose.append({"people": people, "canvas_width": int(Wo), "canvas_height": int(Ho)})
                    frame = {"faces": js_faces}
                    if not shown:
                        no_face_frames.append(i)
                    frame["diagnostics"] = {
                        "box_source": box_source, "tracked_boxes": int(tracked),
                        "finder_boxes": finder_boxes, "faces_inferred": len(found), "faces_accepted": len(shown),
                        "faces_rejected": len(rejected), "rejected_reasons": [why for _f, why in rejected],
                        "points_finite_in_frame": int(finite_in_frame), "points_drawn_on_overlay": drawn_per_class}
                    if len(robust) > 1 or robust["rung"]:
                        frame["diagnostics"]["robust"] = robust
                    if c["screening_keep_rejected_faces"]:
                        frame["rejected_faces"] = [
                            {"reason": why, "box": [round(float(v), 2) for v in O.box(f["box"])],
                             "confidence": None if "conf_face" not in f else round(float(f["conf_face"]), 4),
                             "median_std_px": round(float(f["median_std_px"]), 3)} for f, why in rejected]
                    if cut:
                        frame["scene_cut"] = True
                    js_frames.append(frame)
                    lm_imgs.append(torch.from_numpy(lm[:, :, ::-1].copy()))
                    overlays.append(torch.from_numpy(ov[:, :, ::-1].copy()))
                    masks.append(torch.from_numpy(fm) if mask_f is None else torch.from_numpy(mask_f))
                    pbar.update_absolute(i + 1, frames)
            finally:
                if device != "cpu":
                    det.to("cpu")
                    mm.soft_empty_cache()
        out_dev = mm.intermediate_device()
        lm_t = (torch.stack(lm_imgs).float() / 255.0).to(out_dev)
        ov_t = (torch.stack(overlays).float() / 255.0).to(out_dev)
        if shaped_mask:
            mk_t = torch.stack(masks).to(out_dev)
        else:
            mk_t = (torch.stack(masks).float() / 255.0).to(out_dev)
        info = {"format": "c2c-facepoints-1", "model": f"{REPO}@{REVISION[:7]}",
                "weights_licence": "CC BY-NC 4.0 (non-commercial)", "status": status,
                "image_size": [int(Wo), int(Ho)], "classes_620": det.config["classes"],
                "std_note": f"points_620 third value = the model's std in px; x{STD_CALIBRATION} calibrates it",
                "frames": js_frames}
        notes = []
        if c["strength_connection_lines"] > 0 and landmark_set == SETS[2]:
            notes.append("connection lines are drawn for the 68 and 98 sets; the 620 dense points have no "
                         "published contour order, so none were drawn")
        if index_missing:
            notes.append(f"mask_face_index {int(c['mask_face_index'])} is past the number of faces in "
                         f"{len(index_missing)} frame(s) (first: {index_missing[0]}); face_mask is empty there")
        # an all-black landmark_image is only expected where no face was accepted: say so, never silently
        diag = {"outputs": {name: _image_stats(t) for name, t in
                            (("landmark_image", lm_t), ("overlay", ov_t), ("face_mask", mk_t))},
                "frames_without_face": len(no_face_frames), "class_warnings": class_warn,
                "frames_found_by_enhanced_detection": len(robust_frames)}
        if robust_frames:
            notes.append(f"enhanced detection found the face in {len(robust_frames)} frame(s) (first: "
                         f"{robust_frames[0]}) that plain detection missed (low light / colour / blur); see each "
                         "frame's diagnostics.robust. Outputs show the original footage.")
        if no_face_frames:
            first = js_frames[no_face_frames[0]]["diagnostics"]
            why = ("the face finder found no face" if first["faces_inferred"] == 0
                   else "every face was rejected: " + "; ".join(sorted(set(first["rejected_reasons"]))))
            notes.append(f"no face in {len(no_face_frames)} of {frames} frame(s) (first: {no_face_frames[0]}: {why}); "
                         "landmark_image is black there and overlay shows no points. Try face_finder = whole image "
                         "for a face crop, lower finder_confidence, or a larger processing resolution.")
            LOG.warning("Face Landmarks: %s", notes[-1])
        if any("no colour for" in w_ for w_ in class_warn):
            notes.append(next(w_ for w_ in class_warn if "no colour for" in w_))
        info["diagnostics"] = diag
        if notes:
            info["notes"] = notes
        if changed:                                     # only then, so the default JSON stays byte-identical
            info["controls"] = {k: c[k] for k in changed}
            info["source_size"] = [int(W), int(H)]
            info["processing_size"] = [int(P.width), int(P.height)]
            info["output_scale"] = [O.sx, O.sy]
            info["coordinates"] = "output pixels (image_size); source pixels = output / output_scale"
        return (pose, lm_t, ov_t, mk_t, json.dumps(info))


def _with_layout(spec: dict) -> dict:
    """A28 (ORDERS 29, docs/audit/A28/README.md section 5): which controls show when, read by js/_c2c_node_layout.js.
    Display only - every input keeps its name, order, type and default, is saved and reaches run() as before."""
    finders = [FINDER_YUNET] + [n for n in spec["required"]["face_finder"][0] if n.startswith(FINDER_YOLO)]
    layout = {
        "face_finder": {"c2c_hide_if_linked": ["face_mask"]},
        "max_faces": {"c2c_hide_if_linked": ["face_mask"], "c2c_show_if": {"face_finder": finders}},
        "finder_confidence": {"c2c_hide_if_linked": ["face_mask"], "c2c_show_if": {"face_finder": finders}},
        "passes": {"c2c_section": "Tracking"},
        "track_video": {"c2c_section": "Tracking", "c2c_hide_if_linked": ["face_mask"]},
        "confident_pct": {"c2c_section": "Confidence"},
        "landmark_set": {"c2c_section": "Output"},
        "point_size": {"c2c_section": "Output"},
        # shown only while the weights are missing (GET /c2c/node_status, layout_status below); consent stays the
        # user's tick, and run() still goes through _download_policy.allow_downloads
        "allow_download": {"c2c_section": "Model setup", "c2c_show_if_status": "model_missing"},
    }
    for group in ("required", "optional"):
        for name, s in spec[group].items():
            if name in layout:
                s[1].update(layout[name])
    return spec


def layout_status() -> dict:
    """The model status the front end needs for allow_download (A28 c2c_show_if_status). It only LOOKS, with the same
    resolvers run() uses (find_release, the YuNet file in models/facepoints): never downloads, never loads."""
    release = find_release() is not None
    yunet = any(os.path.isfile(os.path.join(d, YUNET_FILE)) for d in _folders())
    return {"model_missing": not (release and yunet), "release_installed": release, "yunet_installed": yunet}


try:                                                       # A28: the layout engine's status route
    from ._node_status import register_status
    register_status("FacepointsLandmarksC2C", layout_status)
except Exception:  # pragma: no cover - imported outside the pack (tests load the file directly)
    pass


def _screen_pose(face70: List[float], scale: float, pmin: float) -> List[int]:
    """Pose confidence scaling (values only) and point screening (c = 0, x / y kept). Returns screened indices."""
    screened: List[int] = []
    if scale == 1.0 and pmin <= 0.0:
        return screened
    for j in range(len(face70) // 3):
        cj = face70[3 * j + 2]
        missing = j >= 68 and face70[3 * j] == 0.0 and face70[3 * j + 1] == 0.0     # no pupil found
        if pmin > 0.0 and cj < pmin and not missing:
            screened.append(j)
            face70[3 * j + 2] = 0.0
        else:
            face70[3 * j + 2] = float(min(1.0, max(0.0, cj * scale)))
    return screened


def _temporal(FC, c, shown, tracks, next_ident, prev_bgr, bgr, mesh, classes, smoothing):
    """Face identities (matched by position), scene cuts, tracking-loss resets and landmark smoothing, in place."""
    cut = False
    thr = float(c["temporal_scene_cut"])
    if thr > 0 and prev_bgr is not None and FC.frame_change(prev_bgr, bgr) > thr:
        tracks, cut = [], True
    cents = [np.asarray(f["points"][mesh]).mean(0) for f in shown]
    match = FC.match_faces(tracks, cents, [f["face_size_px"] for f in shown])
    seen = set()
    for f, ti, cen in zip(shown, match, cents):
        if ti is None:
            t = FC.FaceTrack(next_ident, cen, f["face_size_px"])
            next_ident += 1
            tracks.append(t)
        else:
            t = tracks[ti]
        t.centre, t.size, t.missed = cen, f["face_size_px"], 0
        f["identity"] = t.ident
        seen.add(t.ident)
        if smoothing:
            f["raw"] = {"points": np.asarray(f["points"]).copy(), "300w": np.asarray(f["300w"]).copy(),
                        "wflw": np.asarray(f["wflw"]).copy(), "iris": np.asarray(f["iris"]).copy()}
            for key, conf in (("points", f["conf620"]), ("300w", np.full(len(f["300w"]), f["conf_face"])),
                              ("wflw", np.full(len(f["wflw"]), f["conf_face"]))):
                flt = t.filters.setdefault(key, FC.PointFilter(float(c["temporal_strength"]),
                                                               bool(c["temporal_confidence_aware"]),
                                                               float(c["temporal_max_offset"])))
                f[key] = flt(np.asarray(f[key], np.float64), conf, f["face_size_px"])
            f["iris"] = np.asarray(f["points"])[classes == "iris"]
    for t in tracks:
        if t.ident not in seen:
            t.missed += 1
            if c["temporal_reset_on_loss"]:
                for flt in t.filters.values():
                    flt.reset()
    tracks = FC.prune_tracks(tracks)                   # L9.252 F4
    if c["temporal_stable_order"]:
        shown.sort(key=lambda f: f["identity"])
    return tracks, next_ident, cut


def _face_json(f: dict, g: dict, O) -> dict:
    js = {
        "box": [round(float(v), 2) for v in g["box"]],
        "median_std_px": round(f["median_std_px"] if O.identity else float(O.length(f["median_std_px"])), 3),
        "face_size_px": round(g["face_size_px"], 1),
        "confidence": round(f["conf_face"], 4),
        "points_620": [[round(float(x), 2), round(float(y), 2), round(float(s), 3)]
                       for (x, y), s in zip(g["points"], g["std"])],
        "set_68": np.round(np.asarray(g["300w"], np.float64), 2).tolist(),
        "set_98": np.round(np.asarray(g["wflw"], np.float64), 2).tolist(),
        "iris": np.round(np.asarray(g["iris"], np.float64), 2).tolist(),
    }
    if "identity" in f:
        js["identity"] = int(f["identity"])
    if f.get("screened_pose"):
        js["screened_pose_points"] = f["screened_pose"]
    if "raw" in f:                                      # unsmoothed, in the same (output) pixels
        js["raw"] = {k: np.round(np.asarray(O.points(f["raw"][k]), np.float64), 2).tolist()
                     for k in ("points", "300w", "wflw", "iris")}
    return js


NODE_CLASS_MAPPINGS = {"FacepointsLandmarksC2C": FacepointsLandmarksC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"FacepointsLandmarksC2C": "Face Landmarks (Facepoints, 620 + 68/98)"}
