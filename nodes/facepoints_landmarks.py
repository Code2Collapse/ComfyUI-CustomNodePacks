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
CLASS_BGR = {"lmk": (229, 136, 30), "skin": (0, 140, 255), "lid": (80, 200, 0), "teeth_u": (150, 50, 230),
             "teeth_l": (150, 50, 230), "iris": (0, 215, 255)}   # BGR, as the model card's figure

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


# ── node ───────────────────────────────────────────────────────────────────────────────────────────────────────────

# A27 FACE-001: appended controls. Every default reproduces the earlier output exactly (tests compare bit for bit).
# Name prefixes are the groups: appearance_ / strength_ / resolution_ / screening_ / mask_ / temporal_.
SCHEMES = ("model card (default)", "custom")
LOW_CONF = ("dim (default)", "fade", "hide", "keep")
DEFAULT_HEX = {"lmk": "#1E88E5", "skin": "#FF8C00", "lid": "#00C850", "teeth_u": "#E63296", "teeth_l": "#E63296",
               "iris": "#FFD700"}        # = CLASS_BGR as RGB hex
DEFAULT_SET_HEX = "#E63C3C"              # the 68 / 98 overlay colour (BGR 60, 60, 230)


def _adv(spec: dict) -> dict:
    spec["advanced"] = True
    return spec


def _controls_spec() -> dict:
    from .facepoints_controls import MASK_PREVIEWS, OUTPUT_SIZES, PROCESSING_SIZES
    c = {}
    # Appearance
    c["appearance_scheme"] = (list(SCHEMES), _adv({"default": SCHEMES[0], "tooltip": "Appearance. model card: the "
                              "original colours (white dots on black; 620 points coloured by kind). custom: the "
                              "colours below. Colours change the images only, never the POSE_KEYPOINT numbers."}))
    c["appearance_landmark_colour"] = ("STRING", _adv({"default": "#FFFFFF", "tooltip": "Appearance (custom): dot "
                                       "colour in landmark_image, #RRGGBB."}))
    c["appearance_background_colour"] = ("STRING", _adv({"default": "#000000", "tooltip": "Appearance (custom): "
                                         "landmark_image background, #RRGGBB."}))
    c["appearance_overlay_colour"] = ("STRING", _adv({"default": "", "tooltip": "Appearance (custom): one colour "
                                      "for every dot in overlay, #RRGGBB. Empty = the per-set / per-kind colours "
                                      "below."}))
    for k in ("lmk", "skin", "lid", "teeth_u", "teeth_l", "iris"):
        c[f"appearance_colour_{k}"] = ("STRING", _adv({"default": DEFAULT_HEX[k], "tooltip": f"Appearance (custom): "
                                       f"overlay colour of the 620-point kind '{k}', #RRGGBB."}))
    c["appearance_colour_68"] = ("STRING", _adv({"default": DEFAULT_SET_HEX, "tooltip": "Appearance (custom): "
                                 "overlay colour of the 68 + iris set, #RRGGBB."}))
    c["appearance_colour_98"] = ("STRING", _adv({"default": DEFAULT_SET_HEX, "tooltip": "Appearance (custom): "
                                 "overlay colour of the 98 set, #RRGGBB."}))
    c["appearance_kinds_on_landmark_image"] = ("BOOLEAN", _adv({"default": False, "tooltip": "Appearance (custom): "
                                               "landmark_image uses the overlay's per-set / per-kind colours "
                                               "instead of one dot colour."}))
    # Strength
    c["strength_overlay_opacity"] = ("FLOAT", _adv({"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Strength: how opaque the overlay's dots are over the image (real "
                                                "alpha blending). 1 = solid."}))
    c["strength_dot_intensity"] = ("FLOAT", _adv({"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Strength: landmark_image dot brightness, blended toward the "
                                              "background (a weaker control image). 1 = full colour."}))
    c["strength_low_confidence_opacity"] = ("FLOAT", _adv({"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Strength: how strong an unsure point is drawn - its colour "
                                                       "factor in 'dim' mode, its opacity in 'fade' mode."}))
    c["strength_render_threshold"] = ("FLOAT", _adv({"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Strength: a point whose confidence is below this is drawn as "
                                                 "unsure (see screening_low_confidence_points). Drawing only."}))
    c["strength_pose_confidence_scale"] = ("FLOAT", _adv({"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                                           "tooltip": "Strength: multiplies the confidence values in "
                                                      "pose_keypoint (clamped to 0..1). Coordinates are unchanged."}))
    # Resolution
    c["resolution_processing"] = (list(PROCESSING_SIZES), _adv({"default": "source", "tooltip": "Resolution: "
                                  "size the faces are FOUND and cropped at. source = the input; a number = the "
                                  "longer side at most that (downscale only, aspect kept). The network itself "
                                  "always sees the checkpoint's own input size. Points come back in source "
                                  "pixels."}))
    c["resolution_processing_max"] = ("INT", _adv({"default": 1024, "min": 64, "max": 16384, "step": 8,
                                      "tooltip": "Resolution: the longer side for resolution_processing = custom."}))
    c["resolution_output"] = (list(OUTPUT_SIZES), _adv({"default": "original", "tooltip": "Resolution: size of "
                              "landmark_image, overlay, face_mask - and the coordinates in pose_keypoint and JSON "
                              "follow it, so everything stays aligned. original = the input size."}))
    c["resolution_output_width"] = ("INT", _adv({"default": 1024, "min": 16, "max": 16384, "step": 8,
                                    "tooltip": "Resolution: width for resolution_output = custom."}))
    c["resolution_output_height"] = ("INT", _adv({"default": 1024, "min": 16, "max": 16384, "step": 8,
                                     "tooltip": "Resolution: height for resolution_output = custom."}))
    c["resolution_keep_aspect"] = ("BOOLEAN", _adv({"default": True, "tooltip": "Resolution: fit inside the "
                                   "output size keeping the aspect ratio (the output is the fitted size). Off: "
                                   "stretch to exactly that size."}))
    # Confidence / screening
    c["screening_face_confidence_min"] = ("FLOAT", _adv({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Screening: faces whose overall landmark confidence is below "
                                                     "this are left out of the images and pose_keypoint (the face "
                                                     "finder's own threshold is finder_confidence)."}))
    c["screening_point_confidence_min"] = ("FLOAT", _adv({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Screening: pose_keypoint points below this confidence get "
                                                      "c = 0 (consumers skip them); their x, y stay. JSON always "
                                                      "keeps every point and lists the screened ones."}))
    c["screening_low_confidence_points"] = (list(LOW_CONF), _adv({"default": LOW_CONF[0], "tooltip": "Screening: "
                                            "how unsure 620-set points (below strength_render_threshold) are DRAWN "
                                            "- dim (the original look, overlay only), fade (transparent), hide, or "
                                            "keep (like sure points). fade / hide apply to landmark_image too. Data "
                                            "is never removed by this."}))
    c["screening_keep_rejected_faces"] = ("BOOLEAN", _adv({"default": False, "tooltip": "Screening: list rejected "
                                          "faces (no face in the box, below the face confidence) in the JSON with "
                                          "the reason, for diagnosis. They never enter the images or pose."}))
    # Face mask
    c["mask_grow"] = ("INT", _adv({"default": 0, "min": 0, "max": 512, "tooltip": "Face mask: expand (dilate) by "
                      "this many px."}))
    c["mask_shrink"] = ("INT", _adv({"default": 0, "min": 0, "max": 512, "tooltip": "Face mask: contract (erode) "
                        "by this many px (after mask_grow)."}))
    c["mask_feather"] = ("FLOAT", _adv({"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip": "Face "
                         "mask: Gaussian blur, sigma in px (soft on both sides of the edge)."}))
    c["mask_edge_softness"] = ("FLOAT", _adv({"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip":
                               "Face mask: fade in from the edge over this many px (the outer edge stays put)."}))
    c["mask_threshold"] = ("FLOAT", _adv({"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Face "
                           "mask: > 0 makes the mask hard at this level (applied last). 0 = off."}))
    c["mask_hull_smoothing"] = ("INT", _adv({"default": 0, "min": 0, "max": 6, "tooltip": "Face mask: rounds the "
                                "hull outline (corner-cutting passes). 0 = the original hull."}))
    c["mask_preview"] = (list(MASK_PREVIEWS), _adv({"default": "off", "tooltip": "Face mask: show the mask on the "
                         "overlay image (tint or outline). The MASK output is not changed by this."}))
    # Temporal stabilisation
    c["temporal_smoothing"] = ("BOOLEAN", _adv({"default": False, "tooltip": "Temporal: smooth landmark "
                               "trajectories over a video (adaptive: still points steady, real motion followed). "
                               "Independent of track_video. Raw points stay in the JSON."}))
    c["temporal_strength"] = ("FLOAT", _adv({"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip":
                              "Temporal: smoothing of still points. 0 = none; higher = steadier."}))
    c["temporal_confidence_aware"] = ("BOOLEAN", _adv({"default": True, "tooltip": "Temporal: points the model is "
                                      "less sure of than before move less toward their new position."}))
    c["temporal_max_offset"] = ("FLOAT", _adv({"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5, "tooltip":
                                "Temporal: the most (px) smoothing may hold a point away from its detected "
                                "position in a frame. 0 = no limit."}))
    c["temporal_scene_cut"] = ("FLOAT", _adv({"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                               "Temporal: a frame that differs from the previous one by more than this (mean "
                               "absolute difference, 0..1) is a scene cut: smoothing and identities restart. 0 = "
                               "off."}))
    c["temporal_reset_on_loss"] = ("BOOLEAN", _adv({"default": True, "tooltip": "Temporal: when a face is not "
                                   "found in a frame, its smoothing restarts when it comes back."}))
    c["temporal_stable_order"] = ("BOOLEAN", _adv({"default": False, "tooltip": "Temporal: keep each face's "
                                  "position in pose_keypoint / JSON by identity across frames (matched by "
                                  "position). Off: left to right in every frame (the original order)."}))
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
        "\n\nAdvanced controls (show advanced inputs): appearance_ (colours), strength_ (opacity, intensity, pose "
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
        return {
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
        }

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
        prev_bgr = None
        smoothing = bool(c["temporal_smoothing"])
        shaped_mask = _mask_shaped(c)
        with allow_downloads(allow_download):
            det = _detector(device)
            status = model_status()
            try:
                classes = det.cls
                mesh = det.mesh
                slots: List[Optional[dict]] = []        # tracked faces, PROCESSING px (left-to-right order kept)
                for i in range(frames):
                    mm.throw_exception_if_processing_interrupted()
                    rgb = (img[i, :, :, :3].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    work = FC.resize_image(bgr, P)
                    starts: List[tuple] = []
                    if face_mask is not None:
                        mb = mask_box(face_mask, i)
                        if mb and not P.identity:
                            mb = (mb[0] * P.sx, mb[1] * P.sy, mb[2] * P.sx, mb[3] * P.sy, mb[4])
                        starts = [square_box(*mb[:4])] if mb else []
                        slots = []
                    else:
                        live = [s for s in slots if s is not None] if (track_video and i > 0) else []
                        starts = [box_of_points(s["points"][mesh]) for s in live]
                        if len(starts) < int(max_faces):
                            for fb in find_faces(work, face_finder, finder_confidence):
                                cx, cy = fb[0] + fb[2] / 2, fb[1] + fb[3] / 2
                                if any(b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[2] for b in starts):
                                    continue        # already tracked
                                starts.append(square_box(*fb[:4]))
                                if len(starts) >= int(max_faces):
                                    break
                    found = det.detect_many([work] * len(starts), starts, int(passes)) if starts else []
                    kept, rejected = [], []
                    for f in found:
                        if not P.identity:                      # back to source pixels
                            f = {**f, "proc_points": f["points"], "points": back.points(f["points"]),
                                 "std": back.length(f["std"]), "300w": back.points(f["300w"]),
                                 "wflw": back.points(f["wflw"]), "box": back.box(f["box"])}
                        side = face_size(f["points"][mesh])
                        f["face_size_px"] = side
                        med = float(np.median(f["std"][mesh])) * STD_CALIBRATION
                        f["median_std_px"] = med
                        f["iris"] = np.asarray(f["points"])[classes == "iris"]
                        if med > LOST_FACTOR * float(confident_pct) / 100.0 * side:
                            rejected.append((f, "no face in the box (error estimate too large)"))
                            continue                    # no face in this box (the card: "threshold it")
                        kept.append(f)
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
                        lm_alpha = None
                        if landmark_set == SETS[2] and low_mode in ("fade", "hide"):
                            lm_alpha, _ = FC.point_alpha(f["conf620"], len(lm_pts), rthr, low_mode, low_op)
                        if pal["kinds_on_lm"] and landmark_set == SETS[2]:
                            for kk, col in pal["kinds"].items():
                                sel = classes == kk
                                FC.draw_points(lm, lm_pts[sel], pal["overlay"] or col, r,
                                               None if lm_alpha is None else lm_alpha[sel])
                        else:
                            col = lm_col
                            if pal["kinds_on_lm"]:
                                col = pal["overlay"] or (pal["set98"] if landmark_set == SETS[1] else pal["set68"])
                            FC.draw_points(lm, lm_pts, col, r, lm_alpha)
                        FC.fill_hull(fm, g["points"][mesh], int(c["mask_hull_smoothing"]))
                    if shaped_mask:
                        mask_f = FC.shape_mask(fm, int(c["mask_grow"]), int(c["mask_shrink"]),
                                               float(c["mask_feather"]), float(c["mask_edge_softness"]),
                                               float(c["mask_threshold"]))
                    else:
                        mask_f = None
                    if c["mask_preview"] != "off":
                        FC.preview_mask(ov, mask_f if mask_f is not None else fm.astype(np.float32) / 255.0,
                                        c["mask_preview"])
                    for f, g in zip(shown, outs):
                        if landmark_set == SETS[2]:
                            alpha, factor = FC.point_alpha(f["conf620"], len(g["points"]), rthr, low_mode, low_op,
                                                           opacity)
                            for kk, col in pal["kinds"].items():
                                sel = classes == kk
                                FC.draw_points(ov, g["points"][sel], pal["overlay"] or col, max(1, r - 1),
                                               alpha[sel], factor[sel])
                        else:
                            col = pal["overlay"] or (pal["set98"] if landmark_set == SETS[1] else pal["set68"])
                            FC.draw_points(ov, _set_points(g, landmark_set), col, r, opacity)
                        js_faces.append(_face_json(f, g, O))
                    pose.append({"people": people, "canvas_width": int(Wo), "canvas_height": int(Ho)})
                    frame = {"faces": js_faces}
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
        if changed:                                     # only then, so the default JSON stays byte-identical
            info["controls"] = {k: c[k] for k in changed}
            info["source_size"] = [int(W), int(H)]
            info["processing_size"] = [int(P.width), int(P.height)]
            info["output_scale"] = [O.sx, O.sy]
            info["coordinates"] = "output pixels (image_size); source pixels = output / output_scale"
        return (pose, lm_t, ov_t, mk_t, json.dumps(info))


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
