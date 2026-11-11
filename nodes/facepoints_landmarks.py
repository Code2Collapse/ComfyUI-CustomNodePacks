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

class FacepointsLandmarksC2C:
    DESCRIPTION = (
        "Dense face landmarks for any model: 620 points on Google's GNM Head mesh, each with its own error estimate, "
        "plus the standard 68-point (iBUG) and 98-point (WFLW) sets and both pupils, from myned-ai/facepoints. "
        "Outputs OpenPose POSE_KEYPOINT (face = 68 + 2 pupils, pixels), a white-dot control image, an overlay, a face "
        "mask and JSON with every set. Video: the box follows the face's own points frame to frame. Weights are "
        "CC BY-NC 4.0 (non-commercial use only) and are downloaded only with allow_download, into models/facepoints."
    )

    @classmethod
    def INPUT_TYPES(cls):
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
            "optional": {
                "face_mask": ("MASK", {"tooltip": "A mask over the face (per frame, or one for all): its box is used "
                                                  "instead of the face finder. One face."}),
                "finder_confidence": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 0.99, "step": 0.01,
                                                "tooltip": "Minimum score for yunet / yolo faces."}),
                "confident_pct": ("FLOAT", {"default": CONFIDENT_PCT, "min": 0.1, "max": 10.0, "step": 0.1,
                                            "tooltip": "A point counts as sure when its calibrated error estimate is "
                                                       "below this % of the face size (2.5 = a little above a typical good point); its confidence c is 0.5 "
                                                       "exactly there. Unsure points are drawn faint in overlay."}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("pose_keypoint", "landmark_image", "overlay", "face_mask", "landmarks_json")
    FUNCTION = "run"
    CATEGORY = "C2C/Face"

    def run(self, image, face_finder=FINDER_YUNET, max_faces=1, passes=3, track_video=True,
            landmark_set=SETS[0], point_size=3, allow_download=False, face_mask=None, finder_confidence=0.6,
            confident_pct=CONFIDENT_PCT):
        import cv2

        import comfy.model_management as mm
        import comfy.utils

        from ._download_policy import allow_downloads, model_status
        from .facepoints_engine import box_of_points, square_box

        img = image if image.ndim == 4 else image[None]
        frames, H, W = img.shape[0], img.shape[1], img.shape[2]
        device = str(mm.get_torch_device())
        pbar = comfy.utils.ProgressBar(frames)
        pose, lm_imgs, overlays, masks, js_frames = [], [], [], [], []
        status = ""
        with allow_downloads(allow_download):
            det = _detector(device)
            status = model_status()
            try:
                classes = det.cls
                mesh = det.mesh
                slots: List[Optional[dict]] = []        # tracked faces (left-to-right order kept)
                for i in range(frames):
                    mm.throw_exception_if_processing_interrupted()
                    rgb = (img[i, :, :, :3].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    starts: List[tuple] = []
                    if face_mask is not None:
                        mb = mask_box(face_mask, i)
                        starts = [square_box(*mb[:4])] if mb else []
                        slots = []
                    else:
                        live = [s for s in slots if s is not None] if (track_video and i > 0) else []
                        starts = [box_of_points(s["points"][mesh]) for s in live]
                        if len(starts) < int(max_faces):
                            for fb in find_faces(bgr, face_finder, finder_confidence):
                                cx, cy = fb[0] + fb[2] / 2, fb[1] + fb[3] / 2
                                if any(b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[2] for b in starts):
                                    continue        # already tracked
                                starts.append(square_box(*fb[:4]))
                                if len(starts) >= int(max_faces):
                                    break
                    faces = det.detect_many([bgr] * len(starts), starts, int(passes)) if starts else []
                    kept = []
                    for f in faces:
                        side = face_size(f["points"][mesh])
                        f["face_size_px"] = side
                        med = float(np.median(f["std"][mesh])) * STD_CALIBRATION
                        f["median_std_px"] = med
                        f["iris"] = np.asarray(f["points"])[classes == "iris"]
                        if med > LOST_FACTOR * float(confident_pct) / 100.0 * side:
                            continue                    # no face in this box (the card: "threshold it")
                        kept.append(f)
                    kept.sort(key=lambda f: float(np.mean(f["points"][:, 0])))
                    slots = kept
                    people, js_faces = [], []
                    lm = np.zeros((H, W, 3), np.uint8)
                    ov = bgr.copy()
                    fm = np.zeros((H, W), np.uint8)
                    r = int(point_size)
                    for f in kept:
                        side = f["face_size_px"]
                        conf620 = confidence(f["std"], side, confident_pct)
                        conf_face = float(confidence(np.array([f["median_std_px"] / STD_CALIBRATION]), side,
                                                     confident_pct)[0])
                        people.append({"pose_keypoints_2d": [],
                                       "face_keypoints_2d": openpose_face70(f, classes, conf_face,
                                                                            conf620[classes == "iris"]),
                                       "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": []})
                        draw(lm, _set_points(f, landmark_set), (255, 255, 255), r)
                        if landmark_set == SETS[2]:
                            for k, col in CLASS_BGR.items():
                                sel = classes == k
                                draw(ov, f["points"][sel], col, max(1, r - 1), conf620[sel])
                        else:
                            draw(ov, _set_points(f, landmark_set), (60, 60, 230), r)
                        hull = cv2.convexHull(np.round(f["points"][mesh]).astype(np.int32))
                        cv2.fillConvexPoly(fm, hull, 255)
                        js_faces.append({
                            "box": [round(float(v), 2) for v in f["box"]],
                            "median_std_px": round(f["median_std_px"], 3),
                            "face_size_px": round(f["face_size_px"], 1),
                            "confidence": round(conf_face, 4),
                            "points_620": [[round(float(x), 2), round(float(y), 2), round(float(s), 3)]
                                           for (x, y), s in zip(f["points"], f["std"])],
                            "set_68": np.round(np.asarray(f["300w"], np.float64), 2).tolist(),
                            "set_98": np.round(np.asarray(f["wflw"], np.float64), 2).tolist(),
                            "iris": np.round(f["iris"].astype(np.float64), 2).tolist(),
                        })
                    pose.append({"people": people, "canvas_width": int(W), "canvas_height": int(H)})
                    js_frames.append({"faces": js_faces})
                    lm_imgs.append(torch.from_numpy(lm[:, :, ::-1].copy()))
                    overlays.append(torch.from_numpy(ov[:, :, ::-1].copy()))
                    masks.append(torch.from_numpy(fm))
                    pbar.update_absolute(i + 1, frames)
            finally:
                if device != "cpu":
                    det.to("cpu")
                    mm.soft_empty_cache()
        out_dev = mm.intermediate_device()
        lm_t = (torch.stack(lm_imgs).float() / 255.0).to(out_dev)
        ov_t = (torch.stack(overlays).float() / 255.0).to(out_dev)
        mk_t = (torch.stack(masks).float() / 255.0).to(out_dev)
        info = {"format": "c2c-facepoints-1", "model": f"{REPO}@{REVISION[:7]}",
                "weights_licence": "CC BY-NC 4.0 (non-commercial)", "status": status,
                "image_size": [int(W), int(H)], "classes_620": det.config["classes"],
                "std_note": f"points_620 third value = the model's std in px; x{STD_CALIBRATION} calibrates it",
                "frames": js_frames}
        return (pose, lm_t, ov_t, mk_t, json.dumps(info))


NODE_CLASS_MAPPINGS = {"FacepointsLandmarksC2C": FacepointsLandmarksC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"FacepointsLandmarksC2C": "Face Landmarks (Facepoints, 620 + 68/98)"}
