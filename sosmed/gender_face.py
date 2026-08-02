"""
Face + gender detection for person clips, using OpenCV's DNN face detector
and a small gender-classification model.

This module adds *who* information to the active-speaker crop path:
  * face detection inside each YOLO person box (so the close-up centers on the
    speaker's FACE/mouth, not their torso)
  * gender classification (Male/Female), used to label or pick the crop framing

Model weights are downloaded automatically on first use (matching the
project's "weights download on first run" convention used by yolov8n.pt),
and cached under the project .cache directory so subsequent runs are offline.

Requires the optional ``opencv-python-headless`` extra (``dnn`` submodule).
If OpenCV DNN is unavailable the helpers degrade gracefully and return None
rather than raising, so callers can fall back to the existing person-box crop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import log

# ── Model definitions (downloaded on first use) ──────────────────────────────
_FACE_PROTO = "opencv_face_detector.pbtxt"
_FACE_MODEL = "opencv_face_detector_uint8.pb"
_GENDER_PROTO = "gender_deploy.prototxt"
_GENDER_MODEL = "gender_net.caffemodel"
_GENDER_LABELS = ["Female", "Male"]

_MODELS_URLS = {
    _FACE_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
    _FACE_MODEL: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector_uint8.pb",
    _GENDER_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/gender_age_train/models/gender_deploy.prototxt",
    _GENDER_MODEL: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/gender_age_train/models/gender_net.caffemodel",
}


def _cache_dir() -> Path:
    """Directory for downloaded DNN weights (project .cache/dnn)."""
    d = Path.cwd() / ".cache" / "dnn"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_if_missing(name: str) -> Path:
    """Download a model file if it is not already in the cache.

    Raises RuntimeError (no silent fallback) when the download fails, so the
    caller is forced to handle the missing dependency explicitly rather than
    shipping a wrong crop.
    """
    path = _cache_dir() / name
    if path.exists() and path.stat().st_size > 0:
        return path
    url = _MODELS_URLS[name]
    log("INFO", f"Downloading DNN model {name} ...")
    try:
        import urllib.request

        urllib.request.urlretrieve(url, str(path))
    except Exception as e:  # network / blocked
        if path.exists():
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download DNN model '{name}' from {url}: {e}. "
            f"Gender/face detection requires network access on first run."
        ) from e
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"DNN model '{name}' downloaded empty from {url}.")
    log("OK", f"Downloaded DNN model → {path}")
    return path


# Lazily-created singletons so importing this module is cheap.
_FACE_NET: Any = None
_GENDER_NET: Any = None


def _load_face_net():
    global _FACE_NET
    if _FACE_NET is None:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python-headless is required for face detection.")
        proto = _download_if_missing(_FACE_PROTO)
        model = _download_if_missing(_FACE_MODEL)
        _FACE_NET = cv2.dnn.readNetFromTensorflow(str(model), str(proto))
    return _FACE_NET


def _load_gender_net():
    global _GENDER_NET
    if _GENDER_NET is None:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python-headless is required for gender detection.")
        proto = _download_if_missing(_GENDER_PROTO)
        model = _download_if_missing(_GENDER_MODEL)
        _GENDER_NET = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    return _GENDER_NET


def detect_face_in_box(
    frame: Any,
    box: dict[str, Any],
    confidence_threshold: float = 0.5,
) -> dict[str, int] | None:
    """Detect a face inside a person box.

    Returns a face box ``{"x1","y1","x2","y2","conf"}`` in the SAME coordinate
    space as ``box`` (frame pixels), or ``None`` if no face is found / DNN is
    unavailable.

    The face box is what the crop should center on (mouth/eyes region) rather
    than the whole body, so the close-up actually shows the speaker's face.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    x1, y1, x2, y2 = (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
    # Clamp to frame bounds.
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, max(x1 + 1, x2)); y2 = min(h - 1, max(y1 + 1, y2))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    try:
        net = _load_face_net()
    except RuntimeError as e:
        log("WARN", f"Face detector unavailable: {e}")
        return None

    blob = cv2.dnn.blobFromImage(roi, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    net.setInput(blob)
    dets = net.forward()
    best: dict[str, int] | None = None
    best_conf = 0.0
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < confidence_threshold:
            continue
        bx1, by1, bx2, by2 = dets[0, 0, i, 3:7]
        # Map normalized coords back into ROI pixels, then into frame pixels.
        fx1 = x1 + int(bx1 * (x2 - x1))
        fy1 = y1 + int(by1 * (y2 - y1))
        fx2 = x1 + int(bx2 * (x2 - x1))
        fy2 = y1 + int(by2 * (y2 - y1))
        if conf > best_conf:
            best_conf = conf
            best = {"x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2, "conf": round(conf, 3)}
    return best


def classify_gender(face_box: dict[str, int], frame: Any) -> str | None:
    """Classify gender from a face box.

    Returns ``"Male"`` / ``"Female"`` or ``None`` when DNN is unavailable or the
    crop is degenerate.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    if not face_box:
        return None
    x1, y1, x2, y2 = (
        int(face_box["x1"]), int(face_box["y1"]),
        int(face_box["x2"]), int(face_box["y2"]),
    )
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, max(x1 + 1, x2)); y2 = min(h - 1, max(y1 + 1, y2))
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    try:
        net = _load_gender_net()
    except RuntimeError as e:
        log("WARN", f"Gender model unavailable: {e}")
        return None

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 [78.4263377603, 87.7689143744, 114.895847746],
                                 swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    idx = int(preds[0].argmax())
    return _GENDER_LABELS[idx]


def detect_face_and_gender(
    frame: Any,
    box: dict[str, Any],
    confidence_threshold: float = 0.5,
) -> tuple[dict[str, int] | None, str | None]:
    """Convenience: return (face_box, gender) for a person box in one call.

    If no face is found, gender is ``None`` and the caller should fall back to
    the whole person box for framing.
    """
    face = detect_face_in_box(frame, box, confidence_threshold)
    gender = classify_gender(face, frame) if face else None
    return face, gender
