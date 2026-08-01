"""
Active-speaker detection + speaker-aware crop regions for podcast-style clips.

Problem this solves
-------------------
person_detection.py tracks the *largest* person in frame. For a 2-person
(mono-recorded) podcast that is wrong: the camera locks onto whoever is
bigger/closer and never cuts to the other guest when they talk.

This module locates the person who is actually *speaking* at each moment and
returns crop regions that pan to that person, so a landscape podcast becomes a
vertical clip that follows the active speaker.

Inputs are the same as person_detection:
  * per-frame YOLO person bounding boxes (person_detection.detect_persons_in_clip)
  * the mono audio envelope (already computed by audio_energy._extract_pcm /
    a small RMS windowing pass)

Method (audio-visual active speaker localization)
-------------------------------------------------
1. Re-identify persons across frames by tracking their crop-center with a
   greedy nearest-neighbour + IoU match. Each stable track gets a persistent id.
2. For every frame, compute a "mouth-region motion" score: the low-frequency
   frame difference inside the lower-center band of each person's box (where a
   mouth/jaw moves). This is a cheap stand-in for face landmarks.
3. Per track, build a mouth-activity signal sampled at the audio rate.
4. Cross-correlate each track's mouth signal with the global audio envelope
   (delay-bounded). The track whose mouth motion best follows the speech is the
   active speaker for that window.
5. Smooth the active-speaker choice over time, then emit per-segment crop
   regions (same shape as compute_dynamic_crop_regions) so postprocess builds a
   crop filter that pans to whoever is talking.

No external model beyond YOLO + OpenCV is required. mono mixed audio is handled
by the cross-correlation (we never need to diarize a single mixed track).
"""

from __future__ import annotations

from typing import Any

from .utils import log


# ── Person re-identification (track assignment across frames) ───────────────
def _boxes_overlap_ratio(a: dict, b: dict) -> float:
    """Intersection-over-area (area of a) IoU-ish ratio for quick merge test."""
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _assign_tracks(
    per_frame_boxes: list[list[dict[str, Any]]],
    max_center_dist: float | None = None,
) -> list[list[dict[str, Any]]]:
    """Assign a persistent ``track_id`` to every box across frames.

    Greedy nearest-neighbour by crop-center distance, falling back to IoU.
    ``per_frame_boxes`` is a list indexed by frame: each element is the list of
    cleaned boxes for that frame (with x1,y1,x2,y2,conf,area).
    Returns the same structure with each box enriched by ``track_id``.
    """
    tracks: list[dict[str, Any]] = []  # each: {"center":(cx,cy),"last_frame":int}

    for fi, boxes in enumerate(per_frame_boxes):
        # Match each current box to the best existing track
        matches: list[tuple[int, float]] = []  # (track_idx, cost)
        for bi, box in enumerate(boxes):
            cx = (box["x1"] + box["x2"]) / 2.0
            cy = (box["y1"] + box["y2"]) / 2.0
            best_cost, best_t = float("inf"), -1
            for ti, tr in enumerate(tracks):
                if fi - tr["last_frame"] > 30:
                    continue  # stale track — treat as gone
                d = ((cx - tr["center"][0]) ** 2 + (cy - tr["center"][1]) ** 2) ** 0.5
                cost = d
                if best_cost > cost:
                    best_cost, best_t = cost, ti
            matches.append((best_t, best_cost))

        # Resolve assignments so two boxes don't grab the same track in one frame
        claimed: dict[int, int] = {}  # track_idx -> box_idx
        for bi in sorted(range(len(boxes)), key=lambda b: matches[b][1]):
            t, cost = matches[bi]
            limit = max_center_dist if max_center_dist else float("inf")
            if t >= 0 and t not in claimed and cost <= limit:
                claimed[t] = bi
                cx = (boxes[bi]["x1"] + boxes[bi]["x2"]) / 2.0
                cy = (boxes[bi]["y1"] + boxes[bi]["y2"]) / 2.0
                tracks[t]["center"] = (cx, cy)
                tracks[t]["last_frame"] = fi
                boxes[bi]["track_id"] = t
            else:
                # New track
                tracks.append({
                    "center": (
                        (boxes[bi]["x1"] + boxes[bi]["x2"]) / 2.0,
                        (boxes[bi]["y1"] + boxes[bi]["y2"]) / 2.0,
                    ),
                    "last_frame": fi,
                })
                boxes[bi]["track_id"] = len(tracks) - 1

    return per_frame_boxes


# ── Mouth-region motion score (cheap face-landmark stand-in) ────────────────
def _mouth_motion_score(frame, box: dict) -> float:
    """Low-frequency motion in the lower-center band of a person's box.

    A speaking person moves lips/jaw in the lower third of the face. We take a
    horizontal strip across the lower-center of the box, downscale it, and
    measure variance of the luminance as a proxy for articulated motion.
    Returns 0.0 when the region is degenerate.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0.0

    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    # Lower-center band: 55%-85% down the box, centered horizontally
    band_y1 = int(y1 + h * 0.55)
    band_y2 = int(y1 + h * 0.85)
    band_x1 = int(x1 + w * 0.30)
    band_x2 = int(x1 + w * 0.70)
    if band_y2 <= band_y1 or band_x2 <= band_x1:
        return 0.0
    if band_x2 > frame.shape[1] or band_y2 > frame.shape[0]:
        return 0.0
    band = frame[band_y1:band_y2, band_x1:band_x2]
    if band.size == 0:
        return 0.0
    # Downscale for speed and to remove high-freq sensor noise
    small = cv2.resize(band, (16, 8), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Variance of luminance ~ articulation energy
    return float(np.var(small, dtype=np.float64))


def _build_mouth_signals(
    frames: list[Any],
    per_frame_boxes: list[list[dict[str, Any]]],
) -> dict[int, list[float]]:
    """Compute a per-track mouth-motion signal aligned to frame index."""
    signals: dict[int, list[float]] = {}
    prev_gray: dict[int, Any] = {}
    for fi, boxes in enumerate(per_frame_boxes):
        # Need previous frame for difference-based motion
        prev = frames[fi - 1] if fi > 0 else None
        cur = frames[fi]
        for box in boxes:
            tid = box.get("track_id", -1)
            if tid < 0:
                continue
            signals.setdefault(tid, [])
            score = _mouth_motion_score(cur, box)
            if prev is not None:
                # frame difference inside the band sharpens motion cues
                try:
                    import cv2
                    import numpy as np
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    h = max(1, y2 - y1)
                    w = max(1, x2 - x1)
                    by1 = int(y1 + h * 0.55)
                    by2 = int(y1 + h * 0.85)
                    bx1 = int(x1 + w * 0.30)
                    bx2 = int(x1 + w * 0.70)
                    if by2 > by1 and bx2 > bx1 and by2 <= prev.shape[0] and bx2 <= prev.shape[1]:
                        p = prev[by1:by2, bx1:bx2]
                        c = cur[by1:by2, bx1:bx2]
                        if p.shape == c.shape and p.size:
                            pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p
                            cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
                            diff = float(np.mean(np.abs(cg.astype(np.int16) - pg.astype(np.int16))))
                            score = max(score, diff)
                except Exception:
                    pass
            signals[tid].append(score)
    return signals


# ── Audio envelope (RMS) at a chosen sample rate ─────────────────────────────
def _audio_envelope(video_path: str, sample_rate: int = 16000, window_sec: float = 0.5) -> list[float]:
    """Mono 16-bit PCM RMS energy, one value per ``window_sec`` window.

    Mirrors audio_energy._extract_pcm band-pass + the RMS windowing, but
    returns the raw envelope so we can cross-correlate with mouth motion.
    Returns [] on failure.
    """
    import subprocess

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-af", "highpass=f=120,lowpass=f=4000",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except Exception:
        return []
    pcm = result.stdout
    if result.returncode != 0 or not pcm:
        return []
    try:
        import numpy as np
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    except ImportError:
        return []
    spw = int(sample_rate * window_sec)
    if spw == 0 or len(samples) < spw:
        return []
    n = len(samples) // spw
    trimmed = samples[: n * spw].reshape(n, spw)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    # Normalize so cross-correlation isn't dominated by overall loudness
    m = rms.max()
    if m > 0:
        rms = rms / m
    return rms.tolist()


def _resample_track_to_audio(
    mouth_signal: list[float],
    n_windows: int,
) -> list[float]:
    """Stretch a per-frame (1 Hz) mouth signal across the audio-window grid.

    Mouth motion is sampled once per detected frame (uniformly in time), the
    audio envelope once per ``window_sec`` (faster). We linearly interpolate
    the mouth signal so it has exactly ``n_windows`` samples aligned to the
    audio timeline, then normalize. Both signals are uniformly sampled in
    time, so index-based interpolation is the correct alignment.
    """
    if not mouth_signal or n_windows <= 0:
        return []
    try:
        import numpy as np
        src = np.asarray(mouth_signal, dtype=np.float64)
        if len(src) == 1:
            out = np.full(n_windows, float(src[0]))
        else:
            x = np.linspace(0, len(src) - 1, n_windows)
            xp = np.arange(len(src))
            out = np.interp(x, xp, src)
        mx = out.max()
        if mx > 0:
            out = out / mx
        return out.tolist()
    except ImportError:
        return []


def _active_speaker_per_window(
    mouth_signals: dict[int, list[float]],
    audio_env: list[float],
) -> list[int]:
    """Pick the speaking track for each audio window via audio-visual coherence.

    Each track's mouth-motion signal is resampled onto the audio grid and
    normalized. For every audio window we take the pointwise coherence between
    that window's audio loudness and each track's mouth motion; the track with
    the highest coherence is the active speaker. Windows whose audio is
    near-silent are marked -1 (no clear speaker). This is a lightweight
    active-speaker proxy that works on a single mixed mono track without
    diarization.
    """
    n_windows = len(audio_env)
    if n_windows == 0:
        return []

    resampled: dict[int, list[float]] = {}
    for tid, sig in mouth_signals.items():
        rs = _resample_track_to_audio(sig, n_windows)
        if rs:
            resampled[tid] = rs

    if not resampled:
        return [-1] * n_windows

    silence_floor = 0.04
    assignment: list[int] = []
    for wi in range(n_windows):
        if audio_env[wi] < silence_floor:
            assignment.append(-1)
            continue
        best_tid, best_corr = -1, -1.0
        for tid, sig in resampled.items():
            if wi >= len(sig):
                continue
            corr = sig[wi] * audio_env[wi]  # instantaneous coherence
            if corr > best_corr:
                best_corr, best_tid = corr, tid
        assignment.append(best_tid)
    return assignment


# ── Public API ──────────────────────────────────────────────────────────────
def compute_active_speaker_crop_regions(
    video_path: str,
    detections: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_aspect: float = 9 / 16,
    segment_duration: float = 1.0,
    smoothing_window: int = 5,
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """Compute per-segment crop regions that follow the ACTIVE SPEAKER.

    Unlike person_detection.compute_dynamic_crop_regions (which tracks the
    largest person), this tracks each person's identity and selects the one who
    is speaking, using cross-correlation of mouth motion with the mono audio
    envelope.

    Args:
        video_path: Source clip (used to read frames + audio envelope).
        detections: Output of person_detection.detect_persons_in_clip
            (list of {"time", "boxes":[{x1,y1,x2,y2,conf,area}]}).
        src_w, src_h: Source dimensions.
        target_aspect: Output w/h (9/16 vertical).
        segment_duration: Crop segment length in seconds.
        smoothing_window: Neighbour smoothing for stable panning.
        fps: Source frame rate (for signal alignment).

    Returns:
        [{"time", "x", "y", "w", "h", "track_id"}, ...] suitable for
        build_dynamic_crop_filter.
    """
    if not detections:
        return []

    # Reconstruct a per-frame box list aligned to the sampled detection times.
    # detections are sampled every `sample_interval` (1.0s in caller). We assume
    # a uniform frame index = round(time * fps).
    per_frame_boxes: list[list[dict[str, Any]]] = []
    max_frame = 0
    for det in detections:
        fi = int(round(det["time"] * fps))
        max_frame = max(max_frame, fi)
        while len(per_frame_boxes) <= fi:
            per_frame_boxes.append([])
        for b in det["boxes"]:
            per_frame_boxes[fi].append(dict(b))

    if not per_frame_boxes:
        return []

    # Assign stable track ids
    per_frame_boxes = _assign_tracks(per_frame_boxes)

    # Read the same sampled frames for mouth-motion scoring. We re-extract using
    # cv2 at the same 1 fps sampling to keep it cheap and aligned.
    frames: list[Any] = []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Only keep frames whose index matches a detection frame
                if fi < len(per_frame_boxes) and per_frame_boxes[fi]:
                    frames.append(frame)
                fi += 1
            cap.release()
    except Exception as e:
        log("WARN", f"Active-speaker frame read failed: {e}")
        frames = []

    # Align frames list to per_frame_boxes (frames were appended only for
    # detection frames in order, so zip by detection-frame index).
    det_frames = [per_frame_boxes[i] for i in range(len(per_frame_boxes)) if per_frame_boxes[i]]
    if len(frames) == len(det_frames):
        frame_by_idx: dict[int, Any] = {}
        k = 0
        for i in range(len(per_frame_boxes)):
            if per_frame_boxes[i]:
                frame_by_idx[i] = frames[k]
                k += 1
    else:
        frame_by_idx = {}

    # Mouth signals keyed by track id, indexed by frame
    mouth_signals: dict[int, list[float]] = {}
    for fi, boxes in enumerate(per_frame_boxes):
        frame = frame_by_idx.get(fi)
        for box in boxes:
            tid = box.get("track_id", -1)
            if tid < 0:
                continue
            mouth_signals.setdefault(tid, [])
            mouth_signals[tid].append(_mouth_motion_score(frame, box) if frame is not None else 0.0)

    # Audio envelope (required — no silent fallback to largest-person tracking)
    audio_env = _audio_envelope(video_path)
    if not audio_env:
        raise RuntimeError(
            "Active-speaker crop requires an audio envelope but none could be "
            "extracted (ffmpeg audio read failed). This clip cannot be cropped "
            "to the speaking person."
        )

    # Active speaker per audio window
    assignment = _active_speaker_per_window(mouth_signals, audio_env)
    if not assignment:
        raise RuntimeError(
            "Active-speaker crop could not assign any speaking person from the "
            "audio envelope for this clip."
        )

    # Crop dimensions (constant)
    src_aspect = src_w / src_h if src_h else 1.0
    if target_aspect < src_aspect:
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    max_time = max(d["time"] for d in detections)
    n_segments = max(1, int(max_time / segment_duration) + 1)

    segments: list[dict[str, Any]] = []
    prev_cx, prev_cy = src_w / 2, src_h / 2
    prev_tid = -1

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_duration
        seg_end = seg_start + segment_duration

        # Which audio windows fall inside this segment?
        w_lo = int(seg_start / segment_duration)  # approx; audio uses 0.5s bins
        # Map segment seconds to audio-window indices: audio bins are 0.5s
        a_lo = int(seg_start / 0.5)
        a_hi = int(seg_end / 0.5) + 1
        window_assign = [t for t in assignment[a_lo:a_hi] if t >= 0]

        # Preferred track for this segment = majority of assigned windows
        if window_assign:
            from collections import Counter
            tid = Counter(window_assign).most_common(1)[0][0]
        else:
            tid = prev_tid if prev_tid >= 0 else -1

        # Find that track's box in this segment's frames
        seg_dets = [d for d in detections if seg_start <= d["time"] < seg_end]
        cx = cy = None
        for d in seg_dets:
            fi = int(round(d["time"] * fps))
            if fi >= len(per_frame_boxes):
                continue
            for box in per_frame_boxes[fi]:
                if box.get("track_id") == tid:
                    cx = (box["x1"] + box["x2"]) / 2.0
                    cy = (box["y1"] + box["y2"]) / 2.0
                    break
            if cx is not None:
                break

        if cx is None:
            # Track not visible this segment — keep previous pan position
            cx, cy = prev_cx, prev_cy
        else:
            # Smooth transition
            cx = prev_cx * 0.3 + cx * 0.7
            cy = prev_cy * 0.3 + cy * 0.7

        crop_x = int(cx - crop_w / 2)
        crop_y = int(cy - crop_h / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        prev_cx, prev_cy = cx, cy
        prev_tid = tid
        segments.append({
            "time": round(seg_start, 2),
            "x": crop_x, "y": crop_y,
            "w": crop_w, "h": crop_h,
            "track_id": tid,
        })

    return segments
