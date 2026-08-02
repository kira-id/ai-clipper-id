"""
Person detection and close-up cropping using YOLO.

Detects persons in video frames and generates crop coordinates
to create close-up shots — essential for converting horizontal
video to vertical shorts/reels format.
"""

import subprocess
import json
from pathlib import Path
from typing import Any

from .utils import get_ffmpeg, get_ffprobe, log


def _get_video_dimensions(video_path: str) -> tuple[int, int, float]:
    """Get video width, height, and fps."""
    try:
        result = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        w = int(stream["width"])
        h = int(stream["height"])
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)
        return w, h, fps
    except Exception as e:
        log("WARN", f"Could not get video dimensions: {e}")
        return 0, 0, 30.0


def _extract_frames_with_ffmpeg(
    video_path: str,
    sample_interval: float = 1.0,
) -> list[tuple[float, bytes]]:
    """Extract frames using FFmpeg and return as raw bytes for OpenCV.
    
    More robust for corrupted videos than cv2.VideoCapture.
    
    Returns list of (timestamp, frame_bytes) tuples.
    """
    try:
        result = subprocess.run(
            [
                get_ffmpeg(), "-i", video_path,
                "-vf", f"fps=1/{sample_interval}",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-"
            ],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            return []
        
        import cv2
        import numpy as np
        
        w, h, _ = _get_video_dimensions(video_path)
        if w == 0 or h == 0:
            return []
        
        frame_bytes = result.stdout
        frame_size = w * h * 3
        
        frames = []
        timestamp = 0.0
        for i in range(0, len(frame_bytes), frame_size):
            if i + frame_size > len(frame_bytes):
                break
            frame_data = frame_bytes[i:i+frame_size]
            frames.append((timestamp, frame_data, w, h))
            timestamp += sample_interval
        
        return frames
    except Exception as e:
        log("DEBUG", f"FFmpeg frame extraction failed: {e}")
        return []


def detect_persons_in_clip(
    video_path: str,
    sample_interval: float = 1.0,
    confidence_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect persons in video frames using YOLO.

    Samples frames at regular intervals and returns bounding boxes
    for detected persons. Tries cv2.VideoCapture first, falls back
    to FFmpeg extraction for corrupted videos.

    Args:
        video_path: Path to the video clip.
        sample_interval: Seconds between sampled frames.
        confidence_threshold: Minimum YOLO confidence to accept detection.

    Returns:
        List of detections: [{"time": float, "boxes": [{"x1","y1","x2","y2","conf"}]}]
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log("WARN", "ultralytics not installed. pip install ultralytics — skipping person detection")
        return []

    import cv2
    import numpy as np

    w, h, fps = _get_video_dimensions(video_path)
    if w == 0 or h == 0:
        return []

    # Load YOLO model (cached after first load)
    model = YOLO("yolov8n.pt")  # nano model — fast, good enough for person detection

    detections: list[dict[str, Any]] = []
    
    # Try cv2.VideoCapture first (faster for normal videos)
    frames_to_process = []
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        log("DEBUG", f"Using cv2.VideoCapture for person detection")
        frame_interval = int(fps * sample_interval) if fps > 0 else 1
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / fps if fps > 0 else 0
                frames_to_process.append((time_sec, frame))
            frame_idx += 1
        cap.release()
    else:
        # cv2.VideoCapture failed, try FFmpeg extraction
        log("DEBUG", f"cv2.VideoCapture failed, trying FFmpeg frame extraction")
        ffmpeg_frames = _extract_frames_with_ffmpeg(video_path, sample_interval)
        if ffmpeg_frames:
            for timestamp, frame_bytes, w_frame, h_frame in ffmpeg_frames:
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h_frame, w_frame, 3))
                frames_to_process.append((timestamp, frame))
    
    if not frames_to_process:
        log("WARN", f"Could not extract frames from video: {video_path}")
        return []
    
    # Run YOLO on extracted frames
    for time_sec, frame in frames_to_process:
        results = model(frame, verbose=False, classes=[0])
        
        boxes = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "conf": round(conf, 3),
                        "area": int((x2 - x1) * (y2 - y1)),
                    })
        
        if boxes:
            # Sort by area descending (largest person = likely speaker)
            boxes.sort(key=lambda b: -b["area"])
            detections.append({"time": round(time_sec, 2), "boxes": boxes})
    
    return detections


def compute_crop_region(
    detections: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_aspect: float = 9 / 16,
    padding_pct: float = 0.25,
    smoothing_window: int = 5,
) -> dict[str, int] | None:
    """Compute a stable crop region that keeps the main person centered.

    Algorithm:
    1. For each frame, find the largest person (likely speaker)
    2. Compute the center of that person across all frames
    3. Smooth the center position to avoid jittery panning
    4. Calculate crop box that centers on the person with padding

    Args:
        detections: Output from detect_persons_in_clip
        src_w: Source video width
        src_h: Source video height
        target_aspect: Target width/height ratio (9/16 for vertical)
        padding_pct: Extra padding around person (fraction of crop size)
        smoothing_window: Frames to average for smooth tracking

    Returns:
        {"x": int, "y": int, "w": int, "h": int} or None if no persons found
    """
    if not detections:
        return None

    # Collect center-x positions of the largest person per frame
    centers_x: list[float] = []
    centers_y: list[float] = []
    for det in detections:
        box = det["boxes"][0]  # largest person
        cx = (box["x1"] + box["x2"]) / 2.0
        cy = (box["y1"] + box["y2"]) / 2.0
        centers_x.append(cx)
        centers_y.append(cy)

    # Smooth center positions
    def _smooth(values: list[float], window: int) -> list[float]:
        if len(values) <= window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(sum(values[start:end]) / (end - start))
        return smoothed

    smooth_cx = _smooth(centers_x, smoothing_window)
    smooth_cy = _smooth(centers_y, smoothing_window)

    # Use median center as the stable crop point
    avg_cx = sorted(smooth_cx)[len(smooth_cx) // 2]
    avg_cy = sorted(smooth_cy)[len(smooth_cy) // 2]

    # Calculate crop dimensions
    # For horizontal→vertical: crop width is narrow, height is full
    src_aspect = src_w / src_h
    if target_aspect < src_aspect:
        # Source is wider than target → crop width (most common: landscape→portrait)
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)
    else:
        # Source is taller than target → crop height
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)

    # Ensure even dimensions (required by libx264)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    # Center crop on the person
    crop_x = int(avg_cx - crop_w / 2)
    crop_y = int(avg_cy - crop_h / 2)

    # Clamp to video bounds
    crop_x = max(0, min(crop_x, src_w - crop_w))
    crop_y = max(0, min(crop_y, src_h - crop_h))

    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def compute_dynamic_crop_regions(
    detections: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_aspect: float = 9 / 16,
    segment_duration: float = 3.0,
    smoothing_window: int = 5,
) -> list[dict[str, Any]]:
    """Compute per-segment crop regions for smooth person tracking.

    Instead of a single static crop, divides the clip into segments
    and computes crop position for each, enabling smooth panning
    to follow the active speaker.

    Returns:
        [{"time": float, "x": int, "y": int, "w": int, "h": int}, ...]
    """
    if not detections:
        return []

    # Group detections by time segment
    max_time = max(d["time"] for d in detections)
    n_segments = max(1, int(max_time / segment_duration) + 1)

    # Calculate crop dimensions (constant for all segments)
    src_aspect = src_w / src_h
    if target_aspect < src_aspect:
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    segments: list[dict[str, Any]] = []
    prev_cx, prev_cy = src_w / 2, src_h / 2  # default center

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_duration
        seg_end = seg_start + segment_duration

        # Find detections in this segment
        seg_dets = [d for d in detections if seg_start <= d["time"] < seg_end]

        if seg_dets:
            # Use median center of largest person
            cxs = [(d["boxes"][0]["x1"] + d["boxes"][0]["x2"]) / 2 for d in seg_dets]
            cys = [(d["boxes"][0]["y1"] + d["boxes"][0]["y2"]) / 2 for d in seg_dets]
            cx = sorted(cxs)[len(cxs) // 2]
            cy = sorted(cys)[len(cys) // 2]
            # Smooth transition from previous segment
            cx = prev_cx * 0.3 + cx * 0.7
            cy = prev_cy * 0.3 + cy * 0.7
            prev_cx, prev_cy = cx, cy
        else:
            cx, cy = prev_cx, prev_cy

        crop_x = int(cx - crop_w / 2)
        crop_y = int(cy - crop_h / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        segments.append({
            "time": round(seg_start, 2),
            "x": crop_x, "y": crop_y,
            "w": crop_w, "h": crop_h,
        })

    return segments


def build_crop_filter(
    crop_region: dict[str, int] | None,
    target_w: int = 1080,
    target_h: int = 1920,
) -> str | None:
    """Build FFmpeg crop+scale filter string for static crop.

    Args:
        crop_region: Output from compute_crop_region
        target_w: Output width
        target_h: Output height

    Returns:
        FFmpeg filter string like "crop=608:1080:336:0,scale=1080:1920"
        or None if no crop needed
    """
    if not crop_region:
        return None

    x, y, w, h = crop_region["x"], crop_region["y"], crop_region["w"], crop_region["h"]
    return f"crop={w}:{h}:{x}:{y},scale={target_w}:{target_h}:flags=lanczos"


def build_dynamic_crop_filter(
    crop_regions: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
) -> str:
    """Build FFmpeg filter for dynamic crop with smooth pan/zoom transitions.

    Uses FFmpeg's zoompan filter to create smooth camera movement that
    follows the detected person across frames.

    Args:
        crop_regions: List of per-segment crop regions from compute_dynamic_crop_regions
        src_w: Source video width
        src_h: Source video height
        target_w: Output width
        target_h: Output height

    Returns:
        FFmpeg filter string with zoompan for smooth tracking
    """
    if not crop_regions:
        # Fallback to center crop if no regions
        return f"crop={target_w}:{target_h}:{(src_w - target_w) // 2}:{(src_h - target_h) // 2},scale={target_w}:{target_h}"

    # Build keyframe expressions for zoompan filter
    # zoompan syntax: zoompan=z=zoom:x=x_pos:y=y_pos:d=duration:primary=1:interp=linear

    # Calculate zoom level (crop window relative to output size)
    # For landscape→portrait: we crop a vertical slice from the landscape frame
    src_aspect = src_w / src_h
    target_aspect = target_w / target_h

    if target_aspect < src_aspect:
        # Source is wider than target → crop width
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)
    else:
        # Source is taller than target → crop height
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)

    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    # Zoom factor: how much we're zooming in
    zoom_x = crop_w / src_w
    zoom_y = crop_h / src_h
    zoom = min(zoom_x, zoom_y)

    # Build time-keyed keyframes of the crop window's TOP-LEFT corner. We drive
    # ffmpeg's `crop` filter directly (x/y accept per-frame expressions) rather
    # than `zoompan`. zoompan re-derives its own zoom/scale state and its
    # `primary`/`interp` options are not accepted by all ffmpeg builds, which
    # made the previous filter graph fail to parse outright.
    max_x = max(0, src_w - crop_w)
    max_y = max(0, src_h - crop_h)

    keys_x: list[tuple[float, float]] = []
    keys_y: list[tuple[float, float]] = []
    keys_track: list[int] = []
    for region in crop_regions:
        t = float(region["time"])
        keys_x.append((t, min(max(0.0, float(region["x"])), float(max_x))))
        keys_y.append((t, min(max(0.0, float(region["y"])), float(max_y))))
        keys_track.append(int(region.get("track_id", -1)))

    x_expr = _build_interpolation_expr(keys_x, keys_track)
    y_expr = _build_interpolation_expr(keys_y, keys_track)

    # Commas and single quotes are filter-graph separators. Escape every comma
    # inside the expressions so the whole `if(...)` tree survives ffmpeg's
    # two-stage (graph, then filter-option) parser.
    x_expr = x_expr.replace(",", "\\,")
    y_expr = y_expr.replace(",", "\\,")

    return (
        f"crop={crop_w}:{crop_h}:x={x_expr}:y={y_expr},"
        f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
    )


def _build_interpolation_expr(
    keyframes: list[tuple[float, float]],
    tracks: list[int] | None = None,
) -> str:
    """Build an FFmpeg expression that pans to each keyframe.

    Uses the timeline variable ``t`` (seconds) rather than the frame counter
    ``n`` so the result is correct at any frame rate. The previous version
    hardcoded 30fps, so tracking drifted badly on 25/50/60fps sources.

    When ``tracks`` is provided, the expression HOLDS (step) the value across a
    speaker change (two adjacent keyframes with different ``track_id``) instead
    of linearly interpolating. That produces a hard CUT / jump between people —
    a smooth linear pan between two speakers would sweep across the scene and
    make viewers dizzy. Within the same speaker (same track_id) it still
    interpolates smoothly so a single talking head stays stable.

    Args:
        keyframes: List of (time_seconds, value) tuples
        tracks: Optional parallel list of track ids (one per keyframe)

    Returns:
        FFmpeg expression string using nested if() for piecewise motion.
    """
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.2f}"

    keyframes = sorted(keyframes, key=lambda kv: kv[0])
    if tracks is None:
        tracks = [-1] * len(keyframes)
    # Legacy callers (largest-person crop) pass no track ids (all -1). In that
    # case behave exactly as before: always interpolate smoothly.
    _has_track_info = any(t >= 0 for t in tracks)

    exprs: list[str] = []
    for i in range(len(keyframes) - 1):
        t1, v1 = keyframes[i]
        t2, v2 = keyframes[i + 1]
        same_speaker = (
            _has_track_info
            and (tracks[i] == tracks[i + 1])
            and tracks[i] >= 0
        )

        if t2 <= t1:
            # Duplicate / non-monotonic timestamp — hold the earlier value.
            exprs.append(f"if(lte(t,{t1:.3f}),{v1:.2f},")
            continue

        if not _has_track_info:
            # No speaker/track info (legacy largest-person crop): always
            # interpolate smoothly — there is only one tracked subject, so a
            # smooth pan is correct and desirable.
            slope = (v2 - v1) / (t2 - t1)
            interp = f"{v1:.2f}+({slope:.4f})*(t-{t1:.3f})"
            exprs.append(f"if(lte(t,{t2:.3f}),{interp},")
        elif same_speaker:
            # Smooth pan within the same speaker.
            slope = (v2 - v1) / (t2 - t1)
            interp = f"{v1:.2f}+({slope:.4f})*(t-{t1:.3f})"
            exprs.append(f"if(lte(t,{t2:.3f}),{interp},")
        else:
            # Speaker change: HARD CUT. Hold the OLD speaker's position (v1)
            # for the whole interval up to t2, then the next branch snaps to the
            # new speaker (v2) at t >= t2. No slide across the scene.
            exprs.append(f"if(lte(t,{t2:.3f}),{v1:.2f},")

    # Past the final keyframe: hold the last value.
    exprs.append(f"{keyframes[-1][1]:.2f}")

    return "".join(exprs) + ")" * (len(exprs) - 1)


def build_split_screen_filter(
    detections: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    face_ratio: float = 0.4,
    input_label: str = "[0:v]",
    output_label: str = "[vout]",
) -> str:
    """Build FFmpeg filter for split-screen: face close-up on top, gameplay on bottom.

    For landscape gaming/reaction videos converted to vertical format.
    Top section shows a close-up crop of the detected person (webcam/face).
    Bottom section shows the center of the frame (gameplay).

    Args:
        detections: Person detection results from detect_persons_in_clip
        src_w: Source video width
        src_h: Source video height
        target_w: Output width (default 1080)
        target_h: Output height (default 1920)
        face_ratio: Fraction of output height for the face section (default 0.4)
        input_label: FFmpeg input stream label
        output_label: FFmpeg output stream label

    Returns:
        FFmpeg filter_complex fragment with split, crop, scale, and vstack
    """
    face_out_h = int(target_h * face_ratio)
    face_out_h -= face_out_h % 2  # ensure even
    game_out_h = target_h - face_out_h
    game_out_h -= game_out_h % 2  # ensure even
    # Adjust face_out_h so both sum exactly to target_h
    face_out_h = target_h - game_out_h

    # --- Face crop region ---
    # Use median bounding box from detections for stable positioning
    all_boxes = [det["boxes"][0] for det in detections if det.get("boxes")]

    if all_boxes:
        med_x1 = sorted(b["x1"] for b in all_boxes)[len(all_boxes) // 2]
        med_y1 = sorted(b["y1"] for b in all_boxes)[len(all_boxes) // 2]
        med_x2 = sorted(b["x2"] for b in all_boxes)[len(all_boxes) // 2]
        med_y2 = sorted(b["y2"] for b in all_boxes)[len(all_boxes) // 2]
    else:
        # Fallback: bottom-left quadrant (common webcam position)
        med_x1, med_y1 = 0, src_h // 2
        med_x2, med_y2 = src_w // 3, src_h

    person_cx = (med_x1 + med_x2) / 2
    person_cy = (med_y1 + med_y2) / 2
    person_w = max(med_x2 - med_x1, 1)
    person_h = max(med_y2 - med_y1, 1)

    # Face crop aspect ratio must match target_w / face_out_h
    face_aspect = target_w / face_out_h

    # Expand person bbox with padding, then adjust to match face_aspect
    padding = 0.3
    padded_w = person_w * (1 + padding)
    padded_h = person_h * (1 + padding)

    if padded_w / padded_h > face_aspect:
        face_crop_w = int(padded_w)
        face_crop_h = int(face_crop_w / face_aspect)
    else:
        face_crop_h = int(padded_h)
        face_crop_w = int(face_crop_h * face_aspect)

    # Minimum crop size (at least 20% of source width)
    min_w = int(src_w * 0.2)
    if face_crop_w < min_w:
        face_crop_w = min_w
        face_crop_h = int(face_crop_w / face_aspect)

    # Clamp to source, then ensure even
    face_crop_w = min(face_crop_w, src_w)
    face_crop_h = min(face_crop_h, src_h)
    face_crop_w -= face_crop_w % 2
    face_crop_h -= face_crop_h % 2

    # Ensure minimum dimensions (at least 2x2 after rounding)
    face_crop_w = max(face_crop_w, 2)
    face_crop_h = max(face_crop_h, 2)

    # Center on person, clamp to bounds
    face_x = int(person_cx - face_crop_w / 2)
    face_y = int(person_cy - face_crop_h / 2)
    face_x = max(0, min(face_x, src_w - face_crop_w))
    face_y = max(0, min(face_y, src_h - face_crop_h))

    # --- Gameplay crop region ---
    # Center of frame horizontally, full height vertically
    game_aspect = target_w / game_out_h

    if src_w / src_h > game_aspect:
        # Source wider than game aspect — crop width from center
        game_crop_h = src_h
        game_crop_w = int(game_crop_h * game_aspect)
    else:
        # Source taller — crop height from center
        game_crop_w = src_w
        game_crop_h = int(game_crop_w / game_aspect)

    game_crop_w -= game_crop_w % 2
    game_crop_h -= game_crop_h % 2

    game_x = (src_w - game_crop_w) // 2
    game_y = (src_h - game_crop_h) // 2

    log("DEBUG", f"Split-screen: face crop {face_crop_w}x{face_crop_h}+{face_x}+{face_y} → {target_w}x{face_out_h}")
    log("DEBUG", f"Split-screen: game crop {game_crop_w}x{game_crop_h}+{game_x}+{game_y} → {target_w}x{game_out_h}")

    return (
        f"{input_label}split=2[_face_in][_game_in];"
        f"[_face_in]crop={face_crop_w}:{face_crop_h}:{face_x}:{face_y},"
        f"scale={target_w}:{face_out_h}:flags=lanczos[_face];"
        f"[_game_in]crop={game_crop_w}:{game_crop_h}:{game_x}:{game_y},"
        f"scale={target_w}:{game_out_h}:flags=lanczos[_game];"
        f"[_face][_game]vstack=inputs=2{output_label}"
    )


def needs_crop(src_w: int, src_h: int, target_aspect: str = "vertical") -> bool:
    """Check if the source video needs cropping for the target format.

    Args:
        src_w: Source width
        src_h: Source height
        target_aspect: "vertical" (9:16), "horizontal" (16:9), or "square" (1:1)
    """
    src_aspect = src_w / src_h if src_h > 0 else 1.0

    if target_aspect == "vertical":
        return src_aspect > 0.7  # source is landscape or square → needs vertical crop
    elif target_aspect == "horizontal":
        return src_aspect < 1.3  # source is portrait or square → needs horizontal crop
    elif target_aspect == "square":
        return abs(src_aspect - 1.0) > 0.15
    return False
