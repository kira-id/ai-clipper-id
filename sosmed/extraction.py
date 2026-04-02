"""
Clip extraction: uses ffmpeg to extract video segments.
"""

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import get_defaults
from .utils import get_ffmpeg, get_ffprobe, log


def _is_videotoolbox_available() -> bool:
    """Check if h264_videotoolbox encoder is available (macOS hardware acceleration)."""
    try:
        result = subprocess.run(
            [get_ffmpeg(), "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_videotoolbox" in result.stdout
    except Exception:
        return False


def _get_video_duration(path: str) -> float | None:
    """Get video duration in seconds via ffprobe."""
    try:
        out = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _extract_one(
    video_path: str,
    clip: dict[str, Any],
    output_dir: Path,
    *,
    padding: float = 0.35,
    encoding_preset: str | None = None,
    encoding_crf: int | None = None,
) -> str:
    """Extract a single clip with ffmpeg (frame-accurate). Returns output file path.

    Historically this function used ``-c:v copy`` and relied on "output
    seeking" (``-ss`` after ``-i``) plus ``-copyts``.  That worked for
    keyframe-aligned cuts but produced files whose internal start timestamp
    matched the nearest preceding I-frame.  When the first keyframe was
    several seconds past the requested start the resulting clip would appear
    to freeze for those seconds and even report a non-zero ``start`` value
    when inspected with ffprobe.

    To guarantee accurate trimming regardless of keyframe placement we now
    re-encode the video with libx264.  It's a little slower, but ensures the
    output begins at the correct frame and the player doesn't show a
    multi-second freeze.  Padding is still applied to avoid cutting into
    speech.
    
    Encoding quality is controlled via encoding_preset and encoding_crf parameters
    (or config defaults): preset affects speed, CRF affects quality.
    """
    start = max(0.0, clip["start"] - padding)
    end = clip["end"] + padding
    duration = end - start

    safe = re.sub(r"[^\w\s-]", "", clip.get("title", f"clip_{clip['rank']}"))
    safe = re.sub(r"\s+", "_", safe)[:50]
    out_path = output_dir / f"rank{clip['rank']:02d}_{safe}.mp4"
    # record raw filename in metadata (postprocess can override later)
    clip["filename"] = out_path.name

    # Load encoding settings from config if not provided
    if encoding_preset is None or encoding_crf is None:
        defaults = get_defaults()
        if encoding_preset is None:
            encoding_preset = defaults.get("encoding_preset", "veryfast")
        if encoding_crf is None:
            encoding_crf = defaults.get("encoding_crf", 23)

    # -i first, then -ss/-t  → output seeking. we re-encode rather than copy
    # so that trimming is frame-accurate even when the desired start doesn't
    # coincide with a keyframe. previously we used -c:v copy which caused
    # clips to begin at the nearest preceding keyframe (often several seconds
    # early) and the file header retained the original timestamp, leading to
    # what looked like a freeze for the first 2–3 seconds.
    base_cmd = [
        get_ffmpeg(), "-y", "-hide_banner",
        "-i", video_path,
        "-ss", f"{start:.3f}",
        "-t",  f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
    ]
    # encode with libx264 so ffmpeg will decode up to the requested start
    # and the resulting clip has correct timing/stamps.
    # Quality: CRF 23 (good balance for social media), preset for speed
    
    # Check hardware acceleration setting from config
    defaults = get_defaults()
    use_hwaccel = defaults.get("hwaccel", True)
    
    # Use hardware acceleration on macOS (VideoToolbox) for faster encoding
    # Fall back to libx264 if hwaccel disabled or not available
    if use_hwaccel and _is_videotoolbox_available():
        # VideoToolbox: very fast, good quality for social media
        # Note: h264_videotoolbox doesn't support CRF, use quality/speed instead
        encode_args = ["-c:v", "h264_videotoolbox", "-quality", "speed"]
        log("DEBUG", f"Clip #{clip['rank']}: using VideoToolbox hardware acceleration")
    else:
        # libx264 with configurable preset/CRF
        encode_args = ["-c:v", "libx264", "-preset", encoding_preset, "-crf", str(encoding_crf)]
    
    audio_flags = ["-c:a", "aac", "-b:a", "192k"]  # Higher bitrate for better audio
    output_flags = ["-movflags", "+faststart", "-loglevel", "error", str(out_path)]

    cmd = base_cmd + encode_args + audio_flags + output_flags
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log("DEBUG", f"Clip #{clip['rank']} encoded")
        return str(out_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_detail = e.stderr[:150] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        error_msg = error_detail
        raise RuntimeError(f"Encoding failed for clip #{clip['rank']}: {error_msg}")


def extract_clips(
    video_path: str,
    clips: list[dict[str, Any]],
    output_dir: Path,
    max_workers: int = 4,
    encoding_preset: str | None = None,
    encoding_crf: int | None = None,
) -> list[str]:
    """Extract all clips in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log("INFO", f"Extracting {len(clips)} clips → {output_dir}/ ({max_workers} workers)")

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_extract_one, video_path, c, output_dir, encoding_preset=encoding_preset, encoding_crf=encoding_crf): c
            for c in clips
        }
        for fut in as_completed(futs):
            clip = futs[fut]
            try:
                out = fut.result()
                mb = os.path.getsize(out) / 1_048_576
                log("OK", f"  #{clip['rank']:>2} {clip['title'][:40]:<40}  "
                          f"{Path(out).name} ({mb:.1f} MB)")
                results.append(out)
            except Exception as exc:
                log("ERROR", f"  #{clip['rank']:>2} failed: {exc}")

    return sorted(results)
