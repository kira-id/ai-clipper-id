"""
Background music selection and mixing for video clips.

Uses a curated library of royalty-free music organized by mood/category.
LLM matches clip content to appropriate background music.
Music is mixed at very low volume as ambient vibe, with a fade-out at the
end of the clip.
"""

import hashlib
import json
import subprocess
import tempfile
from array import array
from pathlib import Path
from typing import Any

from .utils import get_ffmpeg, log
from .config import get_music_library


# ── Royalty-free music library ───────────────────────────────────────────────
# Loaded from config.yaml (or config.yaml.example)
# Categories map to moods/vibes that match different clip content.
MUSIC_LIBRARY: list[dict[str, str]] = get_music_library()


def get_media_duration(media_path: str | Path) -> float:
    """Return the duration of a media file in seconds."""
    from .utils import get_ffprobe

    try:
        result = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(media_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout).get("format", {}).get("duration", 0))
    except Exception as e:
        log("WARN", f"Could not determine duration for {media_path}: {e}")
        return 0.0


def choose_music_start_offset(
    music_path: str | Path,
    clip_duration: float,
    seed_source: str | None = None,
) -> float:
    """Pick a stable start offset inside a music track.

    The offset is biased away from the opening intro and ending tail so the
    chosen section is less likely to sound like the very beginning of the
    track. The exact section is deterministic for a given music file and
    seed, which keeps reruns stable.
    """
    music_duration = get_media_duration(music_path)
    if music_duration <= 0:
        return 0.0

    intro_buffer = min(10.0, music_duration * 0.12)
    outro_buffer = min(10.0, music_duration * 0.12)
    usable_duration = music_duration - intro_buffer - outro_buffer
    if usable_duration <= 0:
        return 0.0

    # Keep enough tail so the selected section can cover the whole clip.
    # This avoids late starts that can make the music end early in output.
    max_start_offset = max(0.0, music_duration - max(0.0, clip_duration))
    if max_start_offset <= 0:
        return 0.0

    effective_min = min(intro_buffer, max_start_offset)
    effective_max = min(intro_buffer + usable_duration, max_start_offset)
    if effective_max <= effective_min:
        return effective_min

    seed = f"{Path(music_path).as_posix()}|{clip_duration:.3f}|{seed_source or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    base_offset = effective_min + (fraction * (effective_max - effective_min))

    # Sample multiple deterministic candidates and choose the loudest section
    # so background music remains audible even when some regions are very quiet.
    offsets = _build_music_offset_candidates(base_offset, effective_min, effective_max)
    analysis_window = min(8.0, max(2.0, clip_duration * 0.25))

    best_offset = offsets[0]
    best_rms = -1.0
    for off in offsets:
        rms = _compute_audio_window_rms(music_path, off, analysis_window)
        if rms > best_rms:
            best_rms = rms
            best_offset = off

    return best_offset


def _build_music_offset_candidates(
    base_offset: float,
    min_offset: float,
    max_offset: float,
) -> list[float]:
    """Build deterministic candidate offsets within the usable range."""
    if max_offset <= min_offset:
        return [max(0.0, min_offset)]

    span = max_offset - min_offset
    rel_base = base_offset - min_offset
    # Spread probes across the track to avoid landing on an unusually quiet spot.
    shifts = [0.0, 0.21, 0.43, 0.67]
    candidates: list[float] = []
    for s in shifts:
        rel = (rel_base + span * s) % span
        candidates.append(min_offset + rel)
    return candidates


def _compute_audio_window_rms(
    media_path: str | Path,
    start_offset: float,
    window_duration: float,
) -> float:
    """Compute RMS for a short audio window using ffmpeg PCM output."""
    if window_duration <= 0:
        return 0.0

    try:
        result = subprocess.run(
            [
                get_ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, start_offset):.3f}",
                "-t",
                f"{window_duration:.3f}",
                "-i",
                str(media_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
            check=True,
        )
    except Exception:
        return 0.0

    if not result.stdout:
        return 0.0

    samples = array("h")
    samples.frombytes(result.stdout)
    if not samples:
        return 0.0

    sum_sq = 0.0
    for s in samples:
        sum_sq += float(s) * float(s)
    return (sum_sq / len(samples)) ** 0.5


def download_music_library(
    music_dir: str | Path = "music",
    api_key: str | None = None,
    min_duration: int | None = None,
) -> list[str]:
    """Pixabay download is intentionally disabled.

    Music is expected to come from local files, primarily assets/background_music.mp3.
    """
    _ = (music_dir, api_key, min_duration)
    log("INFO", "Pixabay music download is disabled. Using local music files only.")
    return []


def get_available_music(music_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Return music entries that have actual files available on disk.

    Args:
        music_dir: Directory containing music files.
                   If None, uses "music/" relative to cwd.
    """
    base = Path(music_dir) if music_dir else Path("music")
    available = []
    for entry in MUSIC_LIBRARY:
        file_path = base / entry["file"]
        if file_path.exists() and file_path.stat().st_size > 10_000:
            available.append({**entry, "file": str(file_path)})

    return available


def match_music_to_clip(
    clip: dict[str, Any],
    available_music: list[dict[str, str]],
    llm_model: str | None = None,
    api_key: str | None = None,
) -> dict[str, str] | None:
    """Use LLM to match the best background music to a clip's content.

    Args:
        clip: Clip metadata with topic, title, caption, hook, scores
        available_music: List of available music entries
        llm_model: LLM model override
        api_key: API key override

    Returns:
        Best matching music entry, or None
    """
    if not available_music:
        return None

    from .llm.backends import call_llm

    music_options = "\n".join([
        f"- {m['id']}: {m['description']} (mood: {m['mood']})"
        for m in available_music
    ])

    clip_info = (
        f"Title: {clip.get('title', '')}\n"
        f"Topic: {clip.get('topic', '')}\n"
        f"Hook: {clip.get('hook', '')}\n"
        f"Caption: {clip.get('caption', '')}\n"
        f"Scores — emotion: {clip.get('score_emotion', 0)}, "
        f"hook: {clip.get('score_hook', 0)}, "
        f"retention: {clip.get('score_retention', 0)}"
    )

    system = (
        "You are a music supervisor for short-form video content. "
        "Select the best background music that fits the clip's mood and content. "
        "The music should enhance the video without being distracting — it plays at very low volume. "
        "Return ONLY a JSON object with the key 'music_id' containing the selected music ID."
    )

    user = (
        f"Select the best background music for this clip:\n\n"
        f"CLIP:\n{clip_info}\n\n"
        f"AVAILABLE MUSIC:\n{music_options}\n\n"
        f"Return ONLY: {{\"music_id\": \"selected_id\"}}"
    )

    try:
        result = call_llm(system, user, api_key, llm_model, enable_reasoning=False)
        if result and isinstance(result, list) and len(result) > 0:
            music_id = result[0].get("music_id", "")
        elif result and isinstance(result, dict):
            music_id = result.get("music_id", "")
        else:
            music_id = ""

        if music_id:
            for m in available_music:
                if m["id"] == music_id:
                    return m

        # Fallback: return first available if LLM fails
        log("WARN", f"LLM music match returned unknown ID '{music_id}', using default")
    except Exception as e:
        log("WARN", f"LLM music match failed: {e}")

    return available_music[0] if available_music else None


def match_music_batch(
    clips: list[dict[str, Any]],
    available_music: list[dict[str, str]],
    llm_model: str | None = None,
    api_key: str | None = None,
) -> dict[int, dict[str, str]]:
    """Match music to multiple clips in a single LLM call.

    Returns:
        Dict mapping clip rank → music entry
    """
    if not available_music or not clips:
        return {}

    from .llm.backends import call_llm

    music_options = "\n".join([
        f"- {m['id']}: {m['description']} (mood: {m['mood']})"
        for m in available_music
    ])

    clips_info = []
    for c in clips:
        clips_info.append(
            f"Clip #{c.get('rank', 0)}: "
            f"title=\"{c.get('title', '')}\", "
            f"topic=\"{c.get('topic', '')}\", "
            f"hook=\"{c.get('hook', '')}\", "
            f"scores(emotion={c.get('score_emotion', 0)}, "
            f"hook={c.get('score_hook', 0)})"
        )

    system = (
        "You are a music supervisor for short-form video. "
        "Match background music to each clip. Music plays at very low volume as ambient vibe. "
        "Return ONLY a JSON array of objects with 'rank' and 'music_id' keys."
    )

    user = (
        f"Match background music for these clips:\n\n"
        f"CLIPS:\n" + "\n".join(clips_info) + "\n\n"
        f"AVAILABLE MUSIC:\n{music_options}\n\n"
        f"Return: [{{\"rank\": 1, \"music_id\": \"...\"}}, ...]"
    )

    result_map: dict[int, dict[str, str]] = {}

    try:
        result = call_llm(system, user, api_key, llm_model, enable_reasoning=False)
        if result and isinstance(result, list):
            music_by_id = {m["id"]: m for m in available_music}
            for item in result:
                rank = item.get("rank")
                mid = item.get("music_id", "")
                if rank is not None and mid in music_by_id:
                    result_map[int(rank)] = music_by_id[mid]
    except Exception as e:
        log("WARN", f"Batch music matching failed: {e}")

    # Fill in any missing clips with default
    default = available_music[0] if available_music else None
    if default:
        for c in clips:
            rank = c.get("rank", 0)
            if rank not in result_map:
                result_map[rank] = default

    return result_map


def build_music_filter(
    music_idx: int,
    clip_duration: float,
    start_offset: float = 0.0,
    volume: float = 0.20,
) -> str:
    """Build FFmpeg filter string for the background music stream.

    The music is looped, trimmed to clip duration, faded in/out, and
    attenuated to the given volume. Output label is [bgm].

    Args:
        music_idx: FFmpeg input index of the music file (e.g. 1 for second -i).
        clip_duration: Duration of the clip in seconds.
        start_offset: Start position inside the music stream in seconds.
        volume: Music volume (0.0–1.0), default 0.20.

    Returns:
        Filter string to include in filter_complex.
    """
    fade_in = min(1.0, clip_duration * 0.1)
    fade_out = min(2.0, clip_duration * 0.15)
    fade_out_start = max(0, clip_duration - fade_out)
    trim_start = max(0.0, start_offset)
    trim_end = trim_start + max(0.0, clip_duration)

    return (
        f"[{music_idx}:a]"
        f"aloop=loop=-1:size=2e+09,"
        f"atrim={trim_start:.3f}:{trim_end:.3f},"
        f"asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f},"
        f"volume={volume:.3f}"
        f"[bgm]"
    )


def apply_music_to_clip(
    clip_path: str,
    output_path: str,
    music_path: str,
    music_volume: float = 0.20,
) -> bool:
    """Mix background music into an existing video clip.

    Main audio is normalized for social media loudness (-10 LUFS) with
    enough headroom to avoid capping, then music is added additively
    (amix normalize=0) so voice volume is not reduced.

    Args:
        clip_path: Input video file path.
        output_path: Output video file path (can be same as input).
        music_path: Background music file path.
        music_volume: Music volume (0.0–1.0). Default 0.20.

    Returns:
        True on success, False on failure.
    """
    import tempfile

    duration = get_media_duration(clip_path)
    if duration <= 0:
        log("WARN", f"Zero duration for {clip_path}, skipping music")
        return False

    # Speech-optimized: -13 LUFS, LRA=7 for tight, clear narration
    # No pre-amp — source audio already has good levels
    voice_filter = "[0:a]loudnorm=I=-13:LRA=7:TP=-1.0[voice]"
    music_start_offset = choose_music_start_offset(music_path, duration, seed_source=clip_path)
    music_filter = build_music_filter(1, duration, music_start_offset, music_volume)
    mix_filter = "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
    if music_start_offset > 0:
        log("DEBUG", f"Using music start offset {music_start_offset:.2f}s for {Path(music_path).name}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_out = tmp.name

    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner",
        "-i", clip_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", f"{voice_filter};{music_filter};{mix_filter}",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        "-loglevel", "error",
        tmp_out,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        Path(tmp_out).replace(output_path)
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR", f"Music mixing failed: {e.stderr[:300] if e.stderr else str(e)}")
        try:
            Path(tmp_out).unlink(missing_ok=True)
        except OSError:
            pass
        return False
