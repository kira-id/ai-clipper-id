"""
Background music selection and mixing for video clips.

Uses a curated library of royalty-free music organized by mood/category.
LLM matches clip content to appropriate background music.
Music is mixed at very low volume as ambient vibe.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests

from .utils import get_ffmpeg, log
from .config import get_music_library, get_pixabay_settings


# ── Royalty-free music library ───────────────────────────────────────────────
# Loaded from config.yaml (or config.yaml.example)
# Categories map to moods/vibes that match different clip content.
MUSIC_LIBRARY: list[dict[str, str]] = get_music_library()


# ── Pixabay search queries per music category ──────────────────────────────
# Maps each library entry to a Pixabay search query + category filter.
# Auto-generated from music library IDs.
_PIXABAY_QUERIES: dict[str, dict[str, str]] = {
    entry["id"]: {"q": entry["id"].replace("_", " ") + " " + entry.get("category", ""), "category": "music"}
    for entry in MUSIC_LIBRARY
}


def download_music_library(
    music_dir: str | Path = "music",
    api_key: str | None = None,
    min_duration: int | None = None,
) -> list[str]:
    """Download copyright-free music from Pixabay for each mood category.

    Pixabay music is 100% royalty-free, no attribution required, safe for
    commercial use including YouTube, TikTok, etc.

    Get a free API key at: https://pixabay.com/api/docs/

    Args:
        music_dir: Directory to save music files into.
        api_key: Pixabay API key. Falls back to PIXABAY_API_KEY env var.
        min_duration: Minimum track duration in seconds. If None, uses config.

    Returns:
        List of downloaded file paths.
    """
    api_key = api_key or os.environ.get("PIXABAY_API_KEY", "")
    if not api_key:
        log("ERROR", "Pixabay API key required for music download. "
                      "Get a free key at https://pixabay.com/api/docs/ "
                      "then set PIXABAY_API_KEY env var.")
        return []

    music_path = Path(music_dir)
    music_path.mkdir(parents=True, exist_ok=True)

    # Load Pixabay settings from config
    pixabay_settings = get_pixabay_settings()
    if min_duration is None:
        min_duration = pixabay_settings.get("min_duration", 30)
    per_page = pixabay_settings.get("per_page", 5)

    downloaded: list[str] = []

    for entry in MUSIC_LIBRARY:
        mid = entry["id"]
        target_file = music_path / f"{mid}.mp3"

        # Skip if already downloaded
        if target_file.exists() and target_file.stat().st_size > 10_000:
            log("OK", f"  {mid}: already exists ({target_file.stat().st_size // 1024} KB)")
            downloaded.append(str(target_file))
            continue

        query_info = _PIXABAY_QUERIES.get(mid)
        if not query_info:
            log("WARN", f"  {mid}: no search query defined, skipping")
            continue

        # Search Pixabay audio API
        try:
            resp = requests.get(
                "https://pixabay.com/api/",
                params={
                    "key": api_key,
                    "q": query_info["q"],
                    "media_type": "music",
                    "per_page": per_page,
                    "min_duration": min_duration,
                    "order": "popular",
                    "editors_choice": "true",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Try without editors_choice filter
            try:
                resp = requests.get(
                    "https://pixabay.com/api/",
                    params={
                        "key": api_key,
                        "q": query_info["q"],
                        "media_type": "music",
                        "per_page": per_page,
                        "min_duration": min_duration,
                        "order": "popular",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e2:
                log("WARN", f"  {mid}: Pixabay search failed: {e2}")
                continue

        hits = data.get("hits", [])
        if not hits:
            log("WARN", f"  {mid}: no results for query '{query_info['q']}'")
            continue

        # Pick the first (most popular) hit
        hit = hits[0]
        audio_url = hit.get("audio") or hit.get("previewURL") or ""
        if not audio_url:
            # Try alternative URL fields
            for h in hits:
                audio_url = h.get("audio") or h.get("previewURL") or ""
                if audio_url:
                    hit = h
                    break

        if not audio_url:
            log("WARN", f"  {mid}: no download URL found")
            continue

        # Download the audio file
        try:
            log("INFO", f"  {mid}: downloading '{hit.get('title', 'unknown')}'...")
            audio_resp = requests.get(audio_url, timeout=60, stream=True)
            audio_resp.raise_for_status()

            with open(target_file, "wb") as f:
                for chunk in audio_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = target_file.stat().st_size // 1024
            log("OK", f"  {mid}: downloaded ({size_kb} KB) → {target_file}")
            downloaded.append(str(target_file))
        except Exception as e:
            log("WARN", f"  {mid}: download failed: {e}")
            if target_file.exists():
                target_file.unlink()

    return downloaded


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
    volume: float = 0.06,
) -> str:
    """Build FFmpeg filter string for the background music stream.

    The music is looped, trimmed to clip duration, faded in/out, and
    attenuated to the given volume. Output label is [bgm].

    Args:
        music_idx: FFmpeg input index of the music file (e.g. 1 for second -i).
        clip_duration: Duration of the clip in seconds.
        volume: Music volume (0.0–1.0), default 0.06 (very quiet).

    Returns:
        Filter string to include in filter_complex.
    """
    fade_in = min(1.0, clip_duration * 0.1)
    fade_out = min(2.0, clip_duration * 0.15)
    fade_out_start = max(0, clip_duration - fade_out)

    return (
        f"[{music_idx}:a]"
        f"aloop=loop=-1:size=2e+09,"
        f"atrim=0:{clip_duration:.3f},"
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
    music_volume: float = 0.06,
) -> bool:
    """Mix background music into an existing video clip.

    Main audio is normalized for social media loudness (-10 LUFS) with
    enough headroom to avoid capping, then music is added additively
    (amix normalize=0) so voice volume is not reduced.

    Args:
        clip_path: Input video file path.
        output_path: Output video file path (can be same as input).
        music_path: Background music file path.
        music_volume: Music volume (0.0–1.0). Default 0.06 (very quiet).

    Returns:
        True on success, False on failure.
    """
    import json
    import tempfile

    from .utils import get_ffprobe

    # Get clip duration via ffprobe
    try:
        result = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                clip_path,
            ],
            capture_output=True, text=True, check=True,
        )
        duration = float(json.loads(result.stdout).get("format", {}).get("duration", 0))
    except Exception as e:
        log("WARN", f"Could not determine duration for {clip_path}: {e}")
        return False

    if duration <= 0:
        log("WARN", f"Zero duration for {clip_path}, skipping music")
        return False

    # Speech-optimized: -13 LUFS, LRA=7 for tight, clear narration
    # No pre-amp — source audio already has good levels
    voice_filter = "[0:a]loudnorm=I=-13:LRA=7:TP=-1.0[voice]"
    music_filter = build_music_filter(1, duration, music_volume)
    mix_filter = "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"

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
