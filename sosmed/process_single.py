"""
Process a single video (or a folder of videos) to extract the best clip with overlay subtitles.
Output: topic and title printed to console.
Usage:
  python sosmed/process_single.py path/to/video.mp4 [options]
  python sosmed/process_single.py path/to/folder/  [options]   # batch mode
"""

import argparse
import glob as glob_module
import json
import sys
import time
from pathlib import Path

from .transcription import transcribe
from .prefilter import prefilter_segments
from .llm import generate_single_clip_metadata, translate_subtitle_words
from .extraction import extract_clips, _get_video_duration
from .postprocess import postprocess_clips
from .config import get_defaults, get_cta_settings
from .utils import (
    log, BOLD, RESET, CYAN, GREEN, YELLOW,
    strip_internal_fields, get_internal_fields, get_clips_cache_dir
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def _get_transcript_cache_path(video_path: str) -> Path:
    """Return cache path for transcript based on video filename."""
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video.stem}_transcript.json"
    return cache_file


def _build_full_video_clip(
    video: Path,
    video_duration: float,
    segments: list[dict],
    title: str | None = None,
    caption: str | None = None,
) -> dict:
    """Build a single clip that covers the entire video."""
    clip_start = 0.0
    if segments:
        clip_start = max(0.0, min(float(segments[0].get("start", 0.0)), video_duration))

    return {
        "rank": 1,
        "start": clip_start,
        "end": video_duration,
        "title": title or "",
        "topic": "",
        "caption": caption or "",
        "reason": "Single-video mode returns the full source video.",
        "hook": "",
        "closing_line": "",
        "comment_bait": "",
        "score_hook": 100,
        "score_insight_density": 100,
        "score_retention": 100,
        "score_emotional_payoff": 100,
        "score_clarity": 100,
        "clip_score": 100,
    }


def _should_translate_to_indonesian(detected_language: dict | None) -> bool:
    """Return True when subtitle translation to Indonesian is still needed."""
    if not detected_language:
        return True

    lang = str(detected_language.get("language", "")).lower()
    prob = float(detected_language.get("language_probability", 0.0) or 0.0)
    return not (lang == "id" and prob > 0.6)


def process_single_video(
    video_path: str,
    model: str = "turbo",
    lang: str | None = "id",
    device: str = "auto",
    compute_type: str = "auto",
    no_vad: bool = False,
    vad_min_silence: int = 400,
    vad_speech_pad: int = 200,
    batch: int = 16,
    chunk_duration: float = 360.0,
    chunk_overlap: float = 60.0,
    min_duration: int = 5,
    max_duration: int = 60,
    output_dir: str = "output",
    api_key: str | None = None,
    llm_model: str | None = None,
    subtitles: bool = True,
    subtitle_position: str = "lower",
    subtitle_margin_pct: float | None = None,
    title: str | None = None,
    caption: str | None = None,
    cta: bool | None = None,
    music: bool = False,
    music_dir: str = "music",
    music_volume: float = 0.06,
    encoding_preset: str | None = None,
    encoding_crf: int | None = None,
    skip_existing: bool = False,
) -> dict:
    """
    Process a single video and extract the best clip.

    Args:
        skip_existing: If True, skip processing if final output file already exists.

    Returns:
        dict with keys: 'topic', 'title', 'video_path', 'start', 'end', 'clip' (full clip dict)
    """
    video = Path(video_path)
    if not video.exists():
        log("ERROR", f"File not found: {video}")
        sys.exit(1)

    output_path = Path(output_dir) / video.stem

    # ── Check for existing output ───────────────────────────────────────────
    skip_to_postprocess = False
    existing_clips_data = None
    existing_segments = None
    existing_detected_language = None
    raw_outputs: list[str] = []
    
    if skip_existing:
        clips_json = output_path / "all_clips.json"
        if not clips_json.exists():
            clips_json = output_path / "clips.json"
        if clips_json.exists():
            clips_data = json.loads(clips_json.read_text())
            if clips_data:
                final_filename = clips_data[0].get("filename")
                if final_filename:
                    final_file = output_path / final_filename
                    if final_file.exists():
                        # Check if post-processing is needed (CTA, subtitles, music)
                        cta_defaults = get_cta_settings()
                        cta_enabled = cta if cta is not None else cta_defaults.get("enabled", False)
                        needs_postprocess = subtitles or cta_enabled or music
                        
                        # Check if raw clip still exists — only re-postprocess when we have it
                        _raw_pat = str(output_path / "rank*_*.mp4")
                        _raw_avail = [f for f in glob_module.glob(_raw_pat)
                                      if "_final" not in f and "_ctatmp" not in f]

                        if not needs_postprocess or not _raw_avail:
                            log("OK", f"Output already exists: {final_file}")
                            log("OK", "Skipping processing (use --no-skip-existing to force re-run)")
                            best_clip = clips_data[0]
                            return {
                                "topic": best_clip.get("topic", ""),
                                "title": best_clip.get("title", ""),
                                "video_path": str(final_file),
                                "start": best_clip.get("start", 0),
                                "end": best_clip.get("end", 0),
                                "clip": best_clip,
                            }
                        # Raw clip available and post-processing needed — skip transcription/LLM/extraction
                        # but run post-process below
                        log("INFO", "Found existing clip - skipping to post-processing...")
                        skip_to_postprocess = True
                        existing_clips_data = clips_data
                        # Try to load cached transcript for subtitles
                        cache_path = _get_transcript_cache_path(str(video))
                        if cache_path.exists():
                            cache_data = json.loads(cache_path.read_text())
                            if isinstance(cache_data, dict) and "segments" in cache_data:
                                existing_segments = cache_data["segments"]
                                existing_detected_language = cache_data.get("language_info", {"language": "unknown", "language_probability": 0.0})

    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   Single Video Processor")
    print(f"{'═' * 50}{RESET}")
    print(f"  Video     : {video.name}")
    print(f"  Model     : {model}")
    print(f"  Language  : {lang or 'auto-detect'}")
    print()

    t_total = time.time()

    # ── 1. Transcribe ────────────────────────────────────────────────────────
    segments = existing_segments if skip_to_postprocess else None
    detected_language = existing_detected_language if skip_to_postprocess else {"language": "unknown", "language_probability": 0.0}
    
    if not skip_to_postprocess:
        cache_path = _get_transcript_cache_path(str(video))
        if cache_path.exists():
            log("INFO", f"Loading cached transcript from {cache_path}")
            cache_data = json.loads(cache_path.read_text())
            if isinstance(cache_data, dict) and "segments" in cache_data:
                segments = cache_data["segments"]
                detected_language = cache_data.get("language_info", detected_language)
            else:
                segments = cache_data
            log("OK", f"Loaded {len(segments)} segments from cache")
        else:
            segments, detected_language = transcribe(
                str(video),
                model_size=model,
                language=lang,
                device=device,
                compute_type=compute_type,
                vad_filter=not no_vad,
                vad_min_silence_ms=vad_min_silence,
                vad_speech_pad_ms=vad_speech_pad,
                batch_size=batch,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {"segments": segments, "language_info": detected_language}
            cache_path.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
            log("OK", f"Transcript cached → {cache_path}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    if not skip_to_postprocess:
        filtered, stats = prefilter_segments(segments)
        log("OK", f"Pre-filtered: {stats['original']} → {stats['kept']} segments")

        if not filtered:
            log("ERROR", "All segments filtered out. Try a different whisper model or looser filters.")
            sys.exit(1)

    # ── 3. Full-video single clip ───────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)
    
    if skip_to_postprocess and existing_clips_data:
        # Use existing clip data
        clips = existing_clips_data
        best_clip = existing_clips_data[0]
        log("OK", f"Loaded existing clip: rank {best_clip['rank']}")
        log("OK", f"  Start: {best_clip['start']:.1f}s, End: {best_clip['end']:.1f}s")
        log("OK", f"  Topic: {best_clip.get('topic', 'N/A')}")
        log("OK", f"  Title: {best_clip.get('title', 'N/A')}")
        # Create a raw output pointing to the existing file (without postprocess)
        # We need to find the raw extracted clip (before postprocess)
        # Look for files matching the pattern but not the final _final.mp4
        raw_pattern = str(output_path / "rank*_*.mp4")
        raw_files = [f for f in glob_module.glob(raw_pattern) if "_final" not in f and "_ctatmp" not in f]
        if raw_files:
            raw_outputs = [raw_files[0]]
            log("INFO", f"Found existing raw clip: {raw_files[0]}")
        else:
            # No raw clip found, use the final file as input
            final_filename = best_clip.get("filename")
            if final_filename:
                final_file = output_path / final_filename
                if final_file.exists():
                    raw_outputs = [str(final_file)]
                    log("INFO", f"Using existing final clip as input: {final_file}")
                else:
                    log("ERROR", f"Expected clip file not found: {final_file}. Delete output/{output_path.name} and re-run.")
                    sys.exit(1)
            else:
                log("ERROR", "No clip filename recorded in all_clips.json. Delete output and re-run.")
                sys.exit(1)
    else:
        video_dur = _get_video_duration(str(video))
        if not video_dur or video_dur <= 0:
            video_dur = float(segments[-1]["end"])

        clips = [_build_full_video_clip(video, video_dur, segments, title=title, caption=caption)]
        log("INFO", "Single-video mode skips LLM chunking and uses the entire video as one clip")

        # Check for cached LLM metadata before calling the API
        _meta_cache_path = get_clips_cache_dir(output_path) / "metadata_cache.json"
        _meta_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _loaded_from_cache = False
        if _meta_cache_path.exists():
            try:
                _cached_meta = json.loads(_meta_cache_path.read_text())
                if not isinstance(_cached_meta, dict):
                    raise ValueError("cache is not a dict")
                log("INFO", f"Loading cached LLM metadata from {_meta_cache_path}")
                # Don't overwrite fields the user explicitly provided via CLI
                _cached_meta_to_apply = {
                    k: v for k, v in _cached_meta.items()
                    if not (k == "title" and title) and not (k == "caption" and caption)
                }
                clips[0].update(_cached_meta_to_apply)
                _loaded_from_cache = True
            except Exception as e:
                log("WARN", f"Metadata cache unreadable ({e}), regenerating...")
                _meta_cache_path.unlink(missing_ok=True)

        if not _loaded_from_cache:
            log("INFO", "Generating title, topic, caption, reason, and hook from transcript...")
            clips[0] = generate_single_clip_metadata(
                clips[0],
                filtered,
                llm_model=llm_model,
                api_key=api_key,
            )
            # Cache the generated metadata fields
            _meta_fields = ["title", "topic", "caption", "reason", "hook", "closing_line", "comment_bait"]
            _cached_meta = {k: clips[0][k] for k in _meta_fields if k in clips[0]}
            _meta_cache_path.write_text(json.dumps(_cached_meta, indent=2, ensure_ascii=False))
            log("OK", f"LLM metadata cached → {_meta_cache_path}")

        # Select best clip (rank 1)
        best_clip = clips[0] if clips else None
        if not best_clip:
            log("ERROR", "No clip available for single-video processing")
            sys.exit(1)

        log("OK", f"Selected full video clip: rank {best_clip['rank']} (score: {best_clip.get('clip_score', '?')})")
        log("OK", f"  Start: {best_clip['start']:.1f}s, End: {best_clip['end']:.1f}s")
        log("OK", f"  Topic: {best_clip.get('topic', 'N/A')}")
        log("OK", f"  Title: {best_clip.get('title', 'N/A')}")

        # ── 3c. Translate subtitle words to Indonesian ───────────────────────────
        if subtitles:
            from .subtitles import get_clip_words
            raw_words = get_clip_words(
                segments,
                clip_start=best_clip["start"],
                clip_end=best_clip["end"],
            )
            if _should_translate_to_indonesian(detected_language):
                log("INFO", "Translating subtitle words to Indonesian (with Whisper error fixing)...")
                best_clip["_subtitle_words"] = translate_subtitle_words(
                    raw_words,
                    llm_model=llm_model,
                    api_key=api_key,
                    fix_errors=True,  # Enable Whisper error fixing
                )
                log("OK", f"Subtitle translation complete: {len(best_clip['_subtitle_words'])} words")
            else:
                log("INFO", "Whisper detected Indonesian, but still fixing transcription errors...")
                # Even if language is Indonesian, still fix transcription errors
                best_clip["_subtitle_words"] = translate_subtitle_words(
                    raw_words,
                    llm_model=llm_model,
                    api_key=api_key,
                    fix_errors=True,  # Enable Whisper error fixing
                )
                log("OK", f"Subtitle fixing complete: {len(best_clip['_subtitle_words'])} words")

        # ── 4. Extract best clip ─────────────────────────────────────────────────
        log("INFO", "Extracting best clip...")
        raw_outputs = extract_clips(
            str(video),
            [best_clip],  # Only best clip
            output_dir=output_path,
            max_workers=1,
            encoding_preset=encoding_preset,
            encoding_crf=encoding_crf,
        )

        if not raw_outputs:
            log("ERROR", "Failed to extract clip")
            sys.exit(1)

    # ── 4b. Prepare background music ─────────────────────────────────────
    music_entries: dict = {}
    if music:
        from pathlib import Path as _Path
        from .music import get_available_music, download_music_library
        available = get_available_music(music_dir)
        if not available:
            log("INFO", "No music files found — attempting auto-download from Pixabay...")
            downloaded = download_music_library(music_dir=music_dir)
            if downloaded:
                available = get_available_music(music_dir)
        if not available:
            assets_music = _Path(__file__).parent.parent / "assets" / "background_music.mp3"
            if assets_music.exists():
                log("INFO", f"Using fallback background music from assets: {assets_music}")
                fallback = {"id": "background_music", "file": str(assets_music), "description": "Background music", "mood": "neutral"}
                music_entries = {best_clip.get("rank", 1): fallback}
            else:
                log("WARN", "No background music files found. Skipping music.")
        else:
            fallback = available[0]
            music_entries = {best_clip.get("rank", 1): fallback}

    # ── 3c. Translate subtitle words (for skip_to_postprocess case) ────────
    # Only fix subtitles if the existing file doesn't have them yet
    # (detected by checking if filename contains "_final")
    existing_has_subtitles = False
    if skip_to_postprocess and best_clip.get("filename"):
        existing_has_subtitles = "_final" in best_clip.get("filename", "")
    
    if skip_to_postprocess and subtitles and segments and "_subtitle_words" not in best_clip and not existing_has_subtitles:
        from .subtitles import get_clip_words
        raw_words = get_clip_words(
            segments,
            clip_start=best_clip["start"],
            clip_end=best_clip["end"],
        )
        if _should_translate_to_indonesian(detected_language):
            log("INFO", "Translating subtitle words to Indonesian (with Whisper error fixing)...")
            best_clip["_subtitle_words"] = translate_subtitle_words(
                raw_words,
                llm_model=llm_model,
                api_key=api_key,
                fix_errors=True,
            )
            log("OK", f"Subtitle translation complete: {len(best_clip['_subtitle_words'])} words")
        else:
            log("INFO", "Whisper detected Indonesian, but still fixing transcription errors...")
            best_clip["_subtitle_words"] = translate_subtitle_words(
                raw_words,
                llm_model=llm_model,
                api_key=api_key,
                fix_errors=True,
            )
            log("OK", f"Subtitle fixing complete: {len(best_clip['_subtitle_words'])} words")
    elif skip_to_postprocess and subtitles and existing_has_subtitles:
        log("INFO", "Existing file already has subtitles - skipping LLM subtitle fixing")

    # ── 5. Post-process with subtitles ──────────────────────────────────
    cta_defaults = get_cta_settings()
    cta_cfg = {**cta_defaults, "enabled": cta if cta is not None else cta_defaults.get("enabled", False)}

    # If the input is already a _final file (already postprocessed), only run CTA
    _input_is_final = raw_outputs and "_final" in Path(raw_outputs[0]).stem
    if _input_is_final and cta_cfg.get("enabled") and raw_outputs:
        from .cta import append_instagram_cta
        log("INFO", "Input already post-processed — appending CTA only...")
        cta_result = append_instagram_cta(
            raw_outputs[0],
            raw_outputs[0],
            name=str(cta_cfg.get("name", "Samuel Academy")),
            username=str(cta_cfg.get("username", "@samuelkoesnadi")),
            duration=float(cta_cfg.get("duration", 3.0)),
            fade_duration=float(cta_cfg.get("fade_duration", 0.5)),
        )
        outputs = [cta_result]
    elif (subtitles or cta_cfg["enabled"] or music) and raw_outputs:
        log("INFO", "Adding subtitles overlay...")
        outputs = postprocess_clips(
            raw_outputs,
            [best_clip],
            segments,
            output_dir=output_path,
            subtitles=subtitles,
            subtitle_position=subtitle_position,
            subtitle_margin_pct=subtitle_margin_pct,
            enable_music=music,
            music_entries=music_entries,
            music_volume=music_volume,
            cta_config=cta_cfg,
            encoding_preset=encoding_preset,
            encoding_crf=encoding_crf,
        )
    else:
        outputs = raw_outputs

    elapsed_total = time.time() - t_total

    # ── 6. Output and reporting ─────────────────────────────────────────
    if outputs:
        print(f"\n{GREEN}{BOLD}✓ Complete!{RESET}")
        print(f"  Output: {outputs[0]}")
        print(f"  Time: {elapsed_total:.0f}s")
        print()
        print(f"{BOLD}Topic:{RESET} {best_clip.get('topic', 'N/A')}")
        print(f"{BOLD}Title:{RESET} {best_clip.get('title', 'N/A')}")
        print(f"{BOLD}Caption:{RESET} {best_clip.get('caption', 'N/A')}")

        # Save clip JSONs (public to disk, internal to .cache)
        # Best clip
        best_clip_public = strip_internal_fields([best_clip])[0]
        best_clip_path = output_path / "best_clip.json"
        best_clip_path.write_text(json.dumps(best_clip_public, indent=2, ensure_ascii=False))
        log("OK", f"Saved best clip → {best_clip_path}")

        # Save internal fields to cache
        internal = get_internal_fields([best_clip])
        if internal:
            cache_dir = get_clips_cache_dir(output_path)
            cache_file = cache_dir / "clips_internal.json"
            cache_file.write_text(json.dumps(internal, indent=2, ensure_ascii=False))

        # Save all clips for reference
        all_clips_public = strip_internal_fields(clips)
        all_clips_path = output_path / "all_clips.json"
        all_clips_path.write_text(json.dumps(all_clips_public, indent=2, ensure_ascii=False))
        log("OK", f"Saved all {len(clips)} clips → {all_clips_path}")

        best_clip_public["video_path"] = str(outputs[0])
        return {
            "topic": best_clip.get("topic", ""),
            "title": best_clip.get("title", ""),
            "video_path": str(outputs[0]),
            "start": best_clip["start"],
            "end": best_clip["end"],
            "clip": best_clip_public,
        }
    else:
        log("ERROR", "No output generated")
        sys.exit(1)


def add_cta_to_existing(
    video_path: str,
    output_dir: str = "output",
    cta_name: str = "Samuel Academy",
    cta_username: str = "@samuelkoesnadi",
    cta_duration: float = 3.0,
    fade_duration: float = 0.5,
) -> dict:
    """
    Add Instagram CTA to an existing _final.mp4 clip.

    This is a lightweight operation that only appends the CTA screen
    without re-transcribing, re-extracting, or re-processing the video.

    Args:
        video_path: Path to the input video (can be _final.mp4 or any MP4)
        output_dir: Output directory for the result
        cta_name: Display name for the CTA
        cta_username: Instagram handle for the CTA
        cta_duration: Duration of the CTA screen in seconds
        fade_duration: Fade transition duration

    Returns:
        dict with 'video_path' pointing to the output file
    """
    from .cta import append_instagram_cta

    video = Path(video_path)
    if not video.exists():
        log("ERROR", f"File not found: {video}")
        sys.exit(1)

    output_path = Path(output_dir) / video.stem

    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   CTA Append Mode")
    print(f"{'═' * 50}{RESET}")
    print(f"  Input     : {video.name}")
    print(f"  CTA Name  : {cta_name}")
    print(f"  CTA Handle: {cta_username}")
    print(f"  Duration  : {cta_duration}s")
    print()

    t_start = time.time()

    output_path.mkdir(parents=True, exist_ok=True)

    # Determine output filename - always use a different file from input
    if "_final" in video.stem:
        # Already a _final file, output to same directory with _cta suffix
        output_file = video.parent / f"{video.stem}_cta.mp4"
    else:
        # Not a _final file, create new output
        output_file = output_path / f"{video.stem}_with_cta.mp4"

    log("INFO", f"Appending Instagram CTA to {video.name}...")

    result = append_instagram_cta(
        str(video),
        str(output_file),
        name=cta_name,
        username=cta_username,
        duration=cta_duration,
        fade_duration=fade_duration,
    )

    elapsed = time.time() - t_start

    if result == str(video):
        log("ERROR", "CTA append failed - input file left unchanged")
        sys.exit(1)

    print(f"\n{GREEN}{BOLD}✓ Complete!{RESET}")
    print(f"  Output: {result}")
    print(f"  Time: {elapsed:.1f}s")
    print()

    return {
        "video_path": result,
        "input_path": str(video),
    }


def process_folder(
    folder_path: str,
    output_dir: str = "output",
    **kwargs,
) -> list[dict]:
    """
    Process all video files in a folder using process_single_video.

    Returns:
        list of result dicts from each processed video (same shape as process_single_video)
        Also writes a combined all_clips.json to the output_dir root.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        log("ERROR", f"Not a directory: {folder}")
        sys.exit(1)

    video_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        log("ERROR", f"No video files found in {folder} (looked for {', '.join(VIDEO_EXTENSIONS)})")
        sys.exit(1)

    total = len(video_files)
    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   Batch Processor  ({total} video{'s' if total != 1 else ''})")
    print(f"{'═' * 50}{RESET}")
    for i, vf in enumerate(video_files, 1):
        print(f"  [{i}/{total}] {vf.name}")
    print()

    results: list[dict] = []
    all_clips: list[dict] = []
    failed: list[str] = []

    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{BOLD}{YELLOW}── [{idx}/{total}] Processing: {video_file.name} ──{RESET}")
        try:
            result = process_single_video(
                video_path=str(video_file),
                output_dir=output_dir,
                **kwargs,
            )
            results.append(result)
            clip_entry = dict(result.get("clip") or {})
            clip_entry["source_video"] = video_file.name
            all_clips.append(clip_entry)
        except SystemExit:
            log("ERROR", f"Failed to process {video_file.name}, skipping")
            failed.append(video_file.name)

    # Write combined all_clips.json to output root
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_path = out_root / "all_clips.json"
    combined_path.write_text(json.dumps(all_clips, indent=2, ensure_ascii=False))

    print(f"\n{BOLD}{GREEN}{'═' * 50}")
    print(f"   Batch Complete")
    print(f"{'═' * 50}{RESET}")
    print(f"  Processed : {len(results)}/{total}")
    if failed:
        print(f"  Failed    : {', '.join(failed)}")
    print(f"  All clips : {combined_path}")
    print()

    return results


def main() -> None:
    """CLI entry point for process_single.py"""
    defaults = get_defaults()

    ap = argparse.ArgumentParser(
        description="Process a single video (or folder of videos) and extract the best clip with overlay subtitles.",
    )
    ap.add_argument("video", help="Path to input video file or folder (folder = batch mode)")
    ap.add_argument("--cta-only", action="store_true",
                    help="Only append CTA to an existing video (skip transcription, extraction, post-processing). "
                         "Use with --cta-name, --cta-username, --cta-duration to customize.")
    ap.add_argument("--cta-name", default="Samuel Academy",
                    help="Display name for CTA (default: Samuel Academy)")
    ap.add_argument("--cta-username", default="@samuelkoesnadi",
                    help="Instagram handle for CTA (default: @samuelkoesnadi)")
    ap.add_argument("--cta-duration", type=float, default=3.0,
                    help="CTA duration in seconds (default: 3.0)")
    ap.add_argument("--model", default=defaults.get("whisper_model", "turbo"),
                    choices=["tiny", "base", "small", "medium",
                             "large-v2", "large-v3", "distil-large-v3", "turbo"],
                    help="Whisper model size (default: from config or turbo)")
    ap.add_argument("--lang", default="id",
                    help="Language code — 'id' Indonesian, 'en' English, "
                         "or None for auto-detect (default: id)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="Compute device (default: auto)")
    ap.add_argument("--compute-type", default="auto",
                    choices=["auto", "float16", "int8", "int8_float16"],
                    help="Compute type (default: auto)")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable VAD filtering for transcription")
    ap.add_argument("--vad-min-silence", type=int, default=400,
                    help="VAD min silence duration in ms (default: 400)")
    ap.add_argument("--vad-speech-pad", type=int, default=200,
                    help="VAD speech padding in ms (default: 200)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Whisper batch size (default: 16; lower if OOM)")
    ap.add_argument("--chunk-duration", type=float, default=360.0,
                    help="LLM chunk duration in seconds (default: 360)")
    ap.add_argument("--chunk-overlap", type=float, default=60.0,
                    help="Overlap between chunks in seconds (default: 60)")
    ap.add_argument("--min-duration", type=int, default=5,
                    help="Minimum clip duration in seconds (default: 5)")
    ap.add_argument("--max-duration", type=int, default=180,
                    help="Maximum clip duration in seconds (default: 180)")
    ap.add_argument("--output", default="output",
                    help="Output directory (default: ./output)")
    ap.add_argument("--api-key", default=None,
                    help="API key (overrides env vars)")
    ap.add_argument("--llm-model", default=None,
                    help="Override LLM model name for OpenRouter")
    ap.add_argument("--subtitles", action=argparse.BooleanOptionalAction,
                    default=defaults.get("subtitles_enabled", True),
                    help="TikTok-style word-by-word subtitles (default: from config or on)")
    ap.add_argument("--subtitle-position",
                    default=defaults.get("subtitle_position", "lower"),
                    choices=["center", "upper", "lower"],
                    help="Subtitle position (default: from config or lower)")
    ap.add_argument("--subtitle-margin", type=float,
                    default=defaults.get("subtitle_margin_pct"),
                    help="Subtitle margin from bottom for 'lower' position in %% (default: from config or 25)")
    ap.add_argument("--title", default=None,
                    help="Manually set the title (overrides auto-generated title; single-file mode only)")
    ap.add_argument("--caption", default=None,
                    help="Manually set the caption (overrides auto-generated caption; single-file mode only)")
    _cta_defaults = get_cta_settings()
    ap.add_argument("--cta", action=argparse.BooleanOptionalAction,
                    default=_cta_defaults.get("enabled", False),
                    help="Append Instagram follow CTA at the end "
                         f"(default: {'on' if _cta_defaults.get('enabled') else 'off'})")
    ap.add_argument("--music", action=argparse.BooleanOptionalAction,
                    default=defaults.get("music_enabled", False),
                    help="Add background music (default: from config or off)")
    ap.add_argument("--music-dir", default=defaults.get("music_dir", "music"),
                    help="Directory containing music files (default: music/)")
    ap.add_argument("--music-volume", type=float, default=defaults.get("music_volume", 0.06),
                    help="Background music volume 0.0-1.0 (default: 0.06)")
    ap.add_argument("--encoding-preset", default=defaults.get("encoding_preset", "veryfast"),
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                    help="ffmpeg x264 encoding preset (default: from config or veryfast)")
    ap.add_argument("--encoding-crf", type=int, default=defaults.get("encoding_crf", 23),
                    help="ffmpeg x264 CRF quality (18-28, lower=better; default: from config or 23)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip processing if final output file already exists")

    args = ap.parse_args()
    lang = None if args.lang.lower() == "none" else args.lang

    # CTA-only mode: just append CTA to existing video
    if args.cta_only:
        add_cta_to_existing(
            video_path=args.video,
            output_dir=args.output,
            cta_name=args.cta_name,
            cta_username=args.cta_username,
            cta_duration=args.cta_duration,
        )
        return

    shared_kwargs = dict(
        model=args.model,
        lang=lang,
        device=args.device,
        compute_type=args.compute_type,
        no_vad=args.no_vad,
        vad_min_silence=args.vad_min_silence,
        vad_speech_pad=args.vad_speech_pad,
        batch=args.batch,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output,
        api_key=args.api_key,
        llm_model=args.llm_model,
        subtitles=args.subtitles,
        subtitle_position=args.subtitle_position,
        subtitle_margin_pct=args.subtitle_margin,
        cta=args.cta,
        music=args.music,
        music_dir=args.music_dir,
        music_volume=args.music_volume,
        encoding_preset=args.encoding_preset,
        encoding_crf=args.encoding_crf,
        skip_existing=args.skip_existing,
    )

    input_path = Path(args.video)
    if input_path.is_dir():
        process_folder(str(input_path), **shared_kwargs)
    else:
        process_single_video(
            video_path=args.video,
            title=args.title,
            caption=args.caption,
            **shared_kwargs,
        )


if __name__ == "__main__":
    main()
