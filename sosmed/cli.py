"""
CLI interface: argument parsing and orchestration.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .transcription import transcribe
from .prefilter import prefilter_segments
from .llm import find_clips, fix_and_improve_clips
from .extraction import extract_clips, _get_video_duration
from .postprocess import postprocess_clips
from .utils import (
    log, BOLD, RESET, CYAN, GREEN, YELLOW,
    MAX_CLIPS_HARD_LIMIT,
    save_clips_to_disk, load_clips_with_internal_fields
)
from .smart_clip_boundaries import smart_adjust_clip_boundaries
from .audio_energy import analyze_audio_energy
from .config import get_defaults, load_config, get_cta_settings


def _get_transcript_cache_path(video_path: str) -> Path:
    """Return cache path for transcript based on video filename."""
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video.stem}_transcript.json"
    return cache_file


def _make_clip_filename(clip: dict) -> str:
    """Generate clip filename for raw extracted clips (before postprocessing)."""
    rank = clip.get("rank", 0)
    safe = re.sub(r"[^\w\s-]", "", clip.get("title", f"clip_{rank}"))
    safe = re.sub(r"\s+", "_", safe)[:50]
    return f"rank{rank:02d}_{safe}.mp4"


def _ensure_filenames(clips_list: list[dict]) -> bool:
    """Ensure every clip dict has a ``filename`` key. Returns True if any changes made."""
    changed = False
    for c in clips_list:
        expected = _make_clip_filename(c)
        if c.get("filename") != expected:
            c["filename"] = expected
            changed = True
    return changed


def _prepare_music(clips, args):
    """Prepare background music entries for clips if music is enabled."""
    if not args.music:
        return {}

    from .music import get_available_music, download_music_library, match_music_batch
    music_dir = getattr(args, 'music_dir', None) or "music"
    available = get_available_music(music_dir)

    if not available:
        log("INFO", "No music files found — attempting auto-download from Pixabay...")
        downloaded = download_music_library(music_dir=music_dir)
        if downloaded:
            log("OK", f"Downloaded {len(downloaded)} music tracks")
            available = get_available_music(music_dir)

    if not available:
        assets_music = Path(__file__).parent.parent / "assets" / "background_music.mp3"
        if assets_music.exists():
            log("INFO", f"Using fallback background music from assets: {assets_music}")
            fallback = {"id": "background_music", "file": str(assets_music), "description": "Background music", "mood": "neutral"}
            return {c.get("rank", 0): fallback for c in clips}
        log("WARN", "No background music files found. "
                     "Set PIXABAY_API_KEY env var to auto-download, "
                     "or place .mp3 files in music/ directory. Skipping music.")
        return {}

    log("INFO", f"Matching background music for {len(clips)} clips "
                f"({len(available)} tracks available)...")
    music_entries = match_music_batch(
        clips, available,
        llm_model=args.llm_model,
        api_key=args.api_key,
    )
    matched = sum(1 for c in clips if c.get("rank") in music_entries)
    log("OK", f"Music matched for {matched}/{len(clips)} clips")
    return music_entries


def _prepare_subtitles(clips, segments, args, detected_language):
    """Prepare subtitle words for all clips."""
    if not args.subtitles or not segments:
        return

    from .subtitles import get_clip_words
    
    # Always enable fix_errors - even if English, fix Whisper transcription errors
    log("INFO", "Fixing Whisper transcription errors and translating subtitles to English...")
    from .llm import translate_subtitle_words
    
    for clip in clips:
        try:
            raw_words = get_clip_words(
                segments,
                clip_start=clip["start"],
                clip_end=clip["end"],
            )
            # Always use fix_errors=True to fix Whisper mistakes
            clip["_subtitle_words"] = translate_subtitle_words(
                raw_words,
                llm_model=args.llm_model,
                api_key=args.api_key,
                fix_errors=True,  # Enable Whisper error fixing
            )
        except Exception as e:
            log("WARN", f"Could not translate subtitles for clip #{clip['rank']}: {e}")
            clip["_subtitle_words"] = []


def _get_crop_target_from_orientation(orientation: str) -> str:
    """Map output orientation to crop target for person detection.
    
    Args:
        orientation: "auto", "portrait", or "landscape"
    
    Returns:
        crop_target: "vertical", "horizontal", or "square"
    """
    if orientation in ("portrait", "vertical"):
        return "vertical"
    elif orientation in ("landscape", "horizontal"):
        return "horizontal"
    else:  # "auto" or "square"
        return "vertical"  # Default to vertical for social media


def main() -> None:
    """Main CLI entry point."""
    # Load .env file for API keys and config
    load_dotenv()

    # Load configuration from config.yaml
    load_config()
    defaults = get_defaults()

    ap = argparse.ArgumentParser(
        description="AI Video Clipper — English-optimized, auto clip count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  OPENROUTER_API_KEY   OpenRouter key (default backend, free model)\n"
            "  OPENROUTER_MODEL     Override default model on OpenRouter\n"
            "  ANTHROPIC_API_KEY    Anthropic Claude key\n"
            "  OPENAI_API_KEY       OpenAI key\n"
            "  OLLAMA_MODEL         Local Ollama model name (default: llama3.1)\n"
            "\n"
            "Configuration:\n"
            "  Place config.yaml in project root to override defaults.\n"
            "  See config.yaml.example for all options.\n"
        ),
    )
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--model", default=defaults["whisper_model"],
                    choices=["tiny", "base", "small", "medium",
                             "large-v2", "large-v3", "distil-large-v3", "turbo"],
                    help=f"Whisper model size (default: {defaults['whisper_model']})")
    ap.add_argument("--lang", default=defaults["language"],
                    help=f"Language code — 'en' English, 'id' Indonesian, "
                         f"or None for auto-detect (default: {defaults['language']})")
    ap.add_argument("--min", type=int, default=defaults["min_clip_duration"],
                    help=f"Min clip duration in seconds (default: {defaults['min_clip_duration']})")
    ap.add_argument("--max", type=int, default=defaults["max_clip_duration"],
                    help=f"Max clip duration in seconds (default: {defaults['max_clip_duration']})")
    ap.add_argument("--max-clips", type=int, default=defaults["max_clips"],
                    help=f"Maximum number of clips (default: {defaults['max_clips']})")
    ap.add_argument("--min-score", type=int, default=defaults["min_score"],
                    help=f"Minimum engagement score to keep a clip (default: {defaults['min_score']})")
    ap.add_argument("--device", default=defaults["device"],
                    choices=["auto", "cuda", "cpu"],
                    help=f"Compute device (default: {defaults['device']})")
    ap.add_argument("--compute-type", default=defaults["compute_type"],
                    choices=["auto", "float16", "int8", "int8_float16"],
                    help=f"Compute type (default: {defaults['compute_type']})")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable VAD filtering for transcription" if defaults["vad_enabled"] else " (VAD disabled by config)")
    ap.add_argument("--vad-min-silence", type=int, default=defaults["vad_min_silence_ms"],
                    help=f"VAD min silence duration in ms (default: {defaults['vad_min_silence_ms']})")
    ap.add_argument("--vad-speech-pad", type=int, default=defaults["vad_speech_pad_ms"],
                    help=f"VAD speech padding in ms (default: {defaults['vad_speech_pad_ms']})")
    ap.add_argument("--batch", type=int, default=defaults["batch_size"],
                    help=f"Whisper batch size (default: {defaults['batch_size']}; lower if OOM)")
    ap.add_argument("--workers", type=int, default=1,
                    help="(ignored) previously used for parallel ffmpeg workers")
    ap.add_argument("--chunk-duration", type=float, default=defaults["chunk_duration"],
                    help=f"LLM chunk duration in seconds (default: {defaults['chunk_duration']})")
    ap.add_argument("--chunk-overlap", type=float, default=defaults["chunk_overlap"],
                    help=f"Overlap between chunks in seconds (default: {defaults['chunk_overlap']})")
    ap.add_argument("--output", default=defaults["output_dir"],
                    help=f"Output directory (default: ./{defaults['output_dir']}/)")
    ap.add_argument("--save-transcript", action="store_true",
                    help="Save full transcript JSON")
    ap.add_argument("--api-key", default=None,
                    help="API key (overrides env vars)")
    ap.add_argument("--llm-model", default=None,
                    help="Override LLM model name for OpenRouter")

    # ── Post-processing options ──────────────────────────────────────────
    ap.add_argument("--subtitles", action=argparse.BooleanOptionalAction,
                    default=defaults["subtitles_enabled"],
                    help=f"TikTok-style word-by-word subtitles "
                         f"(default: {'on' if defaults['subtitles_enabled'] else 'off'})")
    ap.add_argument("--subtitle-position", default=defaults["subtitle_position"],
                    choices=["center", "upper", "lower"],
                    help=f"Subtitle position (default: {defaults['subtitle_position']})")
    ap.add_argument("--title", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Overlay clip title at the top of the video (default: off)")
    ap.add_argument("--orientation", default="auto",
                    choices=["auto", "portrait", "landscape"],
                    help="Force output orientation: portrait (9:16), landscape (16:9), "
                         "or auto to keep original (default: auto)")

    # ── Person detection & crop ──────────────────────────────────────────
    ap.add_argument("--crop", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Enable YOLO person detection + close-up crop (default: off, uses simple center crop for orientation)")
    ap.add_argument("--crop-target", default="vertical",
                    choices=["vertical", "horizontal", "square"],
                    help=f"Target aspect ratio for crop (default: vertical)")
    ap.add_argument("--split-screen", action="store_true", default=False,
                    help="Split-screen layout: face close-up on top, gameplay on bottom "
                         "(for landscape→vertical; implies --crop)")

    # ── Background music ─────────────────────────────────────────────────
    ap.add_argument("--music", action=argparse.BooleanOptionalAction,
                    default=defaults["music_enabled"],
                    help=f"Add background music matched to clip mood "
                         f"(default: {'on' if defaults['music_enabled'] else 'off'})")
    ap.add_argument("--music-dir", default=defaults["music_dir"],
                    help=f"Directory containing music files (default: ./{defaults['music_dir']}/)")
    ap.add_argument("--music-volume", type=float, default=defaults["music_volume"],
                    help=f"Background music volume 0.0-1.0 (default: {defaults['music_volume']})")

    # ── Encoding quality & speed ─────────────────────────────────────────
    ap.add_argument("--encoding-preset", default=defaults.get("encoding_preset", "veryfast"),
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                    help="ffmpeg x264 encoding preset (default: from config or veryfast)")
    ap.add_argument("--encoding-crf", type=int, default=defaults.get("encoding_crf", 23),
                    help="ffmpeg x264 CRF quality (18-28, lower=better; default: from config or 23)")

    # ── Silence removal ──────────────────────────────────────────────────
    ap.add_argument("--remove-silence", action=argparse.BooleanOptionalAction,
                    default=defaults["silence_removal_enabled"],
                    help=f"Remove silent gaps from clips "
                         f"(default: {'on' if defaults['silence_removal_enabled'] else 'off'})")
    ap.add_argument("--max-silence", type=float, default=defaults["max_silence_duration"],
                    help=f"Maximum silence gap in seconds before removal (default: {defaults['max_silence_duration']})")

    # ── Instagram CTA ────────────────────────────────────────────────────
    _cta_defaults = get_cta_settings()
    ap.add_argument("--cta", action=argparse.BooleanOptionalAction,
                    default=_cta_defaults.get("enabled", False),
                    help="Append Instagram follow CTA at the end of each clip "
                         f"(default: {'on' if _cta_defaults.get('enabled') else 'off'})")

    # ── Caching & skip options ───────────────────────────────────────────
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip processing if final output file already exists")

    # ── Testing options ──────────────────────────────────────────────────
    ap.add_argument("--example", action="store_true",
                    help="Load example clips from file (skip transcription/LLM)")
    ap.add_argument("--example-count", type=int, default=3,
                    help="Number of example clips to use (default: 3)")
    ap.add_argument("--clips-file", default="clips/clips.json", metavar="FILE",
                    help="Path to clips JSON file for example mode (default: clips/clips.json)")

    args = ap.parse_args()
    lang = None if args.lang.lower() == "none" else args.lang
    video = Path(args.video)

    # --split-screen implies --crop (needs person detection)
    if args.split_screen:
        args.crop = True

    # ── Example mode: skip to extraction ─────────────────────────────────
    if args.example:
        if not video.exists():
            log("ERROR", f"File not found: {video}")
            sys.exit(1)

        output_dir = Path(args.output) / video.stem
        clips_file = output_dir / "clips.json"
        if not clips_file.exists():
            log("ERROR", f"Clips file not found: {clips_file}")
            sys.exit(1)

        all_clips = json.loads(clips_file.read_text())
        if _ensure_filenames(all_clips):
            clips_file.write_text(json.dumps(all_clips, indent=2, ensure_ascii=False))
            log("OK", f"Augmented existing clips with filenames → {clips_file}")
        clips = all_clips[:args.example_count]

        print(f"\n{BOLD}{CYAN}{'═' * 50}")
        print(f"   AI Video Clipper — Example Mode (Testing)")
        print(f"{'═' * 50}{RESET}")
        print(f"  Video     : {video.name}")
        print(f"  Clips     : {len(clips)} example clips (from {clips_file.name})")

        pp_features = []
        if args.subtitles: pp_features.append("TikTok Subs")
        if args.title: pp_features.append("Title Overlay")
        if args.orientation != "auto": pp_features.append(f"Force {args.orientation}")
        if args.split_screen: pp_features.append("Split-Screen")
        elif args.crop: pp_features.append(f"Crop({args.crop_target})")
        if args.music: pp_features.append("Music")
        if args.remove_silence: pp_features.append("Silence-Rm")
        if args.cta: pp_features.append("Instagram-CTA")
        print(f"  Features  : {', '.join(pp_features) or 'Raw clips only'}")
        print()

        # Summary table
        print(f"{BOLD}{'#':<4} {'Score':<6} {'E/H/R/P':<16} {'Start':>7} {'End':>7} {'Dur':>5}  Topic{RESET}")
        print("─" * 90)
        for c in clips:
            d = c["end"] - c["start"]
            se = c.get("score_emotion", "?")
            sh = c.get("score_hook", "?")
            sr = c.get("score_retention", "?")
            sp = c.get("score_personality", "?")
            print(f"  {c['rank']:<3} {c.get('clip_score', '?'):<6} "
                  f"{se}/{sh}/{sr}/{sp}  "
                  f"{c['start']:>7.1f} {c['end']:>7.1f} {d:>4.0f}s  {c.get('topic', c['title'])}")
        print()

        t_total = time.time()

        raw_outputs = extract_clips(
            str(video),
            clips,
            output_dir=output_dir,
            max_workers=args.workers,
        )

        # Load cached transcript
        cache_path = _get_transcript_cache_path(str(video))
        detected_language = {"language": "unknown", "language_probability": 0.0}
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
            segments = []
            for clip in clips:
                segments.append({
                    "id": 0, "seek": 0,
                    "start": clip["start"], "end": clip["end"],
                    "text": f"{clip.get('title', '')} - {clip.get('topic', '')}",
                    "tokens": [], "temperature": 0.0,
                    "avg_logprob": 0.0, "compression_ratio": 0.0,
                    "no_speech_prob": 0.0, "words": [],
                })

        # Prepare subtitles
        _prepare_subtitles(clips, segments, args, detected_language)

        # Prepare music
        music_entries = _prepare_music(clips, args)

        # Post-process
        _cta_cfg = {**_cta_defaults, "enabled": args.cta}
        log("DEBUG", f"Example mode postprocess: orientation={args.orientation}, crop={args.crop}")
        any_postprocess = args.subtitles or args.title or args.orientation != "auto" or args.crop or args.music or args.remove_silence or args.cta
        if any_postprocess and raw_outputs:
            outputs = postprocess_clips(
                raw_outputs,
                clips,
                segments,
                output_dir=output_dir,
                subtitles=args.subtitles,
                subtitle_position=args.subtitle_position,
                enable_title=args.title,
                orientation=args.orientation,
                enable_crop=args.crop,
                crop_target=args.crop_target,
                enable_split_screen=args.split_screen,
                enable_music=args.music,
                music_entries=music_entries,
                music_volume=args.music_volume,
                enable_silence_removal=args.remove_silence,
                max_silence=args.max_silence,
                cta_config=_cta_cfg,
                encoding_preset=args.encoding_preset,
                encoding_crf=args.encoding_crf,
            )
        else:
            outputs = raw_outputs

        # Save metadata (preserve all_clips in file, only processed clips subset)
        meta = save_clips_to_disk(all_clips, output_dir)

        elapsed_total = time.time() - t_total
        print(f"\n{GREEN}{BOLD}✓ Done!{RESET} "
              f"{len(outputs)}/{len(clips)} clips extracted (from {len(all_clips)} total) → {output_dir}/ "
              f"({elapsed_total:.0f}s total)")
        if any_postprocess:
            print(f"  Enhanced  : {', '.join(pp_features)}")
        print(f"  Metadata  → {meta}")
        return

    if not video.exists():
        log("ERROR", f"File not found: {video}")
        sys.exit(1)

    output_dir = Path(args.output) / video.stem

    # ── Check for existing output ───────────────────────────────────────────
    if args.skip_existing:
        clips_json = output_dir / "all_clips.json"
        if not clips_json.exists():
            clips_json = output_dir / "clips.json"
        if clips_json.exists():
            clips_data = json.loads(clips_json.read_text())
            if clips_data:
                final_filename = clips_data[0].get("filename")
                if final_filename:
                    final_file = output_dir / final_filename
                    if final_file.exists():
                        log("OK", f"Output already exists: {final_file}")
                        log("OK", "Skipping processing (use without --skip-existing to force re-run)")
                        best_clip = clips_data[0]
                        print(f"\n{GREEN}{BOLD}✓ Complete!{RESET}")
                        print(f"  Output: {final_file}")
                        print()
                        print(f"{BOLD}Topic:{RESET} {best_clip.get('topic', 'N/A')}")
                        print(f"{BOLD}Title:{RESET} {best_clip.get('title', 'N/A')}")
                        print(f"{BOLD}Caption:{RESET} {best_clip.get('caption', 'N/A')}")
                        return

    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   AI Video Clipper — English-optimized")
    print(f"{'═' * 50}{RESET}")
    print(f"  Video     : {video.name}")
    print(f"  Model     : {args.model}")
    print(f"  Language  : {lang or 'auto-detect'}")
    print(f"  Duration  : {args.min}–{args.max}s per clip")
    print(f"  Max clips : {args.max_clips} (LLM decides actual count)")
    # Show post-processing features
    pp_features = []
    if args.subtitles: pp_features.append("TikTok Subs")
    if args.title: pp_features.append("Title Overlay")
    if args.orientation != "auto": pp_features.append(f"Force {args.orientation}")
    if args.crop: pp_features.append(f"Crop({args.crop_target})")
    if args.music: pp_features.append("Music")
    if args.remove_silence: pp_features.append("Silence-Rm")
    print(f"  Features  : {', '.join(pp_features) or 'Raw clips only'}")
    print()

    # ── 1. Transcribe ────────────────────────────────────────────────────────
    t_total = time.time()

    cache_path = _get_transcript_cache_path(str(video))
    detected_language = {"language": "unknown", "language_probability": 0.0}
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
            model_size=args.model,
            language=lang,
            device=args.device,
            compute_type=args.compute_type,
            vad_filter=not args.no_vad,
            vad_min_silence_ms=args.vad_min_silence,
            vad_speech_pad_ms=args.vad_speech_pad,
            batch_size=args.batch,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {"segments": segments, "language_info": detected_language}
        cache_path.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
        log("OK", f"Transcript cached → {cache_path}")

    if args.save_transcript:
        output_dir.mkdir(parents=True, exist_ok=True)
        tx = output_dir / "transcript.json"
        tx.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        log("OK", f"Transcript → {tx}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    filtered, stats = prefilter_segments(segments)

    print(f"\n{BOLD}Pre-filter:{RESET}")
    print(f"  {stats['original']} → {stats['after_filter']} after filter "
          f"({stats['dropped']} dropped, {stats['drop_pct']})"
          f" → {stats['kept']} after merge ({stats['merged']} merged)")
    if stats["reasons"]:
        print(f"  Reasons: {', '.join(f'{k}={v}' for k, v in stats['reasons'].items())}")
    print()

    if not filtered:
        log("ERROR", "All segments filtered out. Try a different whisper model or looser filters.")
        sys.exit(1)

    # ── 2b. Audio energy analysis ─────────────────────────────────────────
    energy_events = analyze_audio_energy(str(video), segments=filtered)

    # ── 3. LLM analysis ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_cache_file = output_dir / "clips.json"
    raw_clips_cache_file = output_dir / ".clips_raw.json"
    clips_from_cache = False

    if clips_cache_file.exists():
        log("INFO", f"Loading cached clips from {clips_cache_file}")
        clips = json.loads(clips_cache_file.read_text())
        if _ensure_filenames(clips):
            clips_cache_file.write_text(json.dumps(clips, indent=2, ensure_ascii=False))
            log("OK", f"Patched filenames in cache → {clips_cache_file}")
        log("OK", f"Loaded {len(clips)} clips from cache (skipped LLM, tighten, improve)")
        clips_from_cache = True
    else:
        video_dur = _get_video_duration(str(video))
        clips = find_clips(
            filtered,
            min_duration=args.min,
            max_duration=args.max,
            max_clips=min(args.max_clips, MAX_CLIPS_HARD_LIMIT),
            min_score=args.min_score,
            llm_model=args.llm_model,
            api_key=args.api_key,
            video_duration=video_dur,
            chunk_duration=args.chunk_duration,
            chunk_overlap=args.chunk_overlap,
            raw_clips_cache_file=raw_clips_cache_file,
            energy_events=energy_events,
        )
        _ensure_filenames(clips)

    if not clips:
        log("WARN", "No engaging clips found. Exiting.")
        sys.exit(0)

    if not clips_from_cache:
        # Smart-adjust clip boundaries for viral-worthy hooks and natural endings
        original_durations = [c["end"] - c["start"] for c in clips]
        clips = smart_adjust_clip_boundaries(
            clips, segments,
            min_duration=5.0,
            max_duration=float(args.max),
            validate_hook_closing=True,
            aggressive_optimization=True,  # Actively find power words and best endings
        )
        new_durations = [c["end"] - c["start"] for c in clips]
        time_saved = sum(original_durations) - sum(new_durations)
        if time_saved > 1:
            log("OK", f"Removed {time_saved:.1f}s of gaps/filler (avg {time_saved/len(clips):.1f}s per clip)")
        else:
            log("OK", "Clip boundaries optimized for viral hooks and natural endings")

        # ── 3b. Improve and fix clips ──────────────────────────────────────
        log("INFO", "Improving clips: translate to English, fix captions, deduplicate topics...")
        clips = fix_and_improve_clips(
            clips,
            llm_model=args.llm_model,
            api_key=args.api_key,
            detected_language=detected_language,
        )
        _ensure_filenames(clips)
        log("OK", f"Clip improvement complete: {len(clips)} clips after deduplication")

    # Save metadata early
    meta = save_clips_to_disk(clips, output_dir)
    log("OK", f"Metadata saved early → {meta}")

    # Summary table
    print(f"\n{BOLD}{'#':<4} {'Score':<6} {'E/H/R/P':<16} {'Start':>7} {'End':>7} {'Dur':>5}  Topic{RESET}")
    print("─" * 90)
    for c in clips:
        d = c["end"] - c["start"]
        se = c.get("score_emotion", "?")
        sh = c.get("score_hook", "?")
        sr = c.get("score_retention", "?")
        sp = c.get("score_personality", "?")
        print(f"  {c['rank']:<3} {c.get('clip_score', '?'):<6} "
              f"{se}/{sh}/{sr}/{sp}  "
              f"{c['start']:>7.1f} {c['end']:>7.1f} {d:>4.0f}s  {c.get('topic', c['title'])}")
    print()

    # Show captions
    print(f"{BOLD}Captions (ready to paste):{RESET}")
    print("─" * 60)
    for c in clips:
        caption = c.get("caption", "")
        if caption:
            print(f"  {YELLOW}#{c['rank']}{RESET} {c['title']}")
            print(f"     {caption}")
            print()
    print()

    # ── 4. Extract raw clips ─────────────────────────────────────────────────
    raw_outputs = extract_clips(
        str(video),
        clips,
        output_dir=output_dir,
        max_workers=args.workers,
    )

    # ── 5. Prepare subtitles ─────────────────────────────────────────────────
    _prepare_subtitles(clips, segments, args, detected_language)

    # ── 5b. Prepare background music ─────────────────────────────────────────
    music_entries = _prepare_music(clips, args)

    # ── 6. Post-process ──────────────────────────────────────────────────────
    _cta_cfg = {**_cta_defaults, "enabled": args.cta}
    any_postprocess = args.subtitles or args.title or args.orientation != "auto" or args.crop or args.music or args.remove_silence or args.cta
    if any_postprocess and raw_outputs:
        crop_target = _get_crop_target_from_orientation(args.orientation)
        outputs = postprocess_clips(
            raw_outputs,
            clips,
            segments,
            output_dir=output_dir,
            subtitles=args.subtitles,
            subtitle_position=args.subtitle_position,
            enable_title=args.title,
            orientation=args.orientation,
            enable_crop=args.crop,
            crop_target=crop_target,
            enable_split_screen=args.split_screen,
            enable_music=args.music,
            music_entries=music_entries,
            music_volume=args.music_volume,
            enable_silence_removal=args.remove_silence,
            max_silence=args.max_silence,
            cta_config=_cta_cfg,
            encoding_preset=args.encoding_preset,
            encoding_crf=args.encoding_crf,
        )
    else:
        outputs = raw_outputs

    # Save final metadata
    meta = save_clips_to_disk(clips, output_dir)

    elapsed_total = time.time() - t_total
    print(f"\n{GREEN}{BOLD}✓ Done!{RESET} "
          f"{len(outputs)}/{len(clips)} clips extracted → {output_dir}/ "
          f"({elapsed_total:.0f}s total)")
    if any_postprocess:
        print(f"  Enhanced  : {', '.join(pp_features)}")
    print(f"  Metadata  → {meta}")
