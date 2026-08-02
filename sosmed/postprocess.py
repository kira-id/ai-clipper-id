"""
Post-processing orchestrator: subtitles, person detection, silence removal.

Takes raw extracted clips and applies visual/audio enhancements
in a single FFmpeg pass for efficiency.
"""

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import get_defaults
from .subtitles import generate_ass_subtitles, generate_title_overlay
from .utils import get_ffmpeg, get_ffprobe, log


def _escape_ass_path(path: str) -> str:
    """Escape file path for FFmpeg's libass subtitle filter in filter_complex.

    FFmpeg filter parsing is sensitive to separators and special characters.
    In filter_complex, Windows drive colon needs double escaping so it survives
    both graph and filter-option parsing (e.g. ``C\\\\:/Users/.../file.ass``).
    """
    escaped = path.replace("\\", "/")

    # Escape Windows drive-letter colon only (C: -> C\\:).
    if len(escaped) >= 2 and escaped[1] == ":" and escaped[0].isalpha():
        escaped = f"{escaped[0]}\\\\:{escaped[2:]}"

    for ch in "'[];, ":
        escaped = escaped.replace(ch, f"\\{ch}")

    return escaped


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


def _compute_orientation_target(
    src_w: int,
    src_h: int,
    orientation: str,
) -> tuple[int, int] | None:
    """Compute target resolution based on orientation setting.

    Args:
        src_w: Source video width
        src_h: Source video height
        orientation: "vertical", "horizontal", "square", "portrait", "landscape", or "auto"

    Returns:
        Tuple of (target_width, target_height) or None if no conversion needed.
    """
    if orientation == "auto":
        return None

    if orientation in ("vertical", "portrait"):
        return (1080, 1920)
    elif orientation in ("horizontal", "landscape"):
        return (1920, 1080)
    elif orientation == "square":
        return (1080, 1080)

    return None


def _get_video_info(video_path: str) -> dict[str, Any]:
    """Get video width, height, duration, fps, and audio presence via ffprobe."""
    try:
        result = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            log("WARN", f"ffprobe found no video streams in {video_path}")
            return {
                "width": 0, "height": 0, "fps": 0,
                "duration": 0, "has_audio": False, "has_video": False,
            }
        stream = streams[0]
        fmt = data.get("format", {})

        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))

        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)

        dur = float(stream.get("duration", 0) or fmt.get("duration", 0))

        # Check for audio stream and get sample rate + audio duration
        result2 = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type,sample_rate,duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True,
        )
        has_audio = bool(result2.stdout.strip())
        sample_rate = 44100  # default
        audio_duration = 0.0
        if has_audio and result2.stdout.strip():
            # Output format: codec_type,sample_rate,duration
            parts = result2.stdout.strip().split(",")
            if len(parts) >= 2:
                try:
                    sample_rate = int(parts[1])
                except ValueError:
                    pass
            if len(parts) >= 3:
                try:
                    audio_duration = float(parts[2])
                except ValueError:
                    pass

        return {
            "width": w, "height": h, "fps": fps,
            "duration": dur, "has_audio": has_audio, "has_video": True,
            "sample_rate": sample_rate,
            "audio_duration": audio_duration,
        }
    except Exception as e:
        log("WARN", f"ffprobe failed: {e}")
        return {
            "width": 0, "height": 0, "fps": 0,
            "duration": 0, "has_audio": False, "has_video": False,
        }


def _postprocess_one(
    raw_clip_path: str,
    clip: dict[str, Any],
    segments: list[dict[str, Any]],
    output_dir: Path,
    *,
    subtitles: bool = True,
    subtitle_position: str = "lower",
    subtitle_margin_pct: float | None = None,
    subtitle_font_size_pct: float | None = None,
    enable_title: bool = False,
    orientation: str = "auto",
    enable_crop: bool = False,
    crop_target: str = "vertical",
    enable_split_screen: bool = False,
    enable_active_speaker: bool = True,
    enable_silence_removal: bool = False,
    max_silence: float = 1.5,
    cta_config: dict[str, Any] | None = None,
    encoding_preset: str | None = None,
    encoding_crf: int | None = None,
    use_hwaccel: bool = True,
) -> str:
    """Post-process a single clip with all enhancements.

    Features applied in order:
    1. Person detection + crop (if enabled)
    2. Silence removal (if enabled)
    3. Subtitles overlay
    4. Title overlay (if enabled)
    5. Audio loudnorm

    Returns the path to the post-processed clip.
    """
    raw_path = Path(raw_clip_path)
    out_path = output_dir / f"{raw_path.stem}_final.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)

    info = _get_video_info(raw_clip_path)
    if not info.get("has_video", True):
        log("WARN", f"Clip #{clip.get('rank')} has no video stream — skipping postprocess")
        return raw_clip_path
    src_w, src_h = info["width"], info["height"]
    clip_duration = info["duration"]
    has_audio = info["has_audio"]

    if clip_duration <= 0:
        clip_duration = clip["end"] - clip["start"]

    out_w = src_w
    out_h = src_h

    # ── 0. Determine output resolution (orientation conversion) ──────────
    orient_target = _compute_orientation_target(src_w, src_h, orientation)
    orient_scale_filter: str | None = None
    if orient_target:
        out_w, out_h = orient_target
        log("DEBUG", f"Clip #{clip.get('rank')}: {src_w}x{src_h} → {out_w}x{out_h} ({orientation})")
        # Only add scale when the source dimensions don't already match the target
        if (src_w, src_h) != (out_w, out_h):
            if enable_crop:
                # Scale to fill (crop-to-fill, no black bars): scale up so the
                # shorter dimension meets the target, then center-crop the overflow.
                orient_scale_filter = (
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                    f"crop={out_w}:{out_h}"
                )
            else:
                # Scale to fit with black bars (letterbox/pillarbox): scale down
                # so the larger dimension meets the target, pad the remaining space.
                orient_scale_filter = (
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
                )
        else:
            log("DEBUG", f"Clip #{clip.get('rank')}: source already matches target, no scale needed")
    else:
        log("DEBUG", f"Clip #{clip.get('rank')}: orientation={orientation}, no conversion needed")

    # ── 0b. Person detection + crop ───────────────────────────────────────────
    # Run person-detection crop ONLY when explicitly requested (enable_crop).
    # Orientation conversion uses orient_scale_filter (simple center crop), not person detection.
    crop_filter = None
    split_screen = False
    split_screen_detections: list[dict] | None = None
    _effective_crop_target = crop_target if enable_crop else None

    if _effective_crop_target:
        from .person_detection import (
            detect_persons_in_clip, compute_crop_region,
            build_crop_filter, needs_crop,
        )
        aspect_map = {"vertical": 9 / 16, "horizontal": 16 / 9, "square": 1.0}
        target_aspect = aspect_map.get(_effective_crop_target, 9 / 16)

        if needs_crop(src_w, src_h, _effective_crop_target):
            log("DEBUG", f"Clip #{clip.get('rank')}: detecting persons for {_effective_crop_target} crop...")
            detections = detect_persons_in_clip(raw_clip_path, sample_interval=1.0)

            if not detections:
                # Person detection required but failed - no fallback allowed
                raise RuntimeError(
                    f"Clip #{clip.get('rank')}: Person detection failed for {_effective_crop_target} crop. "
                    f"Video may have codec issues or contain no detectable persons."
                )

            if _effective_crop_target == "vertical":
                target_w, target_h = 1080, 1920
            elif _effective_crop_target == "horizontal":
                target_w, target_h = 1920, 1080
            else:
                target_w, target_h = 1080, 1080

            # Split-screen layout: face close-up on top, gameplay on bottom
            if enable_split_screen and src_w > src_h:
                split_screen = True
                split_screen_detections = detections
                log("DEBUG", f"Clip #{clip.get('rank')}: using split-screen layout (face + gameplay)")
            else:
                # Use dynamic crop regions with interpolation between detections
                from .person_detection import build_dynamic_crop_filter

                if enable_active_speaker:
                    # Follow the PERSON WHO IS SPEAKING, not the largest person.
                    # Required for podcasts (2+ speakers on a single mono track):
                    # re-identify each person and pan to the active speaker.
                    from .active_speaker import compute_active_speaker_crop_regions
                    crop_regions = compute_active_speaker_crop_regions(
                        raw_clip_path, detections, src_w, src_h,
                        target_aspect=target_aspect,
                        segment_duration=1.0,  # 1-second segments for smooth tracking
                        smoothing_window=5,
                        fps=max(1.0, info.get("fps", 30.0) or 30.0),
                    )
                else:
                    # Largest-person tracking (legacy behaviour, no audio analysis)
                    from .person_detection import compute_dynamic_crop_regions
                    crop_regions = compute_dynamic_crop_regions(
                        detections, src_w, src_h,
                        target_aspect=target_aspect,
                        segment_duration=1.0,
                        smoothing_window=5,
                    )

                if not crop_regions:
                    raise RuntimeError(
                        f"Clip #{clip.get('rank')}: Could not compute dynamic crop regions from detected persons "
                        f"for {_effective_crop_target} crop."
                    )

                log("DEBUG", f"Clip #{clip.get('rank')}: computed {len(crop_regions)} dynamic crop regions")

                # Build dynamic crop filter with zoompan for smooth transitions
                crop_filter = build_dynamic_crop_filter(
                    crop_regions, src_w, src_h,
                    target_w, target_h,
                )
            out_w, out_h = target_w, target_h

    # ── 1. Silence removal ───────────────────────────────────────────────────
    silence_filter_v = None
    silence_filter_a = None
    subtitle_words = clip.get("_subtitle_words") or []

    if enable_silence_removal and subtitle_words:
        from .silence_removal import (
            compute_silence_removal, build_silence_removal_filter,
            adjust_subtitle_times,
        )
        keep_regions = compute_silence_removal(
            subtitle_words, clip_duration,
            max_silence=max_silence,
            min_kept_duration=5.0,
        )
        if keep_regions:
            silence_filter_v, silence_filter_a = build_silence_removal_filter(keep_regions)
            # Adjust subtitle word times for the shortened clip
            subtitle_words = adjust_subtitle_times(subtitle_words, keep_regions)
            new_duration = sum(end - start for start, end in keep_regions)
            log("DEBUG", f"Clip #{clip.get('rank')}: silence removal "
                         f"{clip_duration:.1f}s → {new_duration:.1f}s")
            clip_duration = new_duration

    # ── 2. Generate subtitles ────────────────────────────────────────────────
    ass_path = None
    if subtitles:
        words = subtitle_words
        if words:
            ass_content = generate_ass_subtitles(
                words,
                play_res_x=out_w,
                play_res_y=out_h,
                position=subtitle_position,
                subtitle_margin_pct=subtitle_margin_pct,
                font_size_pct=subtitle_font_size_pct if subtitle_font_size_pct is not None else 3.2,
            )
            tmp = tempfile.NamedTemporaryFile(
                suffix=".ass", prefix="sosmed_sub_",
                delete=False, mode="w", encoding="utf-8",
            )
            tmp.write(ass_content)
            tmp.close()
            ass_path = tmp.name

    # ── 3. Generate title overlay ────────────────────────────────────────────
    title_ass_path = None
    title = clip.get("title") or clip.get("topic") or ""
    if enable_title and title:
        title_content = generate_title_overlay(
            title,
            play_res_x=out_w,
            play_res_y=out_h,
            duration=3.0,
        )
        tmp_title = tempfile.NamedTemporaryFile(
            suffix=".ass", prefix="sosmed_title_",
            delete=False, mode="w", encoding="utf-8",
        )
        tmp_title.write(title_content)
        tmp_title.close()
        title_ass_path = tmp_title.name

    # ── 4. Build complete FFmpeg command ─────────────────────────────────────
    cmd: list[str] = [get_ffmpeg(), "-y", "-hide_banner"]
    cmd.extend(["-i", raw_clip_path])

    # Build filter_complex
    filter_parts: list[str] = []
    current_v_label = "[0:v]"
    current_a_label = "[0:a]"
    label_counter = 0

    def _next_label(prefix: str = "tmp") -> str:
        nonlocal label_counter
        label_counter += 1
        return f"[{prefix}{label_counter}]"

    # Video filter chain
    has_video_filters = False

    if split_screen:
        # Split-screen layout: face close-up on top, gameplay on bottom
        from .person_detection import build_split_screen_filter

        v_input = current_v_label
        if silence_filter_v:
            filter_parts.append(f"{v_input}{silence_filter_v}[v_pre_split]")
            v_input = "[v_pre_split]"

        # Post-split linear filters (subtitles, title)
        post_split: list[str] = []
        if ass_path:
            post_split.append(f"ass={_escape_ass_path(ass_path)}")
        if title_ass_path:
            post_split.append(f"ass={_escape_ass_path(title_ass_path)}")

        split_out = "[split_out]" if post_split else "[vout]"
        filter_parts.append(build_split_screen_filter(
            split_screen_detections, src_w, src_h, out_w, out_h,
            input_label=v_input, output_label=split_out,
        ))

        if post_split:
            cur = split_out
            for i, pf in enumerate(post_split):
                out_lbl = "[vout]" if i == len(post_split) - 1 else f"[vp{i}]"
                filter_parts.append(f"{cur}{pf}{out_lbl}")
                cur = out_lbl

        has_video_filters = True
    else:
        # Standard linear video filter chain
        vfilters_chain: list[str] = []

        # Silence removal (video)
        if silence_filter_v:
            vfilters_chain.append(silence_filter_v)

        # Crop filter (includes its own scale to target resolution)
        if crop_filter:
            vfilters_chain.append(crop_filter)
        elif orient_scale_filter:
            # Orientation conversion without person-crop
            vfilters_chain.append(orient_scale_filter)

        # Subtitle filters
        if ass_path:
            vfilters_chain.append(f"ass={_escape_ass_path(ass_path)}")
        if title_ass_path:
            vfilters_chain.append(f"ass={_escape_ass_path(title_ass_path)}")

        # Build video filter chain with labels
        if vfilters_chain:
            if len(vfilters_chain) == 1:
                filter_parts.append(f"{current_v_label}{vfilters_chain[0]}[vout]")
            else:
                parts = []
                for i, vf in enumerate(vfilters_chain):
                    if i == 0:
                        out_label = _next_label("v")
                        parts.append(f"{current_v_label}{vf}{out_label}")
                    elif i == len(vfilters_chain) - 1:
                        parts.append(f"{out_label}{vf}[vout]")
                    else:
                        new_label = _next_label("v")
                        parts.append(f"{out_label}{vf}{new_label}")
                        out_label = new_label
                filter_parts.append(";".join(parts))

        has_video_filters = bool(vfilters_chain)

    # Audio filter chain
    afilters: list[str] = []

    # Silence removal (audio)
    if silence_filter_a:
        afilters.append(silence_filter_a)

    # Audio normalization for social media speech — loud and clear.
    # Target: -13 LUFS (matches TikTok/Instagram/YouTube normalization).
    # LRA=7 keeps speech tight and consistent — quiet parts stay intelligible.
    # TP=-1.0 dB prevents harsh limiting; platforms normalize down, not up.
    # No pre-amp needed — source audio already has good levels.
    if has_audio:
        afilters.append("loudnorm=I=-13:LRA=7:TP=-1.0")

    if has_audio and afilters:
        afilter_str = ",".join(afilters)
        filter_parts.append(f"{current_a_label}{afilter_str}[aout]")

    if filter_parts:
        full_filter = ";".join(filter_parts)
        cmd.extend(["-filter_complex", full_filter])

        if has_video_filters:
            cmd.extend(["-map", "[vout]"])
        else:
            cmd.extend(["-map", "0:v:0"])

        if has_audio:
            cmd.extend(["-map", "[aout]"])
        else:
            cmd.append("-an")
    else:
        cmd.extend(["-map", "0:v:0"])
        if has_audio:
            cmd.extend(["-map", "0:a:0?"])
        else:
            cmd.append("-an")

    # Encoding settings
    if has_audio:
        audio_enc = ["-c:a", "aac", "-b:a", "192k"]
    else:
        audio_enc = ["-c:a", "aac"]

    # Load encoding settings from config if not provided
    if encoding_preset is None or encoding_crf is None:
        defaults = get_defaults()
        if encoding_preset is None:
            encoding_preset = defaults.get("encoding_preset", "veryfast")
        if encoding_crf is None:
            encoding_crf = defaults.get("encoding_crf", 23)

    # Check hardware acceleration setting from config
    defaults = get_defaults()
    use_hwaccel = use_hwaccel and defaults.get("hwaccel", True)

    # Use hardware acceleration on macOS (VideoToolbox) for faster encoding
    # Fall back to libx264 if hwaccel disabled or not available
    if has_video_filters or silence_filter_v:
        if use_hwaccel and _is_videotoolbox_available():
            # VideoToolbox: very fast, good quality for social media
            # Note: h264_videotoolbox doesn't support CRF, use quality/speed instead
            video_enc = ["-c:v", "h264_videotoolbox", "-quality", "speed"]
            log("DEBUG", f"Clip #{clip.get('rank')}: using VideoToolbox hardware acceleration")
        else:
            # libx264 with configurable preset/CRF
            video_enc = ["-c:v", "libx264", "-preset", encoding_preset, "-crf", str(encoding_crf)]
    else:
        video_enc = ["-c:v", "copy"]

    output_flags = ["-shortest", "-movflags", "+faststart", "-loglevel", "error"]

    full_cmd = cmd + video_enc + audio_enc + output_flags + [str(out_path)]
    try:
        subprocess.run(full_cmd, check=True, capture_output=True, text=True)
        log("DEBUG", f"Clip #{clip.get('rank')} post-processed")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_detail = e.stderr[:300] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        log("DEBUG", f"ffmpeg cmd: {' '.join(full_cmd)}")
        raise RuntimeError(f"Post-processing failed for clip #{clip.get('rank', '?')}: {error_detail}")

    # ── Instagram CTA ─────────────────────────────────────────────────────────
    if cta_config and cta_config.get("enabled"):
        from .cta import append_instagram_cta
        append_instagram_cta(
            str(out_path),
            str(out_path),
            name=str(cta_config.get("name", "Samuel Academy")),
            username=str(cta_config.get("username", "@samuelkoesnadi")),
            duration=float(cta_config.get("duration", 3.0)),
            fade_duration=float(cta_config.get("fade_duration", 0.5)),
        )

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if ass_path:
        try:
            os.unlink(ass_path)
        except OSError:
            pass
    if title_ass_path:
        try:
            os.unlink(title_ass_path)
        except OSError:
            pass

    # Remove raw clip (replaced by post-processed version)
    try:
        if out_path.exists() and raw_path.exists() and raw_path != out_path:
            raw_path.unlink()
    except OSError:
        pass

    return str(out_path)


def postprocess_clips(
    raw_clip_paths: list[str],
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_workers: int = 2,
    subtitles: bool = True,
    subtitle_position: str = "lower",
    subtitle_margin_pct: float | None = None,
    subtitle_font_size_pct: float | None = None,
    enable_title: bool = False,
    orientation: str = "auto",
    enable_crop: bool = False,
    crop_target: str = "vertical",
    enable_split_screen: bool = False,
    enable_active_speaker: bool = True,
    enable_silence_removal: bool = False,
    max_silence: float = 1.5,
    cta_config: dict[str, Any] | None = None,
    encoding_preset: str | None = None,
    encoding_crf: int | None = None,
    use_hwaccel: bool = True,
) -> list[str]:
    """Post-process all extracted clips.

    Args:
        raw_clip_paths: Paths to raw extracted clips
        clips: Clip metadata dicts
        segments: Whisper segments for word timestamps
        output_dir: Output directory
        subtitles: Enable subtitle overlay
        subtitle_position: "lower", "center", or "upper"
        subtitle_margin_pct: Margin percentage for subtitle position (e.g. 25 for "lower" position)
        enable_title: Enable title text overlay at top of video
        enable_crop: Enable person-detection crop
        crop_target: "vertical", "horizontal", or "square"
        enable_split_screen: Enable split-screen layout (face on top, gameplay on bottom)
        enable_silence_removal: Enable silence gap removal
        max_silence: Max silence gap to allow (seconds)
    """
    if not raw_clip_paths:
        return []

    # Load CTA config from config.yaml if not explicitly provided
    if cta_config is None:
        from .config import get_cta_settings
        cta_config = get_cta_settings()

    features = []
    if subtitles:
        features.append("subtitles")
    if enable_title:
        features.append("title-overlay")
    if enable_crop:
        features.append(f"crop({crop_target})")
    if enable_split_screen:
        features.append("split-screen")
    if enable_silence_removal:
        features.append("silence-removal")
    if cta_config and cta_config.get("enabled"):
        features.append("instagram-cta")
    log("INFO", f"Post-processing {len(raw_clip_paths)} clips: "
                f"{', '.join(features) or 'none'}")

    rank_to_clip = {c["rank"]: c for c in clips}

    results: list[str] = []
    effective_workers = min(max_workers, max(1, len(raw_clip_paths)))

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {}
        for raw_path in raw_clip_paths:
            fname = Path(raw_path).stem
            rank = None
            for c in clips:
                expected_prefix = f"rank{c['rank']:02d}_"
                if fname.startswith(expected_prefix):
                    rank = c["rank"]
                    break

            if rank is None:
                log("WARN", f"Could not match {fname} to any clip — skipping postprocess")
                results.append(raw_path)
                continue

            clip = rank_to_clip[rank]

            fut = pool.submit(
                _postprocess_one,
                raw_path, clip, segments, output_dir,
                subtitles=subtitles,
                subtitle_position=subtitle_position,
                subtitle_margin_pct=subtitle_margin_pct,
                subtitle_font_size_pct=subtitle_font_size_pct,
                enable_title=enable_title,
                orientation=orientation,
                enable_crop=enable_crop,
                crop_target=crop_target,
                enable_split_screen=enable_split_screen,
                enable_active_speaker=enable_active_speaker,
                enable_silence_removal=enable_silence_removal,
                max_silence=max_silence,
                cta_config=cta_config,
                encoding_preset=encoding_preset,
                encoding_crf=encoding_crf,
                use_hwaccel=use_hwaccel,
            )
            futures[fut] = clip

        failures: list[str] = []
        for fut in as_completed(futures):
            clip = futures[fut]
            try:
                out = fut.result()
                mb = os.path.getsize(out) / 1_048_576
                clip["filename"] = Path(out).name
                log("OK", f"  #{clip['rank']:>2} {clip['title'][:35]:<35}  "
                          f"post-processed ({mb:.1f} MB)")
                results.append(out)
            except Exception as exc:
                log("ERROR", f"  #{clip['rank']:>2} postprocess failed: {exc}")
                failures.append(f"#{clip.get('rank')}: {exc}")

    # A postprocess failure previously just logged and dropped the clip, so the
    # caller silently shipped the RAW extracted video — no subtitles, no crop,
    # no title, and no visible error. Surface it instead.
    if failures and not results:
        raise RuntimeError(
            "Post-processing failed for every clip; no subtitles/crop/title "
            "were applied. First errors: " + " | ".join(failures[:3])
        )
    if failures:
        log("WARN", f"{len(failures)} of {len(futures)} clips failed post-processing "
                    f"and were dropped: {' | '.join(failures[:3])}")

    return sorted(results)


# ── Single-video single-pass render ──────────────────────────────────────────
# The single-video ("subtitles for one video") dashboard used to run the *full*
# clip pipeline on a single clip spanning the whole video: extract_clips re-encodes
# the entire source (4K, ~7 min for a 21-min video) to produce a raw clip, then
# postprocess_clips re-encodes that raw clip AGAIN to burn subtitles + loudnorm.
# Two full-resolution transcodes of the same content — the main reason long videos
# take so long, and the reason a concurrent run clobbered a 7-minute in-progress
# write into a corrupt MP4.
#
# render_single_video collapses that into ONE ffmpeg pass straight from the
# source: seek -> decode -> ass(subtitles) -> loudnorm(audio) -> encode.
# No intermediate raw file is written, halving both wall time and disk churn.
def render_single_video(
    video_path: str,
    output_path: str,
    subtitle_words: list[dict[str, Any]] | None = None,
    *,
    subtitle_position: str = "lower",
    subtitle_margin_pct: float | None = None,
    subtitle_font_size_pct: float | None = None,
    start: float = 0.0,
    end: float | None = None,
    encoding_preset: str | None = None,
    encoding_crf: int | None = None,
    use_hwaccel: bool = True,
) -> str:
    """Render a subtitled, loudnorm'd copy of ``video_path`` in a single pass.

    The source is decoded once and the final MP4 is written directly to
    ``output_path`` — no intermediate raw clip is produced. Uses the exact same
    ASS generator (``generate_ass_subtitles``) and `_escape_ass_path` as the
    multi-clip pipeline, so burnt subtitles look identical across modes.

    Returns ``output_path`` on success; raises RuntimeError on failure.
    """
    defaults = get_defaults()
    if encoding_preset is None:
        encoding_preset = defaults.get("encoding_preset", "veryfast")
    if encoding_crf is None:
        encoding_crf = defaults.get("encoding_crf", 23)

    info = _get_video_info(video_path)
    if not info.get("has_video", True):
        raise RuntimeError(
            f"Source {video_path} has no decodable video stream.")
    src_w, src_h = info["width"], info["height"]
    src_dur = info.get("duration") or 0.0
    has_audio = info.get("has_audio", False)
    play_res_x, play_res_y = src_w, src_h

    # Seek window
    if end is None:
        end = src_dur if src_dur > 0 else 0.0
    end = max(end, start + 0.001)
    seg_duration = end - start

    # ── subtitle ASS (reuse production generator) ─────────────────────────
    ass_path = None
    if subtitle_words and (subtitle_position or True):
        ass_content = generate_ass_subtitles(
            subtitle_words,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            position=subtitle_position,
            subtitle_margin_pct=subtitle_margin_pct,
            font_size_pct=(subtitle_font_size_pct
                           if subtitle_font_size_pct is not None else 3.2),
        )
        tmp = tempfile.NamedTemporaryFile(
            suffix=".ass", prefix="sosmed_single_sub_",
            delete=False, mode="w", encoding="utf-8",
        )
        tmp.write(ass_content)
        tmp.close()
        ass_path = tmp.name

    try:
        cmd: list[str] = [get_ffmpeg(), "-y", "-hide_banner"]
        # Input seeking: accurate to the ms. -ss before -i is fast (keyframe +
        # precise decode); fine for whole-video subtitling where start==0.
        if start > 0:
            cmd += ["-ss", f"{start:.3f}", "-accurate_seek"]
        cmd += ["-i", video_path, "-t", f"{seg_duration:.3f}"]
        cmd += ["-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "0:a:0?"]

        # video filter: subtitles (ass) + loudnorm audio
        vfilters: list[str] = []
        if ass_path:
            vfilters.append(f"ass={_escape_ass_path(ass_path)}")
        afilters: list[str] = []
        if has_audio:
            afilters.append("loudnorm=I=-13:LRA=7:TP=-1.0")

        if vfilters:
            cmd += ["-vf", ",".join(vfilters)]
        if afilters:
            cmd += ["-af", ",".join(afilters)]

        # Encode: VideoToolbox (macOS hwaccel) when available + requested, else
        # libx264 with CRF/preset. We always re-encode here because the ass
        # subtitle filter is a decode-time filter (can't -c:v copy).
        if (use_hwaccel and defaults.get("hwaccel", True)
                and _is_videotoolbox_available()):
            video_enc = ["-c:v", "h264_videotoolbox", "-quality", "speed"]
            log("DEBUG", "Single-pass render: using VideoToolbox hardware acceleration")
        else:
            video_enc = ["-c:v", "libx264", "-preset", encoding_preset,
                         "-crf", str(encoding_crf)]
        cmd += video_enc

        if has_audio:
            audio_enc = ["-c:a", "aac", "-b:a", "192k"]
        else:
            audio_enc = ["-an"]
        cmd += audio_enc

        cmd += ["-shortest", "-movflags", "+faststart",
                "-loglevel", "error", str(output_path)]

        log("DEBUG", "Single-pass render cmd: " + " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log("DEBUG", f"Single-video render complete -> {output_path}")
        return str(output_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        detail = (e.stderr[:300] if isinstance(e, subprocess.CalledProcessError)
                  and e.stderr else str(e))
        log("DEBUG", f"Single-pass render failed: {detail}")
        raise RuntimeError(f"Single-pass render failed: {detail}")
    finally:
        if ass_path:
            try:
                os.unlink(ass_path)
            except OSError:
                pass

