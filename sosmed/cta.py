"""
Instagram CTA — fades to black and shows a follow prompt at the end of clips.

Appends a professional 3-second CTA screen with:
  • Fade-from-video transition to black
  • Instagram logo (assets/iglogo.png, scaled & faded in)
  • Account name (large, white)
  • Username (light blue, with @)
  • "Follow" button (light-blue fill, dark text)
  • Click sound when the button appears
"""

import subprocess
from pathlib import Path

from .postprocess import _get_video_info
from .utils import get_ffmpeg, get_ffprobe, log


def _esc(text: str) -> str:
    """Escape text for FFmpeg drawtext filter value."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    return text


def _fade(t0: float, t1: float) -> str:
    """Smooth 0→1 alpha expression for FFmpeg drawtext."""
    span = t1 - t0
    return f"if(lt(t,{t0:.3f}),0,if(lt(t,{t1:.3f}),(t-{t0:.3f})/{span:.3f},1))"


def append_instagram_cta(
    clip_path: str,
    output_path: str,
    name: str,
    username: str,
    duration: float = 3.0,
    fade_duration: float = 0.5,
) -> str:
    """Append an Instagram follow CTA to the end of a video clip.

    OPTIMIZED: Uses concatenation instead of re-encoding the entire video.
    Only the CTA portion is rendered; the main clip is copied without re-encoding.

    Fades from the clip to a black screen and shows a styled CTA with
    animated text, a follow button, and a click sound effect.

    Args:
        clip_path:     Path to the input (post-processed) clip.
        output_path:   Path to write the final clip with CTA appended.
        name:          Display name of the Instagram account.
        username:      Instagram handle (@ prefix added automatically if missing).
        duration:      Duration of the CTA screen in seconds.
        fade_duration: Length of the cross-fade transition.

    Returns:
        output_path on success; clip_path on failure (leaves input intact).
    """
    info = _get_video_info(clip_path)
    if not info.get("has_video"):
        log("WARN", "CTA: no video stream in clip, skipping")
        return clip_path

    w = info["width"] or 1080
    h = info["height"] or 1920
    fps = max(float(info.get("fps") or 30.0), 15.0)
    main_dur = float(info.get("duration") or 0)
    has_audio = info["has_audio"]
    audio_dur = float(info.get("audio_duration") or 0)

    if main_dur <= 0:
        log("WARN", "CTA: could not determine clip duration, skipping")
        return clip_path

    if username and not username.startswith("@"):
        username = "@" + username

    # Clamp fade so it never exceeds 40 % of the clip or 80% of CTA duration
    fade_duration = min(fade_duration, main_dur * 0.4, duration * 0.8)

    # ── Accent colour ─────────────────────────────────────────────────────────
    ACCENT = "0x29B6F6"   # light blue

    # ── Reference dimension: min(w,h) prevents text overflow on narrow canvases
    ref = min(w, h)
    is_portrait = h > w

    # ── Font sizes ────────────────────────────────────────────────────────────
    name_fs = max(64, int(ref * 0.082))   # ~89 px on ref=1080
    user_fs = max(44, int(ref * 0.054))   # ~58 px
    btn_fs  = max(36, int(ref * 0.042))   # ~45 px

    # ── Instagram logo — bigger on portrait ──────────────────────────────────
    logo_path = Path(__file__).parent.parent / "assets" / "iglogo.png"
    if is_portrait:
        logo_size = max(180, int(ref * 0.230))  # ~248 px on ref=1080
    else:
        logo_size = max(130, int(ref * 0.160))  # ~173 px

    # ── Button dimensions — horizontal padding so it feels like a real button ─
    btn_padding_x = max(40, int(ref * 0.060))   # ~65 px per side
    btn_h = max(70, int(ref * 0.085))           # ~92 px
    # Minimum width = "Follow" text width + 2× padding (rough: 6 chars × ~0.6×btn_fs)
    btn_w = max(int(btn_fs * 6 * 0.6) + 2 * btn_padding_x, max(260, int(ref * 0.420)))
    btn_x = (w - btn_w) // 2

    # ── Vertical layout — gap values are pixel distances between visual edges ─
    # drawtext y = TOP of the text box (cap-height), so each element occupies
    # rows [y .. y + font_size].  Gaps are measured between the bottom of one
    # element and the top of the next.
    gap_logo_name = max(12, int(ref * 0.018))   # logo bottom → name top  (tight)
    gap_name_user = max(10, int(ref * 0.014))   # name bottom → username top
    gap_user_btn  = max(32, int(ref * 0.046))   # username bottom → button top

    # Total group height, centered vertically in the frame
    group_h = (
        logo_size
        + gap_logo_name + name_fs
        + gap_name_user + user_fs
        + gap_user_btn  + btn_h
    )
    group_top = (h - group_h) // 2

    # y = top of each element
    icon_top = group_top
    name_y   = icon_top  + logo_size    + gap_logo_name
    user_y   = name_y    + name_fs      + gap_name_user
    btn_y    = user_y    + user_fs      + gap_user_btn
    btn_ty   = btn_y + (btn_h - btn_fs) // 2

    # ── Animation timing (t = seconds into CTA segment) ──────────────────────
    t_icon = 0.10
    t_name = (0.25, 0.60)
    t_user = (0.42, 0.75)
    t_btn  = 1.00    # button pop-in + click sound

    name_esc = _esc(name)
    user_esc = _esc(username)

    # ── CTA visual filter chain (drawtext/drawbox — no logo here) ───────────
    filters: list[str] = [
        # ── Account name ──────────────────────────────────────────────────────

        # Drop shadow
        (f"drawtext=font=Montserrat:fontsize={name_fs}"
         f":text='{name_esc}'"
         f":x=(w-tw)/2+3:y={name_y + 3}"
         f":fontcolor=0x000000@0.60"
         f":alpha='{_fade(*t_name)}'"),

        # Main text (white)
        (f"drawtext=font=Montserrat:fontsize={name_fs}"
         f":text='{name_esc}'"
         f":x=(w-tw)/2:y={name_y}"
         f":fontcolor=white"
         f":bordercolor=0x111111:borderw=3"
         f":alpha='{_fade(*t_name)}'"),

        # ── @username (light blue) ────────────────────────────────────────────
        (f"drawtext=font=Montserrat:fontsize={user_fs}"
         f":text='{user_esc}'"
         f":x=(w-tw)/2:y={user_y}"
         f":fontcolor={ACCENT}"
         f":bordercolor=0x111111:borderw=2"
         f":alpha='{_fade(*t_user)}'"),

        # ── Follow button (light-blue fill, dark text) ────────────────────────
        (f"drawbox=x={btn_x}:y={btn_y}:w={btn_w}:h={btn_h}"
         f":color={ACCENT}:t=fill"
         f":enable='gte(t,{t_btn:.3f})'"),

        (f"drawtext=font=Montserrat:fontsize={btn_fs}"
         f":text='Follow'"
         f":x=(w-tw)/2:y={btn_ty}"
         f":fontcolor=0x0D1B2A"
         f":enable='gte(t,{t_btn:.3f})'"),
    ]

    cta_chain = ",".join(filters)

    # Click plays at t_btn seconds into the CTA portion
    click_ms = int(t_btn * 1000)

    # Click SFX file (assets/click.mp3)
    click_sfx_path = Path(__file__).parent.parent / "assets" / "click.mp3"
    use_click_file = click_sfx_path.exists()

    # ── OPTIMIZED APPROACH: Generate CTA separately, then concatenate ────────
    # Step 1: Generate CTA clip (only 3 seconds, fast encoding)
    # Step 2: Concatenate main clip (copy) + CTA clip (no re-encoding of main clip)

    ffmpeg = get_ffmpeg()
    out = Path(output_path)
    tmp_cta = out.parent / (out.stem + "_cta_tmp" + out.suffix)
    tmp_final = out.parent / (out.stem + "_final_tmp" + out.suffix)

    try:
        # ── Step 1: Generate CTA clip (black background with overlays) ────────
        cmd_cta: list[str] = [ffmpeg, "-y", "-hide_banner"]

        # [0] black CTA background (lavfi color source)
        cmd_cta.extend([
            "-f", "lavfi", "-t", f"{duration:.4f}",
            "-i", f"color=c=black:size={w}x{h}:rate={fps:.4f}",
        ])

        # [1] click sound
        if use_click_file:
            cmd_cta.extend(["-i", str(click_sfx_path)])
        else:
            click_expr = (
                "0.8*sin(2*PI*1200*t)*exp(-90*t)"
                "+0.4*sin(2*PI*2800*t)*exp(-110*t)"
                "+0.2*sin(2*PI*600*t)*exp(-70*t)"
            )
            cmd_cta.extend([
                "-f", "lavfi", "-t", "0.15",
                "-i", f"aevalsrc={click_expr}:s=44100",
            ])

        # [2] Instagram logo PNG (looped for the CTA duration)
        cmd_cta.extend(["-loop", "1", "-t", f"{duration:.4f}", "-i", str(logo_path)])

        # ── filter_complex for CTA ────────────────────────────────────────────
        fc_cta: list[str] = []

        # Apply drawtext/drawbox to the black background
        fc_cta.append(f"[0:v]{cta_chain}[cta_text]")

        # Scale logo and fade it in, then overlay centered on CTA
        fc_cta.append(
            f"[2:v]scale={logo_size}:{logo_size}"
            f",fade=t=in:st={t_icon:.3f}:d=0.30:alpha=1"
            f",format=rgba[logo_scaled]"
        )
        fc_cta.append(
            f"[cta_text][logo_scaled]overlay"
            f"=x=(W-w)/2:y={icon_top}:format=auto[cta_v]"
        )

        # Audio: click sound with silence padding to fill CTA duration
        if use_click_file:
            # Delay the click sound, then pad with silence to fill duration
            # click_ms is the delay in milliseconds from the start of the CTA
            fc_cta.append(f"[1:a]adelay=delays={click_ms}:all=1,apad=whole_dur={duration:.4f}[cta_a]")
        else:
            fc_cta.append(f"[1:a]adelay=delays={click_ms}:all=1,apad=whole_dur={duration:.4f}[cta_a]")

        cmd_cta.extend(["-filter_complex", ",".join(fc_cta)])
        cmd_cta.extend(["-map", "[cta_v]", "-map", "[cta_a]"])
        # Use same codec settings as postprocess to ensure compatibility
        cmd_cta.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
        cmd_cta.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd_cta.extend(["-movflags", "+faststart", "-loglevel", "error"])
        cmd_cta.append(str(tmp_cta))

        log("DEBUG", f"CTA: generating {duration}s CTA clip...")
        result = subprocess.run(cmd_cta, check=True, capture_output=True, text=True)
        
        # Verify CTA tmp file has audio
        if tmp_cta.exists():
            probe_audio = subprocess.run(
                [get_ffprobe(), "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(tmp_cta)],
                capture_output=True, text=True
            )
            if probe_audio.stdout.strip():
                log("DEBUG", f"CTA tmp file has audio: YES")
            else:
                log("ERROR", f"CTA tmp file has NO audio! cmd_cta: {' '.join(cmd_cta)}")
        else:
            log("ERROR", f"CTA tmp file was not created!")

        # ── Step 3: Concatenate main clip + CTA clip ─────────────────────────
        # Use concat filter with proper audio handling for different sample rates
        cmd_concat: list[str] = [ffmpeg, "-y", "-hide_banner"]

        # [0] main clip
        cmd_concat.extend(["-i", clip_path])
        # [1] CTA clip
        cmd_concat.extend(["-i", str(tmp_cta)])

        # Build concat filter - ensure audio matches video duration
        if has_audio:
            # Fade original audio before any silence padding so the fade always
            # affects real content, even when audio stream is shorter than video.
            main_sample_rate = info.get("sample_rate", 44100)
            fade_anchor = main_dur
            if audio_dur > 0:
                fade_anchor = min(main_dur, audio_dur)
            audio_fade_duration = min(1.5, max(0.25, fade_duration * 3), fade_anchor)
            fade_start = max(0.0, fade_anchor - audio_fade_duration)
            concat_filter = (
                f"[0:v][1:v]concat=n=2:v=1[outv];"
                f"[0:a]afade=t=out:st={fade_start:.4f}:d={audio_fade_duration:.4f}[main_faded];"
                f"[main_faded]apad=whole_dur={main_dur:.4f}[main_padded];"
                f"[main_padded]aresample=async=1:osr={main_sample_rate}[main_resampled];"
                f"[1:a]aresample=async=1:osr={main_sample_rate}[cta_resampled];"
                f"[main_resampled][cta_resampled]concat=n=2:a=1:v=0[outa]"
            )
        else:
            # Main clip has no audio — generate silence then concat with CTA audio
            concat_filter = (
                f"[0:v][1:v]concat=n=2:v=1[outv];"
                f"anullsrc=r=44100:cl=stereo[sil];"
                f"[sil]atrim=0:{main_dur:.4f}[sil_t];"
                f"[sil_t][1:a]concat=n=2:v=0:a=1[outa]"
            )

        cmd_concat.extend(["-filter_complex", concat_filter])
        cmd_concat.extend(["-map", "[outv]", "-map", "[outa]"])

        # Re-encode with fast settings
        cmd_concat.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"])
        cmd_concat.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd_concat.extend(["-movflags", "+faststart", "-loglevel", "error"])
        cmd_concat.append(str(tmp_final))

        log("DEBUG", f"CTA: concatenating main clip + CTA...")
        subprocess.run(cmd_concat, check=True, capture_output=True, text=True)

        # Atomic rename
        tmp_final.replace(out)

        # Cleanup
        for tmp_file in [tmp_cta, tmp_final]:
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except OSError:
                pass

        log("DEBUG", f"CTA appended → {out.name} (optimized concat method)")
        return str(out)

    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "")[:400]
        log("ERROR", f"CTA append failed: {detail}")
        log("DEBUG", f"CTA cmd: concatenation approach")
        # Keep temp files on failure for debugging
        log("DEBUG", f"Keeping temp files for debugging: {tmp_cta}, {tmp_final}")
        return clip_path   # leave original untouched on failure
