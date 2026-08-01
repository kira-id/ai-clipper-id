"""
Subtitle generation (ASS format) — clean, non-overlapping, lower-third.

Generates word-by-word highlighted subtitles. Each word appears individually
with yellow highlight on the currently spoken word. Professional design
with modern font styling optimized for TikTok/Reels/Shorts.
"""

from typing import Any


# ── ASS color helpers (format: &HAABBGGRR) ──────────────────────────────────

def _rgb_to_ass(r: int, g: int, b: int, a: int = 0) -> str:
    """Convert RGB(A) to ASS color string &HAABBGGRR."""
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


COLOR_HIGHLIGHT = _rgb_to_ass(255, 225, 53)      # Yellow (current word)
COLOR_NORMAL    = _rgb_to_ass(0, 0, 0)            # Black (other words)
COLOR_OUTLINE   = _rgb_to_ass(255, 255, 255)      # White outline (halo → readable on dark bg)
COLOR_GLOW      = _rgb_to_ass(255, 255, 255, 90)  # White glow (soft halo)


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_words(
    words: list[dict[str, Any]],
    max_words: int = 4,
    max_duration: float = 2.5,
    max_gap: float = 0.5,
) -> list[list[dict[str, Any]]]:
    """Group words into subtitle chunks — single line, max 4 words.

    Splits when:
    - max_words reached
    - cumulative duration exceeds max_duration
    - silence gap between consecutive words exceeds max_gap
    """
    if not words:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in words:
        if current:
            prev_end = current[-1]["end"]
            curr_start = word["start"]
            group_dur = word["end"] - current[0]["start"]
            gap = curr_start - prev_end

            if (len(current) >= max_words
                    or group_dur > max_duration
                    or gap > max_gap):
                groups.append(current)
                current = []

        current.append(word)

    if current:
        groups.append(current)

    return groups


def _adapt_for_aspect_ratio(
    play_res_x: int,
    play_res_y: int,
    font_size_pct: float,
) -> float:
    """Scale font percentage: bump it for portrait (narrow) frames so the
    text stays large and legible, and slightly enlarge for landscape/square.
    """
    aspect = play_res_x / max(1, play_res_y)
    if aspect < 1.0:
        # Portrait: narrow width → grow font so it fills the frame
        return font_size_pct * (1.0 + (1.0 - aspect) * 0.8)
    return font_size_pct * (1.0 + (aspect - 1.0) * 1.3)


def _resolve_font_size(play_res_y: int, font_size_pct: float) -> int:
    """Calculate ASS font size as a percentage of vertical resolution."""
    return max(36, round(play_res_y * font_size_pct / 100.0))


def _resolve_outline(play_res_y: int, font_size_pct: float) -> int:
    """Scale outline thickness relative to resolution and font size."""
    return max(2, round(play_res_y * font_size_pct * 0.07 / 100.0))


def generate_ass_subtitles(
    words: list[dict[str, Any]],
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Montserrat",
    font_size_pct: float = 3.6,
    highlight_color: str | None = None,
    normal_color: str | None = None,
    position: str = "lower",
    max_words_per_group: int = 4,
    subtitle_margin_pct: float | None = None,
) -> str:
    """Generate ASS subtitles with word-by-word yellow highlight.

    Design choices for professional shorts:
    - **Single line** max (no two-line subtitles) — WrapStyle:2 keeps long
      groups on one line with ellipsis rather than overflowing the frame
    - **Word-by-word highlight**: each word turns yellow when spoken,
      NOT karaoke sweep — discrete per-word color change
    - **Bigger** font (3.6% portrait baseline) for high legibility
    - **Lower position** (25% from bottom by default) to avoid face occlusion
    - **Modern font**: Montserrat (falls back to Arial if unavailable)
    - **No temporal overlap** between subtitle groups
    - **Black text** with white halo so it reads on bright footage
    - **Portrait overflow guard**: extra word cap + larger horizontal margins
      so text never runs off the narrow frame edges
    """

    hi_color = highlight_color or COLOR_HIGHLIGHT
    nm_color = normal_color or COLOR_NORMAL

    effective_pct = _adapt_for_aspect_ratio(play_res_x, play_res_y, font_size_pct)
    font_size = _resolve_font_size(play_res_y, effective_pct)
    outline_w = _resolve_outline(play_res_y, effective_pct)
    shadow_depth = max(2, outline_w)  # Larger soft white base for the glow
    GLOW_BLUR = 3  # Edge blur → white halo / glow around the text

    # ── Alignment & margins ──────────────────────────────────────────────
    default_lower_margin = subtitle_margin_pct if subtitle_margin_pct is not None else 25.0
    margin_pct = {
        "lower":  default_lower_margin,  # Adjustable, default 25% from bottom
        "center": 0.0,
        "upper":  5.0,
    }
    alignment_map = {"lower": 2, "center": 5, "upper": 8}
    alignment = alignment_map.get(position, 2)
    margin_v = round(play_res_y * margin_pct.get(position, 5.0) / 100.0)
    # Wider horizontal margin on portrait frames so text never runs off
    # the narrow edges. 10% each side on portrait, 8% otherwise.
    margin_h_pct = 10.0 if (play_res_x / max(1, play_res_y)) < 1.0 else 8.0
    margin_h = round(play_res_x * margin_h_pct / 100.0)

    # ── ASS header with two styles ───────────────────────────────────────
    # Style "Word": normal white word
    # Style "WordHi": highlighted yellow word (currently spoken)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Normal word style (white)
        f"Style: Word,{font_name},{font_size},"
        f"{nm_color},{nm_color},{COLOR_OUTLINE},{COLOR_GLOW},"
        f"-1,0,0,0,100,100,1.0,0,1,{outline_w},{shadow_depth},"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        # Highlighted word style (yellow) — same positioning
        f"Style: WordHi,{font_name},{font_size},"
        f"{hi_color},{hi_color},{COLOR_OUTLINE},{COLOR_GLOW},"
        f"-1,0,0,0,100,100,1.0,0,1,{outline_w},{shadow_depth},"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # ── Group words into single-line chunks ──────────────────────────────
    # Tighter cap on portrait frames: the bigger font means fewer words fit
    # on the narrow width before the line would overflow the edges.
    is_portrait = (play_res_x / max(1, play_res_y)) < 1.0
    cap = 3 if is_portrait else max_words_per_group
    groups = _group_words(words, max_words=cap)
    if not groups:
        return header

    # ── Resolve non-overlapping time ranges ──────────────────────────────
    MIN_GAP = 0.04  # 40 ms

    time_ranges: list[tuple[float, float]] = []
    for g in groups:
        time_ranges.append((g[0]["start"], g[-1]["end"]))

    clamped: list[tuple[float, float]] = []
    for i, (gs, ge) in enumerate(time_ranges):
        if i + 1 < len(time_ranges):
            next_start = time_ranges[i + 1][0]
            ge = min(ge, next_start - MIN_GAP)
        if ge <= gs:
            ge = gs + 0.1
        clamped.append((gs, ge))

    # ── Build dialogue lines with word-by-word highlight ─────────────────
    # For each group, we emit one dialogue line per word-timing where
    # the currently spoken word is yellow and all others are white.
    # This creates a discrete word-by-word highlight (not karaoke sweep).
    dialogue_lines: list[str] = []

    for group, (g_start, g_end) in zip(groups, clamped):
        clean_words = []
        for w in group:
            clean = w["word"].strip()
            if clean:
                clean_words.append((clean, w["start"], w["end"]))

        if not clean_words:
            continue

        # For each word in the group, create a dialogue event where
        # that word is highlighted and others are normal
        for word_idx, (word_text, w_start, w_end) in enumerate(clean_words):
            # This word's display period
            if word_idx == 0:
                event_start = g_start
            else:
                event_start = clean_words[word_idx][1]

            if word_idx < len(clean_words) - 1:
                event_end = clean_words[word_idx + 1][1]
            else:
                event_end = g_end

            # Clamp
            event_start = max(g_start, event_start)
            event_end = min(g_end, event_end)
            if event_end <= event_start:
                event_end = event_start + 0.05

            start_str = _seconds_to_ass_time(event_start)
            end_str = _seconds_to_ass_time(event_end)

            # Build text with inline override: highlight current word
            parts = []
            for j, (wt, _, _) in enumerate(clean_words):
                if j == word_idx:
                    # Yellow highlight for current word
                    parts.append(f"{{\\c{hi_color}}}{wt}{{\\c{nm_color}}}")
                else:
                    parts.append(wt)

            text = " ".join(parts)
            line = f"Dialogue: 0,{start_str},{end_str},Word,,0,0,0,,{{\\blur{GLOW_BLUR}}}{text}"
            dialogue_lines.append(line)

    return header + "\n".join(dialogue_lines) + "\n"


def generate_title_overlay(
    title: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    duration: float = 3.0,
    font_name: str = "Montserrat",
) -> str:
    """Generate ASS title overlay with professional design.

    Modern, clean title with:
    - Montserrat font (professional, widely available)
    - Semi-transparent dark background pill behind text
    - Fade-out animation
    - Positioned at upper-third (25% from top)
    """
    aspect = play_res_x / play_res_y
    if aspect < 1.0:
        # Portrait: make the title big and bold
        font_size = int(play_res_x * 0.085)
    else:
        font_size = int(play_res_y * 0.10)

    outline_w = max(4, int(font_size * 0.15))  # Thicker outline for rounded effect
    shadow_d = max(3, int(font_size * 0.12))   # Larger shadow for soft rounded corners

    # Colors: black text on a yellow background pill
    c_text = _rgb_to_ass(0, 0, 0)
    c_outline = _rgb_to_ass(0, 0, 0)
    c_shadow = _rgb_to_ass(0, 0, 0, 100)

    # Position: 25% from top
    cx = play_res_x // 2
    cy = int(play_res_y * 0.25)

    # Fade timing
    if duration < 1.5:
        fade_out_dur = max(100, int(duration * 200))
    else:
        fade_out_dur = int(0.5 * 1000)
    fade_out_start = max(0, int((duration - (fade_out_dur / 1000)) * 1000))

    margin_h = round(play_res_x * 5.0 / 100.0)

    # Background box style (BorderStyle=3 creates a box behind text)
    bg_color = _rgb_to_ass(255, 225, 53, 235)  # opaque-ish yellow

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # BorderStyle=3 = opaque box background
        f"Style: Title,{font_name},{font_size},"
        f"{c_text},{c_text},{c_outline},{bg_color},"
        f"-1,0,0,0,100,100,0.5,0,3,{outline_w},{shadow_d},"
        f"5,{margin_h},{margin_h},0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )

    t0 = _seconds_to_ass_time(0.0)
    t1 = _seconds_to_ass_time(duration)

    anim = (
        f"{{\\pos({cx},{cy})\\an5"
        f"\\alpha&H00"
        f"\\t({fade_out_start},{fade_out_start + fade_out_dur},\\alpha&HFF)"
        f"}}"
    )

    event = f"Dialogue: 0,{t0},{t1},Title,,0,0,0,,{anim}{title}"

    return header + event + "\n"


def get_clip_words(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:
    """Extract word-level timestamps for a clip's time range.

    Returns words with timestamps adjusted to be relative to clip start
    (0-based) and sorted chronologically.
    """
    words: list[dict[str, Any]] = []

    for seg in segments:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue

        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end   = w.get("end", 0)
            w_text  = w.get("word", "").strip()

            if not w_text:
                continue
            # Only include words whose midpoints fall inside the clip
            w_mid = (w_start + w_end) / 2.0
            if w_mid < clip_start or w_mid > clip_end:
                continue

            words.append({
                "word": w_text,
                "start": max(0.0, w_start - clip_start),
                "end":   max(0.0, w_end   - clip_start),
            })

    words.sort(key=lambda w: w["start"])
    return words
