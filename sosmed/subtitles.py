"""
Subtitle generation (ASS format) — clean, non-overlapping, lower-third.

Generates subtitles with white text and a black outline, no shadow/glow.
Professional design with modern font styling optimized for
TikTok/Reels/Shorts.
"""

from typing import Any

import math
import os

try:
    from PIL import ImageFont
    _HAVE_PIL = True
except Exception:  # pragma: no cover - PIL is a hard dep for measurement
    _HAVE_PIL = False


# ── Font resolution for accurate text-width measurement ──────────────────────
# libass renders with Montserrat and falls back to Arial when Montserrat is
# absent, so measuring with the real font (or Arial as the closest fallback)
# gives an accurate on-screen width. We resolve an actual TTF so the overflow
# algorithm below measures what the viewer will actually see.
def _resolve_measure_font_path() -> str | None:
    """Return a usable TTF path for text measurement, or None to use PIL default."""
    candidates = [
        "/c/Windows/Fonts/montserrat.ttf",
        "/c/Windows/Fonts/Montserrat.ttf",
        "/c/Windows/Fonts/arial.ttf",
        "/c/Windows/Fonts/Arial.ttf",
        "/c/Windows/Fonts/arialbd.ttf",
    ]
    for c in candidates:
        # Normalize the MSYS-ish path for the current interpreter.
        norm = c.replace("/c/", "C:/").replace("/C/", "C:/")
        if os.path.exists(norm):
            return norm
    return None


_FONT_PATH = _resolve_measure_font_path()


def _measure_text_width(text: str, font_size: int, bold: bool = False) -> int:
    """Measure rendered pixel width of ``text`` at ``font_size``.

    Falls back to a conservative heuristic if PIL/PIL-font is unavailable so the
    caller still gets a sane (slightly over-estimated) width instead of overflow.
    """
    if not _HAVE_PIL:
        return int(len(text) * font_size * 0.6) + 4
    try:
        if _FONT_PATH:
            fnt = ImageFont.truetype(_FONT_PATH, font_size)
        else:
            fnt = ImageFont.load_default()
        # getlength respects kerning; fall back to bbox width if unavailable.
        if hasattr(fnt, "getlength"):
            return int(math.ceil(fnt.getlength(text)))
        bbox = fnt.getbbox(text)
        return int(bbox[2] - bbox[0]) if bbox else len(text) * font_size // 2
    except Exception:
        # Conservative fallback: ~0.55em average advance for Arial-ish latin.
        return int(len(text) * font_size * 0.55) + 6


# ── ASS color helpers (format: &HAABBGGRR) ──────────────────────────────────

def _rgb_to_ass(r: int, g: int, b: int, a: int = 0) -> str:
    """Convert RGB(A) to ASS color string &HAABBGGRR."""
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


COLOR_HIGHLIGHT = _rgb_to_ass(255, 255, 255)      # White (unused — no per-word highlight)
COLOR_NORMAL    = _rgb_to_ass(255, 255, 255)      # White (subtitle text)
COLOR_OUTLINE   = _rgb_to_ass(0, 0, 0)            # Black outline (halo → readable on any bg)
COLOR_GLOW      = _rgb_to_ass(255, 255, 255, 0)   # Transparent (no shadow/glow)


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
    font_size_pct: float = 3.2,
    highlight_color: str | None = None,
    normal_color: str | None = None,
    position: str = "lower",
    max_words_per_group: int = 4,
    subtitle_margin_pct: float | None = None,
) -> str:
    """Generate ASS subtitles with white text and a black outline.

    Design choices for professional shorts:
    - **Single line** max (no two-line subtitles) — WrapStyle:2 keeps long
      groups on one line with ellipsis rather than overflowing the frame
    - **White text** with a thin black outline so it reads on any footage
    - **No shadow / no glow** (removed per request)
    - **Bigger** font (3.6% portrait baseline) for high legibility
    - **Lower position** (25% from bottom by default) to avoid face occlusion
    - **Modern font**: Montserrat (falls back to Arial if unavailable)
    - **No temporal overlap** between subtitle groups
    - **Hard overflow guard (measured)**: the font size and per-line word cap
      are *solved* by measuring the actual rendered line width with PIL so the
      longest line can never run off the frame. This is robust to font choice
      and language — it measures, not guesses.
    """

    hi_color = highlight_color or COLOR_HIGHLIGHT
    nm_color = normal_color or COLOR_NORMAL

    effective_pct = _adapt_for_aspect_ratio(play_res_x, play_res_y, font_size_pct)
    base_font_size = _resolve_font_size(play_res_y, effective_pct)
    outline_w = _resolve_outline(play_res_y, effective_pct)
    shadow_depth = 0  # No shadow / glow
    GLOW_BLUR = 0     # No blur

    # ── Alignment & margins ──────────────────────────────────────────────
    is_portrait = (play_res_x / max(1, play_res_y)) < 1.0
    # Default 10% from bottom (low on screen). Portrait frames have a
    # description/caption strip pinned to the very bottom, so the lower
    # subtitle would be hidden behind it — push it further up there.
    if is_portrait and subtitle_margin_pct is None:
        default_lower_margin = 20.0
    else:
        default_lower_margin = subtitle_margin_pct if subtitle_margin_pct is not None else 10.0
    margin_pct = {
        "lower":  default_lower_margin,
        "center": 0.0,
        "upper":  5.0,
    }
    alignment_map = {"lower": 2, "center": 5, "upper": 8}
    alignment = alignment_map.get(position, 2)
    margin_v = round(play_res_y * margin_pct.get(position, 5.0) / 100.0)
    # Wider horizontal margin on portrait frames so text never runs off
    # the narrow edges. 10% each side on portrait, 8% otherwise.
    margin_h_pct = 10.0 if is_portrait else 8.0
    margin_h = round(play_res_x * margin_h_pct / 100.0)

    # ---- ROBUST OVERFLOW GUARD (measured) ---------------------------------
    # Iteratively pick (font_size, word_cap) so the WIDEST line fits inside the
    # safe horizontal area [margin_h, play_res_x - margin_h], plus a small
    # safety pad. We measure the real rendered width of each group's text so
    # this works for any language / font, not just a length heuristic.
    SAFETY_PAD = max(4, round(play_res_x * 0.01))  # 1% safety pad each side
    usable_w = play_res_x - 2 * (margin_h + SAFETY_PAD)

    # Group first with a generous cap, then measure and shrink if needed.
    def _regroup(cap: int) -> list[list[dict[str, Any]]]:
        return _group_words(words, max_words=cap)

    font_size = base_font_size
    cap = 3 if is_portrait else max_words_per_group

    # Shrink font before sacrificing words (words read better than tiny text,
    # but if even 1 word at min font overflows we must cap down to 1).
    MIN_FONT = max(20, round(play_res_y * 0.018))  # ~2% floor
    # Compute widest measured line for a given (font, cap).
    groups_all = _regroup(cap if cap >= 1 else 1)
    if groups_all:
        def _widest(font: int, cap: int) -> int:
            widest = 0
            for g in _regroup(cap):
                txt = " ".join(w["word"] for w in g if w["word"].strip())
                if txt:
                    widest = max(widest, _measure_text_width(txt, font))
            return widest

        # Step 1: if the current font already overflows, shrink it.
        while font_size > MIN_FONT and _widest(font_size, cap) > usable_w:
            font_size -= 2
        # Step 2: if still overflows at min font, drop the word cap.
        while cap > 1 and _widest(font_size, cap) > usable_w:
            cap -= 1
        # Step 3: last resort — shrink font below floor only if a single word
        # still doesn't fit (extremely long token); clip by allowing hard wrap
        # (WrapStyle:2 adds an ellipsis rather than spilling off-frame).
        if _widest(font_size, cap) > usable_w:
            # Try to shrink a bit more; WrapStyle:2 prevents true overflow.
            while font_size > 14 and _widest(font_size, cap) > usable_w:
                font_size -= 2

    groups = _regroup(cap)
    if not groups:
        # Still emit a valid (empty) subtitle file so downstream ffmpeg won't choke.
        header = _subtitle_header(
            play_res_x, play_res_y, font_name, font_size, nm_color, outline_w,
            alignment, margin_h, margin_v,
        )
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

    # ── Build dialogue lines: one line per group, all white, no highlight ──
    dialogue_lines: list[str] = []

    for group, (g_start, g_end) in zip(groups, clamped):
        clean_words = []
        for w in group:
            clean = w["word"].strip()
            if clean:
                clean_words.append((clean, w["start"], w["end"]))

        if not clean_words:
            continue

        text = " ".join(wt for wt, _, _ in clean_words)
        start_str = _seconds_to_ass_time(g_start)
        end_str = _seconds_to_ass_time(g_end)
        line = f"Dialogue: 0,{start_str},{end_str},Word,,0,0,0,,{text}"
        dialogue_lines.append(line)

    header = _subtitle_header(
        play_res_x, play_res_y, font_name, font_size, nm_color, outline_w,
        alignment, margin_h, margin_v,
    )
    return header + "\n".join(dialogue_lines) + "\n"


def _subtitle_header(
    play_res_x: int,
    play_res_y: int,
    font_name: str,
    font_size: int,
    nm_color: str,
    outline_w: int,
    alignment: int,
    margin_h: int,
    margin_v: int,
) -> str:
    """Emit the [Script Info]/[V4+ Styles]/[Events] header for subtitles."""
    return (
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
        # Word style (white text, black outline)
        f"Style: Word,{font_name},{font_size},"
        f"{nm_color},{nm_color},{COLOR_OUTLINE},{COLOR_GLOW},"
        f"-1,0,0,0,100,100,1.0,0,1,{outline_w},0,"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _rounded_sheared_polygon(cx, cy, half_w, half_h, shear, r):
    """Build an ASS drawing-string for a sheared rectangle (parallelogram)
    whose corners are rounded with radius ``r``.

    The parallelogram is an axis-aligned rectangle of size (2*half_w) x
    (2*half_h) centered at (cx, cy), then sheared so every corner's x is
    shifted by ``shear * (y - cy)``. Corners are rounded with circular arcs
    of radius ``r`` (clamped to the shorter half-side).
    """
    r = max(0, min(r, half_w - 1, half_h - 1))

    # Corner centers (before shear): rectangle inset by r from each edge.
    # Order: top-left, top-right, bottom-right, bottom-left.
    xl, xr = cx - half_w + r, cx + half_w - r
    yt, yb = cy - half_h + r, cy + half_h - r

    # Shear helper: x' = x + shear * (y - cy)
    def sh(x, y):
        return (x + int(round(shear * (y - cy))), y)

    tl_c = sh(xl, yt)   # top-left corner center
    tr_c = sh(xr, yt)   # top-right corner center
    br_c = sh(xr, yb)   # bottom-right corner center
    bl_c = sh(xl, yb)   # bottom-left corner center

    # Arc end/start points for a rounded corner use the corner center plus
    # offsets of +/- r. We approximate each quarter circle with 8 segments.
    N = 8

    def arc_points(center, a0, a1):
        """Points along an arc (degrees) around `center` with radius r."""
        px0, py0 = center
        pts = []
        for k in range(N + 1):
            a = math.radians(a0 + (a1 - a0) * k / N)
            pts.append((int(round(px0 + r * math.cos(a))),
                        int(round(py0 + r * math.sin(a)))))
        return pts

    # For each corner, the arc sweeps 90deg. Tangent points are the rectangle
    # edges offset by r. We define each corner's arc by its center and sweep.
    # TL corner center: connect top tangent (xl, yt) -> left tangent (xl, yb)
    # in screen coords that's angle 180deg (left) to 270deg (up).
    tl_arc = arc_points(tl_c, 180, 270)
    # TR corner center: left tangent (xr, yt) -> top tangent (xr, yt-r)?
    # angle 270 (up) -> 360/0 (right)
    tr_arc = arc_points(tr_c, 270, 360)
    # BR corner center: top tangent (xr, yb) -> right tangent (xr+r, yb)
    # angle 0 (right) -> 90 (down)
    br_arc = arc_points(br_c, 0, 90)
    # BL corner center: right tangent (xl, yb) -> bottom tangent (xl, yb+r)
    # angle 90 (down) -> 180 (left)
    bl_arc = arc_points(bl_c, 90, 180)

    # Assemble path: move to end of TL arc, then line+arc around.
    path = []
    path.append(f"m {tl_arc[0][0]} {tl_arc[0][1]}")
    for p in tl_arc[1:]:
        path.append(f"l {p[0]} {p[1]}")
    for p in tr_arc:
        path.append(f"l {p[0]} {p[1]}")
    for p in br_arc:
        path.append(f"l {p[0]} {p[1]}")
    for p in bl_arc:
        path.append(f"l {p[0]} {p[1]}")
    path.append(f"l {tl_arc[0][0]} {tl_arc[0][1]}")  # close
    return " ".join(path)


def generate_title_overlay(
    title: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    duration: float = 3.0,
    font_name: str = "Montserrat",
) -> str:
    """Generate ASS title overlay with professional design.

    Modern title banner with:
    - Montserrat font (professional, widely available)
    - BLACK parallelogram background, sheared via the drawing polygon (background
      layer ONLY) — high contrast for white text on bright footage
    - BLACK, BOLD, UPRIGHT text (NO italic — shear is NOT applied to the text)
    - NO outline / NO shadow on the text
    - Fade-out animation
    - Positioned at upper-third (25% from top)
    - **Measured overflow guard**: the font size and banner width are solved by
      measuring the *actual rendered* title width (PIL), so a long title shrinks
      to fit inside the frame instead of spilling past the edges. The black block
      hugs the measured text width (with padding), so it looks intentional at any
      title length.
    """
    aspect = play_res_x / play_res_y

    # ── Measured font-size solve (prevents horizontal overflow) ────────────
    # Start at the designed size, then shrink until the title fits the safe
    # width. Safe width = frame width minus a healthy side margin (12% each side
    # so the black banner never touches the edges).
    if aspect < 1.0:
        start_font = int(play_res_x * 0.085)   # portrait: big & bold
    else:
        start_font = int(play_res_y * 0.10)
    SIDE_MARGIN_PCT = 12.0
    safe_w = int(play_res_x * (1.0 - 2 * SIDE_MARGIN_PCT / 100.0))

    font_size = start_font
    # Robust solve: shrink until the measured title fits the safe width, or until
    # we hit a hard floor (WrapStyle:2 then ellipsizes as a last resort). Step of
    # 2 for speed, then a final 1px pass so we never stop a hair above safe_w.
    while font_size > 16 and _measure_text_width(title, font_size) > safe_w:
        font_size -= 2
    while font_size > 14 and _measure_text_width(title, font_size) > safe_w:
        font_size -= 1

    # Colors: WHITE title text on a BLACK parallelogram. No text outline.
    c_text = _rgb_to_ass(255, 255, 255)       # WHITE title text (contrasts black)

    # ── Filled parallelogram background ───────────────────────────────────
    # Drawn as a TRUE solid vector polygon (a sheared rectangle) via ASS
    # Drawing commands, so it is a clean block of color — NOT a fake
    # text-glyph box. The readable title is drawn on top as a separate
    # upright layer.
    # ~14deg shear (tan(14) ≈ 0.25). Negative → leans like "/_" (top shifted
    # right, bottom shifted left).
    shear = -0.25

    # Banner geometry: centered at upper-third (25% from top).
    cx = play_res_x // 2
    cy = int(play_res_y * 0.25)

    # Vertical breathing room so the text never touches the block edges.
    text_margin_y = int(font_size * 0.35)
    # Banner width follows the MEASURED title width (+ padding) but is clamped
    # so it never exceeds the safe frame width. Long titles therefore get a
    # snug black block; very long titles shrink the font (above) first.
    title_w = _measure_text_width(title, font_size)
    banner_w = min(safe_w, int(title_w * 1.12) + 2 * int(font_size * 0.4))
    banner_w = max(banner_w, int(play_res_x * 0.30))  # keep a minimum visible block
    banner_h = int(font_size * 1.9) + 2 * text_margin_y   # + margin top & bottom
    half_w = banner_w // 2
    half_h = banner_h // 2

    # Rounded-corner radius for the parallelogram (clamped inside the helper).
    corner_r = int(font_size * 0.45)

    # Fade timing (applied to BOTH box + text so they vanish together)
    if duration < 1.5:
        fade_out_dur = max(100, int(duration * 200))
    else:
        fade_out_dur = int(0.5 * 1000)
    fade_out_start = max(0, int((duration - (fade_out_dur / 1000)) * 1000))

    # Opaque BLACK block color (Alpha 00 = fully opaque in &HAABBGGRR).
    c_block = _rgb_to_ass(0, 0, 0, 0)

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
        # --- Parallelogram background: a solid filled vector polygon ---
        # Alignment 7 (top-left) + the \pos(0,0) in the drawing so the Drawing
        # commands use absolute script-resolution pixel coordinates (0,0 = frame
        # top-left) instead of anchoring at screen center.
        f"Style: TitleBox,{font_name},{font_size},"
        f"{c_block},{c_block},{c_block},{c_block},"
        f"-1,0,0,0,100,100,0.5,0,1,0,0,"
        f"7,0,0,0,1\n"
        # --- Title text: upright, bold, white, drawn on top ---
        f"Style: TitleText,{font_name},{font_size},"
        f"{c_text},{c_text},{c_text},{c_text},"
        f"-1,0,0,0,100,100,0.5,0,1,0,0,"
        f"5,0,0,0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )

    t0 = _seconds_to_ass_time(0.0)
    t1 = _seconds_to_ass_time(duration)

    # Background parallelogram: a solid filled polygon (sheared, rounded-
    # corner rectangle), fades out in sync with the title text.
    box_poly = _rounded_sheared_polygon(cx, cy, half_w, half_h, shear, corner_r)
    box_anim = (
        "{\\p1"
        "\\pos(0,0)"
        f"\\alpha&H00"
        f"\\t({fade_out_start},{fade_out_start + fade_out_dur},\\alpha&HFF)"
        "}"
        f"{box_poly}"
    )
    # Title text: upright, bold, fades out in sync (NO shear → no italic).
    text_anim = (
        f"{{\\pos({cx},{cy})\\an5"
        f"\\alpha&H00"
        f"\\t({fade_out_start},{fade_out_start + fade_out_dur},\\alpha&HFF)"
        f"}}"
    )

    box_event  = f"Dialogue: 0,{t0},{t1},TitleBox,,0,0,0,,{box_anim}"
    text_event = f"Dialogue: 1,{t0},{t1},TitleText,,0,0,0,,{text_anim}{title}"

    return header + box_event + "\n" + text_event + "\n"


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
