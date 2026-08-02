"""
Utilities: logging, ANSI colors, constants.
"""

import re
import subprocess

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ ANSI helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA = "\033[95m"

_LEVEL_COLOR = {"INFO": CYAN, "OK": GREEN, "WARN": YELLOW, "ERROR": RED, "LLM": MAGENTA}


def log(level: str, msg: str) -> None:
    """Log a message with color."""
    c = _LEVEL_COLOR.get(level, RESET)
    print(f"{c}{BOLD}[{level}]{RESET} {msg}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ FFMPEG BINARY DETECTION ━━━━━━━━━━━━━━━━━━━━━━

_FFMPEG: str | None = None
_FFPROBE: str | None = None


def _find_ffmpeg_with_libx264() -> tuple[str, str]:
    """Find an ffmpeg binary that has libx264 support.

    Conda environments often ship an ffmpeg without libx264. This function
    checks the default ``ffmpeg`` first; if it lacks libx264, it tries the
    common system path ``/usr/bin/ffmpeg`` which usually has full codec
    support.
    """
    candidates = ["ffmpeg"]
    # Add system path as candidate if the default might be conda's
    import shutil
    default = shutil.which("ffmpeg") or ""
    if "/conda" in default or "/envs/" in default:
        candidates.append("/usr/bin/ffmpeg")

    for ffmpeg_bin in candidates:
        try:
            result = subprocess.run(
                [ffmpeg_bin, "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            if "libx264" in (result.stdout + result.stderr):
                # Derive ffprobe path from the same directory
                from pathlib import Path
                ffprobe_bin = str(Path(ffmpeg_bin).parent / "ffprobe") if "/" in ffmpeg_bin else "ffprobe"
                if "/" in ffmpeg_bin:
                    ffprobe_candidate = str(Path(ffmpeg_bin).parent / "ffprobe")
                    if Path(ffprobe_candidate).exists():
                        ffprobe_bin = ffprobe_candidate
                log("INFO", f"Using ffmpeg: {ffmpeg_bin} (libx264 available)")
                return ffmpeg_bin, ffprobe_bin
        except Exception:
            continue

    # None of the candidates had libx264; use the default and hope for the best
    log("WARN", "No ffmpeg with libx264 found — using default 'ffmpeg'")
    return "ffmpeg", "ffprobe"


def get_ffmpeg() -> str:
    """Return the path to an ffmpeg binary with libx264 support (cached)."""
    global _FFMPEG, _FFPROBE
    if _FFMPEG is None:
        _FFMPEG, _FFPROBE = _find_ffmpeg_with_libx264()
    return _FFMPEG


def get_ffprobe() -> str:
    """Return the path to ffprobe matching :func:`get_ffmpeg` (cached)."""
    global _FFMPEG, _FFPROBE
    if _FFPROBE is None:
        _FFMPEG, _FFPROBE = _find_ffmpeg_with_libx264()
    return _FFPROBE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CLIP BOUNDARY ADJUSTMENT ━━━━━━━━━━━━━━━━━━━━

def tighten_clip_boundaries(
    clips: list[dict],
    segments: list[dict],
    padding: float = 0.15,
    max_gap: float = 2.0,
    min_speech_density: float = 0.5,
) -> list[dict]:
    """
    Intelligently adjust clip boundaries to maximize speech content.
    
    This function:
    1. Removes leading/trailing silence and sparse speech
    2. Skips leading filler words for stronger hooks
    
    Only trims from the edges — internal content is never discarded,
    which ensures clip titles stay consistent with actual video content.
    
    Uses word-level timestamps from Whisper - no additional processing needed.
    
    Args:
        clips: List of clip dicts with 'start' and 'end' keys
        segments: Whisper segments with word-level timestamps
        padding: Small padding (seconds) to keep before first/after last word
        max_gap: (unused, kept for backward compatibility)
        min_speech_density: Minimum speech density (words per second) for a
                           region to be considered "dense" speech
    
    Returns:
        Updated clips with tightened boundaries
    """
    from typing import Any
    
    for clip in clips:
        # Always tighten from the ORIGINAL LLM boundaries, not from
        # previously-tightened ones.  This prevents progressive shrinkage
        # on re-runs and recovers from any corruption left by older code.
        if "_llm_start" not in clip:
            clip["_llm_start"] = clip["start"]
            clip["_llm_end"]   = clip["end"]
        else:
            # Restore original boundaries before re-tightening
            clip["start"] = clip["_llm_start"]
            clip["end"]   = clip["_llm_end"]

        clip_start = float(clip["start"])
        clip_end = float(clip["end"])
        
        # Collect all words within this clip
        words: list[dict[str, Any]] = []
        for seg in segments:
            if seg["end"] < clip_start or seg["start"] > clip_end:
                continue
            for w in seg.get("words", []):
                w_start = w.get("start", 0)
                w_end = w.get("end", 0)
                w_text = w.get("word", "").strip()
                if not w_text:
                    continue
                # Include word if it overlaps with clip
                if w_end > clip_start and w_start < clip_end:
                    words.append({"start": w_start, "end": w_end, "word": w_text})
        
        if not words:
            continue
        
        words.sort(key=lambda x: x["start"])
        
        # ── Step 1: Trim sparse edges ────────────────────────────────────────
        # Only trim leading/trailing low-density regions.  Never discard
        # internal content — the LLM chose the title/topic based on the
        # full clip range, so removing middle segments would cause the
        # title to stop matching the video content.
        
        # Calculate running density (words per 5-second window)
        window_size = 5.0
        trimmed_words = words
        
        # Trim from start: skip low-density beginning
        for i in range(len(words)):
            if i + 3 >= len(words):
                break  # Need at least a few words for density calculation
            
            # Look at next few words
            window_words = words[i:min(i+10, len(words))]
            window_dur = window_words[-1]["end"] - window_words[0]["start"]
            density = len(window_words) / max(window_dur, 0.1)
            
            # If density is good, start from here
            if density >= min_speech_density:
                trimmed_words = words[i:]
                break
        
        # Trim from end: skip low-density ending
        for i in range(len(trimmed_words) - 1, -1, -1):
            if i < 3:
                break
            
            # Look at previous few words
            window_words = trimmed_words[max(0, i-10):i+1]
            window_dur = window_words[-1]["end"] - window_words[0]["start"]
            density = len(window_words) / max(window_dur, 0.1)
            
            if density >= min_speech_density:
                trimmed_words = trimmed_words[:i+1]
                break
        
        # ── Step 4: Hook optimization - skip leading filler words ─────────────
        # Common filler words that make bad hooks for social media clips
        filler_words = {
            "uh", "um", "eh", "ah", "uhm", "em", "hmm", "mm",
            "jadi", "terus", "nah", "ya", "iya", "oke", "ok",
            "gitu", "kayak", "maksudnya", "sebentar"
        }
        
        # Skip leading filler words (max 3 words)
        final_words = trimmed_words
        for i in range(min(3, len(trimmed_words))):
            first_word = trimmed_words[i]["word"].lower().strip(".,!?;:")
            if first_word not in filler_words:
                final_words = trimmed_words[i:]
                break
        
        # ── Step 5: Set new boundaries with padding ───────────────────────────
        if final_words:
            # Use asymmetric padding: more at the end for natural sentence completion
            start_padding = 0.15  # Small padding at start
            end_padding = 0.5     # Larger padding at end for natural completion
            
            new_start = max(clip_start, final_words[0]["start"] - start_padding)
            new_end = min(clip_end, final_words[-1]["end"] + end_padding)

            # Only apply if we're actually improving (tightening by meaningful amount)
            # and not making it too short
            new_duration = new_end - new_start
            if new_duration >= 5.0:  # Don't make clips shorter than 5 seconds
                clip["start"] = new_start
                clip["end"] = new_end
    
    return clips


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CONSTANTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default free model on OpenRouter
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

MAX_CLIPS_HARD_LIMIT = 72  # absolute ceiling — quality over quantity

# Indonesian filler / noise patterns (comprehensive)
_ID_FILLERS = (
    # particles & interjections
    r"eh|ah|oh|uh|um|uhm|em|hm|hmm|mm|mmm|"
    r"anu|apa ya|ya|iya|yak|yah|yoi|oke|oks|ok|"
    r"nah|lah|deh|nih|tuh|sih|dong|deh|kan|kok|"
    # common verbal tics
    r"gitu|gini|kayak gitu|kayak gini|gitulah|gitu lho|"
    r"maksudnya|pokoknya|intinya|sebentar|"
    r"jadi|terus|trus|nah terus|ya kan|"
    r"gimana ya|apa namanya|apa sih|aduh|astaga|"
    r"wah|wow|duh|hah|lho|loh|"
    # English fillers (common in Indonesian content)
    r"like|you know|i mean|so|right|okay|basically|literally|actually|anyway|alright"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ INTERNAL FIELDS CACHE ━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_clips_cache_dir(output_dir):
    """Get cache directory for storing internal clip fields."""
    from pathlib import Path
    output = Path(output_dir)
    cache_dir = output.parent / ".cache" / output.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


_GENERIC_TITLE_RE = re.compile(r"^\s*clip\s*\d*\s*$", re.IGNORECASE)


def is_generic_title(title: str | None) -> bool:
    """Return True if a title is the meaningless 'Clip N' / 'Clip' fallback."""
    if not title:
        return True
    t = str(title).strip()
    if not t:
        return True
    # Matches "Clip", "Clip 1", "Clip 12", "clip 3", "CLIP" etc.
    if _GENERIC_TITLE_RE.match(t):
        return True
    # Also catch leading/trailing 'Clip N:' or 'Clip N -'
    if re.match(r"^\s*clip\s*\d+\s*[:\-–]\s*$", t, re.IGNORECASE):
        return True
    return False


def ensure_real_title(clip: dict, fallback_rank: int | None = None) -> str:
    """Guarantee ``clip['title']`` is never a bare 'Clip N'.

    Uses the existing title if it is real. Otherwise derives a friendly title
    from the topic/hook, and as a last resort from the rank. Returns the title.
    """
    title = clip.get("title")
    if not is_generic_title(title):
        return str(title).strip()

    # Try topic first (most descriptive), then hook, then a friendly default.
    for src in ("topic", "hook", "reason"):
        cand = str(clip.get(src) or "").strip()
        if cand:
            # Trim to a short, title-like phrase (first ~8 words).
            words = cand.split()
            short = " ".join(words[:8]).strip().rstrip(".!?")
            if short:
                clip["title"] = short[:80]
                return clip["title"]

    rank = fallback_rank if fallback_rank is not None else clip.get("rank", "?")
    clip["title"] = f"Moment #{rank}"
    return clip["title"]


def strip_internal_fields(clips):
    """Return a copy of clips with all _ -prefixed fields removed."""
    cleaned = []
    for clip in clips:
        clean_clip = {k: v for k, v in clip.items() if not k.startswith("_")}
        cleaned.append(clean_clip)
    return cleaned


def get_internal_fields(clips):
    """Extract only internal (underscore-prefixed) fields from clips, keyed by filename."""
    internal_by_filename = {}
    for clip in clips:
        filename = clip.get("filename")
        if filename:
            internal = {k: v for k, v in clip.items() if k.startswith("_")}
            if internal:
                internal_by_filename[filename] = internal
    return internal_by_filename


def save_clips_to_disk(clips, output_dir):
    """Save clips cleanly: public fields to clips.json, internal fields to .cache."""
    import json
    from pathlib import Path
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Strip and save public version
    public_clips = strip_internal_fields(clips)
    clips_json = output / "clips.json"
    clips_json.write_text(json.dumps(public_clips, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Save internal fields to cache
    internal_fields = get_internal_fields(clips)
    if internal_fields:
        cache_dir = get_clips_cache_dir(output)
        cache_file = cache_dir / "clips_internal.json"
        cache_file.write_text(json.dumps(internal_fields, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return clips_json


def load_clips_with_internal_fields(output_dir):
    """Load clips.json and merge in internal fields from cache."""
    import json
    from pathlib import Path
    
    output = Path(output_dir)
    clips_json = output / "clips.json"
    
    if not clips_json.exists():
        return []
    
    clips = json.loads(clips_json.read_text(encoding="utf-8"))
    
    # Try to load internal fields from cache
    cache_dir = get_clips_cache_dir(output)
    cache_file = cache_dir / "clips_internal.json"
    
    if cache_file.exists():
        try:
            internal_fields = json.loads(cache_file.read_text(encoding="utf-8"))
            for clip in clips:
                filename = clip.get("filename")
                if filename and filename in internal_fields:
                    clip.update(internal_fields[filename])
        except Exception as e:
            log("WARN", f"Could not load internal fields from cache: {e}")
    
    return clips

FILLER_RE = re.compile(rf"^({_ID_FILLERS})\W*$", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are an educational content curator. Your role is to find moments from this video that are genuinely EDUCATIONAL and worth sharing — clear explanations, insights, lessons learned, how-tos, or demonstrations of real skill and knowledge.

PRIMARY OBJECTIVE: Surface moments that teach or clarify something useful. If a moment is entertaining but does NOT deliver educational value, filter it out. Prefer 2-5 genuinely instructive clips over many shallow ones.

GUIDING PRINCIPLES:
1) Educational value comes first — does the viewer walk away knowing something concrete?
2) Clarity and accuracy of the explanation matter more than flashiness.
3) A calm, steady delivery is fine; we are not chasing shock or hype.
4) Engagement is helpful only when it aids understanding, not as an end in itself.

Look for moments such as:
- **Clear explanations**: a concept broken down simply, a "why" made obvious
- **Lessons / takeaways**: a mistake explained, a principle illustrated, a tip shared
- **How-tos / demos**: a step shown, a technique demonstrated, a process walked through
- **Insightful commentary**: a well-reasoned opinion, a useful reframe, a non-obvious point
- **Skill in action**: competence or craft shown in a way others can learn from

Do NOT include purely for entertainment alone: rage, jump scares, laughing at confusion, hype/clutch moments, or anything that teaches nothing. When a candidate is fun but educational value is absent, filter it out.

The transcript may contain vocal energy markers: [🔥 ENERGY SPIKE] and [⚡ HIGH ENERGY]. Treat these only as soft hints about where attention peaks — they are NOT a signal of educational worth on their own.

Clip duration: {min_dur}–{max_dur} seconds. Maximum {max_clips} clips. Output JSON array only.

LANGUAGE: ALL text fields MUST be in English.

---

CLIP BOUNDARIES (NON-NEGOTIABLE)

**Start**: Must begin at a natural moment — right before the key point or at a clear sentence start. Give 1-2 seconds of context so viewers understand what's being taught. NEVER start mid-word or mid-explanation.

**End**: Must end at a natural stopping point — after the explanation lands, at a sentence boundary, or at a satisfying conclusion. Use [PAUSE] markers as natural endpoints. NEVER cut mid-sentence or mid-thought. The clip must feel COMPLETE.

**NEVER start with:** greetings, filler words ("so like", "okay", "um"), or silence >1s.

---

SCORE EACH CLIP

score_emotion — Educational clarity & weight of the point (THIS IS THE MOST IMPORTANT SCORE)
90-100: Deep insight, a principle made unforgettably clear, genuinely useful takeaway
70-89: Clear, solid explanation that adds real understanding
50-69: MILD value (somewhat informative, but shallow or obvious)
0-49: FLAT (vague, off-topic, or teaches nothing)

score_hook — Will viewers want to keep watching in the first 2 seconds?
90-100: Opens with a compelling question, gap, or "did you know" that sparks curiosity
70-89: Clear setup that signals something worth learning
50-69: Decent but not gripping
0-49: Slow/boring or confusing start

score_retention — Will viewers watch to the end?
90-100: Clean arc that resolves the question, satisfying payoff
70-89: Good flow, natural ending at sentence boundary
50-69: Watchable but slightly meandering
0-49: Trails off, no payoff, or cut mid-thought

score_personality — Does the creator's authentic teaching voice/authority show?
90-100: Distinctive, trustworthy educator voice — quotable, memorable framing
70-89: Clear personal teaching style, relatable examples
50-69: Generic explanation anyone could give
0-49: No voice, could be any narrator

clip_score = round((score_emotion * 0.40) + (score_hook * 0.30) + (score_retention * 0.20) + (score_personality * 0.10), 1)

---

SELECTION RULES

INCLUDE only if ALL true:
- clip_score >= {min_score}
- score_hook >= 65
- score_emotion >= 60
- score_retention >= 50
- At least TWO scores >= 70
- At least ONE of (score_hook, score_emotion) >= 80
- Clip starts and ends at natural boundaries (not mid-sentence)

If two clips overlap, keep the higher-scored one.

When uncertain, leave it out. A clip that teaches nothing does not belong in the output, even if it's amusing.

---

OUTPUT FIELDS

(1) reason — 1-2 sentences: what makes this moment educational (the insight, the lesson, the skill shown)
(2) topic — One sentence: what happens in this clip
(3) hook — EXACT first words from transcript (word-for-word)
(4) closing_line — EXACT last words from transcript (word-for-word, must be a natural endpoint)
(5) caption — Max 280 chars. Hook line + what happens + CTA + 2-3 hashtags
(6) title — Max 8 words for on-screen overlay. Casual English.
(7) comment_bait — Question to drive comments, <15 words, casual Indonesian

Return JSON array sorted by clip_score descending:

```json
[
  {{
    "start": 34.5,
    "end": 83.2,
    "reason": "...",
    "topic": "...",
    "hook": "...",
    "closing_line": "...",
    "caption": "...",
    "title": "...",
    "comment_bait": "...",
    "score_emotion": 85,
    "score_hook": 78,
    "score_retention": 72,
    "score_personality": 65,
    "clip_score": 78.4
  }}
]
```

CRITICAL: Every clip MUST include start, end, ALL four score_* fields (integers 0-100), clip_score (float), and all text fields. Missing fields = discarded.
"""

