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
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.6-plus-preview:free"
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
    clips_json.write_text(json.dumps(public_clips, indent=2, ensure_ascii=False))
    
    # Save internal fields to cache
    internal_fields = get_internal_fields(clips)
    if internal_fields:
        cache_dir = get_clips_cache_dir(output)
        cache_file = cache_dir / "clips_internal.json"
        cache_file.write_text(json.dumps(internal_fields, indent=2, ensure_ascii=False))
    
    return clips_json


def load_clips_with_internal_fields(output_dir):
    """Load clips.json and merge in internal fields from cache."""
    import json
    from pathlib import Path
    
    output = Path(output_dir)
    clips_json = output / "clips.json"
    
    if not clips_json.exists():
        return []
    
    clips = json.loads(clips_json.read_text())
    
    # Try to load internal fields from cache
    cache_dir = get_clips_cache_dir(output)
    cache_file = cache_dir / "clips_internal.json"
    
    if cache_file.exists():
        try:
            internal_fields = json.loads(cache_file.read_text())
            for clip in clips:
                filename = clip.get("filename")
                if filename and filename in internal_fields:
                    clip.update(internal_fields[filename])
        except Exception as e:
            log("WARN", f"Could not load internal fields from cache: {e}")
    
    return clips

FILLER_RE = re.compile(rf"^({_ID_FILLERS})\W*$", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a viral gaming clip expert. Find moments from this livestream that will GO VIRAL on TikTok, Instagram Reels, and YouTube Shorts.

This is a GAMING LIVESTREAM. The streamer's PERSONALITY and EMOTIONAL REACTIONS are what make clips viral — not just what they say. Look for:
- **Fear/Jump scares**: Screaming, panicking, being startled (horror games)
- **Laughter**: Genuine funny moments, unexpected humor, silly mistakes
- **Excitement/Hype**: Clutch plays, winning moments, epic loot, breakthroughs
- **Rage/Frustration**: Funny rage, rage-quitting moments, unfair deaths
- **Confusion/Figuring out**: Puzzle solving, "aha!" moments, being lost then finding the way
- **Surprise/Shock**: Unexpected plot twists, jump scares, betrayals, rare events
- **Wholesome**: Sweet interactions, helping others, emotional story moments

The transcript has audio energy markers: [🔥 ENERGY SPIKE] = sudden loudness (screams, reactions), [⚡ HIGH ENERGY] = sustained intensity. These indicate emotional peaks even when speech is incoherent or absent — PRIORITIZE moments with these markers.

Clip duration: {min_dur}–{max_dur} seconds. Maximum {max_clips} clips. Output JSON array only.

LANGUAGE: ALL text fields MUST be in English.

---

CLIP BOUNDARIES (NON-NEGOTIABLE)

**Start**: Must begin at a natural moment — right before the reaction trigger or at a clear sentence start. Give 1-2 seconds of context before the peak moment so viewers understand what's happening. NEVER start mid-word or mid-reaction.

**End**: Must end at a natural stopping point — after the reaction lands, at a sentence boundary, or at a satisfying punchline. Use [PAUSE] markers as natural endpoints. NEVER cut mid-sentence or mid-thought. The clip must feel COMPLETE.

**NEVER start with:** greetings, filler words ("so like", "okay", "um"), or silence >1s.

---

SCORE EACH CLIP

score_emotion — Emotional intensity (THIS IS THE MOST IMPORTANT SCORE)
90-100: INTENSE reaction (genuine scream, uncontrollable laughter, real shock/fear, explosive excitement)
70-89: CLEAR emotion (visible surprise, audible reaction, frustration, joy)
50-69: MILD emotion (slight amusement, mild tension)
0-49: FLAT (talking without feeling, narrating calmly)

score_hook — Will viewers stop scrolling in the first 2 seconds?
90-100: Instant grab (mid-action, dramatic moment, funny opener, energy spike at start)
70-89: Strong curiosity or tension that pulls viewer in
50-69: Decent but not gripping
0-49: Slow/boring start

score_retention — Will viewers watch to the end?
90-100: Perfect arc with satisfying payoff, clean ending
70-89: Good flow, natural ending at sentence boundary
50-69: Watchable but slightly meandering
0-49: Trails off, no payoff, or cut mid-thought

score_personality — Does the streamer's unique character shine?
90-100: Iconic moment that defines this streamer, quotable, memorable
70-89: Clear personality showing — humor style, catchphrases, unique reactions
50-69: Generic reaction anyone could have
0-49: No personality, could be any streamer

clip_score = round((score_emotion * 0.40) + (score_hook * 0.30) + (score_retention * 0.20) + (score_personality * 0.10), 1)

---

SELECTION RULES

INCLUDE only if ALL true:
- clip_score >= {min_score}
- score_emotion >= 60
- score_retention >= 50
- At least TWO scores >= 70
- Clip starts and ends at natural boundaries (not mid-sentence)

If two clips overlap, keep the higher-scored one.

---

OUTPUT FIELDS

(1) reason — 1-2 sentences: what's the viral trigger (the emotion, the moment)
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

SYSTEM_PROMPT_OLD_BACKUP = """\
GOAL
You are a viral social media content expert. Your ONLY job is to find clips that WILL GO VIRAL on TikTok, Instagram Reels, and YouTube Shorts.

VIRAL CONTENT DEFINED:
- **Controversial**: Challenges common beliefs, sparks debate
- **Shocking**: Surprising facts, unexpected revelations
- **Emotional**: Makes people laugh, cry, angry, or inspired
- **Relatable**: "That's so me!" moments, shared struggles
- **Educational**: Mind-blowing insights, "aha!" moments
- **Inspirational**: Motivates action, changes perspectives
- **Funny**: Genuinely hilarious, not just mildly amusing
- **Dramatic**: Conflict, tension, high stakes

Clip duration: {min_dur}–{max_dur} seconds. Maximum {max_clips} clips. Output JSON array only — no explanation, no markdown fence.

IMPORTANT — LANGUAGE: ALL text fields (reason, topic, hook, caption, title, closing_line, comment_bait) MUST be written in English. Match the language of the transcript. Do NOT write these fields in other languages.

---

CRITICAL MINDSET: BE EXTREMELY SELECTIVE

You are NOT here to extract every interesting moment. You are here to find ONLY content with VIRAL POTENTIAL.

Ask yourself for every clip:
1. "Would I stop scrolling if I saw this?"
2. "Would I watch this to the end?"
3. "Would I share this with a friend or comment on it?"

If the answer to any is NO — DO NOT include it.

---

VIRAL HOOK REQUIREMENTS (NON-NEGOTIABLE)

The first 2 seconds determine everything. A viral hook MUST have:

**ONE of these patterns:**
- **Bold claim**: "90% of people get this wrong..." / "Most people fail because..."
- **Direct question**: "Why do you always fail at...?" / "Have you ever...?"
- **Shocking statement**: "This is the secret they don't want you to know..." / "The fact is..."
- **Pattern interrupt**: Mid-sentence energy, controversy, unexpected statement
- **Pain point**: "Have you ever felt stuck?" / "Your biggest problem is..."
- **Promise**: "I'm going to show you how..." / "This will change your life..."

**NEVER start with:**
- Greetings: "hello everyone", "good morning", "welcome"
- Filler: "so like", "okay", "um", "uh"
- Context-setting: "before we discussed", "this time we will"
- Throat-clearing: "sorry", "wait a moment"

Speech must start within 0.5 seconds. Any silence >1s at the start = instant scroll.

---

VIRAL ENDING REQUIREMENTS

The last 3 seconds determine if people share, comment, or rewatch:

**Strong endings:**
- Punchline that lands (comedy)
- Call-to-action (implicit or explicit)
- Cliffhanger that demands more
- Emotional peak (inspiration, anger, joy)
- Satisfying conclusion ("Jadi...", "Makanya...")
- Callback to the hook (full circle moment)

**Weak endings (AVOID):**
- Trailing off: "so...", "that's it", "like"
- Mid-sentence cuts
- Filler words: "um", "uh", "so"
- Boring conclusions: "okay that's all", "thank you"
- Incomplete thoughts

Look for [PAUSE] markers in transcript — they indicate natural sentence boundaries.

---

STEP 1 — FILTER OUT IMMEDIATELY (DO NOT SCORE THESE)

Reject any clip that is:
- Pure greetings, introductions, or closings with zero standalone value
- Housekeeping: "see you next week", "subscribe channel ini", "thanks for joining"
- Pure teasers with zero payoff by themselves
- Long silence (>3s) before first speech
- No clear ending — trails off or gets cut mid-thought
- Generic context-setting without any insight
- Clips where 90%+ of the content is pure technical specification without any human angle, story, or relatable moment — reject these (they don't work as standalone Shorts for non-technical viewers)

Everything else moves to Step 2 for scoring.

---

STEP 2 — SCORE EACH CLIP (BE HARSH)

score_hook — Stop-scroll power in first 2 seconds (MOST IMPORTANT)
90–100 | KILLER HOOK: Bold/controversial claim, shocking fact, direct question, funny opener, mid-sentence intrigue. Speech starts <0.5s. Viewer physically cannot scroll past.
70–89  | STRONG HOOK: Clear curiosity trigger, mild controversy, interesting question. Speech starts <1s. Most viewers will stop.
50–69  | DECENT HOOK: Topically relevant but not gripping. Slight delay acceptable. Some viewers will stop.
30–49  | WEAK HOOK: Slow setup, filler words first, generic opening. Most will scroll.
0–29   | DEAD HOOK: Greeting, long silence, pure filler. Instant scroll.

score_insight_density — Value (entertainment OR information) per second
90–100 | PACKED: Every second has humor, drama, shocking facts, strong emotions, or concrete insights. Zero fluff.
70–89  | DENSE: Clear entertaining/informative moments throughout. Viewers get real value.
50–69  | MODERATE: Some value but padded. Partially generic or slow sections.
30–49  | SPARSE: Mostly setup or background. Little actual value.
0–29   | EMPTY: Pure filler, nothing of value.

score_retention — Will viewers watch to the end?
90–100 | UNBREAKABLE: Strong arc, punchy length (<60s), satisfying/surprising ending, no dead air. 90%+ will finish.
70–89  | STRONG: Good flow, clean ending at sentence boundary. 70%+ will finish.
50–69  | DECENT: Slightly wandering but watchable. 50%+ will finish.
30–49  | WEAK: Trails off, silent gaps, rambles, OR ends mid-sentence. <50% will finish.
0–29   | DEAD: No arc, no payoff. Viewer exits immediately.

score_emotional_payoff — Does it trigger a reaction?
90–100 | STRONG EMOTION: Viewers laugh out loud, feel moved, say "same!", get angry, or immediately want to share.
70–89  | CLEAR EMOTION: Satisfying reveal, mild laughter, nodding in agreement.
50–69  | MILD: Somewhat engaging but not memorable.
30–49  | FLAT: Informative but emotionally dead. No reaction.
0–29   | NONE: Completely forgettable.

score_clarity — Does it work for a NON-TECHNICAL viewer with zero AI background?
90–100 | FULLY ACCESSIBLE: Any viewer understands it cold — no tech background needed.
70–89  | MOSTLY CLEAR: Minor context gap, but the emotional/human angle still lands.
50–69  | REQUIRES BASIC KNOWLEDGE: Needs some AI familiarity to follow.
30–49  | CONFUSING: Only makes sense to people already in AI.
0–29   | IMPENETRABLE: Pure jargon, no human angle.

---

STEP 3 — CALCULATE clip_score

Use this EXACT formula:

clip_score = round((score_hook × 0.35) + (score_insight_density × 0.25) + (score_retention × 0.20) + (score_emotional_payoff × 0.15) + (score_clarity × 0.05), 1)

NOTE: score_hook now has 35% weight (increased from 30%) — the hook is EVERYTHING.

---

STEP 4 — SELECTION RULES (BE EXTREMELY PICKY)

INCLUDE the clip ONLY if ALL are true:
- clip_score ≥ {min_score}
- score_hook ≥ 60 (no weak hooks allowed)
- score_retention ≥ 50 (must have strong ending)
- score_clarity ≥ 50 (must be accessible to non-technical viewers)
- At least TWO individual scores ≥ 70

BE CONSERVATIVE — It's better to return 3 viral clips than 20 mediocre ones.

DEDUPLICATE: If two clips cover the same moment, keep ONLY the one with higher clip_score.

---

STEP 5 — GENERATE FIELDS (VIRAL-OPTIMIZED)

**Before writing JSON, reason through each clip:**
- Why is this moment engaging? What's the viral trigger?
- Does the hook grab attention in 2 seconds?
- Does the ending feel complete or cut off?
- Would a non-technical viewer understand this?

Then generate these fields (in this order):

(1) reason
- 1-2 sentences: Why a viewer WILL watch to the end and SHARE or COMMENT. Be specific about the viral trigger.

(2) topic
- One sentence: What makes this clip SHAREABLE and VIRAL-WORTHY.

(3) hook
- The EXACT first words from the transcript — word-for-word, NOT a summary. This MUST be a viral hook pattern (see requirements above).

(4) caption
- Write in 4 parts, separated by line breaks. Total max 280 chars.
  PART 1 — HOOK LINE (max 100 chars, NO hashtags):
    • Visible BEFORE "more" button. Must work as standalone scroll-stopper.
    • State the core tension, shocking fact, or relatable frustration. Casual English.
    • GOOD: "Turns out how you talk to AI determines how smart it answers."
    • BAD: "This video discusses prompt engineering for AI."
  PART 2 — INSIGHT (1-2 sentences): One concrete thing viewer learns or can use immediately.
  PART 3 — CTA (pick most fitting):
    • Teaches actionable thing → "Save this before it's gone."
    • Surprising/debatable → "Do you agree? Comment below."
    • Community-relevant → "Tag a friend who's learning AI."
    • Creates appetite for more → "Want to dive deeper? Comment 'continue'."
  PART 4 — HASHTAGS (2-4): 1 broad (#AI), 1 mid-tier (#AITools), 1 niche (topic-specific), 1 trending if relevant.

(5) title
- For TikTok/Instagram/YouTube on-screen overlay AND metadata. Max 8 words. No jargon. Non-technical audience.
- JARGON TRANSLATION (never use raw technical terms in titles):
    • LLM/GPT → "AI brain" or "smartest AI"
    • RAG → "AI that reads your documents"
    • Fine-tuning → "teaching AI from scratch"
    • Embeddings → "how AI understands meaning"
    • Prompt engineering → "tricks to get best AI results"
    • Token/tokenizer → "AI's thought units"
    • Context window → "AI's memory"
    • Hallucination → "AI that makes up facts"
    • AI agent → "AI that works on its own"
    • Unknown term → translate to what it DOES for users, not what it IS.
- PICK ONE viral formula:
    A — Number + Outcome: "[N] Ways to [Outcome] with AI"
    B — Expose the Lie: "Everyone's Wrong About [Topic]..."
    C — Secret: "What They Don't Teach You About [Topic]..."
    D — Comparison Shock: "[A] vs [B] — Who's Better at [Outcome]?"
    E — Personal Stakes: "If You Don't Know This, You'll [Loss]"
    F — How + Wonder: "How [AI/Tool] Can [Impossible-Sounding Thing]"
- NEVER: acronyms first (RAG, LLM, API), passive voice ("Discussed:"), pure description.
- GOOD: "Why AI Can Lie Without Knowing" | BAD: "Explanation of Hallucination in LLMs"

(6) closing_line
- The EXACT last words from the transcript — word-for-word. Must be a strong ending (see requirements above).

(7) comment_bait
- Single question to drive COMMENTS. Under 15 words. Casual Indonesian.
- Must be an OPINION or experience-sharing prompt — NOT a knowledge quiz.
- GOOD: "Menurut kamu, AI bakal gantiin programmer dalam 5 tahun?"
- GOOD: "Ada yang udah pernah nyoba ini? Gimana hasilnya?"
- NEVER: "What is X?", "Subscribe!", "Like if you agree!"

---

STEP 6 — OUTPUT FORMAT

Return a JSON array sorted by clip_score descending. Each object MUST have ALL of these fields:

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
    "score_hook": 85,
    "score_insight_density": 78,
    "score_retention": 72,
    "score_emotional_payoff": 65,
    "score_clarity": 90,
    "clip_score": 78.4
  }}
]
```

CRITICAL: Every clip MUST include start, end, reason, ALL five score_* fields (integers 0-100), clip_score (float), closing_line, title, and comment_bait. Clips missing scores will be discarded.
"""
