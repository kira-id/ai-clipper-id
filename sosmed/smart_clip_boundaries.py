"""
Smart clip boundary adjustment: optimize for viral-worthy content.

This module intelligently adjusts clip start/end times to:
1. Start at strong hook points (power words, not filler)
2. End at natural sentence boundaries (complete thoughts, punchlines)
3. Validate and correct LLM's hook/closing_line to match actual transcript
4. Score and select the best ending points for maximum retention
"""

from typing import Any


# ── Filler words that make WEAK hooks ──────────────────────────────────────
WEAK_HOOK_WORDS = {
    # Indonesian fillers
    "uh", "um", "eh", "ah", "uhm", "em", "hmm", "mm", "mmm",
    "jadi", "terus", "nah", "ya", "iya", "oke", "ok",
    "gitu", "kayak", "maksudnya", "sebentar", "anu",
    "apa", "sih", "dong", "deh", "nih", "tuh", "lah", "kan", "kok",
    "gini", "gitulah", "lho", "loh", "aduh", "astaga",
    "eung", "euh", "oh", "hah", "wah",
    # English fillers
    "the", "a", "an", "so", "like", "you know", "i mean",
    "well", "actually", "basically", "right", "okay", "alright",
    # Hesitation
    "hello", "halo", "hai", "good morning", "good afternoon",
    "selamat pagi", "selamat siang", "selamat sore",
}

# Power words that make STRONG hooks (start here if possible)
POWER_HOOK_WORDS = {
    # Questions
    "kenapa", "mengapa", "bagaimana", "apa", "siapa", "kapan", "di mana", "kemana", "darimana",
    "how", "what", "why", "when", "who", "where", "which",
    # Bold claims
    "sebenarnya", "faktanya", "kenyataannya", "padahal", "nyatanya",
    "actually", "fact", "truth", "honestly", "believe", "percaya",
    # Numbers/quantifiers
    "satu", "dua", "tiga", "pertama", "kedua", "ketiga",
    "1", "2", "3", "first", "second", "third",
    # Emotional/emphatic
    "yang", "paling", "sangat", "benar", "tidak", "never", "ever",
    "harus", "wajib", "wajib", "penting", "critical", "crucial",
    # Direct address
    "kalian", "kamu", "anda", "lo", "gue", "you", "we", "kita", "kami",
    # Surprising/controversial
    "tapi", "tetapi", "namun", "meskipun", "walaupun",
    "but", "however", "although", "despite",
    # Action words
    "bikin", "buat", "buat", "create", "make", "do", "lakukan",
    "get", "dapatkan", "take", "ambil", "give", "kasih", "beri",
}

# Words that make BAD endings (incomplete thought)
BAD_ENDING_WORDS = {
    "jadi", "terus", "nah", "ya", "iya", "oke", "ok",
    "gitu", "kayak", "maksudnya", "sebentar", "anu",
    "uh", "um", "eh", "ah", "hmm",
    "the", "a", "an", "so", "like", "well",
    # Conjunctions (lead to more content)
    "dan", "atau", "tapi", "tetapi", "namun", "serta",
    "because", "since", "although", "while", "if", "when",
    "karena", "kalau", "jika", "saat", "ketika", "sedangkan",
    # Prepositions ending
    "di", "ke", "dari", "pada", "dalam", "untuk", "dengan",
    "in", "on", "at", "to", "for", "from", "with",
}

# Words that make GOOD endings (conclusive)
GOOD_ENDING_WORDS = {
    # Pronouns (often end complete thoughts)
    "itu", "ini", "dia", "mereka", "kita", "aku", "saya",
    # Concluding phrases
    "begitu", "demikian", "saja", "aja", "sih", "lah",
    # Past/completed
    "sudah", "telah", "udah", "done", "finished", "complete",
    # Emphatic endings
    "banget", "sekali", "very", "really", "totally", "definitely",
    # Strong statements
    "pasti", "yakin", "sure", "certain", "jelas", "clear",
}


def _find_words_in_range(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    tolerance: float = 0.5,
) -> list[dict[str, Any]]:
    """Extract all words within a time range with some tolerance."""
    words = []
    for seg in segments:
        if seg["end"] < clip_start - tolerance or seg["start"] > clip_end + tolerance:
            continue
        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end = w.get("end", 0)
            w_text = w.get("word", "").strip()
            if not w_text:
                continue
            if w_end > clip_start - tolerance and w_start < clip_end + tolerance:
                words.append({
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "seg_id": seg.get("id"),
                })
    words.sort(key=lambda x: x["start"])
    return words


def _find_text_in_transcript(
    segments: list[dict[str, Any]],
    search_text: str,
    search_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]] | None:
    """Find a text phrase in the transcript and return matching words."""
    search_normalized = search_text.lower().strip()
    search_words = search_normalized.split()

    if not search_words:
        return None

    if search_range:
        words = _find_words_in_range(segments, search_range[0], search_range[1])
    else:
        words = _find_words_in_range(segments, 0, float("inf"))

    if not words:
        return None

    word_texts = [w["word"].lower().strip(".,!?;:") for w in words]
    search_len = len(search_words)

    for i in range(len(word_texts) - search_len + 1):
        window = word_texts[i:i + search_len]
        if window == search_words:
            return words[i:i + search_len]
        # Fuzzy match
        matches = sum(1 for sw, ww in zip(search_words, window) if sw in ww or ww in sw)
        if matches >= max(1, search_len - 1):
            return words[i:i + search_len]

    return None


def _find_sentence_boundaries(
    words: list[dict[str, Any]],
    min_gap: float = 0.7,
) -> list[int]:
    """Find indices where sentence boundaries likely occur."""
    boundaries = []

    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        if gap >= min_gap:
            boundaries.append(i)

    # Punctuation-based boundaries
    sentence_enders = {".", "?", "!", "。", "؟"}
    for i, w in enumerate(words):
        if w["word"].strip() and w["word"].strip()[-1] in sentence_enders:
            if i not in boundaries:
                boundaries.append(i)

    return sorted(set(boundaries))


def _find_strong_hook_position(
    words: list[dict[str, Any]],
    proposed_start: float,
    lookforward_window: float = 4.0,
) -> float:
    """
    Find the strongest hook position - optimized for viral content.

    Priority:
    1. Power words (questions, bold claims, numbers, emotional words)
    2. First substantive content word (skip ALL fillers)
    3. Ensure speech begins within 0.5s
    """
    if not words:
        return proposed_start

    words_in_window = [w for w in words if w["start"] >= proposed_start - 0.2 and w["start"] <= proposed_start + lookforward_window]

    if not words_in_window:
        return proposed_start

    # Skip leading fillers (up to 5 words)
    first_content_idx = 0
    for i in range(min(5, len(words_in_window))):
        word_lower = words_in_window[i]["word"].lower().strip(".,!?;:")
        if word_lower not in WEAK_HOOK_WORDS:
            first_content_idx = i
            break

    # Look for power words in the next 3 positions
    for i in range(first_content_idx, min(first_content_idx + 3, len(words_in_window))):
        word_lower = words_in_window[i]["word"].lower().strip(".,!?;:")
        if word_lower in POWER_HOOK_WORDS:
            # Start slightly before power word for natural lead-in
            return max(proposed_start, words_in_window[i]["start"] - 0.1)

    # Default: start at first non-filler content word
    first_word = words_in_window[first_content_idx]
    return max(proposed_start, first_word["start"] - 0.12)


def _score_ending_quality(
    words: list[dict[str, Any]],
    boundary_idx: int,
) -> int:
    """
    Score how good an ending point is (higher = better for viral retention).

    Scoring factors:
    - Gap after word (natural pause = complete thought)
    - Not ending on filler/conjunction
    - Ending on strong/conclusive words
    - Punctuation ending
    """
    word = words[boundary_idx]
    word_text = word["word"].lower().strip(".,!?;:")
    word_end = word["end"]

    # Gap after this word
    gap_after = 0.0
    if boundary_idx < len(words) - 1:
        gap_after = words[boundary_idx + 1]["start"] - word_end

    score = 0

    # Large gap = strong sentence boundary (+30)
    if gap_after >= 1.2:
        score += 30
    elif gap_after >= 0.8:
        score += 22
    elif gap_after >= 0.5:
        score += 14
    elif gap_after >= 0.3:
        score += 8

    # Not ending on weak word (+25)
    if word_text not in BAD_ENDING_WORDS:
        score += 25

    # Ending on strong/conclusive word (+20)
    if word_text in GOOD_ENDING_WORDS:
        score += 20

    # Punctuation ending (+20)
    if word["word"].strip()[-1] in {".", "?", "!", "。"}:
        score += 20

    # Bonus for question ending (creates curiosity) (+10)
    if word["word"].strip()[-1] == "?":
        score += 10

    # Bonus for exclamation (emotional peak) (+10)
    if word["word"].strip()[-1] == "!":
        score += 10

    return score


def _find_best_ending(
    words: list[dict[str, Any]],
    proposed_end: float,
    lookback_window: float = 6.0,
    lookahead_window: float = 5.0,
    min_duration: float = 5.0,
    max_end: float | None = None,
) -> float:
    """
    Find the BEST ending point for viral retention.

    Searches BOTH directions around ``proposed_end``:
      - backward, to trim a trailing half-sentence, and
      - forward, to *finish* a thought the LLM cut off mid-sentence.

    Only searching backward (the previous behaviour) meant a clip whose
    proposed end landed in the middle of a sentence could never be extended to
    complete it, so clips routinely ended abruptly on a conjunction.

    Strategy:
    1. Collect candidate sentence boundaries in the whole window.
    2. Score each for ending quality.
    3. Penalise distance from proposed_end, asymmetrically — extending a little
       to complete a sentence is much cheaper than truncating content away.
    4. Pick the highest scorer that respects min_duration / max_end.
    """
    if not words:
        return proposed_end

    window_start = max(0.0, proposed_end - lookback_window)
    window_end = proposed_end + lookahead_window
    if max_end is not None:
        window_end = min(window_end, max_end)

    words_in_window = [
        w for w in words
        if w["end"] >= window_start and w["end"] <= window_end
    ]

    if not words_in_window:
        return proposed_end

    boundaries = _find_sentence_boundaries(words_in_window, min_gap=0.6)

    first_word = words[0] if words else None
    min_end_time = (first_word["start"] if first_word else 0.0) + min_duration

    if not boundaries:
        # No clear boundary anywhere: back off to the last non-weak word.
        for i in range(len(words_in_window) - 1, -1, -1):
            if words_in_window[i]["end"] > proposed_end:
                continue
            word_text = words_in_window[i]["word"].lower().strip(".,!?;:")
            if word_text not in BAD_ENDING_WORDS:
                return words_in_window[i]["end"] + 0.25
        return min(proposed_end, words_in_window[-1]["end"] + 0.3)

    scored_boundaries: list[tuple[float, float, int]] = []
    for idx in boundaries:
        boundary_time = words_in_window[idx]["end"]
        if boundary_time < min_end_time:
            continue
        if max_end is not None and boundary_time > max_end:
            continue

        score = float(_score_ending_quality(words_in_window, idx))

        delta = boundary_time - proposed_end
        if delta >= 0:
            # Extending forward to complete a thought — cheap.
            score -= delta * 1.5
        else:
            # Truncating backward throws content away — expensive.
            score -= (-delta) * 4.0

        scored_boundaries.append((score, boundary_time, idx))

    if not scored_boundaries:
        return min(proposed_end, words_in_window[boundaries[-1]]["end"] + 0.2)

    scored_boundaries.sort(key=lambda x: -x[0])
    best_boundary_idx = scored_boundaries[0][2]

    return words_in_window[best_boundary_idx]["end"] + 0.25


def _validate_and_fix_hook_closing(
    clip: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate hook/closing_line match actual transcript.
    Auto-correct if they don't match exactly.
    """
    clip_start = clip.get("start", 0)
    clip_end = clip.get("end", 0)
    actual_words = _find_words_in_range(segments, clip_start, clip_end, tolerance=1.0)

    if not actual_words:
        return clip

    # Validate hook
    hook = clip.get("hook", "").strip()
    if hook:
        hook_search_end = min(clip_start + 5.0, clip_end)
        hook_words = _find_text_in_transcript(
            segments, hook,
            search_range=(clip_start, hook_search_end)
        )

        if hook_words:
            actual_hook = " ".join(w["word"] for w in hook_words)
            if actual_hook.lower() != hook.lower():
                clip["_hook_original"] = hook
                clip["hook"] = actual_hook
                clip["_hook_start_adjusted"] = clip["start"]
                clip["start"] = max(clip_start, hook_words[0]["start"] - 0.12)

    # Validate closing_line
    closing = clip.get("closing_line", "").strip()
    if closing:
        # Search a window that extends slightly PAST the proposed end too — the
        # LLM often quotes a closing line that finishes just after its own cut,
        # and clamping with min() would silently discard the real ending.
        closing_words = _find_text_in_transcript(
            segments, closing,
            search_range=(max(clip_end - 8.0, clip_start), clip_end + 5.0)
        )

        if closing_words:
            actual_closing = " ".join(w["word"] for w in closing_words)
            if actual_closing.lower() != closing.lower():
                clip["_closing_original"] = closing
                clip["closing_line"] = actual_closing
                clip["_closing_end_adjusted"] = clip["end"]
                clip["end"] = closing_words[-1]["end"] + 0.25

    return clip


def smart_adjust_clip_boundaries(
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    min_duration: float = 5.0,
    max_duration: float = 180.0,
    validate_hook_closing: bool = True,
    aggressive_optimization: bool = True,
) -> list[dict[str, Any]]:
    """
    Intelligently adjust clip boundaries for viral-worthy content.

    This function:
    1. Validates and corrects LLM's hook/closing_line to match actual transcript
    2. Finds power words for strong hooks (questions, bold claims, numbers)
    3. Scores ending points and selects the best for retention
    4. Ensures clips maintain minimum/maximum duration constraints

    Args:
        clips: List of clip dicts with 'start', 'end', 'hook', 'closing_line'
        segments: Transcript segments with word-level timestamps
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        validate_hook_closing: Whether to validate/correct hook and closing_line
        aggressive_optimization: If True, be more aggressive about finding optimal points

    Returns:
        Updated clips with viral-optimized boundaries
    """
    for clip in clips:
        # Store original LLM boundaries
        if "_llm_start" not in clip:
            clip["_llm_start"] = clip["start"]
            clip["_llm_end"] = clip["end"]

        # Step 1: Validate hook/closing_line
        if validate_hook_closing:
            clip = _validate_and_fix_hook_closing(clip, segments)

        # Step 2: Get words around the clip range. The tolerance must cover the
        # ending search's lookahead window, otherwise there are no candidate
        # words past clip["end"] and the ending can only ever be truncated.
        words = _find_words_in_range(
            segments,
            clip["start"],
            clip["end"],
            tolerance=6.0,
        )

        if not words:
            continue

        orig_start = clip["start"]
        orig_end = clip["end"]

        # Step 3: Find optimal hook position
        new_start = _find_strong_hook_position(words, orig_start, lookforward_window=4.0)

        # Step 4: Find optimal ending position (may extend past orig_end to
        # complete a sentence, bounded by max_duration).
        new_end = _find_best_ending(
            words,
            orig_end,
            lookback_window=6.0,
            lookahead_window=5.0,
            min_duration=min_duration,
            max_end=new_start + max_duration,
        )

        # Step 5: Apply constraints
        if new_end - new_start < min_duration:
            if orig_end - new_start >= min_duration:
                new_end = orig_end
            elif new_end - orig_start >= min_duration:
                new_start = orig_start
            else:
                continue

        if new_end - new_start > max_duration:
            new_end = new_start + max_duration

        # Apply if the start skipped filler, or the end moved to a real
        # sentence boundary (in either direction).
        if new_start > orig_start + 0.2 or abs(new_end - orig_end) > 0.4:
            clip["start"] = new_start
            clip["end"] = new_end

        # Log optimization details for debugging
        if new_start != clip.get("_llm_start") or new_end != clip.get("_llm_end"):
            clip["_boundary_optimized"] = True

    return clips


# Backward compatibility
def tighten_clip_boundaries(
    clips: list[dict],
    segments: list[dict],
    padding: float = 0.15,
    max_gap: float = 2.0,
    min_speech_density: float = 0.5,
) -> list[dict]:
    """Legacy function - redirects to smart_adjust_clip_boundaries."""
    return smart_adjust_clip_boundaries(
        clips,
        segments,
        min_duration=5.0,
        max_duration=180.0,
        validate_hook_closing=True,
    )
