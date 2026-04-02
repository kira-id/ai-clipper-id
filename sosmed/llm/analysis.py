"""
LLM analysis: find engaging clips in transcript.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

from ..utils import log, SYSTEM_PROMPT, MAX_CLIPS_HARD_LIMIT
from .backends import call_llm


def _build_transcript_text(segments: list[dict[str, Any]]) -> str:
    """
    Build transcript with sentence structure hints for better clip selection.
    
    Format: [start-end] text | gap_to_next
    - Shows gap (in seconds) to next segment to help LLM identify natural pauses
    - Gaps >1.0s marked with [PAUSE] to indicate sentence boundaries
    """
    lines: list[str] = []
    for i, s in enumerate(segments):
        # Use integer seconds when possible to save chars
        st = f"{s['start']:.0f}" if s['start'] == int(s['start']) else f"{s['start']:.1f}"
        en = f"{s['end']:.0f}" if s['end'] == int(s['end']) else f"{s['end']:.1f}"
        
        # Calculate gap to next segment (for sentence boundary detection)
        gap_marker = ""
        if i < len(segments) - 1:
            gap = segments[i + 1]["start"] - s["end"]
            if gap >= 1.0:
                gap_marker = f" [PAUSE {gap:.1f}s]"
            elif gap >= 0.5:
                gap_marker = f" [gap {gap:.2f}s]"
        
        lines.append(f"[{st}-{en}]{s['text']}{gap_marker}")
    return "\n".join(lines)


def _chunk_segments(
    segments: list[dict[str, Any]],
    chunk_duration: float = 480.0,
    overlap_duration: float = 60.0,
) -> list[list[dict[str, Any]]]:
    """
    Split segments into time-based chunks for iterative LLM processing.

    Each chunk covers ~chunk_duration seconds of transcript with overlap_duration
    overlap to avoid missing clips that span chunk boundaries.
    """
    if not segments:
        return []

    total_start = segments[0]["start"]
    total_end = segments[-1]["end"]
    total_dur = total_end - total_start

    # If total duration fits in one chunk, no need to split
    if total_dur <= chunk_duration * 1.3:
        return [segments]

    chunks: list[list[dict[str, Any]]] = []
    window_start = total_start

    while window_start < total_end:
        window_end = window_start + chunk_duration
        # Gather segments that overlap with this window
        chunk = [
            s for s in segments
            if s["end"] > window_start and s["start"] < window_end + overlap_duration
        ]
        if chunk:
            chunks.append(chunk)
        window_start += chunk_duration

    log("INFO", f"Split {total_dur:.0f}s transcript into {len(chunks)} chunks "
               f"(~{chunk_duration:.0f}s each, {overlap_duration:.0f}s overlap)")
    return chunks


def _build_user_prompt(
    transcript: str,
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    chunk_info: str = "",
) -> str:
    """Build the user prompt for LLM."""
    header = (
        f"Analyze this transcript and extract every clip with a realistic shot at high views.\n"
        f"Duration: {min_dur}–{max_dur}s. Max {max_clips} clips. clip_score ≥ {min_score}.\n"
    )
    if chunk_info:
        header += f"{chunk_info}\n"
    return f"{header}\nTRANSKRIP:\n{transcript}"


def _compute_clip_score(c: dict[str, Any]) -> float:
    """
    Compute clip_score using the educational virality weighting.
    
    If the clip has an existing valid clip_score and all new dimension scores 
    are missing (legacy format), preserve the original score.
    """
    scores = _normalize_score_fields(c)
    hook = scores["score_hook"]
    insight_density = scores["score_insight_density"]
    retention = scores["score_retention"]
    emotional_payoff = scores["score_emotional_payoff"]
    clarity = scores["score_clarity"]

    # If all new scores ended up coming from legacy mapping and original clip_score exists,
    # prefer the original (legacy format clips)
    has_original_score = "clip_score" in c and isinstance(c.get("clip_score"), (int, float))
    has_new_fields = any(c.get(f) is not None for f in [
        "score_hook", "score_insight_density", "score_retention",
        "score_emotional_payoff", "score_clarity"
    ])

    if has_original_score and not has_new_fields and c["clip_score"] > 0:
        # Legacy format clip with original score - return it
        return float(c["clip_score"])

    total = hook + insight_density + retention + emotional_payoff + clarity
    if total > 0:
        return round(
            hook * 0.30
            + insight_density * 0.25
            + retention * 0.20
            + emotional_payoff * 0.15
            + clarity * 0.10,
            1,
        )

    # No scores at all — the LLM returned this clip without scoring fields.
    # The LLM chose it, so assign a default passing score.
    return 70.0


def _to_score(value: Any) -> int:
    """Parse score-like values into a bounded integer in [0, 100]."""
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _normalize_score_fields(c: dict[str, Any]) -> dict[str, int]:
    """
    Normalize per-dimension score fields to keep output stable.
    Falls back to legacy score fields if new ones are missing.
    """
    # Try new field names first
    hook = _to_score(c.get("score_hook"))
    insight_density = _to_score(c.get("score_insight_density"))
    retention = _to_score(c.get("score_retention"))
    emotional_payoff = _to_score(c.get("score_emotional_payoff"))
    clarity = _to_score(c.get("score_clarity"))
    
    # If all new fields are missing, try legacy fields
    if hook == 0 and insight_density == 0 and retention == 0 and emotional_payoff == 0 and clarity == 0:
        # Map legacy fields to new ones
        hook = _to_score(c.get("score_newsworthy", c.get("score_shareability", 0)))  # newsworthy acts as hook
        insight_density = _to_score(c.get("score_informative", c.get("score_educational", 0)))
        retention = _to_score(c.get("score_energy", 0))  # energy acts as retention
        emotional_payoff = _to_score(c.get("score_entertainment", 0))
        clarity = _to_score(c.get("score_easy", 0))

    return {
        "score_hook": hook,
        "score_insight_density": insight_density,
        "score_retention": retention,
        "score_emotional_payoff": emotional_payoff,
        "score_clarity": clarity,
    }


# Titles/topics that indicate non-viral content (intros, outros, etc.)
# NOTE: Q&A / tanya jawab is NOT excluded — often contains great insights
_LOW_VALUE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:selamat datang|pembuka|opening|penutup|closing|terima kasih|"
    r"thank you|perkenalan|introduction|polling|vote|subscribe|"
    r"topik yang akan|webinar akan|rekaman|pendekatan materi|"
    r"ajak partisipasi|sapa penonton|ucapan|disclaimer|house\s*keeping)"
)


def _is_low_value_clip(c: dict[str, Any]) -> bool:
    """Check if clip is likely low-value (intro, outro, generic Q&A, etc.)."""
    title = (c.get("title", "") or "").strip()
    topic = (c.get("topic", "") or "").strip()
    combined = f"{title} {topic}"
    return bool(_LOW_VALUE_PATTERNS.search(combined))


def _validate_clips(
    clips: list[dict[str, Any]],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Sanitize, deduplicate, and cap the clip list — strict quality gate for viral hits only."""
    valid: list[dict[str, Any]] = []
    seen_ranges: list[tuple[float, float]] = []

    # Sort by score descending first so higher-quality clips get priority
    scored_clips = []
    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        score = _compute_clip_score(c)
        # Penalize low-value clips (intros, outros, disclaimers)
        if _is_low_value_clip(c):
            score = max(0, score - 25)
        c["_score"] = score
        scored_clips.append(c)
    # Tiebreaker: score_hook (stop-scrolling power is #1 virality predictor)
    scored_clips.sort(key=lambda x: (-x["_score"], -int(x.get("score_hook", 0) or 0)))

    for c in scored_clips:
        s, e = float(c["start"]), float(c["end"])
        dur = e - s
        title = c.get("title", "?")
        
        # Duration check
        if dur < min_dur or dur > max_dur:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: duration {dur:.0f}s outside {min_dur}-{max_dur}s")
            continue
        
        # Video duration check
        if video_duration and e > video_duration + 2:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: end exceeds video duration {video_duration:.0f}s")
            continue
        
        score = c["_score"]
        
        # Minimum score check
        if score < min_score:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: score {score:.1f} < {min_score}")
            continue
        
        # VIRAL REQUIREMENTS: Enforce strict score floors
        hook_score = int(c.get("score_hook", 0) or 0)
        retention_score = int(c.get("score_retention", 0) or 0)
        
        # No weak hooks allowed (must be ≥60)
        if hook_score < 60:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: hook score {hook_score} < 60 (weak hook)")
            continue
        
        # Must have strong ending (retention ≥50)
        if retention_score < 50:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: retention score {retention_score} < 50 (weak ending)")
            continue
        
        # At least two scores must be ≥70 (viral-tier quality)
        all_scores = [
            hook_score,
            int(c.get("score_insight_density", 0) or 0),
            retention_score,
            int(c.get("score_emotional_payoff", 0) or 0),
            int(c.get("score_clarity", 0) or 0),
        ]
        high_scores = sum(1 for score in all_scores if score >= 70)
        if high_scores < 2:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: only {high_scores} scores ≥70 (need 2+)")
            continue
        
        # Normalize score fields for output
        normalized = _normalize_score_fields(c)
        
        # Overlap check — no near-duplicates
        def _overlap_ratio(s1: float, e1: float, s2: float, e2: float) -> float:
            overlap = max(0, min(e1, e2) - max(s1, s2))
            shorter = min(e1 - s1, e2 - s2)
            return overlap / shorter if shorter > 0 else 0

        overlaps = any(_overlap_ratio(s, e, rs, re) > 0.5 for rs, re in seen_ranges)
        if overlaps:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: overlaps with existing clip")
            continue
        
        seen_ranges.append((s, e))
        
        # Ensure required fields
        c.setdefault("rank", len(valid) + 1)
        c.setdefault("title", f"Clip {c['rank']}")
        c.setdefault("reason", "")
        c.setdefault("topic", "")
        c.setdefault("caption", "")
        c.setdefault("hook", "")
        c.setdefault("closing_line", "")
        c.setdefault("comment_bait", "")
        c.update(normalized)
        c["clip_score"] = score
        c.pop("_score", None)
        c.pop("engagement_score", None)
        
        # Remove score fields from older algorithms if present
        for legacy in (
            "score_shareability",
            "score_educational",
            "score_entertainment",
            "score_easy",
            "score_informative",
            "score_energy",
            "score_newsworthy",
        ):
            c.pop(legacy, None)
        
        valid.append(c)
        if len(valid) >= max_clips:
            break

    # Re-rank by clip_score, tiebreak by hook strength (strongest virality signal)
    valid.sort(key=lambda x: (-x.get("clip_score", 0), -int(x.get("score_hook", 0) or 0)))
    for i, c in enumerate(valid, 1):
        c["rank"] = i

    return valid


def _merge_chunk_clips(
    all_clips: list[dict[str, Any]],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """
    Merge clips from multiple LLM chunks, removing near-duplicates.

    Two clips are considered duplicates if they overlap by > 30%.
    When duplicates are found, keep the one with the higher score.
    """
    # Sort all clips by start time
    all_clips.sort(key=lambda c: (float(c.get("start", 0)), -_compute_clip_score(c)))

    deduped: list[dict[str, Any]] = []
    for clip in all_clips:
        try:
            s, e = float(clip["start"]), float(clip["end"])
        except (KeyError, ValueError, TypeError):
            continue

        # Check for near-duplicate with already accepted clips
        is_dup = False
        for i, existing in enumerate(deduped):
            es, ee = float(existing["start"]), float(existing["end"])
            overlap = max(0, min(e, ee) - max(s, es))
            shorter = min(e - s, ee - es)
            if shorter > 0 and overlap / shorter > 0.7:
                # Keep the higher-scoring one
                if _compute_clip_score(clip) > _compute_clip_score(existing):
                    deduped[i] = clip
                is_dup = True
                break
        if not is_dup:
            deduped.append(clip)

    log("INFO", f"Merged {len(all_clips)} raw clips → {len(deduped)} after dedup")

    # Now validate the deduped list
    return _validate_clips(deduped, min_dur, max_dur, max_clips, min_score, video_duration)


def _find_gaps(
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    min_gap: float = 30.0,
) -> list[tuple[float, float]]:
    """
    Find time ranges in the transcript not covered by any clip.

    Returns list of (start, end) tuples for gaps >= min_gap seconds.
    """
    if not segments:
        return []

    total_start = segments[0]["start"]
    total_end = segments[-1]["end"]

    # Sort clips by start time
    sorted_clips = sorted(clips, key=lambda c: float(c.get("start", 0)))

    gaps: list[tuple[float, float]] = []
    cursor = total_start

    for c in sorted_clips:
        cs, ce = float(c["start"]), float(c["end"])
        if cs > cursor + min_gap:
            gaps.append((cursor, cs))
        cursor = max(cursor, ce)

    # Check trailing gap
    if total_end > cursor + min_gap:
        gaps.append((cursor, total_end))

    return gaps


def _segments_in_range(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Return segments that overlap with the given time range."""
    return [s for s in segments if s["end"] > start and s["start"] < end]


def find_clips(
    segments: list[dict[str, Any]],
    *,
    min_duration: int = 15,
    max_duration: int = 60,
    max_clips: int = MAX_CLIPS_HARD_LIMIT,
    min_score: int = 60,
    llm_model: str | None = None,
    api_key: str | None = None,
    video_duration: float | None = None,
    chunk_duration: float = 480.0,
    chunk_overlap: float = 60.0,
    raw_clips_cache_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Ask LLM to find ALL engaging clips (up to *max_clips*).

    For long videos (> ~8 min of transcript), splits transcript into
    overlapping chunks and calls the LLM iteratively on each chunk,
    then merges and deduplicates results across all chunks.

    After the first pass, identifies large gaps (uncovered time ranges)
    and does a second LLM pass to extract additional clips from those gaps.
    
    If raw_clips_cache_file is provided and exists, loads cached raw LLM results
    instead of calling LLM again.
    """
    if raw_clips_cache_file:
        raw_clips_cache_file = Path(raw_clips_cache_file)
        if raw_clips_cache_file.exists():
            log("INFO", f"Loading cached raw LLM clips from {raw_clips_cache_file}")
            all_raw_clips = json.loads(raw_clips_cache_file.read_text())
            log("OK", f"Loaded {len(all_raw_clips)} raw clips from cache (skipped LLM calls)")
        else:
            # Will generate and cache below
            all_raw_clips = None
    else:
        all_raw_clips = None
    
    # If not cached, call LLM
    if all_raw_clips is None:
        # Split into chunks for iterative processing
        chunks = _chunk_segments(segments, chunk_duration, chunk_overlap)

        system = SYSTEM_PROMPT.format(
            min_dur=min_duration,
            max_dur=max_duration,
            max_clips=max_clips,
            min_score=min_score,
        )

        all_raw_clips = []
        n_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, 1):
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]
            # Generous per-chunk budget — maximize output
            clips_per_chunk = max(10, math.ceil(max_clips / max(n_chunks, 1) * 1.5))
            clips_per_chunk = min(clips_per_chunk, max_clips)

            transcript = _build_transcript_text(chunk)
            chunk_info = ""
            if n_chunks > 1:
                chunk_info = (
                    f"Ini bagian {idx}/{n_chunks} video "
                    f"({chunk_start:.0f}s-{chunk_end:.0f}s). "
                    f"Ekstrak SEMUA momen menarik di bagian ini — setiap subtopik yang layak harus jadi klip."
                )
                log("INFO", f"Chunk {idx}/{n_chunks}: {chunk_start:.0f}s → {chunk_end:.0f}s "
                           f"({len(chunk)} segments, asking for ≤{clips_per_chunk} clips)")

            user = _build_user_prompt(
                transcript, min_duration, max_duration,
                clips_per_chunk, min_score, chunk_info,
            )

            raw_clips = call_llm(system, user, api_key, llm_model)
            log("OK", f"Chunk {idx}/{n_chunks}: LLM returned {len(raw_clips)} clips")
            all_raw_clips.extend(raw_clips)

        log("INFO", f"Total raw clips from {n_chunks} chunk(s): {len(all_raw_clips)}")
        
        # Cache raw results for future runs
        if raw_clips_cache_file:
            raw_clips_cache_file = Path(raw_clips_cache_file)
            raw_clips_cache_file.parent.mkdir(parents=True, exist_ok=True)
            raw_clips_cache_file.write_text(json.dumps(all_raw_clips, indent=2, ensure_ascii=False))
            log("OK", f"Cached {len(all_raw_clips)} raw LLM clips → {raw_clips_cache_file}")

    # Merge, deduplicate, and validate across all chunks
    # Determine number of chunks from segments (for merge logic)
    chunks_for_merge = _chunk_segments(segments, chunk_duration, chunk_overlap)
    n_chunks = len(chunks_for_merge)
    
    if n_chunks > 1:
        clips = _merge_chunk_clips(
            all_raw_clips, min_duration, max_duration,
            max_clips, min_score, video_duration,
        )
    else:
        clips = _validate_clips(
            all_raw_clips, min_duration, max_duration,
            max_clips, min_score, video_duration,
        )

    if not clips:
        log("WARN", "LLM returned 0 valid clips. The video may not have engaging segments.")
    else:
        log("OK", f"{len(clips)} valid clips after validation")
    return clips
