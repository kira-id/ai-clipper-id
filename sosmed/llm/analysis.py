"""
LLM analysis: find engaging clips in transcript.
"""

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..utils import log, SYSTEM_PROMPT, MAX_CLIPS_HARD_LIMIT
from .backends import call_llm


def _build_transcript_text(
    segments: list[dict[str, Any]],
    energy_events: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build transcript with sentence structure hints and audio energy markers.

    Format: [start-end] text [marker] [gap]
    - Gaps >1.0s marked with [PAUSE] to indicate sentence boundaries
    - Energy spikes/high moments annotated with [ENERGY SPIKE] / [HIGH ENERGY]
    - Non-verbal energy clusters (no overlapping transcript line, in_speech
      False) are listed in a trailing section so the LLM can still clip a
      genuine scream/reaction Whisper failed to transcribe, without
      misattributing it to a nearby spoken line.
    """
    # Pre-sort energy events by start time for efficient lookup via pointer scan
    sorted_energy = sorted(energy_events, key=lambda e: e["start"]) if energy_events else []

    lines: list[str] = []
    ev_ptr = 0  # pointer into sorted_energy — advances as segments progress
    used_for_line: set[int] = set()
    for i, s in enumerate(segments):
        st = f"{s['start']:.0f}" if s['start'] == int(s['start']) else f"{s['start']:.1f}"
        en = f"{s['end']:.0f}" if s['end'] == int(s['end']) else f"{s['end']:.1f}"

        # Check for energy events overlapping this segment (scan forward only)
        energy_marker = ""
        if sorted_energy:
            # Advance pointer past events that end before this segment starts
            while ev_ptr < len(sorted_energy) and sorted_energy[ev_ptr]["end"] <= s["start"]:
                ev_ptr += 1
            # Check events from pointer onward (they start before or during this segment)
            for j in range(ev_ptr, len(sorted_energy)):
                ev = sorted_energy[j]
                if ev["start"] >= s["end"]:
                    break  # all remaining events are past this segment
                # Overlap confirmed
                label = "🔥 ENERGY SPIKE" if ev["kind"] == "spike" else "⚡ HIGH ENERGY"
                energy_marker = f" [{label}]"
                used_for_line.add(j)
                break

        # Gap to next segment for sentence boundary detection
        gap_marker = ""
        if i < len(segments) - 1:
            gap = segments[i + 1]["start"] - s["end"]
            if gap >= 1.0:
                gap_marker = f" [PAUSE {gap:.1f}s]"
            elif gap >= 0.5:
                gap_marker = f" [gap {gap:.2f}s]"

        lines.append(f"[{st}-{en}]{s['text']}{energy_marker}{gap_marker}")

    # Trailing section: non-verbal energy moments (no overlapping spoken line).
    unmatched = [
        ev for j, ev in enumerate(sorted_energy)
        if j not in used_for_line and not ev.get("in_speech", True)
    ]
    if unmatched:
        lines.append("")
        lines.append("[NON-VERBAL ENERGY MOMENTS — sudden loud reactions with no transcript; "
                     "clip these standalone, do NOT attach to nearby text]")
        for ev in unmatched:
            label = "🔥 ENERGY SPIKE" if ev["kind"] == "spike" else "⚡ HIGH ENERGY"
            lines.append(f"[{ev['start']:.1f}-{ev['end']:.1f}] {label}")

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
        f"Analyze this transcript and extract ONLY clips with genuine educational value.\n"
        f"Prioritize clear explanations, useful lessons, how-tos, and demonstrable skill over pure entertainment.\n"
        f"Filter out clips that are merely amusing or interesting but teach nothing.\n"
        f"Duration: {min_dur}–{max_dur}s. Max {max_clips} clips. clip_score ≥ {min_score}.\n"
    )
    if chunk_info:
        header += f"{chunk_info}\n"
    return f"{header}\nTRANSKRIP:\n{transcript}"


def _clip_score(c: dict[str, Any]) -> float:
    """
    Compute clip_score from the four model-emitted sub-scores.

    Weights (from SYSTEM_PROMPT): educational clarity 40%, hook 30%,
    retention 20%, teaching voice 10%.  Returns 0.0 when no scores are
    present so the validator's min_score gate drops unscoreable clips.
    """
    emotion = _to_score(c.get("score_emotion"))
    hook = _to_score(c.get("score_hook"))
    retention = _to_score(c.get("score_retention"))
    personality = _to_score(c.get("score_personality"))
    total = emotion + hook + retention + personality
    if total == 0:
        return 0.0
    return round(
        emotion * 0.40 + hook * 0.30 + retention * 0.20 + personality * 0.10,
        1,
    )


def _to_score(value: Any) -> int:
    """Parse score-like values into a bounded integer in [0, 100]."""
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _validate_clips(
    clips: list[dict[str, Any]],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Sanitize, deduplicate, and cap the clip list — keep the best educational clips.

    Selection is driven solely by the model-emitted score_* sub-scores via
    ``_clip_score`` and the caller's ``min_score`` floor.  We deliberately do
    NOT re-impose per-dimension viral floors: the live ``SYSTEM_PROMPT``
    instructs the model to emit inclusive, education-first scores, and a
    second, contradictory gate here only drops otherwise-good clips.
    """
    valid: list[dict[str, Any]] = []
    seen_ranges: list[tuple[float, float]] = []

    # Pre-compute scores once; drop any clip missing both score_* fields and a
    # usable clip_score (the model must grade what it returns).
    scored_clips = []
    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        score = _clip_score(c)
        if score == 0.0 and not (
            isinstance(c.get("clip_score"), (int, float)) and c["clip_score"] > 0
        ):
            log("DEBUG", f"Skip '{c.get('title', '?')}' [{s:.0f}-{e:.0f}]: no score_* fields")
            continue
        c["_score"] = score
        scored_clips.append(c)
    # Highest clip_score first; emotion is the secondary tiebreak (the
    # strongest educational-weight signal per SYSTEM_PROMPT).
    scored_clips.sort(key=lambda x: (-x["_score"], -int(x.get("score_emotion", 0) or 0)))

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

        # Minimum score check (single, explicit floor from the caller)
        if score < min_score:
            log("DEBUG", f"Skip '{title}' [{s:.0f}-{e:.0f}]: score {score:.1f} < {min_score}")
            continue

        # Overlap check — no near-duplicates (see _merge_chunk_clips for the
        # keep-the-longer-clip policy applied earlier in the pipeline).
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
        c.setdefault("reason", "")
        c.setdefault("topic", "")
        c.setdefault("caption", "")
        c.setdefault("hook", "")
        c.setdefault("closing_line", "")
        c.setdefault("comment_bait", "")
        c.setdefault("social_description", "")
        # Never allow a bare "Clip N" title — derive a real one if needed.
        from .utils import ensure_real_title
        ensure_real_title(c, fallback_rank=c.get("rank"))
        c["clip_score"] = score
        c.pop("_score", None)
        c.pop("engagement_score", None)

        valid.append(c)
        if len(valid) >= max_clips:
            break

    # Re-rank by clip_score, tiebreak by emotion (strongest educational-weight signal).
    valid.sort(key=lambda x: (-x.get("clip_score", 0), -int(x.get("score_emotion", 0) or 0)))
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
    When duplicates are found, keep the more complete (longer) clip.
    """
    # Sort all clips by start time
    all_clips.sort(key=lambda c: (float(c.get("start", 0)), -_clip_score(c)))

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
                # Pick the more COMPLETE clip: a clip that fully contains (or is
                # much longer than) another is the better standalone moment to
                # ship.  Prefer the longer one; only fall back to the higher
                # score when lengths are within ~5s of each other.
                clip_dur = e - s
                existing_dur = ee - es
                if clip_dur > existing_dur + 5 or (
                    abs(clip_dur - existing_dur) <= 5
                    and _clip_score(clip) > _clip_score(existing)
                ):
                    deduped[i] = clip
                is_dup = True
                break
        if not is_dup:
            deduped.append(clip)

    log("INFO", f"Merged {len(all_clips)} raw clips → {len(deduped)} after dedup")

    # Now validate the deduped list
    return _validate_clips(deduped, min_dur, max_dur, max_clips, min_score, video_duration)


def _resolve_raw_cache_path(
    raw_clips_cache_file: "str | Path | None",
    *,
    min_duration: int,
    max_duration: int,
    max_clips: int,
    min_score: int,
    chunk_duration: float,
    chunk_overlap: float,
    n_segments: int,
    system_prompt: str | None = None,
) -> "Path | None":
    """Resolve the param-aware raw-clip cache path (deterministic, single pass).

    The caller passes a *base* path. We fold the LLM-affecting parameters into a
    short hash so changing them never silently reuses stale LLM output. This is
    the ONLY place the transformation happens — read and write both call it, so
    the two sides can never drift (a double-transform would otherwise read
    `base.sig.json` but write `base.sig.sig.json` and always miss).
    """
    if not raw_clips_cache_file:
        return None
    p = Path(raw_clips_cache_file)
    if p.suffix != ".json":
        return p
    params_sig = json.dumps(
        {
            "min_duration": min_duration,
            "max_duration": max_duration,
            "max_clips": max_clips,
            "min_score": min_score,
            "chunk_duration": chunk_duration,
            "chunk_overlap": chunk_overlap,
            "n_segments": n_segments,
            "system_prompt": system_prompt or "",
        },
        sort_keys=True,
    )
    sig = hashlib.sha1(params_sig.encode("utf-8")).hexdigest()[:10]
    return p.with_name(f"{p.stem}.{sig}{p.suffix}")


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
    energy_events: list[dict[str, Any]] | None = None,
    llm_parallel: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Ask LLM to find educational clips (up to *max_clips*).

    For long videos, splits the transcript into overlapping chunks and calls
    the LLM on each chunk, then merges and deduplicates results across chunks
    (overlapping/duplicate moments are collapsed, keeping the more complete clip).

    If raw_clips_cache_file is provided and exists, loads cached raw LLM
    results instead of calling the LLM again.
    """
    if raw_clips_cache_file:
        raw_clips_cache_file = _resolve_raw_cache_path(
            raw_clips_cache_file,
            min_duration=min_duration,
            max_duration=max_duration,
            max_clips=max_clips,
            min_score=min_score,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap,
            n_segments=len(segments),
            system_prompt=system_prompt,
        )
        if raw_clips_cache_file.exists():
            text = raw_clips_cache_file.read_text(encoding="utf-8")
            if not text.strip():
                log("WARN", f"Cache file {raw_clips_cache_file} is empty — treating as cache miss")
                all_raw_clips = None
            else:
                try:
                    all_raw_clips = json.loads(text)
                    log("OK", f"Loaded {len(all_raw_clips)} raw clips from cache (skipped LLM calls)")
                except json.JSONDecodeError as e:
                    log("WARN", f"Cache file {raw_clips_cache_file} is corrupt ({e}) — treating as cache miss")
                    all_raw_clips = None
        else:
            # Will generate and cache below
            all_raw_clips = None
    else:
        all_raw_clips = None
    
    # If not cached, call LLM
    if all_raw_clips is None:
        # Split into chunks for iterative processing
        chunks = _chunk_segments(segments, chunk_duration, chunk_overlap)

        effective_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
        system = effective_prompt.format(
            min_dur=min_duration,
            max_dur=max_duration,
            max_clips=max_clips,
            min_score=min_score,
        )

        n_chunks = len(chunks)

        def _process_chunk(idx: int, chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Build the prompt for one chunk and call the LLM. Returns raw clips.

            On a parse failure that yields 0 clips from a non-empty chunk, we
            retry ONCE with the chunk split in half. A failed (e.g. reasoning
            model that dumps CoT into content) call otherwise silently loses
            the whole time range — in the observed run chunk 3/3 returned 0
            clips, dropping ~356s (716s–1072s) of the video with no recovery.
            """
            raw_clips = _call_chunk_llm(idx, chunk, min_duration, max_duration,
                                        max_clips, min_score, n_chunks)
            if not raw_clips and len(chunk) > 1:
                # Recover: split this chunk and try each half once.
                mid = len(chunk) // 2
                log("WARN", f"Chunk {idx}/{n_chunks} returned 0 clips — "
                            f"retrying with halved sub-chunks to recover coverage")
                halves = [chunk[:mid], chunk[mid:]]
                recovered: list[dict[str, Any]] = []
                for h_i, half in enumerate(halves, 1):
                    sub = _call_chunk_llm(f"{idx}.{h_i}", half, min_duration,
                                          max_duration, max_clips, min_score, n_chunks)
                    recovered.extend(sub)
                log("OK", f"Chunk {idx}/{n_chunks} recovered {len(recovered)} clips "
                          f"from halved sub-chunks")
                return recovered
            return raw_clips

        def _call_chunk_llm(idx, chunk, min_duration, max_duration, max_clips,
                            min_score, n_chunks) -> list[dict[str, Any]]:
            """Single LLM attempt for a (sub-)chunk."""
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]
            # Generous per-chunk budget — maximize output
            clips_per_chunk = max(10, math.ceil(max_clips / max(n_chunks, 1) * 1.5))
            clips_per_chunk = min(clips_per_chunk, max_clips)

            transcript = _build_transcript_text(chunk, energy_events)
            chunk_info = ""
            if n_chunks > 1:
                chunk_info = (
                    f"Ini bagian {idx}/{n_chunks} video "
                    f"({chunk_start:.0f}s-{chunk_end:.0f}s). "
                    f"Ekstrak momen edukatif di bagian ini — hanya bagian yang benar-benar mengajarkan sesuatu yang layak jadi klip."
                )
                log("INFO", f"Chunk {idx}/{n_chunks}: {chunk_start:.0f}s → {chunk_end:.0f}s "
                           f"({len(chunk)} segments, asking for ≤{clips_per_chunk} clips)")

            user = _build_user_prompt(
                transcript, min_duration, max_duration,
                clips_per_chunk, min_score, chunk_info,
            )

            raw_clips = call_llm(system, user, api_key, llm_model)
            log("OK", f"Chunk {idx}/{n_chunks}: LLM returned {len(raw_clips)} clips")
            return raw_clips

        all_raw_clips: list[dict[str, Any]] = []
        if llm_parallel and n_chunks > 1:
            # Run chunks concurrently — the OpenAI client is thread-safe and the
            # chunks are independent, so this collapses N serial calls into ~1 call
            # of wall-clock time (bounded by the slowest single chunk).
            log("INFO", f"Running {n_chunks} chunks in PARALLEL")
            results_by_idx: dict[int, list[dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=min(n_chunks, 4)) as pool:
                futures = {
                    pool.submit(_process_chunk, idx, chunk): idx
                    for idx, chunk in enumerate(chunks, 1)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        results_by_idx[idx] = fut.result()
                    except Exception as e:  # surface but don't silently drop
                        log("ERROR", f"Chunk {idx} failed: {e}")
                        results_by_idx[idx] = []
            # Preserve chunk order so merged output stays deterministic
            for idx in range(1, n_chunks + 1):
                all_raw_clips.extend(results_by_idx.get(idx, []))
        else:
            for idx, chunk in enumerate(chunks, 1):
                all_raw_clips.extend(_process_chunk(idx, chunk))

        log("INFO", f"Total raw clips from {n_chunks} chunk(s): {len(all_raw_clips)}")

        # Cache raw results for future runs. raw_clips_cache_file is already the
        # fully-resolved param-aware path (set on the read side above), so we
        # write to it directly — re-resolving here would double-hash the name.
        if raw_clips_cache_file:
            raw_clips_cache_file.parent.mkdir(parents=True, exist_ok=True)
            raw_clips_cache_file.write_text(json.dumps(all_raw_clips, indent=2, ensure_ascii=False), encoding="utf-8")
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
