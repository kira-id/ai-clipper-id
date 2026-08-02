"""Fix and improve clips: translate to English, fix caption-topic mismatches, deduplicate topics."""

import json
from typing import Any

from ..utils import log
from .backends import call_llm
from .prompts import get_prompt


def generate_single_clip_metadata(
    clip: dict[str, Any],
    segments: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate title/topic/caption/reason/hook for a full-video clip from transcript.

    Preserves user-provided title and caption if they are already set.
    """
    transcript = _build_transcript_for_metadata(segments)
    if not transcript:
        raise RuntimeError("Cannot generate single clip metadata without transcript text")

    prompt_text = _read_prompt("Generate Single Clip Metadata")
    clip_json = json.dumps([clip], ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nTranscript:\n{transcript}\n\nClip:\n{clip_json}"
    system_message = (
        "You are an expert short-form video strategist. "
        "Generate accurate, compelling metadata from the transcript and return only valid JSON."
    )

    result = call_llm(system_message, user_message, api_key, llm_model)
    if not result or not isinstance(result, list) or not isinstance(result[0], dict):
        raise RuntimeError(f"Single clip metadata generation failed: {result!r}")

    generated = result[0]
    updated = clip.copy()

    # Preserve user-provided title and caption, otherwise use LLM-generated
    user_title = str(clip.get("title", "") or "").strip()
    user_caption = str(clip.get("caption", "") or "").strip()

    for field in ["title", "topic", "caption", "reason", "hook", "closing_line", "comment_bait", "social_description"]:
        # Skip if user already provided this field
        if field == "title" and user_title:
            continue
        if field == "caption" and user_caption:
            continue

        value = str(generated.get(field, "") or "").strip()
        if not value:
            raise RuntimeError(f"Single clip metadata missing required field: {field}")
        updated[field] = value

    return updated


def _normalize_target_language(target_language: str | None) -> str:
    """Normalize the user-facing target-language option into a canonical key.

    Returns one of: "auto", "en", "id", or any open language code/label.
    "auto" means "keep the detected/transcribed language" (no translation).
    """
    if not target_language:
        return "en"
    t = str(target_language).strip().lower()
    if t in ("", "none", "auto", "source", "keep", "keep-original",
             "keep_original", "original"):
        return "auto"
    return t


def _target_label(target_language: str) -> str:
    """Human-readable language label for prompts.

    "auto" -> "the same language as the transcript" (caller should avoid this
    path; we only translate when target is an explicit language).
    """
    t = str(target_language).strip()
    if t.lower() in ("en", "english"):
        return "English"
    if t.lower() in ("id", "indonesian", "bahasa", "bahasa indonesia"):
        return "Indonesian"
    return t  # open language: used verbatim in the prompt


def _call_llm_clips(
    prompt_text: str,
    system_message: str,
    clips: list[dict[str, Any]],
    llm_model: str | None,
    api_key: str | None,
    max_retries: int = 2,
) -> list[dict[str, Any]] | None:
    """
    Serialize ``clips`` to JSON, call the LLM once, and return the parsed clip
    list (or ``None`` on total failure, so the caller can keep the originals).

    Centralizes the retry / malformed-response handling that the old three-step
    pipeline duplicated for every pass.
    """
    if not clips:
        return None

    clips_json = json.dumps(clips, ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nClips:\n{clips_json}"

    result: list[dict[str, Any]] | None = None
    for attempt in range(max_retries + 1):
        try:
            raw_result = call_llm(system_message, user_message, api_key, llm_model)
            if raw_result and isinstance(raw_result, list) and all(isinstance(c, dict) for c in raw_result):
                result = raw_result
                break
            else:
                log("WARN", f"Refine attempt {attempt + 1}/{max_retries + 1}: "
                             f"malformed response, retrying...")
        except Exception as e:
            log("WARN", f"Refine attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")
    return result


def _merge_refined_clips(
    clips: list[dict[str, Any]],
    result: list[dict[str, Any]],
    fields: tuple[str, ...] = ("title", "topic", "caption", "hook", "reason", "closing_line", "social_description"),
) -> list[dict[str, Any]]:
    """
    Merge LLM-refined fields back into the original clips by (start, end) key.

    Preserves every original metadata/timing field and only overwrites the
    listed content fields with the model's output. Clips the LLM dropped (it may
    have deduplicated) are re-introduced from ``clips`` (copy + merged fields) so
    no clip is silently lost, then re-ranked by score — matching the previous
    per-step behaviour exactly.
    """
    if not result:
        return clips

    orig_map = {
        (round(c.get("start", 0), 2), round(c.get("end", 0), 2)): c
        for c in clips
    }
    merged: list[dict[str, Any]] = []
    for improved in result:
        key = (round(improved.get("start", 0), 2), round(improved.get("end", 0), 2))
        if key in orig_map:
            m = orig_map[key].copy()
            for f in fields:
                if f in improved and str(improved[f] or "").strip():
                    m[f] = improved[f]
            merged.append(m)
            orig_map.pop(key, None)  # consume so we don't double-add below
        else:
            # LLM produced a clip not in originals — keep as-is (e.g. new merge).
            merged.append(improved)

    # Any original clip the LLM dropped gets re-added unchanged (no silent loss).
    for c in orig_map.values():
        merged.append(c.copy())

    merged.sort(key=lambda x: (-x.get("clip_score", 0), x.get("rank", 999)))
    for i, c in enumerate(merged, 1):
        c["rank"] = i
    return merged


def _refine_clips_combined(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    detected_language: dict[str, Any] | None = None,
    target_language: str | None = None,
) -> list[dict[str, Any]]:
    """
    Single-pass replacement for the old 3-step pipeline (translate → fix
    caption/topic → improve+dedupe). One LLM call now does all three, which
    roughly triples the speed of the ``refine`` step because the LLM was the
    only real cost and it was being billed three times in series.

    Falls back to returning the original clips (no mutation) if the single call
    fails entirely — the previous behaviour on a failed step.
    """
    if not clips:
        return clips

    target = _normalize_target_language(target_language)
    log("INFO", f"Refining {len(clips)} clips in one LLM pass (target language: {target})")

    # Pick the prompt that encodes translate + fix + improve for the language mode.
    if target == "auto":
        prompt_text = _read_prompt("Refine Clips (No Translate)")
        system_message = (
            "You are an expert social media strategist. Optimize strictly for "
            "virality first (shares/comments/replays), not completeness. Keep the "
            "original language."
        )
    elif target.lower().startswith("en"):
        # Skip translation when Whisper already detected English with high confidence.
        skip_translate = False
        if detected_language:
            lang = detected_language.get("language", "").lower()
            prob = detected_language.get("language_probability", 0.0)
            if lang == "en" and prob > 0.6:
                skip_translate = True
        prompt_key = (
            "Improve and Deduplicate Clips"
            if skip_translate
            else "Refine Clips (Translate to English)"
        )
        prompt_text = _read_prompt(prompt_key)
        system_message = (
            "You are an expert social media strategist. Optimize strictly for "
            "virality first (shares/comments/replays), not completeness or neutral "
            "informativeness."
        )
    else:
        label = _target_label(target)
        prompt_text = _read_prompt("Refine Clips (Translate to Target Language)", label)
        system_message = (
            "You are an expert social media strategist and translator. Optimize "
            "strictly for virality first (shares/comments/replays), not completeness."
        )

    result = _call_llm_clips(prompt_text, system_message, clips, llm_model, api_key)
    if not result:
        log("WARN", "Combined refine call failed after retries; keeping original clips unchanged")
        return clips

    merged = _merge_refined_clips(clips, result)
    log("OK", f"After combined refine: {len(merged)} clips")
    return merged


def fix_and_improve_clips(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    detected_language: dict[str, Any] | None = None,
    target_language: str | None = None,
) -> list[dict[str, Any]]:
    """
    Post-process clips to:
    1. Translate to the target language if needed (default English; "auto" keeps the
       original/transcribed language, which only fixes caption/topic & improves)
    2. Fix mismatched caption/topic pairs
    3. Improve titles/hooks/captions and deduplicate topics

    This runs AFTER find_clips() and BEFORE subtitle generation.

    The three operations are now performed in a SINGLE LLM pass
    (``_refine_clips_combined``) instead of three sequential calls, which makes
    the ``refine`` step ~3x faster. The original per-step helpers still exist
    for callers that need them (e.g. ``process_single`` uses ``_translate_to_language``).

    Args:
        detected_language: dict with keys "language" and "language_probability" from Whisper
        target_language: "en" (default), "id", "auto" (keep original), or any open
            language label/code. Controls which language clip metadata is translated into.
    """
    if not clips:
        log("WARN", "No clips to fix")
        return clips

    return _refine_clips_combined(
        clips,
        llm_model=llm_model,
        api_key=api_key,
        detected_language=detected_language,
        target_language=target_language,
    )


def _translate_to_language(
    clips: list[dict[str, Any]],
    target_label: str,
    llm_model: str | None = None,
    api_key: str | None = None,
    detected_language: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """
    Translate title, topic, caption, hook, reason to ``target_label`` if not already.

    Defaults to English for backward compatibility. Skips translation when the
    source was detected as the target language with >60% confidence (so an
    English-video -> English target does not waste an LLM call).

    Args:
        target_label: human-readable language name passed to the LLM
            (e.g. "English", "Indonesian", or any open language).
    """
    if not clips:
        return clips

    # Skip translation if Whisper already detected the *target* language with
    # high confidence (primarily relevant for English target with an English source).
    if detected_language and target_label.lower().startswith("en"):
        lang = detected_language.get("language", "").lower()
        prob = detected_language.get("language_probability", 0.0)
        if lang == "en" and prob > 0.6:
            log("OK", f"Language detected as English (p={prob:.0%}), skipping translation")
            return clips

    # Choose the right prompt: English target keeps the original wording,
    # any other target uses the parameterized prompt.
    if target_label.lower().startswith("en"):
        prompt_text = _read_prompt("Translate to English")
    else:
        prompt_text = _read_prompt("Translate to Target Language", target_label)

    # Prepare user message with clips
    clips_json = json.dumps(clips, ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nClips to translate:\n{clips_json}"

    # Call LLM with retry
    system_message = (
        f"You are a helpful assistant that translates content to {target_label} "
        f"while preserving meaning and tone."
    )
    result = None
    for attempt in range(max_retries + 1):
        try:
            raw_result = call_llm(system_message, user_message, api_key, llm_model)
            if raw_result and isinstance(raw_result, list) and all(isinstance(c, dict) for c in raw_result):
                result = raw_result
                break
            else:
                log("WARN", f"Translation attempt {attempt + 1}/{max_retries + 1}: malformed response, retrying...")
        except Exception as e:
            log("WARN", f"Translation attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")

    # Graceful fallback: if all retries failed, return original clips
    if not result:
        log("WARN", "Translation failed after retries, keeping original clips unchanged")
        return clips

    # Merge the translated fields back into original clips
    # Use (start, end) as key since clips are identified by timing
    result_map = {(round(c.get("start", 0), 2), round(c.get("end", 0), 2)): c for c in result}
    matched_count = 0
    for orig_clip in clips:
        key = (round(orig_clip.get("start", 0), 2), round(orig_clip.get("end", 0), 2))
        if key in result_map:
            trans_clip = result_map[key]
            # Update translatable fields
            for field in ["title", "topic", "caption", "hook", "social_description"]:
                if field in trans_clip:
                    orig_clip[field] = trans_clip[field]
            matched_count += 1
    log("OK", f"Translation: matched {matched_count}/{len(clips)} clips")

    return clips


def _fix_caption_topic_mismatch(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """
    Ensure caption and topic are aligned.
    Rewrites captions that don't match the topic.
    """
    if not clips:
        return clips

    prompt_text = _read_prompt("Fix Mismatched Caption/Topic")

    clips_json = json.dumps(clips, ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nClips:\n{clips_json}"

    system_message = "You are an expert content strategist ensuring social media content consistency."
    result = None
    for attempt in range(max_retries + 1):
        try:
            raw_result = call_llm(system_message, user_message, api_key, llm_model)
            if raw_result and isinstance(raw_result, list) and all(isinstance(c, dict) for c in raw_result):
                result = raw_result
                break
            else:
                log("WARN", f"Caption-topic fix attempt {attempt + 1}/{max_retries + 1}: malformed response, retrying...")
        except Exception as e:
            log("WARN", f"Caption-topic fix attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")

    # Graceful fallback: if all retries failed, return original clips
    if not result:
        log("WARN", "Caption-topic fix failed after retries, keeping original clips unchanged")
        return clips

    # Use (start, end) as key for matching clips
    result_map = {(round(c.get("start", 0), 2), round(c.get("end", 0), 2)): c for c in result}
    matched_count = 0
    for orig_clip in clips:
        key = (round(orig_clip.get("start", 0), 2), round(orig_clip.get("end", 0), 2))
        if key in result_map:
            fixed_clip = result_map[key]
            # Update caption/topic fields
            for field in ["caption", "topic"]:
                if field in fixed_clip:
                    orig_clip[field] = fixed_clip[field]
            matched_count += 1
    log("OK", f"Caption-topic fix: matched {matched_count}/{len(clips)} clips")

    return clips


def _improve_and_deduplicate(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """
    Improve titles, topics, captions, hooks.
    Deduplicate clips with overlapping topics — keep highest-scoring per topic.
    Filter out low-quality/filler clips.
    """
    if not clips:
        return clips

    prompt_text = _read_prompt("Improve and Deduplicate Clips")

    clips_json = json.dumps(clips, ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nClips:\n{clips_json}"

    system_message = "You are an expert social media strategist. Optimize strictly for virality first (shares/comments/replays), not completeness or neutral informativeness."
    result = None
    for attempt in range(max_retries + 1):
        try:
            raw_result = call_llm(system_message, user_message, api_key, llm_model)
            if raw_result and isinstance(raw_result, list) and all(isinstance(c, dict) for c in raw_result):
                result = raw_result
                break
            else:
                log("WARN", f"Improve-deduplicate attempt {attempt + 1}/{max_retries + 1}: malformed response, retrying...")
        except Exception as e:
            log("WARN", f"Improve-deduplicate attempt {attempt + 1}/{max_retries + 1}: {e}, retrying...")

    # Graceful fallback: if all retries failed, return original clips
    if not result:
        log("WARN", "Improve-deduplicate failed after retries, keeping original clips unchanged")
        return clips

    # Merge improved clips back with original metadata
    # Create map of original clips by timing
    orig_map = {(round(c.get("start", 0), 2), round(c.get("end", 0), 2)): c for c in clips}

    # Process LLM results: preserve metadata from originals, use improvements from LLM
    deduplicated = []
    for improved_clip in result:
        key = (round(improved_clip.get("start", 0), 2), round(improved_clip.get("end", 0), 2))
        if key in orig_map:
            orig = orig_map[key]
            # Merge: keep all original fields, update with improved content fields
            merged = orig.copy()
            for field in ["title", "topic", "caption", "hook", "social_description"]:
                if field in improved_clip:
                    merged[field] = improved_clip[field]
            deduplicated.append(merged)
        else:
            # If not found in originals, use as-is (LLM may have created new clips)
            deduplicated.append(improved_clip)

    # Renumber ranks sequentially after deduplication
    # (LLM dedup may remove clips, leaving gaps in original numbering)
    for i, c in enumerate(deduplicated, 1):
        c["rank"] = i

    return deduplicated


def _redistribute_with_gaps(
    words: list[dict],
    trans_texts: list[str],
    phrase_start: float,
    phrase_end: float,
) -> list[dict]:
    """Map translated words onto the *real* speech gaps of a source phrase.

    The naive redistribution (``phrase_duration / N`` even slices) smears the
    spoken audio across the natural pauses between words: a translated word that
    was actually said after a 0.8s silence shows up ~0.24s early, so the burnt
    caption visibly drifts ahead of the speaker and the speech looks "gap-free".
    This keeps the source's inter-word gaps intact and only redistributes the
    *talking* time, so subtitle timing tracks the speech.

    Args:
        words: source phrase words ``[{word, start, end}]`` (clip-relative).
        trans_texts: already-split translated word strings.
        phrase_start / phrase_end: clip-relative phrase bounds (== words[0].start
            and words[-1].end after translation grouping).

    Returns translated-word dicts with ``start``/``end`` aligned to the source's
    talk/gap structure (clip-relative, preserving inter-word silences).
    """
    n_src = len(words)
    n_dst = len(trans_texts)
    if n_src == 0 or n_dst == 0:
        return []

    src_starts = [float(w["start"]) for w in words]
    src_ends = [float(w["end"]) for w in words]
    talk = [src_ends[i] - src_starts[i] for i in range(n_src)]
    # Gap that FOLLOWS source word i (between word i and i+1).
    gaps = [src_starts[i + 1] - src_ends[i] for i in range(n_src - 1)]

    # 1:1 word count — reuse the exact spoken timing; gaps preserved perfectly.
    if n_dst == n_src:
        return [
            {"word": trans_texts[j], "start": src_starts[j], "end": src_ends[j]}
            for j in range(n_dst)
        ]

    total_talk = sum(talk)
    total_gap = sum(gaps)

    # Talk time shared evenly across translated words.  When the translator
    # merges/splits words there is no word→word mapping, so an even share of
    # the *talk* span is the safe default (and correct when words were merely
    # collapsed/expanded).  The inter-word silences below are preserved exactly.
    talk_per = total_talk / n_dst

    # Spread the source gaps across the (n_dst - 1) boundaries *between*
    # translated words, in source order so the relative pauses survive.
    boundary_gap: list[float] = [0.0] * max(1, n_dst - 1)
    n_src_gaps = len(gaps)
    for k, g in enumerate(gaps):
        if n_dst - 1 >= n_src_gaps:
            b = k
        elif n_src_gaps > 1:
            b = min(n_dst - 2, int(round(k * (n_dst - 2) / (n_src_gaps - 1))))
        else:
            b = 0
        boundary_gap[b] += g

    out: list[dict] = []
    cursor = phrase_start
    for j in range(n_dst):
        s = cursor
        e = cursor + talk_per
        out.append({"word": trans_texts[j], "start": s, "end": e})
        cursor = e
        if j < n_dst - 1:
            # Don't overrun the phrase window on rounding drift.
            remaining = phrase_end - cursor
            if remaining > 0:
                cursor += min(boundary_gap[j], remaining)
    return out


def translate_subtitle_words(
    words: list[dict],
    llm_model: str | None = None,
    api_key: str | None = None,
    max_words_per_group: int = 5,
    fix_errors: bool = True,
    target_language: str | None = None,
    phrases: list[dict] | None = None,
    return_pid: bool = False,
) -> list[dict]:
    """
    Translate word-level subtitle entries to the target language, optionally
    fixing transcription errors.

    Groups ``words`` into short phrases, asks the LLM to translate (and optionally fix) each
    phrase, then redistributes the original timestamps proportionally
    across the translated words.

    Args:
        words: List of ``{"word": str, "start": float, "end": float}``
               already offset to clip-start (0-based), as returned by
               ``get_clip_words``.
        llm_model: LLM model override.
        api_key: API key override.
        max_words_per_group: Max source words per subtitle phrase group.
        fix_errors: If True, use LLM to fix Whisper transcription errors + translate.
                   If False, only translate existing text.
        target_language: "en" (default), "id", "auto" (keep original, fix only),
                   or any open language label/code. Controls the subtitle language.
        phrases: Optional pre-built phrase list ``[{"id", "text", "start", "end"}]``.
                 When supplied (e.g. by a caller merging multiple clips), the
                 grouping step is skipped so all phrases are translated in ONE
                 LLM call. The ``id`` values are trusted as the redistribution key.
        return_pid: If True, keep the transient ``_pid`` (source phrase id) tag on
                    each returned word. Used by ``batch_translate_subtitles`` to
                    remap translated words back to their clips. Direct callers
                    (single_video_runner, process_single) must leave this False so
                    the tag is stripped before ASS rendering.

    Returns:
        Translated word list with the same ``{"word", "start", "end"}`` shape.
        When ``return_pid`` is True, each word additionally carries ``_pid``.
    """
    if not words and not phrases:
        return []

    target = _normalize_target_language(target_language)
    # "auto" => keep the transcribed language: fix errors but do not translate.
    keep_original = target == "auto"
    label = "English" if target == "en" else _target_label(target)

    # ── 1. Group words into short phrases ──────────────────────────────────
    # Allow callers (web_runner, cli) to pass a pre-built/merged phrase list so
    # multiple clips can be translated in ONE LLM call (big speedup). When the
    # caller supplies `phrases`, we trust it and skip regrouping.
    if phrases:
        groups = []
        for p in phrases:
            # Reconstruct groups from caller-supplied phrases for downstream
            # timestamp redistribution. Preserve the real phrase id so the
            # id→translated_text map (keyed on id) lines up after the LLM call.
            groups.append([
                {
                    "id": p.get("id"),
                    "word": p.get("text", ""),
                    "start": p["start"],
                    "end": p["end"],
                }
            ])
        # `phrases` is already the final list; keep it as-is below.
    else:
        groups: list[list[dict]] = []
        current: list[dict] = []
        for w in words:
            current.append(w)
            if len(current) >= max_words_per_group:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        # ── 2. Build phrase list for LLM ────────────────────────────────────────
        phrases = [
            {
                "id": i,
                "text": " ".join(w["word"] for w in grp),
                "start": grp[0]["start"],
                "end": grp[-1]["end"],
            }
            for i, grp in enumerate(groups)
        ]

    # Choose prompt based on whether we're fixing errors, translating, or both
    if keep_original:
        # Keep original language: only fix Whisper errors + punctuation
        prompt_text = _read_prompt("Fix Subtitle Phrases (No Translate)")
        system_message = (
            "You are a professional subtitle transcription editor. "
            "Fix Whisper transcription errors and add natural punctuation, "
            "but preserve the original language."
        )
    elif fix_errors:
        if label.lower().startswith("en"):
            prompt_text = _read_prompt("Fix and Translate Subtitle Phrases")
        else:
            prompt_text = _read_prompt("Translate and Fix Subtitle Phrases", label)
        system_message = (
            f"You are a professional subtitle translator and transcription editor. "
            f"Fix Whisper transcription errors and translate spoken content to natural {label}."
        )
    else:
        if label.lower().startswith("en"):
            prompt_text = _read_prompt("Translate Subtitle Phrases")
        else:
            prompt_text = _read_prompt("Translate Subtitle Phrases to Target Language", label)
        system_message = (
            f"You are a professional subtitle translator. "
            f"Translate spoken content to natural {label} accurately."
        )

    # ── 2b. Call the LLM in chunks ──────────────────────────────────────────
    # A single merged call for long videos emits 600+ phrase objects; the
    # model's JSON output then hits the ``max_tokens`` ceiling and the tail
    # phrases (highest ids) get truncated. Capping phrases-per-call keeps each
    # response under the model's real output ceiling. The per-phrase ``id`` is
    # preserved across chunks so the redistribution map below still lines up.
    #
    # CALIBRATION: the observed free-tier model (laguna-xs) truncated a 127-
    # phrase payload. 200 was far too high. 80 leaves comfortable headroom for
    # ~35 tokens/phrase worst case AND for the retries below to finish in 1-2
    # passes even when the model is being terse.
    MAX_PHRASES_PER_CALL = 80

    id_to_text: dict[int, str] = {}

    def _process_batch(batch: list[dict], batch_label: str) -> tuple[dict[int, str], set[int]]:
        """Translate one batch of phrases. Returns (id->text map, uncovered ids).

        On truncation we retry ONLY the still-uncovered phrases, looping until
        everything is covered or the batch shrinks below a floor we give up on.
        Crucially we never re-send a payload that still exceeds what the model
        can emit: each retry drops the phrases it *did* get, so an oversized
        tail monotonically shrinks and eventually fits (capping attempts avoids
        an infinite loop on a genuinely broken model)."""
        phrases_json = json.dumps(batch, ensure_ascii=False, indent=2)
        user_message = f"{prompt_text}\n\nPhrases to translate:\n{phrases_json}"

        cmap: dict[int, str] = {}
        pending = list(batch)
        attempts = 0
        MAX_ATTEMPTS = 8  # generous; normally 1-2 loops resolve truncation
        while pending and attempts < MAX_ATTEMPTS:
            attempts += 1
            result = None
            try:
                raw_result = call_llm(system_message, user_message, api_key, llm_model)
                if raw_result and isinstance(raw_result, list) and all(isinstance(p, dict) for p in raw_result):
                    result = raw_result
                else:
                    log("WARN", f"Subtitle translation (batch {batch_label}): malformed response, retrying...")
            except Exception as e:
                log("WARN", f"Subtitle translation (batch {batch_label}): {e}, retrying...")

            if result:
                for item in result:
                    pid = item.get("id")
                    text = item.get("text", "").strip()
                    if pid is None or not text:
                        continue
                    unk_ratio = text.count("<unk>") / max(1, len(text.split()))
                    if unk_ratio > 0.2 or len(text) < 2:
                        log("WARN", f"Subtitle phrase id={pid} returned degenerate output "
                                     f"(unk_ratio={unk_ratio:.2f}); keeping original words")
                        continue
                    cmap[int(pid)] = text

            requested = {int(p["id"]) for p in pending}
            covered = set(cmap.keys())
            uncovered = requested - covered
            if not uncovered:
                break

            # Drop the phrases we already got and re-request the rest. Because
            # the pending set shrinks each pass, an over-long tail eventually
            # fits the model's output window.
            pending = [p for p in pending if int(p["id"]) in uncovered]
            phrases_json = json.dumps(pending, ensure_ascii=False, indent=2)
            user_message = f"{prompt_text}\n\nPhrases to translate:\n{phrases_json}"
            log("WARN", f"Subtitle batch {batch_label}: {len(uncovered)} phrase(s) "
                         f"truncated (id {sorted(uncovered)[:5]}{'…' if len(uncovered) > 5 else ''}); "
                         f"retrying {len(pending)} remaining (attempt {attempts})…")

        if pending:
            # Genuinely failed after retries — per the no-fallback rule, leave
            # these unmapped so they fall through to "keep original" and get
            # logged individually below (one warning per phrase), not silently.
            ids_left = {int(p["id"]) for p in pending}
            log("WARN", f"Subtitle batch {batch_label}: {len(ids_left)} phrase(s) "
                         f"still untranslated after {attempts} attempts; keeping original text")
        return cmap, {int(p["id"]) for p in pending}

    n_chunks = (len(phrases) + MAX_PHRASES_PER_CALL - 1) // MAX_PHRASES_PER_CALL
    for c_start in range(0, len(phrases), MAX_PHRASES_PER_CALL):
        chunk = phrases[c_start:c_start + MAX_PHRASES_PER_CALL]
        chunk_idx = c_start // MAX_PHRASES_PER_CALL + 1
        cmap, _ = _process_batch(chunk, f"{chunk_idx}/{n_chunks}")
        id_to_text.update(cmap)

    # ── 3. Redistribute timestamps for translated words ──────────────────────
    translated_words: list[dict] = []
    for i, grp in enumerate(groups):
        # Read the phrase id from the parallel `phrases` list (aligned by
        # index), NOT from the group's first word. Word dicts in the single-
        # clip path carry no "id", so grp[0].get("id") would be None and every
        # phrase would fall through to the "keep original" fallback, silently
        # discarding the translation.
        pid = phrases[i].get("id")
        phrase_start = grp[0]["start"]
        phrase_end = grp[-1]["end"]

        translated_text = id_to_text.get(pid) if pid is not None else None
        if not translated_text:
            log("WARN", f"Subtitle translation missing for phrase id={pid}, keeping original")
            # Fallback: keep original words for this phrase
            for w in grp:
                translated_words.append(w.copy())
            continue

        trans_words = translated_text.split()
        if not trans_words:
            log("WARN", f"Subtitle translation produced empty phrase for id={i}, keeping original")
            for w in grp:
                translated_words.append(w.copy())
            continue

        # Redistribute onto the *real* inter-word gaps instead of a flat even
        # slice.  This keeps the natural pauses between words so the burnt
        # caption tracks the speech instead of smearing it across silences
        # (the "subtitles run ahead of the speaker" symptom).
        for twd in _redistribute_with_gaps(grp, trans_words, phrase_start, phrase_end):
            translated_words.append({
                "word": twd["word"],
                "start": twd["start"],
                "end": twd["end"],
                "_pid": pid,  # transient: marks source phrase for batch remap
            })

    log("OK", f"Subtitle {'fix+translation' if fix_errors else 'translation'}: {len(words)} original words → {len(translated_words)} translated words")
    # Strip the transient _pid tag (used only for batch remapping) so direct
    # callers (single_video_runner, process_single) don't emit it into ASS.
    # When return_pid is True (batch mode) we keep it so the caller can remap
    # words back to their source clip, then strip it itself.
    if not return_pid:
        for w in translated_words:
            w.pop("_pid", None)
    return translated_words


def batch_translate_subtitles(
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    fix_errors: bool = True,
    target_language: str | None = None,
    max_words_per_group: int = 5,
) -> dict[int, list[dict]]:
    """
    Translate subtitle words for ALL clips in a single merged LLM call.

    Replaces the previous one-``translate_subtitle_words``-call-per-clip loop
    (N round-trips — the dominant cost on long videos, ~25 min in the run
    logs). Every clip's raw words are gathered, each phrase gets a globally
    unique id, and they are all translated at once. The returned flat word
    list carries a transient ``_pid`` per word; we remap by pid→clip and strip
    the tag.

    Returns a dict mapping ``clip['rank'] -> translated word list``
    (``[]`` for clips with no words). On total failure every clip falls back
    to its original raw words rather than emitting garbage.
    """
    from ..subtitles import get_clip_words

    if not clips:
        return {}

    target = _normalize_target_language(target_language)

    merged_phrases: list[dict] = []
    clip_phrase_ids: dict[int, list[int]] = {}
    for clip in clips:
        rank = int(clip.get("rank", 0))
        clip_phrase_ids[rank] = []
        raw_words = get_clip_words(
            segments, clip_start=clip["start"], clip_end=clip["end"]
        )
        if not raw_words:
            continue
        current: list[dict] = []
        for w in raw_words:
            current.append(w)
            if len(current) >= max_words_per_group:
                pid = len(merged_phrases)
                merged_phrases.append({
                    "id": pid,
                    "text": " ".join(x["word"] for x in current),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                })
                clip_phrase_ids[rank].append(pid)
                current = []
        if current:
            pid = len(merged_phrases)
            merged_phrases.append({
                "id": pid,
                "text": " ".join(x["word"] for x in current),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            })
            clip_phrase_ids[rank].append(pid)

    if not merged_phrases:
        return {int(c.get("rank", 0)): [] for c in clips}

    log("INFO", f"Batch-translating subtitles for {len(clips)} clips "
                f"({len(merged_phrases)} phrases)")
    # NOTE: chunking is handled inside ``translate_subtitle_words`` so both the
    # batch path (here) and the single-video path (direct caller) are covered.

    translated = translate_subtitle_words(
        [],  # words unused when phrases supplied
        llm_model=llm_model,
        api_key=api_key,
        fix_errors=fix_errors,
        target_language=target_language,
        phrases=merged_phrases,
        return_pid=True,  # keep _pid so we can remap words back to clips
    )

    # Group translated words by their source phrase id.
    words_by_pid: dict[int, list[dict]] = {}
    for w in translated:
        pid = w.pop("_pid", None)  # strip transient tag
        if pid is not None:
            words_by_pid.setdefault(pid, []).append(w)

    # Fallback: if the whole batch produced nothing, keep original raw words.
    if not words_by_pid:
        log("WARN", "Batch subtitle translation returned nothing; "
                    "falling back to original words for all clips")
        out: dict[int, list[dict]] = {}
        for clip in clips:
            rank = int(clip.get("rank", 0))
            out[rank] = get_clip_words(
                segments, clip_start=clip["start"], clip_end=clip["end"]
            )
        return out

    result: dict[int, list[dict]] = {}
    for clip in clips:
        rank = int(clip.get("rank", 0))
        clip_words: list[dict] = []
        for pid in clip_phrase_ids.get(rank, []):
            clip_words.extend(words_by_pid.get(pid, []))
        result[rank] = clip_words
    return result


def _read_prompt(section_name: str, target_language: str = "English") -> str:
    """
    Get a prompt from prompts.py.

    section_name: one of "Translate to English", "Translate to Target Language",
        "Fix Mismatched Caption/Topic", "Improve and Deduplicate Clips",
        "Translate Subtitle Phrases", "Translate Subtitle Phrases to Target Language",
        "Translate and Fix Subtitle Phrases", "Fix and Translate Subtitle Phrases",
        "Fix Subtitle Phrases (No Translate)".
    target_language: human-readable label used to fill the ``{TARGET_LANGUAGE}``
        token in parameterized prompts (default "English").
    """
    from .prompts import render_prompt
    return render_prompt(section_name, target_language)


def _build_transcript_for_metadata(segments: list[dict[str, Any]]) -> str:
    """Serialize transcript segments compactly for metadata generation."""
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", 0.0) or 0.0)
        lines.append(f"[{start:.0f}-{end:.0f}] {text}")
    return "\n".join(lines)
