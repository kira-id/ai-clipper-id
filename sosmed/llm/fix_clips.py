"""
Fix and improve clips: translate to Indonesian, fix caption-topic mismatches, deduplicate topics.
"""

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

    for field in ["title", "topic", "caption", "reason", "hook", "closing_line", "comment_bait"]:
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


def fix_and_improve_clips(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    detected_language: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Post-process clips to:
    1. Translate to Indonesian if needed
    2. Fix mismatched caption/topic pairs
    3. Improve titles/hooks/captions and deduplicate similar topics

    This runs AFTER find_clips() and BEFORE subtitle generation.
    
    Args:
        detected_language: dict with keys "language" and "language_probability" from Whisper
    """
    if not clips:
        log("WARN", "No clips to fix")
        return clips

    log("INFO", f"Starting clip fixing pipeline: {len(clips)} clips")

    # Step 1: Translate to Indonesian (skip if already Indonesian)
    log("INFO", "Step 1/3: Translating to Indonesian...")
    clips = _translate_to_indonesian(clips, llm_model, api_key, detected_language)
    log("OK", f"After translation: {len(clips)} clips")

    # Step 2: Fix mismatched caption/topic
    log("INFO", "Step 2/3: Fixing mismatched caption/topic pairs...")
    clips = _fix_caption_topic_mismatch(clips, llm_model, api_key)
    log("OK", f"After fixing mismatches: {len(clips)} clips")

    # Step 3: Improve and deduplicate
    log("INFO", "Step 3/3: Improving content and deduplicating topics...")
    clips = _improve_and_deduplicate(clips, llm_model, api_key)
    log("OK", f"After improvement and deduplication: {len(clips)} clips")

    # Re-rank by score
    clips.sort(key=lambda x: (-x.get("clip_score", 0), x.get("rank", 999)))
    for i, c in enumerate(clips, 1):
        c["rank"] = i

    return clips


def _translate_to_indonesian(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
    detected_language: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Translate title, topic, caption, hook to Indonesian if not already Indonesian.
    Skips translation if Whisper detected Indonesian with >60% confidence.
    """
    if not clips:
        return clips
    
    # Skip translation if already Indonesian (based on Whisper detection)
    if detected_language:
        lang = detected_language.get("language", "").lower()
        prob = detected_language.get("language_probability", 0.0)
        if lang == "id" and prob > 0.6:
            log("OK", f"Language detected as Indonesian (p={prob:.0%}), skipping translation")
            return clips

    # Read the translation prompt from docs/prompts.md
    prompt_text = _read_prompt("Translate to Indonesian")

    # Prepare user message with clips
    clips_json = json.dumps(clips, ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nClips to translate:\n{clips_json}"

    # Call LLM
    system_message = "You are a helpful assistant that translates content to Indonesian while preserving meaning and tone."
    result = call_llm(system_message, user_message, api_key, llm_model)

    # Validate and merge results
    if result and isinstance(result, list):
        # If LLM returned structured JSON as expected
        if all(isinstance(c, dict) for c in result):
            # Merge the translated fields back into original clips
            # Use (start, end) as key since clips are identified by timing
            result_map = {(round(c.get("start", 0), 2), round(c.get("end", 0), 2)): c for c in result}
            matched_count = 0
            for orig_clip in clips:
                key = (round(orig_clip.get("start", 0), 2), round(orig_clip.get("end", 0), 2))
                if key in result_map:
                    trans_clip = result_map[key]
                    # Update translatable fields
                    for field in ["title", "topic", "caption", "hook"]:
                        if field in trans_clip:
                            orig_clip[field] = trans_clip[field]
                    matched_count += 1
            log("OK", f"Translation: matched {matched_count}/{len(clips)} clips")
        else:
            log("WARN", "Translation LLM did not return structured list")
    else:
        log("WARN", "Translation LLM returned no valid results")

    return clips


def _fix_caption_topic_mismatch(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
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
    result = call_llm(system_message, user_message, api_key, llm_model)

    if result and isinstance(result, list):
        if all(isinstance(c, dict) for c in result):
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
        else:
            log("WARN", "Caption-topic fix LLM did not return structured list")
    else:
        log("WARN", "Caption-topic fix LLM returned no valid results")

    return clips


def _improve_and_deduplicate(
    clips: list[dict[str, Any]],
    llm_model: str | None = None,
    api_key: str | None = None,
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

    system_message = "You are an expert social media strategist optimizing video clips for viral engagement."
    result = call_llm(system_message, user_message, api_key, llm_model)

    if result and isinstance(result, list):
        if all(isinstance(c, dict) for c in result):
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
                    for field in ["title", "topic", "caption", "hook"]:
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
        else:
            log("WARN", "Improve-deduplicate LLM did not return structured list")
    else:
        log("WARN", "Improve-deduplicate LLM returned no valid results")

    return clips


def translate_subtitle_words(
    words: list[dict],
    llm_model: str | None = None,
    api_key: str | None = None,
    max_words_per_group: int = 5,
    fix_errors: bool = True,
) -> list[dict]:
    """
    Translate word-level subtitle entries to Indonesian, optionally fixing transcription errors.

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

    Returns:
        Translated word list with the same ``{"word", "start", "end"}`` shape.
    """
    if not words:
        return []

    # ── 1. Group words into short phrases ──────────────────────────────────
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

    # Choose prompt based on whether we're fixing errors or just translating
    if fix_errors:
        prompt_text = _read_prompt("Fix and Translate Subtitle Phrases")
        system_message = (
            "You are a professional subtitle translator and transcription editor. "
            "Fix Whisper transcription errors and translate spoken content to natural Indonesian."
        )
    else:
        prompt_text = _read_prompt("Translate Subtitle Phrases")
        system_message = (
            "You are a professional subtitle translator. "
            "Translate spoken Indonesian-language content accurately and naturally."
        )

    phrases_json = json.dumps(phrases, ensure_ascii=False, indent=2)
    user_message = f"{prompt_text}\n\nPhrases to translate:\n{phrases_json}"

    result = call_llm(system_message, user_message, api_key, llm_model)

    if not result or not isinstance(result, list) or not all(isinstance(p, dict) for p in result):
        raise RuntimeError(
            f"Subtitle translation failed: LLM returned {result!r}"
        )

    # Build id→translated_text map
    id_to_text: dict[int, str] = {}
    for item in result:
        pid = item.get("id")
        text = item.get("text", "").strip()
        if pid is not None and text:
            id_to_text[int(pid)] = text

    # ── 3. Redistribute timestamps for translated words ──────────────────────
    translated_words: list[dict] = []
    for i, grp in enumerate(groups):
        phrase_start = grp[0]["start"]
        phrase_end = grp[-1]["end"]
        phrase_duration = phrase_end - phrase_start

        translated_text = id_to_text.get(i)
        if not translated_text:
            raise RuntimeError(
                f"Subtitle translation missing for phrase id={i}: "
                f"'{' '.join(w['word'] for w in grp)}'"
            )

        trans_words = translated_text.split()
        if not trans_words:
            raise RuntimeError(
                f"Subtitle translation produced empty phrase for id={i}"
            )

        word_dur = phrase_duration / len(trans_words)
        for j, tw in enumerate(trans_words):
            translated_words.append({
                "word": tw,
                "start": phrase_start + j * word_dur,
                "end": phrase_start + (j + 1) * word_dur,
            })

    log("OK", f"Subtitle {'fix+translation' if fix_errors else 'translation'}: {len(words)} original words → {len(translated_words)} translated words")
    return translated_words


def _read_prompt(section_name: str) -> str:
    """
    Get a prompt from prompts.py.
    section_name: one of "Translate to Indonesian", "Fix Mismatched Caption/Topic", "Improve and Deduplicate Clips"
    """
    prompt = get_prompt(section_name)
    if not prompt:
        log("ERROR", f"Prompt section '{section_name}' not found in prompts.py")
    return prompt


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
