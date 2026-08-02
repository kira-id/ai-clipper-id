"""
LLM prompts for clip processing: translation, caption fixing, and deduplication.

Translation targets are templated with ``{TARGET_LANGUAGE}`` (a human-readable
language label such as "English" or "Indonesian"). We deliberately use string
``.replace()`` rather than ``str.format()`` to fill that token, because several
prompts contain raw JSON ``{...}`` braces that would otherwise be misinterpreted
by ``str.format()``.
"""

PROMPTS = {
    "Generate Single Clip Metadata": """Rewrite metadata for a single full-video clip using the transcript. Maximize viral potential—choose high-engagement wording over bland accuracy.

Rewrite ONLY: reason, title, topic, caption, comment_bait, hook, closing_line, social_description. Keep all timing/score/rank/filename fields unchanged.

Field specs:
- reason: 1-2 sentences on why this clip is compelling
- title: Max 8 words. Friendly, warm and easy to understand even for a child. A gentle, curious hook — NEVER name it "Clip 1", "Clip 2", or any "Clip N". Use simple inviting words (e.g. "Why this little trick changes everything", "The silly mistake we all make"). No clickbait rage.
- topic: One sentence, core idea with emotional/curiosity angle
- caption: hook → insight → CTA → hashtags. Max 280 chars. Friendly and easy to read.
- comment_bait: Casual Indonesian. Opinion/experience question, NOT a quiz. Under 15 words
- hook: Strongest opening line from transcript (tighten if needed, stay faithful)
- closing_line: EXACT last words from transcript, word-for-word
- social_description: Friendly, approachable 2-3 sentence social-media description a child could understand. Starts with a hook, says why the clip is fun or useful, ends with a gentle invite to watch and share. No hashtags here.

Language: All fields in English except comment_bait (Indonesian). Translate if needed. No invented facts.

Return ONLY valid JSON array (one item), no other text.""",

    "Translate to English": """For each clip, translate reason, topic, caption, hook, closing_line, social_description to English if not already. Preserve meaning, tone, and marketing appeal.

Do NOT translate "title" or "comment_bait"—keep as-is. Keep all other fields unchanged.

Return ONLY valid JSON array, no other text.""",

    "Translate to Target Language": """For each clip, translate reason, topic, caption, hook, closing_line, social_description to {TARGET_LANGUAGE} if not already in that language. Preserve meaning, tone, and marketing appeal.

Do NOT translate "title" or "comment_bait"—keep as-is. Keep all other fields unchanged.

Return ONLY valid JSON array, no other text.""",

    "Fix Mismatched Caption/Topic": """For each clip, verify caption and topic are aligned. If caption doesn't match the topic, rewrite caption to fit.

Rules:
- Caption: engaging, TikTok-friendly, max 280 chars, relevant to topic
- Topic: clear and specific
- Both must tell a coherent story
- Keep all other fields unchanged

Return ONLY valid JSON array, no other text.""",

    "Improve and Deduplicate Clips": """Maximize virality per retained clip.

1. Improve content (do NOT change "title"):
   - Captions: enforce hook → insight → CTA → hashtags structure. Add power words
   - Topics: make emotional/curiosity-driven, not generic labels
   - comment_bait: must be opinion/experience question, not quiz
   - Hooks: maximize stop-scroll potential
   - social_description: friendly, approachable 2-3 sentence social-media description a child could understand; hook → why it's fun/useful → gentle invite to watch & share
   - Tone: energetic and viral, not academic

2. Deduplicate conservatively:
   - Remove only if another clip covers the EXACT same moment (not just same theme)
   - Keep the higher clip_score when forced to choose
   - When in doubt, keep it

3. Remove only dead weight:
   - Pure filler ("Welcome everyone, let's get started")
   - clip_score < 40
   - Emotionally flat / zero shareability

4. Re-rank by clip_score descending.

Return ONLY valid JSON array, no other text.""",

    "Refine Clips (No Translate)": """Maximize virality per retained clip. Do NOT translate any text—keep the original language of every field.

1. Improve content (do NOT change "title"):
   - Captions: enforce hook → insight → CTA → hashtags structure. Add power words
   - Topics: make emotional/curiosity-driven, not generic labels
   - comment_bait: must be opinion/experience question, not quiz
   - Hooks: maximize stop-scroll potential
   - social_description: friendly, approachable 2-3 sentence social-media description a child could understand; hook → why it's fun/useful → gentle invite to watch & share
   - Tone: energetic and viral, not academic

2. Fix caption/topic alignment: ensure the caption and topic tell a coherent story. If the caption doesn't match the topic, rewrite the caption to fit.

3. Deduplicate conservatively:
   - Remove only if another clip covers the EXACT same moment (not just same theme)
   - Keep the higher clip_score when forced to choose
   - When in doubt, keep it

4. Remove only dead weight:
   - Pure filler ("Welcome everyone, let's get started")
   - clip_score < 40
   - Emotionally flat / zero shareability

5. Re-rank by clip_score descending.

Return ONLY valid JSON array, no other text.""",

    "Refine Clips (Translate to English)": """Maximize virality per retained clip. Translate reason, topic, caption, hook, closing_line to English if not already (do NOT translate "title" or "comment_bait").

1. Improve content (do NOT change "title"):
   - Captions: enforce hook → insight → CTA → hashtags structure. Add power words
   - Topics: make emotional/curiosity-driven, not generic labels
   - comment_bait: must be opinion/experience question, not quiz
   - Hooks: maximize stop-scroll potential
   - social_description: friendly, approachable 2-3 sentence social-media description a child could understand; hook → why it's fun/useful → gentle invite to watch & share
   - Tone: energetic and viral, not academic

2. Fix caption/topic alignment: ensure the caption and topic tell a coherent story. If the caption doesn't match the topic, rewrite the caption to fit.

3. Deduplicate conservatively:
   - Remove only if another clip covers the EXACT same moment (not just same theme)
   - Keep the higher clip_score when forced to choose
   - When in doubt, keep it

4. Remove only dead weight:
   - Pure filler ("Welcome everyone, let's get started")
   - clip_score < 40
   - Emotionally flat / zero shareability

5. Re-rank by clip_score descending.

Return ONLY valid JSON array, no other text.""",

    "Refine Clips (Translate to Target Language)": """Maximize virality per retained clip. Translate reason, topic, caption, hook, closing_line to {TARGET_LANGUAGE} if not already in that language (do NOT translate "title" or "comment_bait").

1. Improve content (do NOT change "title"):
   - Captions: enforce hook → insight → CTA → hashtags structure. Add power words
   - Topics: make emotional/curiosity-driven, not generic labels
   - comment_bait: must be opinion/experience question, not quiz
   - Hooks: maximize stop-scroll potential
   - social_description: friendly, approachable 2-3 sentence social-media description a child could understand; hook → why it's fun/useful → gentle invite to watch & share
   - Tone: energetic and viral, not academic

2. Fix caption/topic alignment: ensure the caption and topic tell a coherent story. If the caption doesn't match the topic, rewrite the caption to fit.

3. Deduplicate conservatively:
   - Remove only if another clip covers the EXACT same moment (not just same theme)
   - Keep the higher clip_score when forced to choose
   - When in doubt, keep it

4. Remove only dead weight:
   - Pure filler ("Welcome everyone, let's get started")
   - clip_score < 40
   - Emotionally flat / zero shareability

5. Re-rank by clip_score descending.

Return ONLY valid JSON array, no other text.""",

    "Translate Subtitle Phrases": """Translate each subtitle's "text" to English (keep unchanged if already English) and add natural punctuation.

Input: [{"id": 0, "text": "...", "start": 0.5, "end": 2.0}, ...]

Rules:
- Translate "text" only. Keep id/start/end unchanged
- Add periods, commas, question/exclamation marks naturally
- Don't add or remove words—punctuation only
- Same item count in output

Return ONLY valid JSON array.""",

    "Translate Subtitle Phrases to Target Language": """Translate each subtitle's "text" to {TARGET_LANGUAGE} (keep unchanged if already in that language) and add natural punctuation.

Input: [{"id": 0, "text": "...", "start": 0.5, "end": 2.0}, ...]

Rules:
- Translate "text" only. Keep id/start/end unchanged
- Add periods, commas, question/exclamation marks naturally (use {TARGET_LANGUAGE} punctuation conventions)
- Don't add or remove words—punctuation only
- Same item count in output

Return ONLY valid JSON array.""",

    "Translate and Fix Subtitle Phrases": """Fix Whisper transcription errors, translate to {TARGET_LANGUAGE}, and add punctuation.

Common Whisper errors: wrong words from similar sounds, missing punctuation, botched proper nouns/brands, run-on sentences, misheard words.

Input: [{"id": 0, "text": "...", "start": 0.5, "end": 2.0}, ...]

Tasks:
1. Fix transcription errors using context
2. Translate to {TARGET_LANGUAGE} (if needed)
3. Add natural punctuation (use {TARGET_LANGUAGE} punctuation conventions)

Rules:
- Keep id/start/end unchanged, same item count
- Roughly same word count—don't add/remove concepts
- Fix proper nouns, brands, technical terms
- Keep conversational tone

Return ONLY valid JSON array.""",

    "Fix and Translate Subtitle Phrases": """Fix Whisper transcription errors, translate to English, and add punctuation.

Common Whisper errors: wrong words from similar sounds, missing punctuation, botched proper nouns/brands, run-on sentences, misheard words.

Input: [{"id": 0, "text": "...", "start": 0.5, "end": 2.0}, ...]

Tasks:
1. Fix transcription errors using context
2. Translate to English (if needed)
3. Add natural punctuation

Rules:
- Keep id/start/end unchanged, same item count
- Roughly same word count—don't add/remove concepts
- Fix proper nouns, brands, technical terms
- Keep conversational tone

Return ONLY valid JSON array.""",

    "Fix Subtitle Phrases (No Translate)": """Fix Whisper transcription errors in each subtitle's "text" and add natural punctuation. Do NOT translate the language.

Common Whisper errors: wrong words from similar sounds, missing punctuation, botched proper nouns/brands, run-on sentences, misheard words.

Input: [{"id": 0, "text": "...", "start": 0.5, "end": 2.0}, ...]

Tasks:
1. Fix transcription errors using context
2. Add natural punctuation

Rules:
- Keep id/start/end unchanged, same item count
- Roughly same word count—don't add/remove concepts
- Fix proper nouns, brands, technical terms
- Keep conversational tone
- Preserve the original language of "text"

Return ONLY valid JSON array.""",
}


def render_prompt(section_name: str, target_language: str = "English") -> str:
    """Get a prompt, filling the ``{TARGET_LANGUAGE}`` token if present.

    Uses ``str.replace`` (not ``str.format``) so existing JSON ``{...}`` braces
    in prompts are left untouched.
    """
    prompt = PROMPTS.get(section_name, "")
    if not prompt:
        from ..utils import log
        log("WARN", f"Prompt '{section_name}' not found. Available: {list(PROMPTS.keys())}")
        return ""
    return prompt.replace("{TARGET_LANGUAGE}", target_language)


def get_prompt(section_name: str) -> str:
    """
    Get a prompt by section name (legacy, un-parameterized lookup).

    Args:
        section_name: One of the keys in PROMPTS.

    Returns:
        The prompt text, or empty string if not found.
    """
    return render_prompt(section_name)
