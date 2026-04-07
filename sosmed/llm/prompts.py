"""
LLM prompts for clip processing: translation, caption fixing, and deduplication.
"""

PROMPTS = {
    "Generate Single Clip Metadata": """Rewrite metadata for a single full-video clip using the transcript. Maximize viral potential—choose high-engagement wording over bland accuracy.

Rewrite ONLY: reason, title, topic, caption, comment_bait, hook, closing_line. Keep all timing/score/rank/filename fields unchanged.

Field specs:
- reason: 1-2 sentences on why this clip is compelling
- title: Max 8 words, Title Case, emotional. Use viral formats (Comparison, How-To, Receh/Brainrot, Secret, Personal Stakes, Contrarian)
- topic: One sentence, core idea with emotional/curiosity angle
- caption: hook → insight → CTA → hashtags. Max 280 chars
- comment_bait: Casual Indonesian. Opinion/experience question, NOT a quiz. Under 15 words
- hook: Strongest opening line from transcript (tighten if needed, stay faithful)
- closing_line: EXACT last words from transcript, word-for-word

Language: All fields in English except comment_bait (Indonesian). Translate if needed. No invented facts.

Return ONLY valid JSON array (one item), no other text.""",

    "Translate to English": """For each clip, translate reason, topic, caption, hook, closing_line to English if not already. Preserve meaning, tone, and marketing appeal.

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

    "Translate Subtitle Phrases": """Translate each subtitle's "text" to English (keep unchanged if already English) and add natural punctuation.

Input: [{"id": 0, "text": "...", "start": 0.5, "end": 2.0}, ...]

Rules:
- Translate "text" only. Keep id/start/end unchanged
- Add periods, commas, question/exclamation marks naturally
- Don't add or remove words—punctuation only
- Same item count in output

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
}


def get_prompt(section_name: str) -> str:
    """
    Get a prompt by section name.

    Args:
        section_name: One of:
            - "Generate Single Clip Metadata"
            - "Translate to English"
            - "Fix Mismatched Caption/Topic"
            - "Improve and Deduplicate Clips"
            - "Translate Subtitle Phrases"
            - "Fix and Translate Subtitle Phrases"

    Returns:
        The prompt text, or empty string if not found
    """
    prompt = PROMPTS.get(section_name, "")
    if not prompt:
        from ..utils import log
        log("WARN", f"Prompt '{section_name}' not found. Available: {list(PROMPTS.keys())}")
    return prompt
