"""
LLM prompts for clip processing: translation, caption fixing, and deduplication.
"""

PROMPTS = {
    "Generate Single Clip Metadata": """Given a transcript and a single clip JSON object that covers the full video, rewrite only the metadata fields so they accurately reflect the spoken content.

Tasks:
1. Read the transcript carefully and identify the central claim, tension, or insight.
2. Rewrite these fields only: reason, title, topic, caption, comment_bait, hook, closing_line.
3. Keep all timing, score, rank, and filename fields unchanged.
4. Return the same JSON array with exactly one item.

Field requirements (in this order):
- reason: 1-2 sentences explaining why this full-video clip is compelling or worth watching
- title: Max 8 words. No jargon. Non-technical audience. Translate technical terms to plain English (e.g., "large language model" → "AI brain", "fine-tuning" → "AI training"). Use viral formula (Number+Outcome, Expose Lie, Secret, Comparison, Personal Stakes, or How+Wonder).
- topic: one clear sentence describing the core idea or debate
- caption: 4-part structure (hook line → insight → CTA → hashtags). Total max 280 chars.
- comment_bait: Single opinion/experience question to drive comments. Under 15 words. NOT a knowledge quiz. Write in casual Indonesian.
- hook: the strongest opening line or a tightened version that stays faithful to the transcript
- closing_line: The EXACT last words from the transcript — word-for-word. Must be a strong ending.

Language rules:
- Write all fields (reason, title, topic, caption, hook, closing_line) in English
- Write comment_bait in casual Indonesian
- If the transcript is in a different language, translate concepts to English while maintaining meaning
- Do not use the filename or generic placeholders as metadata.
- Do not invent facts not supported by the transcript.

Return ONLY valid JSON array, no other text.""",

    "Translate to English": """Given clips data in JSON format, perform the following:

1. For each clip, analyze the language of the fields: reason, topic, caption, hook, closing_line
2. If any of these fields are NOT in English, translate them to English
3. If they are already in English, keep them unchanged
4. Preserve the meaning, tone, and marketing appeal during translation
5. Keep all other fields unchanged (title, comment_bait, scores, timing, metadata)
6. Return the same JSON array with only the translated/updated fields

IMPORTANT: Do NOT translate the "title" or "comment_bait" fields — keep them exactly as-is.

Example input:
[{"title": "How to cook rice", "topic": "Tips Memasak", "caption": "Pelajari dengan cepat!", "rank": 1, ...}]

Example output:
[{"title": "How to cook rice", "topic": "Cooking Tips", "caption": "Learn fast!", "rank": 1, ...}]

Return ONLY valid JSON array, no other text.""",

    "Fix Mismatched Caption/Topic": """Given clips data in JSON format, perform the following:

1. For each clip, verify that the caption and topic are cohesive and aligned
2. If the caption does NOT match the topic context, rewrite the caption to align with the topic
3. Ensure the caption is engaging, concise (max 280 chars), and relevant to the topic
4. Keep all other fields unchanged
5. Requirements:
   - Caption must be catchy and TikTok/Instagram-friendly
   - Topic must be clear and specific (not generic)
   - Both should tell a coherent story

Example:
Input: {"topic": "Efficient Meeting Strategies", "caption": "How to cook pasta perfectly", ...}
Output: {"topic": "Efficient Meeting Strategies", "caption": "5 ways to run meetings that don't waste time", ...}

Return ONLY valid JSON array with fixed clips, no other text.""",

    "Improve and Deduplicate Clips": """Given clips data in JSON format, perform the following:

1. Improve each clip's content:
   - Do NOT change the "title" field — keep it exactly as-is
   - For captions: verify 4-part structure (hook → insight → CTA → hashtags) is intact. If flat, restructure.
   - Verify comment_bait is a genuine opinion/experience question, not a knowledge quiz.
   - Enhance captions with power words that drive engagement
   - Ensure topics are clear categories or themes
   - Strengthen hooks for maximum stop-scroll potential

2. Deduplicate only near-identical clips:
   - Only remove a clip if another clip covers the EXACT same moment or statement (not just the same broad theme)
   - Different clips covering similar topics from different angles are NOT duplicates — keep all of them
   - When forced to choose between near-identical clips, keep the one with the higher clip_score
   - Be conservative: when in doubt, keep the clip

3. Filter out only truly dead-weight clips:
   - Remove ONLY: pure opening/closing remarks with zero standalone value (e.g. "Welcome everyone, let's get started")
   - Remove ONLY: clips with clip_score < 40
   - Keep everything else — including Q&A, opinions, stories, tangents

4. Re-rank remaining clips by clip_score (highest first)

Return ONLY a valid JSON array of the kept and improved clips, no other text.""",
    "Translate Subtitle Phrases": """Given subtitle phrases in JSON format, translate each phrase to English AND add proper punctuation.

Input format: [{"id": 0, "text": "some phrase", "start": 0.5, "end": 2.0}, ...]

Rules:
- Translate the "text" field to English. If text is already in English, keep it unchanged.
- Add natural punctuation: periods (.), commas (,), question marks (?), exclamation marks (!) as appropriate for spoken content.
- Keep the word count roughly the same — punctuation marks are fine but do NOT add or remove words.
- Keep "id", "start", "end" fields exactly unchanged.
- Maintain natural conversational English.
- Keep the exact same number of items in the output array.

Example:
Input:  [{"id": 0, "text": "oh itu menarik sekali", "start": 0.0, "end": 1.5}]
Output: [{"id": 0, "text": "Oh, that's really interesting.", "start": 0.0, "end": 1.5}]

Return ONLY a JSON array with the exact same structure.
""",

    "Fix and Translate Subtitle Phrases": """Given subtitle phrases from Whisper transcription in JSON format, FIX transcription errors AND translate to English.

Whisper often makes mistakes like:
- Wrong words due to similar sounds (e.g., "I" heard as "eye", "their" as "there")
- Missing punctuation
- Incorrect proper nouns (names, brands, technical terms)
- Run-on sentences without breaks
- Misheard words (e.g., "gonna" as "going to", "dunno" as "don't know")

Input format: [{"id": 0, "text": "some phrase", "start": 0.5, "end": 2.0}, ...]

Your tasks:
1. FIX any transcription errors based on context
2. Translate the corrected text to English (if not already English)
3. Add natural punctuation: periods (.), commas (,), question marks (?), exclamation marks (!)
4. Keep the meaning and tone natural and conversational

Rules:
- Keep "id", "start", "end" fields exactly unchanged
- Maintain roughly the same word count — you can adjust word boundaries for natural English but don't add/remove entire concepts
- Fix proper nouns, technical terms, and brand names based on context
- Add appropriate punctuation for spoken content
- Keep the exact same number of items in the output array
- If text is already correct English, just add punctuation and return unchanged

Example:
Input:  [{"id": 0, "text": "so basically ai is like really powerful now", "start": 0.0, "end": 2.5}]
Output: [{"id": 0, "text": "So basically, AI is like really powerful now.", "start": 0.0, "end": 2.5}]

Return ONLY a JSON array with the exact same structure.
""",}


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
