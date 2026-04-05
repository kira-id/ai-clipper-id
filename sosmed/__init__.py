"""
AI Video Clipper — Indonesian-optimized video clip extraction.

Transcribes video with faster-whisper, filters noise/fillers, uses LLM to
identify engaging clips, then extracts them in parallel with ffmpeg.

Features:
- Hook-first clip selection with strong openings and closings
- Word-by-word subtitle highlighting (not karaoke sweep)
- YOLO person detection with close-up cropping (horizontal→vertical)
- Background music matching via LLM
- Silent gap removal for better pacing
- Non-speech event preservation (laughter, applause, etc.)

Supports OpenRouter (free default), Anthropic, OpenAI, and local Ollama.
"""

from .transcription import transcribe
from .prefilter import prefilter_segments
from .llm import find_clips
from .extraction import extract_clips
from .postprocess import postprocess_clips
from .subtitles import generate_ass_subtitles, generate_title_overlay, get_clip_words
from .utils import log, tighten_clip_boundaries
from .audio_energy import analyze_audio_energy

__version__ = "0.3.0"
__all__ = [
    "transcribe",
    "prefilter_segments",
    "find_clips",
    "extract_clips",
    "postprocess_clips",
    "generate_ass_subtitles",
    "generate_title_overlay",
    "get_clip_words",
    "log",
    "tighten_clip_boundaries",
    "analyze_audio_energy",
]
