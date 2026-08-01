"""
Transcription module: faster-whisper wrapper with VAD and batching.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

from .utils import log


def _detect_device() -> str:
    """Return the best available device (cuda if available, else cpu).

    The original implementation silently fell back to CPU when torch was
    missing or CUDA wasn't available, which made it hard to know why the
    log showed "Loading whisper ... on CPU (int8)".  We now log explicit
    warnings so the user can take corrective action.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - environment may lack torch
        log("WARN",
            "PyTorch is not installed; transcription will run on CPU. "
            "Activate the ai_clipper conda env and install a CUDA-enabled "
            "build (see https://pytorch.org/get-started/locally/).")
        return "cpu"

    if not torch.cuda.is_available():
        log("WARN",
            "PyTorch is installed but CUDA is not available. "
            "Either the environment uses a CPU-only build or drivers are "
            "misconfigured. Run `nvidia-smi` and ensure the correct GPU "
            "build of PyTorch is installed.")
        return "cpu"

    return "cuda"


def _resolve_compute_type(device: str, compute_type: str) -> str:
    """Resolve compute type with safe CPU/GPU defaults."""
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    if device == "cpu" and compute_type in {"float16", "int8_float16"}:
        log("WARN", f"compute_type={compute_type} not supported on CPU; using int8")
        return "int8"
    return compute_type


def transcribe(
    video_path: str,
    model_size: str = "large-v3",
    language: str | None = "id",
    device: str = "auto",
    compute_type: str = "auto",
    vad_filter: bool = True,
    vad_min_silence_ms: int = 400,
    vad_speech_pad_ms: int = 200,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Transcribe video with faster-whisper.

    Returns tuple of (segments, language_info):
      segments: list[{"start": float, "end": float, "text": str, "words": [...], "no_speech_prob": float}]
      language_info: {"language": str, "language_probability": float}

    Tuned for Indonesian:
    - language defaults to "id" (skip auto-detect overhead)
    - VAD aggressively filters silence common in podcast/talk formats
    - word_timestamps for precise cut boundaries
    - batch_size > 1 uses BatchedInferencePipeline
    """
    # Normalize language sentinels. faster-whisper rejects "auto"/"none"
    # as a literal language code; both mean "let the model detect".
    # Centralizing this here guarantees no caller can pass an invalid code.
    if language is not None and str(language).strip().lower() in ("auto", "none", ""):
        language = None

    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        log("ERROR", "faster-whisper not installed.  pip install faster-whisper")
        sys.exit(1)

    # choose device, with helpful diagnostics
    if device == "auto":
        device = _detect_device()
    elif device == "cuda":
        # user explicitly requested CUDA; verify availability
        try:
            import torch
        except ImportError:
            log("ERROR", "device=cuda requested but PyTorch is not installed. "
                          "Install a CUDA-enabled PyTorch build or use device=cpu.")
            sys.exit(1)
        if not torch.cuda.is_available():
            log("ERROR", "device=cuda requested but CUDA is not available. "
                          "Check your drivers or install the correct PyTorch build.")
            sys.exit(1)

    compute_type = _resolve_compute_type(device, compute_type)

    from .utils import BOLD, RESET
    
    # Check if model is cached
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        model_repo = f"Systran/faster-whisper-{model_size}"
        cached = any(model_repo in repo.repo_id for repo in cache_info.repos)
        cache_status = "(from cache)" if cached else "(downloading...)"
    except Exception:
        cache_status = ""
    
    log("INFO", f"Loading whisper {BOLD}{model_size}{RESET} on {device.upper()} ({compute_type}) {cache_status}")
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=min(4, os.cpu_count() or 1),
    )
    log("OK", f"Model loaded in {time.time() - t0:.1f}s")

    log("INFO", f"Transcribing {BOLD}{Path(video_path).name}{RESET} …")
    t0 = time.time()

    condition_prev = False if model_size.startswith("distil-") else True

    # This app always needs word-level timestamps for the burned-in TikTok-style
    # subtitles. faster-whisper's BatchedInferencePipeline does NOT compute them
    # (its own docstring: "word_timestamps ... Set as False", and the batched
    # generate_segments path never calls add_word_timestamps), so forcing
    # batch_size > 1 would silently produce empty `words` on every segment →
    # no subtitles in the output. Force the sequential pipeline regardless of the
    # batch_size setting; the knob stays useful only when word timestamps are
    # ever explicitly disabled downstream.
    word_timestamps = True
    if batch_size > 1:
        log("WARN",
            f"word_timestamps=True requested but batch_size={batch_size} "
            f"disables word-level timestamps in faster-whisper. "
            f"Forcing batch_size=1 so subtitles are generated correctly "
            f"(slower, but the only path that yields word timings).")
        batch_size = 1

    transcribe_kwargs = dict(
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=condition_prev,
    )

    if vad_filter:
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms": vad_speech_pad_ms,
        }

    if batch_size > 1:
        batched = BatchedInferencePipeline(model=model)
        segments_gen, info = batched.transcribe(
            video_path,
            batch_size=batch_size,
            **transcribe_kwargs,
        )
    else:
        segments_gen, info = model.transcribe(
            video_path,
            **transcribe_kwargs,
        )

    segments: list[dict[str, Any]] = []
    for seg in segments_gen:
        words = (
            [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
            if seg.words else []
        )
        seg_data: dict[str, Any] = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
            "no_speech_prob": round(seg.no_speech_prob, 4),
        }
        # Capture non-speech annotations from Whisper
        # These include [music], [laughter], [applause], etc.
        # which can be interesting for clip content
        text_lower = seg.text.strip().lower()
        is_non_speech = any(
            marker in text_lower
            for marker in (
                "[music]", "[laughter]", "[applause]", "[cheering]",
                "[singing]", "[crowd]", "(music)", "(laughter)",
                "(applause)", "[background music]", "[sound effect]",
                "[gasps]", "[sighs]", "[clapping]",
            )
        )
        if is_non_speech:
            seg_data["non_speech_event"] = True
        segments.append(seg_data)

    elapsed = time.time() - t0
    duration = segments[-1]["end"] if segments else 0.0
    rtf = elapsed / duration if duration else 0
    log("OK",
        f"{duration:.0f}s audio → {len(segments)} segments "
        f"| RTF {rtf:.2f}x | {elapsed:.1f}s")
    language_info = {
        "language": info.language or "unknown",
        "language_probability": float(info.language_probability or 0.0),
    }
    if info.language:
        log("INFO", f"Language: {info.language} (p={info.language_probability:.0%})")

    return segments, language_info
