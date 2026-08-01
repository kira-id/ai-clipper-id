"""
Single-video subtitle runner.

Focused pipeline for the single-video dashboard: add subtitles (with or
without translation) to one video. No clip selection, no LLM clip analysis.

Steps:
  upload      -> video received
  transcribe  -> faster-whisper -> word timestamps
  subtitles   -> build/translate word list (optional LLM translate)
  render      -> extract full video + burn subtitles with ffmpeg
  done        -> output ready

The same word-timestamp + ASS primitives as the multi-clip pipeline are
reused so the burnt look is identical. For "subtitles without translation"
we burn the raw transcript words and skip the LLM entirely (works offline).

Usage:
  from .single_video_runner import run_single, get_job, JOBS
  job = run_single(video_path, options)   # background thread
  state = job.to_dict()                    # polled by the web layer
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .utils import log
from .config import get_cta_settings, get_defaults


# ── Ordered pipeline steps (drives the dashboard stepper) ──────────────────
STEPS = [
    "upload",
    "transcribe",
    "subtitles",
    "render",
    "done",
]


class _LogSink(io.StringIO):
    """Capture everything written to stdout during a job run, with level tags."""

    def __init__(self, job: "Job"):
        super().__init__()
        self._job = job

    def write(self, s: str) -> int:
        text = s.rstrip("\n")
        if text:
            if text.startswith("[") and "]" in text:
                level, _, msg = text[1:].partition("]")
                self._job.add_log(msg.strip(), level.strip())
            else:
                self._job.add_log(text, "INFO")
        return len(s)


class Job:
    """One single-video run. Thread-safe for concurrent polling."""

    def __init__(self, job_id: str, video_path: str, options: dict):
        self.id = job_id
        self.video_path = str(video_path)
        self.options = options
        self.created_at = time.time()

        self._lock = threading.RLock()
        self.step = "upload"
        self.step_index = 0
        self.step_status: dict[str, str] = {s: "pending" for s in STEPS}
        self.step_detail: dict[str, str] = {}
        self.logs: list[dict] = []
        self.outputs: list[dict] = []   # final video file(s)
        self.error: str | None = None
        self.finished = False
        self.progress = 0
        self.output_dir: str | None = None
        self.output_path: str | None = None

        self.step_status["upload"] = "active"

    # ── thread-safe mutators ──
    def set_step(self, name: str, detail: str = "") -> None:
        with self._lock:
            if name not in self.step_status:
                return
            if self.step in self.step_status and self.step_status[self.step] == "active":
                self.step_status[self.step] = "done"
            self.step = name
            self.step_index = STEPS.index(name)
            self.step_status[name] = "active"
            if detail:
                self.step_detail[name] = detail
            self.progress = int(round(100 * self.step_index / (len(STEPS) - 1)))

    def complete_step(self, name: str, detail: str = "") -> None:
        with self._lock:
            self.step_status[name] = "done"
            if detail:
                self.step_detail[name] = detail

    def add_log(self, message: str, level: str = "INFO") -> None:
        with self._lock:
            self.logs.append({
                "t": round(time.time() - self.created_at, 1),
                "level": level,
                "msg": message,
            })
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]

    def fail(self, message: str) -> None:
        self.error = message
        self.step_status[self.step] = "error"
        self.finished = True
        self.add_log(message, "ERROR")

    def finish(self) -> None:
        with self._lock:
            self.step_status["done"] = "done"
            for s in STEPS:
                if self.step_status[s] == "active":
                    self.step_status[s] = "done"
            self.progress = 100
            self.finished = True

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "video": Path(self.video_path).name,
                "step": self.step,
                "step_index": self.step_index,
                "progress": self.progress,
                "steps": STEPS,
                "status": dict(self.step_status),
                "detail": dict(self.step_detail),
                "logs": list(self.logs),
                "outputs": list(self.outputs),
                "output_dir": self.output_dir,
                "output_path": self.output_path,
                "error": self.error,
                "finished": self.finished,
            }


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return JOBS.get(job_id)


def run_single(video_path: str, options: dict) -> Job:
    """Create a Job, start it in a background thread, return it immediately."""
    job_id = f"sv-{int(time.time()*1000)}-{Path(video_path).stem}"
    job = Job(job_id, video_path, options)
    with _JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=_run, args=(job,), daemon=True)
    t.start()
    return job


# ── Actual pipeline ────────────────────────────────────────────────────────
def _run(job: Job) -> None:
    opts = job.options
    video = job.video_path

    # Redirect stdout into the job log sink so all pipeline [LEVEL] logs are
    # captured. Restore afterwards.
    old_stdout = sys.stdout
    sys.stdout = _LogSink(job)

    from .transcription import transcribe
    from .prefilter import prefilter_segments
    from .extraction import extract_clips, _get_video_duration
    from .postprocess import postprocess_clips
    from .subtitles import get_clip_words, generate_ass_subtitles

    try:
        from .config import load_config
        load_config()
        load_dotenv_safe()

        output_dir = Path(opts.get("output_dir", "clips")) / (
            opts.get("video_name") or Path(video).stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(output_dir)

        # 1. TRANSCRIBE
        job.set_step("transcribe", "Loading whisper model…")
        cache_path = _transcript_cache_path(video)
        detected_language = {"language": "unknown", "language_probability": 0.0}
        if cache_path.exists() and not opts.get("no_cache", False):
            job.add_log(f"Loading cached transcript from {cache_path}")
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache_data, dict) and "segments" in cache_data:
                segments = cache_data["segments"]
                detected_language = cache_data.get("language_info", detected_language)
            else:
                segments = cache_data
            job.complete_step("transcribe",
                              f"{len(segments)} segments (cached)")
        else:
            _lang_opt = str(opts.get("lang", "auto")).lower()
            _lang = None if _lang_opt in ("auto", "none") else _lang_opt
            segments, detected_language = transcribe(
                video,
                model_size=opts.get("model", "tiny"),
                language=_lang,
                device=opts.get("device", "auto"),
                compute_type=opts.get("compute_type", "auto"),
                vad_min_silence_ms=opts.get("vad_min_silence", 400),
                vad_speech_pad_ms=opts.get("vad_speech_pad", 200),
                batch_size=opts.get("batch", 16),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(
                {"segments": segments, "language_info": detected_language},
                indent=2, ensure_ascii=False,
            ), encoding="utf-8")
            job.complete_step("transcribe",
                              f"{len(segments)} segments · "
                              f"lang {detected_language.get('language','?')}")

        # Filter noise so subtitles are clean.
        filtered, stats = prefilter_segments(segments)
        job.add_log(f"Pre-filtered: {stats['original']} → {stats['kept']} segments")

        if not filtered:
            raise RuntimeError(
                "No speech detected. Try a different whisper model or a "
                "clearer audio source.")

        video_dur = _get_video_duration(video) or float(filtered[-1]["end"])

        # 2. SUBTITLE WORDS
        job.set_step("subtitles", "Building subtitle words…")
        raw_words = get_clip_words(segments, clip_start=0.0, clip_end=video_dur)

        subtitle_mode = str(opts.get("subtitle_mode", "original")).lower()
        full_clip: dict[str, Any] = {
            "rank": 1,
            "start": 0.0,
            "end": video_dur,
            "title": Path(video).stem,
            "topic": "",
            "caption": "",
            "clip_score": 100,
        }

        translated = False
        if subtitle_mode in ("translate", "both") and raw_words:
            # Translate (and fix Whisper errors). Requires an LLM key.
            job.add_log("Translating subtitle words (LLM)…")
            from .llm import translate_subtitle_words
            words = translate_subtitle_words(
                raw_words,
                llm_model=opts.get("llm_model"),
                api_key=opts.get("api_key"),
                fix_errors=True,
                target_language=opts.get("target_language", "en"),
            )
            translated = True
        else:
            # Burn the original transcript words (no LLM call).
            words = raw_words

        # Also include a translated copy as a second track when "both".
        words_translated: list[dict] | None = None
        if subtitle_mode == "both" and raw_words:
            # original words already in `words`; build the translated track
            from .llm import translate_subtitle_words
            words_translated = translate_subtitle_words(
                raw_words,
                llm_model=opts.get("llm_model"),
                api_key=opts.get("api_key"),
                fix_errors=True,
                target_language=opts.get("target_language", "en"),
            )

        full_clip["_subtitle_words"] = words
        job.complete_step("subtitles",
                          f"{len(words)} words · "
                          f"{'translated' if translated else 'original language'}")

        # 3. RENDER — extract the full video, then burn subtitles.
        job.set_step("render", "Extracting + burning subtitles…")
        from .cli import _ensure_filenames
        _ensure_filenames([full_clip])

        raw_outputs = extract_clips(
            video, [full_clip],
            output_dir=output_dir,
            max_workers=1,
            encoding_preset=opts.get("encoding_preset"),
            encoding_crf=opts.get("encoding_crf"),
        )
        if not raw_outputs:
            raise RuntimeError("Failed to extract full video.")

        _cta_cfg = {**get_cta_settings(),
                    "enabled": bool(opts.get("cta", False))}
        outputs = postprocess_clips(
            raw_outputs, [full_clip], segments,
            output_dir=output_dir,
            subtitles=True,
            subtitle_position=opts.get("subtitle_position", "lower"),
            subtitle_margin_pct=opts.get("subtitle_margin_pct"),
            enable_title=False,
            orientation="auto",
            enable_crop=False,
            enable_active_speaker=False,
            enable_silence_removal=opts.get("remove_silence", False),
            max_silence=opts.get("max_silence", 1.5),
            cta_config=_cta_cfg,
            encoding_preset=opts.get("encoding_preset"),
            encoding_crf=opts.get("encoding_crf"),
        )
        if not outputs:
            raise RuntimeError(
                "Rendering produced no output — subtitles were not burned in. "
                "Check the log above for the ffmpeg error.")

        final_path = outputs[0] if outputs else raw_outputs[0]
        job.output_path = str(final_path)
        job.outputs.append({
            "filename": Path(final_path).name,
            "path": str(final_path),
            "size_mb": round(Path(final_path).stat().st_size / 1_048_576, 1),
            "translated": translated,
            "language": detected_language.get("language", "unknown"),
        })
        job.complete_step("render", f"{Path(final_path).name} ready")

        job.set_step("done", "Subtitled video ready")
        job.finish()

    except Exception as e:  # noqa: BLE001
        job.add_log(traceback.format_exc(), "ERROR")
        job.fail(str(e))
    finally:
        sys.stdout = old_stdout


# ── helpers ──────────────────────────────────────────────────────────────
def load_dotenv_safe() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _transcript_cache_path(video_path: str) -> Path:
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{video.stem}_transcript.json"
