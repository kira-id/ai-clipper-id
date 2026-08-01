"""
Web runner: drives the clip pipeline as a tracked Job so the dashboard can
show step-by-step progress, a live log feed, and final output files.

The pipeline mirrors sosmed/cli.py:main() exactly (transcribe -> prefilter ->
audio energy -> LLM clip selection -> refine -> extract -> render) but records
progress into a Job object instead of printing to stdout.

Usage:
  from .web_runner import run_job, get_job, JOBS
  job = run_job(video_path, options)   # starts in background thread
  state = job.to_dict()                # polled by the web layer
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
from .config import get_cta_settings
from . import pipeline_cache as pc

# ── Ordered pipeline steps (drives the dashboard stepper) ──────────────────
STEPS = [
    "upload",
    "transcribe",
    "prefilter",
    "energy",
    "llm_select",
    "refine",
    "extract",
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
            # sosmed.utils.log prints "[LEVEL] message"; parse for coloring.
            if text.startswith("[") and "]" in text:
                level, _, msg = text[1:].partition("]")
                self._job.add_log(msg.strip(), level.strip())
            else:
                self._job.add_log(text, "INFO")
        return len(s)


class Job:
    """One dashboard run. Thread-safe for concurrent polling."""

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
        self.outputs: list[dict] = []   # final clip files
        self.clips_meta: list[dict] = []  # clip metadata (for the results list)
        self.error: str | None = None
        self.finished = False
        self.progress = 0  # 0..100

        # Per-step cache state (surfaced to the dashboard so the user can see
        # which steps are served from cache and clear them).
        self.cache: dict[str, bool] = {}
        self.cache_dir: str | None = None
        self.video_stem: str | None = None

        self.step_status["upload"] = "active"
        self.output_dir: str | None = None

    # ── thread-safe mutators ──
    def set_step(self, name: str, detail: str = "") -> None:
        with self._lock:
            if name not in self.step_status:
                return
            # mark previous active step done
            if self.step in self.step_status and self.step_status[self.step] == "active":
                self.step_status[self.step] = "done"
            self.step = name
            self.step_index = STEPS.index(name)
            self.step_status[name] = "active"
            if detail:
                self.step_detail[name] = detail
            # progress heuristic from step position
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
            # cap memory
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
                "clips": list(self.clips_meta),
                "output_dir": self.output_dir,
                "error": self.error,
                "finished": self.finished,
                "cache": dict(self.cache),
            }


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return JOBS.get(job_id)


def run_job(video_path: str, options: dict) -> Job:
    """Create a Job, start it in a background thread, return it immediately."""
    job_id = f"{int(time.time()*1000)}-{Path(video_path).stem}"
    job = Job(job_id, video_path, options)
    with _JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=_run, args=(job,), daemon=True)
    t.start()
    return job


def clear_job_cache(job_id: str, step: str | None = None) -> int:
    """Clear the pipeline cache for a job's video.

    If ``step`` is given, only that one step's cache is removed; otherwise the
    entire per-video .cache folder is cleared. Returns the number of cache
    files removed. This is a user-initiated action from the dashboard — it
    only affects *future* runs (a running job keeps its in-memory data).
    """
    job = get_job(job_id)
    if not job or not job.cache_dir:
        return 0
    cdir = Path(job.cache_dir)
    if step:
        return pc.clear_step(cdir, step)
    return pc.clear_cache(cdir)


def rerun_job(job_id: str, overrides: dict | None = None) -> Job | None:
    """Re-run a previous job's *same upload* with merged option overrides.

    Returns None if the original job (or its video file) no longer exists.
    This is an explicit, user-initiated action — no automatic fallback.
    """
    old = get_job(job_id)
    if not old:
        return None
    if not Path(old.video_path).exists():
        return None
    merged = dict(old.options)
    if overrides:
        merged.update(overrides)
    # a rerun is a fresh attempt — use the explicit override, else the
    # original job's min_score (never a "lowered" value from a prior retry)
    merged["min_score"] = int(overrides.get("min_score")
                              if overrides else old.options.get("min_score", 55))
    return run_job(old.video_path, merged)


# ── Actual pipeline (mirrors cli.py) ───────────────────────────────────────
def _run(job: Job) -> None:
    opts = job.options
    video = job.video_path

    # Redirect stdout into the job log sink so all pipeline [LEVEL] logs are
    # captured. We keep a reference to restore later.
    old_stdout = sys.stdout
    sys.stdout = _LogSink(job)

    from .transcription import transcribe
    from .prefilter import prefilter_segments
    from .llm import find_clips, fix_and_improve_clips
    from .extraction import extract_clips, _get_video_duration
    from .postprocess import postprocess_clips
    from .smart_clip_boundaries import smart_adjust_clip_boundaries
    from .audio_energy import analyze_audio_energy

    try:
        from .config import load_config
        load_config()  # same as cli.py — merges config.yaml with defaults
        load_dotenv_safe()

        output_dir = Path(opts.get("output_dir", "clips")) / (
            opts.get("video_name") or Path(video).stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(output_dir)

        cdir = pc.cache_dir_for(video, str(opts.get("output_dir", "clips")))
        job.cache_dir = str(cdir)
        job.video_stem = Path(video).stem
        pc.touch(cdir)

        # A per-step signature reflects every input that could change the
        # step's result. If the signature matches a stored cache entry we skip
        # the (expensive) computation and serve the cached result instead.
        sig = {
            "transcribe": pc.signature(
                str(video),
                opts.get("model", "turbo"),
                opts.get("lang", "id"),
                opts.get("vad_min_silence", 400),
                opts.get("vad_speech_pad", 200),
                opts.get("batch", 16),
            ),
            "prefilter": pc.signature(
                "prefilter", opts.get("vad_min_silence", 400)),
            "energy": pc.signature("energy"),
            "llm_select": pc.signature(
                opts.get("min", 15), opts.get("max", 60),
                min(int(opts.get("max_clips", 10)), 72),
                opts.get("min_score", 55),
                opts.get("chunk_duration", 360.0),
                opts.get("chunk_overlap", 60.0),
                opts.get("system_prompt") or "",
            ),
            "refine": pc.signature(
                opts.get("target_language", "en"),
                opts.get("llm_model"),
            ),
            "extract": pc.signature(
                opts.get("encoding_preset"), opts.get("encoding_crf")),
            "render": pc.signature(
                opts.get("subtitles", True), opts.get("title", False),
                opts.get("orientation", "auto"), opts.get("crop", False),
                opts.get("split_screen", False),
                opts.get("active_speaker", True),
                opts.get("remove_silence", False), opts.get("max_silence", 1.5),
                opts.get("cta", False), opts.get("subtitle_position", "lower"),
                opts.get("target_language", "en"),
                opts.get("encoding_preset"), opts.get("encoding_crf"),
            ),
        }

        # 1. TRANSCRIBE
        job.set_step("transcribe", "Loading whisper model…")
        cache_path = pc.transcript_cache_path(video)
        detected_language = {"language": "unknown", "language_probability": 0.0}
        if cache_path.exists():
            job.add_log(f"Loading cached transcript from {cache_path}")
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache_data, dict) and "segments" in cache_data:
                segments = cache_data["segments"]
                detected_language = cache_data.get("language_info", detected_language)
            else:
                segments = cache_data
            job.cache["transcribe"] = True
            job.complete_step("transcribe",
                              f"{len(segments)} segments (cached)")
        else:
            segments, detected_language = transcribe(
                video,
                model_size=opts.get("model", "turbo"),
                language=None if str(opts.get("lang", "id")).lower() == "none"
                else opts.get("lang", "id"),
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
            job.cache["transcribe"] = False
            job.complete_step("transcribe",
                              f"{len(segments)} segments · "
                              f"lang {detected_language.get('language','?')}")

        # 2. PREFILTER
        job.set_step("prefilter", "Removing noise / fillers / duplicates…")
        cached = pc.get_cached(cdir, "prefilter", sig["prefilter"])
        if cached is not None:
            filtered = cached["segments"]
            stats = cached["stats"]
            job.cache["prefilter"] = True
            job.complete_step("prefilter",
                              f"{stats['original']} → {stats['kept']} "
                              f"segments kept (cached)")
        else:
            filtered, stats = prefilter_segments(segments)
            pc.put_cached(cdir, "prefilter",
                          {"segments": filtered, "stats": stats},
                          sig_parts=(sig["prefilter"],))
            job.cache["prefilter"] = False
            job.complete_step("prefilter",
                              f"{stats['original']} → {stats['kept']} segments kept")

        if not filtered:
            raise RuntimeError(
                "All segments filtered out. Try a different whisper model or "
                "looser filters.")

        # 3. AUDIO ENERGY
        job.set_step("energy", "Detecting emotional peaks…")
        cached = pc.get_cached(cdir, "energy", sig["energy"])
        if cached is not None:
            energy_events = cached["events"]
            job.cache["energy"] = True
            job.complete_step("energy",
                              f"{len(energy_events)} energy events found (cached)")
        else:
            energy_events = analyze_audio_energy(video, segments=filtered)
            pc.put_cached(cdir, "energy",
                          {"events": energy_events},
                          sig_parts=(sig["energy"],))
            job.cache["energy"] = False
            job.complete_step("energy",
                              f"{len(energy_events)} energy events found")

        # 4. LLM CLIP SELECTION (with retry — the free LLM can occasionally
        #    return 0 clips on a valid transcript; retry by relaxing the
        #    min-score threshold before giving up.)
        job.set_step("llm_select", "LLM is choosing the best clips…")
        raw_clips_cache = output_dir / ".clips_raw.json"
        clips = None
        last_score = int(opts.get("min_score", 55))
        for attempt in range(3):
            effective_score = max(0, last_score - attempt * 15)
            if attempt:
                job.add_log(f"Retry {attempt}: lowering min-score to "
                            f"{effective_score}", "WARN")
            clips = find_clips(
                filtered,
                min_duration=opts.get("min", 15),
                max_duration=opts.get("max", 60),
                max_clips=min(int(opts.get("max_clips", 10)), 72),
                min_score=effective_score,
                llm_model=opts.get("llm_model"),
                api_key=opts.get("api_key"),
                video_duration=_get_video_duration(video),
                chunk_duration=opts.get("chunk_duration", 360.0),
                chunk_overlap=opts.get("chunk_overlap", 60.0),
                raw_clips_cache_file=raw_clips_cache,
                energy_events=energy_events,
                system_prompt=opts.get("system_prompt") or None,
            )
            if clips:
                break
        job.cache["llm_select"] = raw_clips_cache.exists()
        job.complete_step("llm_select",
                          f"{len(clips)} clips chosen by LLM")

        if not clips:
            raise RuntimeError("No engaging clips found. Try lowering "
                               "--min-score or widening the duration range.")

        # 5. REFINE
        job.set_step("refine", "Optimizing boundaries & improving captions…")
        cached = pc.get_cached(cdir, "refine", sig["refine"])
        if cached is not None:
            clips = cached["clips"]
            job.cache["refine"] = True
            job.complete_step("refine",
                              f"{len(clips)} clips finalized (cached)")
        else:
            clips = smart_adjust_clip_boundaries(
                clips, segments,
                min_duration=5.0,
                max_duration=float(opts.get("max", 60)),
                validate_hook_closing=True,
                aggressive_optimization=True,
            )
            clips = fix_and_improve_clips(
                clips,
                llm_model=opts.get("llm_model"),
                api_key=opts.get("api_key"),
                detected_language=detected_language,
                target_language=opts.get("target_language", "en"),
            )
            # ensure filenames
            from .cli import _ensure_filenames
            _ensure_filenames(clips)
            pc.put_cached(cdir, "refine",
                          {"clips": clips},
                          sig_parts=(sig["refine"],))
            job.cache["refine"] = False
            job.complete_step("refine",
                              f"{len(clips)} clips finalized")

        # persist metadata
        from .utils import save_clips_to_disk
        save_clips_to_disk(clips, output_dir)

        # 6. EXTRACT
        job.set_step("extract", "Cutting raw clips with ffmpeg…")
        # render consumes the raw extract files (postprocess deletes them), so
        # on a cache-only rerun the raw files may no longer be on disk. If the
        # render step is ALSO cached we don't need them — treat extract as a
        # hit anyway. If render is not cached we must re-extract to recover
        # the raw files it needs.
        render_cached = pc.get_cached(cdir, "render", sig["render"]) is not None
        cached = pc.get_cached(cdir, "extract", sig["extract"])
        if cached is not None:
            raw_outputs = cached["outputs"]
            raw_outputs = [p for p in raw_outputs if Path(p).exists()]
            if raw_outputs or render_cached:
                job.cache["extract"] = True
                job.complete_step("extract",
                                  f"{len(raw_outputs)} raw clips (cached)")
            else:
                # signature matched but raw files were consumed and render
                # needs them — fall through to a real re-extract below
                cached = None
        if cached is None:
            raw_outputs = extract_clips(
                video, clips,
                output_dir=output_dir,
                max_workers=1,
                encoding_preset=opts.get("encoding_preset"),
                encoding_crf=opts.get("encoding_crf"),
            )
            pc.put_cached(cdir, "extract",
                          {"outputs": raw_outputs},
                          files=raw_outputs,
                          sig_parts=(sig["extract"],))
            job.cache["extract"] = False
            job.complete_step("extract",
                              f"{len(raw_outputs)} raw clips extracted")

        # 7. RENDER (subtitles / crop / cta / silence removal)
        job.set_step("render", "Burning subtitles & post-processing…")
        any_pp = (opts.get("subtitles", True) or opts.get("title", False)
                  or str(opts.get("orientation", "auto")) != "auto"
                  or opts.get("crop", False) or opts.get("remove_silence", False)
                  or opts.get("cta", False))
        cached = pc.get_cached(cdir, "render", sig["render"])
        if cached is not None:
            outputs = cached["outputs"]
            outputs = [p for p in outputs if Path(p).exists()]
            # re-attach subtitle words so the download/preview still works
            for clip in clips:
                clip["_subtitle_words"] = []
            job.cache["render"] = bool(outputs)
            job.complete_step("render",
                              f"{len(outputs)} clips rendered (cached)")
        elif any_pp and raw_outputs:
            # prepare subtitle words for ALL clips in one batched LLM call
            # (previously one serial call per clip — the dominant runtime cost)
            from .llm import batch_translate_subtitles
            try:
                subtitle_map = batch_translate_subtitles(
                    clips, segments,
                    llm_model=opts.get("llm_model"),
                    api_key=opts.get("api_key"),
                    fix_errors=True,
                    target_language=opts.get("target_language", "en"),
                )
                for clip in clips:
                    clip["_subtitle_words"] = subtitle_map.get(
                        int(clip.get("rank", 0)), [])
            except Exception as e:
                job.add_log(f"Subtitle batch translation failed: {e}", "WARN")
                for clip in clips:
                    clip["_subtitle_words"] = []

            _cta_cfg = {**get_cta_settings(),
                        "enabled": bool(opts.get("cta", False))}
            crop_target = ("vertical"
                           if str(opts.get("orientation", "auto")) in
                           ("portrait", "vertical", "auto")
                           else "horizontal")
            outputs = postprocess_clips(
                raw_outputs, clips, segments,
                output_dir=output_dir,
                subtitles=opts.get("subtitles", True),
                subtitle_position=opts.get("subtitle_position", "lower"),
                enable_title=opts.get("title", False),
                orientation=opts.get("orientation", "auto"),
                enable_crop=opts.get("crop", False),
                crop_target=crop_target,
                enable_split_screen=opts.get("split_screen", False),
                enable_active_speaker=opts.get("active_speaker", True),
                enable_silence_removal=opts.get("remove_silence", False),
                max_silence=opts.get("max_silence", 1.5),
                cta_config=_cta_cfg,
                encoding_preset=opts.get("encoding_preset"),
                encoding_crf=opts.get("encoding_crf"),
            )
            pc.put_cached(cdir, "render",
                          {"outputs": outputs},
                          files=outputs,
                          sig_parts=(sig["render"],))
            job.cache["render"] = False
            job.complete_step("render",
                              f"{len(outputs)} clips rendered")
        else:
            outputs = raw_outputs
            job.cache["render"] = False
            job.complete_step("render", "raw clips (no post-processing)")

        # collect outputs + metadata
        for clip in clips:
            out_path = None
            if "filename" in clip:
                cand = output_dir / clip["filename"]
                if cand.exists():
                    out_path = str(cand)
            job.outputs.append({
                "filename": Path(out_path).name if out_path else None,
                "path": out_path,
                "rank": clip.get("rank"),
                "title": clip.get("title"),
                "topic": clip.get("topic"),
                "caption": clip.get("caption"),
                "score": clip.get("clip_score"),
                "start": clip.get("start"),
                "end": clip.get("end"),
                "size_mb": round(Path(out_path).stat().st_size / 1_048_576, 1)
                if out_path else None,
            })
        # metadata for the results table (public fields only)
        job.clips_meta = [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in clips
        ]

        job.set_step("done", f"{len(outputs)} clips ready")
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


def job_options_to_cli(job: Job) -> dict:
    """Reserved hook — pipeline modules load their own config via config.yaml."""
    return {}
