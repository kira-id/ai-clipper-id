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


def _opt_float(opts: dict, key: str) -> float | None:
    """Read an optional float option (e.g. subtitle font size pct).

    Returns None when the key is missing or empty so callers fall back to
    their configured default.
    """
    v = opts.get(key, None)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

# ── Ordered pipeline steps (drives the dashboard stepper) ──────────────────
STEPS = [
    "upload",
    "transcribe",
    "prefilter",
    "energy",
    "select_refine",
    "build",
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
        # Per-step intermediate results (surfaced to the dashboard as
        # collapsible "show output" panels so the user can inspect what each
        # stage produced before the final clips are rendered).
        self.step_results: dict[str, dict] = {}
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

    def set_result(self, name: str, result: dict) -> None:
        """Attach an intermediate result payload for a step.

        ``result`` should be a small JSON-serialisable dict describing what the
        step produced (counts, lists, a preview, etc.). It is rendered by the
        dashboard as a collapsible "show output" panel. Keep it compact.
        """
        with self._lock:
            self.step_results[name] = result

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
                "step_results": dict(self.step_results),
            }


# Per-video execution guard. Two concurrent runs for the SAME video write to the
# same clips/<VideoName>/ output path (keyed only by video stem); the second
# ffmpeg re-encode truncates the first's in-progress file with -y, producing a
# corrupt MP4 ("Invalid NAL unit size … Error splitting the input into NAL
# units") that then fails post-processing. This dedup guard prevents that.
JOBS: dict[str, Job] = {}
# RLock so re-entrant helper calls (e.g. run_job holding _JOBS_LOCK while
# calling _find_running_job_for, which also locks) don't deadlock.
_JOBS_LOCK = threading.RLock()


def _video_lock_key(video_path: str) -> str:
    """Stable key for a video across submissions (resolved absolute path)."""
    return str(Path(video_path).resolve())


def get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return JOBS.get(job_id)


def _find_running_job_for(video_path: str) -> Job | None:
    """Return an in-flight Job for the same video, if any (else None).

    Two concurrent runs for the SAME video re-encode into the same
    clips/<VideoName>/ path; the second ffmpeg -y truncates the first's
    in-progress file, yielding a corrupt MP4 that fails post-processing.
    """
    key = _video_lock_key(video_path)
    with _JOBS_LOCK:
        for job in JOBS.values():
            if (not job.finished
                    and _video_lock_key(job.video_path) == key):
                return job
    return None


def run_job(video_path: str, options: dict) -> Job:
    """Create a Job, start it in a background thread, return it immediately.

    If another run for the SAME video is already in flight, returns that
    running Job instead of starting a competing one. Competing ffmpeg
    re-encodes would clobber each other's output (same clips/<Video>/ path),
    writing a corrupt MP4.
    """
    existing = _find_running_job_for(video_path)
    if existing is not None:
        log("WARN",
            f"A job for {Path(video_path).name} is already running "
            f"(job_id={existing.id}); returning it instead of starting a "
            f"duplicate run that would overwrite its output.")
        return existing

    job_id = f"{int(time.time()*1000)}-{Path(video_path).stem}"
    job = Job(job_id, video_path, options)
    with _JOBS_LOCK:
        # re-check under lock in case two requests raced past the first check
        if _find_running_job_for(video_path) is not None:
            dup = _find_running_job_for(video_path)
            log("WARN",
                f"Duplicate concurrent submission for {Path(video_path).name} "
                f"raced through; returning existing job {dup.id}.")
            return dup
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
            "select_refine": pc.signature(
                opts.get("min", 15), opts.get("max", 60),
                min(int(opts.get("max_clips", 10)), 72),
                opts.get("min_score", 55),
                opts.get("chunk_duration", 360.0),
                opts.get("chunk_overlap", 60.0),
                opts.get("system_prompt") or "",
                opts.get("target_language", "en"),
                opts.get("llm_model"),
            ),
            "build": pc.signature(
                opts.get("encoding_preset"), opts.get("encoding_crf"),
                opts.get("subtitles", True), opts.get("title", False),
                opts.get("orientation", "auto"), opts.get("crop", False),
                opts.get("split_screen", False),
                opts.get("active_speaker", True),
                opts.get("remove_silence", False), opts.get("max_silence", 1.5),
                opts.get("cta", False), opts.get("subtitle_position", "lower"),
                opts.get("subtitle_font_size_pct"),
                opts.get("target_language", "en"),
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
            job.set_result("transcribe", {
                "segments": len(segments),
                "language": detected_language.get("language", "unknown"),
                "language_probability": round(
                    detected_language.get("language_probability", 0.0), 3),
                "cached": True,
                "preview": [{
                    "start": round(s.get("start", 0), 1),
                    "end": round(s.get("end", 0), 1),
                    "text": _trim(s.get("text", ""), 180),
                } for s in segments[:8]],
            })
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
            job.set_result("transcribe", {
                "segments": len(segments),
                "language": detected_language.get("language", "unknown"),
                "language_probability": round(
                    detected_language.get("language_probability", 0.0), 3),
                "cached": False,
                "preview": [{
                    "start": round(s.get("start", 0), 1),
                    "end": round(s.get("end", 0), 1),
                    "text": _trim(s.get("text", ""), 180),
                } for s in segments[:8]],
            })

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
            job.set_result("prefilter", _prefilter_result(filtered, stats, True))
        else:
            filtered, stats = prefilter_segments(segments)
            pc.put_cached(cdir, "prefilter",
                          {"segments": filtered, "stats": stats},
                          sig_parts=(sig["prefilter"],))
            # A cache entry now exists on disk, so the next run will be fast.
            # The badge reflects "is cached" (truthful after a fresh run), not
            # "was served from cache this run" — that distinction lives in the
            # step detail / step_result.cached field.
            job.cache["prefilter"] = True
            job.complete_step("prefilter",
                              f"{stats['original']} → {stats['kept']} segments kept")
            job.set_result("prefilter", _prefilter_result(filtered, stats, False))

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
            job.set_result("energy", _energy_result(energy_events, cached=True))
        else:
            energy_events = analyze_audio_energy(video, segments=filtered)
            pc.put_cached(cdir, "energy",
                          {"events": energy_events},
                          sig_parts=(sig["energy"],))
            job.cache["energy"] = True
            job.complete_step("energy",
                              f"{len(energy_events)} energy events found")
            job.set_result("energy", _energy_result(energy_events, cached=False))

        # 5. SELECT & REFINE (merged: LLM clip selection + boundary/caption
        #    refinement). Cached as a single step — re-running only costs a
        #    cache lookup, not a fresh LLM selection + refinement pass.
        job.set_step("select_refine", "LLM selecting + refining clips…")
        raw_clips_cache = output_dir / ".clips_raw.json"
        # Capture cache state BEFORE running find_clips — that call writes
        # .clips_raw.json, so checking existence afterwards would always be
        # True (and falsely report "cached" on a fresh run).
        llm_was_cached = raw_clips_cache.exists()
        cached = pc.get_cached(cdir, "select_refine", sig["select_refine"])
        if cached is not None:
            clips = cached["clips"]
            job.cache["select_refine"] = True
            job.complete_step("select_refine",
                              f"{len(clips)} clips chosen & finalized (cached)")
            job.set_result("select_refine", _refine_result(clips, cached=True))
        else:
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
            if not clips:
                raise RuntimeError("No engaging clips found. Try lowering "
                                   "--min-score or widening the duration range.")
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
            pc.put_cached(cdir, "select_refine",
                          {"clips": clips},
                          sig_parts=(sig["select_refine"],))
            job.cache["select_refine"] = True
            job.complete_step("select_refine",
                              f"{len(clips)} clips chosen & finalized")
            job.set_result("select_refine",
                           _refine_result(clips, cached=False))

        # persist metadata
        from .utils import save_clips_to_disk
        save_clips_to_disk(clips, output_dir)

        # 6. BUILD (merged: extract raw clips + render subtitles/crop/cta).
        #    The raw intermediate clips are an internal detail of this stage —
        #    only the final rendered outputs are cached, so a re-run that hits
        #    this cache skips BOTH the ffmpeg cut and the post-processing pass.
        job.set_step("build", "Cutting + rendering clips…")
        any_pp = (opts.get("subtitles", True) or opts.get("title", False)
                  or str(opts.get("orientation", "auto")) != "auto"
                  or opts.get("crop", False) or opts.get("remove_silence", False)
                  or opts.get("cta", False))
        cached = pc.get_cached(cdir, "build", sig["build"])
        if cached is not None:
            outputs = cached["outputs"]
            outputs = [p for p in outputs if Path(p).exists()]
            # re-attach subtitle words so the download/preview still works
            for clip in clips:
                clip["_subtitle_words"] = []
            job.cache["build"] = bool(outputs)
            job.complete_step("build",
                              f"{len(outputs)} clips built (cached)")
            job.set_result("build", _render_result(outputs, cached=True))
        else:
            raw_outputs = extract_clips(
                video, clips,
                output_dir=output_dir,
                max_workers=1,
                encoding_preset=opts.get("encoding_preset"),
                encoding_crf=opts.get("encoding_crf"),
            )
            if any_pp and raw_outputs:
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
                    subtitle_font_size_pct=_opt_float(opts, "subtitle_font_size_pct"),
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
            else:
                outputs = raw_outputs
            pc.put_cached(cdir, "build",
                          {"outputs": outputs},
                          files=outputs,
                          sig_parts=(sig["build"],))
            job.cache["build"] = True
            job.complete_step("build",
                              f"{len(outputs)} clips built"
                              + ("" if any_pp else " (raw, no post-processing)"))
            job.set_result("build", _render_result(
                outputs, cached=False,
                note="" if any_pp else "no post-processing"))

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
def _trim(text: str, n: int) -> str:
    """Truncate a string for preview payloads."""
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _prefilter_result(filtered: list, stats: dict, cached: bool) -> dict:
    return {
        "original": stats.get("original"),
        "after_filter": stats.get("after_filter"),
        "kept": stats.get("kept"),
        "dropped": stats.get("dropped"),
        "merged": stats.get("merged"),
        "drop_pct": stats.get("drop_pct"),
        "reasons": stats.get("reasons", {}),
        "cached": cached,
        "preview": [{
            "start": round(s.get("start", 0), 1),
            "end": round(s.get("end", 0), 1),
            "text": _trim(s.get("text", ""), 140),
        } for s in filtered[:8]],
    }


def _energy_result(events: list, cached: bool) -> dict:
    labels = {}
    for e in events:
        labels[e.get("label", "peak")] = labels.get(e.get("label", "peak"), 0) + 1
    return {
        "count": len(events),
        "cached": cached,
        "labels": labels,
        "preview": [{
            "start": round(e.get("start", 0), 1),
            "end": round(e.get("end", 0), 1),
            "label": e.get("label", ""),
            "intensity": round(e.get("intensity", 0), 2)
            if e.get("intensity") is not None else None,
        } for e in events[:12]],
    }


def _refine_result(clips: list, cached: bool) -> dict:
    return {
        "count": len(clips),
        "cached": cached,
        "clips": [{
            "rank": c.get("rank"),
            "start": round(float(c.get("start", 0)), 1),
            "end": round(float(c.get("end", 0)), 1),
            "score": round(float(c.get("clip_score", 0) or 0), 1),
            "title": _trim(c.get("title", ""), 120),
            "caption": _trim(c.get("caption", ""), 160),
        } for c in clips[:24]],
    }


def _render_result(outputs: list, cached: bool, note: str = "") -> dict:
    return {
        "count": len(outputs),
        "cached": cached,
        "note": note,
        "files": [Path(p).name for p in outputs[:24]],
    }


def load_dotenv_safe() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def job_options_to_cli(job: Job) -> dict:
    """Reserved hook — pipeline modules load their own config via config.yaml."""
    return {}
