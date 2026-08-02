"""
Mock output preview — verify dashboard rendering WITHOUT running the pipeline.

The full pipeline (whisper transcription -> LLM clip selection -> ffmpeg extract
-> render) takes minutes and needs a real video + API keys. To *quickly* check
that titles, subtitles, positions, orientation and topic chips show up correctly
in the dashboard, this module synthesizes a short sample clip and burns the
overlays through the EXACT same ASS generators the real render uses
(generate_ass_subtitles / generate_title_overlay) so the preview is faithful —
not a fake placeholder.

The preview reuses the real web_runner.Job so it appears in the dashboard exactly
like a real run (steps stepper, live log, output cards).

Usage (programmatic):
    from sosmed.mock import build_mock_job
    job = build_mock_job()           # registers in JOBS, returns Job

HTTP: POST /api/mock  ->  {"job_id": "...", "state": {...}}
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .utils import get_ffmpeg
from .subtitles import generate_ass_subtitles, generate_title_overlay
from .postprocess import _escape_ass_path
from .web_runner import Job, JOBS, _JOBS_LOCK, _opt_float
from .single_video_runner import Job as SingleJob, JOBS as SINGLE_JOBS, _JOBS_LOCK as _SINGLE_LOCK


# ── Sample data ─────────────────────────────────────────────────────────────
# Word-level timestamps (relative to clip start) for a fake caption track.
SAMPLE_WORDS: list[dict[str, Any]] = [
    {"word": "Ini", "start": 0.2, "end": 0.7},
    {"word": "cara", "start": 0.7, "end": 1.1},
    {"word": "tercepat", "start": 1.1, "end": 1.7},
    {"word": "bikin", "start": 1.7, "end": 2.1},
    {"word": "video", "start": 2.1, "end": 2.5},
    {"word": "pendek", "start": 2.5, "end": 3.0},
    {"word": "dengan", "start": 3.3, "end": 3.8},
    {"word": "AI", "start": 3.8, "end": 4.2},
    {"word": "agent", "start": 4.2, "end": 4.7},
    {"word": "yang", "start": 5.0, "end": 5.3},
    {"word": "bekerja", "start": 5.3, "end": 5.8},
    {"word": "otomatis", "start": 5.8, "end": 6.4},
    {"word": "langsung", "start": 6.4, "end": 6.9},
    {"word": "dari", "start": 7.0, "end": 7.3},
    {"word": "browser", "start": 7.3, "end": 7.9},
    {"word": "kamu", "start": 7.9, "end": 8.2},
]

SAMPLE_CLIPS: list[dict[str, Any]] = [
    {
        "rank": 1,
        "score": 92.4,
        "title": "Cara Tercepat Bikin Video Pendek dengan AI",
        "topic": "AI video editing · tutorial",
        "start": 0.0,
        "end": 8.5,
        "caption": "AI agent bisa bikin video pendek otomatis langsung dari browser.",
        "subtitle_words": SAMPLE_WORDS,
    },
    {
        "rank": 2,
        "score": 87.1,
        "title": "Why Pixel Clicking Still Beats MCP",
        "topic": "computer-use agents · deep dive",
        "start": 0.0,
        "end": 8.5,
        "caption": "Pixel-level control outperforms protocol-based agents on long tasks.",
        "subtitle_words": SAMPLE_WORDS,
    },
    {
        "rank": 3,
        "score": 79.6,
        "title": "3 Langkah Adopsi AI yang Simpel",
        "topic": "AI adoption · tips",
        "start": 0.0,
        "end": 8.5,
        "caption": "Tiga langkah simpel untuk adopsi AI di tim kamu.",
        "subtitle_words": SAMPLE_WORDS,
    },
]


def _build_background(out_path: Path, width: int, height: int,
                       duration: float = 9.0) -> None:
    """Generate a moving-gradient background clip with ffmpeg (no external assets).

    Uses a synthetic test source (no input file needed) so it works offline and
    in CI. A colored gradient + subtle motion makes burned overlays easy to read.
    """
    fps = 30
    # testsrc2 = smooth color gradient with a moving counter, fully synthetic.
    vf = (f"testsrc2=size={width}x{height}:rate={fps}:duration={duration},"
          f"format=yuv420p")
    cmd = [
        get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", vf,
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        str(out_path),
    ]
    import subprocess
    subprocess.run(cmd, check=True)


def _burn_one(clip: dict, output_dir: Path, *, subtitles: bool,
              title: bool, orientation: str, subtitle_position: str,
              width: int, height: int,
              subtitle_font_size_pct: float | None = None) -> str | None:
    """Burn the overlays for a single clip onto a generated background.

    Returns the output filename (relative), or None if ffmpeg failed.
    Uses the SAME ASS generators as the production render path.
    """
    rank = clip.get("rank", 1)
    tmp_bg = output_dir / f".mock_bg_{rank}.mp4"
    try:
        _build_background(tmp_bg, width, height)
    except Exception as e:  # pragma: no cover - environment issue
        print(f"[mock] background generation failed: {e}")
        return None

    filters: list[str] = []
    if subtitles and clip.get("subtitle_words"):
        ass = tempfile.NamedTemporaryFile(suffix=".ass", prefix="mock_sub_",
                                          delete=False, mode="w",
                                          encoding="utf-8")
        ass.write(generate_ass_subtitles(
            clip["subtitle_words"], play_res_x=width, play_res_y=height,
            position=subtitle_position,
            font_size_pct=subtitle_font_size_pct if subtitle_font_size_pct is not None else 3.2))
        ass.close()
        filters.append(f"ass={_escape_ass_path(ass.name)}")
    if title and clip.get("title"):
        tass = tempfile.NamedTemporaryFile(suffix=".ass", prefix="mock_title_",
                                           delete=False, mode="w",
                                           encoding="utf-8")
        tass.write(generate_title_overlay(
            clip["title"], play_res_x=width, play_res_y=height, duration=3.0))
        tass.close()
        filters.append(f"ass={_escape_ass_path(tass.name)}")

    out_name = f"rank{rank:02d}_mock_preview_final.mp4"
    out_path = output_dir / out_name
    cmd = [get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(tmp_bg)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", str(out_path)]
    try:
        import subprocess
        subprocess.run(cmd, check=True)
    except Exception as e:  # pragma: no cover - environment issue
        print(f"[mock] burn failed for clip {rank}: {e}")
        return None
    finally:
        try:
            tmp_bg.unlink()
        except OSError:
            pass
    return out_name if out_path.exists() else None


def _run(job: Job) -> None:
    """Background worker: build mock clips and register them on the Job."""
    opts = job.options
    try:
        orientation = str(opts.get("orientation", "auto"))
        portrait = orientation in ("portrait", "vertical")
        if portrait:
            width, height = 1080, 1920
        elif orientation in ("landscape", "horizontal"):
            width, height = 1920, 1080
        else:
            width, height = 1080, 1920  # default portrait, matches the form

        output_dir = Path(opts.get("output_dir", "clips")) / "mock_preview"
        output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(output_dir)

        job.set_step("upload", "mock preview (no upload)")
        job.complete_step("upload", "mock")

        # Steps 2-7 are surfaced as instantaneous "preview" steps so the
        # stepper reflects the real flow without running the heavy pipeline.
        job.set_step("transcribe", "mock transcript")
        job.complete_step("transcribe", "8 sample segments (mock)")
        job.set_step("prefilter", "mock prefilter")
        job.complete_step("prefilter", "kept (mock)")
        job.set_step("energy", "mock energy")
        job.complete_step("energy", "3 peaks (mock)")
        job.set_step("select_refine", "mock select & refine")
        job.complete_step("select_refine", f"{len(SAMPLE_CLIPS)} clips (mock)")
        job.set_step("build", "Building mock clips…")
        for clip in SAMPLE_CLIPS:
            fname = _burn_one(
                clip, output_dir,
                subtitles=bool(opts.get("subtitles", True)),
                title=bool(opts.get("title", False)),
                orientation=orientation,
                subtitle_position=opts.get("subtitle_position", "lower"),
                subtitle_font_size_pct=_opt_float(opts, "subtitle_font_size_pct"),
                width=width, height=height,
            )
            if fname:
                fpath = output_dir / fname
                job.outputs.append({
                    "filename": fname,
                    "path": str(fpath),
                    "rank": clip.get("rank"),
                    "title": clip.get("title"),
                    "topic": clip.get("topic"),
                    "caption": clip.get("caption"),
                    "score": clip.get("score"),
                    "start": clip.get("start"),
                    "end": clip.get("end"),
                    "size_mb": round(fpath.stat().st_size / 1_048_576, 1),
                })
        job.complete_step("build",
                          f"{len(job.outputs)} mock clips built")

        job.clips_meta = [
            {k: v for k, v in c.items() if not k.startswith("_")}
            for c in SAMPLE_CLIPS
        ]
        job.set_step("done", f"{len(job.outputs)} mock clips ready")
        job.finish()
    except Exception as e:  # noqa: BLE001
        import traceback
        job.add_log(traceback.format_exc(), "ERROR")
        job.fail(str(e))


def build_mock_job(options: dict | None = None) -> Job:
    """Create a mock-preview Job, register it, and start it in the background.

    Returns the Job immediately (the dashboard polls /api/job/<id> for progress).
    """
    job_id = f"{int(time.time()*1000)}-mock"
    base_opts = {
        "orientation": "portrait",
        "subtitles": True,
        "title": False,
        "subtitle_position": "lower",
        "output_dir": str(Path(__file__).resolve().parent.parent / "clips"),
    }
    if options:
        base_opts.update(options)
    job = Job(job_id, "<mock>", base_opts)
    with _JOBS_LOCK:
        JOBS[job_id] = job
    t = threading.Thread(target=_run, args=(job,), daemon=True)
    t.start()
    return job


# ── Single-video dashboard variant ─────────────────────────────────────────
# Reuses the single_video_runner Job registry + output schema so the existing
# single.html / single.js pipeline (stepper + /files/single/<id>/<file>) works
# unchanged. The preview burns the SAME subtitle ASS the real render uses.
SINGLE_SAMPLE_WORDS: list[dict[str, Any]] = SAMPLE_WORDS
SINGLE_LANGUAGE = "id"


def _run_single(job: SingleJob) -> None:
    """Single-video mock: one subtitled sample clip, registered as a real Job."""
    opts = job.options
    try:
        out_dir = Path(opts.get("output_dir", "clips")) / "mock_preview_single"
        out_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(out_dir)

        job.set_step("upload", "mock preview (no upload)")
        job.complete_step("upload", "mock")
        job.set_step("transcribe", "mock transcript")
        job.complete_step("transcribe", "8 sample segments (mock)")
        job.set_step("subtitles", "mock subtitle words")
        job.complete_step("subtitles",
                          f"{len(SINGLE_SAMPLE_WORDS)} words (mock)")

        job.set_step("render", "Burning mock subtitles…")
        clip = {
            "rank": 1,
            "title": "Contoh Subtitle — AI Video Clipper",
            "subtitle_words": SINGLE_SAMPLE_WORDS,
        }
        # Single dashboard always burns subtitles; position is honoured.
        fname = _burn_one(
            clip, out_dir,
            subtitles=True,
            title=False,
            orientation="auto",
            subtitle_position=opts.get("subtitle_position", "lower"),
            subtitle_font_size_pct=_opt_float(opts, "subtitle_font_size_pct"),
            width=1080, height=1920,
        )
        if fname:
            fpath = out_dir / fname
            job.outputs.append({
                "filename": fname,
                "path": str(fpath),
                "size_mb": round(fpath.stat().st_size / 1_048_576, 1),
                "translated": bool(opts.get("translated", False)),
                "language": SINGLE_LANGUAGE,
            })
        job.complete_step("render",
                          f"{len(job.outputs)} mock video rendered")
        job.set_step("done", "Mock subtitled video ready")
        job.finish()
    except Exception as e:  # noqa: BLE001
        import traceback
        job.add_log(traceback.format_exc(), "ERROR")
        job.fail(str(e))


def build_single_mock_job(options: dict | None = None) -> SingleJob:
    """Create a single-video mock Job into the single_video_runner registry."""
    job_id = f"sv-mock-{int(time.time()*1000)}"
    base_opts = {
        "subtitle_position": "lower",
        "translated": False,
        "output_dir": str(Path(__file__).resolve().parent.parent / "clips"),
    }
    if options:
        base_opts.update(options)
    job = SingleJob(job_id, "<mock-single>", base_opts)
    with _SINGLE_LOCK:
        SINGLE_JOBS[job_id] = job
    t = threading.Thread(target=_run_single, args=(job,), daemon=True)
    t.start()
    return job
