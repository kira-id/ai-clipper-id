#!/usr/bin/env python3
"""
AI Video Clipper — Web Dashboard (zero external dependencies).

Serves dashboard/index.html, accepts a video upload, runs the clip pipeline
via sosmed.web_runner, and streams step-by-step progress + output files.

Run:
    python web_app.py
    python web_app.py --port 8080 --host 0.0.0.0

Then open  http://localhost:8000

No Flask/eventlet needed — pure stdlib http.server. Long-running steps run in
background threads; the browser polls /api/job/<id> for live state.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Allow running from project root (python web_app.py) or via module.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sosmed.web_runner import run_job, rerun_job, get_job, JOBS  # noqa: E402

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# MIME overrides for browsers that choke on uncommon types
MIME_OVERRIDES = {".js": "text/javascript", ".mjs": "text/javascript"}


# ── options builder (shared by /api/process) ─────────────────────────────
def build_options(form: dict) -> dict:
    """Translate the upload form fields into the pipeline options dict."""
    return {
        "model": form.get("model", "tiny"),
        # Normalize 'auto'/'none' -> None so faster-whisper auto-detects
        # the language (it rejects the literal string 'auto').
        "lang": (None if str(form.get("lang", "auto")).strip().lower()
                 in ("auto", "none", "") else form.get("lang", "auto")),
        "min": int(form.get("min", 15)),
        "max": int(form.get("max", 60)),
        "max_clips": int(form.get("max_clips", 28)),
        "min_score": int(form.get("min_score", 55)),
        "device": form.get("device", "auto"),
        "compute_type": form.get("compute_type", "auto"),
        "vad_min_silence": int(form.get("vad_min_silence", 400)),
        "vad_speech_pad": int(form.get("vad_speech_pad", 200)),
        "batch": int(form.get("batch", 16)),
        "chunk_duration": float(form.get("chunk_duration", 360.0)),
        "chunk_overlap": float(form.get("chunk_overlap", 60.0)),
        "subtitles": form.get("subtitles", "on") == "on",
        "subtitle_position": form.get("subtitle_position", "lower"),
        "subtitle_font_size_pct": form.get("subtitle_font_size_pct", ""),
        "title": form.get("title", "off") == "on",
        "orientation": form.get("orientation", "auto"),
        "crop": form.get("crop", "off") == "on",
        "split_screen": form.get("split_screen", "off") == "on",
        "active_speaker": form.get("active_speaker", "on") == "on",
        "remove_silence": form.get("remove_silence", "off") == "on",
        "max_silence": float(form.get("max_silence", 1.5)),
        "cta": form.get("cta", "off") == "on",
        "target_language": form.get("target_language", "en") or "en",
        "encoding_preset": form.get("encoding_preset", "veryfast"),
        "encoding_crf": int(form.get("encoding_crf", 23)),
        "output_dir": str(ROOT / "clips"),
        "video_name": form.get("video_name", ""),
        "llm_model": form.get("llm_model") or None,
        "api_key": form.get("api_key") or None,
        "system_prompt": form.get("system_prompt") or None,
    }


# ── multipart upload (stdlib, no external deps) ───────────────────────────
def parse_multipart(body: bytes, boundary: str):
    """Return {field_name: (filename, data)} for the first file part.

    Minimal parser good enough for our single-file upload form.
    """
    delim = (b"--" + boundary.encode()).rstrip(b"--")
    parts = body.split(delim)
    file_part = None
    form: dict[str, str] = {}
    for part in parts:
        if not part or part in (b"--\r\n", b"\r\n", b"--"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        # header / body split
        if b"\r\n\r\n" not in part:
            continue
        header, _, content = part.partition(b"\r\n\r\n")
        header_text = header.decode("utf-8", "replace")
        m = re.findall(r'Content-Disposition: form-data; name="([^"]+)"'
                       r'(?:; filename="([^"]*)")?', header_text)
        if not m:
            continue
        name, filename = m[0]
        # strip trailing boundary CRLF on content
        if content.endswith(b"\r\n"):
            content = content[:-2]
        if filename:
            file_part = (name, filename, content)
        else:
            form[name] = content.decode("utf-8", "replace")
    return file_part, form


# ── request handler ────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "AIClipperDashboard/1.0"

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"),
                   ctype="application/json")

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("  [web] " + (fmt % args) + "\n")

    # ── GET ──
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self._serve_file(ROOT / "dashboard" / "index.html",
                                    "text/html; charset=utf-8")
        if path == "/health":
            return self._send_json({"ok": True, "jobs": len(JOBS)})

        if path == "/api/prompt-default":
            from sosmed.utils import SYSTEM_PROMPT
            return self._send_json({"prompt": SYSTEM_PROMPT})

        if path.startswith("/api/job/"):
            # urlparse does NOT percent-decode, so a job id derived from a
            # filename with spaces arrives as "…V1%20.mp4"; decode it before
            # the dict lookup or get_job() never matches (dashboard hangs).
            job_id = urllib.parse.unquote(path.split("/")[-1])
            job = get_job(job_id)
            if not job:
                return self._send_json({"error": "job not found"}, 404)
            return self._send_json(job.to_dict())

        if path.startswith("/api/jobs"):
            with _jobs_lock:
                return self._send_json({
                    "jobs": [j.to_dict() for j in list(JOBS.values())[-20:]]
                })

        if path.startswith("/files/"):
            # /files/<job_id>/<filename>  -> stream the output clip
            _, _, rest = path.partition("/files/")
            job_id, _, fname = rest.partition("/")
            job_id = urllib.parse.unquote(job_id)
            job = get_job(job_id)
            if not job or not job.output_dir:
                return self._send_json({"error": "not found"}, 404)
            fpath = Path(job.output_dir) / urllib.parse.unquote(fname)
            if not fpath.exists() or not str(fpath.resolve()).startswith(
                    str(Path(job.output_dir).resolve())):
                return self._send_json({"error": "file not found"}, 404)
            return self._serve_file_stream(fpath)

        if path.startswith("/api/download-all/"):
            # /api/download-all/<job_id>  -> zip every file in the job's
            # output folder and stream it as one archive.
            job_id = urllib.parse.unquote(path.split("/")[-1])
            job = get_job(job_id)
            if not job or not job.output_dir:
                return self._send_json({"error": "not found"}, 404)
            return self._serve_zip(job)

        # static assets under dashboard/
        if path.startswith("/dashboard/"):
            fpath = ROOT / urllib.parse.unquote(path.lstrip("/"))
            if fpath.exists() and fpath.is_file():
                ctype = MIME_OVERRIDES.get(fpath.suffix,
                                           mimetypes.guess_type(str(fpath))[0]
                                           or "application/octet-stream")
                return self._serve_file(fpath, ctype)
            return self._send_json({"error": "not found"}, 404)

        return self._send_json({"error": "not found"}, 404)

    # ── POST ──
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # ── CLEAR CACHE: remove a job's cached pipeline steps ──
        #   POST /api/clear-cache/<job_id>            -> clear all steps
        #   POST /api/clear-cache/<job_id>/<step>      -> clear one step
        if path.startswith("/api/clear-cache/"):
            parts = path.split("/")
            # ["", "api", "clear-cache", "<job_id>", ("<step>")?]
            job_id = urllib.parse.unquote(parts[3]) if len(parts) > 3 else ""
            step = parts[4] if len(parts) > 4 else None
            from sosmed.web_runner import clear_job_cache
            n = clear_job_cache(job_id, step)
            return self._send_json({
                "cleared": n,
                "job_id": job_id,
                "step": step,
                "ok": True if n >= 0 else False,
            })

        # ── RETRY: re-run a previous job's same upload with relaxed params ──
        if path.startswith("/api/retry/"):
            job_id = urllib.parse.unquote(path.split("/")[-1])
            old = get_job(job_id)
            if not old:
                return self._send_json({"error": "job not found"}, 404)
            # explicit, user-initiated relaxed defaults (no silent fallback)
            body = b""
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    body = self.rfile.read(length)
            except Exception:
                body = b""
            overrides: dict = {}
            if body:
                try:
                    overrides = json.loads(body) or {}
                except Exception:
                    overrides = {}
            job = rerun_job(job_id, overrides)
            if not job:
                return self._send_json(
                    {"error": "original upload no longer available"}, 410)
            return self._send_json({"job_id": job.id, "state": job.to_dict()})

        # ── MOCK PREVIEW: render a sample clip with the REAL overlay code ──
        # Lets you verify the dashboard (title / subtitle / position / topic)
        # WITHOUT running the full pipeline. POST /api/mock (JSON or form ok).
        if path == "/api/mock":
            from sosmed.mock import build_mock_job
            mock_opts: dict = {}
            if self.headers.get("Content-Type", "").startswith("application/json"):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if length:
                        mock_opts = json.loads(self.rfile.read(length)) or {}
                except Exception:
                    mock_opts = {}
            else:
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    try:
                        _, form = parse_multipart(self.rfile.read(length),
                                                  self.headers.get("Content-Type", "")
                                                  .split("boundary=")[-1].strip().strip('"'))
                        mock_opts = dict(form)
                    except Exception:
                        mock_opts = {}
            # Only the render-affecting options are honoured for the preview.
            job = build_mock_job({
                "orientation": mock_opts.get("orientation", "auto"),
                "subtitles": str(mock_opts.get("subtitles", "on")) == "on",
                "title": str(mock_opts.get("title", "off")) == "on",
                "subtitle_position": mock_opts.get("subtitle_position", "lower"),
                "subtitle_font_size_pct": mock_opts.get("subtitle_font_size_pct", ""),
            })
            return self._send_json({"job_id": job.id, "state": job.to_dict()})

        if path != "/api/process":
            return self._send_json({"error": "unknown endpoint"}, 404)

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._send_json({"error": "expected multipart/form-data"},
                                   400)

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        boundary = ctype.split("boundary=")[-1].strip().strip('"')
        file_part, form = parse_multipart(body, boundary)

        # ── LOCAL FILE SHORTCUT ──────────────────────────────────────────
        # When the dashboard and the video live on the SAME machine (the usual
        # case for a big local file), re-uploading a copy wastes time and disk.
        # If the form carries an absolute `local_path`, we read that file
        # DIRECTLY instead of saving an uploaded copy. Falls back to a normal
        # upload when no local_path is supplied.
        local_path = (form.get("local_path") or "").strip()
        direct = False
        if local_path:
            lp = Path(local_path)
            if not lp.exists() or not lp.is_file():
                return self._send_json(
                    {"error": f"local file not found: {local_path}"}, 400)
            dest = str(lp)
            filename = lp.name
            orig_stem = lp.stem
            direct = True
            source_detail = f"local: {lp.name}"
        elif file_part:
            # save upload (timestamp-prefixed to avoid collisions in uploads/)
            _, filename, data = file_part
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
            dest = UPLOAD_DIR / f"{int(time.time()*1000)}_{safe_name}"
            dest.write_bytes(data)
            # output folder should mirror the ORIGINAL video name (not the
            # timestamp-prefixed upload) so clips/<VideoName>/ is predictable.
            orig_stem = Path(filename).stem
            source_detail = f"upload: {filename}"
        else:
            return self._send_json({"error": "no video file in upload"}, 400)

        # build options from form
        options = build_options(form)
        options["video_name"] = orig_stem

        job = run_job(str(dest), options)
        with _jobs_lock:
            pass
        return self._send_json({"job_id": job.id, "state": job.to_dict()})

    # ── file serving ──
    def _serve_file(self, fpath: Path, ctype: str):
        if not fpath.exists():
            return self._send_json({"error": "not found"}, 404)
        data = fpath.read_bytes()
        self._send(200, data, ctype)

    def _serve_file_stream(self, fpath: Path):
        size = fpath.stat().st_size
        ctype = MIME_OVERRIDES.get(fpath.suffix,
                                  mimetypes.guess_type(str(fpath))[0]
                                  or "application/octet-stream")
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):].split(",")[0].strip()
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            length = end - start + 1
            with open(fpath, "rb") as f:
                f.seek(start)
                chunk = f.read(length)
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(chunk)
            return
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_zip(self, job):
        """Stream a zip of every file in the job's output folder."""
        import io
        out_dir = Path(job.output_dir)
        files = sorted(p for p in out_dir.iterdir()
                       if p.is_file() and not p.name.endswith(".zip"))
        if not files:
            return self._send_json({"error": "no output files to zip"}, 404)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        data = buf.getvalue()
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", job.id)
        fname = f"clips_{safe_id}.zip"
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition",
                         f'attachment; filename="{fname}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_jobs_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser(description="AI Clipper Web Dashboard")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  AI Video Clipper — Dashboard")
    print(f"  ─────────────────────────────")
    print(f"  Open: {url}")
    print(f"  Uploads: {UPLOAD_DIR}")
    print(f"  Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down…")
        server.shutdown()


if __name__ == "__main__":
    main()
