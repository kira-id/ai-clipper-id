#!/usr/bin/env python3
"""
AI Video Clipper — Single-Video Subtitle Dashboard (zero external deps).

A focused, simple dashboard: upload ONE video, add subtitles (with or without
translation), get back one subtitled video. Reuses the same word-timestamp +
ASS burning primitives as the multi-clip pipeline so the look is identical.

Run:
    python single_video_app.py
    python single_video_app.py --port 8081 --host 0.0.0.0

Then open  http://localhost:8081/single

No Flask/eventlet needed — pure stdlib http.server. The pipeline runs in a
background thread; the browser polls /api/single/job/<id> for live state.

The multi-clip dashboard (web_app.py) serves http://localhost:8000/ and this
app serves http://localhost:8081/single — run both side by side, or run only
this one if you only need subtitles.
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sosmed.single_video_runner import run_single, get_job, JOBS  # noqa: E402

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MIME_OVERRIDES = {".js": "text/javascript", ".mjs": "text/javascript"}


# ── multipart upload (stdlib, no external deps) ───────────────────────────
def parse_multipart(body: bytes, boundary: str):
    """Return (file_part, form) like web_app.py.

    file_part = (name, filename, data) or None
    form = {field_name: value}
    """
    delim = (b"--" + boundary.encode()).rstrip(b"--")
    file_part = None
    form: dict[str, str] = {}
    for part in body.split(delim):
        if not part or part in (b"--\r\n", b"\r\n", b"--"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if b"\r\n\r\n" not in part:
            continue
        header, _, content = part.partition(b"\r\n\r\n")
        header_text = header.decode("utf-8", "replace")
        m = re.findall(r'Content-Disposition: form-data; name="([^"]+)"'
                       r'(?:; filename="([^"]*)")?', header_text)
        if not m:
            continue
        name, filename = m[0]
        if content.endswith(b"\r\n"):
            content = content[:-2]
        if filename:
            file_part = (name, filename, content)
        else:
            form[name] = content.decode("utf-8", "replace")
    return file_part, form


# ── request handler ────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "AIClipperSingle/1.0"

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
        sys.stderr.write("  [single-web] " + (fmt % args) + "\n")

    # ── GET ──
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/single" or path == "/single/":
            return self._serve_file(ROOT / "dashboard" / "single.html",
                                    "text/html; charset=utf-8")

        if path == "/health":
            return self._send_json({"ok": True, "jobs": len(JOBS)})

        if path.startswith("/api/single/job/"):
            # urlparse does NOT percent-decode, so a job id from a filename
            # with spaces arrives as "…V1%20.mp4"; decode before the lookup.
            job_id = urllib.parse.unquote(path.split("/")[-1])
            job = get_job(job_id)
            if not job:
                return self._send_json({"error": "job not found"}, 404)
            return self._send_json(job.to_dict())

        # /api/single/jobs -> recent list
        if path.startswith("/api/single/jobs"):
            with _jobs_lock:
                return self._send_json({
                    "jobs": [j.to_dict() for j in list(JOBS.values())[-20:]]
                })

        # /files/single/<job_id>/<filename> -> stream the rendered video
        if path.startswith("/files/single/"):
            _, _, rest = path.partition("/files/single/")
            job_id, _, fname = rest.partition("/")
            job_id = urllib.parse.unquote(job_id)
            job = get_job(job_id)
            if not job or not job.output_dir:
                return self._send_json({"error": "not found"}, 404)
            fpath = Path(job.output_dir) / urllib.parse.unquote(fname)
            root = Path(job.output_dir).resolve()
            if (not fpath.exists()
                    or not str(fpath.resolve()).startswith(str(root))):
                return self._send_json({"error": "file not found"}, 404)
            return self._serve_file_stream(fpath)

        # static assets under dashboard/
        if path.startswith("/dashboard/"):
            fpath = ROOT / urllib.parse.unquote(path.lstrip("/"))
            if fpath.exists() and fpath.is_file():
                ctype = MIME_OVERRIDES.get(
                    fpath.suffix,
                    mimetypes.guess_type(str(fpath))[0]
                    or "application/octet-stream")
                return self._serve_file(fpath, ctype)
            return self._send_json({"error": "not found"}, 404)

        return self._send_json({"error": "not found"}, 404)

    # ── POST ──
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # ── MOCK PREVIEW (single dashboard) ──────────────────────────────
        # Render a sample subtitled clip via the REAL ASS burning code so you
        # can verify the subtitle look/position WITHOUT a real upload or API
        # key. POST /api/single/mock (form or JSON). Mirrors /api/single/process.
        if path == "/api/single/mock":
            from sosmed.mock import build_single_mock_job
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
                        _, form = parse_multipart(
                            self.rfile.read(length),
                            self.headers.get("Content-Type", "")
                            .split("boundary=")[-1].strip().strip('"'))
                        mock_opts = dict(form)
                    except Exception:
                        mock_opts = {}
            job = build_single_mock_job({
                "subtitle_position": mock_opts.get("subtitle_position", "lower"),
                "subtitle_font_size_pct": mock_opts.get("subtitle_font_size_pct", ""),
                "translated": str(mock_opts.get("subtitle_mode", "original"))
                == "translate",
            })
            return self._send_json({"job_id": job.id, "state": job.to_dict()})

        if path != "/api/single/process":
            return self._send_json({"error": "unknown endpoint"}, 404)

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._send_json(
                {"error": "expected multipart/form-data"}, 400)

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        boundary = ctype.split("boundary=")[-1].strip().strip('"')
        file_part, form = parse_multipart(body, boundary)

        # ── LOCAL FILE SHORTCUT ──────────────────────────────────────────
        # When the dashboard and the video live on the SAME machine, re-uploading
        # a big file wastes time and disk. If the form carries an absolute
        # `local_path`, we read that file DIRECTLY instead of saving an upload
        # copy. Falls back to a normal upload when no local_path is supplied.
        local_path = (form.get("local_path") or "").strip()
        if local_path:
            lp = Path(local_path)
            if not lp.exists() or not lp.is_file():
                return self._send_json(
                    {"error": f"local file not found: {local_path}"}, 400)
            dest = str(lp)
            filename = lp.name
            orig_stem = lp.stem
        elif file_part:
            _, filename, data = file_part
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
            dest = UPLOAD_DIR / f"{int(time.time()*1000)}_{safe_name}"
            dest.write_bytes(data)
            orig_stem = Path(filename).stem
        else:
            return self._send_json({"error": "no video file in upload"}, 400)

        _lang_opt = form.get("lang", "auto").lower()
        _lang = None if _lang_opt in ("auto", "none") else _lang_opt

        if _lang == "auto":
            _lang = None

        options = {
            "model": form.get("model", "tiny"),
            "lang": _lang,
            "device": form.get("device", "auto"),
            "compute_type": form.get("compute_type", "auto"),
            "vad_min_silence": int(form.get("vad_min_silence", 400)),
            "vad_speech_pad": int(form.get("vad_speech_pad", 200)),
            "batch": int(form.get("batch", 16)),
            "subtitle_mode": form.get("subtitle_mode", "original"),
            "subtitle_position": form.get("subtitle_position", "lower"),
            "subtitle_font_size_pct": form.get("subtitle_font_size_pct", ""),
            "subtitle_margin_pct": float(form.get("subtitle_margin_pct", 25.0))
            if form.get("subtitle_margin_pct") else None,
            "no_cache": form.get("no_cache", "off") == "on",
            "remove_silence": form.get("remove_silence", "off") == "on",
            "max_silence": float(form.get("max_silence", 1.5)),
            "target_language": form.get("target_language", "en") or "en",
            "encoding_preset": form.get("encoding_preset", "veryfast"),
            "encoding_crf": int(form.get("encoding_crf", 23)),
            "output_dir": str(ROOT / "clips"),
            "video_name": orig_stem,
            "llm_model": form.get("llm_model") or None,
            "api_key": form.get("api_key") or None,
        }

        job = run_single(str(dest), options)
        return self._send_json({"job_id": job.id, "state": job.to_dict()})

    # ── file serving ──
    def _serve_file(self, fpath: Path, ctype: str):
        if not fpath.exists():
            return self._send_json({"error": "not found"}, 404)
        data = fpath.read_bytes()
        self._send(200, data, ctype)

    def _serve_file_stream(self, fpath: Path):
        size = fpath.stat().st_size
        ctype = MIME_OVERRIDES.get(
            fpath.suffix,
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
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
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


_jobs_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser(description="AI Clipper — Single-Video Dashboard")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}/single"
    print(f"\n  AI Video Clipper — Single-Video Subtitle Dashboard")
    print(f"  ───────────────────────────────────────────────────")
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
