"""
Per-video, per-step pipeline cache for the web dashboard.

Every pipeline step that is expensive (or that the user may want to replay
without recomputing) is cached under a predictable folder:

    <clips_root>/<video_stem>/.cache/<step>.json      (structured result)
    <clips_root>/<video_stem>/.cache/<step>.files.json (list of output files)

Caching is *opt-in per step* and keyed by a stable signature of the inputs that
affect that step. When the signature matches a stored cache, the step is
short-circuited and the dashboard shows a "cached" badge instead of re-running
it.

The whole cache for one video (or a single step) can be cleared from the
dashboard via the web API, so nothing in the pipeline is ever *silently*
stuck on stale data — clearing is an explicit, user-initiated action.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# Ordered pipeline steps (mirrors web_runner.STEPS minus the synthetic
# upload/done markers). These are the steps we support caching for.
CACHEABLE_STEPS = [
    "transcribe",   # transcript segments (already cached elsewhere, kept for status)
    "prefilter",
    "energy",
    "llm_select",
    "refine",
    "extract",
    "render",
]


def cache_dir_for(video_path: str, clips_root: str) -> Path:
    """The .cache folder lives next to the clip outputs for a video.

    ``clips_root`` is the top-level clips directory (e.g. ``clips/``); the
    per-video folder is ``<clips_root>/<video_stem>/`` and the cache lives in
    ``<clips_root>/<video_stem>/.cache/``.
    """
    video = Path(video_path)
    return Path(clips_root) / video.stem / ".cache"


def transcript_cache_path(video_path: str) -> Path:
    """Global transcript cache path (mirrors sosmed.cli._get_transcript_cache_path)."""
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{video.stem}_transcript.json"


def signature(*parts: Any) -> str:
    """Short stable signature of the inputs that affect a step."""
    return _sig(*parts)


def is_cached(cdir: Path, step: str, *sig_parts: Any) -> bool:
    """True if a live cache entry exists for `step` with these inputs."""
    return get_cached(cdir, step, *sig_parts) is not None


def _sig(*parts: Any) -> str:
    """Short stable signature of the inputs that affect a step."""
    payload = json.dumps([_jsonable(p) for p in parts], sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _jsonable(p: Any) -> Any:
    if isinstance(p, Path):
        return str(p)
    if isinstance(p, dict):
        return {k: _jsonable(v) for k, v in p.items()}
    if isinstance(p, (list, tuple)):
        return [_jsonable(v) for v in p]
    return p


def _step_path(cdir: Path, step: str) -> Path:
    return cdir / f"{step}.json"


def _files_path(cdir: Path, step: str) -> Path:
    return cdir / f"{step}.files.json"


def _read_sig(cdir: Path, step: str) -> str | None:
    p = _step_path(cdir, step)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("_sig")
    except Exception:
        return None


def _write(cdir: Path, step: str, payload: dict, files: list[str] | None) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    _step_path(cdir, step).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if files is not None:
        _files_path(cdir, step).write_text(
            json.dumps({"files": files}, ensure_ascii=False), encoding="utf-8")


def _is_signature(value: Any) -> bool:
    """True if `value` is a single pre-computed 12-char hex signature.

    Callers (e.g. web_runner) sometimes pre-compute ``pc.signature(...)`` and
    pass that string directly to get_cached/put_cached. In that case we must
    NOT re-hash it (doing so double-hashes and silently breaks cache
    matching). Treat a lone 12-char lowercase-hex string as already a sig.
    """
    return (
        isinstance(value, str)
        and len(value) == 12
        and all(c in "0123456789abcdef" for c in value)
    )


def get_cached(cdir: Path, step: str, *sig_parts: Any) -> dict | None:
    """Return the cached payload for `step` if the signature matches.

    Returns None on miss / corruption / signature mismatch.
    """
    if not cdir.exists():
        return None
    # If a single pre-computed signature string was passed, use it directly
    # instead of re-hashing the hash (which would double-hash and break matching).
    if len(sig_parts) == 1 and _is_signature(sig_parts[0]):
        sig = sig_parts[0]
    else:
        sig = _sig(*sig_parts)
    if _read_sig(cdir, step) != sig:
        return None
    p = _step_path(cdir, step)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data.pop("_sig", None)
        return data
    except Exception:
        return None


def put_cached(cdir: Path, step: str, payload: dict,
               files: list[str] | None = None,
               sig_parts: list[Any] | tuple[Any, ...] | None = None) -> None:
    """Persist `payload` (and optional output file list) for `step`."""
    if (sig_parts is not None and len(sig_parts) == 1
            and _is_signature(sig_parts[0])):
        sig = sig_parts[0]
    else:
        sig = _sig(*(sig_parts or []))
    data = dict(payload)
    data["_sig"] = sig
    _write(cdir, step, data, files)


def cached_files(cdir: Path, step: str) -> list[str]:
    """Absolute paths of output files recorded for `step`."""
    p = _files_path(cdir, step)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("files", [])
    except Exception:
        return []


def cache_status(cdir: Path, signatures: dict[str, str]) -> dict[str, bool]:
    """Return {step: is_cached} given a {step: signature} map."""
    out: dict[str, bool] = {}
    for step in CACHEABLE_STEPS:
        sig = signatures.get(step)
        if sig is None:
            out[step] = False
            continue
        out[step] = (_read_sig(cdir, step) == sig)
    return out


def clear_cache(cdir: Path) -> int:
    """Delete the entire per-video .cache folder. Returns #files removed."""
    if not cdir.exists():
        return 0
    n = 0
    for f in cdir.iterdir():
        if f.is_file():
            f.unlink()
            n += 1
    try:
        cdir.rmdir()
    except OSError:
        pass
    return n


def clear_step(cdir: Path, step: str) -> int:
    """Delete one step's cache files. Returns #files removed (0 or 1/2)."""
    if step not in CACHEABLE_STEPS:
        return 0
    n = 0
    for p in (_step_path(cdir, step), _files_path(cdir, step)):
        if p.exists():
            p.unlink()
            n += 1
    return n


def touch(cdir: Path) -> None:
    """Ensure the cache dir exists (called when a fresh run starts)."""
    cdir.mkdir(parents=True, exist_ok=True)
