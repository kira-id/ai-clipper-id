"""
Audio energy analysis: detect emotional moments from loudness spikes.

Gaming livestreams have emotional moments (screams, laughter, excitement,
fear reactions) that show up as sudden loudness changes — even when there's
no coherent speech for Whisper to transcribe.

This module extracts per-window RMS energy from the audio track and
identifies spikes relative to a local baseline.  All detection is
**relative** (ratio to local median) so results are independent of
recording volume / mic gain.
"""

import subprocess
import math
from typing import Any

from .utils import log

# Minimum RMS below which we consider the audio effectively silent.
# 16-bit PCM range is -32768..32767; RMS 50 ≈ -50 dBFS — well into noise floor.
_SILENCE_FLOOR = 50.0


def _build_speech_regions(
    segments: list[dict[str, Any]] | None,
    max_gap: float = 0.45,
    min_duration: float = 0.12,
) -> list[tuple[float, float]]:
    """Build merged speech regions from transcript segments.

    Regions are derived from segment start/end timestamps and merged when the
    silence gap between adjacent regions is small.
    """
    if not segments:
        return []

    spans: list[tuple[float, float]] = []
    for seg in segments:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append((start, end))

    if not spans:
        return []

    spans.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = spans[0]

    for start, end in spans[1:]:
        if start - cur_end <= max_gap:
            cur_end = max(cur_end, end)
            continue
        if cur_end - cur_start >= min_duration:
            merged.append((round(cur_start, 3), round(cur_end, 3)))
        cur_start, cur_end = start, end

    if cur_end - cur_start >= min_duration:
        merged.append((round(cur_start, 3), round(cur_end, 3)))

    return merged


def _extract_pcm(video_path: str, sample_rate: int = 16000) -> bytes:
    """Extract mono 16-bit PCM audio from video via ffmpeg.

    A speech-focused band-pass filter (roughly 120-4000 Hz) is applied to
    reduce background music influence and keep vocal energy dominant.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn",                     # no video
        "-af", "highpass=f=120,lowpass=f=4000",
        "-ac", "1",                # mono
        "-ar", str(sample_rate),   # resample
        "-f", "s16le",             # raw 16-bit little-endian PCM
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        log("WARN", "ffmpeg PCM extraction timed out (10 min)")
        return b""
    except FileNotFoundError:
        log("ERROR", "ffmpeg not found — cannot analyze audio energy")
        return b""
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:200]
        log("ERROR", f"ffmpeg PCM extraction failed: {stderr}")
        return b""
    return result.stdout


def _compute_rms_windows(
    pcm: bytes,
    sample_rate: int = 16000,
    window_sec: float = 0.5,
) -> list[float]:
    """
    Compute RMS energy for fixed-size windows using numpy for speed.

    Returns a plain list of RMS floats, one per window.
    Window *i* covers time [i*window_sec, (i+1)*window_sec).
    """
    try:
        import numpy as np
    except ImportError:
        log("WARN", "numpy not installed — falling back to pure-python RMS (slow)")
        return _compute_rms_windows_pure(pcm, sample_rate, window_sec)

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    samples_per_window = int(sample_rate * window_sec)
    if samples_per_window == 0:
        return []

    # Trim to whole windows
    n_windows = len(samples) // samples_per_window
    if n_windows == 0:
        return []
    trimmed = samples[: n_windows * samples_per_window]
    blocks = trimmed.reshape(n_windows, samples_per_window)
    rms_arr = np.sqrt(np.mean(blocks ** 2, axis=1))
    return rms_arr.tolist()


def _compute_rms_windows_pure(
    pcm: bytes,
    sample_rate: int = 16000,
    window_sec: float = 0.5,
) -> list[float]:
    """Pure-python fallback for RMS computation (no numpy)."""
    import struct

    samples_per_window = int(sample_rate * window_sec)
    bytes_per_window = samples_per_window * 2
    windows: list[float] = []
    offset = 0

    while offset + bytes_per_window <= len(pcm):
        chunk = pcm[offset: offset + bytes_per_window]
        samples = struct.unpack(f"<{samples_per_window}h", chunk)
        sum_sq = sum(s * s for s in samples)
        windows.append(math.sqrt(sum_sq / samples_per_window))
        offset += bytes_per_window

    return windows


def _detect_spikes(
    rms_values: list[float],
    window_sec: float = 0.5,
    local_radius: int = 20,
    spike_threshold: float = 2.0,
    high_energy_percentile: float = 80.0,
    speech_regions: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """
    Detect energy spikes — moments where RMS is significantly above local baseline.

    All detection is **relative** (ratio to local median), so recording
    volume / mic gain does not matter.

    Args:
        rms_values: Per-window RMS energy.
        window_sec: Duration of each window in seconds.
        local_radius: Number of windows on each side for local baseline.
        spike_threshold: Multiplier above local median to count as spike.
        high_energy_percentile: Percentile above which energy is "high".

    Returns list of {"time": float, "rms": float, "kind": str}.
    """
    n = len(rms_values)
    if n < 3:
        return []

    sorted_rms = sorted(rms_values)
    global_median = sorted_rms[n // 2]

    # If the entire audio is essentially silent, nothing to detect
    if global_median < _SILENCE_FLOOR:
        log("DEBUG", f"Audio mostly silent (median RMS {global_median:.0f}), skipping spike detection")
        return []

    # Percentile threshold for "sustained high energy"
    p_idx = min(int(n * high_energy_percentile / 100), n - 1)
    high_threshold = sorted_rms[p_idx]

    events: list[dict[str, Any]] = []

    speech_regions = speech_regions or []

    def _is_in_speech_region(t: float) -> bool:
        if not speech_regions:
            return False
        for start, end in speech_regions:
            if start <= t <= end:
                return True
        return False

    for i, rms in enumerate(rms_values):
        t = i * window_sec
        if not _is_in_speech_region(t):
            continue

        # Skip near-silent windows — no emotion here
        if rms < _SILENCE_FLOOR:
            continue

        # Local median baseline (more robust than mean against outliers)
        lo = max(0, i - local_radius)
        hi = min(n, i + local_radius + 1)
        neighborhood = sorted(rms_values[lo:hi])
        local_median = neighborhood[len(neighborhood) // 2]
        # Floor: don't let baseline drop below half the global median
        local_median = max(local_median, global_median * 0.5)

        # Spike: sudden jump above local baseline
        if rms > local_median * spike_threshold:
            events.append({"time": round(t, 2), "rms": round(rms, 1), "kind": "spike"})
        # Sustained high energy (top percentile AND meaningfully above median)
        elif rms >= high_threshold and rms > global_median * 1.5:
            events.append({"time": round(t, 2), "rms": round(rms, 1), "kind": "high"})

    return events


def _cluster_events(
    events: list[dict[str, Any]],
    window_sec: float = 0.5,
    merge_gap: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Merge nearby events into clusters with start/end times and peak info.

    Returns list of {"start": float, "end": float, "peak_rms": float, "kind": str}.
    """
    if not events:
        return []

    events = sorted(events, key=lambda e: e["time"])
    clusters: list[dict[str, Any]] = []

    cur = {
        "start": events[0]["time"],
        "end": round(events[0]["time"] + window_sec, 2),
        "peak_rms": events[0]["rms"],
        "has_spike": events[0]["kind"] == "spike",
    }

    for ev in events[1:]:
        if ev["time"] - cur["end"] <= merge_gap:
            cur["end"] = round(ev["time"] + window_sec, 2)
            cur["peak_rms"] = max(cur["peak_rms"], ev["rms"])
            if ev["kind"] == "spike":
                cur["has_spike"] = True
        else:
            clusters.append(cur)
            cur = {
                "start": ev["time"],
                "end": round(ev["time"] + window_sec, 2),
                "peak_rms": ev["rms"],
                "has_spike": ev["kind"] == "spike",
            }
    clusters.append(cur)

    # Label clusters
    for c in clusters:
        c["kind"] = "spike" if c["has_spike"] else "high"
        del c["has_spike"]

    return clusters


def analyze_audio_energy(
    video_path: str,
    segments: list[dict[str, Any]] | None = None,
    sample_rate: int = 16000,
    window_sec: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Analyze audio energy in a video and return clustered high-energy moments.

    Returns list of {"start": float, "end": float, "peak_rms": float, "kind": str}
    where kind is "spike" (sudden loudness jump — likely scream/reaction) or
    "high" (sustained high energy — likely excitement/action).

    Detection is volume-independent: uses ratio-to-local-median, so a quiet
    streamer who suddenly screams is detected the same as a loud one.

    Energy moments are only emitted inside transcript speech regions. This
    keeps background music from being treated as emotional peaks.
    """
    log("INFO", "Analyzing audio energy for emotional moments...")

    speech_regions = _build_speech_regions(segments)
    if not speech_regions:
        log("INFO", "No speech regions found in transcript — skipping energy analysis")
        return []

    pcm = _extract_pcm(video_path, sample_rate)
    if not pcm:
        log("WARN", "Could not extract audio — skipping energy analysis")
        return []

    # Sanity: warn on very large audio (>4h at 16kHz mono 16-bit ≈ 460 MB)
    pcm_mb = len(pcm) / (1024 * 1024)
    if pcm_mb > 500:
        log("WARN", f"Large audio ({pcm_mb:.0f} MB) — energy analysis may be slow")

    rms_values = _compute_rms_windows(pcm, sample_rate, window_sec)
    if not rms_values:
        log("WARN", "No audio windows computed — skipping energy analysis")
        return []

    log("DEBUG", f"Computed RMS for {len(rms_values)} windows "
                 f"({len(rms_values) * window_sec:.0f}s audio)")

    events = _detect_spikes(rms_values, window_sec, speech_regions=speech_regions)
    clusters = _cluster_events(events, window_sec)

    if clusters:
        spikes = sum(1 for c in clusters if c["kind"] == "spike")
        highs = sum(1 for c in clusters if c["kind"] == "high")
        log("OK", f"Found {len(clusters)} energy moments ({spikes} spikes, {highs} sustained)")
    else:
        log("INFO", "No significant energy spikes detected")

    return clusters
