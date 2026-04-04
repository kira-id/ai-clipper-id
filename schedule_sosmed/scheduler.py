#!/usr/bin/env python3
# ============================================================
#  scheduler.py  –  Cross-platform video uploader  [v2]
#  Platforms : Instagram Reels | YouTube Shorts | TikTok
#
#  Posting strategy:
#    • Times = statistically best slots for Indonesian audience
#    • Each slot has a tier (1=best engagement)
#    • Highest clip_score goes to best-tier slot
#    • clips.json re-read fresh every job (edit anytime)
#    • Filename read from "filename" field in clips.json
#    • No FFmpeg — raw files uploaded as-is
# ============================================================

import os
import json
import time
import pickle
import random
import shutil
import signal
import logging
import argparse
import schedule
import pytz
import subprocess
from datetime import datetime, timezone, timedelta

from instagrapi import Client as IGClient
from instagrapi.exceptions import LoginRequired
import googleapiclient.discovery
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from tiktok_uploader.upload import TikTokUploader
import atexit

from config import (
    CLIPS_FOLDER,
    INSTAGRAM_CREDENTIAL_FILE, INSTAGRAM_SESSION_FILE,
    YOUTUBE_CLIENT_SECRETS, YOUTUBE_TOKEN_FILE, YOUTUBE_CATEGORY_ID,
    YOUTUBE_CHANNEL_HANDLE,
    YOUTUBE_PRIVACY_STATUS,
    YOUTUBE_DEFAULT_LANGUAGE,
    YOUTUBE_DEFAULT_AUDIO_LANGUAGE,
    YOUTUBE_LICENSE,
    YOUTUBE_EMBEDDABLE,
    YOUTUBE_PUBLIC_STATS_VIEWABLE,
    YOUTUBE_NOTIFY_SUBSCRIBERS,
    YOUTUBE_SELF_DECLARED_MADE_FOR_KIDS,
    TIKTOK_COOKIES_FILE,
    TIKTOK_BROWSER_DATA_DIR,
    CHROME_PROFILE_DIR,
    FFMPEG_BIN,
    FFPROBE_BIN,
    ENABLE_INSTAGRAM, ENABLE_YOUTUBE, ENABLE_TIKTOK,
)

RESET_TIKTOK_BROWSER_DATA = False
RESET_TIKTOK_BROWSER_DATA_HARD = False

# ── Weekly upload pattern (rest + daily 3–4 posts) ───────────────────
# Each week follows a fixed pattern: [0, 1, 1, 1, 1, 1, 1]
# This guarantees 1 rest day per week + 6 active days. On active days we
# randomize between 3 and 4 posts to keep timing organic rather than rigid.
#
# Pattern: [0, 1, 1, 1, 1, 1, 1]
#   • 1 rest day (0) — algorithm fatigue prevention
#   • 6 active days (3–4 posts each)
# Total: 18–24 posts/week with consistent rest spacing.
# The order shuffles each week to maintain organic variation.
_WEEKLY_PATTERN = [0, 1, 1, 1, 1, 1, 1]

# Active-day upload target options. 3-4 posts/day now configured globally.
DAILY_ACTIVE_POSTS = [3, 4]
_weekly_shuffle: list[int] = []
_weekly_start_date = None

# Drawn from the weekly pattern; shared across all platforms since one slot
# = one cross-platform post.
_daily_target: int = 1  # will be overwritten on first reset


def _get_daily_seed(date: datetime.date) -> int:
    """Generate a deterministic seed for a given date.
    
    This ensures that the same schedule is generated for the same date,
    even if the server crashes and restarts mid-day.
    
    Args:
        date: The date to generate seed for
        
    Returns:
        An integer seed based on the date string (YYYY-MM-DD)
    """
    date_str = date.strftime("%Y-%m-%d")
    # Convert date string to a numeric seed
    return int(date_str.replace("-", ""))

# Track daily upload counts (reset at midnight WIB)
_daily_counts = {
    "youtube": 0,
    "instagram": 0,
    "tiktok": 0,
    "last_reset_date": None,
}

# Track last upload timestamp per platform (minimum gap enforcement)
_last_upload_ts: dict[str, float] = {
    "youtube": 0.0,
    "instagram": 0.0,
    "tiktok": 0.0,
}

# Minimum seconds between consecutive uploads to the same platform.
# 90 minutes keeps us under spam-detection thresholds while safely
# accommodating up to 15-min jitter on 2-hour-spaced slots
# (worst case: 2 h − 15 min jitter = 1 h 45 min > 90 min ✓).
MIN_GAP_BETWEEN_UPLOADS_SECS = 90 * 60  # 90 minutes

# ─────────────────────────────────────────────────────────────────────────
# Human-like delays (stealth)
# ─────────────────────────────────────────────────────────────────────────

def _human_pause(min_s: float = 1.5, max_s: float = 4.0) -> None:
    """Random pause between requests to appear human-like."""
    time.sleep(random.uniform(min_s, max_s))


def _post_login_cooldown() -> None:
    """Cool-down after successful login to avoid triggering spam detection."""
    wait = random.uniform(1, 3)
    log.debug(f"⏳ Post-login cooldown: {wait:.1f}s")
    time.sleep(wait)


def _inter_platform_delay() -> None:
    """Delay between platform uploads (60-180s) to avoid fingerprinting."""
    wait = random.randint(60, 180)
    log.info(f"💤 Pausing {wait}s before next platform...")
    time.sleep(wait)


def _remove_path(path: str) -> None:
    """Remove file/symlink/directory path if present."""
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _is_pid_alive(pid: int) -> bool:
    """Return True if process exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _find_profile_processes(profile_dir: str) -> list[int]:
    """Find Chrome/Chromium PIDs that are using the given profile dir."""
    try:
        result = subprocess.run(
            ["ps", "ax", "-o", "pid=", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        log.warning(f"⚠️  TikTok: could not inspect browser processes: {exc}")
        return []

    profile_arg = f"--user-data-dir={profile_dir}"
    current_pid = os.getpid()
    pids: list[int] = []

    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue

        if pid == current_pid:
            continue

        cmd = parts[1] if len(parts) > 1 else ""
        if profile_arg not in cmd:
            continue
        if "Chrome" not in cmd and "Chromium" not in cmd and "msedge" not in cmd:
            continue

        pids.append(pid)

    return sorted(set(pids))


def _terminate_processes(pids: list[int], grace_seconds: float = 3.0) -> tuple[list[int], list[int]]:
    """Terminate PIDs gracefully, then force-kill if needed.

    Returns:
        (terminated_pids, still_alive_pids)
    """
    if not pids:
        return [], []

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception as exc:
            log.warning(f"⚠️  TikTok: failed to SIGTERM pid {pid}: {exc}")

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        alive = [pid for pid in pids if _is_pid_alive(pid)]
        if not alive:
            return sorted(set(pids)), []
        time.sleep(0.15)

    alive = [pid for pid in pids if _is_pid_alive(pid)]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception as exc:
            log.warning(f"⚠️  TikTok: failed to SIGKILL pid {pid}: {exc}")

    time.sleep(0.2)
    still_alive = [pid for pid in alive if _is_pid_alive(pid)]
    terminated = [pid for pid in pids if pid not in still_alive]
    return terminated, still_alive


def ensure_tiktok_profile_unlocked() -> bool:
    """Ensure TikTok persistent profile is not locked by another browser process."""
    if not os.path.isdir(TIKTOK_BROWSER_DATA_DIR):
        return True

    pids = _find_profile_processes(TIKTOK_BROWSER_DATA_DIR)
    if pids:
        log.warning(
            "⚠️  TikTok profile is currently in use by Chrome/Chromium "
            f"(pid: {', '.join(str(pid) for pid in pids)}). Closing those processes..."
        )
        terminated, alive = _terminate_processes(pids)
        if terminated:
            log.info(f"🧹 Closed profile-locking browser process(es): {', '.join(str(pid) for pid in terminated)}")
        if alive:
            log.error(
                "❌ TikTok profile is still locked by running process(es): "
                f"{', '.join(str(pid) for pid in alive)}"
            )
            return False

    singleton_paths = [
        "SingletonLock",
        "SingletonSocket",
        "SingletonCookie",
    ]
    for rel_path in singleton_paths:
        abs_path = os.path.join(TIKTOK_BROWSER_DATA_DIR, rel_path)
        if os.path.exists(abs_path):
            try:
                _remove_path(abs_path)
                log.debug(f"🧹 Removed stale lock artifact: {abs_path}")
            except Exception as exc:
                log.error(f"❌ Could not remove stale lock artifact {abs_path}: {exc}")
                return False

    return True


def reset_tiktok_browser_data(hard_reset: bool = False) -> bool:
    """Reset TikTok browser data.

    Default behavior keeps session/auth state and only clears volatile data
    that often causes profile corruption (locks, shader/cache/temp files).
    Use hard_reset=True to delete the entire profile directory.
    """
    if not os.path.exists(TIKTOK_BROWSER_DATA_DIR):
        log.info(f"🧹 TikTok browser data dir already missing: {TIKTOK_BROWSER_DATA_DIR}")
        return True

    try:
        if not ensure_tiktok_profile_unlocked():
            return False

        if hard_reset:
            shutil.rmtree(TIKTOK_BROWSER_DATA_DIR)
            log.info(f"🧹 Hard reset TikTok browser data: {TIKTOK_BROWSER_DATA_DIR}")
            return True

        root_transient = [
            "SingletonLock",
            "SingletonSocket",
            "SingletonCookie",
            "GraphiteDawnCache",
            "GrShaderCache",
            "ShaderCache",
            "Code Cache",
            "Crashpad",
            "component_crx_cache",
            "extensions_crx_cache",
            "BrowserMetrics",
            "BrowserMetrics-spare.pma",
            "first_party_sets.db-journal",
        ]
        default_transient = [
            "Cache",
            "Code Cache",
            "GPUCache",
            "Service Worker/CacheStorage",
            "Service Worker/ScriptCache",
            "DawnGraphiteCache",
        ]

        removed = 0

        for rel_path in root_transient:
            abs_path = os.path.join(TIKTOK_BROWSER_DATA_DIR, rel_path)
            if os.path.exists(abs_path):
                _remove_path(abs_path)
                removed += 1

        default_dir = os.path.join(TIKTOK_BROWSER_DATA_DIR, "Default")
        if os.path.isdir(default_dir):
            for rel_path in default_transient:
                abs_path = os.path.join(default_dir, rel_path)
                if os.path.exists(abs_path):
                    _remove_path(abs_path)
                    removed += 1

        log.info(
            "🧹 Reset TikTok browser volatile data (session preserved): "
            f"{removed} path(s) removed"
        )
        return True
    except Exception as exc:
        log.error(f"❌ Could not reset TikTok browser data: {exc}")
        return False


def reset_daily_counts_if_needed():
    """Reset upload counters and draw today's target from the weekly pattern.

    The weekly pattern is shuffled once per week (Monday start) to ensure:
      • Exactly 1 rest day per week
      • 5 days at 1 post/day
      • 1 day at 2 posts/day
    
    All randomization is seeded by date to ensure deterministic schedules
    that survive server crashes and restarts.
    """
    global _daily_target, _weekly_shuffle, _weekly_start_date

    today = datetime.now(pytz.timezone("Asia/Jakarta")).date()

    # Calculate week start (Monday)
    week_start = today - timedelta(days=today.weekday())

    # Regenerate shuffle at start of each week
    if _weekly_start_date != week_start:
        _weekly_shuffle = _WEEKLY_PATTERN.copy()
        # Use week-start seed for reproducible weekly shuffle
        random.seed(_get_daily_seed(week_start))
        random.shuffle(_weekly_shuffle)
        _weekly_start_date = week_start
        log.info(f"🔄 Weekly pattern shuffled for week starting {week_start}: {_weekly_shuffle}")

    # Get today's position in the week (0=Monday, 6=Sunday)
    day_of_week = today.weekday()
    scheduled_value = _weekly_shuffle[day_of_week]

    # If the day is a rest day, target is zero. Otherwise randomize based on DAILY_ACTIVE_POSTS.
    if scheduled_value == 0:
        today_target = 0
    else:
        # Use date seed for reproducible daily target selection
        random.seed(_get_daily_seed(today))
        today_target = random.choice(DAILY_ACTIVE_POSTS)

    # Reset daily counts at midnight and set the target once per day
    if _daily_counts["last_reset_date"] != today:
        _daily_counts["youtube"] = 0
        _daily_counts["instagram"] = 0
        _daily_counts["tiktok"] = 0
        _daily_counts["last_reset_date"] = today
        _daily_target = today_target
        log.info(f"📊 Daily counters reset for {today}. Today's upload target: {_daily_target}/platform")
    else:
        # Keep stable target during the day; detect misalignment and correct if needed
        if scheduled_value == 0:
            _daily_target = 0
        elif _daily_target not in (3, 4):
            _daily_target = today_target
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("uploader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

CLIPS_JSON = os.path.join(CLIPS_FOLDER, "clips.json")

# folder where we keep a simple record of every clip that was posted
# - a JSON array is stored in logs/clips.json and a line-oriented journal
#   is kept in logs/uploads.jsonl for easy tailing/debugging
LOGS_FOLDER = os.path.join(os.path.dirname(CLIPS_FOLDER), "logs")

# request read access as well so that we can call channels.list (confirm auth)
# and future features that read channel data.  If the saved token only had
# the upload scope we'll drop it and re-authorize automatically.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    # alternatively use the single broad scope:
    # "https://www.googleapis.com/auth/youtube",
]

# Module-level YouTube client cache
_yt_client = None
_yt_channel_valid = False  # Track if channel validation passed this session

FFMPEG_EXECUTABLE = FFMPEG_BIN
FFPROBE_EXECUTABLE = FFPROBE_BIN


def resolve_executable(executable: str) -> str | None:
    executable = (executable or "").strip()
    if not executable:
        return None

    if os.path.isabs(executable) or os.path.dirname(executable):
        if os.path.isfile(executable):
            return executable
        return None

    return shutil.which(executable)


def ensure_media_tools_ready() -> bool:
    global FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE

    ffmpeg_resolved = resolve_executable(FFMPEG_BIN)
    ffprobe_resolved = resolve_executable(FFPROBE_BIN)

    if not ffmpeg_resolved:
        log.error(
            "❌ ffmpeg executable not found. "
            f"Set FFMPEG_BIN in config.py (current: '{FFMPEG_BIN}')."
        )
    if not ffprobe_resolved:
        log.error(
            "❌ ffprobe executable not found. "
            f"Set FFPROBE_BIN in config.py (current: '{FFPROBE_BIN}')."
        )

    if not ffmpeg_resolved or not ffprobe_resolved:
        log.error("   Install FFmpeg and point both settings to valid executables.")
        return False

    FFMPEG_EXECUTABLE = ffmpeg_resolved
    FFPROBE_EXECUTABLE = ffprobe_resolved
    log.info(f"🎬 ffmpeg : {FFMPEG_EXECUTABLE}")
    log.info(f"🎬 ffprobe: {FFPROBE_EXECUTABLE}")
    return True

# ═══════════════════════════════════════════════════════════
#  Statistically best posting times — Indonesian audience WIB
#
#  Based on: Sprout Social, Later, HubSpot, SocialBee research
#  + Indonesian social media usage patterns (APJII, We Are Social)
#
#  Tier 1 ★★★ — highest engagement windows
#  Tier 2 ★★  — strong engagement
#  Tier 3 ★   — moderate engagement
#
#  21:00  Peak prime time. Highest daily screen time.
#         Everyone is home, relaxed, long scroll sessions.
#  12:00  Lunch break. Commuters + office workers peak.
#         2nd highest engagement across all 3 platforms.
#  19:00  Post-Maghrib / pre-Isya. Screens return after prayer.
#         Reels/Shorts perform especially well here.
#  16:00  End of school / late work. Pre-Maghrib window.
#         Strong for TikTok and Reels. Avoids ~17:45 prayer dip.
#  09:00  Mid-morning work break. Consistent but lower.
#  07:00  Morning commute. Decent reach, lower engagement rate.
#
#  Note:  Maghrib falls ~17:45–18:15 WIB year-round (equatorial).
#         Avoid scheduling into that window for a Muslim-majority
#         audience — engagement notably dips during prayer time.
# ═══════════════════════════════════════════════════════════

SCHEDULE_SLOTS = [
    # (time_WIB, tier, label)
    ("20:45", 1, "Peak prime time"),
    ("11:50", 1, "Lunch peak"),
    ("18:50", 2, "Post-Maghrib"),
    ("15:50", 2, "End of work/school"),
    ("08:50", 3, "Morning work break"),
    ("06:50", 3, "Morning commute"),
]

# ── Day-of-week slot selection ──────────────────────────────
# Not every slot fires every day.  This keeps the posting pattern
# looking organic (humans don't post at exact same 6 times daily).
# Each day randomly picks a subset of slots from the full list.
# Tier-1 slots are always included; lower tiers are included
# probabilistically.  The result is 3–4 posts on active days (0 on rest day)
# instead of a rigid schedule.

# Keep track of which slots are active today (regenerated at midnight)
DAY_ENGAGEMENT = {
    "Monday":    0.85,
    "Tuesday":   0.90,
    "Wednesday": 0.95,
    "Thursday":  1.00,  # Best engagement
    "Friday":    0.92,
    "Saturday":  0.88,
    "Sunday":    0.80,
}

# Keep track of which slots are active today (regenerated at midnight)
_active_slots: set[str] = set()
_active_slots_date = None

def refresh_active_slots():
    """Select today's active slots so the count matches ``_daily_target``.

    Slots are filled by tier priority (Tier 1 first, then 2, then 3).
    Within each tier the order is shuffled so the specific times chosen
    vary day-to-day, preserving organic-looking behaviour.
    Called once per day (or on first run).
    
    Uses date-based seeding for deterministic slot selection that survives
    server crashes and restarts.
    """
    global _active_slots, _active_slots_date
    today = datetime.now(pytz.timezone("Asia/Jakarta")).date()
    if _active_slots_date == today:
        return

    # Ensure we have a fresh daily target for today
    reset_daily_counts_if_needed()

    # Group available slots by tier
    by_tier: dict[int, list[str]] = {}
    for slot_time, tier, _label in SCHEDULE_SLOTS:
        by_tier.setdefault(tier, []).append(slot_time)

    # Fill slots in tier order until the daily target is reached
    selected: list[str] = []
    remaining = _daily_target
    # Use date seed for reproducible slot shuffling (set once before the loop)
    random.seed(_get_daily_seed(today))
    for tier in sorted(by_tier.keys()):
        tier_slots = by_tier[tier][:]
        random.shuffle(tier_slots)
        take = min(remaining, len(tier_slots))
        selected.extend(tier_slots[:take])
        remaining -= take
        if remaining == 0:
            break

    _active_slots = set(selected)
    _active_slots_date = today
    log.info(
        f"📅 Active slots for {today} (target={_daily_target}): "
        f"{sorted(_active_slots)}  ({len(_active_slots)}/{len(SCHEDULE_SLOTS)})"
    )


# ═══════════════════════════════════════════════════════════
#  Clip queue helpers
# ═══════════════════════════════════════════════════════════

def dedupe_clips(clips: list) -> list:
    """Remove duplicate entries based on *filename*.

    Clips are considered identical if they share the same ``filename``.  This
    function keeps the first occurrence and logs a warning for each removed
    duplicate.  Returns a new list in the original order with duplicates
    stripped.  If an entry is missing ``filename`` we leave it in place; the
    caller will warn separately.
    """
    seen = set()
    out = []
    for clip in clips:
        fn = clip.get("filename")
        if fn in seen:
            log.warning(f"duplicate clip entry removed (filename already seen): {fn}")
            continue
        seen.add(fn)
        out.append(clip)
    return out


def load_clips() -> list:
    """Read the queue JSON, dedupe entries and validate fields.

    The queue may be edited by hand or generated by external scripts, so we
    perform a little sanitisation here.  We intentionally *do not* use the
    old ``rank`` field as an identity – only ``filename`` matters.
    """
    with open(CLIPS_JSON, "r", encoding="utf-8") as f:
        clips = json.load(f)

    orig_len = len(clips)
    clips = dedupe_clips(clips)
    if len(clips) != orig_len:
        log.info(f"🧹 removed {orig_len - len(clips)} duplicate clip(s) from queue")

    # warn about missing filenames
    for clip in clips:
        if not clip.get("filename"):
            log.warning(f"clip missing 'filename' field: rank {clip.get('rank')} title '{clip.get('title')}'")

    # validate that all video files exist
    missing_files = []
    for clip in clips:
        filename = clip.get("filename")
        if filename:
            filepath = os.path.join(CLIPS_FOLDER, filename)
            if not os.path.isfile(filepath):
                missing_files.append({
                    "filename": filename,
                    "title": clip.get("title", "N/A"),
                    "rank": clip.get("rank", "N/A")
                })

    if missing_files:
        log.warning(f"⚠️  Found {len(missing_files)} missing video file(s) in clips.json:")
        for item in missing_files:
            log.warning(f"   - {item['filename']} (rank {item['rank']}, title: {item['title']})")

    return clips


def get_posted_filenames() -> set:
    """Return a set of filenames that have already been uploaded.

    We prefer the structured log ``logs/clips.json`` if it exists because it
    contains the full metadata.  As a fallback we scan ``uploader.log`` for
    lines that mention a filename; this is less reliable but better than
    nothing.
    """
    posted = set()
    json_log = os.path.join(LOGS_FOLDER, "clips.json")
    if os.path.exists(json_log):
        try:
            with open(json_log, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                clip = entry.get("clip", {})
                fn = clip.get("filename")
                if fn:
                    posted.add(fn)
        except Exception:
            log.warning("Could not read structured upload log; falling back to text scan")
    if not posted and os.path.exists("uploader.log"):
        import re
        with open("uploader.log", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"Filename\s*:\s*(\S+\.mp4)", line)
                if m:
                    posted.add(m.group(1))
    return posted


def clean_orphan_files(clips: list) -> int:
    """Remove any video files in ``CLIPS_FOLDER`` that aren't in clips.

    Returns the number of files deleted.  Only ``.mp4`` files are considered;
    padded derivatives (``_padded.mp4``) are also swept if their base file is
    missing.
    """
    valid = {c.get("filename") for c in clips if c.get("filename")}
    deleted = 0
    for fname in os.listdir(CLIPS_FOLDER):
        if not fname.lower().endswith(".mp4"):
            continue
        if fname not in valid:
            path = os.path.join(CLIPS_FOLDER, fname)
            try:
                os.remove(path)
                log.info(f"🧹 removed orphan video file: {path}")
                deleted += 1
            except Exception as e:
                log.warning(f"failed to delete orphan {path}: {e}")
    return deleted



def save_clips(clips: list):
    """Write clips to disk atomically to prevent corruption on crash."""
    tmp = CLIPS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CLIPS_JSON)



# Pick best clip for the day, using engagement statistics
def get_clip_for_day(day_name: str) -> tuple:
    """
    Pick the best clip for a given day, using engagement statistics.
    Matching logic:
      - Multiply engagement score by clip_score (or class_score if present)
      - Sort all available clips by this combined score
      - Return the highest scoring clip
    """
    clips = load_clips()
    if not clips:
        return None, None

    engagement = DAY_ENGAGEMENT.get(day_name, 1.0)

    available = []
    for clip in clips:
        filename = clip.get("filename")
        if not filename:
            log.warning(f"Clip rank {clip.get('rank', '?')} missing 'filename' field — skipping")
            continue
        path = os.path.join(CLIPS_FOLDER, filename)
        if os.path.exists(path):
            # Use class_score if present, else clip_score
            score = clip.get("class_score", clip.get("clip_score", 0))
            combined_score = score * engagement
            available.append((clip, path, combined_score))
        else:
            log.warning(f"File not found for rank {clip.get('rank', '?')}: {filename}")

    if not available:
        return None, None

    # Sort by combined score descending
    available.sort(key=lambda x: x[2], reverse=True)
    # Always pick the highest scoring available clip
    return available[0][0], available[0][1]


def get_clip_by_filename(filename: str) -> tuple:
    clips = load_clips()
    for clip in clips:
        if clip.get("filename") == filename:
            path = os.path.join(CLIPS_FOLDER, filename)
            if os.path.exists(path):
                return clip, path
            log.error(f"File for test post not found: {path}")
            return None, None
    
    # If not found in clips.json, check logs/clips.json
    json_log = os.path.join(LOGS_FOLDER, "clips.json")
    if os.path.exists(json_log):
        try:
            with open(json_log, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            for entry in log_data:
                clip = entry.get("clip", {})
                if clip.get("filename") == filename:
                    path = os.path.join(CLIPS_FOLDER, filename)
                    if os.path.exists(path):
                        return clip, path
                    log.error(f"File for test post not found: {path}")
                    return None, None
        except Exception as e:
            log.warning(f"Could not read logs/clips.json: {e}")
    
    log.error(f"No clip entry found in clips.json or logs/clips.json for filename: {filename}")
    return None, None


def log_upload(clip: dict, video_path: str, results: dict) -> None:
    """Write upload results to structured logs.

    Appends to both logs/clips.json (structured array) and logs/uploads.jsonl
    (newline-delimited journal for easy tailing).
    """
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    timestamp = datetime.now(pytz.timezone("Asia/Jakarta")).isoformat()
    
    entry = {
        "timestamp": timestamp,
        "clip": clip,
        "video_path": video_path,
        "results": results,
        "success_count": sum(bool(v) for v in results.values()),
        "total_platforms": len(results),
    }
    
    # Append to newline-delimited journal (easy to tail/grep)
    jsonl_path = os.path.join(LOGS_FOLDER, "uploads.jsonl")
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Could not write to upload journal: {e}")
    
    # Update structured log array
    json_log = os.path.join(LOGS_FOLDER, "clips.json")
    try:
        if os.path.exists(json_log):
            with open(json_log, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        log_data.append(entry)
        
        with open(json_log, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Could not update structured upload log: {e}")


def mark_done(clip: dict, delete_files: bool = True) -> bool:
    """Remove a clipped entry from the queue.

    Identification is solely by ``filename``.  If multiple entries happen to
    share the same filename (which should never happen after deduplication)
    they all are removed to avoid orphaned duplicates.
    
    Args:
        clip: The clip dict to remove
        delete_files: If True, also delete the video file(s). If False, only
                      remove from clips.json (useful for partial uploads)
    
    Returns ``True`` if anything was removed from the queue, ``False`` otherwise.
    """
    fn = clip.get("filename")
    if not fn:
        log.error("mark_done called with clip lacking filename")
        return False

    clips = load_clips()
    remaining = [c for c in clips if c.get("filename") != fn]
    removed = len(clips) - len(remaining)
    if removed:
        save_clips(remaining)
        log.info(f"📋 Removed {removed} clip(s) with filename '{fn}' from clips.json ({len(remaining)} remaining)")
        
        if delete_files:
            # try deleting the file(s)
            path = os.path.join(CLIPS_FOLDER, fn)
            for candidate in (path, f"{os.path.splitext(path)[0]}_padded.mp4"):
                try:
                    if os.path.exists(candidate):
                        os.remove(candidate)
                        log.info(f"🗑️  Deleted source: {candidate}")
                except Exception as e:
                    log.warning(f"Could not delete {candidate}: {e}")
        else:
            log.info(f"💾 Kept video file for manual review: {fn}")
        
        return True
    else:
        log.warning(f"mark_done: clip '{fn}' not found in queue")
        return False


# ═══════════════════════════════════════════════════════════
#  Auth helpers
# ═══════════════════════════════════════════════════════════

def get_ig_client() -> IGClient:
    cl = IGClient()
    if os.path.exists(INSTAGRAM_SESSION_FILE):
        cl.load_settings(INSTAGRAM_SESSION_FILE)
        try:
            cl.get_timeline_feed()
            return cl
        except LoginRequired:
            log.warning("Instagram session expired, re-logging in ...")
            # Create fresh client instance for re-login
            cl = IGClient()
    
    # Load credentials from JSON file
    with open(INSTAGRAM_CREDENTIAL_FILE, "r", encoding="utf-8") as f:
        creds = json.load(f)
    
    try:
        cl.login(creds["username"], creds["password"])
        cl.dump_settings(INSTAGRAM_SESSION_FILE)
    except Exception as e:
        log.error(f"Instagram login failed: {e}")
        if "403" in str(e) or "login_required" in str(e):
            log.error("   Instagram may be blocking automated logins. Try:")
            log.error("   1. Delete ig_session.json and try again")
            log.error("   2. Verify credentials in ig_cred.json are correct")
            log.error("   3. Login manually to Instagram from your IP address first")
            log.error("   4. Wait a few hours before retrying (rate limit)")
        raise
    return cl


def get_youtube_client():
    """
    Get authenticated YouTube client using explicit auth strategy (no fallback).
    Cached at module level to avoid rebuilding on every call.

    Priority:
      1. Cached client (if already authenticated)
      2. Saved token with valid scopes
      3. Refresh if expired
      4. Fresh OAuth flow
    """
    global _yt_client, _yt_channel_valid

    # Return cached client if available
    if _yt_client is not None:
        return _yt_client

    credentials = None

    # Try loading saved token
    if os.path.exists(YOUTUBE_TOKEN_FILE):
        log.debug(f"📂  Loading saved YouTube token...")
        try:
            with open(YOUTUBE_TOKEN_FILE, "rb") as f:
                credentials = pickle.load(f)
        except Exception as e:
            log.warning(f"⚠️  Cannot load token file: {e}")
            credentials = None
    
    # Validate and check scopes
    if credentials:
        needed_scopes = set(YOUTUBE_SCOPES)
        existing_scopes = set(credentials.scopes or [])
        if not needed_scopes.issubset(existing_scopes):
            log.warning(f"⚠️  Token missing required scopes {needed_scopes - existing_scopes}")
            credentials = None
    
    # Attempt refresh if token exists but is expired
    if credentials and hasattr(credentials, "refresh_token") and credentials.refresh_token:
        if credentials.expired:
            log.debug(f"🔄  Refreshing expired YouTube token...")
            try:
                _human_pause(1, 2)
                credentials.refresh(Request())
                with open(YOUTUBE_TOKEN_FILE, "wb") as f:
                    pickle.dump(credentials, f)
                log.info(f"✅ YouTube token refreshed")
                _yt_client = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
                return _yt_client
            except Exception as e:
                log.warning(f"⚠️  Token refresh failed: {e}")
                credentials = None
    
    # If token is valid and not expired, use it
    if credentials and credentials.valid:
        log.debug(f"🔐  Using existing valid YouTube token")
        _yt_client = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
        return _yt_client
    
    # No valid credentials — explicit OAuth flow (not fallback)
    log.info(f"📋️  Starting fresh YouTube OAuth flow...")
    try:
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            YOUTUBE_CLIENT_SECRETS, YOUTUBE_SCOPES
        )
        credentials = flow.run_local_server(port=0)
        with open(YOUTUBE_TOKEN_FILE, "wb") as f:
            pickle.dump(credentials, f)
        log.info(f"✅ YouTube OAuth complete, token saved")
        _yt_client = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
        return _yt_client
    except Exception as e:
        log.error(f"❌ YouTube OAuth failed: {e}")
        raise


def validate_youtube_channel(yt) -> bool:
    """Validate the YouTube channel (cached per session)."""
    global _yt_channel_valid
    
    # Skip validation if already validated this session
    if _yt_channel_valid:
        return True
    
    response = yt.channels().list(part="id,snippet", mine=True).execute()
    items = response.get("items", [])
    if not items:
        log.error("❌ YouTube failed: no authenticated channel found for current token.")
        return False

    channel = items[0]
    channel_id = channel.get("id", "")
    channel_title = channel.get("snippet", {}).get("title", "")
    channel_custom_url = channel.get("snippet", {}).get("customUrl", "")
    log.info(
        f"📺 YouTube auth channel: title='{channel_title}', id='{channel_id}', customUrl='{channel_custom_url}'"
    )

    expected = (YOUTUBE_CHANNEL_HANDLE or "").strip().lower().lstrip("@")
    actual = (channel_custom_url or "").strip().lower().lstrip("@")
    if expected and actual != expected:
        log.error(
            "❌ YouTube failed: authenticated channel does not match YOUTUBE_CHANNEL_HANDLE. "
            f"expected='@{expected}', actual='@{actual or 'unknown'}'"
        )
        log.error("   Delete yt_token.pickle, re-run auth, and login with the correct YouTube channel account.")
        return False

    _yt_channel_valid = True
    return True


def probe_video_info(video_path: str) -> tuple[int, int, float] | tuple[None, None, None]:
    cmd = [
        FFPROBE_EXECUTABLE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        raw_w = stream.get("width")
        raw_h = stream.get("height")
        raw_dur = fmt.get("duration")
        if raw_w is None or raw_h is None or raw_dur is None:
            log.error("❌ ffprobe returned incomplete stream info (missing width/height/duration)")
            return None, None, None
        width = int(raw_w)
        height = int(raw_h)
        duration = float(raw_dur)
        return width, height, duration
    except Exception as e:
        log.error(f"❌ Could not inspect video with ffprobe: {e}")
        return None, None, None


def pad_video_to_vertical(video_path: str) -> str | None:
    """Pad a landscape clip with black bars to make it taller than it is
    wide.

    The output file is created alongside the original with a
    ``_padded`` suffix.  We choose a target height equal to twice the
    original width (a 1:2 aspect) which is safely taller than the
    original.  If FFmpeg fails or the resulting video still isn't vertical
    we return ``None``.
    """
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_padded{ext}"

    # get original dimensions
    w, h, _ = probe_video_info(video_path)
    if w is None or h is None:
        return None

    target_h = int(w * 2)
    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"pad={w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
        "-c:a",
        "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        log.error(f"❌ Could not pad video to vertical: {e}")
        return None

    # verify result is vertical
    w2, h2, _ = probe_video_info(out_path)
    if w2 is None or h2 is None or h2 <= w2:
        log.error(
            "❌ Padding did not produce a vertical video (still "
            f"{w2}x{h2})."
        )
        try:
            os.remove(out_path)
        except Exception:
            pass
        return None

    log.info(f"🔲 Padded {video_path} -> {out_path} with black bars to vertical")
    return out_path


def ensure_shorts_eligible(video_path: str) -> str | None:
    """Verify (and if possible fix) a clip so it can be uploaded as a
    YouTube Short.

    Returns the path that should be used for the upload.  ``None`` means
    the clip cannot be used.
    """
    width, height, duration = probe_video_info(video_path)
    if width is None or height is None or duration is None:
        return None

    if duration > 180:
        log.error(
            f"❌ YouTube Shorts max duration is 180s. Current duration is {duration:.2f}s."
        )
        return None

    # if clip is landscape, pad with black bars instead of rotating
    if height <= width:
        log.warning(
            f"📐 Video is not vertical ({width}x{height}); padding with black bars"
        )
        padded = pad_video_to_vertical(video_path)
        if not padded:
            # helper already logged the error
            return None
        return ensure_shorts_eligible(padded)

    # at this point we know height > width and duration is OK
    return video_path


def unique_ify_video(video_path: str, platform_tag: str = "generic") -> str | None:
    """Create a unique version of the video to defeat perceptual hash matching.

    Each call produces a different output because every parameter is
    randomised.  Use a different ``platform_tag`` for each platform so
    that Instagram, YouTube, and TikTok each receive a distinct file.

    Techniques applied:
    1. Strip all metadata.
    2. Random brightness (±0.03) and contrast (0.97–1.03) — wide enough
       to change the perceptual hash, still imperceptible to viewers.
    3. Random saturation tweak (0.97–1.03).
    4. Tiny random crop (1–3 px per side) then scale back — changes the
       pixel grid so DCT-based hashes differ.
    5. Re-encode video *and* audio (slight pitch-preserving tempo shift
       of ±0.5 %) so the audio fingerprint also changes.
    """
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_unique_{platform_tag}{ext}"

    # --- Video tweaks ---
    brightness = random.uniform(-0.03, 0.03)
    contrast = random.uniform(0.97, 1.03)
    saturation = random.uniform(0.97, 1.03)

    # Tiny random crop (1-3 px per side) → scale back to original dims
    w, h, _ = probe_video_info(video_path)
    crop_px = random.randint(1, 3)
    if w and h:
        cw, ch = w - 2 * crop_px, h - 2 * crop_px
        crop_filter = f"crop={cw}:{ch}:{crop_px}:{crop_px},scale={w}:{h}"
    else:
        crop_filter = None

    vf_parts = [f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"]
    if crop_filter:
        vf_parts.append(crop_filter)
    vf_str = ",".join(vf_parts)

    # --- Audio tweak: tiny tempo shift (±0.5 %) to change audio fingerprint ---
    tempo = random.uniform(0.995, 1.005)
    af_str = f"atempo={tempo}"

    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-i", video_path,
        "-map_metadata", "-1",        # Strip all metadata
        "-vf", vf_str,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",                 # High quality
        "-af", af_str,
        "-c:a", "aac", "-b:a", "192k",  # Re-encode audio
        out_path,
    ]

    try:
        log.info(
            f"✨ Uniquifying video [{platform_tag}]: {video_path} "
            f"(bright={brightness:+.3f}, contrast={contrast:.3f}, "
            f"sat={saturation:.3f}, crop={crop_px}px, tempo={tempo:.4f})"
        )
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out_path
    except Exception as e:
        log.error(f"❌ Could not uniquify video [{platform_tag}]: {e}")
        return None


def generate_thumbnail(video_path: str, width: int, height: int, suffix: str) -> str | None:
    """Create a still image from ``video_path`` at the requested size.

    Seeks to ~10 % into the video duration to avoid black intros.
    The thumbnail is scaled to fit within ``width``x``height`` while
    preserving the original aspect ratio, padded with black bars to
    exactly match the target dimensions.
    """
    thumbs_dir = os.path.join(CLIPS_FOLDER, "_thumbnails")
    os.makedirs(thumbs_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    thumbnail_path = os.path.join(thumbs_dir, f"{base}_{suffix}.jpg")

    # Seek to the first frame
    _w, _h, duration = probe_video_info(video_path)
    seek_secs = 0.0

    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-ss", f"{seek_secs:.2f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        thumbnail_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return thumbnail_path
    except Exception as e:
        log.error(f"❌ Could not generate thumbnail with ffmpeg: {e}")
        return None

# ═══════════════════════════════════════════════════════════
#  Upload functions — raw file, no conversion
# ═══════════════════════════════════════════════════════════

def unique_ify_caption(caption: str) -> str:
    """Randomize caption slightly to avoid exact-duplicate detection.

    Uses only invisible Unicode variation selectors — these are
    imperceptible to readers but alter the byte signature of the string.
    Cyrillic homoglyphs are intentionally avoided: modern spam classifiers
    flag mixed-script text and can trigger shadowbans.
    """
    # Append a random invisible Unicode character from the Variation Selector
    # block (U+FE00–FE0F) — zero visual impact, changes string hash.
    variation_selectors = [chr(c) for c in range(0xFE00, 0xFE10)]
    suffix = random.choice(variation_selectors)

    # Also insert a zero-width non-joiner at a random interior position
    # to further differentiate the byte sequence.
    if len(caption) > 10:
        pos = random.randint(5, len(caption) - 5)
        caption = caption[:pos] + "\u200C" + caption[pos:]

    return caption + suffix


def upload_instagram(video_path: str, clip: dict) -> bool:
    reset_daily_counts_if_needed()
    if _daily_counts["instagram"] >= _daily_target:
        log.warning(f"⚠️  Instagram daily limit reached ({_daily_target}/day) — skipping")
        return False

    try:
        log.debug(f"📹 Instagram: preparing upload for {clip.get('filename')}...")

        thumbnail_path = generate_thumbnail(video_path, 1080, 1920, "instagram")
        if not thumbnail_path:
            log.error(f"❌ Instagram: failed to generate thumbnail")
            return False

        ig = get_ig_client()
        unique_caption = unique_ify_caption(clip.get("caption", ""))
        comment_bait = clip.get("comment_bait", "").strip()

        # Add comment_bait to caption (like TikTok) for engagement
        if comment_bait:
            caption = f"{unique_caption}\n\n💬 {comment_bait}"
        else:
            caption = unique_caption

        # Instagram caption limit: 2200 chars
        caption = caption[:2200]

        log.info(f"📤 Instagram: uploading {clip.get('title', '?')}...")
        _human_pause(1, 3)

        # Upload the clip
        ig.clip_upload(path=video_path, caption=caption, thumbnail=thumbnail_path)
        _post_login_cooldown()

        # Get the media ID to post comment
        media_id = None
        try:
            # Try to get the recently uploaded media
            user_id = ig.user_id
            medias = ig.user_medias(user_id, amount=1)
            if medias:
                media_id = str(medias[0].pk)
                log.debug(f"📎 Instagram: got media ID {media_id} for commenting")
        except Exception as e:
            log.warning(f"⚠️  Instagram: could not get media ID for comment: {e}")

        _daily_counts["instagram"] += 1
        log.info(f"✅ Instagram: {clip.get('title', '?')} uploaded successfully [{_daily_counts['instagram']}/{_daily_target} today]")
        log.info(f"🖼️ Instagram thumbnail uploaded: {thumbnail_path}")

        # Also post comment_bait as first comment (after upload success)
        # This creates engagement signal even if comment_bait is already in caption
        if comment_bait and media_id:
            # Wait a bit before commenting (looks more natural)
            _human_pause(30, 90)  # 30-90 seconds after upload
            post_instagram_comment(ig, media_id, comment_bait)
        
        return True
    except Exception as e:
        log.error(f"❌ Instagram: {type(e).__name__}: {e}")
        return False


def upload_youtube(video_path: str, clip: dict) -> bool:
    """Upload a clip to YouTube Shorts with stealth measures."""
    reset_daily_counts_if_needed()
    if _daily_counts["youtube"] >= _daily_target:
        log.warning(f"⚠️  YouTube daily limit reached ({_daily_target}/day) — skipping")
        return False

    try:
        log.debug(f"📹 YouTube: preparing upload for {clip.get('filename')}...")

        # Get and validate YouTube client
        _human_pause(1, 2)
        yt = get_youtube_client()
        if not validate_youtube_channel(yt):
            log.error(f"❌ YouTube: channel validation failed")
            return False

        # Ensure vertical format
        new_path = ensure_shorts_eligible(video_path)
        if not new_path:
            log.error(f"❌ YouTube: video not eligible for Shorts format")
            return False
        if new_path != video_path:
            log.info(f"✨ Using converted video for upload: {new_path}")
            video_path = new_path

        # Generate thumbnail
        thumbnail_path = generate_thumbnail(video_path, 1280, 720, "youtube")
        if not thumbnail_path:
            log.error(f"❌ YouTube: failed to generate thumbnail")
            return False

        # Prepare metadata
        title_base = clip.get("title", "").strip()
        title = f"{title_base} #Shorts"
        if len(title) > 100:
            title = f"{title_base[:91].rstrip()} #Shorts"

        desc = unique_ify_caption(clip.get("caption", "").strip())
        comment_bait = clip.get("comment_bait", "").strip()

        # Add comment_bait to description for engagement (like TikTok & Instagram)
        if comment_bait:
            desc = f"{desc}\n\n💬 {comment_bait}"

        if "#shorts" not in desc.lower():
            desc = f"{desc}\n\n#Shorts"
        desc = desc[:5000]

        tags = ["Shorts", "shorts", "AI", "Indonesia", "fyp", "viral"]

        log.info(f"📤 YouTube: uploading {title[:50]}...")
        _human_pause(2, 5)  # Pre-upload pause

        req = yt.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": desc,
                    "tags": tags,
                    "categoryId": YOUTUBE_CATEGORY_ID,
                    "defaultLanguage": YOUTUBE_DEFAULT_LANGUAGE,
                    "defaultAudioLanguage": YOUTUBE_DEFAULT_AUDIO_LANGUAGE,
                },
                "status": {
                    "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                    "selfDeclaredMadeForKids": YOUTUBE_SELF_DECLARED_MADE_FOR_KIDS,
                    "license": YOUTUBE_LICENSE,
                    "embeddable": YOUTUBE_EMBEDDABLE,
                    "publicStatsViewable": YOUTUBE_PUBLIC_STATS_VIEWABLE,
                },
            },
            media_body=MediaFileUpload(
                video_path,
                mimetype="video/mp4",
                chunksize=8 * 1024 * 1024,
                resumable=True,
            ),
            notifySubscribers=YOUTUBE_NOTIFY_SUBSCRIBERS,
        )
        resp = req.execute()
        video_id = resp["id"]

        # Upload thumbnail
        _human_pause(1, 3)
        yt.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
        ).execute()

        _post_login_cooldown()
        _daily_counts["youtube"] += 1
        log.info(f"✅ YouTube: https://youtube.com/shorts/{video_id} [{_daily_counts['youtube']}/{_daily_target} today]")
        log.info(f"🖼️ YouTube thumbnail: {thumbnail_path}")

        # Post comment_bait as first comment (after upload success)
        if comment_bait:
            # Wait a bit before commenting (looks more natural)
            _human_pause(60, 180)  # 1-3 minutes after upload
            post_youtube_comment(yt, video_id, comment_bait)
        
        return True
    except Exception as e:
        error_str = str(e)
        if "quotaExceeded" in error_str or ("403" in error_str and "insufficientPermissions" in error_str):
            log.error("❌ YouTube: Daily API quota exceeded for comments")
            log.error("   YouTube API has 10,000 units/day; video uploads = 1,600 units each (~6/day max)")
            log.error("   Quota resets at midnight PT")
            log.error("   Check: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas")
            log.error("   Tip: Apply for quota increase or wait for reset")
        elif "403" in error_str:
            log.error(f"❌ YouTube: 403 Forbidden - {e}")
            log.error("   This may be a permissions issue. Try re-authenticating.")
        else:
            log.error(f"❌ YouTube: {e}")
        return False



def ensure_vertical(video_path: str) -> str | None:
    """Pad a landscape video with black bars until it is taller than it is wide.

    TikTok favors vertical uploads; if the source clip is wider than it is
    tall we call ``pad_video_to_vertical`` (which is already used for Instagram
    thumbnails) and recurse until the result is vertical.  ``None`` is
    returned on failure.
    """
    w, h, _ = probe_video_info(video_path)
    if w is None or h is None:
        return None
    if w > h:
        log.warning(
            f"📐 Video is not vertical ({w}x{h}); padding with black bars"
        )
        padded = pad_video_to_vertical(video_path)
        if not padded:
            return None
        return ensure_vertical(padded)
    return video_path


def upload_tiktok(video_path: str, clip: dict) -> bool:
    """Upload a clip to TikTok with stealth measures and robust error handling.

    Uses your Chrome profile directly - all cookies, sessions, and logins
    are automatically available. No manual cookie export needed.
    """
    reset_daily_counts_if_needed()
    if _daily_counts["tiktok"] >= _daily_target:
        log.warning(f"⚠️  TikTok daily limit reached ({_daily_target}/day) — skipping")
        return False

    try:
        log.debug(f"📹 TikTok: preparing upload for {clip.get('filename')}...")

        # Ensure vertical format
        video_path = ensure_vertical(video_path)
        if not video_path:
            log.error(f"❌ TikTok: cannot convert video to vertical format")
            return False

        # Copy Chrome profile to temp location for Playwright to use
        log.info(f"📋 Preparing Chrome profile from: {CHROME_PROFILE_DIR}")
        
        if not os.path.exists(CHROME_PROFILE_DIR):
            log.error(f"❌ Chrome profile not found: {CHROME_PROFILE_DIR}")
            log.error("   Make sure you're logged into TikTok in Chrome")
            return False
        
        # Extract fresh cookies from Chrome automatically
        log.info(f"🍪 Extracting fresh TikTok cookies from Chrome...")
        try:
            import browser_cookie3
            cj = browser_cookie3.chrome(domain_name='tiktok.com')
            
            tiktok_cookies = []
            for cookie in cj:
                if 'tiktok.com' in cookie.domain:
                    tiktok_cookies.append({
                        'name': cookie.name,
                        'value': cookie.value,
                        'domain': cookie.domain,
                        'path': cookie.path,
                        'secure': cookie.secure,
                        'expires': cookie.expires
                    })
            
            # Check for sessionid
            sessionids = [c for c in tiktok_cookies if 'sessionid' in c['name'].lower()]
            if not sessionids:
                log.error(f"❌ No sessionid found in Chrome - please login to TikTok in Chrome first")
                return False
            
            # Save in Netscape format
            with open(TIKTOK_COOKIES_FILE, "w", encoding="utf-8") as f:
                f.write("# Netscape HTTP Cookie File\n")
                f.write(f"# Auto-extracted at {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M:%S WIB')}\n\n")
                for cookie in tiktok_cookies:
                    domain = cookie['domain'].lstrip('.')
                    include_subdomains = "TRUE" if cookie['domain'].startswith('.') else "FALSE"
                    secure = "TRUE" if cookie['secure'] else "FALSE"
                    expires = cookie['expires'] if cookie['expires'] else int(time.time()) + 31536000
                    f.write(f"{domain}\t{include_subdomains}\t{cookie['path']}\t{secure}\t{expires}\t{cookie['name']}\t{cookie['value']}\n")
            
            log.info(f"✅ Extracted {len(tiktok_cookies)} cookies ({len(sessionids)} session cookies)")
            
        except Exception as e:
            log.error(f"❌ Failed to extract cookies: {e}")
            log.error("   Make sure you're logged into TikTok in Chrome")
            return False
        
        _human_pause(2, 5)  # Pre-upload pause

        # Use undetected-playwright for full automation with bot detection bypass
        log.info(f"🛡️  Using undetected-playwright for stealth upload...")
        try:
            from playwright.sync_api import sync_playwright
            import browser_cookie3
            from pathlib import Path

            UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload?from=upload&lang=en"
            STEALTH_PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiktok_stealth_profile")

            # Extract fresh cookies
            cj = browser_cookie3.chrome(domain_name='tiktok.com')
            tiktok_cookies_list = []
            for cookie in cj:
                if 'tiktok.com' in cookie.domain:
                    tiktok_cookies_list.append({
                        'name': cookie.name,
                        'value': cookie.value,
                        'domain': cookie.domain,
                        'path': cookie.path,
                        'secure': bool(cookie.secure),
                        'expires': int(cookie.expires) if cookie.expires else int(time.time()) + 31536000,
                        'httpOnly': False,
                        'sameSite': 'Lax'
                    })

            log.info(f"🍪 Extracted {len(tiktok_cookies_list)} cookies")

            # Ensure stealth profile directory exists
            os.makedirs(STEALTH_PROFILE, exist_ok=True)

            with sync_playwright() as p:
                # Launch Chrome with persistent profile
                browser = p.chromium.launch_persistent_context(
                    user_data_dir=STEALTH_PROFILE,
                    channel="chrome",
                    headless=True,
                    viewport={"width": 1280, "height": 900},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                    locale="en-US",
                    timezone_id="Asia/Jakarta",
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-infobars",
                        "--start-maximized",
                    ],
                )

                # Add extracted cookies
                browser.add_cookies(tiktok_cookies_list)
                log.info(f"🍪 Added {len(tiktok_cookies_list)} cookies to browser")

                page = browser.new_page()

                log.info(f"🌐 Navigating to TikTok upload page...")
                page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=60000)
                time.sleep(5)

                current_url = page.url
                log.info(f"✅ Upload page loaded: {current_url}")

                if "login" in current_url.lower():
                    raise Exception(f"Redirected to login: {current_url}. Please ensure you're logged into TikTok in Chrome.")

                # Build description
                unique_caption = unique_ify_caption(clip.get("caption", ""))
                comment_bait = clip.get("comment_bait", "").strip()
                description = f"{unique_caption}\n\n💬 {comment_bait}" if comment_bait else unique_caption
                description = description[:2200]

                # Upload video - TikTok uses a hidden file input triggered by UI buttons
                log.info(f"📤 Uploading video...")
                
                # Wait for the file input to exist in DOM (it's hidden but present)
                file_input = page.wait_for_selector('input[type="file"]', state='attached', timeout=15000)
                
                if not file_input:
                    log.error(f"❌ Could not find file input. Current URL: {page.url}")
                    log.error(f"   Page title: {page.title()}")
                    raise Exception("File input not found on page - TikTok UI may have changed")
                
                # Set files on the hidden input (Playwright can set files on hidden inputs)
                file_input.set_input_files(video_path)
                log.info(f"✅ Video selected: {os.path.basename(video_path)}")

                # Wait for upload to process
                log.info(f"⏳ Waiting for upload to process (this may take 1-2 minutes)...")
                time.sleep(30)

                # Dismiss any overlays/popups
                try:
                    overlay = page.query_selector('[data-test-id="overlay"]')
                    if overlay:
                        overlay.evaluate("el => el.style.display = 'none'")
                        log.info(f"✅ Dismissed tutorial overlay")
                except:
                    pass

                try:
                    got_it = page.query_selector('button:has-text("Got it")')
                    if got_it and got_it.is_visible():
                        got_it.click()
                        time.sleep(1)
                        log.info(f"✅ Clicked 'Got it'")
                except:
                    pass

                # Add description
                log.info(f"📝 Adding description...")
                desc_box = page.wait_for_selector('[contenteditable="true"]', timeout=30000)
                desc_box.click()
                time.sleep(1)
                desc_box.fill(description)
                log.info(f"📝 Description added")

                # Wait for post button to be enabled and click it
                log.info(f"⏳ Waiting for Post button...")
                post_clicked = False
                for i in range(40):  # Up to 2 minutes
                    # Dismiss overlays before each attempt
                    try:
                        overlay = page.query_selector('[data-test-id="overlay"]')
                        if overlay:
                            overlay.evaluate("el => el.style.display = 'none'")
                    except:
                        pass

                    try:
                        got_it = page.query_selector('button:has-text("Got it")')
                        if got_it and got_it.is_visible():
                            got_it.click()
                            time.sleep(0.5)
                    except:
                        pass

                    post_btn = page.query_selector('button[data-e2e="post_video_button"]')
                    if post_btn and post_btn.is_enabled():
                        log.info(f"✅ Post button is enabled!")
                        post_btn.click()
                        log.info(f"🚀 Post clicked!")
                        post_clicked = True
                        time.sleep(10)
                        break
                    
                    time.sleep(3)

                if not post_clicked:
                    raise Exception("Post button not enabled after 2 minutes")

                # Handle "Continue to post" / "Post now" confirmation popup
                log.info(f"⏳ Handling post confirmation...")
                confirm_clicked = False
                for i in range(10):
                    try:
                        # Dismiss any overlays first
                        overlay = page.query_selector('[data-test-id="overlay"]')
                        if overlay:
                            overlay.evaluate("el => el.style.display = 'none'")
                    except:
                        pass

                    try:
                        # Look for dialog/modal confirmation buttons
                        modal = page.query_selector('[role="dialog"], [class*="modal"], [class*="Dialog"]')
                        if modal:
                            buttons = modal.query_selector_all('button')
                            for btn in buttons:
                                btn_text = btn.evaluate('el => el.textContent?.trim()')
                                if btn_text and any(word in btn_text.lower() for word in ['continue', 'post', 'confirm', 'yes']):
                                    if btn.is_visible() and btn.is_enabled():
                                        log.info(f"✅ Found confirmation button: '{btn_text}'")
                                        btn.click()
                                        confirm_clicked = True
                                        time.sleep(3)
                                        break

                        if not confirm_clicked:
                            # Fallback: scan all buttons on page
                            all_buttons = page.query_selector_all('button')
                            for btn in all_buttons:
                                btn_text = btn.evaluate('el => el.textContent?.trim()?.toLowerCase()')
                                if btn_text and any(word in btn_text for word in [
                                    'continue', 'post now', 'publish', 'submit',
                                    'post video', 'confirm', 'yes'
                                ]):
                                    if btn.is_visible() and btn.is_enabled():
                                        log.info(f"✅ Found confirmation button: '{btn_text}'")
                                        btn.click()
                                        confirm_clicked = True
                                        time.sleep(3)
                                        break
                    except:
                        pass

                    if confirm_clicked:
                        break

                    time.sleep(2)

                if not confirm_clicked:
                    log.warning(f"⚠️  No confirmation popup found, proceeding anyway")

                # Wait for post to complete (page may redirect or show success)
                log.info(f"⏳ Waiting for post to complete...")
                time.sleep(15)

                # Success
                _daily_counts["tiktok"] += 1
                log.info(f"✅ TikTok: {clip.get('title', '?')} [{_daily_counts['tiktok']}/{_daily_target} today]")
                _post_login_cooldown()
                
                browser.close()
                return True

        except Exception as e:
            log.error(f"❌ TikTok upload failed: {e}")
            return False

    except Exception as e:
        error_str = str(e)
        if "No valid authentication" in error_str or "expired" in error_str.lower():
            log.error(f"❌ TikTok: not logged in - please login to TikTok in Chrome first")
            log.error(f"   Go to https://www.tiktok.com and login")
        else:
            log.error(f"❌ TikTok: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  Auto-comment functions — post comment_bait as first comment
# ═══════════════════════════════════════════════════════════

def post_tiktok_comment(uploader: "TikTokUploader", comment_text: str, video_url: str | None = None) -> bool:
    """Post a comment on the most recently uploaded TikTok video.

    Note: TikTok commenting via automation is unreliable due to frequent UI changes.
    This function uses a best-effort approach but may not always succeed.

    Args:
        uploader: TikTokUploader instance with active browser
        comment_text: The comment text to post
        video_url: Optional direct URL to the uploaded video. If provided,
                   skips profile scanning and navigates directly to the video.

    Returns:
        True if successful, False otherwise
    """
    if not comment_text or not comment_text.strip():
        log.debug("⏭️  TikTok: no comment_bait to post")
        return True  # Not an error, just nothing to do

    try:
        page = uploader.page
        log.info(f"💬 TikTok: attempting to post comment...")

        # Navigate directly to the video if URL is known
        if video_url:
            log.debug(f"📹 TikTok: navigating to known video URL")
            page.goto(video_url, wait_until="domcontentloaded", timeout=60000)
        else:
            # Fallback: navigate to user's profile to find latest video
            log.debug("📹 TikTok: no video URL provided, scanning profile")
            cookies = page.context.cookies()
            session_cookie = next((c for c in cookies if "sessionid" in c["name"]), None)
            if not session_cookie:
                log.warning("⚠️  TikTok: no session cookie found, skipping comment")
                return False

            page.goto("https://www.tiktok.com/@me?lang=en", wait_until="domcontentloaded", timeout=60000)
            _human_pause(3, 5)

            # Try to find the first (most recent) video on the profile
            # Note: TikTok may not show videos in chronological order
            video_links = page.locator("xpath=//a[contains(@href, '/video/')]")

            # Wait for at least one video link to appear
            try:
                video_links.first.wait_for(state="attached", timeout=15000)
            except Exception:
                log.warning("⚠️  TikTok: no videos found on profile, skipping comment")
                return False

            # Click on the first video (hopefully most recent)
            log.debug(f"📹 TikTok: found {video_links.count()} videos, clicking first")
            video_links.first.click()

        # Wait for video page to load and look for comment section
        _human_pause(3, 5)

        # Look for comment input field with proper wait
        comment_input = page.locator("xpath=//div[contains(@placeholder, 'Add a comment') or contains(@placeholder, 'Add comment')]")

        try:
            comment_input.first.wait_for(state="attached", timeout=15000)
        except Exception:
            log.warning("⚠️  TikTok: comment input not found, skipping comment")
            return False

        # Fill in the comment
        comment_input.first.click()
        _human_pause(1, 2)
        comment_input.first.fill(comment_text.strip())
        _human_pause(1, 2)

        # Look for send/post button - use CSS selectors (more reliable for class-based matching)
        send_selectors = [
            "button[class*='send']",
            "button:has-text('Post')",
            "div[class*='comment-action']",
            "xpath=//button[contains(@class, 'send')]",
        ]

        send_button = None
        for selector in send_selectors:
            try:
                btn = page.locator(selector)
                btn.first.wait_for(state="visible", timeout=3000)
                send_button = btn.first
                break
            except Exception:
                continue

        if send_button:
            send_button.click()
            _human_pause(2, 3)
            log.info("✅ TikTok: comment posted successfully")
            return True
        else:
            # Fallback: try pressing Enter
            log.debug("📝 TikTok: send button not found, trying Enter key")
            comment_input.first.press("Enter")
            _human_pause(2, 3)
            log.info("✅ TikTok: comment posted (via Enter key)")
            return True

    except Exception as e:
        log.warning(f"⚠️  TikTok: comment posting failed (non-fatal): {e}")
        return False  # Non-fatal, upload succeeded


def post_instagram_comment(ig: IGClient, media_id: str, comment_text: str) -> bool:
    """Post a comment on an Instagram media (clip/reel).
    
    Args:
        ig: Instagrapi client instance
        media_id: Instagram media ID (numeric string)
        comment_text: The comment text to post
        
    Returns:
        True if successful, False otherwise
    """
    if not comment_text or not comment_text.strip():
        log.debug("⏭️  Instagram: no comment_bait to post")
        return True  # Not an error, just nothing to do
    
    try:
        log.info(f"💬 Instagram: posting comment on media {media_id}...")
        _human_pause(2, 5)  # Human-like delay before commenting
        
        # Post the comment
        result = ig.media_comment(media_id=media_id, text=comment_text.strip())
        
        if result:
            log.info(f"✅ Instagram: comment posted successfully")
            return True
        else:
            log.error(f"❌ Instagram: media_comment returned False")
            return False
    except Exception as e:
        log.error(f"❌ Instagram comment failed: {type(e).__name__}: {e}")
        return False


def post_youtube_comment(yt, video_id: str, comment_text: str) -> bool:
    """Post a comment on a YouTube video.
    
    Args:
        yt: YouTube API client instance
        video_id: YouTube video ID
        comment_text: The comment text to post
        
    Returns:
        True if successful, False otherwise
    """
    if not comment_text or not comment_text.strip():
        log.debug("⏭️  YouTube: no comment_bait to post")
        return True  # Not an error, just nothing to do
    
    try:
        log.info(f"💬 YouTube: posting comment on video {video_id}...")
        _human_pause(3, 8)  # Longer human-like delay for YouTube
        
        # Create comment via YouTube Data API v3
        response = yt.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {
                            "videoId": video_id,
                            "textOriginal": comment_text.strip()
                        }
                    }
                }
            }
        ).execute()
        
        comment_id = response.get("id", "unknown")
        log.info(f"✅ YouTube: comment posted successfully (ID: {comment_id})")
        return True
    except Exception as e:
        error_str = str(e)
        if "quotaExceeded" in error_str or ("403" in error_str and "insufficientPermissions" in error_str):
            log.error("❌ YouTube: Daily API quota exceeded for comments")
            log.error("   Quota resets at midnight PT")
            log.error("   Check: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas")
            log.error("   Tip: Apply for quota increase or wait for reset")
        elif "403" in error_str:
            log.error(f"❌ YouTube: 403 Forbidden - {e}")
            log.error("   This may be a permissions issue. Try re-authenticating.")
        elif "commentsDisabled" in error_str or "comments disabled" in error_str.lower():
            log.warning(f"⚠️  YouTube: comments are disabled on this video")
            return True  # Not our fault, video has comments disabled
        else:
            log.error(f"❌ YouTube comment failed: {type(e).__name__}: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  Job factory — one closure per time slot
# ═══════════════════════════════════════════════════════════


# Post ONE clip per time slot (prevents quota exhaustion)
def make_post_job(slot_time: str, tier: int, label: str):
    def post_job():
        # --- Day-of-week slot filtering ---
        refresh_active_slots()
        if slot_time not in _active_slots:
            log.info(f"⏭️  Slot {slot_time} not active today — skipping (organic variation)")
            return

        # --- Shadowban prevention: Jitter ---
        # Instead of posting exactly at the minute, wait 0-15 minutes
        jitter_mins = random.randint(0, 15)
        jitter_secs = random.randint(0, 59)
        total_wait = (jitter_mins * 60) + jitter_secs
        
        log.info(f"🎲 Shadowban protection: delaying post for {jitter_mins}m {jitter_secs}s...")
        time.sleep(total_wait)
        # ------------------------------------

        now_dt = datetime.now(pytz.timezone("Asia/Jakarta"))
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S WIB")
        day_name = now_dt.strftime("%A")
        log.info("─" * 60)
        log.info(f"⏰  {slot_time} WIB  —  {label}  (tier {tier})  |  {now} | {day_name}")
        
        reset_daily_counts_if_needed()

        # Post ONE clip per slot (not all clips!)
        clip, video_path = get_clip_for_day(day_name)
        if clip is None:
            log.warning("📭 Queue empty — nothing to post.")
            return

        log.info(f"📹 Posting   : [score={clip.get('class_score', clip.get('clip_score', '?'))}]  rank {clip.get('rank', '?')}  — {clip.get('title', '?')}")
        log.info(f"   File      : {clip.get('filename')}")

        # --- Build the list of enabled platforms in RANDOM order ---
        # Randomising the order prevents a fixed IG→YT→TT fingerprint.
        platform_fns = []
        if ENABLE_INSTAGRAM:
            platform_fns.append(("instagram", upload_instagram))
        if ENABLE_YOUTUBE:
            platform_fns.append(("youtube", upload_youtube))
        if ENABLE_TIKTOK:
            platform_fns.append(("tiktok", upload_tiktok))
        random.shuffle(platform_fns)
        log.info(f"📤 Platform order this slot: {[p[0] for p in platform_fns]}")

        results = {}
        unique_files: list[str] = []  # track all generated files for cleanup

        for idx, (platform_name, upload_fn) in enumerate(platform_fns):
            # --- Minimum-gap enforcement ---
            elapsed = time.time() - _last_upload_ts[platform_name]
            if elapsed < MIN_GAP_BETWEEN_UPLOADS_SECS:
                wait_remaining = MIN_GAP_BETWEEN_UPLOADS_SECS - elapsed
                log.warning(
                    f"⏳ {platform_name}: last upload was {elapsed/60:.0f}m ago "
                    f"(min gap {MIN_GAP_BETWEEN_UPLOADS_SECS/60:.0f}m) — skipping this slot"
                )
                results[platform_name] = False
                continue

            # --- Per-platform unique video ---
            try:
                unique_path = unique_ify_video(video_path, platform_tag=platform_name)
                upload_path = unique_path if unique_path else video_path
                if unique_path:
                    unique_files.append(unique_path)
                    log.debug(f"✨ Generated unique-ified video [{platform_name}]: {unique_path}")
            except Exception as e:
                log.error(f"❌ Failed to create unique-ified video [{platform_name}]: {e}")
                results[platform_name] = False
                continue

            # --- Upload to platform ---
            try:
                success = upload_fn(upload_path, clip)
                results[platform_name] = success
                if success:
                    _last_upload_ts[platform_name] = time.time()
            except Exception as e:
                log.error(f"❌ Uncaught exception during {platform_name} upload: {e}")
                results[platform_name] = False

            # --- Pause between platforms ---
            if idx < len(platform_fns) - 1:
                _inter_platform_delay()

        ok = sum(bool(v) for v in results.values())
        total = len(results)
        log.info(f"📊 Upload summary: {ok}/{total} platforms succeeded — {results}")

        # --- Log upload attempt ---
        try:
            log_upload(clip, video_path, results)
        except Exception as e:
            log.warning(f"⚠️  Could not log upload: {e}")

        # --- Cleanup unique-ified video files ---
        for uf in unique_files:
            if os.path.exists(uf):
                try:
                    os.remove(uf)
                    log.debug(f"🗑️  Removed temporary file: {uf}")
                except Exception as e:
                    log.warning(f"⚠️  Could not cleanup temporary file {uf}: {e}")

        # --- Queue management based on results ---
        if ok == total and total > 0:
            log.info(f"✅ All {total} platforms succeeded — removing from queue and deleting files")
            mark_done(clip, delete_files=True)
        elif ok > 0:
            log.warning(f"⚠️  Partial success ({ok}/{total}) — removing from queue but keeping files for review")
            mark_done(clip, delete_files=False)
        else:
            log.error(f"❌ All platforms failed — keeping entry and file for retry next slot")

    post_job.__name__ = f"post_{slot_time.replace(':', '')}"
    return post_job


def run_test_post(filename: str, platform: str | None = None):
    """Run a single test upload for manual validation."""
    now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S WIB")
    log.info("=" * 70)
    log.info("🧪 TEST MODE — single-file upload")
    log.info(f"     Triggered: {now}")
    log.info(f"     Filename : {filename}")
    if platform:
        log.info(f"     Platform : {platform.upper()}")
    log.info("=" * 70)

    clip, video_path = get_clip_by_filename(filename)
    if clip is None:
        log.error(f"❌ Clip not found: {filename}")
        return

    results = {}
    # If platform is specified, only post to that platform
    if platform:
        platform_lower = platform.lower()
        if platform_lower == "tiktok" and ENABLE_TIKTOK:
            log.info(f"📱 Testing TikTok upload for: {clip.get('title', '?')}")
            results["tiktok"] = upload_tiktok(video_path, clip)
        elif platform_lower == "instagram" and ENABLE_INSTAGRAM:
            log.info(f"📷 Testing Instagram upload for: {clip.get('title', '?')}")
            results["instagram"] = upload_instagram(video_path, clip)
        elif platform_lower == "youtube" and ENABLE_YOUTUBE:
            log.info(f"📺 Testing YouTube upload for: {clip.get('title', '?')}")
            results["youtube"] = upload_youtube(video_path, clip)
        else:
            log.error(f"❌ Platform '{platform}' not recognized or not enabled")
            return
    else:
        # If no platform specified, test all enabled platforms
        log.info(f"📹 Testing all enabled platforms for: {clip.get('title', '?')}")
        if ENABLE_TIKTOK:
            _human_pause(1, 2)
            results["tiktok"] = upload_tiktok(video_path, clip)
        if ENABLE_INSTAGRAM:
            _inter_platform_delay() if results else None
            results["instagram"] = upload_instagram(video_path, clip)
        if ENABLE_YOUTUBE:
            _inter_platform_delay() if results else None
            results["youtube"] = upload_youtube(video_path, clip)

    ok = sum(results.values())
    log.info("=" * 70)
    log.info(f"📊 Test result: {ok}/{len(results)} platforms succeeded")
    for platform_name, success in results.items():
        status = "✅" if success else "❌"
        log.info(f"     {status} {platform_name.upper()}")
    log.info("=" * 70)

    # Log the test upload
    try:
        log_upload(clip, video_path, results)
    except Exception as e:
        log.warning(f"⚠️  Could not log test upload: {e}")

    if ok > 0:
        log.info(f"🧪 TEST MODE: clips.json entry and source file were NOT deleted (manual cleanup required)")
        log.info(f"   Source: {video_path}")
    else:
        log.error(f"❌ All platforms failed — check configuration and credentials")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def rebuild_queue(master_path: str, dry_run: bool = False) -> None:
    """Regenerate ``CLIPS_JSON`` from a master file by removing already-posted clips.

    If ``dry_run`` is True the new queue is printed but not written.
    """
    if not os.path.exists(master_path):
        log.error(f"master file not found: {master_path}")
        return
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)
    master = dedupe_clips(master)
    posted = get_posted_filenames()
    new_queue = [c for c in master if c.get("filename") not in posted]
    log.info(f"master contains {len(master)} clips, {len(posted)} already posted")
    log.info(f"resulting queue would have {len(new_queue)} clips")
    if dry_run:
        for c in new_queue:
            log.info(f"  {c.get('filename')} – {c.get('title')}")
        return
    save_clips(new_queue)
    log.info(f"wrote rebuilt queue to {CLIPS_JSON}")


def main(test_file: str | None = None,
         rebuild: bool = False,
         master: str | None = None,
         clean_orphans_flag: bool = False,
         prune_posted: bool = False,
         dry_run: bool = False,
        platform: str | None = None,
        reset_tiktok_browser: bool = False,
        reset_tiktok_browser_hard: bool = False):
    global RESET_TIKTOK_BROWSER_DATA, RESET_TIKTOK_BROWSER_DATA_HARD
    RESET_TIKTOK_BROWSER_DATA = reset_tiktok_browser
    RESET_TIKTOK_BROWSER_DATA_HARD = reset_tiktok_browser_hard

    log.info("=" * 60)
    log.info("  Cross-Platform Video Scheduler")
    log.info(f"  Timezone  : Asia/Jakarta (WIB, UTC+7)")
    log.info("  Platforms : " + ", ".join(
        p for p, en in [
            ("Instagram", ENABLE_INSTAGRAM),
            ("YouTube",   ENABLE_YOUTUBE),
            ("TikTok",    ENABLE_TIKTOK),
        ] if en
    ))
    log.info(f"  Clips dir : {CLIPS_FOLDER}")
    log.info("=" * 60)

    if not ensure_media_tools_ready():
        return

    # maintenance commands
    if rebuild:
        rebuild_queue(master or CLIPS_JSON, dry_run=dry_run)
        return

    if not os.path.exists(CLIPS_JSON):
        log.error(f"clips.json not found: {CLIPS_JSON}")
        return

    clips = load_clips()

    if prune_posted:
        posted = get_posted_filenames()
        before = len(clips)
        clips = [c for c in clips if c.get("filename") not in posted]
        removed = before - len(clips)
        if removed:
            log.info(f"🧹 pruned {removed} already-posted clip(s) from queue")
            if not dry_run:
                save_clips(clips)
        if dry_run:
            return

    if clean_orphans_flag:
        deleted = clean_orphan_files(clips)
        log.info(f"🧹 cleaned {deleted} orphan file(s)")
        return

    if test_file:
        run_test_post(test_file, platform=platform)
        return

    log.info(f"\n  {'Time':<8}  {'Tier':<8}  Description")
    log.info(f"  {'─'*8}  {'─'*8}  {'─'*30}")
    for t, tier, label in sorted(SCHEDULE_SLOTS, key=lambda s: s[0]):
        stars = "★" * (4 - tier)
        log.info(f"  {t:<8}  {stars:<8}  {label}")

    # Refresh and log today's active schedule
    reset_daily_counts_if_needed()
    refresh_active_slots()
    
    today = datetime.now(pytz.timezone("Asia/Jakarta")).date()
    today_name = today.strftime("%A")
    engagement = DAY_ENGAGEMENT.get(today_name, 1.0)
    
    log.info(f"\n📅 Today's Schedule ({today_name}, {today})")
    log.info(f"   Engagement multiplier: {engagement:.2f}")
    log.info(f"   Daily upload target: {_daily_target}/platform")
    log.info(f"   Active slots: {len(_active_slots)}")
    
    if _active_slots:
        # Build a lookup for slot details
        slot_lookup = {t: (tier, label) for t, tier, label in SCHEDULE_SLOTS}
        log.info(f"\n   🕐 Today's active posting times:")
        for slot_time in sorted(_active_slots):
            tier, label = slot_lookup.get(slot_time, ("?", "Unknown"))
            stars = "★" * (4 - tier)
            log.info(f"      • {slot_time} ({stars}) - {label}")
    else:
        log.info("   🚫 No active slots today (rest day)")

    log.info(f"\n📋 {len(clips)} clips in queue at startup")
    log.info("   (clips.json is re-read fresh before every upload)")
    log.info("")
    log.info(f"📊 Weekly uploads: 6 active days + 1 rest day, active days {DAILY_ACTIVE_POSTS} posts — shuffled each Monday")
    log.info(f"   Distribution: randomized {DAILY_ACTIVE_POSTS} posts/day on active days, with one recovery day")

    for slot_time, tier, label in SCHEDULE_SLOTS:
        job_fn = make_post_job(slot_time, tier, label)
        schedule.every().day.at(slot_time, "Asia/Jakarta").do(job_fn)

    log.info("🚀 Running — Ctrl+C to stop.\n")

    # Track last logged date for midnight crossing detection
    last_logged_date = datetime.now(pytz.timezone("Asia/Jakarta")).date()

    while True:
        schedule.run_pending()

        # Check if we've crossed midnight
        current_date = datetime.now(pytz.timezone("Asia/Jakarta")).date()
        if current_date != last_logged_date:
            log.info("\n" + "="*60)
            log.info("🌙 Midnight passed — new day detected!")
            log.info("="*60)

            # Refresh and log the new day's schedule
            reset_daily_counts_if_needed()
            refresh_active_slots()

            today_name = current_date.strftime("%A")
            engagement = DAY_ENGAGEMENT.get(today_name, 1.0)

            log.info(f"\n📅 Today's Schedule ({today_name}, {current_date})")
            log.info(f"   Engagement multiplier: {engagement:.2f}")
            log.info(f"   Daily upload target: {_daily_target}/platform")
            log.info(f"   Active slots: {len(_active_slots)}")

            if _active_slots:
                slot_lookup = {t: (tier, label) for t, tier, label in SCHEDULE_SLOTS}
                log.info(f"\n   🕐 Today's active posting times:")
                for slot_time in sorted(_active_slots):
                    tier, label = slot_lookup.get(slot_time, ("?", "Unknown"))
                    stars = "★" * (4 - tier)
                    log.info(f"      • {slot_time} ({stars}) - {label}")
            else:
                log.info("   🚫 No active slots today (rest day)")

            log.info(f"\n📋 {len(load_clips())} clips in queue")
            log.info("="*60 + "\n")

            last_logged_date = current_date

        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-platform video scheduler")
    parser.add_argument(
        "--test-file",
        help="Run one immediate test post using filename from clips.json; no deletion is performed.",
    )
    parser.add_argument(
        "--rebuild-queue",
        action="store_true",
        help="Regenerate the queue by removing clips that have already been posted. Requires --master if you want to use a different source."
    )
    parser.add_argument(
        "--master",
        help="Path to master clips.json to rebuild from (defaults to clips.json).",
    )
    parser.add_argument(
        "--clean-orphans",
        action="store_true",
        help="Delete any mp4 files in the clips folder that are not referenced in the current queue.",
    )
    parser.add_argument(
        "--prune-posted",
        action="store_true",
        help="Remove already-posted clips from the queue (based on upload logs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When used with other maintenance flags, do not write changes; only show what would happen.",
    )
    parser.add_argument(
        "--platform",
        help="Specify a platform to post to (instagram, youtube, or tiktok). Used with --test-file to post to a single platform.",
    )
    parser.add_argument(
        "--reset-tiktok-browser-data",
        action="store_true",
        help="Reset TikTok Playwright transient browser data while preserving session.",
    )
    parser.add_argument(
        "--reset-tiktok-browser-data-hard",
        action="store_true",
        help="Hard reset TikTok Playwright profile (deletes full profile and login state).",
    )
    args = parser.parse_args()
    main(
        test_file=args.test_file,
        rebuild=args.rebuild_queue,
        master=args.master,
        clean_orphans_flag=args.clean_orphans,
        prune_posted=args.prune_posted,
        dry_run=args.dry_run,
        platform=args.platform,
        reset_tiktok_browser=args.reset_tiktok_browser_data,
        reset_tiktok_browser_hard=args.reset_tiktok_browser_data_hard,
    )