# ============================================================
#  config.py  –  Edit this file with your credentials & paths
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Folder that contains clips.json + video files ───────────
CLIPS_FOLDER = os.path.join(BASE_DIR, "clips")   # <-- CHANGE THIS

# ── Upload times (24-h, server local time) ──────────────────
SCHEDULE_TIMES = ["06:00", "10:00", "14:00", "18:00", "20:00", "23:00"]

# ── Instagram ────────────────────────────────────────────────
INSTAGRAM_SESSION_FILE = os.path.join(BASE_DIR, "ig_session.json")
INSTAGRAM_CREDENTIAL_FILE = os.path.join(BASE_DIR, "ig_credentials.json")  # {"username": "your_ig_username", "password": "your_ig_password"}

# ── YouTube ──────────────────────────────────────────────────
YOUTUBE_CLIENT_SECRETS = os.path.join(BASE_DIR, "client_secrets.json")  # downloaded from Google Cloud
YOUTUBE_TOKEN_FILE     = os.path.join(BASE_DIR, "yt_token.pickle")      # saved after first auth
YOUTUBE_CATEGORY_ID    = "28"                   # 22 = People & Blogs
YOUTUBE_CHANNEL_HANDLE = "@samuel.koesnadi"     # @yourchannelhandle (optional, for validation)
YOUTUBE_PRIVACY_STATUS = "public"               # public, unlisted, or private
YOUTUBE_DEFAULT_LANGUAGE = "id"                 # id = Indonesian
YOUTUBE_DEFAULT_AUDIO_LANGUAGE = "id"           # id = Indonesian
YOUTUBE_LICENSE = "youtube"                     # youtube or creativeCommon
YOUTUBE_EMBEDDABLE = True
YOUTUBE_PUBLIC_STATS_VIEWABLE = True
YOUTUBE_NOTIFY_SUBSCRIBERS = False              # Don't spam subscribers for each upload
YOUTUBE_SELF_DECLARED_MADE_FOR_KIDS = False

# ── TikTok ───────────────────────────────────────────────────
# ── TikTok ───────────────────────────────────────────────────
TIKTOK_COOKIES_FILE = os.path.join(BASE_DIR, "tiktok_cookies.txt")      # exported from browser
TIKTOK_BROWSER_DATA_DIR = os.path.join(BASE_DIR, "tiktok_browser_data") # persistent browser cookies/cache/storage

# Use your actual Chrome profile (all cookies, sessions, logins included)
CHROME_PROFILE_DIR = os.path.expanduser(
    "~/Library/Application Support/Google/Chrome/Default"
)

# ── FFmpeg video output settings ────────────────────────────
VIDEO_RESOLUTION = "1080:1920"   # vertical 9:16
VIDEO_FPS        = 30
MAX_DURATION_YT  = 180           # YouTube Shorts max = 3 min
MAX_DURATION_IG  = 90            # Instagram Reels max = 90 sec
MAX_DURATION_TT  = 180           # TikTok max (3 min; up to 10 min allowed)

# ── Which platforms to post to ───────────────────────────────
ENABLE_INSTAGRAM = True
ENABLE_YOUTUBE   = True
ENABLE_TIKTOK    = True

# ── Delete video file after successful upload to ALL platforms?
DELETE_AFTER_UPLOAD = True

# ── FFmpeg paths ─────────────────────────────────────────────
FFMPEG_BIN = "ffmpeg"    # or full path like "C:\\ffmpeg\\bin\\ffmpeg.exe"
FFPROBE_BIN = "ffprobe"  # or full path like "C:\\ffmpeg\\bin\\ffprobe.exe"
