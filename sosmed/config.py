"""
Configuration loader for config.yaml

The default configuration lives in YAML (DEFAULT_CONFIG_YAML below) so it is
defined in the same format as config.yaml and can be kept in sync by editing
YAML, not a Python dict. At runtime the values from config.yaml (project root)
are merged on top of these defaults.
"""

import os
from pathlib import Path
from typing import Any

import yaml

from .utils import log

# Canonical default configuration, written in YAML so it matches config.yaml.
# config.yaml (project root) overrides these values when present.
DEFAULT_CONFIG_YAML = """
defaults:
  whisper_model: turbo  # Options: tiny, base, small, medium, large-v2, large-v3, distil-large-v3, turbo
  language: en  # 'id' for Indonesian, 'en' for English, null for auto-detect
  min_clip_duration: 15
  max_clip_duration: 180
  max_clips: 28
  min_score: 55
  device: auto  # Options: auto, cuda, cpu
  compute_type: auto  # Options: auto, float16, int8, int8_float16
  batch_size: 16
  vad_enabled: true
  vad_min_silence_ms: 400
  vad_speech_pad_ms: 200
  chunk_duration: 360.0  # seconds
  chunk_overlap: 60.0  # seconds
  llm_parallel: false  # run LLM chunk + per-clip subtitle calls concurrently
  output_dir: clips
  subtitles_enabled: true
  subtitle_position: lower  # Options: center, upper, lower
  subtitle_font_size_pct: 3.2  # font size as % of vertical resolution
  subtitle_margin_pct: 10.0  # 10% from bottom for "lower" position
  silence_removal_enabled: true
  max_silence_duration: 1.5  # seconds
  encoding_preset: veryfast  # ffmpeg x264 preset
  encoding_crf: 23  # Quality: 18-28 (lower=better, 23=good balance for social media)
  hwaccel: true  # Hardware acceleration (VideoToolbox / NVENC)

pixabay:
  min_duration: 30
  per_page: 5

cta:
  enabled: false
  name: "Samuel Academy"
  username: "@samuelkoesnadi"
  duration: 3.0
  fade_duration: 0.5

openrouter:
  model: openrouter/free
  base_url: https://openrouter.ai/api/v1
"""

DEFAULT_CONFIG: dict[str, Any] = yaml.safe_load(DEFAULT_CONFIG_YAML) or {}
del DEFAULT_CONFIG_YAML

_config_cache: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from config.yaml.

    Args:
        config_path: Path to config file. If None, looks for:
                     1. ./config.yaml
                     2. ./config.yaml.example (fallback)

    Returns:
        Configuration dictionary with defaults merged in.
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    if config_path is None:
        config_path = Path("config.yaml")
        if not config_path.exists():
            example_path = Path("config.yaml.example")
            if example_path.exists():
                config_path = example_path
                log("INFO", f"Using example config: {config_path}")
            else:
                log("INFO", "No config.yaml found, using defaults")
                _config_cache = DEFAULT_CONFIG
                return _config_cache

    config_path = Path(config_path)

    if not config_path.exists():
        log("WARN", f"Config file not found: {config_path}, using defaults")
        _config_cache = DEFAULT_CONFIG
        return _config_cache

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Merge with defaults
        merged = _merge_configs(DEFAULT_CONFIG, user_config)
        _config_cache = merged
        log("OK", f"Config loaded: {config_path}")
        return merged

    except Exception as e:
        log("ERROR", f"Failed to load config: {e}")
        log("INFO", "Using default configuration")
        _config_cache = DEFAULT_CONFIG
        return _config_cache


def _merge_configs(defaults: dict, user: dict) -> dict:
    """Recursively merge user config into defaults."""
    result = defaults.copy()

    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_defaults() -> dict[str, Any]:
    """Get default CLI parameters."""
    config = load_config()
    return config.get("defaults", DEFAULT_CONFIG["defaults"])


def get_pixabay_settings() -> dict[str, Any]:
    """Get Pixabay download settings."""
    config = load_config()
    return config.get("pixabay", DEFAULT_CONFIG["pixabay"])


def get_cta_settings() -> dict[str, Any]:
    """Get Instagram CTA settings."""
    config = load_config()
    return config.get("cta", DEFAULT_CONFIG["cta"])


def get_openrouter_settings() -> dict[str, Any]:
    """Get OpenRouter LLM settings."""
    config = load_config()
    return config.get("openrouter", DEFAULT_CONFIG["openrouter"])


def reload_config() -> dict[str, Any]:
    """Force reload configuration from disk."""
    global _config_cache
    _config_cache = None
    return load_config()
