# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

APP_NAME = "Media Trickplay Generator"
APP_VERSION = "1.0.0"
CONFIG_SCHEMA = 1
STATE_SCHEMA = 1
FFMPEG_OFFICIAL_URL = "https://ffmpeg.org/download.html"
# FFmpeg is taken from this folder beside main.py first. Only on Linux the FFmpeg of the system (PATH) is the fallback; a configured path is never used.
FFMPEG_DIR_NAME = "ffmpeg"

KNOWN_VIDEO_EXTENSIONS = {
    ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".qt", ".wmv", ".asf", ".ts", ".m2ts",
    ".mts", ".webm", ".mpg", ".mpeg", ".mpe", ".vob", ".ogv", ".3gp", ".3g2", ".flv",
    ".f4v", ".rm", ".rmvb", ".divx", ".mpv", ".mxf", ".nut", ".y4m", ".dav", ".wtv", ".dvr-ms",
}
KNOWN_NONVIDEO_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif",
    ".avif", ".ico", ".svg", ".srt", ".ass", ".ssa", ".sub", ".idx", ".vtt", ".sup", ".smi",
    ".sami", ".ttml", ".dfxp", ".lrc", ".nfo", ".txt", ".xml", ".json", ".yaml", ".yml",
    ".log", ".ini", ".cfg", ".conf", ".md", ".html", ".htm", ".url", ".mp3", ".flac", ".aac",
    ".m4a", ".ogg", ".opus", ".wav", ".wma", ".cue", ".m3u", ".m3u8", ".pls", ".zip", ".7z",
    ".rar", ".tar", ".gz", ".bz2", ".xz", ".iso", ".exe", ".dll", ".bat", ".cmd", ".ps1",
    ".lnk", ".db", ".sqlite", ".part", ".tmp", ".torrent", ".csv", ".tsv",
}

PROBE_WORKERS = 8      # ffprobe workers per storage group while scanning

# What Jellyfin makes itself (library setting "Trickplay" defaults: 320 px wide, one picture every 10 s, 10 x 10 pictures per tile).
JELLYFIN_IMAGE_PRESET: dict[str, Any] = {"width": 320, "interval_seconds": 10.0, "tile_columns": 10, "tile_rows": 10, "ffmpeg_qscale": 4, "jpeg_quality": 90}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": CONFIG_SCHEMA, "language": "en-US", "media_roots": [], "excluded_folders": [],
    "trickplay": {**JELLYFIN_IMAGE_PRESET, "same_width_policy": "skip_valid", "other_width_policy": "keep_add", "keyframe_only": False, "keyframe_repeat_limit_percent": 10.0},
    "acceleration": {"mode": "auto", "preferred_backend": "auto", "gpu_speed_threshold": 1.15, "max_gpu_jobs": 2, "hdr_validation": None, "gpu_decoding": None, "path_choice": None, "benchmark_seen": None},
    "workers": {"storage_mode": "auto", "per_storage_group": "auto", "manual_groups": {}},
    "disk_speed": {},      # MB/s the benchmark read from the drives of the library, by kind of disk ("hdd", "ssd"); for the time estimate
    "source_detection": {"strong_fingerprint": False, "fingerprint_chunk_mib": 1},
    "verification": {"verify_all_tiles": True}, "runtime": {"prevent_sleep": True},
    "ffmpeg": {"selected_hash": None},
    "diagnostics": {"benchmark_folder": "", "hdr_folder": ""},
}


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # modules/constants.py -> application root is one level above modules/.
    return Path(__file__).resolve().parent.parent


APP_ROOT = _app_root()

LOCALES_DIR = APP_ROOT / "locales"           # beside main.py: a language is added by putting its file here

# Mutable on purpose: tests (and nothing else) redirect the application directories by updating this dict in place.
DIRS: dict[str, Path] = {
    "config": APP_ROOT / "config",
    "data": APP_ROOT / "data",
    "logs": APP_ROOT / "logs",
    "diagnostics": APP_ROOT / "diagnostics",
    "temp": APP_ROOT / "temp",
}
