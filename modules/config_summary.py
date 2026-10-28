# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The current settings as short texts, for the screens that show what is set."""
from __future__ import annotations

from typing import Any

from .constants import JELLYFIN_IMAGE_PRESET
from .localization import t


def image(settings: dict[str, Any]) -> str:
    return t("summary.image", width=settings["width"], interval=f"{float(settings['interval_seconds']):g}", columns=settings["tile_columns"], rows=settings["tile_rows"],
              jpeg=settings["jpeg_quality"], qscale=settings["ffmpeg_qscale"])


def preset(settings: dict[str, Any]) -> str:
    return t("summary.preset_jellyfin") if all(settings.get(key) == value for key, value in JELLYFIN_IMAGE_PRESET.items()) else t("summary.preset_custom")


def key_frames(settings: dict[str, Any]) -> str:
    if not settings.get("keyframe_only", False): return t("common.off")
    limit = float(settings.get("keyframe_repeat_limit_percent", 10.0))
    return t("summary.key_frames_on_limit", limit=f"{limit:g}") if limit > 0 else t("summary.key_frames_on_never")


def fingerprint(source_detection: dict[str, Any]) -> str:
    return t("common.on") if source_detection.get("strong_fingerprint", False) else t("common.off")
