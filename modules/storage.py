# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

from pathlib import Path

from . import platform_api


def canonical_media_path(path: Path) -> str:
    """Stable identity string: mapped drives / mount points that reach the same share resolve to one string."""
    return platform_api.canonical_media_path(path)


def physical_storage_group(path: Path) -> str:
    """Name of the physical disk or network server a path lives on; used to limit concurrency per storage."""
    return platform_api.storage_group(path)


def storage_kind(path: Path) -> str:
    """"ssd" (SSD, NVMe), "hdd" (a hard disk or a network share) or "unknown" (could not be identified; handled like a hard disk)."""
    try: return platform_api.storage_kind(path)
    except Exception: return "unknown"


def mount_group(path: Path) -> str:
    """Coarser grouping used when the user treats each drive letter / mount / share as separate storage."""
    return platform_api.mount_group(path)
