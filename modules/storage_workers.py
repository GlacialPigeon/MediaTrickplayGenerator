# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""How many videos are processed at once on one disk.

A hard disk reads one video well and two badly (the heads jump between the files: with 2 videos at once a run took half as long again,
with 4 more than twice as long). An SSD or NVMe disk reads several at once without trouble, and two videos at once let the graphics
card work while the other video is being read, started and packed. So the default depends on the kind of disk ("auto"); a number set by
the person is used for every disk as it is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import storage

AUTO = "auto"
AUTO_WORKERS = {"ssd": 2, "hdd": 1}
MIN_WORKERS, MAX_WORKERS = 1, 4


def is_auto(setting: Any) -> bool:
    """True for "auto" and for anything that is not a number of workers (a damaged settings file must not stop Generate)."""
    return _number(setting) is None


def _number(setting: Any) -> int | None:
    if isinstance(setting, bool) or setting is None: return None
    try: return max(MIN_WORKERS, min(MAX_WORKERS, int(setting)))
    except (TypeError, ValueError): return None


def per_disk(setting: Any, kind: str) -> int:
    """Videos at once on a disk of this kind ("ssd", "hdd" or "unknown") under the setting (a number, or "auto")."""
    number = _number(setting)
    return number if number is not None else AUTO_WORKERS.get(kind, AUTO_WORKERS["hdd"])


def group_kind(paths: Iterable[Path]) -> str:
    """"ssd" when every one of the videos is on an SSD, "hdd" when one is on a hard disk or a share, else "unknown" (a group that mixes
    kinds is limited by its slowest disk; unknown disks get the hard disk number of videos but are not called hard disks)."""
    seen: dict[Path, str] = {}
    for path in paths:
        folder = path.parent
        if folder not in seen: seen[folder] = storage.storage_kind(path)
    kinds = set(seen.values())
    if "hdd" in kinds: return "hdd"
    if not kinds or "unknown" in kinds: return "unknown"
    return "ssd"


def kind_label(kind: str) -> str:
    """How a kind of disk is written in reports and logs: "SSD", "HDD", "network" or "unknown (treated as HDD)"."""
    if kind == "network": return "network"
    if kind == "unknown": return "unknown (treated as HDD)"
    return kind.upper()


def layout(items: Iterable[Any], setting: Any) -> dict[str, tuple[str, int]]:
    """For every storage group: (kind of disk, videos at once). `items` are plan items (`storage_group`, `meta.path`)."""
    paths: dict[str, list[Path]] = {}
    for item in items: paths.setdefault(item.storage_group, []).append(item.meta.path)
    result: dict[str, tuple[str, int]] = {}
    for group, group_paths in paths.items():
        kind = group_kind(group_paths)
        result[group] = (kind, per_disk(setting, kind))
    return result
