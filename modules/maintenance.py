# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from .constants import DIRS
from .tiles import folder_size

KEEP_NAMES = {".gitkeep"}
LOG_RETENTION_DAYS = 90
# What the logs folder holds: a folder per start of the program (<time>_<id>/), crash logs and the files of earlier versions.
SESSION_FOLDER = re.compile(r"\d{4}-\d\d-\d\d_\d{6}_[0-9a-f]{8}(_[\w-]+)?")


def usage_report() -> dict[str, int]:
    """Bytes used by each application folder."""
    return {name: folder_size(DIRS[name]) for name in ("logs", "diagnostics", "temp", "data", "config")}


def item_size(path: Path) -> int:
    try:
        return folder_size(path) if path.is_dir() else path.stat().st_size
    except OSError:
        return 0


def old_items(base: Path, days: int, *, include_dirs: bool, now: float | None = None) -> list[Path]:
    """Entries directly inside `base` older than `days` days (files only unless include_dirs)."""
    cutoff = (time.time() if now is None else now) - days * 86400
    found: list[Path] = []
    if not base.is_dir():
        return found
    for entry in base.iterdir():
        if entry.name in KEEP_NAMES:
            continue
        try:
            if entry.stat().st_mtime < cutoff and (include_dirs or entry.is_file()):
                found.append(entry)
        except OSError:
            continue
    return found


def all_items(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return [entry for entry in base.iterdir() if entry.name not in KEEP_NAMES]


def remove_items(items: list[Path]) -> tuple[int, list[tuple[Path, str]]]:
    """Delete files / folders; returns (removed count, [(path, error)])."""
    removed = 0
    failed: list[tuple[Path, str]] = []
    for entry in items:
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except Exception as exc:
            failed.append((entry, str(exc)))
    return removed, failed


def last_activity(entry: Path) -> float:
    """When a log was last written: a folder of a start is as new as its session.log (which is appended to all through the start)."""
    stamp = entry.stat().st_mtime
    if entry.is_dir():
        try: stamp = max(stamp, (entry / "session.log").stat().st_mtime)
        except OSError: pass
    return stamp


def old_logs(days: int, *, keep: Path | None = None, now: float | None = None) -> list[Path]:
    """The folders of starts and the log files in the logs folder that are older than `days` days (never `keep`)."""
    cutoff = (time.time() if now is None else now) - days * 86400
    found: list[Path] = []
    if not DIRS["logs"].is_dir(): return found
    for entry in DIRS["logs"].iterdir():
        if entry == keep or entry.name in KEEP_NAMES: continue
        try:
            if entry.is_dir():
                if not SESSION_FOLDER.fullmatch(entry.name): continue
            elif entry.suffix not in {".log", ".json", ".txt", ".jsonl"}: continue
            if last_activity(entry) < cutoff: found.append(entry)
        except OSError: continue
    return found


def prune_old_logs(days: int = LOG_RETENTION_DAYS, *, keep: Path | None = None, now: float | None = None) -> int:
    """Remove the logs older than `days` days (never the one named in `keep`). Returns how many were removed.
    Problems are ignored: tidying up must never stop a run."""
    try:
        removed, _ = remove_items(old_logs(days, keep=keep, now=now))
        return removed
    except OSError:
        return 0
