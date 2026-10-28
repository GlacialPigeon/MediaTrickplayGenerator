# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

from .datetime_helpers import local_stamp
from .models import TrickplaySettings, VideoMeta
from .paths import could_be_video
from .state_db import StateDB
from .tiles import (
    building_dir_for, describe_frame_deviation, folder_size, frame_count_deviation, list_other_widths, parse_trickplay_width_dir, target_dir_for,
    trickplay_root_for, validate_tile_directory,
)

VALID = "VALID"
MISSING = "MISSING"
INVALID = "INVALID"
SOURCE_CHANGED = "SOURCE_CHANGED"
SETTINGS_CHANGED = "SETTINGS_CHANGED"


OLD_MARKER = ".__old__."
FAILED_MARKER = ".__failed_finalize__."
_SPRITE_NAME = re.compile(r"\d+\.jpg", re.IGNORECASE)


@dataclass
class Leftover:
    """A folder left behind by an interrupted or failed replacement, and whether it may be removed."""
    path: Path
    kind: str                      # "old" (copy of the previous output) | "failed" (new output that failed validation)
    safe: bool
    reason: str
    size: int = 0
    protected_by: Optional[Path] = None                 # "old" only: the output that replaced it and must still be valid
    protected_settings: Optional[TrickplaySettings] = None
    protected_tiles: Optional[int] = None
    duration: float = 0.0


@dataclass
class Restorable:
    """A previous output (`.__old__` folder) that is still valid and can take the place of a missing or damaged output."""
    path: Path          # the .__old__ folder
    target: Path        # where it belongs
    tiles: int
    duration: float
    settings: TrickplaySettings


@dataclass
class VerifyRow:
    path: Path
    status: str
    reason: str = ""
    tiles: int = 0
    stale_building: bool = False
    leftovers: tuple[str, ...] = ()
    leftover_items: tuple[Leftover, ...] = ()
    restorable: Optional[Restorable] = None
    other_widths: tuple[str, ...] = ()      # folders of another width / tile layout next to the target (640 - 10x10): not a problem, just reported
    suspect: str = ""              # a valid output whose number of thumbnails does not fit the video length (see frame_count_deviation)


@dataclass
class Orphan:
    """A `.trickplay` folder whose video is gone, and whether it may be removed."""
    path: Path
    safe: bool
    reason: str
    size: int = 0


def _recognized_output_folder(child: Path) -> bool:
    """A width folder (`640 - 10x10`), or a building / backup / failed copy of one, holding nothing but sprite images."""
    name = child.name
    for marker in (OLD_MARKER, FAILED_MARKER):
        if marker in name:
            name = name.split(marker, 1)[0]
            break
    else:
        if name.endswith(".__building__"): name = name[:-len(".__building__")]
    return parse_trickplay_width_dir(Path(name)) is not None and _only_sprites(child)


def assess_orphan(path: Path) -> Orphan:
    """Removable only when it holds nothing but trickplay output of this program (or Jellyfin); anything else is left to the user."""
    size = folder_size(path)
    try: children = list(path.iterdir())
    except OSError as exc: return Orphan(path, False, f"cannot be read: {exc}", size)
    for child in children:
        if not _recognized_output_folder(child):
            return Orphan(path, False, f"contains something that is not trickplay output: {child.name}", size)
    return Orphan(path, True, "the video is gone; only trickplay images inside", size)


def remove_orphan(item: Orphan) -> Optional[str]:
    """Remove one orphaned folder after checking everything again. Returns None on success, otherwise why it was kept."""
    path = item.path
    if not item.safe: return "not marked safe to remove"
    if not path.name.casefold().endswith(".trickplay"): return "not a .trickplay folder"
    if path.is_symlink() or not path.is_dir(): return "not a plain folder"
    stem = path.name[:-len(".trickplay")].casefold()
    try:
        with os.scandir(path.parent) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and could_be_video(entry.name) and Path(entry.name).stem.casefold() == stem:
                    return "a video with this name is there again"
    except OSError as exc: return f"could not check the folder of the video: {exc}"
    if not assess_orphan(path).safe: return "content changed: it holds something that is not trickplay output"
    try: shutil.rmtree(path)
    except OSError as exc: return str(exc)
    return None


def summary_template(total: int) -> dict[str, int]:
    return {"total": total, "valid": 0, "missing": 0, "invalid": 0, "source_changed": 0,
            "settings_changed": 0, "unexpected": 0, "stale_building": 0, "leftover_dirs": 0, "orphan_dirs": 0, "unreadable_videos": 0, "suspect": 0, "canceled": 0}


def find_leftovers(video: Path) -> tuple[str, ...]:
    """Folders left behind by an interrupted or failed replacement (backup / failed-finalize copies)."""
    root = trickplay_root_for(video)
    if not root.is_dir():
        return ()
    try:
        return tuple(sorted(child.name for child in root.iterdir()
                            if child.is_dir() and (".__old__." in child.name or ".__failed_finalize__." in child.name)))
    except OSError:
        return ()


def _leftover_kind(name: str) -> Optional[tuple[str, str]]:
    for marker, kind in ((OLD_MARKER, "old"), (FAILED_MARKER, "failed")):
        if marker in name:
            return kind, name.split(marker, 1)[0]
    return None


def _only_sprites(path: Path) -> bool:
    """True for a real (non-link) folder that contains nothing but N.jpg files."""
    try:
        if path.is_symlink() or not path.is_dir(): return False
        return all(child.is_file() and not child.is_symlink() and _SPRITE_NAME.fullmatch(child.name) for child in path.iterdir())
    except OSError:
        return False


def assess_leftovers(meta: VideoMeta, settings: TrickplaySettings, *, current_valid: bool, current_tiles: int = 0,
                     verify_all: bool = True) -> list[Leftover]:
    """Decide for every leftover folder of a video whether removing it can lose data.

    * a failed-finalize folder never held a valid output, so it is removable;
    * an old copy is removable only while the output that replaced it is valid (otherwise it may be the last good one);
    * anything that contains something other than sprite images is never removable.
    """
    root = trickplay_root_for(meta.path)
    items: list[Leftover] = []
    for name in find_leftovers(meta.path):
        parsed = _leftover_kind(name)
        if parsed is None: continue
        kind, prefix = parsed
        path = root / name
        size = folder_size(path)
        if not _only_sprites(path):
            items.append(Leftover(path, kind, False, "contains files other than sprite images", size))
            continue
        if kind == "failed":
            items.append(Leftover(path, kind, True, "new output that failed validation", size))
            continue
        if prefix == settings.folder_name:
            protected, protected_settings = target_dir_for(meta.path, settings), settings
            valid, tiles = current_valid, current_tiles
        else:
            dims = parse_trickplay_width_dir(root / prefix)
            if dims is None:
                items.append(Leftover(path, kind, False, "unrecognized folder name", size))
                continue
            try: protected_settings = replace(settings, width=dims[0], cols=dims[1], rows=dims[2])
            except ValueError:
                items.append(Leftover(path, kind, False, "unrecognized folder name", size))
                continue
            protected = root / prefix
            check = validate_tile_directory(protected, protected_settings, meta.duration, verify_all=verify_all)
            valid, tiles = check.valid, check.tiles
        if valid:
            items.append(Leftover(path, kind, True, "previous copy; the current output is valid", size, protected, protected_settings, tiles, meta.duration))
        else:
            items.append(Leftover(path, kind, False, "may be the last good output: the current output is missing or invalid", size))
    return items


def remove_leftover(item: Leftover) -> Optional[str]:
    """Remove one leftover folder after re-checking every condition. Returns None on success, otherwise why it was kept."""
    if not item.safe: return "not marked safe to remove"
    parsed = _leftover_kind(item.path.name)
    if parsed is None or parsed[0] != item.kind: return "unexpected folder name"
    if not item.path.parent.name.casefold().endswith(".trickplay"): return "not inside a .trickplay folder"
    if not _only_sprites(item.path): return "folder content changed or is not a plain sprite folder"
    if item.kind == "old":
        if item.protected_by is None or item.protected_settings is None: return "no replacement output recorded"
        check = validate_tile_directory(item.protected_by, item.protected_settings, item.duration,
                                        expected_exact_tiles=item.protected_tiles or None, verify_all=True)
        if not check.valid: return "the current output is no longer valid; previous copy kept"
    try: shutil.rmtree(item.path)
    except OSError as exc: return str(exc)
    return None


def find_restorable(meta: VideoMeta, settings: TrickplaySettings, row: VerifyRow, *, verify_all: bool = True) -> Optional[Restorable]:
    """The newest `.__old__` copy of the current target that is still a valid output - only for a missing or damaged output
    whose source did not change (an old copy of a changed source would be outdated, too)."""
    if row.status not in {MISSING, INVALID} or not row.leftovers: return None
    root = trickplay_root_for(meta.path)
    for name in sorted(row.leftovers, reverse=True):                       # names carry a sortable time stamp: newest first
        parsed = _leftover_kind(name)
        if parsed is None or parsed[0] != "old" or parsed[1] != settings.folder_name: continue
        old = root / name
        if not _only_sprites(old): continue
        check = validate_tile_directory(old, settings, meta.duration, verify_all=verify_all)
        if check.valid:
            return Restorable(old, target_dir_for(meta.path, settings), check.tiles, meta.duration, settings)
    return None


def restore_previous(item: Restorable) -> Optional[str]:
    """Put a previous output back in place by renaming it (nothing is copied). A damaged output in its way is moved aside,
    never deleted. Returns None on success, otherwise why nothing was changed."""
    old, target = item.path, item.target
    parsed = _leftover_kind(old.name)
    if parsed is None or parsed[0] != "old" or parsed[1] != target.name: return "unexpected folder name"
    root = old.parent
    if not root.name.casefold().endswith(".trickplay") or target.parent != root: return "not inside the video's .trickplay folder"
    if not _only_sprites(old): return "folder content changed or is not a plain sprite folder"
    check = validate_tile_directory(old, item.settings, item.duration, verify_all=True)
    if not check.valid: return "the previous output is no longer valid: " + check.reason
    moved_aside: Optional[Path] = None
    if target.is_symlink(): return "the target is a link; left alone"
    if target.exists():
        if not _only_sprites(target): return "the target folder contains files other than sprite images; left alone"
        moved_aside = root / f"{target.name}{FAILED_MARKER}{local_stamp()}.{uuid.uuid4().hex[:6]}"
        try: os.replace(target, moved_aside)
        except OSError as exc: return f"could not move the damaged output aside: {exc}"
    try:
        os.replace(old, target)
    except OSError as exc:
        if moved_aside is not None:
            try: os.replace(moved_aside, target)
            except OSError: return f"{exc} (the damaged output is at {moved_aside})"
        return str(exc)
    return None


def verify_video(meta: VideoMeta, settings: TrickplaySettings, state: StateDB, *, verify_all: bool = True) -> VerifyRow:
    """Check one video's target trickplay directory against the processing history and the files on disk."""
    row = _verify_core(meta, settings, state, verify_all=verify_all)
    row.other_widths = tuple(sorted(path.name for path in list_other_widths(meta.path, settings)))
    if row.leftovers:
        row.leftover_items = tuple(assess_leftovers(meta, settings, current_valid=row.status in {VALID, SETTINGS_CHANGED},
                                                    current_tiles=row.tiles, verify_all=verify_all))
        row.restorable = find_restorable(meta, settings, row, verify_all=verify_all)
    return row


def _verify_core(meta: VideoMeta, settings: TrickplaySettings, state: StateDB, *, verify_all: bool = True) -> VerifyRow:
    target = target_dir_for(meta.path, settings)
    stale = building_dir_for(meta.path, settings).exists()
    leftovers = find_leftovers(meta.path)
    previous = state.current_target_output(meta.path, settings)

    source_changed = previous is not None and (int(previous["source_size"]) != meta.size or int(previous["source_mtime_ns"]) != meta.mtime_ns)
    if source_changed:
        return VerifyRow(meta.path, SOURCE_CHANGED, "source file changed since the trickplay was generated", stale_building=stale, leftovers=leftovers)
    if not target.exists():
        return VerifyRow(meta.path, MISSING, "no trickplay directory for the current settings", stale_building=stale, leftovers=leftovers)

    exact = int(previous["tiles"]) if previous else None
    validation = validate_tile_directory(target, settings, meta.duration, expected_exact_tiles=exact, verify_all=verify_all)
    if not validation.valid:
        reason = validation.reason
        if validation.unexpected:
            reason += ": " + ", ".join(validation.unexpected[:5])
        return VerifyRow(meta.path, INVALID, reason, tiles=validation.tiles, stale_building=stale, leftovers=leftovers)
    suspect = _suspect_note(previous)
    if previous and previous["settings_hash"] != settings.hash:
        return VerifyRow(meta.path, SETTINGS_CHANGED, "valid output, but generated with different interval/quality settings",
                         tiles=validation.tiles, stale_building=stale, leftovers=leftovers, suspect=suspect)
    return VerifyRow(meta.path, VALID, "", tiles=validation.tiles, stale_building=stale, leftovers=leftovers, suspect=suspect)


def _suspect_note(previous: Optional[sqlite3.Row]) -> str:
    """Text when the recorded output has a number of thumbnails that does not fit the video length it was made from, else "".
    The record holds the thumbnails and the length, so outputs made before this check existed are found too."""
    if previous is None or not previous["source_duration"] or not previous["frames"]: return ""
    interval = float(previous["interval"])
    deviation = frame_count_deviation(int(previous["frames"]), float(previous["source_duration"]), interval)
    return describe_frame_deviation(deviation[0], deviation[1], interval) if deviation else ""


def verify_videos(videos: Iterable[VideoMeta], settings: TrickplaySettings, state: StateDB, *, verify_all: bool = True,
                  on_row: Optional[Callable[[int, int, VerifyRow], None]] = None,
                  should_stop: Optional[Callable[[], bool]] = None) -> tuple[dict[str, int], list[VerifyRow]]:
    items = list(videos)
    summary = summary_template(len(items))
    problems: list[VerifyRow] = []
    for number, meta in enumerate(items, 1):
        if should_stop is not None and should_stop():
            summary["canceled"] = len(items) - number + 1
            break
        row = verify_video(meta, settings, state, verify_all=verify_all)
        key = {VALID: "valid", MISSING: "missing", INVALID: "invalid", SOURCE_CHANGED: "source_changed",
               SETTINGS_CHANGED: "settings_changed"}[row.status]
        summary[key] += 1
        if row.status == INVALID and "unexpected" in row.reason:
            summary["unexpected"] += 1
        if row.stale_building:
            summary["stale_building"] += 1
        if row.leftovers:
            summary["leftover_dirs"] += len(row.leftovers)
        if row.suspect:
            summary["suspect"] += 1
        if row.status != VALID or row.stale_building or row.leftovers or row.suspect:
            problems.append(row)
        if on_row is not None:
            on_row(number, len(items), row)
    return summary, problems
