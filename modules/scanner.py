# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from . import constants
from . import platform_api
from .constants import PROBE_WORKERS, KNOWN_NONVIDEO_EXTENSIONS, KNOWN_VIDEO_EXTENSIONS
from .ffmpeg_tools import FFmpegTools
from .formatting import human_duration
from .localization import number, t
from .logger import RunLogger
from .models import VideoMeta
from .paths import could_be_video, is_relative_to_path, normalize_path, remove_outer_quotes
from .state_db import StateDB
from .storage import canonical_media_path, physical_storage_group


# Folders of the system and of NAS boxes: never media, always skipped. Windows refuses to list the first two ("Access is denied"); Synology
# keeps generated preview videos in @eaDir, and the recycle bins, snapshots and trash folders only hold copies of files that live elsewhere.
SYSTEM_FOLDERS = frozenset({
    "$recycle.bin", "system volume information",
    "@eadir", "#recycle", "#snapshot", "@recycle", ".@__thumb", "lost+found", ".trash",
})
TRASH_PREFIX = ".trash-"        # ".Trash-1000": the trash folder of one user on a Linux disk


def sanitize_media_roots(raw_roots: Iterable[str], *, announce: bool = True) -> list[Path]:
    """Normalize user-entered roots: drop duplicates, nested roots already covered by another root, and the program folder."""
    roots: list[Path] = []
    canonical: list[str] = []
    tool_root = normalize_path(constants.APP_ROOT)
    for raw in raw_roots:
        if not remove_outer_quotes(raw): continue
        try: raw = platform_api.whole_drive(platform_api.resolve_media_path(str(raw)))
        except ValueError:
            if announce: print(t("roots.shortcut_unresolved", path=remove_outer_quotes(raw)))
            continue
        if not raw: continue
        if not os.path.isabs(os.path.expanduser(raw)):
            # A relative name would mean a different folder depending on where the program was started.
            if announce: print(t("roots.not_absolute", path=raw))
            continue
        p = Path(raw).expanduser()
        try: p = p.resolve(strict=False)
        except Exception: p = p.absolute()
        n = normalize_path(p)
        if n == tool_root or is_relative_to_path(n, tool_root):
            if announce: print(t("scan.skip_program_folder", path=p))
            continue
        c = os.path.normcase(os.path.normpath(canonical_media_path(p)))
        if any(c == old for old in canonical):
            if announce: print(t("scan.skip_duplicate", path=p))
            continue
        if any(platform_api.path_is_within(c, old) for old in canonical):
            if announce: print(t("scan.skip_nested", path=p))
            continue
        keep = [(p0, c0) for p0, c0 in zip(roots, canonical) if not platform_api.path_is_within(c0, c)]
        roots = [x[0] for x in keep]
        canonical = [x[1] for x in keep]
        roots.append(p)
        canonical.append(c)
    return roots


class MediaScanner:
    def __init__(self, tools: FFmpegTools, state: StateDB, config: dict[str, Any], logger: Optional[RunLogger] = None) -> None:
        self.tools = tools
        self.state = state
        self.config = config
        self.logger = logger
        self.unknown_video_extensions: set[str] = set()
        self.unreadable_videos: list[tuple[Path, str]] = []   # files with a video extension that ffprobe could not read, and why
        # What the last walk saw, to find `.trickplay` folders whose video is gone without listing the folders a second time.
        self.trickplay_dirs: list[Path] = []
        self._dir_stems: dict[str, set[str]] = {}
        self._unreadable_dirs: set[str] = set()

    def sanitize_roots(self, raw_roots: Iterable[str]) -> list[Path]: return sanitize_media_roots(raw_roots, announce=True)

    def _skip_dir(self, path: Path, excluded: set[str]) -> bool:
        if path.name.casefold().endswith(".trickplay"): return True
        if platform_api.is_link_or_reparse_point(path): return True
        name = path.name.casefold()
        return name in excluded or name in SYSTEM_FOLDERS or name.startswith(TRASH_PREFIX)

    def _walk_files(self, root: Path) -> Iterable[Path]:
        excluded = {x.casefold() for x in self.config.get("excluded_folders", [])}
        tool_root = normalize_path(constants.APP_ROOT)
        stack = [root]
        while stack:
            current = stack.pop()
            if normalize_path(current) == tool_root: continue
            stems = self._dir_stems.setdefault(str(current), set())
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        try:
                            p = Path(entry.path)
                            if entry.is_dir(follow_symlinks=False):
                                if p.name.casefold().endswith(".trickplay") and not platform_api.is_link_or_reparse_point(p): self.trickplay_dirs.append(p)
                                if not self._skip_dir(p, excluded): stack.append(p)
                            elif entry.is_file(follow_symlinks=False):
                                if could_be_video(entry.name): stems.add(Path(entry.name).stem.casefold())
                                yield p
                        except OSError:
                            self._unreadable_dirs.add(str(current))
            except OSError as exc:
                self._unreadable_dirs.add(str(current))
                if self.logger: self.logger.log("WARN", f"Could not scan {current}: {exc}")

    def orphan_trickplay_dirs(self) -> list[Path]:
        """`.trickplay` folders found by the last scan whose video is no longer next to them. A folder whose listing was incomplete is never named."""
        found: list[Path] = []
        for folder in self.trickplay_dirs:
            parent = str(folder.parent)
            if parent in self._unreadable_dirs or parent not in self._dir_stems: continue
            if folder.name[:-len(".trickplay")].casefold() not in self._dir_stems[parent]: found.append(folder)
        return sorted(found, key=lambda path: str(path).casefold())

    def scan(self, roots: list[Path], *, show_progress: bool = True) -> list[VideoMeta]:
        videos: list[VideoMeta] = []
        seen: set[str] = set()
        files_seen = 0
        probes = 0
        start = time.monotonic()
        pending: dict[str, list[tuple[Path, str]]] = {}
        self.trickplay_dirs.clear()
        self._dir_stems.clear()
        self._unreadable_dirs.clear()
        self.unreadable_videos.clear()
        failures = getattr(self.tools, "probe_failures", None)
        for root in roots:
            if not root.exists():
                print(t("scan.offline", path=root))
                continue
            for path in self._walk_files(root):
                files_seen += 1
                ext = path.suffix.casefold()
                if ext in KNOWN_NONVIDEO_EXTENSIONS: continue
                n = normalize_path(path)
                if n in seen: continue
                seen.add(n)
                try: st = path.stat()
                except OSError: continue
                meta = self.state.cached_video(path, st.st_size, st.st_mtime_ns)
                if meta is not None:
                    videos.append(meta)
                else:
                    probes += 1
                    pending.setdefault(physical_storage_group(path), []).append((path, ext))
                if show_progress and (len(videos) + probes) % 100 == 0:
                    elapsed=max(.001,time.monotonic()-start)
                    print("\r" + t("scan.progress", videos=number(len(videos)), files=number(files_seen), elapsed=human_duration(elapsed)),end="",flush=True)
        executors=[]
        futures={}
        try:
            for group, items in pending.items():
                workers=PROBE_WORKERS
                ex=concurrent.futures.ThreadPoolExecutor(max_workers=workers,thread_name_prefix="ffprobe")
                executors.append(ex)
                for path,ext in items: futures[ex.submit(self.tools.probe,path)]=(path,ext)
            completed=0
            for future in concurrent.futures.as_completed(futures):
                path,ext=futures[future]
                completed+=1
                try: meta=future.result()
                except Exception as exc:
                    meta=None
                    if self.logger: self.logger.log("WARN",f"Probe failed for {path}: {exc}")
                    if isinstance(failures, dict): failures[str(path)] = f"{type(exc).__name__}: {exc}"
                if meta is not None:
                    self.state.upsert_video(meta)
                    videos.append(meta)
                    if ext not in KNOWN_VIDEO_EXTENSIONS: self.unknown_video_extensions.add(ext or "<no extension>")
                elif isinstance(failures, dict):
                    reason = failures.pop(str(path), None)
                    if reason and ext in KNOWN_VIDEO_EXTENSIONS:        # an unknown extension that is no video is normal; a video that cannot be read is not
                        self.unreadable_videos.append((path, reason))
                        if self.logger: self.logger.log("WARN", f"Unreadable video: {path} ({reason})")
                if show_progress and (completed % 100 == 0 or completed == len(futures)):
                    elapsed=max(.001,time.monotonic()-start)
                    print("\r" + t("scan.progress", videos=number(len(videos)), files=number(files_seen), elapsed=human_duration(elapsed)),end="",flush=True)
        finally:
            for ex in executors: ex.shutdown(wait=True,cancel_futures=True)        # an interrupted scan drops the probes that have not started; the running ones finish
        videos.sort(key=lambda m: str(m.path).casefold())
        self.unreadable_videos.sort(key=lambda item: str(item[0]).casefold())
        if show_progress:
            elapsed = time.monotonic() - start
            print("\r" + t("scan.complete", videos=number(len(videos)), files=number(files_seen), elapsed=human_duration(elapsed)) + " " * 24)
        return videos
