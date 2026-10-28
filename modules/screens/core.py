# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import os
import re
import shutil
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .. import localization, platform_api, system_info
from ..activity_log import ACTIVITY
from ..config import ConfigManager
from ..constants import APP_NAME, DIRS, FFMPEG_OFFICIAL_URL
from ..ffmpeg_registry import FFmpegCandidate, FFmpegRegistry
from ..ffmpeg_tools import FFmpegTools
from ..file_io import ensure_directories
from ..formatting import display_path, human_bytes
from ..localization import english, number, t
from ..logger import RunLogger
from ..models import PlanItem, VideoMeta
from ..generator import library_disks
from ..prompts import choose, fit, format_labelled, pause, read_main_menu_choice
from ..scanner import MediaScanner
from ..state_db import StateDB
from ..storage import canonical_media_path, mount_group, physical_storage_group
from ..tiles import PIL_AVAILABLE, folder_size


LEAVE_MENU: Any = object()                  # an action returns this to leave the menu it was chosen in (after the first choice of a language, say)


def text(label: Union[str, Callable[[], str]]) -> str:
    """A title or label is a text, or a function that makes it when it is shown (in the language of that moment)."""
    return label() if callable(label) else label


def in_english(label: Union[str, Callable[[], str]]) -> str:
    """The same in English, for the activity log."""
    with english(): return text(label)


def short_version(version_line: str) -> str:
    """The version as the screens show it: "FFmpeg 9.0.2-essentials_build" from "ffmpeg version 9.0.2-essentials_build-www.gyan.dev Copyright ...";
    the builder's web address at the end is left out (the log keeps the whole line)."""
    match = re.match(r"ffmpeg version (\S+)", version_line.strip(), re.IGNORECASE)
    if not match: return version_line.strip() or "FFmpeg"
    return "FFmpeg " + (re.sub(r"-(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}$", "", match.group(1), flags=re.IGNORECASE) or match.group(1))


def ffmpeg_summary(version_line: str, gpu_names: list[str], backends: list[str]) -> str:
    """The one line that says which FFmpeg is in use: "FFmpeg 9.0.2 | hardware paths: cuda, d3d11va" (the graphics cards are told below it)."""
    parts = [short_version(version_line)]
    parts.append(t("ffmpeg.summary_paths", backends=", ".join(backends)) if backends else t("ffmpeg.summary_no_path") if gpu_names else t("ffmpeg.summary_cpu_only"))
    return " | ".join(parts)


class AppCore:
    """Shared state, startup/shutdown, FFmpeg discovery, menu helpers, library scan and storage groups."""

    def __init__(self) -> None:
        ensure_directories()
        self.config_mgr = ConfigManager()
        self.config = self.config_mgr.load()
        localization.activate_configured(str(self.config.get("language", "")))
        self.state = StateDB()
        self.registry = FFmpegRegistry()
        self.ffmpeg_candidate: Optional[FFmpegCandidate] = None
        self._logged_ffmpeg: Optional[str] = None            # what the activity log already says about the FFmpeg in use
        self.tools: Optional[FFmpegTools] = None
        self._storage_cache: dict[str, str] = {}
        self.last_summary: Optional[dict[str, Any]] = None  # summary of the last generation run
        self.orphan_candidates: list[Path] = []              # `.trickplay` folders without a video, found by the last library scan
        self.unreadable_videos: list[tuple[Path, str]] = []  # video files ffprobe could not read in the last library scan, and why
        self._trail: list[Union[str, Callable[[], str]]] = []  # the menus that are open, outermost first

    def ffmpeg_lines(self, tools: FFmpegTools, backends: list[str]) -> list[str]:
        """What is in use, for the top of a screen: the FFmpeg and its hardware paths, then the computer (processor, graphics cards,
        memory, the disks of the library), and a note when an Intel graphics chip is there but FFmpeg cannot use it (QSV)."""
        host = system_info.host()
        rows = [(t("system.cpu"), system_info.cpu_text(host)), (t("system.gpu"), "\n".join(tools.gpu_names)),
                (t("system.ram"), system_info.ram_text(host)), (t("system.disks"), ", ".join(system_info.all_disks())),
                (t("system.selected_disks"), ", ".join(library_disks(self.config.get("media_roots", []))))]
        lines = [ffmpeg_summary(tools.candidate.version_line, list(tools.gpu_names), backends), ""]
        lines += format_labelled([(label, value) for label, value in rows if value], indent=2, minimum=8)
        if tools.hw_profile.get("intel_gpu") and "qsv" not in backends: lines += ["", fit(t("system.qsv_missing"), 0)]
        return lines

    def startup_recovery(self, *, interactive: bool = True) -> bool:
        if self.config_mgr.corrupt_error:
            self._screen(t("startup.config_warning_title"))
            print(t("startup.config_warning"))
            print(t("startup.config_invalid", reason=self.config_mgr.corrupt_error))
            if not interactive:
                print(t("startup.nothing_done_config"))
                return False
            if self.config_mgr.loaded_from_backup:
                print(t("startup.backup_loaded"))
                choice = choose([("1", t("startup.restore_backup")), ("2", t("startup.start_fresh")), ("3", t("startup.exit"))], "1")
                if choice == "1":
                    self.config_mgr.restore_backup()
                    self.config = self.config_mgr.config
                    ACTIVITY.log("CONFIG", "invalid configuration: the backup was restored")
                elif choice == "2":
                    self.config_mgr.start_fresh()
                    self.config = self.config_mgr.config
                    ACTIVITY.log("CONFIG", "invalid configuration: started fresh (the old file was archived)")
                else: return False
            else:
                if choose([("1", t("startup.start_fresh")), ("2", t("startup.exit"))], "1") == "1":
                    self.config_mgr.start_fresh()
                    self.config = self.config_mgr.config
                    ACTIVITY.log("CONFIG", "invalid configuration: started fresh (the old file was archived)")
                else: return False

        if self.registry.recovery_note:
            print("\n" + t("startup.registry_warning", note=self.registry.recovery_note))

        try: self.state.open()
        except Exception as exc:
            self._screen(t("startup.history_warning_title"))
            print(t("startup.history_warning", reason=exc))
            if not interactive:
                print(t("startup.nothing_done_history"))
                return False
            if choose([("1", t("startup.rebuild_history")), ("2", t("startup.exit"))], "1") == "1":
                backup = self.state.recover_corrupt()
                print(t("startup.database_archived", name=backup.name))
            else: return False

        self.clean_abandoned_temp(silent=False)
        return True

    def shutdown(self) -> None:
        try: self.state.close()
        except Exception: pass

    def clean_abandoned_temp(self, silent: bool = False) -> None:
        temp = DIRS["temp"]
        removed = 0
        bytes_removed = 0
        for child in list(temp.iterdir()):
            if child.name == ".gitkeep": continue
            try:
                size = folder_size(child) if child.is_dir() else child.stat().st_size
                if child.is_dir(): shutil.rmtree(child)
                else: child.unlink()
                removed += 1
                bytes_removed += size
            except Exception: continue
        if removed: ACTIVITY.log("TEMP", f"removed {removed} abandoned temporary item(s), {human_bytes(bytes_removed)}")
        if not silent and removed: print(localization.tp("startup.temp_removed", removed, size=human_bytes(bytes_removed)))

    @staticmethod
    def _require_pillow() -> bool:
        """Pillow packs the sprites; say how to get it when it is missing."""
        if PIL_AVAILABLE: return True
        print(t("startup.pillow_required"))
        return False

    def _require_tools(self) -> FFmpegTools:
        """FFmpeg tools; every caller runs ensure_ffmpeg() first, so a missing one is a program error, not a user error."""
        if self.tools is None: raise RuntimeError("FFmpeg not initialized")
        return self.tools

    def ensure_ffmpeg(self, *, interactive: bool = True) -> bool:
        while True:
            candidates = self.registry.discover(self.config)
            validated: list[FFmpegCandidate] = []
            if candidates:
                for candidate in candidates:
                    if self.registry.hydrate_cached(candidate):
                        result = candidate
                    else:
                        print(t("ffmpeg.checking"), flush=True)       # the first check of a build takes a moment
                        result = self.registry.validate(candidate)
                        if not result.valid: print("  " + t("ffmpeg.not_usable", reason=result.validation_error))
                    validated.append(result)

                chosen = self.registry.choose(validated, interactive=interactive)
                if chosen:
                    self.ffmpeg_candidate = chosen
                    self.tools = FFmpegTools(chosen)
                    self.config["ffmpeg"]["selected_hash"] = chosen.sha256
                    self.config_mgr.save()
                    backends = self._require_tools().backends()
                    for line in self.ffmpeg_lines(self.tools, backends): print(line)
                    print()
                    host = system_info.host()
                    selected_disks = ', '.join(library_disks(self.config.get("media_roots", []))) or 'none'
                    described = (f"version short: {short_version(chosen.version_line)} | version long: {chosen.version_line} | {chosen.root} | GPU(s): {', '.join(self.tools.gpu_names) or 'none'} | "
                                 f"hardware backends: {', '.join(backends) or 'none'} | CPU: {system_info.cpu_text(host) or 'unknown'} | RAM: {system_info.ram_text(host) or 'unknown'} | "
                                 f"disks: {', '.join(system_info.all_disks()) or 'unknown'} | selected disks: {selected_disks}")
                    if described != self._logged_ffmpeg:             # once per start, again only if it changed
                        self._logged_ffmpeg = described
                        ACTIVITY.log("FFMPEG", described)
                    return True

            if not interactive: return False
            self._screen(t("ffmpeg.screen"))
            ffmpeg_name, ffprobe_name = platform_api.ffmpeg_executable_names()
            print(t("ffmpeg.not_found") + "\n")
            print(t("ffmpeg.expected_location", first=display_path(self.registry.ffmpeg_folder() / 'bin' / ffmpeg_name),
                    second=display_path(self.registry.ffmpeg_folder() / 'bin' / ffprobe_name)) + "\n")
            help_lines, builds_url = platform_api.ffmpeg_setup_help()
            for line in help_lines: print(line)
            print()
            entries = [("1", t("ffmpeg.open_official"))]
            if builds_url: entries.append(("2", t("ffmpeg.open_builds")))
            choice = choose(entries + [("3", t("ffmpeg.look_again")), ("4", t("common.back"))], "3")
            if choice == "1":
                webbrowser.open(FFMPEG_OFFICIAL_URL)
                continue
            if choice == "2" and builds_url:
                webbrowser.open(builds_url)
                continue
            if choice == "3": continue
            return False

    HEADER_WIDTH = 72

    @classmethod
    def show_header(cls, title: str) -> None:
        """Program name, and under a thin line where the person is (Configuration > Trickplay settings > Key frames).
        The main menu is the root of that path, so it shows the tagline there instead."""
        width = cls.HEADER_WIDTH
        print("=" * width)
        print(f" {APP_NAME.upper()}")
        if title == t("screen.main_menu"):
            print("-" * width)
            print(f" {t('app.tagline')}")
        else:
            room = width - 2
            where = title if len(title) <= room else "..." + title[len(title) - room + 3:]            # the end of the path is the part that matters
            print("-" * width)
            print(f" {where}")
        print("=" * width)
        print()

    def _where(self, title: "Union[str, Callable[[], str]]") -> str:
        """The path to a screen: the menus that were opened to get here, then its own title (the main menu itself is just "Main menu")."""
        trail = [text(item) for item in self._trail]
        title = text(title)
        if trail[:1] == [t("screen.main_menu")]: trail = trail[1:]
        if not trail or trail[-1] != title: trail.append(title)
        return " > ".join(trail)

    def _screen(self, title: "Union[str, Callable[[], str]]") -> None:
        """A fresh screen with the program header: every screen starts like this, menus and single screens alike."""
        platform_api.clear_screen()
        self.show_header(self._where(title))

    def _menu(self, title: "Union[str, Callable[[], str]]", items: "Union[list[tuple[Union[str, Callable[[], str]], Callable[[], Any], bool]], Callable[[], list[tuple[Union[str, Callable[[], str]], Callable[[], Any], bool]]]]",
              back_label: "Union[str, Callable[[], str]]" = localization.lazy("common.back"), *, enter_goes_back: bool = True, intro: Optional[Callable[[], None]] = None) -> None:
        """Numbered menu in the style of MediaCatalog / MediaDiversitySampler.

        [0] leaves. In submenus Enter alone also goes back (`Choice [0]:`); the main menu has no default, so an
        empty answer there only redraws the screen (`enter_goes_back=False`). Each item is
        (label, action, wait): `wait` asks for Enter afterwards so the result stays visible, for actions that
        do not already wait for it themselves; an action that returns False has nothing to show (the person went back), so there
        is no Enter either. A label can be a function: it is called every time the menu is drawn, so it can show the current
        value of a setting. `items` can be a function too (the list is built every time the menu is drawn, so an entry can come
        and go), and `intro` writes a few lines under the header. The title and the label of [0] can be functions as well (a menu that
        changes the language of the program shows itself in the new one at once).
        Ctrl+C in a submenu goes one level back (in the main menu it ends the program); Ctrl+C during an action ends only that action.
        """
        main = not enter_goes_back
        self._trail.append(title)
        try:
            while True:
                shown = items() if callable(items) else items
                self._screen(title)
                if intro:
                    intro()
                    print()
                for number, (label, _action, _pause) in enumerate(shown, 1): print(f"[{number}] {text(label)}")
                print(f"\n[0] {text(back_label)}\n")
                try:
                    if enter_goes_back: choice = read_main_menu_choice(range(len(shown) + 1), localization.t("prompt.choice") + " [0]: ", default=0)
                    else: choice = read_main_menu_choice(range(len(shown) + 1))
                except KeyboardInterrupt:
                    if main: raise
                    ACTIVITY.log("MENU", f"{in_english(title)}: left with Ctrl+C")
                    return
                if choice == -1: continue
                if choice == 0: return
                label, action, wait = shown[choice - 1]
                title_en, label_en = in_english(title), in_english(label)                       # the activity log is always English
                ACTIVITY.log("MENU", f"{title_en} > {label_en}")
                started = time.monotonic()
                try:
                    done = action()
                    if wait and done is not False: pause()                      # an action that returns False has nothing to show (the person went back)
                except KeyboardInterrupt:
                    ACTIVITY.log("MENU", f"{title_en} > {label_en}: canceled with Ctrl+C after {time.monotonic() - started:.1f}s")
                    print("\n" + t("menu.canceled"))
                    try: pause()
                    except KeyboardInterrupt: pass
                    continue
                except BaseException as exc:
                    ACTIVITY.log("MENU", f"{title_en} > {label_en}: ended by {type(exc).__name__}" + (f": {exc}" if str(exc) else ""))
                    raise
                ACTIVITY.log("MENU", f"{title_en} > {label_en}: done in {time.monotonic() - started:.1f}s")
                if done is LEAVE_MENU: return                                   # the choice was the whole point of this menu
        finally:
            self._trail.pop()

    def _scanner(self, logger: Optional[RunLogger] = None) -> MediaScanner:
        if not self.tools: raise RuntimeError("FFmpeg not initialized")
        return MediaScanner(self._require_tools(), self.state, self.config, logger)

    def scan_library(self, logger: Optional[RunLogger] = None) -> list[VideoMeta]:
        self.orphan_candidates = []
        self.unreadable_videos = []
        scanner = self._scanner(logger)
        roots = scanner.sanitize_roots(self.config.get("media_roots", []))
        if not roots:
            print(t("msg.no_roots"))
            return []
        videos = scanner.scan(roots)
        self.orphan_candidates = scanner.orphan_trickplay_dirs()
        self.unreadable_videos = list(scanner.unreadable_videos)
        if self.unreadable_videos:
            print("\n" + t("scan.unreadable", count=number(len(self.unreadable_videos))))
            for path, reason in self.unreadable_videos[:10]: print(f"  {path}\n    - {reason}")
            if len(self.unreadable_videos) > 10: print("  " + t("scan.unreadable_more", count=number(len(self.unreadable_videos) - 10)))
            ACTIVITY.log("SCAN", f"{len(self.unreadable_videos)} unreadable video(s)")
        return videos

    def _storage_group_for(self, path: Path) -> str:
        mode = self.config["workers"].get("storage_mode", "auto")
        if mode == "manual":
            groups = self.config["workers"].get("manual_groups", {})
            n = os.path.normcase(os.path.normpath(canonical_media_path(path)))
            for name, roots in groups.items():
                for root in roots:
                    rn = os.path.normcase(os.path.normpath(canonical_media_path(Path(root))))
                    if n == rn or platform_api.path_is_within(n, rn): return f"MANUAL:{name}"
        if mode == "drive":
            return mount_group(path)
        return physical_storage_group(path)

    def _apply_storage_groups(self, plan: list[PlanItem]) -> None:
        for item in plan: item.storage_group = self._storage_group_for(item.meta.path)
