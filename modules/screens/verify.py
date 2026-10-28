# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from ..datetime_helpers import local_stamp
from ..file_io import atomic_write_text
from ..formatting import display_path, human_bytes
from ..localization import english, number, t, tp
from ..logger import RunLogger
from ..models import VideoMeta
from ..paths import normalize_path
from ..prompts import heading, pause, prompt_yes_no
from ..tiles import config_settings
from ..verify import (
    INVALID, MISSING, SETTINGS_CHANGED, SOURCE_CHANGED, VALID, Leftover, Orphan, VerifyRow, assess_orphan, remove_leftover, remove_orphan,
    restore_previous, verify_video, verify_videos,
)
from .generate import GenerateScreens


class VerifyScreens(GenerateScreens):
    """Verify: check existing output, then offer cleanup of leftovers and regeneration."""

    orphans: list[Orphan] = []
    other_widths: dict[str, int] = {}
    _asked = False                              # a question was put to the person during this Verify
    _acted = False                              # and one was answered with yes (something was removed, restored or generated)

    def verify(self, *, single: Optional[Path] = None) -> None:
        self._screen(t("menu.verify"))
        if not single and not self.config.get("media_roots"):
            print(t("msg.no_roots"))
            pause()
            return
        if not self.ensure_ffmpeg(): return
        if not self._require_pillow():
            pause()
            return
        logger = RunLogger("verify")
        self._asked = self._acted = False
        self.state.begin_run(logger.run_id, "verify")
        try:
            if single:
                self.orphan_candidates = []
                self.unreadable_videos = []
                meta = self.tools.probe(single) if self.tools else None
                videos = [meta] if meta else []
            else:
                videos = self.scan_library(logger)
            if not videos:
                print(t("msg.no_videos"))
                pause()
                return
            checked = self._check_videos(videos, logger)
            if checked is None:
                pause()
                return
            problems = checked[1]
            self._offer_orphan_cleanup(self.orphans, logger)
            self._offer_leftover_cleanup([item for row in problems for item in row.leftover_items], logger)
            problems = self._offer_restore(problems, videos, logger)
            self._offer_regeneration(problems, videos)
            if self._acted or not self._asked: pause()          # only declined questions: nothing new to read, so no extra Enter
        finally:
            logger.close()

    def _check_videos(self, videos: list[VideoMeta], logger: RunLogger) -> Optional[tuple[dict[str, Any], list[VerifyRow]]]:
        """Verify `videos`, print the summary, write the problem list. None if interrupted with Ctrl+C."""
        settings = config_settings(self.config)
        print("\n" + tp("verify.checking", len(videos), folder=settings.folder_name, interval=f"{settings.interval:g}"))
        report = logger.folder / "problems.txt"            # the problems of this verification (the log of this start has the same entries)
        print(t("verify.problems_saved", path=display_path(report)))
        print(t("verify.ctrl_c") + "\n")
        verify_all = bool(self.config.get("verification", {}).get("verify_all_tiles", True))

        self.other_widths = {}

        def on_row(position: int, total: int, row: VerifyRow) -> None:
            """Nothing is listed on the screen: a problem goes to the log and to the problem list file; the screen shows the progress."""
            if self._is_problem(row):
                logger.event("VERIFY_" + row.status, path=str(row.path), reason=row.reason, stale_building=row.stale_building, leftovers=list(row.leftovers))
            if row.suspect: logger.event("VERIFY_SUSPECT", path=str(row.path), note=row.suspect)
            for name in row.other_widths: self.other_widths[name] = self.other_widths.get(name, 0) + 1
            if sys.stdout.isatty() and (position % 25 == 0 or position == total):
                print("\r" + t("verify.checked", done=number(position), total=number(total)), end="", flush=True)

        try:
            summary, problems = verify_videos(videos, settings, self.state, verify_all=verify_all, on_row=on_row)
        except KeyboardInterrupt:
            print("\n" + t("menu.canceled"))
            return None
        if sys.stdout.isatty(): print()
        self.orphans = [assess_orphan(path) for path in self.orphan_candidates]
        summary["orphan_dirs"] = len(self.orphans)
        summary["unreadable_videos"] = len(self.unreadable_videos)
        for path, reason in self.unreadable_videos: logger.event("VERIFY_UNREADABLE", path=str(path), reason=reason)
        for orphan in self.orphans: logger.event("VERIFY_ORPHAN", path=str(orphan.path), safe=orphan.safe, reason=orphan.reason, bytes=orphan.size)
        logger.write_summary(summary)
        self.state.end_run(logger.run_id, summary)
        self._print_verify_summary(summary)
        self._print_other_widths(self.other_widths, str(self.config["trickplay"].get("other_width_policy", "keep_add")))
        if self.other_widths: logger.event("VERIFY_OTHER_WIDTHS", folders=dict(self.other_widths))
        with english(): summary_lines = self._summary_lines(summary)                              # the problem list is English
        lines = [f"{row.status}\t{row.reason}\t{row.path}" + ("\tleftovers=" + ",".join(row.leftovers) if row.leftovers else "")
                 for row in problems if self._is_problem(row)]
        lines += [f"SUSPECT\t{row.suspect}\t{row.path}" for row in problems if row.suspect]
        lines += [f"ORPHAN\t{orphan.reason}\t{orphan.path}" for orphan in self.orphans]
        lines += [f"UNREADABLE\t{reason}\t{path}" for path, reason in self.unreadable_videos]
        header = [f"VERIFY SUMMARY {local_stamp()}", *summary_lines]
        for name, count in sorted(self.other_widths.items(), key=lambda item: (-item[1], item[0])): header.append(f"  Other size {name}: on {count:,} video{'' if count == 1 else 's'}")
        body = [f"PROBLEMS ({len(lines):,})" if lines else "PROBLEMS: none", *lines]
        atomic_write_text(report, "\n".join([*header, "", *body]) + "\n")
        if lines: print("\n" + t("verify.problem_list", path=display_path(report)))

        return summary, problems

    def _ask(self, question: str, default: bool) -> bool:
        """A yes/no question of Verify. What was asked and agreed to decides whether the screen waits for Enter at the end."""
        self._asked = True
        answer = prompt_yes_no(question, default)
        if answer: self._acted = True
        return answer

    @staticmethod
    def _is_problem(row: VerifyRow) -> bool:
        """A problem is output that exists and is not valid, or a folder left behind. Missing output (or output that is outdated) is just work for Generate."""
        return row.status == INVALID or row.stale_building or bool(row.leftovers)

    @staticmethod
    def _summary_lines(summary: dict[str, Any]) -> list[str]:
        """One line of numbers (how many were checked and how they came out); what else was found only when it is there."""
        outcomes = [("valid", t("verify.o_valid")), ("missing", t("verify.o_missing")), ("invalid", t("verify.o_invalid")), ("source_changed", t("verify.o_source_changed")),
                    ("settings_changed", t("verify.o_settings_changed")), ("unexpected", t("verify.o_unexpected"))]
        found = [f"{number(summary[key])} {label}" for key, label in outcomes if summary[key]]
        total = summary["total"]
        everything_fine = summary["valid"] == total
        outcome = t("verify.all_valid") if everything_fine and total else ", ".join(found) or t("verify.none_valid")
        lines = ["  " + tp("verify.checked_summary", total, outcome=outcome)]
        extras = [(key, label) for key, label in (("stale_building", t("verify.e_stale")), ("leftover_dirs", t("verify.e_leftover")), ("orphan_dirs", t("verify.e_orphan")),
                                                  ("unreadable_videos", t("verify.e_unreadable")), ("suspect", t("verify.e_suspect"))) if summary[key]]
        width = max(32, max((len(label) for _, label in extras), default=0) + 1)
        for key, label in extras: lines.append(f"  {label + ':':<{width}} {number(summary[key])}")
        return lines

    def _print_verify_summary(self, summary: dict[str, Any]) -> None:
        heading(t("verify.summary_heading"))
        for line in self._summary_lines(summary): print(line)

    @staticmethod
    def _print_other_widths(found: dict[str, int], policy: str) -> None:
        """Trickplay of other widths next to the target (not a problem): which ones, on how many videos, and what Generate does with them."""
        if not found: return
        heading(t("verify.other_sizes_heading"))
        for name, count in sorted(found.items(), key=lambda item: (-item[1], item[0])): print(f"  {name:16} " + tp("verify.on_videos", count))
        print("  " + (t("verify.other_replace") if policy == "replace" else t("verify.other_keep")))

    def _offer_orphan_cleanup(self, orphans: list[Orphan], logger: RunLogger) -> None:
        """Offer to remove `.trickplay` folders whose video no longer exists (deleted, renamed or moved)."""
        if not orphans: return
        safe = [item for item in orphans if item.safe]
        kept = [item for item in orphans if not item.safe]

        def show(title: str, group: list[Orphan]) -> None:
            print(f"  {title} {number(len(group))} ({human_bytes(sum(item.size for item in group))})")
            for item in group[:10]: print(f"    {item.path}\n      - {item.reason}")
            if len(group) > 10: print("    " + t("verify.more", count=number(len(group) - 10)))

        heading(t("verify.orphans_heading"))
        print(t("verify.orphans_explain"))
        if safe: show(t("verify.safe_to_remove"), safe)
        if kept: show(t("verify.kept_not_safe"), kept)
        if not safe: return
        if not self._ask("\n" + tp("verify.orphans_question", len(safe)), False): return
        removed = 0
        for item in safe:
            problem = remove_orphan(item)
            if problem is None:
                removed += 1
                logger.event("ORPHAN_REMOVED", path=str(item.path), bytes=item.size)
            else:
                print("  " + t("verify.kept", path=item.path, reason=problem))
                logger.event("ORPHAN_KEPT", path=str(item.path), reason=problem)
        print(tp("verify.removed_of", len(safe), removed=number(removed)))

    def _offer_restore(self, rows: list[VerifyRow], videos: list[VideoMeta], logger: RunLogger) -> list[VerifyRow]:
        """Offer to put back the previous output where a replacement did not finish and the new output is missing or damaged.
        Returns the problem rows that still need attention (restored videos are checked again and dropped when they are fine)."""
        candidates = [row for row in rows if row.restorable is not None]
        if not candidates: return rows
        heading(t("verify.restore_heading"))
        print(tp("verify.restore_explain", len(candidates)))
        for row in candidates[:10]:
            restorable = row.restorable
            if restorable is not None: print(f"  {row.path}\n    - " + t("verify.restore_line", status=row.status, name=restorable.path.name))
        if len(candidates) > 10: print("  " + t("verify.more", count=number(len(candidates) - 10)))
        if not self._ask(tp("verify.restore_question", len(candidates)), False): return rows
        by_path = {normalize_path(video.path): video for video in videos}
        settings = config_settings(self.config)
        verify_all = bool(self.config.get("verification", {}).get("verify_all_tiles", True))
        restored: dict[str, VerifyRow] = {}
        for row in candidates:
            restorable = row.restorable
            if restorable is None: continue
            problem = restore_previous(restorable)
            if problem is not None:
                print("  " + t("verify.not_restored", path=row.path, reason=problem))
                logger.event("RESTORE_FAILED", path=str(row.path), previous=str(restorable.path), reason=problem)
                continue
            logger.event("RESTORED", path=str(row.path), previous=str(restorable.path), target=str(restorable.target))
            meta = by_path.get(normalize_path(row.path))
            restored[normalize_path(row.path)] = verify_video(meta, settings, self.state, verify_all=verify_all) if meta else row
        print(tp("verify.restored_of", len(candidates), restored=number(len(restored))))
        result: list[VerifyRow] = []
        for row in rows:
            fresh = restored.get(normalize_path(row.path), row)
            if fresh is not row and fresh.status == VALID and not fresh.stale_building and not fresh.leftovers: continue
            result.append(fresh)
        return result

    def _offer_regeneration(self, rows: list[VerifyRow], videos: list[VideoMeta]) -> None:
        """After Verify: offer to (re)generate the videos whose output is missing, damaged or outdated."""
        by_path = {normalize_path(video.path): video for video in videos}

        def pick(statuses: set[str]) -> list[VideoMeta]:
            return [by_path[key] for row in rows if row.status in statuses and (key := normalize_path(row.path)) in by_path]

        core = pick({MISSING, INVALID, SOURCE_CHANGED})
        outdated = pick({SETTINGS_CHANGED})
        if not core and not outdated: return

        print()
        selected: list[VideoMeta] = []
        if core and self._ask(tp("verify.regen_question", len(core)), False): selected += core
        if outdated and self._ask(tp("verify.regen_outdated_question", len(outdated)), False): selected += outdated
        if not selected: return
        self._screen(t("screen.generate"))                      # on to a clean Generate screen, with the videos Verify found (no second scan)
        tools = self._require_tools()
        for line in self.ffmpeg_lines(tools, tools.backends()): print(line)
        logger = RunLogger("generate")
        self.state.begin_run(logger.run_id, logger.mode)
        try:
            self._acted = self._plan_and_run(selected, logger)         # the screen was cleared: only a finished run leaves a summary to read
        finally:
            logger.close()

    def _offer_leftover_cleanup(self, items: list[Leftover], logger: RunLogger) -> None:
        """Offer to remove leftover backup / failed-finalize folders that cannot hold the only good copy of an output."""
        if not items: return
        safe = [item for item in items if item.safe]
        kept = [item for item in items if not item.safe]

        def show(title: str, group: list[Leftover]) -> None:
            print(f"  {title} {number(len(group))} ({human_bytes(sum(item.size for item in group))})")
            for item in group[:10]: print(f"    {item.path}\n      - {item.reason}")
            if len(group) > 10: print("    " + t("verify.more", count=number(len(group) - 10)))

        heading(t("verify.leftover_heading"))
        if safe: show(t("verify.safe_to_remove"), safe)
        if kept: show(t("verify.kept_not_safe"), kept)
        if not safe: return
        if not self._ask("\n" + tp("verify.leftover_question", len(safe)), False): return
        removed = 0
        for item in safe:
            problem = remove_leftover(item)
            if problem is None:
                removed += 1
                logger.event("LEFTOVER_REMOVED", path=str(item.path), kind=item.kind, reason=item.reason, bytes=item.size)
            else:
                print("  " + t("verify.kept", path=item.path, reason=problem))
                logger.event("LEFTOVER_KEPT", path=str(item.path), kind=item.kind, reason=problem)
        print(tp("verify.removed_of", len(safe), removed=number(removed)))
