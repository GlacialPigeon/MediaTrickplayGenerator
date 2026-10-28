# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..activity_log import ACTIVITY
from .. import constants
from ..constants import DIRS
from ..formatting import human_bytes
from ..localization import tp, t
from ..maintenance import all_items, item_size, old_items, old_logs, remove_items, usage_report
from ..prompts import choose, prompt_yes_no
from .core import AppCore


class MaintenanceScreens(AppCore):
    """Maintenance actions: disk usage, cleaning, resetting history (the menu itself is in ui.py)."""

    def _delete_items(self, items: list[Path], label: str) -> Optional[bool]:
        """False when the person declined (nothing to read afterwards, so no Enter)."""
        items = [p for p in items if p not in (ACTIVITY.folder, ACTIVITY.path)]            # the log of this very session stays
        total = sum(item_size(p) for p in items)
        if not items:
            print(t("maintenance.nothing_to_remove"))
            return None
        print(tp("maintenance.will_remove", len(items), size=human_bytes(total)))
        print(t("maintenance.not_touched") + "\n")
        if not prompt_yes_no(t("common.proceed"), False):
            print(t("common.nothing_changed"))
            return False
        names = ", ".join(p.name for p in items[:20]) + (f", ... and {len(items) - 20} more" if len(items) > 20 else "")
        removed, failed = remove_items(items)
        print(tp("maintenance.removed", removed))
        for path, error in failed: print(t("maintenance.could_not_remove", path=path, error=error))
        ACTIVITY.log("MAINTENANCE", f"removed {removed} of {len(items)} {label} item(s), about {human_bytes(total)}: {names}")
        for path, error in failed: ACTIVITY.log("MAINTENANCE", f"could not remove {path}: {error}")
        return None

    def _clean_temp_now(self) -> None:
        self._screen(t("maintenance.clean_temp_title"))
        if not any(child.name != ".gitkeep" for child in DIRS["temp"].iterdir()): print(t("maintenance.temp_empty"))
        else: self.clean_abandoned_temp()

    def _show_usage(self) -> None:
        self._screen(t("maintenance.usage_title"))
        print(t("maintenance.program_folder", path=constants.APP_ROOT))
        print(t("maintenance.relative_paths") + "\n")
        for name, size in usage_report().items(): print(f"  {name:12}: {human_bytes(size)}")

    def _reset_processing_history(self) -> Optional[bool]:
        self._screen(t("maintenance.reset_title"))
        print(t("maintenance.reset_explain") + "\n")
        print(t("maintenance.reset_not_touched") + "\n")
        if prompt_yes_no(t("common.proceed"), False):
            backup = self.state.reset_with_backup()
            print(t("maintenance.reset_done", name=backup.name))
            ACTIVITY.log("MAINTENANCE", f"processing history reset; the previous database was archived as {backup.name}")
            return None
        print(t("common.nothing_changed"))
        ACTIVITY.log("MAINTENANCE", "reset of the processing history declined")
        return False

    def _forget_ffmpeg_history(self) -> Optional[bool]:
        self._screen(t("maintenance.forget_title"))
        print(t("maintenance.forget_explain") + "\n")
        if prompt_yes_no(t("common.proceed"), False):
            registry_backup = self.registry.forget_with_backup()
            print(t("maintenance.forget_done") + (" " + t("maintenance.forget_backup", name=registry_backup.name) if registry_backup else ""))
            ACTIVITY.log("MAINTENANCE", "FFmpeg build history forgotten" + (f"; backup: {registry_backup.name}" if registry_backup else ""))
            return None
        print(t("common.nothing_changed"))
        return False

    def _clean_old_files(self) -> Optional[bool]:
        """Run logs and diagnostics together, by age."""
        self._screen(t("maintenance.clean_logs_title"))
        choice = choose([("1", t("maintenance.older_30")), ("2", t("maintenance.older_90")), ("3", t("maintenance.everything")), ("0", t("common.back"))], "0")
        if choice == "0": return False
        if choice == "3": items = all_items(DIRS["logs"]) + all_items(DIRS["diagnostics"])
        else:
            days = {"1": 30, "2": 90}[choice]
            items = old_logs(days) + old_items(DIRS["diagnostics"], days, include_dirs=True)
        return self._delete_items(items, "log and diagnostics")
