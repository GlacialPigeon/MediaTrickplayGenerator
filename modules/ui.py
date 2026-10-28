# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

from . import platform_api
from .cli import EXIT_CANCELED, EXIT_NOT_RUN, EXIT_OK, EXIT_PROBLEMS, CliArgs
from .localization import lazy, t
from .logger import RunLogger
from .screens.verify import VerifyScreens
from .verify import VALID


class Application(VerifyScreens):
    """The terminal application: main menu and command-line mode on top of the screen layers (modules/screens/)."""

    def run_command(self, args: CliArgs) -> int:
        """Run one action from the command line without any questions (for schedulers). Returns the exit code."""
        if args.command == "self-test":
            return EXIT_OK if self._run_self_test()["failed"] == 0 else EXIT_PROBLEMS
        if not self.config.get("media_roots"):
            print(t("msg.no_roots"))
            return EXIT_NOT_RUN
        if not self.ensure_ffmpeg(interactive=False):
            ffmpeg_name, _ = platform_api.ffmpeg_executable_names()
            print(t("ffmpeg.not_found_expected", path=self.registry.ffmpeg_folder() / 'bin' / ffmpeg_name))
            return EXIT_NOT_RUN
        if not self._require_pillow(): return EXIT_NOT_RUN
        verify = args.command == "verify"
        logger = RunLogger("verify" if verify else "dry-run" if args.dry_run else "generate")
        self.state.begin_run(logger.run_id, logger.mode)
        try:
            videos = self.scan_library(logger)
            if not videos:
                print(t("msg.no_videos"))
                return EXIT_NOT_RUN
            if verify:
                checked = self._check_videos(videos, logger)
                if checked is None: return EXIT_CANCELED
                return EXIT_PROBLEMS if any(row.status != VALID for row in checked[1]) or self.unreadable_videos else EXIT_OK
            ran = self._plan_and_run(videos, logger, dry_run=args.dry_run, interactive=False)
            if args.dry_run: return EXIT_OK
            if not ran or self.last_summary is None: return EXIT_NOT_RUN
            if self.last_summary["canceled"]: return EXIT_CANCELED       # Ctrl+C: what was running finished, the rest was not started
            # a folder with unexpected files that was left alone still has no valid output: the scheduler should hear about it
            return EXIT_PROBLEMS if self.last_summary["failed"] or self.last_summary["skipped_unclear_folder"] or self.unreadable_videos else EXIT_OK
        finally:
            logger.close()

    def main_menu(self) -> None:
        self._menu(lazy("screen.main_menu"), [
            (lazy("menu.generate"), self.generate, False),
            (lazy("menu.verify"), self.verify, False),
            (lazy("menu.benchmark"), self.benchmark_menu, False),
            (lazy("screen.configuration"), self.configuration_menu, False),
            (lazy("screen.maintenance"), self.maintenance, False),
        ], back_label=lazy("menu.exit"), enter_goes_back=False)

    def maintenance(self) -> None:
        self._menu(lazy("screen.maintenance"), [
            (lazy("maintenance.self_test"), self.self_test, False),
            (lazy("maintenance.show_usage"), self._show_usage, True),
            (lazy("maintenance.clean_temp"), self._clean_temp_now, True),
            (lazy("maintenance.clean_logs"), self._clean_old_files, True),
            (lazy("maintenance.reset_history"), self._reset_processing_history, True),
            (lazy("maintenance.forget_ffmpeg"), self._forget_ffmpeg_history, True),
        ])
