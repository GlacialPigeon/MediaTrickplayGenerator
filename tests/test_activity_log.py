# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The activity log: one file per start of the program, everything in time order."""
from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import shutil

from tests.helpers import FFMPEG, FFPROBE, IsolatedApp, host_adapter, needs_ffmpeg
from tests.test_screens import ScreenTest

from modules import cli, constants
from modules.activity_log import ACTIVITY, ActivityLog, describe_changes, one_line
from modules.application import run_application
from modules.config import ConfigManager
from modules.logger import RunLogger


def sessions() -> list[Path]:
    """The session.log of every start of the program (each start has a folder of its own)."""
    return sorted(constants.DIRS["logs"].glob("*/session.log"))


def text_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class Changes(unittest.TestCase):
    def test_only_what_differs_is_listed_with_dotted_keys(self) -> None:
        old = {"schema_version": 1, "workers": {"probe_concurrency": {"D:": 4}, "jobs": 2}, "media_roots": ["a"]}
        new = {"schema_version": 2, "workers": {"probe_concurrency": {"D:": 2}, "jobs": 2}, "media_roots": ["a"], "diagnostics": {"folder": "x"}}
        self.assertEqual(describe_changes(old, new), ['diagnostics.folder: (not set) -> "x"', "workers.probe_concurrency.D:: 4 -> 2"])

    def test_nothing_changed_nothing_listed(self) -> None:
        self.assertEqual(describe_changes({"a": {"b": 1}}, {"a": {"b": 1}}), [])

    def test_removed_values_and_long_values_and_many_changes(self) -> None:
        self.assertEqual(describe_changes({"a": 1}, {}), ["a: 1 -> (not set)"])
        # a setting that is null or missing on both sides did not change; null turning into a group shows the group's keys only
        self.assertEqual(describe_changes({"a": {"x": 1}}, {"a": {"x": 1, "gpu": None}}), [])
        self.assertEqual(describe_changes({"a": {"gpu": None}}, {"a": {"gpu": {"x": "PASS"}}}), ["a.gpu.x: (not set) -> \"PASS\""])
        self.assertEqual(describe_changes({"w": {"probe": {}}}, {"w": {"probe": {"D": 4}}}), ["w.probe.D: (not set) -> 4"])
        self.assertEqual(describe_changes({"w": {"probe": {"D": 4}}}, {"w": {"probe": {}}}), ["w.probe.D: 4 -> (not set)"])
        long = describe_changes({}, {"k": "x" * 500})[0]
        self.assertLess(len(long), 200)
        many = describe_changes({}, {f"k{i:03d}": i for i in range(100)})
        self.assertEqual(len(many), 41)
        self.assertIn("60 more", many[-1])

    def test_one_line(self) -> None:
        self.assertEqual(one_line("first\n\n second \nthird"), "first | second | third")
        self.assertEqual(len(one_line("x" * 1000)), 600)


class TheFile(IsolatedApp):
    def tearDown(self) -> None:
        ACTIVITY.close()
        super().tearDown()

    def test_writes_nothing_before_it_is_started(self) -> None:
        log = ActivityLog()
        log.log("MENU", "x")
        log.config_changed({}, {"a": 1})
        self.assertEqual(sessions(), [])

    def test_every_start_gets_its_own_file_with_timestamped_lines(self) -> None:
        log = ActivityLog()
        first = log.start("interactive", platform="test")
        log.log("MENU", "Main menu > Verify\nsecond line")
        log.close()
        log.close()                                                  # harmless twice
        second = log.start("generate")
        log.close()
        self.assertNotEqual(first, second)
        self.assertEqual(len(sessions()), 2)
        lines = text_of(first).splitlines()
        self.assertRegex(lines[0], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d \[START\] .*mode=interactive \| platform=test")
        self.assertRegex(lines[1], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d \[MENU\] Main menu > Verify \| second line$")
        self.assertEqual(lines, sorted(lines, key=lambda l: l[:19]), "in time order")

    def test_configuration_changes_are_logged_wherever_they_are_saved(self) -> None:
        mgr = ConfigManager()
        config = mgr.load()
        config["media_roots"] = ["D:/a"]
        mgr.save()
        ACTIVITY.start("test")
        mgr.config["trickplay"]["width"] = 480
        mgr.save()
        mgr.save()                                                   # nothing new: nothing logged
        lines = [l for l in text_of(ACTIVITY.path).splitlines() if "[CONFIG]" in l]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("trickplay.width: 320 -> 480", lines[0])

    def test_routine_skips_stay_out_of_the_activity_log(self) -> None:
        ACTIVITY.start("test")
        run = RunLogger("generate")
        for i in range(50):
            run.event("SKIPPED_VALID_FOREIGN", path=f"D:/{i}.mkv")
            run.event("SKIPPED_VALID", path=f"D:/v{i}.mkv")
        run.event("INVALID_FINAL_REBUILD", path="D:/x.mkv", reason="tile count 6 outside expected [3]")
        run.close()
        text = text_of(ACTIVITY.path)
        self.assertNotIn("SKIPPED_VALID", text)
        self.assertIn("INVALID_FINAL_REBUILD path=D:/x.mkv", text)
        self.assertEqual(run.path, ACTIVITY.path, "a run writes into the log of this start, not into a file of its own")
        # a run logger with no log of a start (tools, tests) still keeps everything in its own file
        ACTIVITY.close()
        alone = RunLogger("generate")
        for i in range(100): alone.event("SKIPPED_VALID", path=f"D:/{i}.mkv")
        alone.close()
        self.assertEqual(sum("SKIPPED_VALID" in l for l in text_of(alone.path).splitlines()), 100)

    def test_events_json_gets_no_routine_skips(self) -> None:
        run = RunLogger("generate")
        for i in range(20): run.event("SKIPPED_VALID_FOREIGN", path=f"D:/{i}.mkv")
        run.event("FAILED", path="D:/x.mkv", error="boom")
        run.close()
        lines = run.events_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('"status": "FAILED"', lines[0])

    def test_every_run_of_a_start_has_a_folder_of_its_own_and_nothing_is_overwritten(self) -> None:
        ACTIVITY.start("test")
        first = RunLogger("generate")
        first.event("FAILED", path="D:/x.mkv", error="boom")
        first.failed(Path("D:/x.mkv"), "boom")
        first.write_summary({"scanned": 3, "failed": 1})
        first.close()
        second = RunLogger("verify")
        second.write_summary({"scanned": 3, "failed": 0})
        second.close()
        names = sorted(p.name for p in ACTIVITY.folder.iterdir())
        self.assertEqual(names, ["01_generate", "02_verify", "session.log"])
        self.assertEqual(sorted(p.name for p in (ACTIVITY.folder / "01_generate").iterdir()), ["events.jsonl", "failed.txt", "summary.json"])
        self.assertEqual(sorted(p.name for p in (ACTIVITY.folder / "02_verify").iterdir()), ["summary.json"])
        self.assertIn('"failed": 1', (ACTIVITY.folder / "01_generate" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn('"failed": 0', (ACTIVITY.folder / "02_verify" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(p.name for p in constants.DIRS["logs"].iterdir()), [ACTIVITY.folder.name], "nothing else in logs/")

    def test_a_run_that_writes_nothing_leaves_no_folder_and_uses_no_number(self) -> None:
        ACTIVITY.start("test")
        RunLogger("verify").close()                                  # asked nothing, found nothing
        run = RunLogger("generate")
        run.write_summary({"scanned": 0})
        run.close()
        self.assertEqual(sorted(p.name for p in ACTIVITY.folder.iterdir()), ["01_generate", "session.log"])

    def test_a_run_without_a_start_keeps_everything_in_a_folder_of_its_own(self) -> None:
        run = RunLogger("generate")
        run.event("FAILED", path="D:/x.mkv", error="boom")
        run.close()
        self.assertEqual(sorted(p.name for p in run.folder.iterdir()), ["events.jsonl", "run.log"])
        self.assertEqual(run.folder.parent, constants.DIRS["logs"])

    def test_a_run_log_points_to_its_details(self) -> None:
        ACTIVITY.start("test")
        run = RunLogger("generate")
        run.event("FAILED", path="D:/x.mkv", error="boom")
        run.write_summary({"scanned": 3, "failed": 1, "per_storage_group": {"a": 1}})
        run.close()
        text = text_of(ACTIVITY.path)
        self.assertEqual(run.path, ACTIVITY.path)
        self.assertIn("[RUN] generate started\n", text + "\n")
        self.assertIn("[RUN] generate FAILED path=D:/x.mkv error=boom", text)
        self.assertIn("[EVENT]", text, "the detailed lines of the run are in the same file")
        self.assertRegex(text, r"generate finished \| scanned=3 failed=1 \| run folder: 01_generate$")
        self.assertNotIn("per_storage_group", text.split("[RUN] generate finished")[1], "only the simple figures of the summary")
        self.assertEqual(sessions(), [ACTIVITY.path], "one log for the whole start")
        self.assertTrue((ACTIVITY.folder / "01_generate" / "summary.json").exists(), "the summary is in the folder of the run")


class Menus(ScreenTest):
    def tearDown(self) -> None:
        ACTIVITY.close()
        super().tearDown()

    def test_menu_entries_are_logged_with_their_outcome(self) -> None:
        ACTIVITY.start("test")
        self.feed(lambda: self.app._menu("Test menu", [("Do something", lambda: None, False)]), "1", "0")
        text = text_of(ACTIVITY.path)
        self.assertIn("[MENU] Test menu > Do something\n", text)
        self.assertRegex(text, r"\[MENU\] Test menu > Do something: done in \d+\.\ds")

    def test_an_entry_that_fails_is_logged_and_still_raises(self) -> None:
        ACTIVITY.start("test")
        def boom() -> None: raise RuntimeError("disk gone")
        with self.assertRaises(RuntimeError):
            self.feed(lambda: self.app._menu("Test menu", [("Do something", boom, False)]), "1")
        self.assertIn("Test menu > Do something: ended by RuntimeError: disk gone", text_of(ACTIVITY.path))

    def test_maintenance_is_logged_and_never_deletes_the_log_of_this_session(self) -> None:
        ACTIVITY.start("test")
        old = constants.DIRS["logs"] / "old.log"
        old.write_text("x", encoding="utf-8")
        self.feed(lambda: self.app._delete_items([old, ACTIVITY.path], "log"), "y")
        self.assertFalse(old.exists())
        self.assertTrue(ACTIVITY.path.exists())
        self.assertIn("[MAINTENANCE] removed 1 of 1 log item(s)", text_of(ACTIVITY.path))
        self.assertIn("old.log", text_of(ACTIVITY.path))


@needs_ffmpeg
class FfmpegLine(ScreenTest):
    def tearDown(self) -> None:
        ACTIVITY.close()
        super().tearDown()

    def test_the_ffmpeg_in_use_is_written_once_per_start_not_for_every_screen(self) -> None:
        folder = constants.APP_ROOT / "ffmpeg" / "bin"
        folder.mkdir(parents=True)
        for source in (FFMPEG, FFPROBE): shutil.copy2(source, folder / Path(source).name)
        ACTIVITY.start("test")
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3): self.assertTrue(self.app.ensure_ffmpeg(interactive=False))
        self.assertEqual(text_of(ACTIVITY.path).count("[FFMPEG]"), 1)


class WholeProgram(IsolatedApp):
    def run_cli(self, *argv: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), mock.patch("builtins.input", side_effect=AssertionError("asked")):
            return run_application(host_adapter(), platform_name="test", argv=list(argv))

    def test_a_start_leaves_one_log_with_start_and_exit(self) -> None:
        self.assertEqual(self.run_cli("--verify"), cli.EXIT_NOT_RUN)          # no media roots
        [log] = sessions()
        lines = text_of(log).splitlines()
        self.assertRegex(lines[0], r"\[START\] .*mode=verify \| platform=test")
        self.assertRegex(lines[-1], r"\[EXIT\] exit code 2 \| session log: ")
        self.assertIsNone(ACTIVITY.path, "closed again")

    def test_a_second_start_is_a_second_file(self) -> None:
        self.run_cli("--verify")
        self.run_cli("--generate", "--dry-run")
        self.assertEqual(len(sessions()), 2)
        self.assertEqual(sum("mode=generate --dry-run" in text_of(log) for log in sessions()), 1)

    def test_version_and_wrong_arguments_leave_no_log(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_application(host_adapter(), platform_name="test", argv=["--version"])
        self.assertEqual(sessions(), [])

    def test_old_logs_are_removed_at_the_start_but_never_recent_ones(self) -> None:
        import os
        import time
        logs = constants.DIRS["logs"]
        old_time = time.time() - 100 * 86400
        old_start = logs / "2026-01-01_000000_aaaaaaaa"                # a start of the program with a run in it
        (old_start / "01_generate").mkdir(parents=True)
        (old_start / "session.log").write_text("x", encoding="utf-8")
        (old_start / "01_generate" / "events.jsonl").write_text("x", encoding="utf-8")
        recent_start = logs / "2026-01-02_000000_bbbbbbbb"             # the folder itself is old, but its log was written lately
        recent_start.mkdir()
        (recent_start / "session.log").write_text("x", encoding="utf-8")
        legacy = ["2026-01-01_000000_session_aaaaaaaa.log", "x.summary.json", "x.verify.txt", "crash_2026-01-01.log", "events.jsonl", "run_summary.json",
                  "failed.txt", "verify_problems.txt"]                  # the files of earlier versions
        for name in legacy + ["recent.log"]: (logs / name).write_text("x", encoding="utf-8")
        for path in [old_start, old_start / "session.log", recent_start, *(logs / name for name in legacy)]: os.utime(path, (old_time, old_time))
        self.run_cli("--verify")
        self.assertFalse(old_start.exists())
        self.assertTrue(recent_start.exists())
        for gone in legacy: self.assertFalse((logs / gone).exists(), gone)
        self.assertTrue((logs / "recent.log").exists())
        [log] = [p for p in sessions() if "2026-01-0" not in p.parent.name]
        self.assertIn("removed 9 log item(s) older than 90 days", text_of(log))

    @needs_ffmpeg
    def test_a_whole_generate_leaves_one_folder_per_start_with_a_folder_per_run(self) -> None:
        from tests.helpers import make_video
        make_video(self.media / "a.mp4", 12)
        config = ConfigManager()
        config.load()
        config.config["media_roots"] = [str(self.media)]
        config.config["trickplay"].update(width=64, interval_seconds=5.0, tile_columns=3, tile_rows=3)
        config.config["acceleration"]["mode"] = "cpu"
        config.save()
        from modules.ui import Application
        from tests.helpers import build_tools

        def with_tools(app, interactive=True):
            app.tools = build_tools()
            app.ffmpeg_candidate = app.tools.candidate
            return True
        with mock.patch.object(Application, "ensure_ffmpeg", with_tools):
            self.assertEqual(self.run_cli("--generate"), cli.EXIT_OK)
            self.assertEqual(self.run_cli("--verify"), cli.EXIT_OK)
        logs = constants.DIRS["logs"]
        names = sorted(p.name for p in logs.iterdir())
        self.assertEqual(len(names), 2, names)                                              # a folder per start, nothing else
        self.assertEqual(len(sessions()), 2)
        folders = sorted(p.name for p in logs.glob("*/*_*") if p.is_dir())
        self.assertEqual(folders, ["01_generate", "01_verify"], "a folder per run, in the folder of its start")
        self.assertTrue(next(logs.glob("*/01_generate/summary.json")).exists())
        self.assertTrue(next(logs.glob("*/01_verify/problems.txt")).exists())
        generate_log = next(text_of(p) for p in sessions() if "mode=generate" in text_of(p))
        self.assertIn("[INFO]", generate_log)
        self.assertIn("generate finished", generate_log)

    def test_a_crash_is_written_into_the_log_of_the_start(self) -> None:
        from modules.ui import Application
        with mock.patch.object(Application, "run_command", side_effect=RuntimeError("boom")), \
                mock.patch("modules.application.ACTIVITY", ACTIVITY):
            self.assertEqual(self.run_cli("--verify"), cli.EXIT_CRASH)
        [log] = sessions()
        text = text_of(log)
        self.assertIn("RuntimeError: boom", text)
        self.assertIn("Traceback", text)
        self.assertEqual(list(constants.DIRS["logs"].glob("crash_*.log")), [])


if __name__ == "__main__":
    unittest.main()
