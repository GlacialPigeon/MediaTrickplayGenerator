# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The live dashboard of a Generate run: fixed columns, one write per frame, no clearing before drawing, the cursor always comes back."""
from __future__ import annotations

import io
import os
import re
import sys
import unittest
from unittest import mock

from pathlib import Path

from modules import localization
from modules.console_progress import ConsoleProgress

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LONG = "Miraculous - Tales of Ladybug & Cat Noir (2015) - S06E20 - Sadnansi with a very long name.mp4"


class Capture(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    def write(self, s: str) -> int:
        self.writes.append(s)
        return len(s)

    def flush(self) -> None: pass
    def isatty(self) -> bool: return True


class Dashboard(unittest.TestCase):
    def setUp(self) -> None:
        self.progress = ConsoleProgress()
        self.progress._enabled = True
        self.out = Capture()
        for patcher in (mock.patch.object(sys, "stdout", self.out),
                        mock.patch("shutil.get_terminal_size", side_effect=lambda fallback=(120, 30): os.terminal_size((100, 24)))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.progress._cursor_hidden = False

    def start(self, jobs: int = 2) -> list[str]:
        p = self.progress
        p.begin_run(10, 10 * 3600.0)
        keys = []
        for n in range(jobs):
            key = f"[{n + 1}/10] {LONG}"
            keys.append(key)
            p.register_job(key, "X:" if n % 2 == 0 else "Y:", LONG if n == 0 else f"Short {n}.mkv", 3600.0)
            p.update_job(key, LONG, "GPU" if n % 2 == 0 else "CPU", "cuda" if n % 2 == 0 else None, 40.0, 12.3, 30.0, 41.0)
        return keys

    def frame(self) -> str:
        self.out.writes.clear()
        self.progress._last_frame_at = 0
        self.progress.tick()
        self.assertEqual(len(self.out.writes), 1, "one write per frame")
        return self.out.writes[0]

    def test_a_frame_overwrites_the_old_one_without_clearing_it_first(self) -> None:
        self.start()
        self.progress._last_frame_at = 0
        self.progress.tick()
        text = self.frame()
        self.assertTrue(text.startswith("\x1b[?2026h") and text.endswith("\x1b[?2026l"), "synchronized output")
        self.assertNotIn("\x1b[2K", text)
        self.assertNotIn("\x1b[J", text, "nothing to erase when the number of lines did not change")
        self.assertIn("\x1b[4A\r", text, "back to the first line of the dashboard")
        self.assertEqual(text.count("\x1b[K\n"), 4)

    def test_frames_come_at_most_twice_a_second(self) -> None:
        self.start()
        self.out.writes.clear()
        self.progress._last_frame_at = 0
        for _ in range(20): self.progress.tick()
        self.assertEqual(len(self.out.writes), 1)

    def test_workers_changing_the_state_draw_nothing(self) -> None:
        keys = self.start()
        self.out.writes.clear()
        self.progress.update_job(keys[0], LONG, "GPU", "cuda", 55.0, 10.0, 31.0, 20.0)
        self.progress.record_done(3600.0)
        self.progress.finish(keys[1], final=True)
        self.assertEqual(self.out.writes, [])

    def test_fewer_lines_erase_what_is_left_over_in_the_same_write(self) -> None:
        keys = self.start()
        self.frame()
        self.progress.finish(keys[1], final=True)
        text = self.frame()
        self.assertIn("\x1b[J", text)
        self.assertEqual(self.progress.rendered_lines, 3)

    def test_the_header_names_done_failed_active_elapsed_and_time_left(self) -> None:
        self.start()
        self.progress.record_done(3600.0)
        self.progress.run_start -= 600
        header = ANSI.sub("", self.frame()).split("\n")[0].replace("\r", "")
        self.assertRegex(header, r"^Generating: 1 of 10 done \| 0 failed \| 2 active \| 10:\d\d elapsed \| about .+ left")

    def test_job_lines_have_fixed_columns_and_never_pass_the_width(self) -> None:
        self.start()
        lines = [ANSI.sub("", line).replace("\r", "") for line in self.frame().split("\n")]
        jobs = [line for line in lines if line.startswith(("X:", "Y:"))]
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(len(line) <= 99 for line in lines), [len(line) for line in lines])
        self.assertTrue(jobs[0].startswith("X:  Miraculous") and "..." in jobs[0], "a long name is cut with dots")
        columns = [(m.start("mode"), m.start("pct"), m.start("speed"), m.start("eta")) for line in jobs
                   if (m := re.search(r"  (?P<mode>GPU cuda|CPU)\s+(?P<pct>\s*\d+%)  (?P<speed>\s*\d+x speed)\s+(?P<eta>ETA \S+)", line))]
        self.assertEqual(len(columns), 2, jobs)
        self.assertEqual(columns[0][0], columns[1][0], "the path column starts at the same place in every line")
        self.assertRegex(jobs[0], r"GPU cuda +40%  12x speed +ETA 00:41")
        self.assertRegex(jobs[1], r"CPU +40%  12x speed +ETA 00:41")

    def test_many_jobs_are_cut_to_the_height_with_a_count_of_the_rest(self) -> None:
        p = self.progress
        p.begin_run(100, 100 * 3600.0)
        for n in range(40):
            p.register_job(f"k{n}", "X:", f"Video {n}.mkv", 3600.0)
            p.update_job(f"k{n}", f"Video {n}.mkv", "CPU", None, 10.0, 2.0, 5.0, 100.0)
        text = ANSI.sub("", self.frame())
        self.assertRegex(text, r"\+\d+ more")
        self.assertLessEqual(p.rendered_lines, 24 - ConsoleProgress.RESERVED_LINES + 3)

    def test_the_end_of_one_ffmpeg_run_keeps_the_line_of_a_video_that_goes_on(self) -> None:
        keys = self.start()
        self.progress.finish(keys[0])                          # e.g. the first pass is over, a redo follows
        self.progress.update_job(keys[0], LONG, "CPU", None, 5.0, 1.0, 2.0, 30.0)
        lines = [ANSI.sub("", line) for line in self.frame().split("\n")]
        self.assertTrue(any(line.startswith("X:  Miraculous") for line in lines), "still with its disk and name")
        self.assertFalse(any("[1/10]" in line for line in lines))

    def test_a_video_that_has_not_reported_yet_says_it_is_starting(self) -> None:
        self.progress.begin_run(3, 3 * 3600.0)
        self.progress.register_job("k", "X:", "New.mkv", 3600.0)
        text = ANSI.sub("", self.frame())
        self.assertRegex(text, r"X:  New\.mkv +starting +0%")

    def test_the_cursor_is_hidden_during_the_run_and_comes_back_at_the_end(self) -> None:
        self.start()
        self.assertIn("\x1b[?25l", "".join(self.out.writes))
        self.out.writes.clear()
        self.progress.end_run()
        self.assertIn("\x1b[?25h", "".join(self.out.writes))
        self.assertIn("\x1b[J", "".join(self.out.writes), "the dashboard is removed")

    def test_a_failure_stays_above_the_dashboard_in_one_write(self) -> None:
        self.start()
        self.frame()
        self.out.writes.clear()
        self.progress.record_failed("[3/10]", "X:/v/a.mkv", ["GPU/cuda", "CPU"], "ffmpeg failed")
        self.assertEqual(len(self.out.writes), 1)
        text = ANSI.sub("", self.out.writes[0])
        self.assertLess(text.index("FAILED [3/10] X:/v/a.mkv"), text.index("Generating:"))
        self.assertEqual(self.progress.failed, 1)

    def test_the_time_left_waits_for_a_minute_of_work_then_follows_the_speed(self) -> None:
        self.start()
        self.assertEqual(self.progress._left_text(), "calculating time left")
        self.progress.record_done(3600.0)
        self.progress.run_start -= 3600           # 1 h of video finished and 2 x 40 % of an hour under way, in an hour: 4.5 hours still to do
        self.assertEqual(self.progress._left_text(), "about 4 h 30 min left")


class Languages(Dashboard):
    """A longer text in another language must never make a line wider than the terminal: the cursor would lose count of the lines."""

    def test_no_line_is_wider_than_the_terminal_in_any_shipped_language(self) -> None:
        manager = localization.LocaleManager(Path(__file__).resolve().parent.parent / "locales")
        manager.load_canonical()
        with mock.patch.object(localization, "LOCALES", manager):
            for entry in manager.discover():
                manager.activate(entry["Locale"])
                self.start(3)
                self.progress.stop_message = None
                for line in self.progress._dashboard_lines_locked():
                    self.assertLess(len(ANSI.sub("", line)), 100, f"{entry['Locale']}: {line!r}")
                self.progress.end_run()


if __name__ == "__main__":
    unittest.main()
