# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import atexit
import shutil
import sys
import threading
import time
from typing import Optional

from . import platform_api, time_estimate
from .error_summary import diagnostic_error_summary
from .formatting import human_duration
from .localization import number, t


class ConsoleProgress:
    """Thread-safe live dashboard. This is the only owner of live terminal output.

    The screen shows one header line (videos done, failed, active, elapsed, time left) and one line per active job with fixed columns
    (disk, file, path, percent, speed, time left). Workers only change the state; drawing happens from `tick()` a few times a second,
    as one write per frame that overwrites the lines in place (no clearing first, so nothing flickers), inside the terminal's
    "synchronized output" markers. The cursor is hidden while the dashboard is up and always comes back."""
    RESET = "\x1b[0m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    DIM = "\x1b[90m"
    SYNC_ON, SYNC_OFF = "\x1b[?2026h", "\x1b[?2026l"
    CURSOR_HIDE, CURSOR_SHOW = "\x1b[?25l", "\x1b[?25h"

    FRAME_INTERVAL = 0.5            # seconds between two frames (2 per second)
    MODE_WIDTH = 11                 # "GPU d3d11va"
    ETA_WIDTH = 12
    MAX_DISK_WIDTH = 10
    RESERVED_LINES = 6              # header, blank line, "+N more", stop message and the prompt stay on the screen

    def __init__(self) -> None:
        self.lock = threading.RLock()
        # key -> (method, backend, percent, speed, elapsed, eta); the job's disk, file name and length are in job_info
        self.job_states: dict[str, tuple[str, Optional[str], float, float, float, Optional[float]]] = {}
        self.job_info: dict[str, tuple[str, str, float]] = {}
        self.active_keys: set[str] = set()
        self.rendered_lines = 0
        self._enabled: Optional[bool] = None
        self.run_start: Optional[float] = None
        self.done = 0
        self.failed = 0
        self.total = 0
        self.total_seconds = 0.0            # length of the videos to do
        self.done_seconds = 0.0             # length of the videos finished (done or failed)
        self.disk_width = 2
        self.stop_message: Optional[str] = None
        self.stop_immediate = False
        self.last_width = self._terminal_width()
        self._resize_pending = False
        self._resize_changed_at = 0.0
        self._last_frame_at = 0.0
        self._cursor_hidden = False
        atexit.register(self._show_cursor_at_exit)

    @property
    def enabled(self) -> bool:
        """Live dashboard on/off. Decided on first use so importing this module needs no platform adapter."""
        if self._enabled is None:
            self._enabled = bool(platform_api.enable_virtual_terminal())
        return self._enabled

    def _eta_width(self) -> int:
        """The column of the time left is as wide as the longest text in the language of the moment (a longer one would wrap the line,
        and a wrapped line is one the cursor can no longer find its way back over)."""
        return max(self.ETA_WIDTH, len(t("progress.eta", time="00:00:00")))

    def _speed_width(self) -> int:
        """The column of the speed is as wide as its longest text in the language of the moment (up to 9999x)."""
        return len(t("progress.speed", speed="9999"))

    def _terminal_width(self) -> int:
        try:
            return max(40, shutil.get_terminal_size(fallback=(120, 30)).columns)
        except Exception:
            return 120

    def _terminal_height(self) -> int:
        try:
            return max(10, shutil.get_terminal_size(fallback=(120, 30)).lines)
        except Exception:
            return 30

    def _paint(self, text: str, color: str) -> str:
        return f"{color}{text}{self.RESET}" if self.enabled else text

    def _fit(self, text: str, width: Optional[int] = None) -> str:
        width = width or self._terminal_width()
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[:width - 3] + "..."

    # ---- terminal output: always one write ----

    def _write(self, text: str) -> None:
        """One write for a whole frame, between the synchronized-output markers, so the terminal never shows a half-drawn frame."""
        if not text: return
        sys.stdout.write(self.SYNC_ON + text + self.SYNC_OFF)
        sys.stdout.flush()

    def _hide_cursor(self) -> str:
        if self._cursor_hidden: return ""
        self._cursor_hidden = True
        return self.CURSOR_HIDE

    def _show_cursor(self) -> str:
        if not self._cursor_hidden: return ""
        self._cursor_hidden = False
        return self.CURSOR_SHOW

    def _show_cursor_at_exit(self) -> None:
        try:
            if self._cursor_hidden and self.enabled:
                sys.stdout.write(self.CURSOR_SHOW)
                sys.stdout.flush()
        except Exception:
            pass

    def _erase_text(self) -> str:
        """Moves to the first line of the dashboard and clears everything from there down; the next text starts on that line."""
        if self.rendered_lines <= 0: return ""
        text = f"\x1b[{self.rendered_lines}A\r\x1b[J"
        self.rendered_lines = 0
        return text

    # ---- what is shown ----

    def _left_text(self) -> str:
        """About how long it takes still: from the length of the videos finished and in progress against the time spent."""
        if self.run_start is None or self.total_seconds <= 0: return ""
        elapsed = time.monotonic() - self.run_start
        partial = sum(self.job_states[key][2] / 100.0 * self.job_info.get(key, ("", "", 0.0))[2] for key in self.job_states)
        progress = min(self.total_seconds, self.done_seconds + partial)
        if elapsed < 45 or progress <= 0: return t("progress.calculating")
        remaining = (self.total_seconds - progress) * elapsed / progress
        return t("progress.left", time=time_estimate.describe(remaining, remaining, True))

    def _header_text(self, width: int, paint: bool = True) -> str:
        parts = [t("progress.generating", done=number(self.done), total=number(self.total))]
        failed = t("progress.failed", count=number(self.failed))
        parts += [failed, t("progress.active", count=len(self.active_keys)), t("progress.elapsed", time=human_duration(time.monotonic() - (self.run_start or time.monotonic())))]
        left = self._left_text()
        if left: parts.append(left)
        while len(parts) > 1 and len(" | ".join(parts)) > width: parts.pop()
        if not paint or not self.enabled: return " | ".join(parts)
        if len(parts) > 1 and parts[1] == failed and self.failed: parts[1] = self._paint(failed, self.RED)
        return " | ".join(parts)

    def _job_line(self, key: str, name_width: int) -> str:
        method, backend, pct, speed, _elapsed, eta = self.job_states[key]
        disk, name, _seconds = self.job_info.get(key, ("", key, 0.0))
        mode = f"{method} {backend}" if backend and method == "GPU" else (method or t("progress.starting"))
        mode_text = self._fit(mode, self.MODE_WIDTH).ljust(self.MODE_WIDTH)
        eta_text = t("progress.eta", time=human_duration(eta) if eta is not None else "--:--")
        shown = self._fit(name, name_width).ljust(name_width)
        pct_text = self._paint(f"{pct:3.0f}%", self.CYAN)
        eta_part = self._paint(eta_text.ljust(self._eta_width()), self.DIM)
        mode_part = self._paint(mode_text, self.GREEN if method == 'GPU' else self.RESET)
        speed_part = t("progress.speed", speed=f"{speed:.0f}")        # no quotes of the same kind inside an f-string: Python 3.10 and 3.11 need that
        return f"{disk[:self.disk_width]:<{self.disk_width}}  {shown}  {mode_part}  {pct_text}  {speed_part:<{self._speed_width()}}  {eta_part}".rstrip()

    def _dashboard_lines_locked(self) -> list[str]:
        if self.run_start is None: return []
        width = self._terminal_width() - 1
        out = [self._header_text(width), ""]
        fixed = self.disk_width + 2 + 2 + self.MODE_WIDTH + 2 + 4 + 2 + self._speed_width() + 2 + self._eta_width()
        name_width = max(10, width - fixed)
        room = max(1, self._terminal_height() - self.RESERVED_LINES)
        keys = sorted(self.job_states, key=lambda k: (self.job_info.get(k, ("", "", 0.0))[0], k))
        shown = keys if len(keys) <= room else keys[:max(1, room - 1)]
        for key in shown: out.append(self._job_line(key, name_width))
        if len(keys) > len(shown): out.append(self._paint(t("progress.more", count=len(keys) - len(shown)), self.DIM))
        if self.stop_message:
            out.extend(["", self._paint(self._fit(self.stop_message, width), self.RED if self.stop_immediate else self.YELLOW)])
        return out

    def _frame_text(self) -> str:
        """The dashboard as one piece of text that replaces the previous one: each line is overwritten and ends with 'erase to the end of
        the line'; lines the new frame no longer has are erased at the end. Sets `rendered_lines`."""
        lines = self._dashboard_lines_locked()
        text = (f"\x1b[{self.rendered_lines}A\r" if self.rendered_lines > 0 else "\r") + "".join(line + "\x1b[K\n" for line in lines)
        if len(lines) < self.rendered_lines: text += "\x1b[J"
        self.rendered_lines = len(lines)
        return text

    def _draw_locked(self, extra_before: str = "") -> None:
        self._write(self._hide_cursor() + extra_before + self._frame_text())
        self._last_frame_at = time.monotonic()

    # ---- the run ----

    def begin_run(self, total: int, total_seconds: float = 0.0) -> None:
        with self.lock:
            self.run_start = time.monotonic()
            self.total = total
            self.total_seconds = total_seconds
            self.done_seconds = 0.0
            self.done = 0
            self.failed = 0
            self.stop_message = None
            self.stop_immediate = False
            self.job_states.clear()
            self.job_info.clear()
            self.active_keys.clear()
            self.disk_width = 2
            self.rendered_lines = 0
            self.last_width = self._terminal_width()
            self._resize_pending = False
            self._resize_changed_at = 0.0
            if self.enabled: self._draw_locked()

    def end_run(self) -> None:
        with self.lock:
            if self.enabled and (self.rendered_lines > 0 or self._cursor_hidden):
                self._write(self._erase_text() + self._show_cursor())
            self.job_states.clear()
            self.job_info.clear()
            self.active_keys.clear()
            self.run_start = None
            self.stop_message = None
            self.rendered_lines = 0

    def register_job(self, key: str, disk: str = "", name: str = "", seconds: float = 0.0) -> None:
        """A video is started: `disk` (like X:) and `name` are shown in its line, `seconds` is its length (for the time left)."""
        with self.lock:
            self.active_keys.add(key)
            self.job_info[key] = (disk, name or key, seconds)
            self.disk_width = max(self.disk_width, min(len(disk), self.MAX_DISK_WIDTH))
            self.job_states.setdefault(key, ("", None, 0.0, 0.0, 0.0, None))

    def update_job(self, key: str, label: str, method: str, backend: Optional[str], pct: float,
                   speed: float, elapsed: float, eta: Optional[float]) -> None:
        with self.lock:
            if not self.enabled or self.run_start is None or key not in self.job_info:
                return                   # not a video of this run (the benchmark and the checks draw their own line)
            self.job_states[key] = (method, backend, pct, speed, elapsed, eta)

    def finish(self, key: str, final: bool = False) -> None:
        """The video is done (`final`). The end of one FFmpeg run alone changes nothing: the video may need another one (a redo with
        every frame, the CPU after the GPU), and its line stays where it is."""
        with self.lock:
            if not final: return
            self.job_states.pop(key, None)
            self.job_info.pop(key, None)
            self.active_keys.discard(key)

    def record_done(self, seconds: float = 0.0) -> None:
        with self.lock:
            self.done += 1
            self.done_seconds += seconds

    def record_failed(self, index_text: str, path: str, attempts: list[str], error: str, culprit: Optional[list[str]] = None, seconds: float = 0.0) -> None:
        with self.lock:
            self.failed += 1
            self.done_seconds += seconds
            human = diagnostic_error_summary(error, localized=True)
            lines = [self._paint(t("progress.failed_job", index=index_text, path=path), self.RED)]
            if attempts: lines.append(self._paint("  " + t("progress.tried", attempts=" -> ".join(attempts)), self.YELLOW))
            for name in culprit or []:
                if name != path: lines.append(self._paint("  " + t("progress.concerns", name=name), self.YELLOW))
            lines.append(self._paint("  " + t("progress.error", error=human), self.RED))
            lines.append("")
            self._print_above("\n".join(lines) + "\n")

    def set_stop(self, immediate: bool = False) -> None:
        with self.lock:
            self.stop_immediate = immediate
            self.stop_message = t("progress.stopping") if immediate else t("progress.finishing")
            if self.enabled and self.run_start is not None:
                self._draw_locked()
            elif not self.enabled:
                print(self.stop_message)

    def _print_above(self, text: str) -> None:
        """Text that stays on the screen: written where the dashboard was, then the dashboard again below it (one write)."""
        if self.enabled and self.run_start is not None:
            self._write(self._hide_cursor() + self._erase_text() + text + self._frame_text())
            self._last_frame_at = time.monotonic()
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def tick(self) -> None:
        """Draw the next frame (at most twice a second) and settle console resizes."""
        with self.lock:
            if not self.enabled or self.run_start is None:
                return
            now = time.monotonic()
            width = self._terminal_width()

            if width != self.last_width:
                self.last_width = width
                self._resize_pending = True
                self._resize_changed_at = now
                return

            if self._resize_pending:
                if now - self._resize_changed_at < 0.25:
                    return
                # Resize has settled. Old cursor geometry is no longer reliable, so rebuild the visible dashboard once.
                self._resize_pending = False
                self.rendered_lines = 0
                self._write("\x1b[2J\x1b[H")
                self._draw_locked()
                return

            if now - self._last_frame_at < self.FRAME_INTERVAL:
                return
            self._draw_locked()

    def print_line(self, text: str = "") -> None:
        """Persistent non-dashboard message, kept for diagnostics and non-generation output."""
        with self.lock:
            self._print_above(text + "\n")


CONSOLE_PROGRESS = ConsoleProgress()
