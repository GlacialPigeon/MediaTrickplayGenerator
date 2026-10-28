# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from .activity_log import ACTIVITY, one_line
from .console_progress import CONSOLE_PROGRESS
from .constants import APP_NAME, APP_VERSION, DIRS
from .datetime_helpers import local_stamp, utc_now_iso
from .file_io import atomic_write_json
from .localization import english, t


# One of these happens for every video that needs nothing: they stay in the run log only. Neither the activity log nor events.jsonl
# gets them (the summary line has the counts), otherwise a big library would bury everything else.
ROUTINE_EVENTS = frozenset({"SKIPPED_VALID", "SKIPPED_VALID_FOREIGN"})


class RunLogger:
    def __init__(self, mode: str, run_id: Optional[str] = None) -> None:
        self.mode = mode
        self.run_id = run_id or f"{local_stamp()}_{uuid.uuid4().hex[:8]}"
        self.lock = threading.RLock()
        self._folder: Optional[Path] = None             # the folder of this run: made when the first file is written, so a run that does nothing leaves none
        # Inside the program every run writes into the log of this start; a folder of its own with a log only when nothing started one (tools, tests).
        shared = ACTIVITY.path
        self._shared = shared is not None
        self._log_fh: Any = None
        if shared is not None:
            self.path = shared
        else:
            self.path = self.folder / "run.log"
            self._log_fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self.log("INFO", f"{APP_NAME} {APP_VERSION} | mode={mode} | run_id={self.run_id}")
        ACTIVITY.log("RUN", f"{mode} started" + self._details())

    @property
    def folder(self) -> Path:
        """logs/<start>/<nn>_<mode>/ : summary.json, events.jsonl, failed.txt and (Verify) problems.txt of this run."""
        with self.lock:
            if self._folder is None:
                self._folder = ACTIVITY.new_run_folder(self.mode) or DIRS["logs"] / f"{self.run_id}_{self.mode}"
                self._folder.mkdir(parents=True, exist_ok=True)
            return self._folder

    @property
    def events_path(self) -> Path: return self.folder / "events.jsonl"

    @property
    def failed_path(self) -> Path: return self.folder / "failed.txt"

    @property
    def summary_path(self) -> Path: return self.folder / "summary.json"

    def _details(self) -> str:
        return "" if self._shared else f" | details: {self.path}"

    def _write_line(self, line: str) -> None:
        if self._shared: ACTIVITY.write_raw(line)
        elif self._log_fh is not None:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()

    def close(self) -> None:
        with self.lock:
            if self._log_fh is None: return
            try:
                self._log_fh.flush()
                os.fsync(self._log_fh.fileno())
            except Exception: pass
            self._log_fh.close()

    def log(self, level: str, message: str, *, echo: bool = False) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} [{level}] {message}"
        with self.lock:
            self._write_line(line)
            if echo: CONSOLE_PROGRESS.print_line(message)

    def say(self, level: str, key: str, **values: Any) -> None:
        """A message for the log (always English) that is also shown on the screen (in the language of the person)."""
        with english(): message = t(key, **values)
        self.log(level, message)
        CONSOLE_PROGRESS.print_line(t(key, **values))

    def event(self, status: str, **payload: Any) -> None:
        record = {"timestamp": utc_now_iso(), "run_id": self.run_id, "status": status, **payload}
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            if status not in ROUTINE_EVENTS:                       # a big library would write one of these per video and run
                with open(self.events_path, "a", encoding="utf-8", buffering=1) as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            if not (self._shared and status in ROUTINE_EVENTS):   # in the log of the whole start they would bury everything: the summary has the counts
                self._write_line(f"{dt.datetime.now().isoformat(timespec='seconds')} [EVENT] {line}")
        if status not in ROUTINE_EVENTS:
            ACTIVITY.log("RUN", f"{self.mode} {status} " + " ".join(f"{k}={one_line(v, 200)}" for k, v in payload.items()))

    def failed(self, path: Path, error: str) -> None:
        with self.lock:
            with open(self.failed_path, "a", encoding="utf-8") as fh:
                fh.write(f"{utc_now_iso()}\t{path}\t{error.replace(chr(10), ' | ')}\n")
        ACTIVITY.log("RUN", f"{self.mode} FAILED {path}: {one_line(error, 300)}")

    def write_summary(self, summary: dict[str, Any]) -> None:
        payload = {"run_id": self.run_id, "version": APP_VERSION, "mode": self.mode, **summary}
        atomic_write_json(self.summary_path, payload)
        shown = " ".join(f"{k}={v}" for k, v in summary.items() if isinstance(v, (int, float, str, bool)))
        ACTIVITY.log("RUN", f"{self.mode} finished | {shown}" + (f" | run folder: {self.folder.name}" if self._shared else self._details()))


class NullRunLogger(RunLogger):
    """Logger that writes nothing, for diagnostics: they keep their own log and result files in diagnostics/."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._shared = False
        self._log_fh = None
        self._folder = None
        self.run_id = f"{local_stamp()}_{uuid.uuid4().hex[:8]}"
        self.path = DIRS["logs"] / f"{self.run_id}.log"
        self.lock = threading.RLock()

    @property
    def folder(self) -> Path: return DIRS["logs"] / f"{self.run_id}_{self.mode}"        # never made

    def close(self) -> None: pass
    def log(self, level: str, message: str, *, echo: bool = False) -> None:
        if echo: CONSOLE_PROGRESS.print_line(message)

    def event(self, status: str, **payload: Any) -> None: pass
    def failed(self, path: Path, error: str) -> None: pass
    def write_summary(self, summary: dict[str, Any]) -> None: pass
