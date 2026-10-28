# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""One folder for everything that happens during a start of the program.

logs/<time>_<id>/session.log holds the start and end of the program, every menu entry, every change of the configuration,
cleaning and diagnostics, and the details of every Generate / Verify run of that start (RunLogger writes into the same file).
Every such run also gets a folder of its own in it, logs/<time>_<id>/<nn>_<mode>/ (summary.json, events.jsonl, failed.txt,
problems.txt), so nothing is ever overwritten. Diagnostics keep their own reports in diagnostics/<kind>/...
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from .constants import APP_NAME, APP_VERSION, DIRS
from .datetime_helpers import local_stamp

SESSION_LOG = "session.log"
MAX_VALUE = 160                      # longest value shown in a configuration change
MAX_CHANGES = 40                     # most configuration changes written for one save
IGNORED_KEYS = {"schema_version"}


def one_line(text: object, limit: int = 600) -> str:
    flat = " | ".join(part.strip() for part in str(text).splitlines() if part.strip())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and (value or not prefix):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def _show(value: Any) -> str:
    if value is None: return "(not set)"
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= MAX_VALUE else text[: MAX_VALUE - 3] + "..."


def describe_changes(old: Any, new: Any) -> list[str]:
    """`key: old -> new` for every setting that differs between two configurations (nested keys are joined with dots)."""
    before = _flatten(old) if isinstance(old, dict) else {}
    after = _flatten(new) if isinstance(new, dict) else {}
    lines: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key.split(".")[0] in IGNORED_KEYS: continue
        if key in before and key in after and before[key] == after[key]: continue
        # "not set", null and an empty group are the same thing: a group that fills up shows its keys, not "{} -> (not set)"
        if before.get(key) in (None, {}) and after.get(key) in (None, {}): continue
        # An empty dict is a value like any other: {} -> {"a": 1} shows up as the new key "a" and the old "(not set)".
        lines.append(f"{key}: {_show(before.get(key))} -> {_show(after.get(key))}")
    if len(lines) > MAX_CHANGES:
        extra = len(lines) - MAX_CHANGES
        lines = lines[:MAX_CHANGES] + [f"... and {extra} more change(s)"]
    return lines


class ActivityLog:
    """Writes nothing until start() has been called, so tests and tools that merely import the program leave no files."""

    def __init__(self) -> None:
        self.path: Optional[Path] = None             # session.log of this start
        self.folder: Optional[Path] = None           # the folder of this start
        self._runs = 0
        self._fh: Any = None
        self._lock = threading.RLock()

    def start(self, mode: str, **details: Any) -> Optional[Path]:
        with self._lock:
            self.close()
            try:
                folder = DIRS["logs"] / f"{local_stamp()}_{uuid.uuid4().hex[:8]}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / SESSION_LOG
                self._fh = open(path, "a", encoding="utf-8", buffering=1)
                self.path, self.folder, self._runs = path, folder, 0
            except OSError:
                self._fh, self.path, self.folder = None, None, None
                return None
            extra = "".join(f" | {key}={value}" for key, value in details.items())
            self.log("START", f"{APP_NAME} {APP_VERSION} | mode={mode}{extra}")
            return self.path

    def new_run_folder(self, mode: str) -> Optional[Path]:
        """The folder of the next Generate / Verify run of this start: 01_generate, 02_verify, ... None when no start is being logged."""
        with self._lock:
            if self.folder is None: return None
            self._runs += 1
            folder = self.folder / f"{self._runs:02d}_{mode}"
            try: folder.mkdir(parents=True, exist_ok=True)
            except OSError: return None
            return folder

    def log(self, area: str, message: object) -> None:
        with self._lock:
            if self._fh is None: return
            try:
                self._fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} [{area}] {one_line(message)}\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass

    def write_raw(self, text: str) -> None:
        """A line (or several) exactly as given: the detailed log of a run, a traceback. Nothing before start()."""
        with self._lock:
            if self._fh is None: return
            try:
                self._fh.write(text if text.endswith("\n") else text + "\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass

    def config_changed(self, old: Any, new: Any) -> None:
        if not isinstance(old, dict):
            self.log("CONFIG", "configuration created")
            return
        for line in describe_changes(old, new): self.log("CONFIG", line)

    def close(self) -> None:
        with self._lock:
            if self._fh is None: return
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):
                pass
            try: self._fh.close()
            except OSError: pass
            self._fh, self.path, self.folder = None, None, None


ACTIVITY = ActivityLog()
