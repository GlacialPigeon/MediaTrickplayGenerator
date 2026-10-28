# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .constants import DIRS


def ensure_directories() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def fsync_directory(directory: Path) -> None:
    """Make a rename inside `directory` durable (POSIX). Best effort: no-op where directories
    cannot be opened (Windows) or the filesystem rejects fsync on them (some network shares)."""
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        return
    try:
        fd = os.open(directory, os.O_RDONLY | flag)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write to a temporary file in the same folder, flush it to disk and swap it in; the temporary file never outlives a failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".trickplay-write-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try: os.chmod(temp_name, os.stat(path).st_mode & 0o777)   # keep the permissions of the file being replaced
        except OSError: pass
        os.replace(temp_name, path)
        fsync_directory(path.parent)
    finally:
        try: os.unlink(temp_name)
        except FileNotFoundError: pass


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def safe_json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
