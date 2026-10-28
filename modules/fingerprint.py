# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


def source_edge_fingerprint(path: Path, chunk_mib: int = 1) -> str:
    size = path.stat().st_size
    chunk = max(64 * 1024, chunk_mib * 1024 * 1024)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(0, size - chunk))
            h.update(fh.read(chunk))
    return h.hexdigest()
