# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from .constants import DIRS, STATE_SCHEMA
from .media_classification import CLASSIFICATION_VERSION
from .datetime_helpers import local_stamp, utc_now_iso
from .models import TrickplaySettings, VideoMeta


class StateDB:
    def __init__(self) -> None:
        self.path = DIRS["data"] / "state.db"
        self.probe_cache_path = DIRS["data"] / "probe_cache.db"

        self.conn: Optional[sqlite3.Connection] = None
        self.probe_conn: Optional[sqlite3.Connection] = None

        self.lock = threading.RLock()

    def open(self) -> None:
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outputs (
                path TEXT NOT NULL, settings_hash TEXT NOT NULL, width INTEGER NOT NULL, interval REAL NOT NULL,
                cols INTEGER NOT NULL, rows INTEGER NOT NULL, qscale INTEGER NOT NULL, jpeg_quality INTEGER NOT NULL,
                source_size INTEGER NOT NULL, source_mtime_ns INTEGER NOT NULL, source_duration REAL, source_fingerprint TEXT,
                frames INTEGER NOT NULL, tiles INTEGER NOT NULL, output_bytes INTEGER NOT NULL,
                method TEXT NOT NULL, backend TEXT, completed_at TEXT NOT NULL,
                PRIMARY KEY(path, settings_hash)
            );
            CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, mode TEXT NOT NULL, summary_json TEXT);
            CREATE TABLE IF NOT EXISTS backend_failures (
                backend TEXT NOT NULL, codec TEXT NOT NULL, profile TEXT NOT NULL, pix_fmt TEXT NOT NULL,
                failure_class TEXT NOT NULL, error_excerpt TEXT, last_seen TEXT NOT NULL,
                PRIMARY KEY(backend, codec, profile, pix_fmt, failure_class)
            );
            """)
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)", (str(STATE_SCHEMA),))
            # Older versions took a file's colour space tag that FFmpeg refuses for a GPU that does not support the whole codec.
            self.conn.execute("DELETE FROM backend_failures WHERE error_excerpt LIKE '%Color spaces:%'")
            self.probe_conn = sqlite3.connect(
                self.probe_cache_path,
                timeout=30,
                check_same_thread=False,
            )
            self.probe_conn.row_factory = sqlite3.Row

            with self.probe_conn:
                self.probe_conn.execute("PRAGMA journal_mode=WAL")
                self.probe_conn.execute("PRAGMA synchronous=NORMAL")
                self.probe_conn.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    duration REAL,
                    codec TEXT,
                    profile TEXT,
                    pix_fmt TEXT,
                    width INTEGER,
                    height INTEGER,
                    hdr INTEGER NOT NULL DEFAULT 0,
                    fingerprint TEXT,
                    last_seen TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_videos_last_seen
                    ON videos(last_seen);
                """)
                self._migrate_probe_cache()

    # columns added after the first version of the probe cache: name -> definition
    _PROBE_COLUMNS = {"bits": "INTEGER", "chroma": "TEXT", "resolution": "TEXT", "hdr_pool": "TEXT", "hdr_type": "TEXT",
                      "dv_profile": "TEXT", "hdr_note": "TEXT", "class_version": "INTEGER NOT NULL DEFAULT 0"}

    def _migrate_probe_cache(self) -> None:
        """Add the picture-format columns to an older probe cache. Old rows keep class_version 0, so those videos are
        probed again once (the cache only holds what ffprobe says; nothing about outputs is touched)."""
        if self.probe_conn is None: return
        have = {row["name"] for row in self.probe_conn.execute("PRAGMA table_info(videos)")}
        for name, definition in self._PROBE_COLUMNS.items():
            if name not in have: self.probe_conn.execute(f"ALTER TABLE videos ADD COLUMN {name} {definition}")

    def close(self) -> None:
        if self.probe_conn is not None:
            try:
                self.probe_conn.commit()
                self.probe_conn.close()
            finally:
                self.probe_conn = None

        if self.conn is not None:
            try:
                self.conn.commit()
                self.conn.close()
            finally:
                self.conn = None

    def reset_with_backup(self) -> Path:
        self.close()
        backup = self.path.with_name(f"state.pre_reset.{local_stamp()}.db")
        if self.path.exists(): shutil.move(str(self.path), str(backup))
        for suffix in ("-wal", "-shm"):
            p = Path(str(self.path) + suffix)
            p.unlink(missing_ok=True)
        self.open()
        return backup

    def recover_corrupt(self) -> Path:
        self.close()
        backup = self.path.with_name(f"state.corrupt.{local_stamp()}.db")
        if self.path.exists(): shutil.move(str(self.path), str(backup))
        for suffix in ("-wal", "-shm"):
            p = Path(str(self.path) + suffix)
            p.unlink(missing_ok=True)
        self.open()
        return backup

    def _execute(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if self.conn is None: raise RuntimeError("state database not open")
        return self.conn.execute(sql, args)

    def _connection(self) -> sqlite3.Connection:
        if self.conn is None: raise RuntimeError("state database not open")
        return self.conn

    def _probe_connection(self) -> sqlite3.Connection:
        if self.probe_conn is None: raise RuntimeError("probe cache database not open")
        return self.probe_conn

    def _probe_execute(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if self.probe_conn is None:
            raise RuntimeError("probe cache database not open")
        return self.probe_conn.execute(sql, args)

    def begin_run(self, run_id: str, mode: str) -> None:
        with self.lock, self._connection():
            self._execute("INSERT OR REPLACE INTO runs(run_id, started_at, mode) VALUES(?,?,?)", (run_id, utc_now_iso(), mode))

    def end_run(self, run_id: str, summary: dict[str, Any]) -> None:
        with self.lock, self._connection():
            self._execute("UPDATE runs SET ended_at=?, summary_json=? WHERE run_id=?", (utc_now_iso(), json.dumps(summary, ensure_ascii=False), run_id))

    def upsert_video(self, meta: "VideoMeta", fingerprint: Optional[str] = None) -> None:
        with self.lock, self._probe_connection():
            self._probe_execute("""
            INSERT INTO videos(
                path, size, mtime_ns, duration, codec, profile, pix_fmt,
                width, height, hdr, fingerprint, last_seen,
                bits, chroma, resolution, hdr_pool, hdr_type, dv_profile, hdr_note, class_version
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                size=excluded.size,
                mtime_ns=excluded.mtime_ns,
                duration=excluded.duration,
                codec=excluded.codec,
                profile=excluded.profile,
                pix_fmt=excluded.pix_fmt,
                width=excluded.width,
                height=excluded.height,
                hdr=excluded.hdr,
                bits=excluded.bits,
                chroma=excluded.chroma,
                resolution=excluded.resolution,
                hdr_pool=excluded.hdr_pool,
                hdr_type=excluded.hdr_type,
                dv_profile=excluded.dv_profile,
                hdr_note=excluded.hdr_note,
                class_version=excluded.class_version,
                fingerprint=COALESCE(excluded.fingerprint, videos.fingerprint),
                last_seen=excluded.last_seen
            """, (
                str(meta.path),
                meta.size,
                meta.mtime_ns,
                meta.duration,
                meta.codec,
                meta.profile,
                meta.pix_fmt,
                meta.width,
                meta.height,
                int(meta.hdr),
                fingerprint,
                utc_now_iso(),
                meta.bits,
                meta.chroma,
                meta.resolution,
                meta.hdr_pool,
                meta.hdr_type,
                meta.dv_profile,
                meta.hdr_note,
                CLASSIFICATION_VERSION,
            ))

    def last_probed(self, path: Path) -> Optional[str]:
        """When the program last probed this file (UTC text, sorts chronologically), or None if it never did."""
        with self.lock:
            row = self._probe_execute("SELECT last_seen FROM videos WHERE path=?", (str(path),)).fetchone()
            return str(row["last_seen"]) if row else None

    def get_meta(self, key: str) -> Optional[str]:
        with self.lock:
            row = self._execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return str(row["value"]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.lock, self._connection():
            self._execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    def cached_video(self, path: Path, size: int, mtime_ns: int) -> Optional["VideoMeta"]:
        with self.lock:
            row = self._probe_execute("SELECT * FROM videos WHERE path=?", (str(path),)).fetchone()
            if not row or row["size"] != size or row["mtime_ns"] != mtime_ns: return None
            if int(row["class_version"] or 0) < CLASSIFICATION_VERSION: return None     # classified by older rules: probe again
            return VideoMeta(path=path, size=row["size"], mtime_ns=row["mtime_ns"], duration=float(row["duration"] or 0),
                             codec=row["codec"] or "unknown", profile=row["profile"] or "", pix_fmt=row["pix_fmt"] or "",
                             width=int(row["width"] or 0), height=int(row["height"] or 0), hdr=bool(row["hdr"]),
                             bits=int(row["bits"] or 0), chroma=row["chroma"] or "", resolution=row["resolution"] or "",
                             hdr_pool=row["hdr_pool"] or "SDR", hdr_type=row["hdr_type"] or "", dv_profile=row["dv_profile"] or "",
                             hdr_note=row["hdr_note"] or "")

    def previous_output(self, path: Path, settings_hash: str) -> Optional[sqlite3.Row]:
        with self.lock: return self._execute("SELECT * FROM outputs WHERE path=? AND settings_hash=?", (str(path), settings_hash)).fetchone()

    def current_target_output(self, path: Path, settings: "TrickplaySettings") -> Optional[sqlite3.Row]:
        with self.lock:
            return self._execute(
                "SELECT * FROM outputs WHERE path=? AND width=? AND cols=? AND rows=? ORDER BY completed_at DESC, rowid DESC LIMIT 1",
                (str(path), settings.width, settings.cols, settings.rows),
            ).fetchone()

    def record_output(self, meta: "VideoMeta", settings: "TrickplaySettings", frames: int, tiles: int, output_bytes: int, method: str, backend: Optional[str], source_fingerprint: Optional[str]) -> None:
        with self.lock, self._connection():
            self._execute("DELETE FROM outputs WHERE path=? AND width=? AND cols=? AND rows=?",
                          (str(meta.path), settings.width, settings.cols, settings.rows))
            self._execute("""
            INSERT OR REPLACE INTO outputs(path,settings_hash,width,interval,cols,rows,qscale,jpeg_quality,
            source_size,source_mtime_ns,source_duration,source_fingerprint,frames,tiles,output_bytes,method,backend,completed_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (str(meta.path), settings.hash, settings.width, settings.interval, settings.cols, settings.rows,
                  settings.qscale, settings.jpeg_quality, meta.size, meta.mtime_ns, meta.duration, source_fingerprint,
                  frames, tiles, output_bytes, method, backend, utc_now_iso()))

    def estimate_bytes_per_second(self, settings_hash: str) -> Optional[float]:
        with self.lock:
            row = self._execute("SELECT SUM(output_bytes) AS b, SUM(source_duration) AS d FROM outputs WHERE settings_hash=? AND source_duration > 0", (settings_hash,)).fetchone()
            if row and row["b"] and row["d"] and row["d"] > 0: return float(row["b"]) / float(row["d"])
            return None

    def known_unsupported(self, backend: str, meta: "VideoMeta") -> bool:
        with self.lock:
            row = self._execute("SELECT 1 FROM backend_failures WHERE backend=? AND codec=? AND profile=? AND pix_fmt=? AND failure_class='unsupported' LIMIT 1",
                                (backend, meta.codec, meta.profile or "", meta.pix_fmt or "")).fetchone()
            return bool(row)

    def record_backend_failure(self, backend: str, meta: "VideoMeta", failure_class: str, error: str) -> None:
        excerpt = error[-2000:]
        with self.lock, self._connection():
            self._execute("""
            INSERT OR REPLACE INTO backend_failures(backend,codec,profile,pix_fmt,failure_class,error_excerpt,last_seen) VALUES(?,?,?,?,?,?,?)
            """, (backend, meta.codec, meta.profile or "", meta.pix_fmt or "", failure_class, excerpt, utc_now_iso()))
