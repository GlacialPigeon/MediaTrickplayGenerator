# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import shutil
from typing import Any, Optional

from .activity_log import ACTIVITY
from .constants import CONFIG_SCHEMA, DEFAULT_CONFIG, DIRS
from .datetime_helpers import local_stamp
from .file_io import atomic_write_json, safe_json_load


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            over = overlay.get(key, {}) if isinstance(overlay.get(key, {}), dict) else {}
            out[key] = deep_merge(value, over)
        else:
            out[key] = overlay.get(key, value)
    for key, value in overlay.items():
        if key not in out: out[key] = value
    return out

class ConfigManager:
    def __init__(self) -> None:
        self.path = DIRS["config"] / "config.json"
        self.backup_path = DIRS["config"] / "config.backup.json"
        self.config: dict[str, Any] = deep_merge(DEFAULT_CONFIG, {})
        self.corrupt_error: Optional[str] = None
        self.loaded_from_backup = False
        self.warnings: list[str] = []
        self.fresh = not self.path.exists()               # no configuration yet: this is the first start

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING: {message}")

    def _migrate(self, raw: dict[str, Any]) -> dict[str, Any]:
        try: version = int(raw.get("schema_version", 1))
        except Exception: version = 1
        if version > CONFIG_SCHEMA: raise ValueError(f"config schema {version} is newer than supported")
        raw["schema_version"] = CONFIG_SCHEMA
        self._drop_retired_settings(raw)
        return raw

    @staticmethod
    def _drop_retired_settings(raw: dict[str, Any]) -> None:
        """Settings of removed features: dropped, and removed choices fall back to the safe default ("ask" and "skip any" were never usable unattended)."""
        workers, acceleration, trickplay = raw.get("workers"), raw.get("acceleration"), raw.get("trickplay")
        if isinstance(workers, dict): workers.pop("probe_concurrency", None)
        if isinstance(acceleration, dict): acceleration.pop("hdr_gpu_disabled", None)
        if isinstance(trickplay, dict):
            if trickplay.get("same_width_policy") not in (None, "skip_valid", "rebuild_all"): trickplay["same_width_policy"] = "skip_valid"
            if trickplay.get("other_width_policy") not in (None, "keep_add", "replace"): trickplay["other_width_policy"] = "keep_add"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.config = deep_merge(DEFAULT_CONFIG, {})
            return self.config
        try:
            raw = safe_json_load(self.path)
            if not isinstance(raw, dict): raise ValueError("config root must be an object")
            raw = self._migrate(raw)
            self.config = deep_merge(DEFAULT_CONFIG, raw)
            return self.config
        except Exception as exc:
            self.corrupt_error = str(exc)
            if self.backup_path.exists():
                try:
                    raw = safe_json_load(self.backup_path)
                    if isinstance(raw, dict):
                        raw = self._migrate(raw)
                        self.config = deep_merge(DEFAULT_CONFIG, raw)
                        self.loaded_from_backup = True
                        return self.config
                except Exception as backup_exc: self._warn(f"The configuration backup could not be read either: {backup_exc}")
            self.config = deep_merge(DEFAULT_CONFIG, {})
            return self.config

    def save(self) -> None:
        self.config["schema_version"] = CONFIG_SCHEMA
        existing: Any = None
        if self.path.exists():
            try:
                existing = safe_json_load(self.path)
                if isinstance(existing, dict): atomic_write_json(self.backup_path, existing)
            except Exception as exc: self._warn(f"The configuration backup could not be refreshed: {exc}")
        atomic_write_json(self.path, self.config)
        ACTIVITY.config_changed(existing, self.config)

    def restore_backup(self) -> bool:
        if not self.backup_path.exists(): return False
        try:
            raw = safe_json_load(self.backup_path)
            if not isinstance(raw, dict): return False
            self._drop_retired_settings(raw)
            broken = self.path.with_name(f"config.json.broken.{local_stamp()}")
            if self.path.exists(): shutil.move(str(self.path), str(broken))
            atomic_write_json(self.path, raw)
            self.config = deep_merge(DEFAULT_CONFIG, raw)
            self.corrupt_error = None
            return True
        except Exception: return False

    def start_fresh(self) -> None:
        if self.path.exists(): shutil.move(str(self.path), str(self.path.with_name(f"config.json.broken.{local_stamp()}")))
        self.config = deep_merge(DEFAULT_CONFIG, {})
        self.save()
        self.corrupt_error = None
