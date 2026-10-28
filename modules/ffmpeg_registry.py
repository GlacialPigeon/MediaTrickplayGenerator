# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import constants
from . import platform_api
from .constants import DIRS
from .datetime_helpers import local_stamp, utc_now_iso
from .file_io import atomic_write_json, safe_json_load
from .fingerprint import sha256_file
from .paths import normalize_path
from .localization import t
from .prompts import choose


@dataclass
class FFmpegCandidate:
    root: Path
    ffmpeg: Path
    ffprobe: Path
    source: str
    folder_name: str
    folder_date: Optional[str] = None
    sha256: Optional[str] = None
    ffprobe_sha256: Optional[str] = None
    version_line: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    validation_error: str = ""
    cached: bool = False

class FFmpegRegistry:
    def __init__(self) -> None:
        self.path = DIRS["data"] / "ffmpeg_registry.json"
        self.data: dict[str, Any] = {"schema_version": 1, "selected_hash": None, "installations": []}
        self.recovery_note: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists(): return
        try:
            raw = safe_json_load(self.path)
            if isinstance(raw, dict): self.data = {**self.data, **raw}
        except Exception as exc:
            broken = self.path.with_name(f"ffmpeg_registry.broken.{local_stamp()}.json")
            try:
                shutil.move(str(self.path), str(broken))
                self.recovery_note = f"FFmpeg registry was invalid and was archived: {exc}"
            except Exception: self.recovery_note = f"FFmpeg registry is invalid: {exc}"

    def save(self) -> None: atomic_write_json(self.path, self.data)

    def forget_with_backup(self) -> Optional[Path]:
        if not self.path.exists():
            self.data = {"schema_version": 1, "selected_hash": None, "installations": []}
            return None
        backup = self.path.with_name(f"ffmpeg_registry.pre_reset.{local_stamp()}.json")
        shutil.move(str(self.path), str(backup))
        self.data = {"schema_version": 1, "selected_hash": None, "installations": []}
        self.save()
        return backup

    def _candidate_from_folder(self, folder: Path, source: str) -> Optional[FFmpegCandidate]:
        possibilities = [folder, folder / "bin"]
        ffmpeg_name, ffprobe_name = platform_api.ffmpeg_executable_names()
        for base in possibilities:
            ffmpeg = base / ffmpeg_name
            ffprobe = base / ffprobe_name
            if ffmpeg.is_file() and ffprobe.is_file():
                m = re.search(r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?!\d)", folder.name)
                folder_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
                return FFmpegCandidate(root=folder, ffmpeg=ffmpeg, ffprobe=ffprobe, source=source, folder_name=folder.name, folder_date=folder_date)
        return None

    @staticmethod
    def ffmpeg_folder() -> Path:
        """The `ffmpeg` folder beside main.py: the place FFmpeg is looked for first (on Windows the only one)."""
        return constants.APP_ROOT / constants.FFMPEG_DIR_NAME

    def discover(self, config: dict[str, Any]) -> list[FFmpegCandidate]:
        """Builds inside `ffmpeg/`: `ffmpeg/bin/<exe>`, `ffmpeg/<exe>`, or one build per sub-folder (`ffmpeg/<build>/bin/<exe>`).

        On Linux the FFmpeg of the system (`apt install ffmpeg`, `dnf install ffmpeg-free`) is one more candidate, and it is used only when
        there is no build beside the program. User-configured locations are ignored so the same, known FFmpeg is always used.
        """
        candidates: list[FFmpegCandidate] = []
        seen: set[str] = set()
        base = self.ffmpeg_folder()
        folders = [base]
        try:
            if base.is_dir():
                folders += sorted((child for child in base.iterdir() if child.is_dir() and child.name.casefold() != "bin"), key=lambda p: p.name.casefold())
        except OSError: pass
        for folder in folders:
            cand = self._candidate_from_folder(folder, "portable")
            if cand and normalize_path(cand.ffmpeg) not in seen:
                seen.add(normalize_path(cand.ffmpeg))
                candidates.append(cand)
        system = platform_api.system_ffmpeg()
        if system and system[0].is_file() and system[1].is_file() and normalize_path(system[0]) not in seen:
            candidates.append(FFmpegCandidate(root=system[0].parent, ffmpeg=system[0], ffprobe=system[1], source="system", folder_name="system"))
        return candidates

    def _run_text(self, args: list[str], timeout: int = 20) -> tuple[int, str]:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return proc.returncode, proc.stdout

    def hydrate_cached(self, candidate: FFmpegCandidate) -> bool:
        try:
            ffst = candidate.ffmpeg.stat()
            fpst = candidate.ffprobe.stat()
        except OSError: return False
        for record in self.data.get("installations", []):
            try:
                if normalize_path(record.get("ffmpeg_path", "")) != normalize_path(candidate.ffmpeg): continue
                if normalize_path(record.get("ffprobe_path", "")) != normalize_path(candidate.ffprobe): continue
                if int(record.get("ffmpeg_size", -1)) != ffst.st_size or int(record.get("ffmpeg_mtime_ns", -1)) != ffst.st_mtime_ns: continue
                if int(record.get("ffprobe_size", -1)) != fpst.st_size or int(record.get("ffprobe_mtime_ns", -1)) != fpst.st_mtime_ns: continue
                if record.get("last_validation") != "PASS": continue
                candidate.sha256 = record.get("ffmpeg_sha256")
                candidate.ffprobe_sha256 = record.get("ffprobe_sha256")
                candidate.version_line = record.get("reported_version", "")
                candidate.folder_date = candidate.folder_date or record.get("folder_date")
                candidate.capabilities = record.get("capabilities", {}) or {}
                candidate.valid = True
                candidate.cached = True
                record["path"] = str(candidate.root)
                record["last_seen"] = utc_now_iso()
                self.save()
                return True
            except Exception: continue
        return False

    def validate(self, candidate: FFmpegCandidate) -> FFmpegCandidate:
        try:
            ff0 = candidate.ffmpeg.stat()
            fp0 = candidate.ffprobe.stat()
            if ff0.st_size == 0 or fp0.st_size == 0: raise RuntimeError("zero-byte executable")
            time.sleep(0.25)
            ff1 = candidate.ffmpeg.stat()
            fp1 = candidate.ffprobe.stat()
            if (ff0.st_size, ff0.st_mtime_ns, fp0.st_size, fp0.st_mtime_ns) != (ff1.st_size, ff1.st_mtime_ns, fp1.st_size, fp1.st_mtime_ns):
                raise RuntimeError("FFmpeg files are still changing")

            rc, ffv = self._run_text([str(candidate.ffmpeg), "-hide_banner", "-version"])
            if rc != 0: raise RuntimeError(f"ffmpeg -version failed ({rc})")
            rc, fpv = self._run_text([str(candidate.ffprobe), "-hide_banner", "-version"])
            if rc != 0: raise RuntimeError(f"ffprobe -version failed ({rc})")

            candidate.version_line = (ffv.splitlines() or ["unknown"])[0].strip()
            if not candidate.folder_date:
                vm = re.search(r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?!\d)", candidate.version_line)
                if vm: candidate.folder_date = f"{vm.group(1)}-{vm.group(2)}-{vm.group(3)}"

            _, hw = self._run_text([str(candidate.ffmpeg), "-hide_banner", "-hwaccels"])
            _, filters = self._run_text([str(candidate.ffmpeg), "-hide_banner", "-filters"])
            candidate.capabilities = {
                "cuda": bool(re.search(r"(?mi)^\s*cuda\s*$", hw)),
                "qsv": bool(re.search(r"(?mi)^\s*qsv\s*$", hw)),
                "d3d11va": bool(re.search(r"(?mi)^\s*d3d11va\s*$", hw)),
                "dxva2": bool(re.search(r"(?mi)^\s*dxva2\s*$", hw)),
                "vaapi": bool(re.search(r"(?mi)^\s*vaapi\s*$", hw)),
                "scale_cuda": "scale_cuda" in filters,
                "scale_qsv": "scale_qsv" in filters,
                "scale_vaapi": "scale_vaapi" in filters,
                "hwaccels_raw": hw.strip(),
            }
            candidate.sha256 = sha256_file(candidate.ffmpeg)
            candidate.ffprobe_sha256 = sha256_file(candidate.ffprobe)
            candidate.valid = True
            self._remember(candidate)
        except Exception as exc:
            candidate.valid = False
            candidate.validation_error = str(exc)
        return candidate

    def _remember(self, candidate: FFmpegCandidate) -> None:
        now = utc_now_iso()
        records = self.data.setdefault("installations", [])
        record = None
        for item in records:
            if item.get("ffmpeg_sha256") == candidate.sha256:
                record = item
                break
        ffst = candidate.ffmpeg.stat()
        fpst = candidate.ffprobe.stat()
        payload = {
            "path": str(candidate.root), "ffmpeg_path": str(candidate.ffmpeg), "ffprobe_path": str(candidate.ffprobe),
            "ffmpeg_size": ffst.st_size, "ffmpeg_mtime_ns": ffst.st_mtime_ns, "ffprobe_size": fpst.st_size, "ffprobe_mtime_ns": fpst.st_mtime_ns,
            "source": candidate.source, "folder_name": candidate.folder_name, "folder_date": candidate.folder_date,
            "ffmpeg_sha256": candidate.sha256, "ffprobe_sha256": candidate.ffprobe_sha256, "reported_version": candidate.version_line,
            "last_validation": "PASS" if candidate.valid else "FAIL", "last_seen": now, "capabilities": candidate.capabilities,
        }
        if record is None:
            payload["first_seen"] = now
            records.append(payload)
        else:
            first_seen = record.get("first_seen", now)
            record.clear()
            record.update(payload)
            record["first_seen"] = first_seen
        self.save()

    def choose(self, candidates: list[FFmpegCandidate], interactive: bool = True) -> Optional[FFmpegCandidate]:
        valid = [c for c in candidates if c.valid]
        if not valid: return None
        portable = [c for c in valid if c.source.startswith("portable")]
        selected_hash = self.data.get("selected_hash")
        dated = [c for c in portable if c.folder_date]
        if dated:
            dated.sort(key=lambda c: c.folder_date or "", reverse=True)
            chosen = dated[0]
            self.data["selected_hash"] = chosen.sha256
            self.save()
            return chosen
        if selected_hash:
            for c in (portable or valid):      # a build beside the program always wins over the one of the system
                if c.sha256 == selected_hash: return c
        if len(portable) == 1:
            chosen = portable[0]
            self.data["selected_hash"] = chosen.sha256
            self.save()
            return chosen
        if len(portable) > 1 and interactive:
            print("\n" + t("ffmpeg.several"))
            picked = choose([(str(i), f"{c.folder_name}  |  {c.version_line}") for i, c in enumerate(portable, 1)], "1")
            chosen = portable[int(picked) - 1]
            self.data["selected_hash"] = chosen.sha256
            self.save()
            return chosen
        chosen = valid[0]
        self.data["selected_hash"] = chosen.sha256
        self.save()
        return chosen
