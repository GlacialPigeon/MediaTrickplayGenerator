# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import constants, platform_api  # noqa: E402
from modules.platform_linux import LinuxPlatform  # noqa: E402

IS_WINDOWS = sys.platform == "win32"


def _find_ffmpeg() -> tuple[Optional[str], Optional[str]]:
    """Same rule as the program: the `ffmpeg` folder beside main.py (`ffmpeg/bin` or `ffmpeg/<build>/bin`), on Linux then the system's FFmpeg."""
    names = ("ffmpeg.exe", "ffprobe.exe") if IS_WINDOWS else ("ffmpeg", "ffprobe")
    base = ROOT / constants.FFMPEG_DIR_NAME
    folders = [base, base / "bin"]
    try:
        for build in sorted((c for c in base.iterdir() if c.is_dir() and c.name.casefold() != "bin"), key=lambda p: p.name, reverse=True):
            folders += [build, build / "bin"]
    except OSError:
        pass
    for folder in folders:
        if (folder / names[0]).is_file() and (folder / names[1]).is_file():
            return str(folder / names[0]), str(folder / names[1])
    if not IS_WINDOWS:
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if ffmpeg and ffprobe:
            return ffmpeg, ffprobe
    return None, None


FFMPEG, FFPROBE = _find_ffmpeg()
HAVE_FFMPEG = bool(FFMPEG and FFPROBE)
try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False

needs_ffmpeg = unittest.skipUnless(
    HAVE_FFMPEG and HAVE_PIL,
    "FFmpeg, FFprobe and Pillow are required (put it in ffmpeg/bin beside main.py)")


def can_symlink() -> bool:
    """Windows only allows symlinks with Developer Mode or an elevated shell."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            (Path(tmp) / "link").symlink_to(Path(tmp), target_is_directory=True)
            return True
        except (OSError, NotImplementedError):
            return False


needs_symlinks = unittest.skipUnless(can_symlink(), "creating symbolic links is not permitted here")
needs_linux = unittest.skipUnless(sys.platform.startswith("linux"), "Linux only")
needs_windows = unittest.skipUnless(IS_WINDOWS, "Windows only")


def host_adapter():
    """The adapter main.py would pick on this machine."""
    if IS_WINDOWS:
        from modules.platform_windows import WindowsPlatform
        return WindowsPlatform()
    return LinuxPlatform()


class IsolatedApp(unittest.TestCase):
    """Redirects every application folder into a temporary directory and installs this machine's platform adapter."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="trickplay_test_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self._saved_root = constants.APP_ROOT
        self._saved_dirs = dict(constants.DIRS)
        self._saved_platform = platform_api._PLATFORM
        constants.APP_ROOT = self.root / "app"
        for name in constants.DIRS:
            constants.DIRS[name] = constants.APP_ROOT / name
            constants.DIRS[name].mkdir(parents=True, exist_ok=True)
        platform_api.configure_platform(host_adapter())
        self.addCleanup(self._restore)
        self.media = self.root / "media"
        self.media.mkdir()

    def _restore(self) -> None:
        constants.APP_ROOT = self._saved_root
        constants.DIRS.clear()
        constants.DIRS.update(self._saved_dirs)
        platform_api._PLATFORM = self._saved_platform


def make_video(path: Path, seconds: float = 25, size: str = "320x180", rate: int = 10, codec: str = "libx264", gop: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG or "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc2=size={size}:rate={rate}", "-t", str(seconds), "-c:v", codec,
           "-pix_fmt", "yuv420p", "-preset", "ultrafast", *(["-g", str(gop)] if gop else []), "-y", str(path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return path


def build_tools():
    """FFmpegTools bound to the FFmpeg found on PATH."""
    from modules.ffmpeg_registry import FFmpegCandidate
    from modules.ffmpeg_tools import FFmpegTools
    cand = FFmpegCandidate(root=Path(FFMPEG).parent, ffmpeg=Path(FFMPEG), ffprobe=Path(FFPROBE), source="PATH",
                           folder_name="PATH", capabilities={}, valid=True, version_line="test")
    cand.sha256 = "test-hash"
    return FFmpegTools(cand)


def config_with(**overrides):
    import copy
    cfg = copy.deepcopy(constants.DEFAULT_CONFIG)
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        cfg[section][key] = value
    return cfg


def new_logger(mode: str = "test"):
    from modules.logger import RunLogger
    return RunLogger(mode)


def open_state():
    from modules.state_db import StateDB
    state = StateDB()
    state.open()
    return state
