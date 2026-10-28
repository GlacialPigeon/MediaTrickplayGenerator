# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import contextlib
import glob
import os
import posixpath
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .localization import t
from .paths import remove_outer_quotes

_NETWORK_FILESYSTEMS = {"cifs", "smb3", "smbfs", "nfs", "nfs4"}
_AUTOFS_FILESYSTEMS = {"autofs"}                       # a trigger layer that mounts the real file system when somebody opens the folder
_SYSTEM_MOUNTS = ("/boot", "/snap", "/run", "/sys", "/proc", "/dev", "/var/lib/docker", "/var/snap")   # mounts of the system itself, not places for media
_PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
_MOUNT_CACHE_SECONDS = 5.0


@dataclass(frozen=True)
class MountInfo:
    mount_id: int
    parent_id: int
    major_minor: str
    root: str
    mount_point: str
    filesystem: str
    source: str


def unescape_mountinfo(value: str) -> str:
    """mountinfo escapes space, tab, newline and backslash as three-digit octal sequences (for example \\040)."""
    out: list[str] = []
    i = 0
    while i < len(value):
        if value[i] == "\\" and i + 3 < len(value) and value[i + 1:i + 4].isdigit():
            try:
                out.append(chr(int(value[i + 1:i + 4], 8)))
                i += 4
                continue
            except ValueError:
                pass
        out.append(value[i])
        i += 1
    return "".join(out)


def parse_mountinfo(text: str) -> list[MountInfo]:
    mounts: list[MountInfo] = []
    for line in text.splitlines():
        try:
            left, right = line.split(" - ", 1)
            lf = left.split()
            rf = right.split()
            if len(lf) < 6 or len(rf) < 2:
                continue
            mounts.append(MountInfo(
                mount_id=int(lf[0]), parent_id=int(lf[1]), major_minor=lf[2],
                root=unescape_mountinfo(lf[3]), mount_point=unescape_mountinfo(lf[4]),
                filesystem=rf[0], source=unescape_mountinfo(rf[1]),
            ))
        except Exception:
            continue
    return mounts


def find_mount(real_path: str, mounts: list[MountInfo]) -> Optional[MountInfo]:
    """Mount with the longest mount point containing the path; among equals the most recently mounted wins."""
    best: Optional[MountInfo] = None
    for mount in mounts:
        mp = mount.mount_point
        inside = real_path == mp or real_path.startswith(mp.rstrip("/") + "/") or mp == "/"
        if not inside:
            continue
        if best is None or len(mp) >= len(best.mount_point):
            best = mount
    return best


def nfs_source(source: str) -> tuple[str, str]:
    """("host", "/export") of an NFS source "host:/export". The export always starts with a slash, so the colons of an IPv6 address
    ("[fe80::1]:/export", "fe80::1:/export") are not mistaken for the separator."""
    host, sep, export = source.partition(":/")
    return (host, "/" + export) if sep else (source, "")


def network_host(mount: MountInfo) -> str:
    source = mount.source
    if mount.filesystem in {"cifs", "smb3", "smbfs"}:
        host = source.lstrip("/").split("/", 1)[0]
    else:  # nfs: host:/export
        host = nfs_source(source)[0]
    return host.casefold() or "unknown"


def network_share(mount: MountInfo) -> str:
    """The share a network mount comes from, in the same case rules everywhere (host names are not case sensitive)."""
    if mount.filesystem in {"cifs", "smb3", "smbfs"}:
        parts = mount.source.lstrip("/").split("/")
        return "//" + "/".join(parts[:2]).casefold()
    host, export = nfs_source(mount.source)
    return host.casefold() + ":" + export if export else mount.source


def filesystem_relative_path(real_path: str, mount: MountInfo) -> str:
    """The path of `real_path` inside the file system the mount shows: the folder where it is mounted plus the part of the file system
    the mount starts at. A folder bound from a subfolder of a share ("root" /Movies) is not the root of that share."""
    rel = posixpath.relpath(real_path, mount.mount_point) if real_path != mount.mount_point else ""
    rel = "" if rel == "." else rel
    root = (mount.root or "/").strip("/")
    joined = posixpath.join(root, rel) if root and rel else (root or rel)
    return posixpath.normpath(joined).strip("/") if joined else ""


class _LinuxInstanceLock:
    """flock on a file in the application data folder: released by the OS if the process exits or crashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Optional[Any] = None
        self.failure: Optional[str] = None  # "busy" = another instance runs; otherwise a short error text

    def acquire(self) -> bool:
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # O_NOFOLLOW: never follow a pre-planted symlink at the lock path.
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600)  # type: ignore[attr-defined]  # Linux-only attribute, this module is only used on Linux
        except OSError as exc:
            self.failure = f"cannot open lock file: {exc}"
            return False
        handle = os.fdopen(fd, "r+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
        except OSError:
            handle.close()
            self.failure = "busy"
            return False
        self.handle = handle
        return True

    def release(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class LinuxPlatform:
    name = "linux"

    def __init__(self, *, mountinfo_path: str = "/proc/self/mountinfo", sys_root: str = "/sys",
                 dri_dir: str = "/dev/dri", cpuinfo_path: str = "/proc/cpuinfo", meminfo_path: str = "/proc/meminfo") -> None:
        self.mountinfo_path = mountinfo_path
        self.sys_root = sys_root
        self.dri_dir = dri_dir
        self.cpuinfo_path = cpuinfo_path
        self.meminfo_path = meminfo_path
        self._mount_cache: tuple[float, list[MountInfo]] = (0.0, [])
        self._disk_cache: dict[tuple[str, str], str] = {}

    # --- console / desktop ---
    def configure_console(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    def enable_virtual_terminal(self) -> bool:
        return bool(sys.stdout.isatty()) and os.environ.get("TERM", "") not in {"", "dumb"}

    def clear_screen(self) -> None:
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

    def open_path(self, path: Path) -> None:
        opener = shutil.which("xdg-open")
        if not opener:
            print(t("platform.no_opener", path=path))
            return
        try:
            subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            print(t("platform.could_not_open", path=path, error=exc))

    def ffmpeg_setup_help(self) -> tuple[list[str], Optional[str]]:
        return (t("ffmpeg.setup_linux").split("\n"), None)

    @contextlib.contextmanager
    def prevent_sleep(self, enabled: bool) -> Iterator[None]:
        """Hold a systemd sleep inhibitor for the duration of the block when systemd-inhibit exists; otherwise a no-op."""
        proc: Optional[subprocess.Popen[bytes]] = None
        inhibit = shutil.which("systemd-inhibit") if enabled else None
        if inhibit:
            try:
                proc = subprocess.Popen(
                    [inhibit, "--what=sleep:idle", "--who=Media Trickplay Generator", "--why=Generating trickplay",
                     "--mode=block", "sleep", "2147483647"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                )
            except OSError:
                proc = None
        try:
            yield
        finally:
            if proc is not None and sys.platform != "win32":        # the process only exists on Linux; the check also keeps type checkers on Windows quiet
                # systemd-inhibit runs "sleep" for the length of the block. It has its own session, so ending the whole process group
                # also ends a `sleep` that was started as a child and would otherwise keep the inhibitor alive.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=3)
                        break
                    except subprocess.TimeoutExpired: continue
                    except (OSError, ProcessLookupError): break

    # --- process / filesystem behaviour ---
    def create_instance_lock(self, path: Path) -> _LinuxInstanceLock:
        return _LinuxInstanceLock(path)

    def ffmpeg_executable_names(self) -> tuple[str, str]:
        return "ffmpeg", "ffprobe"

    def system_ffmpeg(self) -> Optional[tuple[Path, Path]]:
        """(ffmpeg, ffprobe) the system provides (found on PATH), used when there is no build beside the program; None unless both exist."""
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        return (Path(ffmpeg), Path(ffprobe)) if ffmpeg and ffprobe else None

    def subprocess_process_options(self) -> dict[str, Any]:
        # Own session: terminal Ctrl+C stays with the parent; the second interrupt explicitly terminates FFmpeg.
        return {"start_new_session": True}

    def is_link_or_reparse_point(self, path: Path) -> bool:
        try:
            return path.is_symlink()
        except OSError:
            return False

    def is_transient_lock_error(self, exc: BaseException) -> bool:
        return False  # renaming a folder somebody has open just works on Linux; a permission error here is real

    # --- storage identity ---
    def _mounts(self) -> list[MountInfo]:
        now = time.monotonic()
        stamp, cached = self._mount_cache
        if cached and now - stamp < _MOUNT_CACHE_SECONDS:
            return cached
        try:
            text = Path(self.mountinfo_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        mounts = parse_mountinfo(text)
        self._mount_cache = (now, mounts)
        return mounts

    def _real(self, path: Path | str) -> str:
        try:
            return os.path.realpath(os.path.expanduser(str(path)))
        except (OSError, ValueError):
            return posixpath.normpath(str(path))

    def canonical_media_path(self, path: Path) -> str:
        real = self._real(path)
        mount = self._mount_for(real)
        if mount is None or mount.filesystem not in _NETWORK_FILESYSTEMS:
            return real
        rel = filesystem_relative_path(real, mount)
        if mount.filesystem in {"cifs", "smb3", "smbfs"}:
            base = "//" + mount.source.lstrip("/")
            return (base.rstrip("/") + ("/" + rel if rel else "")).casefold()
        host, export = nfs_source(mount.source)
        base = host.casefold() + ":" + export
        return base.rstrip("/") + ("/" + rel if rel else "")

    def _mount_for(self, real: str) -> Optional[MountInfo]:
        """The mount a path really lives on. An autofs layer is only the trigger: opening the folder makes the system mount the real file
        system, so the mount table is read again afterwards (not from the 5 second cache). None while only the trigger is there."""
        mount = find_mount(real, self._mounts())
        if mount is None or mount.filesystem.casefold() not in _AUTOFS_FILESYSTEMS:
            return mount
        try:
            with os.scandir(real): pass
        except OSError:
            pass
        self._mount_cache = (0.0, [])
        mount = find_mount(real, self._mounts())
        if mount is None or mount.filesystem.casefold() in _AUTOFS_FILESYSTEMS:
            return None
        return mount

    def _block_to_disk(self, name: str, depth: int = 0) -> str:
        """Block device name (sda1, nvme0n1p2, dm-0, md0) to the physical disk(s) underneath it."""
        node = os.path.realpath(os.path.join(self.sys_root, "class", "block", name))
        if not os.path.isdir(node):
            return name
        disk = name
        if os.path.exists(os.path.join(node, "partition")):
            disk = os.path.basename(os.path.dirname(node))
        if depth < 4:
            slaves_dir = os.path.join(self.sys_root, "block", disk, "slaves")
            try:
                slaves = sorted(os.listdir(slaves_dir))
            except OSError:
                slaves = []
            if slaves:
                return "+".join(sorted({self._block_to_disk(s, depth + 1) for s in slaves}))
        return disk

    def _local_disk_name(self, mount: MountInfo) -> Optional[str]:
        key = (mount.mount_point, mount.source + "|" + mount.major_minor)
        if key in self._disk_cache:
            return self._disk_cache[key] or None
        name: Optional[str] = None
        if mount.source.startswith("/dev/"):
            name = self._block_to_disk(os.path.basename(os.path.realpath(mount.source)))
        elif not mount.major_minor.startswith("0:"):
            link = os.path.join(self.sys_root, "dev", "block", mount.major_minor)
            real = os.path.realpath(link)
            if os.path.isdir(real):
                name = self._block_to_disk(os.path.basename(real))
        self._disk_cache[key] = name or ""
        return name

    def storage_group(self, path: Path) -> str:
        real = self._real(path)
        mount = self._mount_for(real)
        if mount is None:
            return "UNKNOWN"
        if mount.filesystem in _NETWORK_FILESYSTEMS:
            return f"NET_SERVER_{network_host(mount)}"
        disk = self._local_disk_name(mount)
        if disk:
            return f"PHYSICAL_DISK_{disk}"
        return f"FS_{mount.filesystem}_{mount.mount_point}"

    def storage_kind(self, path: Path) -> str:
        """"ssd" when every disk under the path is not rotational (SSD, NVMe), "hdd" for a hard disk or a network share (a share counts as
        slow), "unknown" when the disk cannot be identified (it is then treated like a hard disk, but not shown as one)."""
        real = self._real(path)
        mount = self._mount_for(real)
        if mount is None: return "unknown"
        if mount.filesystem in _NETWORK_FILESYSTEMS: return "hdd"
        disk = self._local_disk_name(mount)
        if not disk: return "unknown"
        kinds: list[str] = []
        for part in disk.split("+"):
            try: rotational = Path(self.sys_root, "block", part, "queue", "rotational").read_text().strip()
            except OSError: return "unknown"
            kinds.append("ssd" if rotational == "0" else "hdd" if rotational == "1" else "unknown")
        if "hdd" in kinds: return "hdd"
        return "unknown" if "unknown" in kinds else "ssd"

    def list_disks(self) -> list[tuple[str, str]]:
        """The disks the computer has, one entry per physical disk or network share: (where it is mounted, "ssd" / "hdd" / "network").
        System mounts, folders bound from elsewhere and virtual file systems are left out."""
        found: dict[str, tuple[str, str]] = {}
        for mount in sorted(self._mounts(), key=lambda m: (len(m.mount_point), m.mount_point)):
            point = mount.mount_point
            if mount.root != "/" or point.startswith(_SYSTEM_MOUNTS): continue
            if mount.filesystem in _NETWORK_FILESYSTEMS:
                found.setdefault("NET:" + network_share(mount), (point, "network"))
                continue
            disk = self._local_disk_name(mount)
            if disk: found.setdefault(disk, (point, self.storage_kind(Path(point))))
        return sorted(found.values())

    def mount_group(self, path: Path) -> str:
        real = self._real(path)
        mount = self._mount_for(real)
        if mount is None:
            return "UNKNOWN"
        if mount.filesystem in _NETWORK_FILESYSTEMS:
            return f"MOUNT_NET:{network_share(mount)}"
        return f"MOUNT:{mount.mount_point}"

    def resolve_media_path(self, raw: str) -> str:
        return os.path.expanduser(remove_outer_quotes(raw))   # "~" and "~/Movies" are the home folder; shortcuts are a Windows thing

    def whole_drive(self, raw: str) -> str:
        return raw                                          # a name with a colon is just a name here

    def path_is_within(self, child: str, parent: str) -> bool:
        if child == parent:
            return True
        return child.startswith(parent.rstrip("/") + "/")

    # --- hardware ---
    def _gpu_vendors(self) -> list[tuple[str, str]]:
        """(card name, vendor id) for display-class PCI devices, from sysfs."""
        found: list[tuple[str, str]] = []
        for card in sorted(glob.glob(os.path.join(self.sys_root, "class", "drm", "card[0-9]*"))):
            if "-" in os.path.basename(card):
                continue  # connector entries such as card0-HDMI-A-1
            dev = os.path.join(card, "device")
            try:
                vendor = Path(dev, "vendor").read_text().strip().lower()
                klass = Path(dev, "class").read_text().strip().lower()
            except OSError:
                continue
            if klass.startswith("0x03"):
                found.append((os.path.basename(card), vendor))
        return found

    def detect_gpu_names(self) -> list[str]:
        names: list[str] = []
        lspci = shutil.which("lspci")
        if lspci:
            try:
                out = subprocess.run([lspci, "-mm"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace", timeout=10).stdout
                for line in out.splitlines():
                    try: fields = shlex.split(line)
                    except ValueError: continue
                    if len(fields) >= 4 and any(k in fields[1] for k in ("VGA", "3D controller", "Display controller")):
                        names.append(f"{fields[2]} {fields[3]}".strip())
            except Exception:
                pass
        if names:
            return names
        return [f"{_PCI_VENDORS.get(vendor, vendor)} GPU ({card})" for card, vendor in self._gpu_vendors()]

    def system_info(self) -> dict[str, Any]:
        """Processor name, cores (distinct physical cores; threads where the kernel does not say), threads and installed memory."""
        info: dict[str, Any] = {}
        try:
            name, threads, cores = "", 0, set()
            package = core = ""
            for line in Path(self.cpuinfo_path).read_text(errors="replace").splitlines():
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "model name" and not name: name = value
                elif key == "processor": threads += 1
                elif key == "physical id": package = value
                elif key == "core id":
                    core = value
                    cores.add((package, core))
            info.update({"cpu_name": name, "threads": threads, "cores": len(cores) or threads})
        except OSError:
            pass
        try:
            for line in Path(self.meminfo_path).read_text(errors="replace").splitlines():
                if line.startswith("MemTotal:"): info["ram_bytes"] = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return info

    def detect_hardware(self) -> dict[str, bool]:
        hw = {"nvidia_gpu": False, "amd_gpu": False, "intel_gpu": False, "intel_cpu": False}
        for _card, vendor in self._gpu_vendors():
            if vendor == "0x10de": hw["nvidia_gpu"] = True
            elif vendor == "0x1002": hw["amd_gpu"] = True
            elif vendor == "0x8086": hw["intel_gpu"] = True
        if not any(hw.values()):  # no sysfs data (containers, odd kernels): fall back to lspci names
            low = " ".join(self.detect_gpu_names()).lower()
            hw["nvidia_gpu"] = "nvidia" in low
            hw["amd_gpu"] = "amd" in low or "radeon" in low
            hw["intel_gpu"] = "intel" in low
        try:
            hw["intel_cpu"] = "genuineintel" in Path(self.cpuinfo_path).read_text(errors="replace").lower()
        except OSError:
            pass
        return hw

    def render_node(self) -> Optional[str]:
        nodes = sorted(glob.glob(os.path.join(self.dri_dir, "renderD*")))
        return nodes[0] if nodes else None

    def extra_backends(self, capabilities: dict[str, Any], hardware: dict[str, bool]) -> list[str]:
        # VA-API drives Intel and AMD GPUs; NVIDIA uses the CUDA path.
        if (capabilities.get("vaapi") and capabilities.get("scale_vaapi") and self.render_node()
                and (hardware.get("intel_gpu") or hardware.get("amd_gpu"))):
            return ["vaapi"]
        return []

    def backend_filter(self, backend: str, fps: str, width: int) -> Optional[tuple[list[str], str]]:
        if backend == "vaapi":
            node = self.render_node() or "/dev/dri/renderD128"
            args = ["-hwaccel", "vaapi", "-hwaccel_device", node, "-hwaccel_output_format", "vaapi"]
            vf = f"{fps},scale_vaapi=w={width}:h=-2:format=nv12,hwdownload,format=nv12"
            return args, vf
        return None
