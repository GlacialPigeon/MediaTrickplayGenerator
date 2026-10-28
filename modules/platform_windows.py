# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import contextlib
import json
import ntpath
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

from .localization import t
from .paths import remove_outer_quotes


class _WindowsInstanceLock:
    """msvcrt byte-range lock: released by the OS if the process exits or crashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Optional[Any] = None
        self.failure: Optional[str] = None  # "busy" = another instance runs; otherwise a short error text

    def acquire(self) -> bool:
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]  # Windows only
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


def disk_kind(media_type: str, bus_type: str) -> str:
    """What Get-PhysicalDisk says about a disk -> "ssd", "hdd" or "unknown". Only what Windows says clearly is believed; a disk that reports
    "Unspecified" (many USB, RAID and virtual disks do) stays unknown, and unknown disks are handled like hard disks (the slower, safer choice)."""
    media = media_type.strip().casefold()
    if media in {"ssd", "scm"} or bus_type.strip().casefold() == "nvme": return "ssd"
    if media == "hdd": return "hdd"
    return "unknown"


def run_powershell(executable: str, script: str, timeout: float) -> str:
    """Output of a PowerShell script as text. PowerShell writes in the console's OEM code page unless told otherwise, which turns names
    with č, š, ž (or any non-ASCII letter) into garbage, so the output is switched to UTF-8 and decoded as UTF-8."""
    prefix = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8; "
    proc = subprocess.run([executable, "-NoProfile", "-Command", prefix + script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
    return proc.stdout.decode("utf-8-sig", errors="replace")


class WindowsStorageResolver:
    """Resolve Windows paths to stable local-disk or UNC-server storage identities."""

    def __init__(self) -> None:
        self._loaded = False
        self._mappings: list[tuple[str, str]] = []
        self._access_paths: list[tuple[str, int]] = []
        self._disk_kinds: dict[int, str] = {}           # disk number -> "ssd", "hdd" or "unknown"

    @staticmethod
    def _norm_prefix(value: str) -> str:
        value = ntpath.normpath(value).replace("/", "\\")
        return value.rstrip("\\").casefold()

    def _load(self) -> None:
        if self._loaded: return
        self._loaded = True
        ps = shutil.which("powershell.exe") or shutil.which("powershell")
        if not ps: return
        # Raw string: PowerShell treats backslashes literally. The -notlike pattern is \\?\Volume{* (volume-GUID paths are skipped).
        script = r"""
$ErrorActionPreference='SilentlyContinue'
$parts = @()
Get-Partition | ForEach-Object {
  $dn = $_.DiskNumber
  foreach ($a in @($_.AccessPaths)) {
    if ($a -and $a -notlike '\\?\Volume{*') { $parts += [pscustomobject]@{ AccessPath=$a; DiskNumber=$dn } }
  }
  if ($_.DriveLetter) { $parts += [pscustomobject]@{ AccessPath=(("{0}:\" -f $_.DriveLetter)); DiskNumber=$dn } }
}
$maps = @()
Get-SmbMapping | ForEach-Object {
  if ($_.LocalPath -and $_.RemotePath) { $maps += [pscustomobject]@{ LocalPath=$_.LocalPath; RemotePath=$_.RemotePath } }
}
$disks = @()
Get-PhysicalDisk | ForEach-Object {
  $disks += [pscustomobject]@{ DeviceId=[string]$_.DeviceId; MediaType=[string]$_.MediaType; BusType=[string]$_.BusType }
}
[pscustomobject]@{ Partitions=$parts; Mappings=$maps; Disks=$disks } | ConvertTo-Json -Depth 5 -Compress
"""
        try:
            output = run_powershell(ps, script, 12)
            if not output.strip(): return
            self.ingest(json.loads(output))
        except Exception: pass

    def ingest(self, data: dict[str, Any]) -> None:
        """Load the partition / SMB-mapping tables (the PowerShell JSON). Separate from _load so it can be tested."""
        parts = data.get("Partitions") or []
        maps = data.get("Mappings") or []
        disks = data.get("Disks") or []
        if isinstance(parts, dict): parts = [parts]
        if isinstance(maps, dict): maps = [maps]
        if isinstance(disks, dict): disks = [disks]
        for row in disks:
            try: number = int(row.get("DeviceId"))
            except (TypeError, ValueError): continue
            self._disk_kinds[number] = disk_kind(str(row.get("MediaType") or ""), str(row.get("BusType") or ""))
        for row in parts:
            ap = str(row.get("AccessPath") or "").strip()
            try: dn = int(row.get("DiskNumber"))
            except (TypeError, ValueError): continue
            if ap: self._access_paths.append((self._norm_prefix(ap), dn))
        for row in maps:
            local = str(row.get("LocalPath") or "").strip()
            remote = str(row.get("RemotePath") or "").strip()
            if local and remote: self._mappings.append((self._norm_prefix(local), remote.rstrip("\\/")))
        self._access_paths.sort(key=lambda x: len(x[0]), reverse=True)
        self._mappings.sort(key=lambda x: len(x[0]), reverse=True)
        self._loaded = True

    def canonical_path(self, path: Path | str) -> str:
        raw = ntpath.normpath(str(path)).replace("/", "\\")
        if raw.startswith("\\\\"): return raw
        self._load()
        n = self._norm_prefix(raw)
        for local, remote in self._mappings:
            if n == local or n.startswith(local + "\\"):
                suffix = raw[len(local):].lstrip("\\/")
                return remote + (("\\" + suffix) if suffix else "")
        return raw

    def group(self, path: Path | str) -> str:
        canonical = self.canonical_path(path)
        if canonical.startswith("\\\\"):
            parts = canonical.strip("\\").split("\\")
            server = parts[0].casefold() if parts and parts[0] else "unknown"
            return f"UNC_SERVER_{server}"
        self._load()
        n = self._norm_prefix(canonical)
        for access, disk_no in self._access_paths:
            if n == access or n.startswith(access + "\\"): return f"PHYSICAL_DISK_{disk_no}"
        drive = ntpath.splitdrive(canonical)[0].rstrip(":\\/")
        return f"DRIVE_{drive.upper()}" if drive else "UNKNOWN"

    def kind(self, path: Path | str) -> str:
        """"ssd" for a local SSD or NVMe disk, "hdd" for a hard disk or a network share (a share counts as slow), "unknown" for a disk
        that Windows does not identify (it is then handled like a hard disk, but not shown as one)."""
        group = self.group(path)
        if group.startswith("PHYSICAL_DISK_"):
            try: return self._disk_kinds.get(int(group[len("PHYSICAL_DISK_"):]), "unknown")
            except ValueError: return "unknown"
        if group.startswith("UNC_SERVER_"): return "hdd"
        return "unknown"

    def volumes(self) -> list[tuple[str, str]]:
        """Every drive letter of the computer with what it is: "ssd", "hdd" or "unknown" (local disks) or "network" (a mapped share), sorted by letter."""
        self._load()
        found: dict[str, str] = {}
        for access, disk_no in self._access_paths:
            if len(access) == 2 and access[1] == ":" and access[0].isalpha(): found.setdefault(access.upper(), self._disk_kinds.get(disk_no, "unknown"))
        for local, _remote in self._mappings:
            if len(local) == 2 and local[1] == ":" and local[0].isalpha(): found[local.upper()] = "network"
        return sorted(found.items())

    def mount_group(self, path: Path | str) -> str:
        canonical = self.canonical_path(path)
        if canonical.startswith("\\\\"):
            parts = canonical.strip("\\").split("\\")
            return "DRIVE_UNC:" + "\\".join(parts[:2])
        return f"DRIVE_{ntpath.splitdrive(canonical)[0].upper() or 'UNKNOWN'}"


def resolve_mapped_drive(path: str) -> Optional[str]:
    """"Z:\\Movies" -> "\\\\server\\share\\Movies" when Z: is a mapped network drive; None for a local drive or when Windows does not say."""
    import ctypes
    import re
    match = re.match(r"^([A-Za-z]:)(?:\\|/|$)", path)
    if not match: return None
    try:
        function = ctypes.WinDLL("mpr", use_last_error=True).WNetGetConnectionW  # type: ignore[attr-defined]
        function.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
        function.restype = ctypes.c_uint32
        size = ctypes.c_uint32(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if function(match.group(1), buffer, ctypes.byref(size)) != 0: return None
        tail = path[len(match.group(1)):].lstrip("\\/")
        root = buffer.value.rstrip("\\/")
        return root if not tail else root + "\\" + tail.replace("/", "\\")
    except Exception:
        return None


def resolve_shortcut_target(link: str) -> Optional[str]:
    """Where a Windows shortcut (.lnk) points, read through the shell's own IShellLink (no PowerShell, no extra package); None when it cannot be read."""
    import ctypes

    class Guid(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    try:
        ole32 = ctypes.WinDLL("ole32", use_last_error=True)  # type: ignore[attr-defined]
        from_string = ole32.CLSIDFromString
        from_string.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(Guid)]
        from_string.restype = ctypes.c_long

        def guid(text: str) -> Guid:
            value = Guid()
            if from_string(text, ctypes.byref(value)) < 0: raise OSError(f"CLSIDFromString failed for {text}")
            return value

        def method(instance: Any, index: int, restype: Any, *argtypes: Any) -> Any:
            table = ctypes.cast(instance, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])  # type: ignore[attr-defined]

        create = ole32.CoCreateInstance
        create.argtypes = [ctypes.POINTER(Guid), ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(Guid), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_long
        shell_link, link_interface, persist_interface = guid("{00021401-0000-0000-C000-000000000046}"), guid("{000214F9-0000-0000-C000-000000000046}"), guid("{0000010B-0000-0000-C000-000000000046}")
        init = ole32.CoInitializeEx
        init.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        init.restype = ctypes.c_long
        started = init(None, 0x2)
        if started < 0 and started != -2147417850: return None                 # -2147417850: COM already runs in another mode, which is fine
        shell, persist = ctypes.c_void_p(), ctypes.c_void_p()
        try:
            if create(ctypes.byref(shell_link), None, 0x1, ctypes.byref(link_interface), ctypes.byref(shell)) < 0 or not shell: return None
            if method(shell, 0, ctypes.c_long, ctypes.POINTER(Guid), ctypes.POINTER(ctypes.c_void_p))(shell, ctypes.byref(persist_interface), ctypes.byref(persist)) < 0 or not persist: return None
            if method(persist, 5, ctypes.c_long, ctypes.c_wchar_p, ctypes.c_uint32)(persist, link, 0) < 0: return None
            get_path = method(shell, 3, ctypes.c_long, ctypes.POINTER(ctypes.c_wchar), ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
            for flags in (0, 0x4):                                              # 0x4: the path as stored, when the shell cannot resolve it
                buffer = ctypes.create_unicode_buffer(32768)
                if get_path(shell, buffer, len(buffer), None, flags) >= 0 and buffer.value.strip(): return buffer.value.strip()
            return None
        finally:
            for instance in (persist, shell):
                if instance:
                    try: method(instance, 2, ctypes.c_ulong)(instance)
                    except Exception: pass
            if started in (0, 1):
                try: ole32.CoUninitialize()
                except Exception: pass
    except Exception:
        return None


class WindowsPlatform:
    name = "windows"

    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self.storage = WindowsStorageResolver()

    # --- console / desktop ---
    def configure_console(self) -> None:
        # Redirected Windows output otherwise inherits a legacy code page.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    def enable_virtual_terminal(self) -> bool:
        if not sys.stdout.isatty():
            return False
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                return False
            return True
        except Exception:
            return False

    def clear_screen(self) -> None:
        os.system("cls")

    def open_path(self, path: Path) -> None:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            print(t("platform.could_not_open", path=path, error=exc))

    def ffmpeg_setup_help(self) -> tuple[list[str], Optional[str]]:
        return (t("ffmpeg.setup_windows").split("\n"), "https://www.gyan.dev/ffmpeg/builds/")

    @contextlib.contextmanager
    def prevent_sleep(self, enabled: bool) -> Iterator[None]:
        import ctypes
        active = False
        if enabled:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)  # type: ignore[attr-defined]
                active = True
            except Exception:
                pass
        try:
            yield
        finally:
            if active:
                try: ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)  # type: ignore[attr-defined]
                except Exception: pass

    # --- process / filesystem behaviour ---
    def create_instance_lock(self, path: Path) -> _WindowsInstanceLock:
        return _WindowsInstanceLock(path)

    def ffmpeg_executable_names(self) -> tuple[str, str]:
        return "ffmpeg.exe", "ffprobe.exe"

    def system_ffmpeg(self) -> Optional[tuple[Path, Path]]:
        return None     # on Windows FFmpeg is only taken from the `ffmpeg` folder beside the program

    def subprocess_process_options(self) -> dict[str, Any]:
        # Keep console Ctrl+C in the parent; the second interrupt explicitly terminates FFmpeg.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]  # Windows only

    def is_link_or_reparse_point(self, path: Path) -> bool:
        # lstat looks at the link itself: stat() would follow a junction and report the folder it points to, so junctions slipped through.
        try:
            if path.is_symlink(): return True
            return bool(os.lstat(path).st_file_attributes & self.FILE_ATTRIBUTE_REPARSE_POINT)  # type: ignore[attr-defined]
        except Exception:
            return False

    def is_transient_lock_error(self, exc: BaseException) -> bool:
        """"File in use" / "access denied" while renaming: an antivirus scan, the indexer or a media server often lets go within seconds."""
        if not isinstance(exc, OSError): return False
        return getattr(exc, "winerror", None) in (5, 32, 33) or isinstance(exc, PermissionError)

    # --- storage identity ---
    def canonical_media_path(self, path: Path) -> str:
        return self.storage.canonical_path(path)

    def storage_group(self, path: Path) -> str:
        return self.storage.group(path)

    def storage_kind(self, path: Path) -> str:
        return self.storage.kind(path)

    def list_disks(self) -> list[tuple[str, str]]:
        return self.storage.volumes()

    def mount_group(self, path: Path) -> str:
        return self.storage.mount_group(path)

    def resolve_media_path(self, raw: str) -> str:
        """What a person typed or dropped in as a media folder, as a real folder: "~" is the home folder; a shortcut (.lnk) or a Network
        Location (a folder that holds target.lnk) becomes the folder it points to; a mapped drive becomes its \\\\server\\share.
        Raises ValueError when a shortcut cannot be read."""
        value = os.path.expanduser(remove_outer_quotes(raw)).replace("/", "\\")
        link = self._shortcut_file(value)
        if link is not None:
            target = resolve_shortcut_target(link)
            if not target: raise ValueError(f"The shortcut cannot be read: {raw}")
            value = remove_outer_quotes(target).replace("/", "\\")
        value = self.whole_drive(value)
        return resolve_mapped_drive(value) or value

    @staticmethod
    def _shortcut_file(path: str) -> Optional[str]:
        """The .lnk to read for this path: the file itself, or the target.lnk inside a Network Location folder."""
        try:
            candidate = Path(path)
            if candidate.suffix.casefold() == ".lnk" and candidate.is_file(): return str(candidate)
            if (candidate / "target.lnk").is_file(): return str(candidate / "target.lnk")
        except OSError:
            pass
        return None

    def whole_drive(self, raw: str) -> str:
        """"D:" alone is the folder the program last used on drive D, not the drive; the person means the whole drive, so it becomes "D:\\"."""
        return raw + "\\" if len(raw) == 2 and raw[0].isalpha() and raw[1] == ":" else raw

    def path_is_within(self, child: str, parent: str) -> bool:
        child_n = ntpath.normcase(ntpath.normpath(child))
        parent_n = ntpath.normcase(ntpath.normpath(parent))
        return child_n == parent_n or child_n.startswith(parent_n.rstrip("\\/") + "\\")

    # --- hardware ---
    def _powershell(self, command: str) -> str:
        ps = shutil.which("powershell.exe") or shutil.which("powershell")
        if not ps: return ""
        try:
            return run_powershell(ps, command, 10)
        except Exception:
            return ""

    def detect_gpu_names(self) -> list[str]:
        out = self._powershell("Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name")
        return [line.strip() for line in out.splitlines() if line.strip()]

    def system_info(self) -> dict[str, Any]:
        """Processor name, cores, threads and installed memory: one PowerShell call that answers "name|cores|threads|bytes"."""
        out = self._powershell("$c = @(Get-CimInstance Win32_Processor); $m = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
                               "'{0}|{1}|{2}|{3}' -f $c[0].Name, ($c | Measure-Object NumberOfCores -Sum).Sum, "
                               "($c | Measure-Object NumberOfLogicalProcessors -Sum).Sum, $m")
        fields = (out.strip().splitlines() or [""])[-1].split("|")
        if len(fields) != 4: return {}

        def number(text: str) -> int:
            try: return int(float(text.strip()))
            except ValueError: return 0

        return {"cpu_name": fields[0].strip(), "cores": number(fields[1]), "threads": number(fields[2]), "ram_bytes": number(fields[3])}

    def detect_hardware(self) -> dict[str, bool]:
        hw = {"nvidia_gpu": False, "amd_gpu": False, "intel_gpu": False, "intel_cpu": False}
        gpu_names_raw = self._powershell("Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name").lower()
        hw["nvidia_gpu"] = "nvidia" in gpu_names_raw
        hw["amd_gpu"] = "amd" in gpu_names_raw or "radeon" in gpu_names_raw
        hw["intel_gpu"] = "intel" in gpu_names_raw
        cpu_name_raw = self._powershell("Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name").lower()
        hw["intel_cpu"] = "intel" in cpu_name_raw
        return hw

    def extra_backends(self, capabilities: dict[str, Any], hardware: dict[str, bool]) -> list[str]:
        if capabilities.get("d3d11va") and any(hardware.get(k) for k in ("nvidia_gpu", "amd_gpu", "intel_gpu")):
            return ["d3d11va"]
        return []

    def backend_filter(self, backend: str, fps: str, width: int) -> Optional[tuple[list[str], str]]:
        if backend == "d3d11va":
            return ["-hwaccel", "d3d11va"], f"{fps},scale={width}:-2:flags=lanczos"
        return None
