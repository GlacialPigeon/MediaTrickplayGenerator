# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional, Protocol


class InstanceLock(Protocol):
    path: Path
    failure: Optional[str]

    def acquire(self) -> bool: ...
    def release(self) -> None: ...


class PlatformAdapter(Protocol):
    name: str

    # --- console / desktop ---
    def configure_console(self) -> None: ...
    def enable_virtual_terminal(self) -> bool: ...
    def clear_screen(self) -> None: ...
    def open_path(self, path: Path) -> None: ...
    def ffmpeg_setup_help(self) -> tuple[list[str], Optional[str]]: ...
    def prevent_sleep(self, enabled: bool) -> AbstractContextManager[Any]: ...

    # --- process / filesystem behaviour ---
    def create_instance_lock(self, path: Path) -> InstanceLock: ...
    def ffmpeg_executable_names(self) -> tuple[str, str]: ...
    def system_ffmpeg(self) -> Optional[tuple[Path, Path]]: ...
    def subprocess_process_options(self) -> dict[str, Any]: ...
    def is_link_or_reparse_point(self, path: Path) -> bool: ...
    def is_transient_lock_error(self, exc: BaseException) -> bool: ...

    # --- storage identity ---
    def canonical_media_path(self, path: Path) -> str: ...
    def storage_group(self, path: Path) -> str: ...
    def storage_kind(self, path: Path) -> str: ...
    def list_disks(self) -> list[tuple[str, str]]: ...
    def mount_group(self, path: Path) -> str: ...
    def path_is_within(self, child: str, parent: str) -> bool: ...
    def whole_drive(self, raw: str) -> str: ...
    def resolve_media_path(self, raw: str) -> str: ...

    # --- hardware ---
    def detect_gpu_names(self) -> list[str]: ...
    def system_info(self) -> dict[str, Any]: ...
    def detect_hardware(self) -> dict[str, bool]: ...
    def extra_backends(self, capabilities: dict[str, Any], hardware: dict[str, bool]) -> list[str]: ...
    def backend_filter(self, backend: str, fps: str, width: int) -> Optional[tuple[list[str], str]]: ...


_PLATFORM: Optional[PlatformAdapter] = None


def configure_platform(adapter: PlatformAdapter) -> None:
    global _PLATFORM
    _PLATFORM = adapter


def platform() -> PlatformAdapter:
    if _PLATFORM is None:
        raise RuntimeError("Platform adapter is not configured.")
    return _PLATFORM


def configure_console() -> None: platform().configure_console()
def enable_virtual_terminal() -> bool: return platform().enable_virtual_terminal()
def clear_screen() -> None: platform().clear_screen()
def open_path(path: Path) -> None: platform().open_path(path)
def ffmpeg_setup_help() -> tuple[list[str], Optional[str]]: return platform().ffmpeg_setup_help()
def prevent_sleep(enabled: bool) -> AbstractContextManager[Any]: return platform().prevent_sleep(enabled)
def create_instance_lock(path: Path) -> InstanceLock: return platform().create_instance_lock(path)
def ffmpeg_executable_names() -> tuple[str, str]: return platform().ffmpeg_executable_names()
def system_ffmpeg() -> Optional[tuple[Path, Path]]: return platform().system_ffmpeg()
def subprocess_process_options() -> dict[str, Any]: return platform().subprocess_process_options()
def is_link_or_reparse_point(path: Path) -> bool: return platform().is_link_or_reparse_point(path)
def is_transient_lock_error(exc: BaseException) -> bool: return platform().is_transient_lock_error(exc)
def canonical_media_path(path: Path) -> str: return platform().canonical_media_path(path)
def storage_group(path: Path) -> str: return platform().storage_group(path)
def storage_kind(path: Path) -> str: return platform().storage_kind(path)
def mount_group(path: Path) -> str: return platform().mount_group(path)
def path_is_within(child: str, parent: str) -> bool: return platform().path_is_within(child, parent)
def whole_drive(raw: str) -> str: return platform().whole_drive(raw)
def resolve_media_path(raw: str) -> str: return platform().resolve_media_path(raw)
def detect_gpu_names() -> list[str]: return platform().detect_gpu_names()
def system_info() -> dict[str, Any]: return platform().system_info()
def list_disks() -> list[tuple[str, str]]: return platform().list_disks()
def detect_hardware() -> dict[str, bool]: return platform().detect_hardware()
def extra_backends(capabilities: dict[str, Any], hardware: dict[str, bool]) -> list[str]:
    return platform().extra_backends(capabilities, hardware)
def backend_filter(backend: str, fps: str, width: int) -> Optional[tuple[list[str], str]]:
    return platform().backend_filter(backend, fps, width)
