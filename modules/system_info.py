# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""What the computer is: the processor, the memory, and how they are told on a screen, in the log and in the benchmark report."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from . import platform_api, storage_workers
from .localization import t


@dataclass(frozen=True)
class Host:
    cpu_name: str = ""
    cores: int = 0
    threads: int = 0
    ram_bytes: int = 0


_HOST: Optional[Host] = None


def host() -> Host:
    """The processor and memory of this computer; asked once. Whatever cannot be found is left empty (never an error)."""
    global _HOST
    if _HOST is None:
        try: raw = platform_api.system_info()
        except Exception: raw = {}
        _HOST = Host(" ".join(str(raw.get("cpu_name", "")).split()), int(raw.get("cores", 0) or 0), int(raw.get("threads", 0) or 0), int(raw.get("ram_bytes", 0) or 0))
    return _HOST


def cpu_text(info: Host) -> str:
    """"Intel Core i7-14700K, 20 cores / 28 threads" (what is known of it)."""
    counts = t("system.cores_threads", cores=info.cores, threads=info.threads) if info.cores and info.threads else ""
    return ", ".join(part for part in (info.cpu_name, counts) if part)


def ram_text(info: Host) -> str:
    """"32 GB": the installed memory in whole gigabytes (the system reports a little less than is built in)."""
    return f"{max(1, round(info.ram_bytes / 2 ** 30))} GB" if info.ram_bytes else ""


def disk_text(place: str, kind: str) -> str:
    """"D: SSD", "X: HDD", "M: network", "E: unknown (treated as HDD)"."""
    return f"{place} {storage_workers.kind_label(kind)}"


def all_disks() -> list[str]:
    """Every disk of the computer, short and with its kind: ["C: SSD", "D: SSD", "X: HDD"]; empty when the system does not say."""
    try: return [disk_text(place, kind) for place, kind in platform_api.list_disks()]
    except Exception: return []


def as_dict(info: Host, gpu_names: list[str], disks: list[str], selected_disks: list[str]) -> dict[str, Any]:
    """The same for the benchmark report and its result.json."""
    return {**asdict(info), "gpu_names": list(gpu_names), "disks": list(disks), "selected_disks": list(selected_disks)}


def report_lines(system: dict[str, Any]) -> list[str]:
    """The lines of the report that say what the computer is (always English, like the rest of the report)."""
    info = Host(str(system.get("cpu_name", "")), int(system.get("cores", 0) or 0), int(system.get("threads", 0) or 0), int(system.get("ram_bytes", 0) or 0))
    counts = f"{info.cores} cores / {info.threads} threads" if info.cores and info.threads else ""
    pairs = [("CPU", ", ".join(part for part in (info.cpu_name, counts) if part)), ("GPU", ", ".join(system.get("gpu_names") or [])),
             ("RAM", ram_text(info)), ("Disks", ", ".join(system.get("disks") or [])),
             ("Selected disks", ", ".join(system.get("selected_disks") or []))]
    return [f"{label}: {value}" for label, value in pairs if value]
