# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import ROOT, IsolatedApp, needs_linux, needs_symlinks, needs_windows  # noqa: F401  (sets sys.path)

from modules.platform_linux import LinuxPlatform, find_mount, parse_mountinfo, unescape_mountinfo
from modules.platform_windows import WindowsPlatform, WindowsStorageResolver

MOUNTINFO = r"""
22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw
30 22 0:25 / /mnt/space\040dir rw - ext4 /dev/sdb2 rw
31 22 253:0 / /mnt/lvm rw - ext4 /dev/mapper/vg-root rw
32 22 0:44 / /mnt/nas rw - cifs //NAS/Media rw,vers=3.0
33 22 0:45 / /mnt/nas-alias rw - cifs //nas/media rw,vers=3.0
34 22 0:46 /sub /mnt/export rw - nfs4 FileServer:/export/data rw
35 22 0:47 / /run/tmpfs rw - tmpfs tmpfs rw
36 22 0:48 / /mnt/btrfs rw - btrfs /dev/nvme0n1p3 rw,subvol=/@
"""


def fake_sysfs(root: Path) -> None:
    """Block devices: sda (sda1, SSD), sdb (sdb2, hard disk), nvme0n1 (nvme0n1p3, SSD), dm-0 built from sdc (hard disk) + sdd (SSD)."""
    rotational = {"sda": "0", "sdb": "1", "nvme0n1": "0", "sdc": "1", "sdd": "0"}
    devices = root / "devices"
    for disk, parts in {"sda": ["sda1"], "sdb": ["sdb2"], "nvme0n1": ["nvme0n1p3"], "sdc": [], "sdd": []}.items():
        d = devices / disk
        d.mkdir(parents=True)
        (root / "class" / "block").mkdir(parents=True, exist_ok=True)
        (root / "class" / "block" / disk).symlink_to(d)
        (root / "block").mkdir(exist_ok=True)
        (root / "block" / disk / "queue").mkdir(parents=True, exist_ok=True)
        (root / "block" / disk / "queue" / "rotational").write_text(rotational[disk] + "\n")
        for part in parts:
            pd = d / part
            pd.mkdir()
            (pd / "partition").write_text("1")
            (root / "class" / "block" / part).symlink_to(pd)
    dm = devices / "dm-0"
    dm.mkdir()
    (root / "class" / "block" / "dm-0").symlink_to(dm)
    slaves = root / "block" / "dm-0" / "slaves"
    slaves.mkdir(parents=True)
    (slaves / "sdc").symlink_to(devices / "sdc")
    (slaves / "sdd").symlink_to(devices / "sdd")


class MountInfoParsing(unittest.TestCase):
    def test_unescape(self) -> None:
        self.assertEqual(unescape_mountinfo(r"/mnt/space\040dir"), "/mnt/space dir")
        self.assertEqual(unescape_mountinfo(r"a\134b"), "a\\b")
        self.assertEqual(unescape_mountinfo("plain"), "plain")
        self.assertEqual(unescape_mountinfo("trail\\04"), "trail\\04")

    def test_parse_and_longest_mount(self) -> None:
        mounts = parse_mountinfo(MOUNTINFO)
        self.assertEqual(len(mounts), 8)
        self.assertEqual(find_mount("/mnt/space dir/movie.mkv", mounts).source, "/dev/sdb2")
        self.assertEqual(find_mount("/home/user/a.mkv", mounts).source, "/dev/sda1")
        self.assertEqual(find_mount("/mnt/nas/Movies/a.mkv", mounts).filesystem, "cifs")
        self.assertEqual(find_mount("/mnt/space dirty/x", mounts).source, "/dev/sda1", "prefix must match whole path components")

    def test_garbage_lines_are_ignored(self) -> None:
        self.assertEqual(parse_mountinfo("nonsense\n\n1 2 3\n"), [])


@needs_symlinks
class LinuxStorage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        fake_sysfs(base / "sys")
        (base / "mountinfo").write_text(MOUNTINFO)
        self.adapter = LinuxPlatform(mountinfo_path=str(base / "mountinfo"), sys_root=str(base / "sys"),
                                     dri_dir=str(base / "dri"), cpuinfo_path=str(base / "cpuinfo"))
        self.base = base

    def test_partitions_resolve_to_their_physical_disk(self) -> None:
        self.assertEqual(self.adapter.storage_group(Path("/home/x/a.mkv")), "PHYSICAL_DISK_sda")
        self.assertEqual(self.adapter.storage_group(Path("/mnt/space dir/a.mkv")), "PHYSICAL_DISK_sdb")
        self.assertEqual(self.adapter.storage_group(Path("/mnt/btrfs/a.mkv")), "PHYSICAL_DISK_nvme0n1")

    def test_device_mapper_resolves_to_underlying_disks(self) -> None:
        # /dev/mapper/vg-root normally symlinks to ../dm-0; emulate with the realpath basename.
        mounts = parse_mountinfo(MOUNTINFO.replace("/dev/mapper/vg-root", "/dev/dm-0"))
        (self.base / "mountinfo").write_text(MOUNTINFO.replace("/dev/mapper/vg-root", "/dev/dm-0"))
        self.assertTrue(mounts)
        self.adapter._mount_cache = (0.0, [])
        self.assertEqual(self.adapter.storage_group(Path("/mnt/lvm/a.mkv")), "PHYSICAL_DISK_sdc+sdd")

    def test_every_disk_of_the_computer_is_listed_once_with_its_kind(self) -> None:
        self.assertEqual(self.adapter.list_disks(), [("/", "ssd"), ("/mnt/btrfs", "ssd"), ("/mnt/lvm", "unknown"), ("/mnt/nas", "network"), ("/mnt/space dir", "hdd")],
                         "one entry per disk or share; a share mounted twice, a folder bound from elsewhere and tmpfs are not disks")

    def test_a_disk_is_an_ssd_only_when_it_is_not_rotational(self) -> None:
        kind = self.adapter.storage_kind
        self.assertEqual(kind(Path("/home/x/a.mkv")), "ssd")
        self.assertEqual(kind(Path("/mnt/btrfs/a.mkv")), "ssd", "NVMe")
        self.assertEqual(kind(Path("/mnt/space dir/a.mkv")), "hdd")
        self.assertEqual(kind(Path("/run/tmpfs/a")), "unknown", "a file system without a disk is not called a hard disk")

    def test_a_disk_set_with_one_hard_disk_is_a_hard_disk(self) -> None:
        (self.base / "mountinfo").write_text(MOUNTINFO.replace("/dev/mapper/vg-root", "/dev/dm-0"))
        self.adapter._mount_cache = (0.0, [])
        self.assertEqual(self.adapter.storage_kind(Path("/mnt/lvm/a.mkv")), "hdd")

    def test_a_disk_without_the_rotational_file_is_unknown(self) -> None:
        (self.base / "sys" / "block" / "sda" / "queue" / "rotational").unlink()
        self.assertEqual(self.adapter.storage_kind(Path("/home/x/a.mkv")), "unknown")

    def test_an_unexpected_rotational_value_is_unknown(self) -> None:
        (self.base / "sys" / "block" / "sda" / "queue" / "rotational").write_text("maybe\n")
        self.assertEqual(self.adapter.storage_kind(Path("/home/x/a.mkv")), "unknown")

    def test_network_shares_are_hard_disks(self) -> None:
        self.assertEqual(self.adapter.storage_kind(Path("/mnt/nas/a.mkv")), "hdd")
        self.assertEqual(self.adapter.storage_kind(Path("/mnt/export/a.mkv")), "hdd")

    def test_network_shares_group_by_server_and_alias_mounts_share_identity(self) -> None:
        self.assertEqual(self.adapter.storage_group(Path("/mnt/nas/a.mkv")), "NET_SERVER_nas")
        self.assertEqual(self.adapter.storage_group(Path("/mnt/export/a.mkv")), "NET_SERVER_fileserver")
        a = self.adapter.canonical_media_path(Path("/mnt/nas/Movies/a.mkv"))
        b = self.adapter.canonical_media_path(Path("/mnt/nas-alias/Movies/a.mkv"))
        self.assertEqual(a, b)
        self.assertEqual(a, "//nas/media/movies/a.mkv")
        self.assertEqual(self.adapter.mount_group(Path("/mnt/nas/x")), self.adapter.mount_group(Path("/mnt/nas-alias/x")))

    def test_nfs_canonical_keeps_export_case(self) -> None:
        self.assertEqual(self.adapter.canonical_media_path(Path("/mnt/export/Dir/a.mkv")), "fileserver:/export/data/sub/Dir/a.mkv",
                         "the mount shows the /sub folder of the export, so the path below it includes /sub")

    def test_virtual_filesystems_get_a_stable_fallback_group(self) -> None:
        self.assertEqual(self.adapter.storage_group(Path("/run/tmpfs/a")), "FS_tmpfs_/run/tmpfs")

    def test_local_paths_resolve_symlinks(self) -> None:
        real = self.base / "real"
        real.mkdir()
        link = self.base / "link"
        link.symlink_to(real)
        self.assertEqual(self.adapter.canonical_media_path(link), os.path.realpath(real))

    def test_path_is_within(self) -> None:
        w = self.adapter.path_is_within
        self.assertTrue(w("/mnt/a/b", "/mnt/a"))
        self.assertTrue(w("/mnt/a", "/mnt/a"))
        self.assertFalse(w("/mnt/ab", "/mnt/a"))
        self.assertTrue(w("/anything", "/"))
        self.assertTrue(w("//nas/media/x", "//nas/media"))


class LinuxHardware(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def card(self, name: str, vendor: str, klass: str = "0x030000") -> None:
        dev = self.base / "sys" / "class" / "drm" / name / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
        (dev / "class").write_text(klass + "\n")

    def adapter(self) -> LinuxPlatform:
        (self.base / "dri").mkdir(exist_ok=True)
        (self.base / "cpuinfo").write_text("vendor_id\t: GenuineIntel\n")
        return LinuxPlatform(sys_root=str(self.base / "sys"), dri_dir=str(self.base / "dri"), cpuinfo_path=str(self.base / "cpuinfo"))

    def test_vendor_detection(self) -> None:
        self.card("card0", "0x8086")
        self.card("card1", "0x10de")
        (self.base / "sys" / "class" / "drm" / "card0-HDMI-A-1").mkdir()
        hw = self.adapter().detect_hardware()
        self.assertEqual(hw, {"nvidia_gpu": True, "amd_gpu": False, "intel_gpu": True, "intel_cpu": True})

    def test_vaapi_needs_render_node_capability_and_intel_or_amd(self) -> None:
        self.card("card0", "0x1002")
        adapter = self.adapter()
        caps = {"vaapi": True, "scale_vaapi": True}
        hw = adapter.detect_hardware()
        self.assertEqual(adapter.extra_backends(caps, hw), [], "no render node yet")
        (self.base / "dri" / "renderD128").write_text("")
        self.assertEqual(adapter.extra_backends(caps, hw), ["vaapi"])
        self.assertEqual(adapter.extra_backends({"vaapi": True}, hw), [], "scale_vaapi missing")
        self.assertEqual(adapter.extra_backends(caps, {"nvidia_gpu": True}), [], "NVIDIA uses the CUDA path")

    def test_vaapi_filter(self) -> None:
        (self.base / "dri").mkdir(exist_ok=True)
        (self.base / "dri" / "renderD129").write_text("")
        adapter = self.adapter()
        args, vf = adapter.backend_filter("vaapi", "fps=1/10", 640)
        self.assertIn(str(self.base / "dri" / "renderD129"), args)
        self.assertEqual(vf, "fps=1/10,scale_vaapi=w=640:h=-2:format=nv12,hwdownload,format=nv12")
        self.assertIsNone(adapter.backend_filter("d3d11va", "fps=1/10", 640))


class SystemInfoTests(unittest.TestCase):
    def test_linux_reads_the_processor_cores_threads_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cpu = "".join(f"processor\t: {i}\nmodel name\t: Intel(R) Core(TM) i7-14700K\nphysical id\t: 0\ncore id\t\t: {i // 2}\n\n" for i in range(8))
            (base / "cpuinfo").write_text(cpu)
            (base / "meminfo").write_text("MemTotal:       32768000 kB\nMemFree:         1000 kB\n")
            info = LinuxPlatform(cpuinfo_path=str(base / "cpuinfo"), meminfo_path=str(base / "meminfo")).system_info()
        self.assertEqual(info, {"cpu_name": "Intel(R) Core(TM) i7-14700K", "threads": 8, "cores": 4, "ram_bytes": 32768000 * 1024})

    def test_linux_without_the_files_says_nothing_instead_of_failing(self) -> None:
        self.assertEqual(LinuxPlatform(cpuinfo_path="/nonexistent/cpuinfo", meminfo_path="/nonexistent/meminfo").system_info(), {})

    def test_windows_reads_one_line_from_powershell(self) -> None:
        adapter = WindowsPlatform()
        adapter._powershell = lambda command: "Intel(R) Core(TM) i7-14700K|20|28|34240000000\n"  # type: ignore[method-assign]
        self.assertEqual(adapter.system_info(), {"cpu_name": "Intel(R) Core(TM) i7-14700K", "cores": 20, "threads": 28, "ram_bytes": 34240000000})
        adapter._powershell = lambda command: ""  # type: ignore[method-assign]
        self.assertEqual(adapter.system_info(), {})

    def test_the_texts(self) -> None:
        from modules import system_info
        info = system_info.Host("Intel Core i7-14700K", 20, 28, 34240000000)
        self.assertEqual(system_info.cpu_text(info), "Intel Core i7-14700K, 20 cores / 28 threads")
        self.assertEqual(system_info.ram_text(info), "32 GB")
        self.assertEqual(system_info.cpu_text(system_info.Host("Some CPU")), "Some CPU")
        self.assertEqual(system_info.ram_text(system_info.Host()), "")
        report = system_info.report_lines(system_info.as_dict(info, ["NVIDIA GeForce RTX 5060", "Intel UHD"], ["C: SSD", "D: SSD", "X: HDD"], ["D: SSD", "X: HDD"]))
        self.assertEqual(report, ["CPU: Intel Core i7-14700K, 20 cores / 28 threads", "GPU: NVIDIA GeForce RTX 5060, Intel UHD", "RAM: 32 GB",
                                  "Disks: C: SSD, D: SSD, X: HDD", "Selected disks: D: SSD, X: HDD"])
        self.assertEqual(system_info.disk_text("M:", "network"), "M: network")
        self.assertEqual(system_info.disk_text("D:", "ssd"), "D: SSD")
        self.assertEqual(system_info.report_lines({}), [])


class LinuxProcess(unittest.TestCase):
    @needs_linux
    def test_instance_lock_is_exclusive_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data" / "application.lock"
            adapter = LinuxPlatform()
            first, second = adapter.create_instance_lock(path), adapter.create_instance_lock(path)
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            self.assertEqual(second.failure, "busy")
            first.release()
            self.assertTrue(second.acquire())
            second.release()

    @needs_linux
    @needs_symlinks
    def test_instance_lock_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "victim.txt"
            target.write_text("keep")
            lock_path = Path(tmp) / "application.lock"
            lock_path.symlink_to(target)
            lock = LinuxPlatform().create_instance_lock(lock_path)
            self.assertFalse(lock.acquire())
            self.assertEqual(target.read_text(), "keep")

    def test_names_and_options(self) -> None:
        adapter = LinuxPlatform()
        self.assertEqual(adapter.ffmpeg_executable_names(), ("ffmpeg", "ffprobe"))
        self.assertEqual(adapter.subprocess_process_options(), {"start_new_session": True})

    @needs_symlinks
    def test_symlink_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "r"
            real.mkdir()
            (Path(tmp) / "l").symlink_to(real)
            adapter = LinuxPlatform()
            self.assertTrue(adapter.is_link_or_reparse_point(Path(tmp) / "l"))
            self.assertFalse(adapter.is_link_or_reparse_point(real))

    def test_prevent_sleep_never_raises(self) -> None:
        with LinuxPlatform().prevent_sleep(True):
            pass
        with LinuxPlatform().prevent_sleep(False):
            pass


@needs_windows
class WindowsProcess(unittest.TestCase):
    """Runs only on a real Windows host."""

    def test_instance_lock_is_exclusive_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data" / "application.lock"
            adapter = WindowsPlatform()
            first, second = adapter.create_instance_lock(path), adapter.create_instance_lock(path)
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            self.assertEqual(second.failure, "busy")
            first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_names_and_process_options(self) -> None:
        adapter = WindowsPlatform()
        self.assertEqual(adapter.ffmpeg_executable_names(), ("ffmpeg.exe", "ffprobe.exe"))
        self.assertIn("creationflags", adapter.subprocess_process_options())

    def test_storage_group_of_a_local_path_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = WindowsPlatform()
            first = adapter.storage_group(Path(tmp))
            self.assertTrue(first)
            self.assertEqual(first, adapter.storage_group(Path(tmp) / "sub" / "x.mkv"))
            self.assertTrue(adapter.mount_group(Path(tmp)))

    def test_prevent_sleep_never_raises(self) -> None:
        with WindowsPlatform().prevent_sleep(True):
            pass


class WindowsAdapterLogic(unittest.TestCase):
    """The Windows adapter's pure logic uses ntpath, so it is checked on every host OS."""

    def resolver(self) -> WindowsStorageResolver:
        r = WindowsStorageResolver()
        r.ingest({"Partitions": [{"AccessPath": "C:\\", "DiskNumber": 0}, {"AccessPath": "D:\\", "DiskNumber": 1}],
                  "Mappings": {"LocalPath": "M:", "RemotePath": "\\\\NAS\\Media"}})
        return r

    def test_every_drive_letter_is_listed_with_its_kind(self) -> None:
        r = WindowsStorageResolver()
        r.ingest({"Partitions": [{"AccessPath": "C:\\", "DiskNumber": 0}, {"AccessPath": "D:\\", "DiskNumber": 1}, {"AccessPath": "X:\\", "DiskNumber": 2},
                                 {"AccessPath": "C:\\mnt\\extra\\", "DiskNumber": 2}],
                  "Mappings": {"LocalPath": "M:", "RemotePath": "\\\\NAS\\Media"},
                  "Disks": [{"DeviceId": "0", "MediaType": "SSD", "BusType": "NVMe"}, {"DeviceId": "1", "MediaType": "SSD", "BusType": "SATA"}, {"DeviceId": "2", "MediaType": "HDD", "BusType": "SATA"}]})
        self.assertEqual(r.volumes(), [("C:", "ssd"), ("D:", "ssd"), ("M:", "network"), ("X:", "hdd")], "a folder a disk is mounted in is not a drive")

    def test_local_disks_mapped_drives_and_unc(self) -> None:
        r = self.resolver()
        self.assertEqual(r.group("C:/Users/x/a.mkv"), "PHYSICAL_DISK_0")
        self.assertEqual(r.group("D:\\Movies\\b.mkv"), "PHYSICAL_DISK_1")
        self.assertEqual(r.canonical_path("M:\\Movies\\c.mkv"), "\\\\NAS\\Media\\Movies\\c.mkv")
        self.assertEqual(r.group("M:\\Movies\\c.mkv"), "UNC_SERVER_nas")
        self.assertEqual(r.group("\\\\nas\\Media\\d.mkv"), "UNC_SERVER_nas")
        self.assertEqual(r.mount_group("M:\\Movies"), "DRIVE_UNC:NAS\\Media")
        self.assertEqual(r.mount_group("C:\\x"), "DRIVE_C:")

    def test_disk_kinds_from_get_physicaldisk(self) -> None:
        r = WindowsStorageResolver()
        r.ingest({"Partitions": [{"AccessPath": "C:\\", "DiskNumber": 0}, {"AccessPath": "D:\\", "DiskNumber": 1}, {"AccessPath": "X:\\", "DiskNumber": 2},
                                 {"AccessPath": "Y:\\", "DiskNumber": 3}, {"AccessPath": "Z:\\", "DiskNumber": 4}],
                  "Mappings": {"LocalPath": "M:", "RemotePath": "\\\\NAS\\Media"},
                  "Disks": [{"DeviceId": "0", "MediaType": "SSD", "BusType": "SATA"}, {"DeviceId": "1", "MediaType": "Unspecified", "BusType": "NVMe"},
                            {"DeviceId": "2", "MediaType": "HDD", "BusType": "SATA"}, {"DeviceId": "3", "MediaType": "Unspecified", "BusType": "USB"}]})
        self.assertEqual([r.kind(p) for p in ("C:\\a.mkv", "D:\\a.mkv", "X:\\a.mkv", "Y:\\a.mkv")], ["ssd", "ssd", "hdd", "unknown"])
        self.assertEqual(r.kind("Z:\\a.mkv"), "unknown", "a disk Windows did not describe")
        self.assertEqual(r.kind("M:\\Movies\\a.mkv"), "hdd", "a mapped network drive")
        self.assertEqual(r.kind("\\\\nas\\Media\\a.mkv"), "hdd")
        self.assertEqual(r.kind("E:\\a.mkv"), "unknown", "a drive with no known disk")

    def test_a_single_disk_row_and_garbage_are_read(self) -> None:
        r = WindowsStorageResolver()
        r.ingest({"Partitions": {"AccessPath": "D:\\", "DiskNumber": 1}, "Disks": {"DeviceId": "1", "MediaType": "SSD", "BusType": "SATA"}})
        self.assertEqual(r.kind("D:\\a.mkv"), "ssd")
        r.ingest({"Disks": [{"DeviceId": "x", "MediaType": "SSD"}, {"MediaType": "SSD"}]})
        self.assertEqual(r.kind("D:\\a.mkv"), "ssd")

    def test_a_bare_drive_letter_means_the_whole_drive_not_the_folder_last_used_on_it(self) -> None:
        whole = WindowsPlatform().whole_drive
        self.assertEqual([whole(r) for r in ("D:", "x:", "D:\\", "D:\\Media", "Media", "")], ["D:\\", "x:\\", "D:\\", "D:\\Media", "Media", ""])
        self.assertEqual(LinuxPlatform().whole_drive("D:"), "D:", "elsewhere a name with a colon is just a name")

    def test_a_typed_media_folder_becomes_a_real_folder_on_windows(self) -> None:
        w = WindowsPlatform()
        with mock.patch("modules.platform_windows.resolve_mapped_drive", side_effect=lambda p: "\\\\nas\\media" + p[2:] if p.upper().startswith("Z:") else None), \
             mock.patch("modules.platform_windows.resolve_shortcut_target", return_value="Z:\\Movies"), \
             mock.patch.object(WindowsPlatform, "_shortcut_file", side_effect=lambda p: "link.lnk" if p.endswith(("Movies.lnk", "Network Shortcuts\\Movies")) else None):
            self.assertEqual(w.resolve_media_path('  "D:\\Media"  '), "D:\\Media", "a local folder stays as it is, the quotes go")
            self.assertEqual(w.resolve_media_path("D:"), "D:\\", "a bare drive letter is the whole drive")
            self.assertEqual(w.resolve_media_path("Z:\\Shows"), "\\\\nas\\media\\Shows", "a mapped drive becomes its share")
            self.assertEqual(w.resolve_media_path("C:\\Users\\me\\Movies.lnk"), "\\\\nas\\media\\Movies", "a shortcut becomes its target, and a mapped drive in it its share")
            self.assertEqual(w.resolve_media_path("C:\\Network Shortcuts\\Movies"), "\\\\nas\\media\\Movies", "a Network Location folder (it holds target.lnk) the same")
        with mock.patch("modules.platform_windows.resolve_shortcut_target", return_value=None), \
             mock.patch.object(WindowsPlatform, "_shortcut_file", return_value="link.lnk"):
            with self.assertRaises(ValueError): w.resolve_media_path("C:\\x\\Movies.lnk")

    @needs_linux        # '~' is expanded by the operating system's own rules
    def test_a_typed_media_folder_on_linux_expands_the_home_folder(self) -> None:
        with mock.patch.dict(os.environ, {"HOME": "/home/gregor"}):
            self.assertEqual(LinuxPlatform().resolve_media_path(' "~/Movies" '), "/home/gregor/Movies")
            self.assertEqual(LinuxPlatform().resolve_media_path("~"), "/home/gregor")
            self.assertEqual(LinuxPlatform().resolve_media_path("/mnt/nas/a b"), "/mnt/nas/a b")

    def test_path_is_within_is_case_insensitive_with_backslashes(self) -> None:
        w = WindowsPlatform().path_is_within
        self.assertTrue(w("D:\\Media\\Movies", "d:\\media"))
        self.assertTrue(w("\\\\NAS\\Media\\x", "\\\\nas\\media"))
        self.assertFalse(w("D:\\MediaX", "D:\\Media"))

    def test_prevent_sleep_sets_and_clears_the_execution_state(self) -> None:
        import ctypes
        from unittest import mock
        fake = mock.MagicMock()
        adapter = WindowsPlatform()
        with mock.patch.object(ctypes, "windll", fake, create=True):
            with adapter.prevent_sleep(True):
                set_state = fake.kernel32.SetThreadExecutionState
                set_state.assert_called_once_with(adapter.ES_CONTINUOUS | adapter.ES_SYSTEM_REQUIRED)
            self.assertEqual(set_state.call_count, 2)
            set_state.assert_called_with(adapter.ES_CONTINUOUS)
            fake.reset_mock()
            with adapter.prevent_sleep(False):
                pass
            fake.kernel32.SetThreadExecutionState.assert_not_called()

    def test_d3d11va_backend_and_filter(self) -> None:
        w = WindowsPlatform()
        self.assertEqual(w.extra_backends({"d3d11va": True}, {"amd_gpu": True}), ["d3d11va"])
        self.assertEqual(w.extra_backends({"d3d11va": True}, {}), [])
        self.assertEqual(w.extra_backends({"vaapi": True}, {"intel_gpu": True}), [])
        args, vf = w.backend_filter("d3d11va", "fps=1/10", 640)
        self.assertEqual(args, ["-hwaccel", "d3d11va"])
        self.assertEqual(vf, "fps=1/10,scale=640:-2:flags=lanczos")
        self.assertEqual(w.ffmpeg_executable_names(), ("ffmpeg.exe", "ffprobe.exe"))


class PlatformIsolationAudit(unittest.TestCase):
    """OS-specific APIs may appear only in the two adapters and main.py's selector."""

    TOKENS = [r"sys\.platform", r"os\.name\b", r"\bmsvcrt\b", r"\bfcntl\b", r"\bctypes\b", r"windll", r"powershell",
              r"startfile", r"xdg-open", r"CREATE_NEW_PROCESS_GROUP", r"start_new_session", r"/proc/", r"/sys/",
              r"\.exe\b", r"\bcls\b", r"is_windows", r"\bntpath\b"]

    def test_core_modules_contain_no_os_specific_code(self) -> None:
        offenders: list[str] = []
        for path in sorted((ROOT / "modules").glob("*.py")):
            if path.name in {"platform_windows.py", "platform_linux.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            for token in self.TOKENS:
                if path.name == "constants.py" and token == r"\.exe\b":
                    continue  # ".exe" is a non-video file extension in the exclusion list, not an executable name
                if re.search(token, code):
                    offenders.append(f"{path.name}: {token}")
        self.assertEqual(offenders, [])

    def test_every_module_imports_with_the_linux_adapter(self) -> None:
        import importlib
        for path in sorted((ROOT / "modules").glob("*.py")):
            if path.stem != "__init__":
                importlib.import_module(f"modules.{path.stem}")


if __name__ == "__main__":
    unittest.main()
