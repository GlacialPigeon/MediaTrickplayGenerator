# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Behaviour shared with the sibling tools (Media Diversity Sampler, Media Cataloger): paths, quotes, atomic writes, storage kinds, FFmpeg."""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.helpers import IsolatedApp, needs_linux, needs_symlinks  # noqa: F401  (sets sys.path)

from modules import constants, file_io, storage_workers
from modules.paths import remove_outer_quotes
from modules.platform_linux import LinuxPlatform, filesystem_relative_path, network_host, network_share, nfs_source, parse_mountinfo
from modules.platform_windows import WindowsPlatform, WindowsStorageResolver, disk_kind, run_powershell
from modules.scanner import MediaScanner, sanitize_media_roots


class RemoveOuterQuotes(unittest.TestCase):
    def test_every_pair_of_quotes_is_removed(self) -> None:
        for raw in ('"/mnt/a b"', "'/mnt/a b'", "\u201c/mnt/a b\u201d", "\u2018/mnt/a b\u2019", '  "/mnt/a b"  '):
            self.assertEqual(remove_outer_quotes(raw), "/mnt/a b", raw)

    def test_nested_quotes_and_odd_input(self) -> None:
        self.assertEqual(remove_outer_quotes("\"'/mnt/a'\""), "/mnt/a")
        self.assertEqual(remove_outer_quotes('"/mnt/a'), '"/mnt/a', "a quote on one side only belongs to the name")
        self.assertEqual(remove_outer_quotes('"'), '"')
        self.assertEqual(remove_outer_quotes(None), "")
        self.assertEqual(remove_outer_quotes(""), "")

    def test_adapters_use_it(self) -> None:
        self.assertEqual(LinuxPlatform().resolve_media_path("\u201c/mnt/a b\u201d"), "/mnt/a b")
        self.assertEqual(LinuxPlatform().resolve_media_path("'/mnt/a'"), "/mnt/a")


class MediaFolderMustBeFullPath(IsolatedApp):
    def test_a_relative_name_is_skipped_with_a_message(self) -> None:
        folder = self.media / "Movies"
        folder.mkdir()
        with mock.patch("builtins.print") as shown:
            roots = sanitize_media_roots(["Movies", "../media", str(folder)], announce=True)
        self.assertEqual([str(r) for r in roots], [str(folder)])
        self.assertEqual(sum("not a full path" in str(call.args[0]) for call in shown.call_args_list), 2)

    def test_quoted_full_path_is_accepted(self) -> None:
        folder = self.media / "Movies"
        folder.mkdir()
        roots = sanitize_media_roots([f'"{folder}"', f"\u201c{folder}\u201d"], announce=False)
        self.assertEqual([str(r) for r in roots], [str(folder)])

    def test_home_folder_shortcut_is_a_full_path(self) -> None:
        with mock.patch.dict(os.environ, {"HOME": str(self.media)}):
            roots = sanitize_media_roots(["~"], announce=False)
        self.assertEqual(len(roots), 1)


class AtomicWrite(unittest.TestCase):
    def test_writes_and_replaces_without_leftovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "a.txt"
            file_io.atomic_write_text(target, "one\n")
            file_io.atomic_write_text(target, "two\n")
            self.assertEqual(target.read_text(), "two\n")
            self.assertEqual([p.name for p in target.parent.iterdir()], ["a.txt"])

    def test_a_failed_replace_keeps_the_old_file_and_removes_the_temporary_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            target.write_text("old")
            with mock.patch("os.replace", side_effect=OSError("boom")), self.assertRaises(OSError):
                file_io.atomic_write_text(target, "new")
            self.assertEqual(target.read_text(), "old")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["a.txt"])

    def test_a_failed_write_leaves_nothing_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            with mock.patch("os.fsync", side_effect=OSError("disk")), self.assertRaises(OSError):
                file_io.atomic_write_text(target, "new")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    @needs_linux
    def test_the_permissions_of_the_replaced_file_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            target.write_text("old")
            target.chmod(0o640)
            file_io.atomic_write_text(target, "new")
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)

    def test_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.json"
            file_io.atomic_write_json(target, {"name": "č š ž"})
            self.assertEqual(file_io.safe_json_load(target), {"name": "č š ž"})


class JunctionDetection(unittest.TestCase):
    """The Windows adapter must look at the link itself: stat() follows a junction and shows the folder it points to."""

    def test_a_junction_is_found_through_lstat_even_when_stat_hides_it(self) -> None:
        adapter = WindowsPlatform()
        junction = SimpleNamespace(st_file_attributes=0x400)
        target = SimpleNamespace(st_file_attributes=0x10)
        fake = mock.Mock(spec=Path)
        fake.is_symlink.return_value = False
        fake.stat.return_value = target
        with mock.patch("os.lstat", return_value=junction, create=True):
            self.assertTrue(adapter.is_link_or_reparse_point(fake))

    def test_a_plain_folder_is_not_a_link(self) -> None:
        adapter = WindowsPlatform()
        fake = mock.Mock(spec=Path)
        fake.is_symlink.return_value = False
        with mock.patch("os.lstat", return_value=SimpleNamespace(st_file_attributes=0x10), create=True):
            self.assertFalse(adapter.is_link_or_reparse_point(fake))

    def test_a_missing_path_is_not_a_link(self) -> None:
        self.assertFalse(WindowsPlatform().is_link_or_reparse_point(Path("/definitely/not/here")))

    @unittest.skipUnless(os.name == "nt", "needs real Windows")
    def test_a_real_junction_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            real.mkdir()
            junction = Path(tmp) / "junction"
            subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(real)], check=True, stdout=subprocess.DEVNULL)
            adapter = WindowsPlatform()
            self.assertTrue(adapter.is_link_or_reparse_point(junction))
            self.assertFalse(adapter.is_link_or_reparse_point(real))


class ScannerSkipsNasAndSystemFolders(IsolatedApp):
    def test_nas_trash_and_system_folders_are_skipped_at_any_depth(self) -> None:
        scanner = MediaScanner.__new__(MediaScanner)
        for name in ("@eaDir", "#recycle", "#snapshot", "@Recycle", ".@__thumb", "lost+found", ".Trash", ".Trash-1000", "$RECYCLE.BIN",
                     "System Volume Information", "Movie.trickplay"):
            self.assertTrue(scanner._skip_dir(Path("/media") / name, set()), name)

    def test_normal_folders_are_kept(self) -> None:
        scanner = MediaScanner.__new__(MediaScanner)
        for name in ("Movies", "Trash Talk (2020)", "recycle", "Eadir"):
            self.assertFalse(scanner._skip_dir(Path("/media") / name, set()), name)


class DiskKinds(unittest.TestCase):
    def test_windows_believes_only_what_it_is_told(self) -> None:
        self.assertEqual(disk_kind("SSD", "SATA"), "ssd")
        self.assertEqual(disk_kind("Unspecified", "NVMe"), "ssd")
        self.assertEqual(disk_kind("HDD", "SATA"), "hdd")
        self.assertEqual(disk_kind("Unspecified", "USB"), "unknown")
        self.assertEqual(disk_kind("", ""), "unknown")

    def test_volumes_report_unknown_disks(self) -> None:
        r = WindowsStorageResolver()
        r.ingest({"Partitions": [{"AccessPath": "D:\\", "DiskNumber": 1}, {"AccessPath": "E:\\", "DiskNumber": 9}],
                  "Disks": [{"DeviceId": "1", "MediaType": "Unspecified", "BusType": "USB"}]})
        self.assertEqual(r.volumes(), [("D:", "unknown"), ("E:", "unknown")])

    def test_group_kind_of_unknown_disks(self) -> None:
        kinds = {"/a": "ssd", "/b": "unknown", "/c": "hdd"}
        with mock.patch("modules.storage.storage_kind", side_effect=lambda p: kinds[p.parent.as_posix()]):
            self.assertEqual(storage_workers.group_kind([Path("/a/x.mkv")]), "ssd")
            self.assertEqual(storage_workers.group_kind([Path("/a/x.mkv"), Path("/b/x.mkv")]), "unknown")
            self.assertEqual(storage_workers.group_kind([Path("/b/x.mkv"), Path("/c/x.mkv")]), "hdd")

    def test_unknown_disks_get_the_hard_disk_number_of_videos_but_another_name(self) -> None:
        self.assertEqual(storage_workers.per_disk("auto", "unknown"), storage_workers.AUTO_WORKERS["hdd"])
        self.assertEqual(storage_workers.kind_label("unknown"), "unknown (treated as HDD)")
        self.assertEqual([storage_workers.kind_label(k) for k in ("ssd", "hdd", "network")], ["SSD", "HDD", "network"])


class PowerShellOutput(unittest.TestCase):
    def test_output_is_requested_and_read_as_utf8(self) -> None:
        completed = SimpleNamespace(stdout="\ufeff{\"Name\": \"Šola č ž\"}".encode("utf-8"))
        with mock.patch("subprocess.run", return_value=completed) as run:
            text = run_powershell("powershell.exe", "Get-Thing", 5)
        self.assertEqual(text, "{\"Name\": \"Šola č ž\"}")
        command = run.call_args.args[0][-1]
        self.assertIn("OutputEncoding", command)
        self.assertTrue(command.endswith("Get-Thing"))
        self.assertNotIn("text", run.call_args.kwargs)

    def test_broken_bytes_do_not_raise(self) -> None:
        with mock.patch("subprocess.run", return_value=SimpleNamespace(stdout=b"ok \xff")):
            self.assertTrue(run_powershell("powershell.exe", "x", 5).startswith("ok "))


MOUNTS = r"""
22 1 8:1 / / rw - ext4 /dev/sda1 rw
40 22 0:50 / /mnt/share rw - nfs4 NasBox:/Export rw
41 22 0:50 /Movies /mnt/movies rw - nfs4 NasBox:/Export rw
42 22 0:51 / /mnt/v6 rw - nfs4 [fe80::1]:/srv/Data rw
43 22 0:52 /Series /mnt/cifs-sub rw - cifs //NAS/Media rw
44 22 0:53 / /mnt/auto rw - autofs systemd-1 rw
"""


@needs_linux
class LinuxBindMountsAutofsNfs(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mountinfo = Path(self._tmp.name) / "mountinfo"
        self.mountinfo.write_text(MOUNTS)
        self.adapter = LinuxPlatform(mountinfo_path=str(self.mountinfo), sys_root=str(Path(self._tmp.name) / "sys"))
        patcher = mock.patch.object(self.adapter, "_real", side_effect=lambda p: str(p))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_folder_bound_from_a_subfolder_keeps_the_subfolder_in_its_identity(self) -> None:
        whole = self.adapter.canonical_media_path(Path("/mnt/share/Movies/a.mkv"))
        bound = self.adapter.canonical_media_path(Path("/mnt/movies/a.mkv"))
        self.assertEqual(whole, bound)
        self.assertEqual(bound, "nasbox:/Export/Movies/a.mkv")
        self.assertNotEqual(bound, self.adapter.canonical_media_path(Path("/mnt/share/a.mkv")))

    def test_cifs_bind_mount_root(self) -> None:
        self.assertEqual(self.adapter.canonical_media_path(Path("/mnt/cifs-sub/a.mkv")), "//nas/media/series/a.mkv")

    def test_filesystem_relative_path(self) -> None:
        mounts = parse_mountinfo(MOUNTS)
        movies = next(m for m in mounts if m.mount_point == "/mnt/movies")
        share = next(m for m in mounts if m.mount_point == "/mnt/share")
        self.assertEqual(filesystem_relative_path("/mnt/movies", movies), "Movies")
        self.assertEqual(filesystem_relative_path("/mnt/movies/A/b.mkv", movies), "Movies/A/b.mkv")
        self.assertEqual(filesystem_relative_path("/mnt/share", share), "")
        self.assertEqual(filesystem_relative_path("/mnt/share/A", share), "A")

    def test_ipv6_nfs_hosts(self) -> None:
        self.assertEqual(nfs_source("[fe80::1]:/srv/Data"), ("[fe80::1]", "/srv/Data"))
        self.assertEqual(nfs_source("fe80::1:/srv"), ("fe80::1", "/srv"))
        self.assertEqual(nfs_source("Host:/a/b"), ("Host", "/a/b"))
        mount = next(m for m in parse_mountinfo(MOUNTS) if m.mount_point == "/mnt/v6")
        self.assertEqual(network_host(mount), "[fe80::1]")
        self.assertEqual(self.adapter.storage_group(Path("/mnt/v6/a.mkv")), "NET_SERVER_[fe80::1]")

    def test_nfs_share_names_ignore_the_case_of_the_host_only(self) -> None:
        mount = next(m for m in parse_mountinfo(MOUNTS) if m.mount_point == "/mnt/share")
        self.assertEqual(network_share(mount), "nasbox:/Export")
        self.assertEqual(self.adapter.mount_group(Path("/mnt/share/a")), "MOUNT_NET:nasbox:/Export")

    def test_an_autofs_trigger_alone_is_not_a_storage(self) -> None:
        self.assertEqual(self.adapter.storage_group(Path("/mnt/auto/x/a.mkv")), "UNKNOWN")
        self.assertEqual(self.adapter.storage_kind(Path("/mnt/auto/x/a.mkv")), "unknown")

    def test_an_automounted_share_is_read_again_after_the_folder_was_opened(self) -> None:
        auto = Path(self._tmp.name) / "auto"
        trigger = f"46 22 0:70 / {auto} rw - autofs systemd-1 rw\n"
        self.mountinfo.write_text(MOUNTS + trigger)
        self.adapter._mount_cache = (0.0, [])
        self.adapter._mounts()      # the cache now holds the table that shows only the trigger
        self.mountinfo.write_text(MOUNTS + trigger + f"47 46 0:71 / {auto}/share rw - nfs4 NasBox:/Auto rw\n")
        mount = self.adapter._mount_for(str(auto / "share" / "a.mkv"))
        self.assertIsNotNone(mount)
        self.assertEqual(mount.filesystem, "nfs4")
        self.assertEqual(self.adapter.storage_group(auto / "share" / "a.mkv"), "NET_SERVER_nasbox")

    def test_a_disk_that_cannot_be_identified_is_unknown(self) -> None:
        self.assertEqual(self.adapter.storage_kind(Path("/home/x/a.mkv")), "unknown")


class LinuxSystemFfmpeg(unittest.TestCase):
    def test_both_programs_must_be_on_path(self) -> None:
        adapter = LinuxPlatform()
        paths = {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}
        with mock.patch("shutil.which", side_effect=paths.get):
            self.assertEqual(adapter.system_ffmpeg(), (Path("/usr/bin/ffmpeg"), Path("/usr/bin/ffprobe")))
        with mock.patch("shutil.which", side_effect={"ffmpeg": "/usr/bin/ffmpeg"}.get):
            self.assertIsNone(adapter.system_ffmpeg())
        with mock.patch("shutil.which", return_value=None):
            self.assertIsNone(adapter.system_ffmpeg())

    def test_windows_never_uses_the_system_one(self) -> None:
        with mock.patch("shutil.which", return_value="C:\\tools\\ffmpeg.exe"):
            self.assertIsNone(WindowsPlatform().system_ffmpeg())

    def test_the_setup_help_names_apt_and_dnf(self) -> None:
        lines, url = LinuxPlatform().ffmpeg_setup_help()
        text = "\n".join(lines)
        self.assertIn("sudo apt install ffmpeg", text)
        self.assertIn("sudo dnf install ffmpeg-free", text)
        self.assertIsNone(url)


class SleepInhibitor(unittest.TestCase):
    @needs_linux
    def test_the_whole_process_group_is_ended(self) -> None:
        proc = mock.Mock()
        proc.pid = 4242
        proc.wait.return_value = 0
        with mock.patch("shutil.which", return_value="/usr/bin/systemd-inhibit"), mock.patch("subprocess.Popen", return_value=proc), \
                mock.patch("os.killpg") as killpg:
            with LinuxPlatform().prevent_sleep(True):
                killpg.assert_not_called()
        killpg.assert_called_once_with(4242, signal.SIGTERM)

    @needs_linux
    def test_it_is_killed_when_it_ignores_terminate(self) -> None:
        proc = mock.Mock()
        proc.pid = 4242
        proc.wait.side_effect = [subprocess.TimeoutExpired("x", 3), 0]
        with mock.patch("shutil.which", return_value="/usr/bin/systemd-inhibit"), mock.patch("subprocess.Popen", return_value=proc), \
                mock.patch("os.killpg") as killpg:
            with LinuxPlatform().prevent_sleep(True):
                pass
        self.assertEqual([c.args[1] for c in killpg.call_args_list], [signal.SIGTERM, signal.SIGKILL])


class Version(unittest.TestCase):
    def test_version_is_1_0_0_everywhere(self) -> None:
        import json
        self.assertEqual(constants.APP_VERSION, "1.0.0")
        for path in sorted(constants.LOCALES_DIR.glob("*.json")):
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["_meta.version"], "1.0.0", path.name)

    def test_every_python_file_has_the_author_header(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = ["# Media Trickplay Generator", "# Author: GlacialPigeon", "# GitHub: https://github.com/GlacialPigeon"]
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or "ffmpeg" in path.relative_to(root).parts[:1]:
                continue
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[:3], expected, str(path.relative_to(root)))


if __name__ == "__main__":
    unittest.main()
