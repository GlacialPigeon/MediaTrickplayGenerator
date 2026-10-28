# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""How many videos are processed at once on a disk, by kind of disk."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules import storage_workers


def item(group: str, path: str) -> SimpleNamespace:
    return SimpleNamespace(storage_group=group, meta=SimpleNamespace(path=Path(path)))


class PerDisk(unittest.TestCase):
    def test_auto_gives_a_hard_disk_one_and_an_ssd_two(self) -> None:
        self.assertEqual((storage_workers.per_disk("auto", "hdd"), storage_workers.per_disk("auto", "ssd")), (1, 2))

    def test_a_number_is_used_for_every_kind_and_kept_within_the_limits(self) -> None:
        self.assertEqual([storage_workers.per_disk(v, "hdd") for v in (1, 3, "3", 4)], [1, 3, 3, 4])
        self.assertEqual([storage_workers.per_disk(v, "ssd") for v in (0, -2, 9)], [1, 1, 4])

    def test_anything_that_is_not_a_number_means_auto(self) -> None:
        for value in ("auto", None, "", "many", True, [], {}):
            self.assertTrue(storage_workers.is_auto(value), value)
            self.assertEqual(storage_workers.per_disk(value, "ssd"), 2)
        self.assertFalse(storage_workers.is_auto(1))
        self.assertFalse(storage_workers.is_auto("2"))

    def test_an_unknown_kind_is_a_hard_disk(self) -> None:
        self.assertEqual(storage_workers.per_disk("auto", "tape"), 1)


class Layout(unittest.TestCase):
    def kinds(self, mapping: dict[str, str]):
        return mock.patch("modules.storage.storage_kind", side_effect=lambda path: next(k for prefix, k in mapping.items() if Path(path).as_posix().startswith(prefix)))

    def test_every_group_gets_its_own_kind_and_count(self) -> None:
        items = [item("PHYSICAL_DISK_0", "/ssd/a.mkv"), item("PHYSICAL_DISK_0", "/ssd/b.mkv"), item("PHYSICAL_DISK_2", "/hdd/c.mkv")]
        with self.kinds({"/ssd": "ssd", "/hdd": "hdd"}):
            self.assertEqual(storage_workers.layout(items, "auto"), {"PHYSICAL_DISK_0": ("ssd", 2), "PHYSICAL_DISK_2": ("hdd", 1)})
            self.assertEqual(storage_workers.layout(items, 3), {"PHYSICAL_DISK_0": ("ssd", 3), "PHYSICAL_DISK_2": ("hdd", 3)})

    def test_a_group_with_a_hard_disk_among_its_videos_is_a_hard_disk(self) -> None:
        items = [item("MANUAL:all", "/ssd/a.mkv"), item("MANUAL:all", "/hdd/b.mkv")]
        with self.kinds({"/ssd": "ssd", "/hdd": "hdd"}):
            self.assertEqual(storage_workers.layout(items, "auto"), {"MANUAL:all": ("hdd", 1)})

    def test_the_kind_is_asked_once_per_folder(self) -> None:
        items = [item("g", f"/ssd/show/{n}.mkv") for n in range(50)]
        with self.kinds({"/ssd": "ssd"}) as patched:
            storage_workers.layout(items, "auto")
        self.assertEqual(patched.call_count, 1)

    def test_nothing_to_do_gives_no_groups(self) -> None:
        self.assertEqual(storage_workers.layout([], "auto"), {})
        self.assertEqual(storage_workers.group_kind([]), "unknown")


if __name__ == "__main__":
    unittest.main()
