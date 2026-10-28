# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The short texts the configuration menus show for the settings that are set."""
from __future__ import annotations

import unittest

from modules import config_summary as cs
from modules.constants import DEFAULT_CONFIG


class Texts(unittest.TestCase):
    def test_key_frames(self) -> None:
        self.assertEqual(cs.key_frames({"keyframe_only": False}), "off")
        self.assertEqual(cs.key_frames({"keyframe_only": True, "keyframe_repeat_limit_percent": 10.0}), "on (a video with more than 10 % repeated thumbnails is redone with all frames)")
        self.assertEqual(cs.key_frames({"keyframe_only": True, "keyframe_repeat_limit_percent": 0.0}), "on (a video is never redone with all frames)")

    def test_the_defaults_read_well(self) -> None:
        d = DEFAULT_CONFIG
        self.assertEqual(cs.image(d["trickplay"]), f"{d['trickplay']['width']} px, every 10 s, 10x10 tiles, JPEG 90, qscale 4")
        self.assertEqual(cs.fingerprint(d["source_detection"]), "off")

    def test_the_preset_is_recognised_only_when_every_image_value_matches(self) -> None:
        t = dict(DEFAULT_CONFIG["trickplay"])
        self.assertEqual(cs.preset(t), "Jellyfin default")
        for key, other in (("width", 640), ("interval_seconds", 5.0), ("tile_columns", 5), ("tile_rows", 5), ("ffmpeg_qscale", 2), ("jpeg_quality", 80)):
            self.assertEqual(cs.preset({**t, key: other}), "custom", key)
        self.assertEqual(cs.preset({**t, "keyframe_only": True}), "Jellyfin default", "only the image settings count")


if __name__ == "__main__":
    unittest.main()
