# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Picture-format classification (bit depth, resolution class, SDR / HDR type), taken over from MediaDiversitySampler."""
from __future__ import annotations

import sqlite3
import subprocess
import unittest
from pathlib import Path

from tests.helpers import FFMPEG, IsolatedApp, build_tools, needs_ffmpeg, open_state

from modules import media_classification as mc
from modules.diagnostics import select_hdr_candidates
from modules.tiles import classify_video, hdr_label
from tests.test_core import meta

MASTERING = {"side_data_type": "Mastering display metadata", "red_x": "34000/50000", "red_y": "16000/50000", "green_x": "13250/50000",
             "green_y": "34500/50000", "blue_x": "7500/50000", "blue_y": "3000/50000", "white_point_x": "15635/50000",
             "white_point_y": "16450/50000", "min_luminance": "50/10000", "max_luminance": "10000000/10000"}
CLL = {"side_data_type": "Content light level metadata", "max_content": 1000, "max_average": 400}
HDR10_PLUS = {"side_data_type": "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)", "application_version": 1, "num_windows": 1,
              "targeted_system_display_maximum_luminance": "400/1"}


def dovi(profile: int, compat: int, el: int = 0) -> dict:
    return {"side_data_type": "DOVI configuration record", "dv_version_major": 1, "dv_version_minor": 0, "dv_profile": profile,
            "dv_level": 6, "rpu_present_flag": 1, "el_present_flag": el, "bl_present_flag": 1, "dv_bl_signal_compatibility_id": compat}


def stream(**kw) -> dict:
    base = {"codec_name": "hevc", "width": 3840, "height": 2160, "pix_fmt": "yuv420p10le", "bits_per_raw_sample": "10",
            "color_space": "bt2020nc", "color_transfer": "smpte2084", "color_primaries": "bt2020", "field_order": "progressive",
            "side_data_list": []}
    base.update(kw)
    return base


class BitDepthAndResolution(unittest.TestCase):
    def test_bit_depth_comes_from_the_pixel_format_table(self) -> None:
        depth = lambda pf, raw=None: mc.bit_depth(raw, pf)
        self.assertEqual([depth(pf) for pf in ("yuv420p", "yuv420p10le", "p010le", "yuv444p12le", "yuv420p16le")], [8, 10, 10, 12, 16])
        self.assertEqual(depth("nv12"), 8, "the digits of the name are not the bit depth")
        self.assertEqual(depth("yuv410p"), 8, "the digits of the name are not the bit depth")
        self.assertEqual(depth("yuv420p10le", "10"), 10)
        self.assertIsNone(depth("yuv420p", "10"), "more bits than the pixel format can hold: not reliable")
        self.assertEqual(depth("some_new_format", "12"), 12)
        self.assertIsNone(depth(None))

    def test_video_meta_uses_it(self) -> None:
        self.assertEqual(meta(Path("a"), pix_fmt="nv12").bit_depth, 8)
        self.assertEqual(meta(Path("a"), pix_fmt="yuv420p10le").bit_depth, 10)
        self.assertEqual(meta(Path("a"), pix_fmt="yuv420p", bits=10).bit_depth, 10, "the probed value wins")
        self.assertEqual(meta(Path("a"), pix_fmt="").bit_depth, 8)

    def test_resolution_classes(self) -> None:
        cases = {(720, 480): "SD", (1280, 720): "720p", (1920, 1080): "1080p", (1920, 800): "1080p", (2560, 1440): "1440p",
                 (3840, 2160): "4K", (3840, 1600): "4K", (7680, 4320): "8K", (16384, 8640): ">8K", (1080, 1920): "1080p"}
        for (w, h), expected in cases.items():
            with self.subTest(size=f"{w}x{h}"):
                self.assertEqual(mc.resolution_class(w, h), expected)
        self.assertIsNone(mc.resolution_class(0, 0))
        self.assertEqual(meta(Path("a"), width=0, height=0).resolution_class, "SD")


class HdrClassification(unittest.TestCase):
    def classify(self, **kw) -> dict:
        return mc.describe_stream(stream(**kw))

    def test_plain_sdr(self) -> None:
        r = self.classify(codec_name="h264", width=1920, height=1080, pix_fmt="yuv420p", bits_per_raw_sample="8",
                          color_space="bt709", color_transfer="bt709", color_primaries="bt709")
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"]), (False, "SDR", "SDR"))

    def test_bt2020_sdr_is_not_hdr(self) -> None:
        r = self.classify(color_transfer="bt2020-10")
        self.assertEqual((r["hdr"], r["hdr_pool"]), (False, "SDR"))
        r = self.classify(color_transfer="bt2020-10", pix_fmt="yuv420p", bits_per_raw_sample="8")
        self.assertEqual((r["hdr"], r["hdr_pool"]), (False, "SDR Uncertain"), "10-bit transfer in an 8-bit stream is unusual")

    def test_missing_color_information_is_sdr_uncertain(self) -> None:
        r = self.classify(color_transfer=None, color_primaries=None, color_space=None, pix_fmt="yuv420p", bits_per_raw_sample="8")
        self.assertEqual((r["hdr"], r["hdr_pool"]), (False, "SDR Uncertain"))

    def test_hdr10(self) -> None:
        r = self.classify(side_data_list=[MASTERING, CLL])
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"], r["hdr_note"]), (True, "HDR", "HDR10 / PQ", ""))

    def test_pq_without_metadata_blocks_is_still_hdr10(self) -> None:
        r = self.classify()
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"]), (True, "HDR", "HDR10 / PQ"))

    def test_hdr10_plus(self) -> None:
        r = self.classify(side_data_list=[MASTERING, CLL, HDR10_PLUS])
        self.assertEqual((r["hdr"], r["hdr_type"]), (True, "HDR10+"))

    def test_hlg(self) -> None:
        r = self.classify(color_transfer="arib-std-b67")
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"]), (True, "HDR", "HLG"))

    def test_dolby_vision_profiles(self) -> None:
        r = self.classify(side_data_list=[dovi(8, 1)])
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"], r["dv_profile"]), (True, "HDR", "Dolby Vision", "8"))
        r = self.classify(side_data_list=[dovi(5, 0)], color_transfer="smpte2084", color_space="ipt-c2")
        self.assertEqual((r["hdr"], r["hdr_type"], r["dv_profile"]), (True, "Dolby Vision", "5"))
        r = self.classify(codec_name="h264", pix_fmt="yuv420p", bits_per_raw_sample="8", side_data_list=[dovi(9, 2)],
                          color_transfer="bt709", color_primaries="bt709", color_space="bt709")
        self.assertEqual((r["hdr"], r["hdr_type"], r["dv_profile"]), (True, "Dolby Vision", "9"), "DV is HDR even with an SDR base layer")

    def test_hdr_with_missing_details_is_still_hdr(self) -> None:
        r = self.classify(color_primaries=None)
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"]), (True, "HDR Uncertain", "HDR10 / PQ"))
        self.assertIn("primaries", r["hdr_note"])
        r = self.classify(color_transfer="bt709", side_data_list=[MASTERING])
        self.assertEqual((r["hdr"], r["hdr_pool"], r["hdr_type"]), (True, "HDR Uncertain", ""), "HDR metadata, but the type is not confirmed")

    def test_incomplete_side_data_does_not_count(self) -> None:
        r = self.classify(color_transfer="bt709", color_primaries="bt709", color_space="bt709",
                          side_data_list=[{"side_data_type": "Mastering display metadata"}])
        self.assertFalse(r["hdr"])


class Labels(unittest.TestCase):
    def label(self, **kw) -> str:
        info = mc.describe_stream(stream(**kw))
        return classify_video(meta(Path("a"), codec=kw.get("codec_name", "hevc"), pix_fmt=kw.get("pix_fmt", "yuv420p10le"),
                                   width=kw.get("width", 3840), height=kw.get("height", 2160), hdr=info["hdr"], bits=info["bit_depth"],
                                   resolution=info["resolution_class"], hdr_pool=info["hdr_pool"], hdr_type=info["hdr_type"],
                                   dv_profile=info["dv_profile"]))

    def test_hdr_is_shown_at_every_resolution(self) -> None:
        self.assertEqual(self.label(), "HEVC 10-bit 4K HDR10")
        self.assertEqual(self.label(width=1920, height=1080), "HEVC 10-bit 1080p HDR10")
        self.assertEqual(self.label(color_transfer="arib-std-b67", width=1920, height=1080), "HEVC 10-bit 1080p HLG")
        self.assertEqual(self.label(side_data_list=[dovi(8, 1)]), "HEVC 10-bit 4K Dolby Vision P8")
        self.assertEqual(self.label(side_data_list=[MASTERING, CLL, HDR10_PLUS]), "HEVC 10-bit 4K HDR10+")
        self.assertEqual(self.label(color_primaries=None), "HEVC 10-bit 4K HDR10 (uncertain)")

    def test_sdr_is_named_only_where_hdr_would_be_expected(self) -> None:
        sdr = dict(color_transfer="bt709", color_primaries="bt709", color_space="bt709")
        self.assertEqual(self.label(**sdr), "HEVC 10-bit 4K SDR")
        self.assertEqual(self.label(codec_name="h264", pix_fmt="yuv420p", bits_per_raw_sample="8", width=720, height=480, **sdr), "H.264 8-bit SD")

    def test_hdr_label_of_unconfirmed_type(self) -> None:
        self.assertEqual(hdr_label(meta(Path("a"), hdr=True, hdr_pool="HDR Uncertain", hdr_type="")), "HDR (uncertain)")


class HdrCandidates(unittest.TestCase):
    @staticmethod
    def hdr(name: str, hdr_type: str, **kw):
        kw.setdefault("hdr_pool", "HDR")
        kw.setdefault("codec", "hevc")
        kw.setdefault("width", 3840)
        kw.setdefault("height", 2160)
        return meta(Path(name), hdr=True, hdr_type=hdr_type, **kw)

    def names(self, videos) -> list[str]:
        return [m.path.name for m in select_hdr_candidates(videos)]

    def test_one_file_per_kind_of_hdr_in_a_fixed_order(self) -> None:
        sdr = meta(Path("sdr.mkv"), duration=9999.0)
        videos = [sdr, self.hdr("dv8.mkv", "Dolby Vision", dv_profile="8"), self.hdr("hlg.mkv", "HLG"), self.hdr("plus.mkv", "HDR10+"),
                  self.hdr("hdr10.mkv", "HDR10 / PQ"), self.hdr("dv5.mkv", "Dolby Vision", dv_profile="5"), self.hdr("other.mkv", "")]
        self.assertEqual(self.names(videos), ["hdr10.mkv", "plus.mkv", "hlg.mkv", "dv5.mkv", "dv8.mkv", "other.mkv"])

    def test_no_hdr_no_candidates(self) -> None:
        self.assertEqual(self.names([meta(Path("sdr.mkv"))]), [])
        self.assertEqual(self.names([]), [])
        self.assertEqual(self.names([self.hdr("zero.mkv", "HDR10 / PQ", duration=0.0)]), [])

    def test_best_file_of_a_kind_wins(self) -> None:
        small = self.hdr("small.mkv", "HDR10 / PQ", width=1920, height=1080)
        big = self.hdr("big.mkv", "HDR10 / PQ")
        h264 = self.hdr("h264.mkv", "HDR10 / PQ", codec="h264")
        uncertain = self.hdr("unc.mkv", "HDR10 / PQ", hdr_pool="HDR Uncertain", duration=9999.0)
        self.assertEqual(self.names([small, big, h264]), ["big.mkv"])
        self.assertEqual(self.names([uncertain, small]), ["small.mkv"], "confirmed HDR beats uncertain")

    def test_a_kind_with_only_uncertain_files_is_still_tested(self) -> None:
        uncertain = self.hdr("unc.mkv", "HLG", hdr_pool="HDR Uncertain")
        self.assertEqual(self.names([self.hdr("hdr10.mkv", "HDR10 / PQ"), uncertain]), ["hdr10.mkv", "unc.mkv"])


class ProbeCacheMigration(IsolatedApp):
    def test_old_cache_is_extended_and_old_rows_are_probed_again(self) -> None:
        from modules.state_db import StateDB
        db = StateDB()
        db.probe_cache_path.parent.mkdir(parents=True, exist_ok=True)
        old = sqlite3.connect(db.probe_cache_path)
        old.executescript("""
            CREATE TABLE videos (path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, duration REAL, codec TEXT,
                                 profile TEXT, pix_fmt TEXT, width INTEGER, height INTEGER, hdr INTEGER NOT NULL DEFAULT 0,
                                 fingerprint TEXT, last_seen TEXT NOT NULL);
            INSERT INTO videos VALUES ('x.mkv', 10, 20, 60.0, 'hevc', 'Main 10', 'yuv420p10le', 3840, 2160, 1, NULL, 'then');
        """)
        old.commit()
        old.close()
        db.open()
        self.addCleanup(db.close)
        columns = {row["name"] for row in db.probe_conn.execute("PRAGMA table_info(videos)")}
        self.assertTrue({"bits", "chroma", "resolution", "hdr_pool", "hdr_type", "dv_profile", "hdr_note", "class_version"} <= columns)
        self.assertIsNone(db.cached_video(Path("x.mkv"), 10, 20), "classified by older rules: probe again")

        fresh = meta(Path("x.mkv"), size=10, mtime_ns=20, hdr=True, bits=10, chroma="4:2:0", resolution="4K", hdr_pool="HDR Uncertain",
                     hdr_type="Dolby Vision", dv_profile="8", hdr_note="something odd")
        db.upsert_video(fresh)
        back = db.cached_video(Path("x.mkv"), 10, 20)
        self.assertEqual((back.hdr, back.bits, back.chroma, back.resolution, back.hdr_pool, back.hdr_type, back.dv_profile, back.hdr_note),
                         (True, 10, "4:2:0", "4K", "HDR Uncertain", "Dolby Vision", "8", "something odd"))

    def test_opening_twice_is_harmless(self) -> None:
        db = open_state()
        db.close()
        db = open_state()
        self.addCleanup(db.close)
        self.assertIsNone(db.cached_video(Path("none.mkv"), 1, 1))


@needs_ffmpeg
class RealProbe(IsolatedApp):
    def encode(self, name: str, *extra: str) -> Path:
        out = self.media / name
        subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10",
                        "-t", "2", *extra, "-y", str(out)], check=True, capture_output=True)
        return out

    def test_sdr_and_hdr10_files_are_told_apart(self) -> None:
        tools = build_tools()
        sdr = self.encode("sdr.mkv", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
        got = tools.probe(sdr)
        self.assertEqual((got.hdr, got.hdr_pool, got.bit_depth, got.resolution_class), (False, "SDR Uncertain", 8, "SD"))
        try:
            hdr = self.encode("hdr.mkv", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-preset", "ultrafast",
                              "-x265-params", "log-level=none:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                              "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400")
        except subprocess.CalledProcessError:
            self.skipTest("this FFmpeg has no libx265")
        got = tools.probe(hdr)
        self.assertEqual((got.hdr, got.hdr_type, got.bit_depth, got.chroma), (True, "HDR10 / PQ", 10, "4:2:0"), got.hdr_note)
        self.assertEqual(classify_video(got), "HEVC 10-bit SD HDR10")


if __name__ == "__main__":
    unittest.main()
