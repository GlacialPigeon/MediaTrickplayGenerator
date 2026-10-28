# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Classification of a video stream: bit depth, chroma, resolution class and HDR type.

The HDR rules (what counts as SDR, HDR10 / HDR10+ / HLG / Dolby Vision, and which combinations are only "uncertain")
are taken over from MediaDiversitySampler so both programs call the same file the same thing.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

# Raise when the rules above change: videos in the probe cache with an older version are probed again.
CLASSIFICATION_VERSION = 1

PIXEL_FORMAT_INFO: dict[str, tuple[int, str]] = {
    'yuv420p': (8, '4:2:0'), 'yuyv422': (8, '4:2:2'), 'rgb24': (8, 'RGB'),
    'bgr24': (8, 'RGB'), 'yuv422p': (8, '4:2:2'), 'yuv444p': (8, '4:4:4'),
    'yuv410p': (8, '4:1:0'), 'yuv411p': (8, '4:1:1'), 'gray': (8, 'Gray'),
    'monow': (1, 'Gray'), 'monob': (1, 'Gray'), 'yuvj420p': (8, '4:2:0'),
    'yuvj422p': (8, '4:2:2'), 'yuvj444p': (8, '4:4:4'), 'uyvy422': (8, '4:2:2'),
    'uyyvyy411': (8, '4:1:1'), 'bgr8': (0, 'RGB'), 'bgr4': (0, 'RGB'),
    'bgr4_byte': (0, 'RGB'), 'rgb8': (0, 'RGB'), 'rgb4': (0, 'RGB'),
    'rgb4_byte': (0, 'RGB'), 'nv12': (8, '4:2:0'), 'nv21': (8, '4:2:0'),
    'argb': (8, 'RGB'), 'rgba': (8, 'RGB'), 'abgr': (8, 'RGB'), 'bgra': (8, 'RGB'),
    'gray16be': (16, 'Gray'), 'gray16le': (16, 'Gray'), 'yuv440p': (8, '4:4:0'),
    'yuvj440p': (8, '4:4:0'), 'yuva420p': (8, '4:2:0'), 'rgb48be': (16, 'RGB'),
    'rgb48le': (16, 'RGB'), 'rgb565be': (0, 'RGB'), 'rgb565le': (0, 'RGB'),
    'rgb555be': (5, 'RGB'), 'rgb555le': (5, 'RGB'), 'bgr565be': (0, 'RGB'),
    'bgr565le': (0, 'RGB'), 'bgr555be': (5, 'RGB'), 'bgr555le': (5, 'RGB'),
    'yuv420p16le': (16, '4:2:0'), 'yuv420p16be': (16, '4:2:0'),
    'yuv422p16le': (16, '4:2:2'), 'yuv422p16be': (16, '4:2:2'),
    'yuv444p16le': (16, '4:4:4'), 'yuv444p16be': (16, '4:4:4'),
    'rgb444le': (4, 'RGB'), 'rgb444be': (4, 'RGB'), 'bgr444le': (4, 'RGB'),
    'bgr444be': (4, 'RGB'), 'ya8': (8, 'Gray'), 'bgr48be': (16, 'RGB'),
    'bgr48le': (16, 'RGB'), 'yuv420p9be': (9, '4:2:0'), 'yuv420p9le': (9, '4:2:0'),
    'yuv420p10be': (10, '4:2:0'), 'yuv420p10le': (10, '4:2:0'),
    'yuv422p10be': (10, '4:2:2'), 'yuv422p10le': (10, '4:2:2'),
    'yuv444p9be': (9, '4:4:4'), 'yuv444p9le': (9, '4:4:4'),
    'yuv444p10be': (10, '4:4:4'), 'yuv444p10le': (10, '4:4:4'),
    'yuv422p9be': (9, '4:2:2'), 'yuv422p9le': (9, '4:2:2'), 'gbrp': (8, 'RGB'),
    'gbrp9be': (9, 'RGB'), 'gbrp9le': (9, 'RGB'), 'gbrp10be': (10, 'RGB'),
    'gbrp10le': (10, 'RGB'), 'gbrp16be': (16, 'RGB'), 'gbrp16le': (16, 'RGB'),
    'yuva422p': (8, '4:2:2'), 'yuva444p': (8, '4:4:4'),
    'yuva420p9be': (9, '4:2:0'), 'yuva420p9le': (9, '4:2:0'),
    'yuva422p9be': (9, '4:2:2'), 'yuva422p9le': (9, '4:2:2'),
    'yuva444p9be': (9, '4:4:4'), 'yuva444p9le': (9, '4:4:4'),
    'yuva420p10be': (10, '4:2:0'), 'yuva420p10le': (10, '4:2:0'),
    'yuva422p10be': (10, '4:2:2'), 'yuva422p10le': (10, '4:2:2'),
    'yuva444p10be': (10, '4:4:4'), 'yuva444p10le': (10, '4:4:4'),
    'yuva420p16be': (16, '4:2:0'), 'yuva420p16le': (16, '4:2:0'),
    'yuva422p16be': (16, '4:2:2'), 'yuva422p16le': (16, '4:2:2'),
    'yuva444p16be': (16, '4:4:4'), 'yuva444p16le': (16, '4:4:4'),
    'xyz12le': (12, 'Unknown'), 'xyz12be': (12, 'Unknown'), 'nv16': (8, '4:2:2'),
    'nv20le': (10, '4:2:2'), 'nv20be': (10, '4:2:2'), 'rgba64be': (16, 'RGB'),
    'rgba64le': (16, 'RGB'), 'bgra64be': (16, 'RGB'), 'bgra64le': (16, 'RGB'),
    'yvyu422': (8, '4:2:2'), 'ya16be': (16, 'Gray'), 'ya16le': (16, 'Gray'),
    'gbrap': (8, 'RGB'), 'gbrap16be': (16, 'RGB'), 'gbrap16le': (16, 'RGB'),
    '0rgb': (8, 'RGB'), 'rgb0': (8, 'RGB'), '0bgr': (8, 'RGB'), 'bgr0': (8, 'RGB'),
    'yuv420p12be': (12, '4:2:0'), 'yuv420p12le': (12, '4:2:0'),
    'yuv420p14be': (14, '4:2:0'), 'yuv420p14le': (14, '4:2:0'),
    'yuv422p12be': (12, '4:2:2'), 'yuv422p12le': (12, '4:2:2'),
    'yuv422p14be': (14, '4:2:2'), 'yuv422p14le': (14, '4:2:2'),
    'yuv444p12be': (12, '4:4:4'), 'yuv444p12le': (12, '4:4:4'),
    'yuv444p14be': (14, '4:4:4'), 'yuv444p14le': (14, '4:4:4'),
    'gbrp12be': (12, 'RGB'), 'gbrp12le': (12, 'RGB'), 'gbrp14be': (14, 'RGB'),
    'gbrp14le': (14, 'RGB'), 'yuvj411p': (8, '4:1:1'), 'yuv440p10le': (10, '4:4:0'),
    'yuv440p10be': (10, '4:4:0'), 'yuv440p12le': (12, '4:4:0'),
    'yuv440p12be': (12, '4:4:0'), 'ayuv64le': (16, '4:4:4'),
    'ayuv64be': (16, '4:4:4'), 'p010le': (10, '4:2:0'), 'p010be': (10, '4:2:0'),
    'gbrap12be': (12, 'RGB'), 'gbrap12le': (12, 'RGB'), 'gbrap10be': (10, 'RGB'),
    'gbrap10le': (10, 'RGB'), 'gray12be': (12, 'Gray'), 'gray12le': (12, 'Gray'),
    'gray10be': (10, 'Gray'), 'gray10le': (10, 'Gray'), 'p016le': (16, '4:2:0'),
    'p016be': (16, '4:2:0'), 'gray9be': (9, 'Gray'), 'gray9le': (9, 'Gray'),
    'gbrpf32be': (32, 'RGB'), 'gbrpf32le': (32, 'RGB'), 'gbrapf32be': (32, 'RGB'),
    'gbrapf32le': (32, 'RGB'), 'gray14be': (14, 'Gray'), 'gray14le': (14, 'Gray'),
    'grayf32be': (32, 'Gray'), 'grayf32le': (32, 'Gray'),
    'yuva422p12be': (12, '4:2:2'), 'yuva422p12le': (12, '4:2:2'),
    'yuva444p12be': (12, '4:4:4'), 'yuva444p12le': (12, '4:4:4'),
    'nv24': (8, '4:4:4'), 'nv42': (8, '4:4:4'), 'y210be': (10, '4:2:2'),
    'y210le': (10, '4:2:2'), 'x2rgb10le': (10, 'RGB'), 'x2rgb10be': (10, 'RGB'),
    'x2bgr10le': (10, 'RGB'), 'x2bgr10be': (10, 'RGB'), 'p210be': (10, '4:2:2'),
    'p210le': (10, '4:2:2'), 'p410be': (10, '4:4:4'), 'p410le': (10, '4:4:4'),
    'p216be': (16, '4:2:2'), 'p216le': (16, '4:2:2'), 'p416be': (16, '4:4:4'),
    'p416le': (16, '4:4:4'), 'vuya': (8, '4:4:4'), 'rgbaf16be': (16, 'RGB'),
    'rgbaf16le': (16, 'RGB'), 'vuyx': (8, '4:4:4'), 'p012le': (12, '4:2:0'),
    'p012be': (12, '4:2:0'), 'y212be': (12, '4:2:2'), 'y212le': (12, '4:2:2'),
    'xv30be': (10, '4:4:4'), 'xv30le': (10, '4:4:4'), 'xv36be': (12, '4:4:4'),
    'xv36le': (12, '4:4:4'), 'rgbf32be': (32, 'RGB'), 'rgbf32le': (32, 'RGB'),
    'rgbaf32be': (32, 'RGB'), 'rgbaf32le': (32, 'RGB'), 'p212be': (12, '4:2:2'),
    'p212le': (12, '4:2:2'), 'p412be': (12, '4:4:4'), 'p412le': (12, '4:4:4'),
    'gbrap14be': (14, 'RGB'), 'gbrap14le': (14, 'RGB'),
}


def nullable_double(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def nullable_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else None
    except Exception:
        return None


def rational_to_double(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "0/0"}:
        return None
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            n, d = nullable_double(parts[0]), nullable_double(parts[1])
            if n is not None and d is not None and abs(d) > 0.0000001:
                return n / d
        return None
    return nullable_double(text)


def normalized_codec(codec: Optional[str]) -> Optional[str]:
    if not codec or not str(codec).strip():
        return None
    raw = str(codec).strip()
    mapping = {
        "h264": "H264", "hevc": "HEVC", "h265": "HEVC", "av1": "AV1",
        "vp9": "VP9", "vp8": "VP8", "mpeg2video": "MPEG2", "mpeg4": "MPEG4",
        "vc1": "VC1", "wmv3": "WMV3", "theora": "Theora", "prores": "ProRes",
        "dnxhd": "DNxHD/DNxHR",
    }
    return mapping.get(raw.lower(), raw)


def resolution_class(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    long_side, short_side = max(width, height), min(width, height)
    if long_side > 8192 or short_side > 4320: return ">8K"
    if long_side >= 7000 or short_side >= 4000: return "8K"
    if long_side >= 3600 or short_side >= 2000: return "4K"
    if long_side >= 2500 or short_side >= 1300: return "1440p"
    if long_side >= 1800 or short_side >= 900: return "1080p"
    if long_side >= 1200 or short_side >= 650: return "720p"
    return "SD"


def bit_depth(bits_per_raw_sample: Any, pixel_format: Optional[str]) -> Optional[int]:
    explicit = nullable_int(bits_per_raw_sample)
    if explicit is not None and (explicit <= 0 or explicit > 64):
        explicit = None
    descriptor = None
    if pixel_format and pixel_format.lower() in PIXEL_FORMAT_INFO:
        value = int(PIXEL_FORMAT_INFO[pixel_format.lower()][0])
        descriptor = value if value > 0 else None
    if explicit is not None and descriptor is not None and explicit > descriptor:
        return None
    return explicit if explicit is not None else descriptor


def chroma_class(pixel_format: Optional[str]) -> Optional[str]:
    if pixel_format and pixel_format.lower() in PIXEL_FORMAT_INFO:
        value = PIXEL_FORMAT_INFO[pixel_format.lower()][1]
        return None if str(value).casefold() == "unknown" else value
    return None


def scan_type(field_order: Optional[str]) -> Optional[str]:
    if not field_order or not str(field_order).strip():
        return None
    value = str(field_order).strip().lower()
    if value == "progressive": return "Progressive"
    if value in {"tt", "bb", "tb", "bt"}: return "Interlaced"
    return None


def side_data_numbers(block: dict[str, Any], names: Iterable[str]) -> bool:
    usable = False
    for name in names:
        value = block.get(name)
        if value is None: continue
        number = rational_to_double(str(value))
        if number is not None and number >= 0: usable = True
    return usable


def hdr_signals(media: dict[str, Any]) -> dict[str, Any]:
    transfer = str(media.get("ColorTransfer", "")).lower()
    pq = transfer in {"smpte2084", "smpte2084 (pq)", "pq"}
    hlg = transfer in {"arib-std-b67", "hlg"}
    dv = None
    hdr10plus = False
    mastering = False
    cll = False
    for block in media.get("SideDataList") or []:
        if not isinstance(block, dict): continue
        type_ = str(block.get("side_data_type", "")).lower()
        if type_ == "dovi configuration record":
            if side_data_numbers(block, ("dv_profile", "dv_version_major", "rpu_present_flag")): dv = block
        elif type_ == "hdr dynamic metadata smpte2094-40 (hdr10+)":
            if side_data_numbers(block, ("application_version", "num_windows", "targeted_system_display_maximum_luminance")): hdr10plus = True
        elif type_ == "mastering display metadata":
            mastering = side_data_numbers(block, ("red_x", "green_x", "blue_x", "white_point_x", "min_luminance", "max_luminance"))
        elif type_ == "content light level metadata":
            cll = side_data_numbers(block, ("max_content", "max_average"))
    return {"PQ": pq, "HLG": hlg, "DolbyVision": dv is not None, "DolbyVisionConfig": dv, "HDR10Plus": hdr10plus, "MasteringMetadata": mastering, "ContentLightMetadata": cll, "CredibleHDR": pq or hlg or dv is not None or hdr10plus or mastering or cll}


def hdr_metadata_class(signals: dict[str, Any]) -> Optional[str]:
    if signals["DolbyVision"]: return "DolbyVisionMetadata"
    if signals["HDR10Plus"]:
        return "Dynamic+Static" if signals["MasteringMetadata"] or signals["ContentLightMetadata"] else "Dynamic"
    if signals["MasteringMetadata"] and signals["ContentLightMetadata"]: return "Mastering+CLL"
    if signals["MasteringMetadata"]: return "Mastering"
    if signals["ContentLightMetadata"]: return "CLL"
    return None


def dolby_vision_profile(side_data: Iterable[Any]) -> Optional[str]:
    for block in side_data:
        if isinstance(block, dict) and str(block.get("side_data_type", "")).lower() == "dovi configuration record":
            value = nullable_int(block.get("dv_profile"))
            if value is not None and value > 0:
                return str(value)
    return None


def reliable_diversity_value(value: Any) -> bool:
    # PowerShell v3.4 uses -notin here, which is case-insensitive by default.
    # Preserve that behavior explicitly in Python.
    text = str(value if value is not None else "").strip()
    unreliable = {
        "unknown", "n/a", "unspecified", "reserved", "invalid",
        "not detected", "none", "unknown hdr", "sdr / unknown",
        "pq / unknown hdr", "-99",
    }
    return bool(text) and text.lower() not in unreliable


def classify_hdr(media: dict[str, Any]) -> dict[str, Any]:
    """HDR class of one video stream (`media` as built by media_info()): pool SDR / SDR Uncertain / HDR / HDR Uncertain,
    the HDR type, and the reasons for any uncertainty. Taken over unchanged from MediaDiversitySampler."""
    signals = hdr_signals(media)
    flags: set[str] = set()
    reasons: list[str] = []
    transfer = str(media["ColorTransfer"]).lower()
    primaries = str(media["ColorPrimaries"]).lower()
    matrix = str(media["ColorSpace"]).lower()
    hdr_type: Optional[str] = None
    pool = "SDR"
    expected_transfer = None
    proprietary = False
    dv_sdr = False
    expected_depth = None
    is_dv = signals["DolbyVision"]
    if not signals["CredibleHDR"]:
        sdr = {"bt709","gamma22","gamma28","smpte170m","smpte240m","iec61966-2-1","iec61966-2-4","bt1361e","bt2020-10","bt2020-12"}
        hdr_type = "SDR"
        if transfer not in sdr:
            pool = "SDR Uncertain"
            if not reliable_diversity_value(transfer):
                flags.add("MissingMetadata")
                reasons.append("SDR transfer is missing or unspecified.")
            else:
                flags.add("UnusualCombination")
                reasons.append(f"Transfer is outside the confirmed SDR mapping: {transfer}.")
        elif matrix in {"ictcp", "ipt-c2"}:
            pool = "SDR Uncertain"
            flags.add("UnusualCombination")
            reasons.append(f"SDR transfer with an unconfirmed matrix combination: {matrix}.")
        if transfer in {"bt2020-10", "bt2020-12"}:
            expected_depth = 10 if transfer == "bt2020-10" else 12
            depth = media.get("BitDepthNumeric")
            if depth != expected_depth:
                pool = "SDR Uncertain"
                flags.add("UnusualCombination")
                reported = "unspecified" if depth is None else f"{depth}-bit"
                reasons.append(f"BT.2020 transfer {transfer} expects {expected_depth}-bit; reported {reported}.")
            if primaries != "bt2020":
                pool = "SDR Uncertain"
                flags.add("UnusualCombination")
                reasons.append(f"BT.2020 transfer {transfer} expects bt2020 primaries; reported {primaries or 'unspecified'}.")
            if matrix not in {"bt2020nc", "bt2020c"}:
                pool = "SDR Uncertain"
                flags.add("UnusualCombination")
                reasons.append(f"BT.2020 transfer {transfer} expects bt2020nc or bt2020c matrix; reported {matrix or 'unspecified'}.")
    else:
        pool = "HDR"
        if is_dv:
            hdr_type = "Dolby Vision"
            dv = signals["DolbyVisionConfig"] or {}
            profile = nullable_int(dv.get("dv_profile"))
            compat = nullable_int(dv.get("dv_bl_signal_compatibility_id"))
            rpu = nullable_int(dv.get("rpu_present_flag"))
            bl = nullable_int(dv.get("bl_present_flag"))
            el = nullable_int(dv.get("el_present_flag"))
            if any(v is None for v in (profile, compat, rpu, bl, el)):
                flags.add("MissingMetadata")
                reasons.append("DV profile, compatibility ID or RPU/BL/EL flag is missing.")
            allowed = {5:{0}, 7:{6}, 8:{1,4}, 9:{2}, 10:{0,1,4}, 20:{0,4}}
            if profile is None or profile not in allowed:
                flags.add("UnusualCombination")
                reasons.append("DV profile is outside the verified profile matrix.")
            else:
                if compat is not None and compat not in allowed[profile]:
                    flags.add("UnusualCombination")
                    reasons.append(f"Unconfirmed DV profile/compatibility combination: {profile}/{compat}.")
                expected_codec = "HEVC"
                expected_depth = 10
                if profile == 9:
                    expected_codec = "H264"
                    expected_depth = 8
                if profile == 10: expected_codec = "AV1"
                if not reliable_diversity_value(media["Codec"]):
                    flags.add("MissingMetadata")
                    reasons.append("DV codec is missing.")
                elif media["Codec"] != expected_codec:
                    flags.add("UnusualCombination")
                    reasons.append(f"DV profile {profile} expects {expected_codec}; reported {media['Codec']}.")
                expected_el = 1 if profile == 7 else 0
                if (rpu is not None and rpu != 1) or (bl is not None and bl != 1) or (el is not None and el != expected_el):
                    flags.add("UnusualCombination")
                    reasons.append("DV RPU/BL/EL flags do not match the verified configuration.")
            proprietary = compat == 0
            dv_sdr = profile == 9 and compat == 2
            if compat in {1,6}: expected_transfer = "smpte2084"
            if compat == 4: expected_transfer = "arib-std-b67"
            if dv_sdr: expected_transfer = "bt709"
            if media["ScanType"] == "Interlaced":
                flags.add("UnusualCombination")
                reasons.append("DV is reported as interlaced.")
        elif signals["HDR10Plus"]:
            hdr_type = "HDR10+"
            expected_transfer = "smpte2084"
        elif signals["PQ"]:
            hdr_type = "HDR10 / PQ"
            expected_transfer = "smpte2084"
        elif signals["HLG"]:
            hdr_type = "HLG"
            expected_transfer = "arib-std-b67"
        else:
            flags.add("UnknownHDRType")
            reasons.append("Usable static HDR metadata is present, but no supported HDR type is confirmed.")
        if hdr_type is not None:
            depth = media["BitDepthNumeric"]
            if depth is None:
                flags.add("MissingMetadata")
                reasons.append("Component bit depth cannot be determined reliably.")
            elif (expected_depth is not None and depth != expected_depth) or (expected_depth is None and depth not in {10,12}):
                flags.add("UnusualCombination")
                reasons.append(f"Bit depth {depth} is outside the verified configuration.")
            if not reliable_diversity_value(media["Chroma"]):
                flags.add("MissingMetadata")
                reasons.append("Pixel format/chroma cannot be determined reliably.")
            elif is_dv and media["Chroma"] != "4:2:0":
                flags.add("UnusualCombination")
                reasons.append("DV chroma is outside the verified 4:2:0 configuration.")
            if proprietary:
                if reliable_diversity_value(transfer) and transfer != "smpte2084":
                    flags.add("UnusualCombination")
                    reasons.append("Unconfirmed proprietary DV transfer.")
                if reliable_diversity_value(primaries) and primaries != "bt2020":
                    flags.add("UnusualCombination")
                    reasons.append("Unconfirmed proprietary DV primaries.")
                if reliable_diversity_value(matrix) and matrix != "ipt-c2":
                    flags.add("UnusualCombination")
                    reasons.append("Unconfirmed proprietary DV matrix; expected IPT-PQ-C2 or unspecified.")
            elif expected_transfer is not None:
                expected_primaries = "bt2020"
                allowed_matrices = {"bt2020nc"}
                if dv_sdr:
                    expected_primaries = "bt709"
                    allowed_matrices = {"bt709"}
                elif not is_dv and media["Chroma"] == "RGB": allowed_matrices = {"gbr", "rgb"}
                elif not is_dv and media["Chroma"] == "4:4:4": allowed_matrices = {"bt2020nc", "ictcp"}
                checks = [("transfer", transfer, {expected_transfer}), ("primaries", primaries, {expected_primaries}), ("matrix", matrix, allowed_matrices)]
                for name, value, expected in checks:
                    if not reliable_diversity_value(value):
                        flags.add("MissingMetadata")
                        reasons.append(f"HDR {name} is missing or unspecified.")
                    elif value not in expected:
                        if name == "transfer" and (is_dv or signals["HDR10Plus"]) and value in {"smpte2084", "arib-std-b67", "bt709"}:
                            flags.add("InconsistentColorMetadata")
                            reasons.append(f"Reported transfer {value} conflicts with the declared HDR configuration ({'/'.join(sorted(expected))}).")
                        else:
                            flags.add("UnusualCombination")
                            reasons.append(f"Unconfirmed HDR {name}: {value}; expected {'/'.join(sorted(expected))}.")
            if signals["HDR10Plus"] and is_dv and not signals["PQ"] and not proprietary:
                flags.add("UnusualCombination")
                reasons.append("Additional HDR10+ signal without a confirmed PQ base context.")
        if flags: pool = "HDR Uncertain"
    return {"Pool": pool, "HDRType": hdr_type, "HDRMetadataClass": hdr_metadata_class(signals), "HDRSignals": signals, "Anomalies": sorted(flags), "UnknownReason": " ".join(reasons)}


def _display(value: Any) -> Optional[str]:
    if value is None: return None
    text = str(value).strip()
    return text or None


def media_info(stream: dict[str, Any]) -> dict[str, Any]:
    """The fields classify_hdr() needs, from one stream of `ffprobe -show_streams -of json`."""
    width, height = nullable_int(stream.get("width")), nullable_int(stream.get("height"))
    pixel = _display(stream.get("pix_fmt"))
    return {
        "Codec": normalized_codec(stream.get("codec_name")), "Width": width, "Height": height,
        "ResolutionClass": resolution_class(width, height), "PixelFormat": pixel, "Chroma": chroma_class(pixel),
        "BitDepthNumeric": bit_depth(stream.get("bits_per_raw_sample"), pixel), "ScanType": scan_type(stream.get("field_order")),
        "ColorSpace": _display(stream.get("color_space")), "ColorTransfer": _display(stream.get("color_transfer")),
        "ColorPrimaries": _display(stream.get("color_primaries")),
        "SideDataList": [b for b in (stream.get("side_data_list") or []) if isinstance(b, dict)],
    }


def describe_stream(stream: dict[str, Any]) -> dict[str, Any]:
    """Everything the program keeps about a stream's picture format: bit depth, chroma, resolution class and HDR."""
    media = media_info(stream)
    hdr = classify_hdr(media)
    return {
        "bit_depth": media["BitDepthNumeric"] or 0, "chroma": media["Chroma"] or "", "resolution_class": media["ResolutionClass"] or "SD",
        "hdr": hdr["Pool"] in {"HDR", "HDR Uncertain"},      # HDR with a few missing details is still HDR
        "hdr_pool": hdr["Pool"], "hdr_type": hdr["HDRType"] or "", "dv_profile": dolby_vision_profile(media["SideDataList"]) or "",
        "hdr_note": hdr["UnknownReason"],
    }
