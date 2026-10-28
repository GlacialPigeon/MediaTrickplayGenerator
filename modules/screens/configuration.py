# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from .. import config_summary, constants, hardware_decode, hdr_validation, localization, path_choice, storage_workers
from ..constants import JELLYFIN_IMAGE_PRESET, LOCALES_DIR
from ..formatting import display_path
from ..localization import lazy, t
from ..paths import is_relative_to_path, normalize_path, remove_outer_quotes
from ..prompts import choose, parse_number_selection, pause, prompt_float, prompt_int, prompt_yes_no
from ..scanner import sanitize_media_roots
from .core import LEAVE_MENU, text as resolve
from .maintenance import MaintenanceScreens


def parse_excluded_name(raw: str) -> tuple[Optional[str], str]:
    """A folder name for the exclusion list, or (None, why not). Names apply at any depth, so a path makes no sense."""
    name = remove_outer_quotes(raw).rstrip("/\\").strip()
    if name in {"", ".", ".."}: return None, t("excluded.not_a_name")
    if "/" in name or "\\" in name or ":" in name: return None, t("excluded.not_a_path")
    if name.casefold().endswith(".trickplay"): return None, t("excluded.trickplay_skipped")
    return name, ""


class ConfigurationScreens(MaintenanceScreens):
    """Configuration menu and its screens."""

    def _first_run_roots(self) -> None:
        if self.config.get("media_roots"): return
        self.configure_roots()

    def configure_roots(self) -> Optional[bool]:
        """False when the person went back without changing anything (no "Press Enter" after that)."""
        self._screen(t("roots.title"))
        current = self.config.get("media_roots", [])
        if not current: print(t("roots.none"))
        if current:
            print(t("roots.current"))
            for p in current: print(f"  - {p}")
        print()
        choice = choose([("1", t("roots.enter_manually")), ("2", t("roots.load_txt")), ("0", t("common.back"))], "1" if not current else "0")
        if choice == "0": return False

        roots: list[str] = []
        if choice == "1":
            roots = []
            print(t("roots.enter_one_by_one"))
            while True:
                raw = remove_outer_quotes(input(t("roots.add_prompt") + ": "))
                if not raw: break
                p = Path(raw).expanduser()
                roots.append(str(p))
        elif choice == "2":
            txt = remove_outer_quotes(input(t("roots.txt_prompt") + ": "))
            try:
                lines = Path(txt).read_text(encoding="utf-8-sig").splitlines()
                roots = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
            except Exception as exc:
                print(t("roots.txt_failed", error=exc))
                return None

        accepted_roots: list[str] = []
        for raw in roots:
            p = Path(raw).expanduser()
            accepted_roots.append(str(p))

        roots = [str(p) for p in sanitize_media_roots(accepted_roots, announce=True)]
        self.config["media_roots"] = roots
        self.config_mgr.save()
        print(localization.tp("roots.saved", len(roots)))
        tool = normalize_path(constants.APP_ROOT)
        for root in roots:
            if normalize_path(Path(root)) != tool and is_relative_to_path(tool, normalize_path(Path(root))): print(t("roots.holds_program", root=root))
        return None

    def configure_excluded_folders(self) -> None:
        while True:
            self._screen(t("excluded.title"))
            print(t("excluded.explain1"))
            print(t("excluded.explain2") + "\n")
            current = [str(name) for name in self.config.get("excluded_folders", [])]
            if current:
                print(t("excluded.list"))
                for number, name in enumerate(current, 1): print(f"  [{number}] {name}")
            else:
                print(t("excluded.nothing"))
            print("\n" + t("excluded.always"))
            print("  - " + t("excluded.always_program", path=constants.APP_ROOT))
            print("  - " + t("excluded.always_system"))
            print("  - " + t("excluded.always_trickplay"))
            print()
            entries = [("1", t("excluded.add"))]
            if current: entries += [("2", t("excluded.remove_some")), ("3", t("excluded.remove_all"))]
            choice = choose(entries + [("0", t("common.back"))], "0")
            if choice == "0": return
            if choice == "1":
                print(t("excluded.enter_one_by_one"))
                while True:
                    raw = input(t("excluded.name_prompt") + ": ").strip()
                    if not raw: break
                    folder_name, problem = parse_excluded_name(raw)
                    if folder_name is None:
                        print(problem)
                        continue
                    if folder_name.casefold() in {x.casefold() for x in current}:
                        print(t("excluded.already"))
                        continue
                    current.append(folder_name)
            elif choice == "2":
                picked = parse_number_selection(input(t("excluded.remove_prompt") + ": "), len(current))
                current = [name for number, name in enumerate(current, 1) if number not in picked]
            elif choice == "3":
                if prompt_yes_no(t("excluded.remove_all_question"), False): current = []
            self.config["excluded_folders"] = current
            self.config_mgr.save()

    # The settings menus. A menu that leads further shows the names only; the last menu of a branch shows every setting with
    # its value ("Name: value"). Choosing one opens a screen for that one setting and returns to the menu, where the new value is shown.

    def _setting_screen(self, title: str, *paragraphs: str, now: Optional[str] = None, blank_after: bool = True) -> None:
        """The screen of one setting: short paragraphs, then "Now: ..." on a line of its own (and an empty line after it,
        unless a list of choices follows, which starts with one)."""
        self._screen(title)
        for text in paragraphs: print(text + "\n")
        if now is not None: print(t("config.now", value=now) + ("\n" if blank_after else ""))

    def _edit_number(self, title: str, text: str, section: dict, key: str, minimum: float, maximum: float, *,
                     integer: bool = True, warn_above: Optional[int] = None, unit: str = "", convert: tuple[float, float] = (1.0, 0.0)) -> None:
        """Ask for one number and save it. `convert` = (factor, offset): the person sees value * factor + offset (a threshold as a percentage)."""
        factor, offset = convert
        shown = float(section[key]) * factor + offset
        self._setting_screen(title, text, now=f"{shown:g}{unit}")
        if integer: section[key] = prompt_int(title, int(section[key]), int(minimum), int(maximum), warn_above=warn_above)
        else: section[key] = (prompt_float(title, shown, minimum, maximum) - offset) / factor
        self.config_mgr.save()

    def _pick_value(self, title: str, text: str, section: dict, key: str, options: list[tuple[str, str, str]], fallback: str) -> None:
        """Choose one of a few values: options = (stored value, short name, line in the list). Enter keeps what is set."""
        current = section.get(key, fallback)
        names = {value: name for value, name, _ in options}
        self._setting_screen(title, text, now=names.get(current, str(current)), blank_after=False)
        keys = {str(number): value for number, (value, _, _) in enumerate(options, 1)}
        default = next((k for k, v in keys.items() if v == current), "1")
        choice = choose([(str(number), line) for number, (_, _, line) in enumerate(options, 1)], default)
        section[key] = keys[choice]
        self.config_mgr.save()

    def configure_trickplay(self) -> None:
        self._menu(lazy("screen.trickplay_settings"), [
            (lazy("screen.preset"), self.configure_preset, False),
            (lazy("screen.image"), self.configure_trickplay_image, False),
            (lazy("screen.key_frames"), self.configure_key_frames, False),
            (lazy("screen.existing_outputs"), self.configure_existing_outputs, False),
        ])

    def configure_preset(self) -> None:
        settings = self.config["trickplay"]

        def load() -> None:
            settings.update(JELLYFIN_IMAGE_PRESET)
            self.config_mgr.save()

        def intro() -> None:
            print(t("preset.explain") + "\n")
            print(t("config.now", value=config_summary.preset(settings)))
        self._menu(lazy("screen.preset"), [(lambda: t("preset.jellyfin_default", image=config_summary.image(JELLYFIN_IMAGE_PRESET)), load, False)], intro=intro)

    def configure_trickplay_image(self) -> None:
        settings = self.config["trickplay"]

        def edit(name: Callable[[], str], explain: Callable[[], str], key: str, low: int, high: int, **kw: Any) -> Callable[[], None]:
            return lambda: self._edit_number(resolve(name), resolve(explain), settings, key, low, high, **kw)
        self._menu(lazy("screen.image"), [
            (lambda: t("image.width_now", value=settings["width"]), edit(lazy("image.width"), lazy("image.width_explain"), "width", 160, 2048, warn_above=1000, unit=" px"), False),
            (lambda: t("image.interval_now", value=f"{float(settings['interval_seconds']):g}"), edit(lazy("image.interval"), lazy("image.interval_explain"), "interval_seconds", 1, 300, integer=False, unit=" s"), False),
            (lambda: t("image.columns_now", value=settings["tile_columns"]), edit(lazy("image.columns"), lazy("image.columns_explain"), "tile_columns", 1, 50, warn_above=20), False),
            (lambda: t("image.rows_now", value=settings["tile_rows"]), edit(lazy("image.rows"), lazy("image.rows_explain"), "tile_rows", 1, 50, warn_above=20), False),
            (lambda: t("image.qscale_now", value=settings["ffmpeg_qscale"]), edit(lazy("image.qscale"), lazy("image.qscale_explain"), "ffmpeg_qscale", 1, 31), False),
            (lambda: t("image.quality_now", value=settings["jpeg_quality"]), edit(lazy("image.quality"), lazy("image.quality_explain"), "jpeg_quality", 1, 100), False),
        ], intro=lambda: print(t("image.intro")))

    def configure_key_frames(self) -> None:
        settings = self.config["trickplay"]

        def on_off(value: bool) -> str: return t("common.on") if value else t("common.off")

        def only() -> None:
            self._setting_screen(
                t("keyframes.only"),
                t("keyframes.only_explain1"),
                t("keyframes.only_explain2"),
                now=on_off(settings.get("keyframe_only", False)))
            settings["keyframe_only"] = prompt_yes_no(t("keyframes.only_question"), bool(settings.get("keyframe_only", False)))
            self.config_mgr.save()

        def switch() -> None:
            self._edit_number(t("keyframes.switch"), t("keyframes.switch_explain"),
                              settings, "keyframe_repeat_limit_percent", 0, 100, integer=False, unit=" %")

        def items() -> list:
            rows: list = [(lambda: t("keyframes.only_now", value=on_off(settings.get("keyframe_only", False))), only, False)]
            if settings.get("keyframe_only", False):
                limit = float(settings.get("keyframe_repeat_limit_percent", 10.0))
                rows.append((lambda: t("keyframes.switch_now", value=t("keyframes.switch_limit", limit=f"{limit:g}") if limit > 0 else t("keyframes.switch_never")), switch, False))
            return rows
        self._menu(lazy("screen.key_frames"), items, intro=lambda: print(t("keyframes.intro")))

    def configure_existing_outputs(self) -> None:
        settings = self.config["trickplay"]
        same = lambda: [("skip_valid", t("existing.skip_valid"), t("existing.skip_valid_line")), ("rebuild_all", t("existing.rebuild_all"), t("existing.rebuild_all_line"))]
        other = lambda: [("keep_add", t("existing.keep_add"), t("existing.keep_add_line")),
                         ("replace", t("existing.replace"), t("existing.replace_line"))]
        words = lambda options, key, fallback: next((name for value, name, _ in options() if value == settings.get(key, fallback)), str(settings.get(key)))
        self._menu(lazy("screen.existing_outputs"), [
            (lambda: t("existing.same_now", value=words(same, "same_width_policy", "skip_valid")),
             lambda: self._pick_value(t("existing.same"), t("existing.same_explain"), settings, "same_width_policy", same(), "skip_valid"), False),
            (lambda: t("existing.other_now", value=words(other, "other_width_policy", "keep_add")),
             lambda: self._pick_value(t("existing.other"), t("existing.other_explain"), settings, "other_width_policy", other(), "keep_add"), False),
        ], intro=lambda: print(t("existing.intro")))

    # --- Processing ---------------------------------------------------------------------------------------

    def configure_processing(self) -> None:
        self._menu(lazy("screen.processing"), [
            (lazy("screen.acceleration"), self.configure_acceleration, False),
            (lazy("screen.storage_workers"), self.configure_storage_workers, False),
            (lazy("screen.fingerprint"), self.configure_fingerprint, False),
        ])

    def configure_acceleration(self) -> None:
        a = self.config["acceleration"]

        def backends() -> list[str]:
            return list(self._require_tools().backends()) if self.tools else []

        def pick_backend() -> None:
            found = backends()
            options = [("auto", t("accel.auto"), t("accel.choose_automatically"))] + [(b, b, b) for b in found]
            self._pick_value(t("accel.backend"), t("accel.backend_explain"), a, "preferred_backend", options, "auto")

        def items() -> list:
            rows: list = [(lambda: t("accel.mode_now", value=str(a.get("mode", "auto"))), lambda: self._pick_value(
                t("accel.mode"), t("accel.mode_explain"), a, "mode",
                [("auto", "auto", t("accel.mode_auto_line")), ("gpu", "gpu", t("accel.mode_gpu_line")), ("cpu", "cpu", t("accel.mode_cpu_line"))], "auto"), False)]
            if backends(): rows.append((lambda: t("accel.backend_now", value=str(a.get("preferred_backend", "auto"))), pick_backend, False))
            rows.append((lambda: t("accel.gpu_jobs_now", value=a.get("max_gpu_jobs", 2)), lambda: self._edit_number(
                t("accel.gpu_jobs"), t("accel.gpu_jobs_explain"), a, "max_gpu_jobs", 1, 16, warn_above=4), False))
            rows.append((lambda: t("accel.threshold_now", value=f"{(float(a.get('gpu_speed_threshold', 1.15)) - 1) * 100:g}"), lambda: self._edit_number(
                t("accel.threshold"), t("accel.threshold_explain"), a, "gpu_speed_threshold", 1, 200,
                integer=False, unit=" %", convert=(100.0, -100.0)), False))
            stored = a.get("path_choice")
            if path_choice.describe(stored) and isinstance(stored, dict):
                rows.append((lambda: t("accel.saved_choice_now", value=path_choice.summary(stored["results"])), lambda: self._forget_screen(
                    a, "path_choice", t("accel.saved_choice"), t("accel.saved_choice_explain"),
                    path_choice.describe(stored), t("accel.forget_choice")), False))
            for key, name_label, describe in (("gpu_decoding", lazy("accel.saved_decoding"), hardware_decode.describe),
                                                ("hdr_validation", lazy("accel.saved_hdr"), hdr_validation.describe)):
                validation = a.get(key)
                results = validation.get("results") if isinstance(validation, dict) else None
                if results and isinstance(validation, dict):
                    rows.append((lambda name_label=name_label, results=results: t("accel.saved_rated", name=name_label(), count=sum(len(kinds) for kinds in results.values())),
                                 lambda key=key, name_label=name_label, describe=describe, results=results: self._forget_screen(
                        a, key, name_label(), t("accel.saved_explain"), describe(results), t("accel.forget_results")), False))
            return rows
        self._menu(lazy("screen.acceleration"), items, intro=lambda: print(t("accel.intro")))

    def _forget_screen(self, acceleration: dict, key: str, title: str, text: str, lines: list[str], question: str) -> None:
        """What was saved by a check, and a way to forget it."""
        self._setting_screen(title, text)
        for line in lines: print(f"  {line}")
        stored = acceleration.get(key)
        if self.tools and isinstance(stored, dict) and not hdr_validation.same_system(stored, self._require_tools().candidate.sha256, list(self._require_tools().gpu_names)):
            print("\n  " + t("accel.other_system"))
        if prompt_yes_no("\n" + question, False):
            acceleration[key] = None
            self.config_mgr.save()

    def configure_storage_workers(self) -> None:
        w = self.config["workers"]
        def disks() -> str:
            mode = w.get("storage_mode", "auto")
            groups = len(w.get("manual_groups", {}) or {})
            if mode == "auto": return t("disks.auto")
            if mode == "drive": return t("disks.drive")
            return localization.tp("disks.manual", groups)

        def pick_disks() -> None:
            self._setting_screen(t("disks.title"), t("disks.explain"), now=disks(), blank_after=False)
            c = choose([("1", t("disks.pick_auto")), ("2", t("disks.pick_drive")),
                        ("3", t("disks.pick_manual"))], {"auto": "1", "drive": "2", "manual": "3"}.get(w.get("storage_mode", "auto"), "1"))
            w["storage_mode"] = {"1": "auto", "2": "drive", "3": "manual"}[c]
            if c == "3":
                roots = [str(x) for x in self.config.get("media_roots", [])]
                if not roots:
                    print("\n" + t("msg.no_roots"))
                    w["storage_mode"] = "auto"
                    self.config_mgr.save()
                    pause()
                    return
                print("\n" + t("disks.assign"))
                groups: dict[str, list[str]] = {}
                for idx, root in enumerate(roots, 1):
                    default_group = f"disk{idx}"
                    raw_group = input(t("disks.group_prompt", root=root, default=default_group) + ": ")
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_group).strip("_") or default_group
                    groups.setdefault(safe_name, []).append(root)
                w["manual_groups"] = groups
            self.config_mgr.save()

        auto_values = {"hdd": storage_workers.AUTO_WORKERS["hdd"], "ssd": storage_workers.AUTO_WORKERS["ssd"]}

        def per_disk() -> str:
            value = w.get("per_storage_group", storage_workers.AUTO)
            return t("disks.per_disk_auto", **auto_values) if storage_workers.is_auto(value) else str(value)

        def pick_per_disk() -> None:
            automatic = storage_workers.is_auto(w.get("per_storage_group", storage_workers.AUTO))
            self._setting_screen(t("disks.per_disk"), t("disks.per_disk_explain"), now=per_disk(), blank_after=False)
            c = choose([("1", t("disks.pick_workers_auto", **auto_values)), ("2", t("disks.pick_workers_number"))], "1" if automatic else "2")
            if c == "1": w["per_storage_group"] = storage_workers.AUTO
            else:
                current = storage_workers.per_disk(w.get("per_storage_group"), "ssd") if not automatic else storage_workers.AUTO_WORKERS["ssd"]
                w["per_storage_group"] = prompt_int(t("disks.per_disk"), current, storage_workers.MIN_WORKERS, storage_workers.MAX_WORKERS, warn_above=2)
            self.config_mgr.save()

        self._menu(lazy("screen.storage_workers"), [
            (lambda: t("disks.now", value=disks()), pick_disks, False),
            (lambda: t("disks.per_disk_now", value=per_disk()), pick_per_disk, False),
        ], intro=lambda: print(t("disks.intro")))

    def configure_fingerprint(self) -> None:
        source = self.config["source_detection"]

        def toggle() -> None:
            source["strong_fingerprint"] = not bool(source.get("strong_fingerprint", False))
            self.config_mgr.save()

        def intro() -> None:
            print(t("fingerprint.intro"))
        self._menu(lazy("screen.fingerprint"), [
            (lambda: t("fingerprint.now", value=config_summary.fingerprint(source)), toggle, False)], intro=intro)

    # --- Library and the top of the tree ------------------------------------------------------------------

    def configure_library(self) -> None:
        self._menu(lazy("screen.library"), [
            (lazy("roots.title"), self.configure_roots, True),
            (lazy("excluded.title"), self.configure_excluded_folders, False),
        ])

    def configure_language(self, *, first_time: bool = False) -> None:
        """The languages found; `first_time`: the first choice ends the menu (nobody wants a second Enter on the very first start)."""
        loaded = localization.manager()

        def intro() -> None:
            print(t("language.intro") + "\n")
            print(t("language.now", name=next((item["Name"] for item in loaded.discover() if item["Locale"] == loaded.active_locale), loaded.active_locale)))
            if loaded.problems: print("\n" + t("language.file_problems", count=len(loaded.problems), code=loaded.active_locale))
            if len(loaded.discover()) < 2: print("\n" + t("language.no_other", folder=display_path(LOCALES_DIR)))

        def pick(code: str) -> Callable[[], None]:
            def choose_it() -> Any:
                localization.activate_configured(code)
                self.config["language"] = code
                self.config_mgr.save()
                return LEAVE_MENU if first_time else None
            return choose_it

        self._menu(lazy("screen.language"), lambda: [(t("language.entry", name=item["Name"], code=item["Locale"]), pick(item["Locale"]), False) for item in loaded.discover()],
                   lazy("common.back"), intro=intro)

    def offer_language_first_time(self) -> None:
        """On the very first start, when there is a choice, the person picks the language before anything else."""
        if self.config_mgr.fresh and len(localization.manager().discover()) > 1: self.configure_language(first_time=True)

    def configuration_menu(self) -> None:
        self._menu(lazy("screen.configuration"), [
            (lazy("screen.library"), self.configure_library, False),
            (lazy("screen.trickplay_settings"), self.configure_trickplay, False),
            (lazy("screen.processing"), self.configure_processing, False),
            (lazy("screen.language"), self.configure_language, False),
        ])
