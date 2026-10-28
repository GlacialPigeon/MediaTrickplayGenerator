# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Languages: every key the program uses is in en-US.json, other languages may be partial, logs and reports stay English."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules import localization as loc
from modules.constants import APP_ROOT
from tests.test_screens import ScreenTest

CANONICAL = {
    "_meta.code": "en", "_meta.locale": "en-US", "_meta.name": "English", "_meta.version": "1",
    "common.back": "Back", "screen.language": "Language", "found": "{n} found",
    "count.videos": {"one": "{count} video", "other": "{count} videos"},
}


def write(folder: Path, name: str, data: dict) -> None:
    (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def sl(**texts) -> dict:
    return {"_meta.code": "sl", "_meta.locale": "sl-SI", "_meta.name": "Slovenščina", **texts}


class Keys(unittest.TestCase):
    """The keys are in the code; the default language file has all of them (and no others)."""

    def used_keys(self) -> tuple[set[str], list[str]]:
        used: set[str] = set()
        dynamic: list[str] = []
        for path in sorted((APP_ROOT / "modules").rglob("*.py")):
            if path.name == "localization.py": continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            passing_on = {id(call) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "say"
                          for call in ast.walk(fn)}                     # say() hands the key it was given to t() and tp()
            for node in ast.walk(tree):
                if id(node) in passing_on: continue
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) in {"t", "tp", "lazy", "say"} and node.args:
                    name = getattr(node.func, "id", getattr(node.func, "attr", None))
                    first = node.args[0]
                    if name == "say" and len(node.args) > 1 and isinstance(first, ast.Constant) and "." not in str(first.value): first = node.args[1]       # logger.say(level, key)
                    if isinstance(first, ast.Constant) and isinstance(first.value, str): used.add(first.value)
                    else: dynamic.append(f"{path.name}:{node.lineno}")
        return used, dynamic

    def test_the_default_language_has_every_key_the_code_asks_for(self) -> None:
        canonical = json.loads((APP_ROOT / "locales" / "en-US.json").read_text(encoding="utf-8-sig"))
        used, dynamic = self.used_keys()
        self.assertEqual(sorted(used - canonical.keys()), [])
        self.assertEqual(dynamic, [], "keys must be written out in the call so they can be checked")

    def test_the_default_language_has_no_key_the_code_does_not_use(self) -> None:
        canonical = json.loads((APP_ROOT / "locales" / "en-US.json").read_text(encoding="utf-8-sig"))
        used, _ = self.used_keys()
        self.assertEqual(sorted(k for k in canonical if not k.startswith("_meta.")and k not in used), [])

    def test_the_english_file_loads(self) -> None:
        manager = loc.LocaleManager(APP_ROOT / "locales")
        manager.load_canonical()
        self.assertTrue(manager.activate("en-US"))
        self.assertEqual(manager.problems, [])


class Behaviour(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        write(self.dir, "en-US", CANONICAL)
        self.manager = loc.LocaleManager(self.dir)
        self.manager.load_canonical()
        patcher = mock.patch.object(loc, "LOCALES", self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_language_with_a_single_key_works_and_the_rest_is_english(self) -> None:
        write(self.dir, "sl-SI", sl(**{"common.back": "Nazaj"}))
        self.assertTrue(self.manager.activate("sl-SI"))
        self.assertEqual(loc.t("common.back"), "Nazaj")
        self.assertEqual(loc.t("screen.language"), "Language")
        self.assertEqual(loc.t("found", n=3), "3 found")

    def test_an_empty_entry_means_not_translated(self) -> None:
        write(self.dir, "sl-SI", sl(**{"common.back": "", "screen.language": "Jezik", "count.videos": {}}))
        self.manager.activate("sl-SI")
        self.assertEqual(loc.t("common.back"), "Back")
        self.assertEqual(loc.t("screen.language"), "Jezik")
        self.assertEqual(self.manager.problems, [])

    def test_a_wrong_entry_is_ignored_alone_and_reported(self) -> None:
        write(self.dir, "sl-SI", sl(**{"common.back": "Nazaj", "found": "{stevilo} najdenih", "nonsense": "x", "screen.language": "{} Jezik"}))
        self.assertTrue(self.manager.activate("sl-SI"))
        self.assertEqual(loc.t("common.back"), "Nazaj")
        self.assertEqual(loc.t("found", n=2), "2 found", "the placeholders did not match: English instead")
        self.assertEqual(loc.t("screen.language"), "Language")
        self.assertEqual(len(self.manager.problems), 3)
        problems, translated, total = self.manager.check("sl-SI")
        self.assertEqual((translated, total, len(problems)), (1, 4, 3))

    def test_a_missing_or_wrong_file_falls_back_to_english(self) -> None:
        self.assertFalse(self.manager.activate("de-DE"))
        write(self.dir, "fr-FR", sl(**{"common.back": "Retour"}))                 # _meta.locale says sl-SI, the name says fr-FR
        self.assertFalse(self.manager.activate("fr-FR"))
        (self.dir / "xx-XX.json").write_text("{ not json", encoding="utf-8")
        self.assertFalse(self.manager.activate("xx-XX"))
        self.assertEqual(loc.t("common.back"), "Back")
        self.assertEqual([item["Locale"] for item in self.manager.discover()], ["en-US"])

    def test_discover_lists_every_usable_language_file(self) -> None:
        write(self.dir, "sl-SI", sl())
        write(self.dir, "de-DE", {"_meta.code": "de", "_meta.locale": "de-DE", "_meta.name": "Deutsch"})
        self.assertEqual([(i["Locale"], i["Name"]) for i in self.manager.discover()], [("en-US", "English"), ("de-DE", "Deutsch"), ("sl-SI", "Slovenščina")])

    def test_plural_forms_follow_the_language_and_fall_back(self) -> None:
        write(self.dir, "sl-SI", sl(**{"count.videos": {"one": "{count} video", "two": "{count} videa", "few": "{count} videi", "other": "{count} videov"}}))
        self.manager.activate("sl-SI")
        self.assertEqual([loc.tp("count.videos", n) for n in (1, 2, 3, 5, 101, 102)],
                         ["1 video", "2 videa", "3 videi", "5 videov", "101 video", "102 videa"])
        write(self.dir, "de-DE", {"_meta.code": "de", "_meta.locale": "de-DE", "_meta.name": "Deutsch", "count.videos": {"other": "{count} Videos"}})
        self.manager.activate("de-DE")
        self.assertEqual(loc.tp("count.videos", 1), "1 Videos", "only the form 'other' was given")
        self.manager.activate("en-US")
        self.assertEqual((loc.tp("count.videos", 1), loc.tp("count.videos", 0), loc.tp("count.videos", 2)), ("1 video", "0 videos", "2 videos"))

    def test_logs_and_reports_are_made_in_english(self) -> None:
        write(self.dir, "sl-SI", sl(**{"common.back": "Nazaj", "count.videos": {"one": "{count} video", "two": "{count} videa", "other": "{count} videov"}}))
        self.manager.activate("sl-SI")
        with loc.english():
            self.assertEqual((loc.t("common.back"), loc.tp("count.videos", 2)), ("Back", "2 videos"))
        self.assertEqual((loc.t("common.back"), loc.tp("count.videos", 2)), ("Nazaj", "2 videa"))

    def test_numbers_use_the_separators_of_the_language(self) -> None:
        write(self.dir, "sl-SI", sl(**{"_meta.thousands": ".", "_meta.decimal": ","}))
        self.assertEqual((loc.number(10874), loc.number(1234.5, 2)), ("10,874", "1,234.50"))
        self.manager.activate("sl-SI")
        self.assertEqual((loc.number(10874), loc.number(1234.5, 2), loc.number(7)), ("10.874", "1.234,50", "7"))
        with loc.english():
            self.assertEqual(loc.number(10874), "10,874")

    def test_the_default_language_must_be_complete_and_valid(self) -> None:
        for bad in ({**CANONICAL, "common.back": ""}, {**CANONICAL, "found": "{0} found"}, {**CANONICAL, "count.videos": {"one": "{count} video"}},
                    {k: v for k, v in CANONICAL.items() if k != "_meta.name"}):
            write(self.dir, "en-US", bad)
            with self.assertRaises(loc.LocaleError, msg=bad):
                loc.LocaleManager(self.dir).load_canonical()
        (self.dir / "en-US.json").unlink()
        with self.assertRaises(loc.LocaleError):
            loc.LocaleManager(self.dir).load_canonical()

    def test_check_command_reports_what_is_wrong(self) -> None:
        import contextlib
        import io
        write(self.dir, "sl-SI", sl(**{"common.back": "Nazaj", "nonsense": "x"}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = loc.check_command("sl-SI", self.dir)
        self.assertEqual(code, 1)
        self.assertIn("1 of 4 texts translated", out.getvalue())
        self.assertIn("nonsense", out.getvalue())


class LanguageMenu(ScreenTest):
    def setUp(self) -> None:
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        write(self.dir, "en-US", json.loads((APP_ROOT / "locales" / "en-US.json").read_text(encoding="utf-8-sig")))
        manager = loc.LocaleManager(self.dir)
        manager.load_canonical()
        patcher = mock.patch.object(loc, "LOCALES", manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = manager

    def test_with_only_english_the_menu_says_how_to_add_a_language(self) -> None:
        out = self.feed(self.app.configure_language, "")
        self.assertIn("[1] English (en-US)", out)
        self.assertIn("Now: English", out)
        self.assertIn("No other language was found", out)

    def test_choosing_a_language_uses_it_at_once_and_remembers_it(self) -> None:
        write(self.dir, "sl-SI", sl(**{"common.back": "Nazaj", "screen.language": "Jezik", "language.now": "Zdaj: {name}"}))
        out = self.feed(self.app.configure_language, "2", "0")
        self.assertIn("Now: English", out)
        self.assertIn("Zdaj: Slovenščina", out)
        self.assertIn("[0] Nazaj", out)
        self.assertEqual(self.saved()["language"], "sl-SI")
        self.feed(self.app.configure_language, "1", "0")
        self.assertEqual(self.saved()["language"], "en-US")

    def test_a_language_that_has_gone_is_replaced_by_english(self) -> None:
        loc.activate_configured("sl-SI")
        self.assertEqual(self.manager.active_locale, "en-US")
        loc.activate_configured("en")
        self.assertEqual(self.manager.active_locale, "en-US")

    def test_the_first_start_offers_the_choice_only_when_there_is_one(self) -> None:
        self.app.config_mgr.fresh = True
        self.feed(self.app.offer_language_first_time)                           # only English: nothing is asked
        write(self.dir, "sl-SI", sl())
        self.feed(self.app.offer_language_first_time, "2")                      # the choice ends the menu: no second Enter
        self.assertEqual(self.saved()["language"], "sl-SI")
        self.app.config_mgr.fresh = False
        self.feed(self.app.offer_language_first_time)                           # not the first start


if __name__ == "__main__":
    unittest.main()
