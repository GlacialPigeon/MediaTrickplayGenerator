# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The language of the menus and messages.

Every text the person reads on the screen lives in `locales/<language>-<REGION>.json` and is asked for by key: `t("common.back")`,
`tp("count.videos", 3)`. `locales/en-US.json` is the default and must contain every key the program uses; any other language file
may hold as few as one key. A key that a language does not have (or has left empty) is shown in English, so a translation can be
started with a single entry and finished later. Logs, reports and other files are always written in English: code that makes text
for them runs inside `with english():`.

The only sentence in the code itself is the one that says the default language file is missing.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import string
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .activity_log import ACTIVITY
from .constants import LOCALES_DIR

CANONICAL_LOCALE = "en-US"
MISSING_DEFAULT_MESSAGE = "ERROR: Default language file en-US.json is missing or invalid."
PLURAL_CATEGORIES = ("one", "two", "few", "many", "other")
LOCALES: Optional["LocaleManager"] = None
_FORCE_ENGLISH: contextvars.ContextVar[bool] = contextvars.ContextVar("force_english", default=False)


class LocaleError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise LocaleError(f"Locale file not found: {path}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result: raise LocaleError(f"Duplicate key: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8-sig") as fh: data = json.load(fh, object_pairs_hook=no_duplicates)
    except LocaleError: raise
    except Exception as exc: raise LocaleError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(data, dict): raise LocaleError(f"The root of {path.name} must be a JSON object")
    return data


def template_fields(value: str) -> set[str]:
    """The names of the {fields} in a text. Only named fields are allowed: t() passes keyword arguments."""
    fields: set[str] = set()
    try:
        for _, name, _, _ in string.Formatter().parse(value):
            if name is None: continue
            base = name.split(".", 1)[0].split("[", 1)[0]
            if not base or base.isdigit(): raise LocaleError(f"Positional placeholders are not allowed (use named fields): {value!r}")
            fields.add(base)
    except LocaleError: raise
    except Exception as exc: raise LocaleError(f"Invalid text template {value!r}: {exc}") from exc
    return fields


def _check_entry(key: str, value: Any) -> None:
    """Raises LocaleError unless `value` is a text or a plural object (category -> text) with consistent fields."""
    if isinstance(value, str):
        template_fields(value)
        return
    if isinstance(value, dict) and value:
        reference: Optional[set[str]] = None
        for category, template in value.items():
            if category not in PLURAL_CATEGORIES: raise LocaleError(f"Not a plural category: {category!r}")
            if not isinstance(template, str): raise LocaleError(f"Plural form {category!r} must be a text")
            if template == "": continue
            fields = template_fields(template)
            if reference is None: reference = fields
            elif fields != reference: raise LocaleError("The plural forms use different {fields}")
        return
    raise LocaleError("A text, or a plural object, is expected")


def plural_category(locale: str, count: int | float) -> str:
    try: n = int(count)
    except Exception: return "other"
    language = str(locale).split("-", 1)[0].casefold()
    mod10, mod100 = n % 10, n % 100
    if language == "sl":
        return "one" if mod100 == 1 else "two" if mod100 == 2 else "few" if mod100 in (3, 4) else "other"
    if language == "pl":
        if n == 1: return "one"
        return "few" if mod10 in (2, 3, 4) and mod100 not in (12, 13, 14) else "many"
    if language in {"hr", "bs", "sr", "ru", "uk"}:
        if mod10 == 1 and mod100 != 11: return "one"
        if mod10 in (2, 3, 4) and mod100 not in (12, 13, 14): return "few"
        return "many" if language in {"ru", "uk"} else "other"
    if language == "cs" or language == "sk":
        return "one" if n == 1 else "few" if n in (2, 3, 4) else "other"
    if language in {"fr", "pt"}: return "one" if n in (0, 1) else "other"
    if language in {"ja", "zh", "ko", "vi", "th", "id"}: return "other"
    return "one" if n == 1 else "other"


class LocaleManager:
    def __init__(self, locales_dir: Path = LOCALES_DIR) -> None:
        self.dir = Path(locales_dir)
        self.canonical: dict[str, Any] = {}
        self.active: dict[str, Any] = {}
        self.active_locale = CANONICAL_LOCALE
        self.problems: list[str] = []            # what was ignored in the active language file

    @property
    def canonical_path(self) -> Path: return self.dir / f"{CANONICAL_LOCALE}.json"

    def load_canonical(self) -> None:
        data = _load_json(self.canonical_path)
        for key, value in data.items():
            if key.startswith("_meta."):
                if not isinstance(value, str) or not value.strip(): raise LocaleError(f"{key} must be a text")
                continue
            if value == "" or value == {}: raise LocaleError(f"The default language must not leave a text empty: {key}")
            try: _check_entry(key, value)
            except LocaleError as exc: raise LocaleError(f"{key}: {exc}") from exc
            if isinstance(value, dict) and "other" not in value: raise LocaleError(f"{key}: a plural text needs the form 'other'")
        if data.get("_meta.locale") != CANONICAL_LOCALE: raise LocaleError(f"_meta.locale of the default language must be {CANONICAL_LOCALE}")
        for needed in ("_meta.name", "_meta.version"):
            if needed not in data: raise LocaleError(f"{needed} is missing in {self.canonical_path.name}")
        self.canonical = data
        self.active, self.active_locale, self.problems = data, CANONICAL_LOCALE, []

    def _read_override(self, code: str) -> tuple[dict[str, Any], list[str]]:
        """The usable entries of a language file and what was ignored. Raises LocaleError when the file as a whole cannot be used."""
        data = _load_json(self.dir / f"{code}.json")
        if data.get("_meta.locale") != code: raise LocaleError(f"_meta.locale must be {code!r} in {code}.json")
        if not isinstance(data.get("_meta.name"), str) or not data["_meta.name"].strip(): raise LocaleError(f"_meta.name is missing in {code}.json")
        clean: dict[str, Any] = {}
        problems: list[str] = []
        for key, value in data.items():
            if key.startswith("_meta."):
                if isinstance(value, str) and value: clean[key] = value
                continue
            reference = self.canonical.get(key)
            if reference is None:
                problems.append(f"{key}: not a text of the program (ignored)")
                continue
            if value == "" or value == {}: continue                                  # not translated: English is shown
            try:
                _check_entry(key, value)
                if isinstance(reference, str) != isinstance(value, str): raise LocaleError("the kind of text differs from the English one")
                want = template_fields(reference if isinstance(reference, str) else str(reference["other"]))
                for template in ([value] if isinstance(value, str) else [v for v in value.values() if v]):
                    if template_fields(template) != want: raise LocaleError(f"{{fields}} must be {sorted(want)}")
            except LocaleError as exc:
                problems.append(f"{key}: {exc} (ignored)")
                continue
            clean[key] = value if isinstance(value, str) else {k: v for k, v in value.items() if v}
        return clean, problems

    def activate(self, code: str) -> bool:
        """Use the language `code`. False (and English) when its file is missing or unusable; single bad entries only fall back to English."""
        code = str(code or CANONICAL_LOCALE)
        if code == CANONICAL_LOCALE:
            self.active, self.active_locale, self.problems = self.canonical, CANONICAL_LOCALE, []
            return True
        try:
            clean, problems = self._read_override(code)
        except LocaleError as exc:
            self.active, self.active_locale, self.problems = self.canonical, CANONICAL_LOCALE, []
            ACTIVITY.log("LOCALE", f"language {code} not used, English instead: {exc}")
            return False
        self.active, self.active_locale, self.problems = clean, code, problems
        if problems: ACTIVITY.log("LOCALE", f"language {code}: {len(problems)} entries ignored")
        return True

    def discover(self) -> list[dict[str, str]]:
        """The languages that can be chosen: [{"Locale": "sl-SI", "Name": "Slovenščina"}, ...], English first."""
        found = [{"Locale": CANONICAL_LOCALE, "Name": str(self.canonical.get("_meta.name", "English"))}]
        if not self.dir.is_dir(): return found
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.name.casefold()):
            if path.stem == CANONICAL_LOCALE: continue
            try:
                data = _load_json(path)
                if data.get("_meta.locale") != path.stem or not isinstance(data.get("_meta.name"), str) or not data["_meta.name"].strip():
                    raise LocaleError("_meta.locale or _meta.name is wrong")
                found.append({"Locale": path.stem, "Name": data["_meta.name"]})
            except LocaleError as exc:
                ACTIVITY.log("LOCALE", f"language file {path.name} ignored: {exc}")
        return found

    def check(self, code: str) -> tuple[list[str], int, int]:
        """(problems, translated, total) for a language file; raises LocaleError when the file as a whole cannot be used."""
        total = sum(1 for key in self.canonical if not key.startswith("_meta."))
        if code == CANONICAL_LOCALE: return [], total, total
        clean, problems = self._read_override(code)
        return problems, sum(1 for key in clean if not key.startswith("_meta.")), total

    def _entry(self, key: str) -> Any:
        if not _FORCE_ENGLISH.get():
            value = self.active.get(key)
            if value: return value
        if key in self.canonical: return self.canonical[key]
        raise LocaleError(f"Unknown text key: {key}")

    def text(self, key: str, **values: Any) -> str:
        entry = self._entry(key)
        if not isinstance(entry, str):
            entry = self.canonical[key]
            if not isinstance(entry, str): raise LocaleError(f"{key} is a plural text: use tp()")
        try: return entry.format(**values)
        except Exception as exc: raise LocaleError(f"Could not fill in {key}: {exc}") from exc

    def plural(self, key: str, count: int | float, **values: Any) -> str:
        canonical = self.canonical.get(key)
        if not isinstance(canonical, dict): raise LocaleError(f"{key} is not a plural text")
        forced = _FORCE_ENGLISH.get()
        locale = CANONICAL_LOCALE if forced else self.active_locale
        mine = {} if forced else self.active.get(key)
        category = plural_category(locale, count)
        template = (mine.get(category) or mine.get("other")) if isinstance(mine, dict) else None
        if template is None: template = canonical.get(plural_category(CANONICAL_LOCALE, count)) or canonical["other"]
        shown = number(count) if isinstance(count, int) else count
        try: return template.format(**{"count": shown, **values})
        except Exception as exc: raise LocaleError(f"Could not fill in {key}: {exc}") from exc

    def separators(self) -> tuple[str, str]:
        """(thousands, decimal) separator of the language in use; the language file may set _meta.thousands and _meta.decimal."""
        source = self.canonical if _FORCE_ENGLISH.get() else {**self.canonical, **self.active}
        return str(source.get("_meta.thousands", ",")), str(source.get("_meta.decimal", "."))


def initialize() -> LocaleManager:
    global LOCALES
    manager = LocaleManager()
    manager.load_canonical()
    LOCALES = manager
    return manager


def reset() -> None:
    """Forget what was loaded (for tests); the next t() loads the default language again."""
    global LOCALES
    LOCALES = None


def manager() -> LocaleManager:
    return LOCALES if LOCALES is not None else initialize()


def activate_configured(code: str) -> None:
    """Use the language from the configuration; English when it is not available."""
    loaded = manager()
    wanted = str(code or CANONICAL_LOCALE)
    if wanted == "en": wanted = CANONICAL_LOCALE                   # what older configurations hold
    if not loaded.activate(wanted): ACTIVITY.log("LOCALE", f"language {wanted} is not available: English")


def t(key: str, **values: Any) -> str:
    return manager().text(key, **values)


def tp(key: str, count: int | float, **values: Any) -> str:
    return manager().plural(key, count, **values)


def lazy(key: str, **values: Any) -> Callable[[], str]:
    """A text that is made when it is shown (and so in the language of that moment): for menu titles and labels, which are
    drawn again after the language was changed, and which the activity log writes in English."""
    return lambda: t(key, **values)


@contextlib.contextmanager
def english() -> Iterator[None]:
    """Everything made inside is English whatever language is chosen: for text that goes into logs and reports."""
    token = _FORCE_ENGLISH.set(True)
    try: yield
    finally: _FORCE_ENGLISH.reset(token)


def number(value: int | float, places: Optional[int] = None) -> str:
    """A number with the thousands and decimal separators of the language: 10,874 in English, 10.874 in Slovenian."""
    thousands, decimal = manager().separators()
    text = f"{value:,.{places}f}" if places is not None else f"{value:,}"
    return text.replace(",", "\0").replace(".", decimal).replace("\0", thousands)


def check_command(code: str, locales_dir: Path = LOCALES_DIR) -> int:
    """`--check-language CODE`: what is missing or wrong in a language file (exit code 0 when nothing is wrong)."""
    loaded = LocaleManager(locales_dir)
    try: loaded.load_canonical()
    except LocaleError as exc:
        print(MISSING_DEFAULT_MESSAGE)
        print(exc)
        return 2
    try: problems, translated, total = loaded.check(code)
    except LocaleError as exc:
        print(f"{code}: {exc}")
        return 1
    print(f"{code}: {translated} of {total} texts translated; the rest is shown in English.")
    for line in problems: print(f"  {line}")
    return 1 if problems else 0
