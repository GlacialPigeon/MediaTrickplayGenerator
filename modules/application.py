# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import traceback
from typing import Sequence

from . import localization
from .cli import EXIT_CANCELED, EXIT_CRASH, EXIT_NOT_RUN, EXIT_OK, parse_args
from .activity_log import ACTIVITY
from .constants import APP_NAME, APP_VERSION, DIRS
from .datetime_helpers import local_stamp
from .localization import t
from .file_io import atomic_write_text, ensure_directories
from .maintenance import LOG_RETENTION_DAYS, prune_old_logs
from .platform_api import PlatformAdapter, configure_platform


def run_application(adapter: PlatformAdapter, *, platform_name: str, argv: Sequence[str] = ()) -> int:
    args = parse_args(argv)  # before anything is touched: --help, --version and wrong arguments end here
    if args.command == "check-language": return localization.check_command(args.language)
    try: localization.initialize()
    except localization.LocaleError as exc:
        print(localization.MISSING_DEFAULT_MESSAGE)
        print(exc)
        return EXIT_NOT_RUN
    configure_platform(adapter)
    adapter.configure_console()
    ensure_directories()

    lock = adapter.create_instance_lock(DIRS["data"] / "application.lock")
    if not lock.acquire():
        if lock.failure in (None, "busy"):
            print(t("app.already_running"))
        else:
            print(t("app.lock_failed", name=APP_NAME, reason=lock.failure))
            print(t("app.lock_file", path=lock.path))
        return EXIT_NOT_RUN

    from .ui import Application  # imported after the platform adapter is configured

    app = None
    code = EXIT_CRASH
    session = ACTIVITY.start("interactive" if args.interactive else f"{args.command}" + (" --dry-run" if args.dry_run else ""), platform=platform_name)
    try:
        pruned = prune_old_logs(keep=ACTIVITY.folder)
        if pruned: ACTIVITY.log("MAINTENANCE", f"removed {pruned} log item(s) older than {LOG_RETENTION_DAYS} days")
        app = Application()
        if not app.startup_recovery(interactive=args.interactive):
            code = EXIT_NOT_RUN
            return code
        if args.interactive:
            app.offer_language_first_time()
            app.main_menu()
            code = EXIT_OK
            return code
        code = app.run_command(args)
        return code
    except (KeyboardInterrupt, EOFError):
        print("\n" + t("menu.canceled"))
        code = EXIT_CANCELED
        return code
    except Exception as exc:
        ACTIVITY.log("CRASH", f"{type(exc).__name__}: {exc}")
        text = f"{APP_NAME} {APP_VERSION} ({platform_name})\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        crash = ACTIVITY.path or (DIRS["logs"] / f"crash_{local_stamp()}.log")
        if ACTIVITY.path is not None:
            ACTIVITY.write_raw(text)                                  # the traceback goes into the log of this start
        else:
            try: atomic_write_text(crash, text)
            except Exception: pass
        print("\n" + t("app.fatal", kind=type(exc).__name__, error=exc))
        print(t("app.crash_log", path=crash))
        return EXIT_CRASH
    finally:
        if app is not None:
            app.shutdown()
        ACTIVITY.log("EXIT", f"exit code {code}" + (f" | session log: {session}" if session else ""))
        ACTIVITY.close()
        lock.release()
