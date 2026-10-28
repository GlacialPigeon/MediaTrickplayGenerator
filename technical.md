# Media Trickplay Generator: technical reference

The short introduction and the setup are in the [README](README.md). This file holds the details: safety rules, logs, every menu, acceleration, the command line and the project layout.

Terminal tool that generates **trickplay sprite sheets** (the seek-bar preview images used by Jellyfin and similar
media servers) next to your videos, safely and repeatably. Runs on **Windows** and **Linux** with Python 3.10+,
FFmpeg/FFprobe and Pillow.

> Version `1.0.0`.


## What it does

For every video it creates, next to the file:

```
Movie.mkv
Movie.trickplay/
    320 - 10x10/
        0.jpg  1.jpg  2.jpg ...
```

Defaults are Jellyfin's own: 320 px thumbnails, one every 10 s, 10x10 grid per sprite, JPEG quality 90, qscale 4 (Configuration > Trickplay > Preset puts them back). All of this is configurable. A configuration that was saved earlier keeps its values.

**Key frames only** (Configuration > Trickplay > Key frames, `trickplay.keyframe_only`, **off** by default): the decoder skips every
picture that is not a key frame (`-skip_frame nokey`, an option Jellyfin has too), so big files need much less decoding. A thumbnail
then comes from the nearest key frame before its time (a second or two off with usual files), and the `fps` filter still fills every
slot, so the number of thumbnails is the same. If a file or decoder cannot do it, that video is extracted with every frame and nothing
is remembered as a failure of the GPU backend (the reason is in the log). A refused colour space tag is dealt with first and the key
frames stay. Where the key frames of a file are further apart than the interval, the `fps` filter repeats thumbnails (every picture is up to a
whole key frame distance too early on the scrubber). After a key-frame extraction the thumbnails that are exactly the one before are
counted; a video with more than `trickplay.keyframe_repeat_limit_percent` of them (default 10, 0 = never) is extracted again with every
frame (`KEYFRAMES_TOO_SPARSE` event), so only files that really need it pay for the full decode. The `GPU_DONE` / `CPU_DONE` events say `keyframe_only`. A video with too many thumbnails for its length is first extracted
again with every frame (see *Suspicious thumbnail counts*).

Jellyfin reads these folders only when the library setting **Save trickplay images next to media** is on (otherwise it looks in its own
metadata folder). Jellyfin keeps no time for a single thumbnail, only the interval, so the number of thumbnails must fit the video.

### Safety rules (the point of the design)

* Output is built in a `.__building__` folder and only swapped in after it validates. A failed or interrupted run never
  damages an existing valid output.
* Replaced outputs are kept as `.__old__.*` until the new one is validated; leftovers are reported by *Verify*.
* On Windows a folder that is briefly in use (antivirus scan, indexer, media server, a console sitting in it) is waited for: renaming it is retried for about 5 seconds before the video is reported as failed. The old output stays untouched either way.
* If a replacement was interrupted and a still valid `.__old__.*` copy exists, *Generate* puts it back (a rename, instantly) instead of building the output again; a damaged output in the way is moved aside, never deleted. If that is not possible the video is generated as usual.
* Old widths are removed only after the new target is complete, and only if the folder contains nothing but `N.jpg`
  files. Anything else needs explicit approval.
* The source is re-checked before finalizing; if it changed during processing the result is discarded.
* Ambiguous situations (two videos with the same stem, unknown files in a trickplay folder) stop or ask — never guess.
* Valid existing output is skipped, so runs are resumable and cheap to repeat. *Generate* trusts an output it wrote itself when
  the file names and sizes still match what it recorded (no picture is read again, which matters on a network drive);
  *Verify* reads every picture.
* FFmpeg that makes no progress for 10 minutes (a drive that stopped answering) is stopped and the video is reported as failed;
  it is tried again at the next run.
* A video file that FFprobe cannot read (damaged, or a read error) is reported - in the summary of *Generate* and *Verify*, in the
  log, and as exit code `1` - and does not just disappear.
* **Suspicious thumbnail counts:** after a video is done, the number of thumbnails is compared with its length (one every
  `interval` seconds; a difference of a few thumbnails or 1 % is normal). A video that does not fit is reported as *suspicious* -
  a `SUSPECT_FRAME_COUNT` event, a warning, the final summary and `summary.json` of the run (`suspect_videos`, `suspect_details`) - and
  *Verify* lists it again at every run (summary line, `problems.txt` of the Verify run, `SUSPECT`). The output is **kept as it is**:
  no thumbnail is ever repeated or invented to fill the gap, it shows what the file really holds, and it does not count as a
  problem (the exit code stays `0`, the video is not rebuilt at the next run). The cause is in the file: a video stream shorter
  than the length the container states (the preview ends early), or damaged time stamps (the FFmpeg `fps` filter then repeats
  or drops pictures). Remux or replace such a file and generate it again.
  A video with **too many** thumbnails gets, before it is reported, **one more extraction** with the time stamps made from the frame
  count and the file's frame rate (`setpts=N/<fps>/TB`, which Jellyfin does for every file); the result that comes closer to the
  expected number is kept (`TIMESTAMPS_NORMALIZED` event). This repairs damaged time stamps. A video with **too few** thumbnails (a
  stream that is simply shorter) is not extracted again: nothing changes that and every try costs a full decode. A normal video is
  never extracted twice.
* Leftover `.__old__.*` / `.__failed_finalize__.*` folders are only removed from *Verify*, after you confirm, and only
  when that cannot lose the last good output (an old copy is kept unless the output that replaced it is valid; a folder
  holding anything other than `N.jpg` files is never touched). Every removal is logged.
* A `.trickplay` folder whose video was deleted, renamed or moved is reported by *Verify* (count in the summary, listed in the
  problem list, no effect on the exit code) and removed only after you confirm - and only if it holds nothing but trickplay
  images; anything else stays. The check is repeated right before removing (a video with that name must still be missing).
  Folders that could not be listed completely, and everything in excluded folders, are never reported.
* If a replacement did not finish and the new output is missing or damaged, *Verify* offers to put the previous output back
  (a rename, not a copy) when that old copy is still a valid output for the current settings. A damaged output in its way is
  moved to `.__failed_finalize__.*`, never deleted; a folder with foreign files or a link is left alone.

## Logs

Every start of the program gets **one folder** in `logs/`, named `<time>_<id>`, with everything that happened during it:

```
logs/2026-10-28_024649_b579f45f/
  session.log            the story of this start, in time order
  01_generate/           every Generate / Verify run of this start has a folder of its own
    summary.json           the result of the run (counts, suspect videos, timings)
    events.jsonl           one line per event that needs attention (failed, rebuilt, ...)
    failed.txt             the videos that failed, with the error (only when there were any)
  02_verify/
    summary.json
    events.jsonl
    problems.txt           the problems Verify found (and the counts, also when there are none)
```

`session.log` holds the start and exit (with the exit code), every menu entry and how long it took, every change of the configuration
(`key: old -> new`), cleaning and resetting in Maintenance, diagnostics (where the report is, how it ended), the HDR ratings that
Generate will follow, and the detailed log of every run (a `[INFO] ... run_id=...` line opens each run; its events and warnings follow;
the line `generate finished | ... | run folder: 01_generate` says where its files are). Command-line runs write the same, and so does a crash
(the traceback is in `session.log`). Nothing is ever overwritten: a second Verify has a folder of its own. A run that finds nothing to
write leaves no folder. The Benchmark checks keep their reports in `diagnostics/<kind>/...`.
Videos that need nothing (valid output already there) are counted in the summary but not listed anywhere.
Folders older than 90 days are removed when the program starts (also the loose log files of earlier versions); Maintenance never deletes
the folder of the start that is running.

## Menu

```
[1] Generate / Resume trickplay    scan, plan, disk estimate, run
[2] Verify existing trickplay      valid / missing / invalid / source changed / settings changed,
                                   stale builds and leftover folders; problems (output that exists but is not valid, leftover folders, orphans, unreadable videos; not missing output) saved to `problems.txt` in the folder of that run.
                                   Afterwards it offers, each only after you confirm: remove the `.trickplay` folders of
                                   videos that no longer exist, remove safe leftover folders, restore a previous output,
                                   regenerate what is missing / invalid / changed
[3] Benchmark
      [1] CPU vs GPU benchmark     takes one file for every kind of video found in your library (kind = codec, bit depth, picture size,
                                   HDR type; the most common kinds first, at most 30) plus 2 random files, at least 10 files in all;
                                   one 2-minute segment from the middle of each. Every segment is copied (stream copy)
                                   into the program's `temp` folder first, tested there and deleted, so the library drive does not
                                   distort the timings; the copy shows how fast that drive reads. Per file the fastest path is run
                                   once more with key frames only (A/B). Then the segments where a GPU backend won (one of every kind
                                   of video first) are run with 1, 2, 3 ... videos at once, to see how much more work the GPU gets
                                   done on this computer and this library. The screen and the report also list the disks of the
                                   library: kind (HDD or SSD), videos at once on each, and how fast the drive read the copies.
                                   Afterwards it suggests `key frames only`, the repeat limit, the number of GPU jobs (never more
                                   than there are videos at once on your disks, and never fewer than are set) and the fastest path per
                                   kind of video; nothing changes unless you confirm. A test folder of your own can be used instead (asked first when media folders exist).
      [2] HDR compatibility check  (library or test folder) one test file for every kind of HDR found (HDR10, HDR10+, HLG, each Dolby Vision
                                   profile); CPU and GPU frames side by side, you rate each kind and path:
                                   looks correct / too bright / too dark / wrong colors
      [3] Backend compatibility check   does every GPU backend really decode on the GPU? One video per codec, hardware frames
                                   must reach FFmpeg's download step, so a silent fall-back to software is detected;
                                   speed is shown next to software decoding. A path that decodes a codec/bit depth on
                                   the CPU after all (e.g. H.264 10-bit on d3d11va) can be skipped by Generate for that
                                   kind of video (saved per FFmpeg build + GPU; forget it in Configuration > Processing > Hardware acceleration).
                                   Kinds the card cannot decode at all are remembered as "not supported" too
[4] Configuration
      [1] Library                  [1] Media folders   [2] Excluded folders
      [2] Trickplay                [1] Preset (Jellyfin default / custom)   [2] Image (width, interval, tiles, quality)
                                   [3] Key frames   [4] Existing outputs (same-width and other-width policy)
      [3] Processing               [1] Hardware acceleration   [2] Videos at once   [3] Detecting changed videos
      [4] Language                 the language of the menus and messages (English, Slovenščina, ...)
[5] Maintenance
      [1] Self-test                platform, Pillow, folders, config, database, FFmpeg, synthetic video
                                   through the real pipeline
      [2] Show disk usage   [3] Delete temporary files   [4] Clean logs and diagnostics (30 / 90 days / all)
      [5] Reset processing history   [6] Forget FFmpeg build history
[0] Exit
```

Menus that lead on show names only; the last menu before a setting lists each setting with its value (for example `[1] Thumbnail width: 320 px`). Choosing an entry asks for the new value and returns to that menu, which shows it at once. Enter at a question keeps the current value.

*Excluded folders* are folder names (for example `extras`, `trailers`), skipped at any depth under the media folders; capitals do not matter. Always skipped, whatever the list says: the Windows system folders `$RECYCLE.BIN` and `System Volume Information`, and the folders of NAS boxes and disks that never hold media: `@eaDir`, `#recycle`, `#snapshot`, `@Recycle`, `.@__thumb`, `lost+found`, `.Trash` and `.Trash-<uid>`. Media folders must be absolute paths; a relative one is refused with a message.

`[0]` leaves the current menu. In submenus Enter alone also goes back (`Choice [0]:`); the main menu has no default, so an empty answer there only redraws the screen.

Every screen starts with the same header: the program name, and under a thin line where you are (for example `Configuration > Trickplay settings > Key frames`). Settings return to their menu without an extra Enter. **Ctrl+C** in a submenu goes one level back, during an action it cancels only that action (back in the menu), and in the main menu it ends the program; during Generate the first Ctrl+C finishes the running videos, the second stops at once.

Paths inside the program folder (reports, logs, `temp`, `ffmpeg`) are shown relative to it on the screen; the folder itself is named under Maintenance > Show disk usage. Paths elsewhere (your media) and everything written to files and logs stay in full.

While Generate runs, the screen shows one header line (`Generating: 5 of 10,874 done | 0 failed | 2 active | 01:47 elapsed | about 3 h 20 min left`) and one line for each video being made: disk, file name, `GPU cuda` or `CPU`, percent, speed and time left, in fixed columns. The lines are refreshed twice a second in place, so they do not flicker; a failure stays above them. With more videos at once than fit on the screen the rest is counted (`+3 more`). The time left follows the length of the videos finished so far; the preflight shows a first estimate from the benchmark's speeds.

The benchmark prints a short result on the screen (speed per kind of video, the suggested settings); every measurement is in the report file named at the end (`result.txt`). The HDR check first looks for HDR videos and asks Quick or Full only when it found some. Verify shows only a progress counter and the summary; every problem is in `problems.txt` in the folder of that run (under `logs/`).

## Acceleration

* **CPU** always works and is the reference.
* **GPU** (optional): `cuda`, `qsv`; `d3d11va` on Windows; `vaapi` on Linux. A backend is used only when FFmpeg
  advertises it, and a failing GPU attempt falls back to CPU. Known-unsupported video/backend combinations are remembered.
* Identical pictures and similar speed do **not** show where a video was decoded (decoding is exact, and a path may do part of
  the work, such as scaling, on the CPU). The *Backend compatibility check* in Benchmark proves it per codec.
* **Automatic check before Generate** (acceleration mode "auto"): after you confirm the preflight, every kind of video to be
  generated (codec + bit depth) that has no result yet for this FFmpeg build and GPU is checked without any question: about
  20 s of decoding per kind, only on the GPU backends Generate would use. The result (decodes on the GPU / decoded on the CPU
  after all / not supported by the card) is saved, and Generate then skips paths that do not work for that kind instead of
  trying them again and again. A check that could not decide (for example a timeout) is not saved and is repeated next time.
* **HDR and the GPU:** after the HDR check you can switch the GPU off for exactly the kinds of HDR you rated badly (too bright /
  too dark / wrong colors) on a given path (`cuda`, `d3d11va`, ...). Generate then uses the CPU for those videos only; other
  kinds, other paths and SDR keep using the GPU, and so does any kind that was never tested. Ratings are kept per FFmpeg build
  and GPU and are dropped when either changes. They are shown (and can be forgotten) under Configuration > Processing >
  Hardware acceleration. In "GPU only" mode a video whose kind is switched off on every path fails instead of falling back.
* **Benchmark before the first Generate:** when there is no benchmark result for this FFmpeg build and graphics card and at least 3 videos
  need generating, Generate offers the benchmark first (Enter = yes; a few minutes, once). The settings it suggests are applied only after
  your confirmation. The question comes back at every Generate until a benchmark has run to the end for this FFmpeg and GPU (a "no"
  is not remembered); the benchmark is also available under Benchmark. Unattended runs, dry runs and single files never ask.
* **Fastest way per kind of video:** the benchmark compares the CPU and the GPU backends per kind (codec, bit depth, picture size, HDR type)
  and, once you confirm, saves the winner. A GPU backend counts only when it beats the CPU by the benchmark threshold (Configuration >
  Processing > Hardware acceleration); on small SD files the CPU often wins. In acceleration mode "auto" Generate then goes straight to
  the chosen path for those kinds (CPU: no GPU attempt at all; GPU: the fastest path first); kinds that were not measured behave as before.
  Kept per FFmpeg build and GPU, shown (and forgettable) under Hardware acceleration. "GPU only" and "CPU only" are never overruled.
* **Colour space tags:** some files carry a tag newer FFmpeg builds refuse in the scale filter (e.g. `bt2020c`: "Impossible to
  convert between the formats ..."). Generate then tries once more with the tag ignored (`setparams=colorspace=unknown`; it does
  not change the thumbnails) instead of failing, and such a failure is not remembered as "the GPU cannot do this codec".
  The benchmark makes the same second try, so such a file is measured (and listed in the report) instead of failing on every path.
* What counts as HDR is decided by the same rules as in MediaDiversitySampler (`media_classification.py`): HDR10 / PQ,
  HDR10+, HLG and Dolby Vision are recognised from the transfer function and the stream's side data; BT.2020 SDR is SDR;
  an HDR file with a few details missing or unusual is "HDR (uncertain)" and is still treated as HDR. Bit depth comes from
  the pixel-format table, not from digits in its name. The HDR check tests one video per kind of HDR.

> The Linux `vaapi` path and the Windows adapter were written to the same design as the tested code but have **not yet
> been exercised on real GPUs / a real Windows machine**. Use the benchmark and the self-test before trusting them.

## Install and run

```
python -m pip install pillow
python main.py
```

On Linux Pillow can also come from the distribution (`sudo apt install python3-pil`); the [README](README.md#install-on-linux) shows that and the virtual environment way.

FFmpeg is taken from an `ffmpeg` folder beside `main.py` first, never from a configured path:

```
main.py
ffmpeg/
    bin/
        ffmpeg(.exe)
        ffprobe(.exe)
```

Several builds can sit side by side (`ffmpeg/<build>/bin/...`); the newest dated folder name wins. Windows builds:
<https://www.gyan.dev/ffmpeg/builds/>; Linux: a static build from <https://johnvansickle.com/ffmpeg/>.

**Windows** uses only that folder. **Linux** uses it too, and a build in it always wins; when it holds none, the `ffmpeg` and `ffprobe` found on `PATH` are used
(`sudo apt install ffmpeg`, `sudo dnf install ffmpeg-free`), but only when both exist. The system build is shown as `system` in the FFmpeg builds list, and its
benchmark and GPU ratings are kept like those of any other build.
The program creates `config/`, `data/`, `logs/`, `diagnostics/` and `temp/` beside itself (portable, nothing is
written elsewhere). Only one instance can run at a time.

## Command line (schedulers)

Without arguments the interactive menu opens. With an action the program runs once, **never asks anything** and exits:

```
python main.py --generate             generate missing, damaged and outdated trickplay for the configured media folders
python main.py --generate --dry-run   only show what would be done
python main.py --verify               check the existing output (changes nothing)
python main.py --self-test
python main.py --version
python main.py --check-language sl-SI  what is missing or wrong in a language file (for translators; asks nothing)
```

Exit codes: `0` OK, `1` problems (videos not valid / generation failures / a folder with unexpected files left alone / a video that could not be read / a self-test check failed),
`2` could not run (no media folders, FFmpeg missing, media folders unavailable, broken configuration, another instance running,
wrong arguments, or not enough free disk space), `3` unexpected error (the traceback is written to the log), `130` canceled.

Without anybody to ask the program takes the safe side: a trickplay folder that holds unexpected files is skipped or kept,
and a run whose estimated output does not fit on the disk is not started.
Settings come from the saved configuration (set them once in the menu). Logs go to `logs/` as usual.

## Languages

The menus and messages on the screen come from the files in the `locales` folder next to the program. `en-US.json` is the default and holds every text; `sl-SI.json` (Slovenščina) is included. The first start offers a choice when more than one language exists; later it is under Configuration > Language.

Only what the program shows on the screen is translated. **Logs, reports, JSON files and `--help` are always English**, so that a log can be read and compared whoever wrote it. A few details on the screen also stay English: the reasons Verify gives for a problem, the notes on suspicious thumbnails, and the details of the self-test. Technical terms (FFmpeg, GPU, CPU, HDR, JPEG, qscale, trickplay, tile, ...), file names and paths are not translated.

To add a language, copy `en-US.json` to a new file in the same folder (for example `de-DE.json`), set `_meta.code`, `_meta.locale` and `_meta.name`, and translate the texts. The file does not have to be complete: an empty or missing entry is shown in English, and an entry that does not fit (a changed `{placeholder}`, text where a plural form belongs) is ignored alone. Keep the `{placeholders}` exactly as they are. Texts with a number have plural forms (`one`, `two`, `few`, `many`, `other`; `other` is required). The `_meta` entries also set the number format (`_meta.thousands`, `_meta.decimal`) and the words for yes and no (`prompt.yes_words`, `prompt.no_words`); English `y` and `n` always work as well. Then run

```
python main.py --check-language de-DE
```

to see what is missing or wrong. The new language then appears in Configuration > Language; no other change is needed.

## Storage-aware workers

Videos are grouped by physical storage so one disk is not hammered by parallel jobs, while separate disks run in parallel.

* Linux: `/proc/self/mountinfo` + sysfs (partitions → disk, LVM/device-mapper members, CIFS/NFS → one group per server). Bind mounts are followed to the real directory (mountinfo `root`), and systemd automounts are triggered first, so the real storage is identified, not the placeholder.
* Windows: PowerShell `Get-Partition` / `Get-SmbMapping`.
* Override in Configuration: automatic, per drive/mount, or manual groups.
* Scanning (ffprobe) always uses 8 workers per storage group.

### Videos at once per disk: HDD and SSD

How many videos are processed at once on one disk depends on the kind of disk (Configuration > Processing > Videos at once > Videos at once per disk):

* **Automatic** (the default): 1 on a hard disk, 2 on an SSD or NVMe disk. A hard disk reads one video well and two badly, because the heads jump between the files. In a test with the same 35 videos (about 66 GB) one video at a time took 17 minutes from an HDD, 2 at once 26 and 4 at once 37 minutes. An SSD copes with 2 at once (7.5 minutes, against 5.5 with 2 at once and 2 GPU jobs); 3 or 4 bring nothing more.
* **A number**: the same on every disk.
* The kind is read from the system: on Windows from `Get-PhysicalDisk` (`MediaType`, `BusType`), on Linux from `/sys/block/*/queue/rotational`. A network share counts as a hard disk. A disk whose kind cannot be read (a USB disk Windows does not classify, a mount without a physical disk, an unreadable `rotational` value) is shown as **unknown (treated as HDD)**: nothing is invented, it is only handled with the careful hard-disk setting. A group that mixes kinds counts as the slower one.
* The time shown before Generate counts the disk too: a hard disk cannot be read faster than about 60 % of the speed the benchmark read from it, so a library on one hard disk is never shown as faster than that (`disk_speed` in the settings file, written by the benchmark; without a benchmark a typical 110 MB/s is assumed).
* The run log records what was used (`Videos at once per disk: PHYSICAL_DISK_3 (HDD) 1`).

## Project layout

```
main.py                  entry point; the only place that chooses the platform adapter
modules/
  application.py cli.py  startup, instance lock, crash handling; command-line arguments and exit codes
  platform_api.py        platform adapter interface + proxies
  platform_windows.py    Windows adapter (all Windows-specific code)
  platform_linux.py      Linux adapter (all Linux-specific code)
  ui.py                  Application: main menu + command-line mode
  screens/               one module per screen area (core, configuration, maintenance, diagnostics, generate, verify)
  config_summary.py      the settings as short texts for the menus
  prompts.py             terminal prompts
  localization.py        languages: texts, plural forms, number format (the texts are in locales/)
  generator.py           plan / finalize / run (TrickplayGenerator)
  storage_workers.py     videos at once per disk by kind of disk (HDD 1, SSD 2, or a number)
  extraction.py          FFmpeg run, GPU/CPU choice and retries, packing into tiles
  scanner.py tiles.py    discovery, probing cache, sprite validation
  verify.py maintenance.py selftest.py diagnostics.py
  ffmpeg_registry.py ffmpeg_tools.py config.py state_db.py ...
locales/                 en-US.json (default, every text), sl-SI.json, other languages
tests/                   unittest suite
```

OS-specific code lives only in the two adapters; a test enforces this.

## Tests

```
python -m unittest discover -s tests -v
```

Platform, core logic and end-to-end tests (real FFmpeg; skipped if FFmpeg or Pillow is missing).

## Not done yet

* Hardware validation of `vaapi` (Linux) and the Windows adapter.

## License

See [LICENSE](LICENSE).
