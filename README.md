# Media Trickplay Generator

**Let your PC suffer so your movie doesn't buffer.**

Jellyfin makes the seek-bar preview pictures (trickplay) on the server itself, and a small server can chew on that for days.
This tool makes them on your stronger PC instead, with its graphics card if there is one, and puts them next to your videos.
The server then only plays the movies.

## What it does

* Creates Jellyfin trickplay pictures for your whole library, next to each video (`Movie.trickplay/`). Videos that already have valid pictures are skipped, so it is safe to run again and again.
* **Uses the GPU when it is faster** and the CPU when it is not. A built-in benchmark tests your own videos and suggests the best settings.
* **Checks which GPU methods (backends) really work** for each kind of video, so nothing is tried that is known to fail.
* **Never touches your videos.** It writes only trickplay folders, and a failed or interrupted run never damages pictures that were already good.
* Handles several disks at once and keeps hard disks and SSDs apart, so a slow disk is not made slower.
* **Understands network locations.** On Windows, UNC paths (`\\server\share\Movies`), mapped drive letters (`Z:`), shortcuts (`.lnk`) and Network Locations are all resolved to the real `\\server\share`. The program then knows that `Z:` and `\\server\share` are the same place and does not read from one server with too many videos at once.
* **Skips what is not media.** Recycle bins, trash folders and the preview and snapshot folders of NAS boxes (`@eaDir`, `#recycle`, `#snapshot`, ...) are never scanned.
* **Understands `~` on Linux.** A media folder like `~/Movies` becomes the `Movies` folder in your home folder.
* **Links.** A link (symbolic link or junction) that you add as a media folder is resolved to the real folder. Links inside your library are not followed, so nothing is processed twice or in a loop.
* Runs interactively from a menu, or from the command line for a scheduler.

## What you need

| | |
|---|---|
| Python | 3.10 or newer |
| FFmpeg and FFprobe | Windows: in the `ffmpeg/bin` folder next to `main.py`. Linux: the same folder, or the system package (see below) |
| Pillow | `pip install pillow` |
| Jellyfin setting | in the library settings turn on **Save trickplay images next to media** |
| GPU (optional) | NVIDIA (`cuda`) or, on Windows, any GPU (`d3d11va`). Without a GPU everything runs on the CPU. |

Intel Quick Sync (`qsv`) and Linux `vaapi` are included but not tested on real hardware yet.

## Install on Windows

1. Install Python from <https://www.python.org/downloads/>. Tick **Add python.exe to PATH** in the installer.
2. Download this project (green **Code** button, **Download ZIP**) and unpack it, for example to `C:\MediaTrickplayGenerator`.
3. Open a terminal in that folder and run:
   ```
   python -m pip install pillow
   ```
4. Download an FFmpeg build from <https://www.gyan.dev/ffmpeg/builds/> (the *essentials* or *full* zip) and unpack it so that you get:
   ```
   MediaTrickplayGenerator\
       main.py
       ffmpeg\
           bin\
               ffmpeg.exe
               ffprobe.exe
   ```
5. For an NVIDIA card keep the graphics driver up to date. Nothing else is needed.
6. Start it:
   ```
   python main.py
   ```

## Install on Linux

1. Get the project:
   ```
   git clone https://github.com/GlacialPigeon/MediaTrickplayGenerator.git
   cd MediaTrickplayGenerator
   ```
2. Install Python, FFmpeg and Pillow. Choose one of the two ways.

   **Packages from your distribution** (simplest):
   ```
   sudo apt install python3 ffmpeg python3-pil        # Debian, Ubuntu
   sudo dnf install python3 ffmpeg-free python3-pillow   # Fedora
   ```
   Start it with:
   ```
   python3 main.py
   ```

   **Virtual environment** (keeps Pillow out of the system):
   ```
   sudo apt install python3 python3-venv ffmpeg       # Debian, Ubuntu
   python3 -m venv .venv
   .venv/bin/pip install pillow
   ```
   Start it with:
   ```
   .venv/bin/python main.py
   ```
3. For an NVIDIA card install the proprietary driver (the `nvidia-smi` command must work).

**Your own FFmpeg build (optional).** The system FFmpeg is enough. If you want a specific build instead, download a static one into the `ffmpeg/bin` folder next to `main.py`; `ffmpeg` and `ffprobe` there are used before the system one. The same commands work on Debian, Ubuntu and Fedora (use `arm64` instead of `amd64` on a 64-bit ARM machine such as a Raspberry Pi):
```
sudo apt install curl xz-utils        # Debian, Ubuntu
sudo dnf install curl xz              # Fedora
cd MediaTrickplayGenerator
mkdir -p ffmpeg/bin
curl -L -o ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg.tar.xz --strip-components=1 -C ffmpeg/bin --wildcards '*/ffmpeg' '*/ffprobe'
rm ffmpeg.tar.xz
ffmpeg/bin/ffprobe -version
```

On Windows FFmpeg is only taken from the `ffmpeg` folder next to the program. On Linux a build in that folder wins; without one the program uses the FFmpeg and FFprobe the system provides. The program keeps its settings, logs and temporary files in folders beside itself; nothing is written anywhere else.

## First run

1. **Configuration > Library > Media folders**: add the folders that hold your movies and shows.
2. **Benchmark > CPU vs GPU benchmark**: it tests a few of your own videos on the CPU and the GPU and suggests settings. Accept what it suggests. This takes a while, and you do it once.
3. **Generate / Resume trickplay**: it shows what it is going to do and how much disk space it needs, and asks once before it starts.

You can stop with Ctrl+C at any time and start again later. Finished videos are not done twice.

## Command line and schedulers

```
python main.py --generate             make missing, damaged and outdated trickplay (asks nothing)
python main.py --generate --dry-run   only show what would be done
python main.py --verify               check the existing pictures, changes nothing
```

Exit code `0` means everything is fine, `1` means some videos had problems, `2` means it could not run. On Windows use Task Scheduler, on Linux cron or a systemd timer.

## More

All the details (safety rules, logs, every menu, acceleration, languages, project layout) are in the [technical reference](technical.md).

### More tools

Three separate tools that work well together:

* [Media Cataloger](https://github.com/GlacialPigeon/MediaCataloger): catalogs your library with FFprobe and writes Excel, JSON and TXT reports.
* [Media Diversity Sampler](https://github.com/GlacialPigeon/MediaDiversitySampler): picks a small, varied sample of your library, for example for testing.
* Media Trickplay Generator: this tool.

## License

See [LICENSE](LICENSE).
