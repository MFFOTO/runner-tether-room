# Tether Room

Crop and colour-grade race photos as they arrive from a tethered camera.

Point it at the folder your tethering software writes into. Every photo is
cropped to its runners as it lands; once 100 Premium crops have accumulated
they are finished, graded with a LUT of your choosing and written to a numbered
folder, together with a tab-separated delivery note for the uploader.

Two stages run independently, which is the point: **choosing or changing the
look never pauses cropping.**

---

## Install

Needs Python 3.10+ on Windows. An NVIDIA GPU is optional but roughly 3x faster.

```
git clone https://github.com/MFFOTO/runner-tether-room.git
cd runner-tether-room
setup.bat
```

`setup.bat` creates `.venv`, installs the dependencies, and fetches two things
this repo deliberately does not carry:

| fetched | from | why not vendored |
|---|---|---|
| `runner_suite_core.py` | [runner-suite-stage6](https://github.com/MFFOTO/runner-suite-stage6) | the crop engine keeps improving; a copy here would quietly fall behind |
| `luts/*.cube` | [runner-lut-suite](https://github.com/MFFOTO/runner-lut-suite) | the grades are their own project |

Re-run `python fetch_deps.py --update` any time to pick up newer versions of both.

**For an NVIDIA GPU**, add the CUDA build of torch after setup:

```
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

YOLO weights are **not** in the repo — ultralytics downloads `yolov8m-pose.pt`
by name on first run.

## Run

```
run.bat
```

Opens `http://127.0.0.1:8770/` in your browser. Everything is driven from
there; nothing is uploaded anywhere and it works with no internet.

1. **Choose the folder to watch.** Cropping starts immediately.
2. **When the first crops appear**, one dialog asks for everything at once:
   where the finished photos go, where the delivery note goes, the spot name,
   order number and description, and the look — previewed on *your* crops, with
   click-to-enlarge.
3. From then on it runs by itself. **Change look**, **Pause grading** and the
   **crop worker count** are all adjustable mid-run.

## Output

Finished photos land in numbered folders named after the spot:

```
D:\wherever\
    finish1\      100 graded crops
    finish2\      100 graded crops
    finish3\       47 graded crops
```

The delivery note is one tab-separated line per **closed** folder, written once
when that folder is complete, so anything reading it sees only finished work:

```
bogus     <order>   999999   bogus
finish1   876593    100      10k men, wet conditions
finish2   876593    100      10k men, wet conditions
finish3   876593     47      10k men, wet conditions
```

Fields: `spot name`, `order number`, `number of images`, `description`.
The first row is a fixed header. The spot name carries the folder number, so a
row and its folder are the same string.

## Configuration

`setup.bat` copies `settings_merged.example.json` to `settings_merged.json`,
which is git-ignored so each machine keeps its own paths. Worth knowing:

| key | meaning |
|---|---|
| `merged.premium_batch_size` | crops per folder (100) |
| `merged.premium_batch_timeout_sec` | flush a short folder after this idle time |
| `merged.note_dir` | where the delivery note goes. Ships as `D:/ProLoad Bulk Loader/Release` (the bulk loader reads it there); `""` puts it with the photos |
| `merged.recursive` | watch subfolders too — on, for per-card folders |
| `performance.workers` | crop threads; also adjustable live in the UI |
| `lut.lut_folder` | folder of `.cube` files |
| `image_quality.enhance_by_class` | **leave at `none`** unless you want FSRCNN upscaling, which costs ~56% more time |

## How it works

```
watch folder ──► stage 1: crop ──► queue ──► stage 2: finish + grade ──► folders of 100
                 YOLO pose,                  resize to 4000px, sharpen,      + delivery note
                 N threads                   LUT, encode once
```

Stage 1 writes crops at native size and does nothing else, so arrivals are
handled as fast as possible. Stage 2 does resize → sharpen → grade → encode in
a single pass, so every delivered image is encoded exactly once rather than
re-compressed. Only **Premium** crops are delivered; `B_Good` stays in the work
folder.

Grading runs on its own thread. That is what lets the look picker sit open
indefinitely — comparing sixteen LUTs — while cropping carries on underneath.

**Measured** on an RTX 4070 laptop with 24 workers, 6192x4128 sources: ~0.23 s
per photo to crop, ~55 ms per crop to grade, ~2.7 Premium crops per photo.

## Notes

- The work folder (native-size crops) is created next to the watched folder and
  is never written inside it, so crops are not re-ingested.
- Changing the destination mid-run takes effect at the next folder boundary,
  never mid-folder.
- Writes that fail are put back on the queue rather than dropped.
- A hard kill can leave a finished folder with no line in the note — restart
  and it re-queues anything undelivered.

## Licence

Private project. All rights reserved.
