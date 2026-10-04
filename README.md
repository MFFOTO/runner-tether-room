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
4. **Change folder** switches to a different monitored folder without closing
   the program. It finishes and delivers what is queued, then starts on the new
   folder. The setup sheet reopens with the last destination, spot, order number
   and description filled in; nothing is delivered until you confirm it, so a
   new event is never sent out under the old order by accident. After **Stop**
   it also serves as a way to start again.

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
- Writes that fail are put back on the queue rather than dropped, with limits:
  a crop that cannot be read is tried three times and then moved to `_failed`
  in the work folder (the Grade panel shows how many); a destination that
  refuses every file is retried with a growing pause instead of flat out. A
  folder is only created once something is written into it, so a failed batch
  leaves no empty numbered folder and does not use up a number.
- A hard kill can leave a finished folder with no line in the note — restart
  with the same destination and it re-queues anything undelivered.
- Folder numbers continue after the highest one already in the destination, so
  a restart, or a second event sent to the same place, never writes into an
  existing `finish1` or repeats its row in the note.
- The work folder is shared by every run that watches the same folder, so it
  keeps a small record (`_delivery_target.txt`) of where its crops were going.
  Leftover crops are resumed only for that same destination; any others are
  moved to `_previous_<date>` inside the work folder and never delivered.
  Photos still in the watch folder are simply cropped again.
- **Stop** finishes the photos already in flight, delivers what can go out and
  exits. If there is no destination yet, grading is paused, or no look was
  chosen, the queued crops stay in the work folder instead of Stop waiting.

## Grade Room

A second, much smaller tool in the same repo: it applies a LUT to folders of
JPEGs and does nothing else. No cropping, no model, no torch. It shares
`.venv`, `lut.py` and the `luts` folder with Tether Room.

Start it with `run_grade.bat` (opens http://127.0.0.1:8771/). The first run
creates `settings_grade.json` from `settings_grade.example.json`; it is
git-ignored like the Tether Room settings.

1. **Folders.** Choose *Photos to grade* and *Save graded copies to*. The
   output mirrors the input: same subfolders, same file names. The output may
   not be inside the input (or the reverse), and originals are never touched.
2. **Look.** Pick a `.cube` from the strip. Every tile is one of your own
   photos with that look applied. The strength slider blends the look with the
   original, and holding a sample shows the ungraded photo. Double-click a
   sample, or press *Enlarge*, to see it at the size of your window: it appears
   at once and sharpens when the large version arrives. Hold the picture to
   compare with the original, use the arrow keys (or the side buttons) to step
   through the samples, and press Esc to close. If the large preview cannot be
   made, the page says why and keeps showing the small one.
3. **Options.** By default files are graded at native size. Tick *Also finish
   to* to scale to a target height (4000 px, aspect ratio kept, nothing
   trimmed), sharpen and then grade, which is the same finishing as Tether
   Room's stage 2. Also here: JPEG quality, parallel workers (automatic by default), *Keep watching
   for new photos after this batch* (files still being written are left until
   their size stops changing), *Overwrite files that already exist in the
   output*, and *Grade files that Grade Room has already graded*.
4. **Start grading.** Progress is live; Stop ends the run cleanly and keeps what is already written.
   While it runs, the *Parallel workers* box under the progress numbers changes
   the worker count on the spot (1–64) — press Apply and watch the rate and
   cpu figures next to it (the rate covers the last 20 seconds).

Notes:

- Restart-safe: a file whose output already exists is skipped, and outputs are
  written under a temporary name and renamed, so a killed run never leaves a
  half-written JPEG.
- Every output carries `GradeRoom: <look> strength=… ` in its EXIF Software
  tag. A file already stamped is skipped rather than graded twice, unless
  the regrade option is on.
- **Small photos.** Photos whose longer side is under 4000 px are flagged, never
  skipped. After you choose the input folder the page reads every JPEG header
  (no decoding, so it takes seconds) and says how many are affected, with a
  *Show which* list, before you start. During and after the run an amber banner
  shows the running count and the smallest size, the browser tab title starts
  with `(!)`, and `_grade_report.csv` records `width`, `height` and a note for
  each one. The limit is the *Warn about photos smaller than* box in Options
  (`warn_below` in `settings_grade.json`, `--warn-below` on the command line);
  `0` turns the warning off. The rule is on the longer side, so orientation does
  not matter, and 3999 px is flagged while 4000 px is not. It looks at the
  source photo's own size, whether or not *Also finish to* is ticked.
- The colour profile is carried across. The LUTs assume sRGB, so a file tagged
  with anything else is reported.
- `_grade_report.csv` in the output folder lists every file and what happened.
- `settings_grade.json` keys: `lut_folder`, `port`, `quality`, `target_height`,
  `warn_below` (default 4000, `0` = off), `sharpen` (`none`/`light`/`medium`/`strong`), `interpolation` (`lanczos` or
  `cubic` — cubic is about 9x faster when upscaling), `workers` (`0` or `"auto"` = automatic, or a fixed 1–64).
- Workers: automatic means one worker per CPU thread (a 32-thread CPU gets 32),
  so the whole CPU can be used; a fixed number can never use more than that many
  threads, e.g. 20 workers top out around 60% CPU on 32 threads. Automatic is
  held back only if the photos are so big that that many in flight would not
  fit in free memory. If the cpu figure stays well under 100% even so, the
  limit is probably the disk (or Windows Defender scanning each new file); try
  a few more workers than CPU threads to overlap the waiting.
- Memory: each worker holds one photo in flight, roughly 250 MB for a 24 MP
  JPEG (scale with megapixels). Building the look's table at the start briefly
  uses a further 0.3 GB for half a second. If the PC starts swapping or the rate drops as
  you add workers, lower the number.
- **Where the time goes.** Once a few photos are done, the progress card shows
  the average time per photo for each stage (read, decode, grade, encode, write)
  and how much of it was spent computing rather than waiting. A high waiting
  share means the drive, a network share or antivirus is the limit, and more
  workers will not help.
- **Speed test.** `bench_grade.bat` (or `python grade_suite.py bench <folder>`)
  takes about 24 of your photos and measures what limits speed on this PC:
  reading, computing, writing, and the speed at 1, 2, 4 … workers, then says
  which part is the limit. Give it a second folder (`bench_grade.bat <photos>
  <output folder>`) to test the drive you really write to. The result is saved
  as `bench_grade_result.txt`.
- The colour table for a look is built in about half a second and about 270 MB
  (it used to be several seconds and 1.3 GB), and gives exactly the same
  bytes as before.
- `python test_grade.py` is a self-test that needs no photos.

## Licence

Private project. All rights reserved.
