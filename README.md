# Tether Room

Crop and colour-grade race photos as they arrive from a tethered camera.

Point it at the folder your tethering software writes into. Every photo is
cropped to its runners as it lands; once 100 Premium crops have accumulated
they are finished, retouched with a LUT of your choosing and written to a numbered
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

Opens `http://127.0.0.1:8770/` in your browser. Every *Browse* dialog has a **New
folder** button, so the output and delivery-note folders (or the folder to watch) can
be created there instead of in File Explorer. Everything is driven from
there; nothing is uploaded anywhere and it works with no internet.

1. **Check the Crop settings** (right-hand side), then **choose the folder to
   watch.** Cropping starts immediately, using those settings:
   - *Also keep Good crops*: off by default, meaning only Premium crops are
     saved at all. On, Good crops are also saved (to `B_Good` in the work
     folder) but are still never delivered.
   - *Premium needs sharpness of at least*: a crop softer than this cannot be
     Premium (engine default 180; 0 turns the check off).
   - *Runner at least (% of photo height)*: smaller runners are skipped. The
     engine also never accepts one under 220 px tall.
   - *Skip runners cut off by the photo's edge*: on by default. A runner whose
     box runs into the left or right edge of the photo is cut off by the frame,
     so it cannot be Premium. The crop engine only checks the top edge, and its
     close-up path skips the check altogether, which let crops of a runner half
     out of the frame through as Premium. Also stops a close-up that touches the
     top edge from taking that path. Uses `completeness_guard.edge_margin_px`
     (6 px). Switch it off to get the engine's own behaviour back.
   - *Delete each native-size crop once delivered*: on by default.

   All four can be changed while it runs; the next photo uses the new values.
   They are saved to `settings_merged.json`, so the next start remembers them.
2. **When the first crops appear**, one dialog asks for everything at once:
   where the finished photos go, where the delivery note goes, the spot name,
   order number and description, and the look — previewed on *your* crops, with
   click-to-enlarge.
3. From then on it runs by itself. **Change look**, **Pause grading**, **Pause
   cropping** and the **crop worker count** are all adjustable mid-run.
   - **Pause cropping** holds stage 1: photos already being cropped finish, the
     rest wait, nothing is lost, and grading and delivery carry on. **Resume
     cropping** continues exactly where it paused.
   - **Stop** ends cropping as soon as the photos in progress are done (seconds,
     not the whole backlog). The header then says what it is still waiting for,
     for example *stopping · 3 in progress · 40 queued*. Queued crops
     are delivered unless you press **Skip delivery**, in which case they stay in
     the work folder.
   - **Resume** (shown once stopped) starts again on the same folder where it
     stopped: photos already cropped are skipped, and the destination, look and
     order details from before are put back, so it carries on without asking
     again.
4. **Change folder** switches to a different monitored folder without closing
   the program. It finishes and delivers what is queued, then starts on the new
   folder. The setup sheet reopens with the last destination, spot, order number
   and description filled in; nothing is delivered until you confirm it, so a
   new event is never sent out under the old order by accident. After **Stop**
   it also serves as a way to start again, though **Resume** is the usual way.

### Look

The page opens in a light theme. The sun/moon button at the top right switches to a
dark one (a deep blue-grey, not black) and the choice is remembered in that browser.
Tether Room and Grade Room share one theme (`theme.py`), and photos are always shown
on a dark surround in the full-size viewer because that is how they are judged.

## Output

Finished photos land in numbered folders named after the spot:

```
D:\wherever\
    finish1\      100 retouched crops
    finish2\      100 retouched crops
    finish3\       47 retouched crops
```

The delivery note is one tab-separated line per **closed** folder, written once
when that folder is complete, so anything reading it sees only finished work:

```
bogus     <order>   999999   bogus
finish1   876593    100      10k men, wet conditions
finish2   876593    100      10k men, wet conditions
finish3   876593     47      10k men, wet conditions
```

Every delivered JPEG keeps the original photo's EXIF, capture date included,
and is tagged with the **sRGB IEC61966-2.1** colour profile. Only the
Orientation tag (the pixels are already rotated) and the thumbnail of the
uncropped original are dropped, and the pixel dimensions are corrected. The same
goes for the native-size crops in the work folder. Tether Room does not convert
colours: it assumes the camera shoots sRGB and warns once if a file says
otherwise.

Fields: `spot name`, `order number`, `number of images`, `description`.
The first row is a fixed header. The spot name carries the folder number, so a
row and its folder are the same string.

## Where things are saved

| what | where |
|---|---|
| finished photos | `<output folder>\<spot>1`, `<spot>2` ... |
| delivery note | `delivery.txt` in the note folder (`merged.note_dir`, default `D:/ProLoad Bulk Loader/Release`) |
| crop report, one file per run | `<watched folder>_crops\runner_crop_report_<date>_<time>.csv`: every photo the engine looked at, accepted or rejected, and why (box size, sharpness, score, class, reason). Refreshed every two minutes and when the run ends |
| photos already cropped | `<watched folder>_crops\_cropped.log` |
| native-size crops waiting for delivery | `<watched folder>_crops\A_Premium` (plus `B_Good` if Good crops are kept) |
| crops given up on | `<watched folder>_crops\_failed` |

Grade Room writes `_grade_report.csv` into its output folder.

## Configuration

`setup.bat` copies `settings_merged.example.json` to `settings_merged.json`,
which is git-ignored so each machine keeps its own paths. Worth knowing:

| key | meaning |
|---|---|
| `merged.premium_batch_size` | crops per folder (100) |
| `merged.keep_good_crops` | `false` (default): only Premium crops are written. `true`: Good crops are also kept in `B_Good` in the work folder, undelivered |
| `merged.delete_crops_after_delivery` | `true` (default): the native-size crop is removed from the work folder once its finished copy is delivered |
| `selection_filters.premium_min_sharpness` | a crop softer than this cannot be Premium (180; 0 = off) |
| `completeness_guard.edge_truncation_guard` | `true` (default): a runner cut by the photo's left/right edge cannot be Premium, and a close-up touching the left, right or top edge no longer bypasses the completeness check |
| `selection_filters.min_height_ratio` | smallest runner, as a fraction of photo height (0.15) |
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
  is never written inside it, so crops are not re-ingested. A delivered crop is
  removed from it once its finished copy is written (unless
  `delete_crops_after_delivery` is off), so it normally holds only what is
  still waiting to be delivered. A crop whose finished copy could not be
  completed (for example its EXIF or profile failed to write) is kept, with a
  warning in the console.
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
- **Restarting carries on where it finished.** The work folder keeps
  `_cropped.log`: one line per photo already cropped (its size, modified time and
  path inside the watched folder). A restart, whether after Stop, a crash or a
  closed program, skips those photos. A different photo that reuses an old file
  name is cropped, because its size or time differs. To crop everything again,
  tick *Crop photos again that an earlier run of this folder already cropped* on
  the start screen (or pass `--recrop` on the command line), or delete
  `_cropped.log`. Photos that were still waiting when Stop was pressed were never
  cropped, so they are picked up on the next start.
- **Stop** finishes the photos already in flight, delivers what can go out and
  exits. If there is no destination yet, grading is paused, or no look was
  chosen, the queued crops stay in the work folder instead of Stop waiting.
  *Skip delivery* leaves without delivering the rest; they are offered again on
  the next run for the same destination.

## Grade Room

A second, much smaller tool in the same repo: it applies a LUT to folders of
JPEGs and does nothing else. No cropping, no model, no torch. It shares
`.venv`, `lut.py` and the `luts` folder with Tether Room.

Start it with `run_grade.bat` (opens http://127.0.0.1:8771/). The first run
creates `settings_grade.json` from `settings_grade.example.json`; it is
git-ignored like the Tether Room settings.

1. **Folders.** Choose *Photos to grade* and *Save retouched copies to*. The
   output mirrors the input: same subfolders, same file names. The output may
   not be inside the input (or the reverse), and originals are never touched.
2. **Look.** Pick a `.cube` from the strip. Every tile is one of your own
   photos with that look applied. The strength slider blends the look with the
   original, and holding a sample shows the original photo. Double-click a
   sample, or press *Enlarge*, to see it at the size of your window: it appears
   at once and sharpens when the large version arrives. Hold the picture to
   compare with the original, use the arrow keys (or the side buttons) to step
   through the samples, and press Esc to close. If the large preview cannot be
   made, the page says why and keeps showing the small one.
3. **Options.** *Also finish to* starts ticked, so files are scaled to the target
   height (4000 px by default) unless you untick it for native size; set
   `"finish": false` in `settings_grade.json` to make native size the default.
   Finishing scales to a target height (4000 px, aspect ratio kept, nothing
   trimmed), sharpen and then grade, which is the same finishing as Tether
   Room's stage 2. Also here: JPEG quality, parallel workers (automatic by default), *Keep watching
   for new photos after this batch* (files still being written are left until
   their size stops changing), *Overwrite files that already exist in the
   output*, and *Retouch files that Grade Room has already retouched*.
4. **Start grading.** Progress is live; Stop ends the run cleanly and keeps what is already written.
   While it runs, the *Parallel workers* box under the progress numbers changes
   the worker count on the spot (1–64) — press Apply and watch the rate and
   cpu figures next to it (the rate covers the last 20 seconds).

Notes:

- Restart-safe: a file whose output already exists is skipped, and outputs are
  written under a temporary name and renamed, so a killed run never leaves a
  half-written JPEG.
- Every output carries `GradeRoom: <look> strength=… ` in its EXIF Software
  tag. A file already stamped is skipped rather than retouched twice, unless
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
- Every output is tagged **sRGB IEC61966-2.1**. The LUTs assume sRGB, so a
  source tagged with another RGB profile (Adobe RGB, Display P3 ...) is first
  converted to sRGB, and the file is reported. An untagged source is taken to be
  sRGB. EXIF, capture date included, is carried across.
- `_grade_report.csv` in the output folder lists every file and what happened.
- `settings_grade.json` keys: `lut_folder`, `port`, `quality`, `finish` (default
  `true`), `target_height`,
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
- `python test_grade.py` is a self-test that needs no photos. `python test_tether.py`
  does the same for Tether Room's metadata, Good-crop and cleanup handling, and the
  crop settings (no GPU or crop engine needed). `python test_edge_guard.py` runs the
  edge guard against the real, fetched crop engine (no model or GPU needed).

## Licence

Private project. All rights reserved.
