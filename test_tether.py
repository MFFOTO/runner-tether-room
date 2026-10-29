# -*- coding: utf-8 -*-
"""Self-test for Tether Room's own code: the metadata step (EXIF + sRGB profile),
Good-crop handling, cleanup after delivery and the crop settings.

Needs no photos, no GPU and not the crop engine: a small stand-in for
runner_suite_core is used, so it runs on a bare machine.

    python test_tether.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import piexif
from PIL import Image, ImageCms

# Stand-in for the crop engine: only what merged_suite imports, plus the few
# base-class behaviours the code under test calls.
core = types.ModuleType("runner_suite_core")
core.DEFAULT_CONFIG = {"quality_scoring": {"premium_only": False}, "performance": {"fast_path": True},
                       "completeness_guard": {"edge_truncation_guard": True}}


def _deep_merge(base, override):                 # the engine's own, as it is
    result = dict(base)
    for key, value in override.items():
        result[key] = _deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


core.deep_merge = _deep_merge
core.ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
core.tqdm = None


class _Base:
    def _copy_exif_without_orientation(self, src, dst, size=None):
        """What the patched crop engine does (see its test_metadata.py): copy the EXIF except the
        orientation and the thumbnail, with the real pixel dimensions. Stand-in so that Tether
        Room's override, which calls it, can be tested without the engine."""
        import piexif as _px
        d = _px.load(str(src))
        for k in ("0th", "Exif", "GPS", "Interop"):
            d[k] = dict(d.get(k) or {})
        d.pop("1st", None); d.pop("thumbnail", None)
        d["0th"].pop(_px.ImageIFD.Orientation, None)
        wh = size or gs.jpeg_size(dst)
        d["Exif"][_px.ExifIFD.PixelXDimension], d["Exif"][_px.ExifIFD.PixelYDimension] = int(wh[0]), int(wh[1])
        _px.insert(_px.dump(d), str(dst))
        return True

    def resize_final(self, img, enhance_mode="none"):
        return img

    def sharpen_image(self, img):
        return img

    def _quality_folder_name(self, quality_class):
        return {"premium": "A_Premium", "good": "B_Good"}.get(quality_class, "C_Review")


core.HighResRunnerSuite = _Base
sys.modules["runner_suite_core"] = core

import grade_suite as gs      # noqa: E402
import merged_suite as ms     # noqa: E402
import merged_ui as ui        # noqa: E402
import srgb                   # noqa: E402

fails = []


def chk(label, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + label + (f"   [{detail}]" if detail is not None and not ok else ""))
    if not ok:
        fails.append(label)


_td = tempfile.TemporaryDirectory()
root = Path(_td.name)

(root / 'meta').mkdir()
(root / 'logic').mkdir()
arr = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
orig = (root / 'meta') / "orig.jpg"
ex = {"0th": {piexif.ImageIFD.Make: b"Canon", piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.DateTime: b"2026:10:11 08:15:31"},
      "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:10:11 08:15:30", piexif.ExifIFD.DateTimeDigitized: b"2026:10:11 08:15:30",
               piexif.ExifIFD.SubSecTimeOriginal: b"42", piexif.ExifIFD.PixelXDimension: 6192, piexif.ExifIFD.PixelYDimension: 4128,
               piexif.ExifIFD.MakerNote: b"\x01" * 70000},          # too big for one segment
      "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
Image.fromarray(arr).save(orig, "JPEG", quality=95)
big = dict(ex, Exif=dict(ex["Exif"]))
ex["Exif"][piexif.ExifIFD.MakerNote] = b"\x01" * 20000
exif_bytes = piexif.dump(ex)
piexif.insert(exif_bytes, str(orig))

ms_ = ms.MergedSuite.__new__(ms.MergedSuite)
ms_.cfg = {"image_quality": {"preserve_exif": True}}

fails = []
# oversize EXIF (maker note > one JPEG segment) falls back instead of losing everything
fb = gs.build_exif(big, None, (10, 10))
chk("oversize maker note: block fits one segment and keeps the capture date",
    fb is not None and len(fb) + 2 <= 0xFFFF and gs.has_capture_date(fb))

# stage 1: crop written by cv2 (no metadata), hook called with the original
crop = (root / 'meta') / "crop.jpg"; cv2.imwrite(str(crop), arr[50:250, 50:250])
ms_._copy_exif_without_orientation(orig, crop)
got = piexif.load(str(crop))
chk("stage 1: DateTimeOriginal identical to the original", got["Exif"].get(piexif.ExifIFD.DateTimeOriginal) == b"2026:10:11 08:15:30")
chk("stage 1: DateTime, sub-seconds, make kept", got["0th"].get(piexif.ImageIFD.DateTime) == b"2026:10:11 08:15:31"
    and got["Exif"].get(piexif.ExifIFD.SubSecTimeOriginal) == b"42" and got["0th"].get(piexif.ImageIFD.Make) == b"Canon")
chk("stage 1: Orientation dropped, dimensions corrected to 200x200",
    piexif.ImageIFD.Orientation not in got["0th"] and got["Exif"][piexif.ExifIFD.PixelXDimension] == 200 and got["Exif"][piexif.ExifIFD.PixelYDimension] == 200)
icc = gs.extract_icc(crop.read_bytes())
chk("stage 1: tagged sRGB IEC61966-2.1", icc == srgb.srgb_icc() and gs.icc_name(icc) == "sRGB IEC61966-2.1")
chk("stage 1: still decodes", cv2.imread(str(crop)) is not None)

# newer engines pass the written size as a third argument; both call styles must work
three = (tmp := (root / 'meta') / "three.jpg"); cv2.imwrite(str(three), arr[50:250, 50:250])
chk("the metadata step accepts the engine's third argument (the written size) and uses it",
    ms_._copy_exif_without_orientation(orig, three, (200, 200)) is True
    and piexif.load(str(three))["Exif"][piexif.ExifIFD.PixelXDimension] == 200
    and gs.extract_icc(three.read_bytes()) == srgb.srgb_icc())
size_x = (root / 'meta') / "four.jpg"; cv2.imwrite(str(size_x), arr[50:250, 50:250])
ms_._copy_exif_without_orientation(orig, size_x, (1234, 567))
chk("an explicit size wins over the file's header", piexif.load(str(size_x))["Exif"][piexif.ExifIFD.PixelXDimension] == 1234)

# stage 2: finished image written by cv2, hook called with the crop
fin = (root / 'meta') / "fin.jpg"; cv2.imwrite(str(fin), cv2.resize(arr, (800, 600)))
ms_._copy_exif_without_orientation(crop, fin)
got = piexif.load(str(fin))
chk("stage 2: capture date survives the second hop", got["Exif"].get(piexif.ExifIFD.DateTimeOriginal) == b"2026:10:11 08:15:30")
chk("stage 2: dimensions are 800x600", (got["Exif"][piexif.ExifIFD.PixelXDimension], got["Exif"][piexif.ExifIFD.PixelYDimension]) == (800, 600))
chk("stage 2: tagged sRGB IEC61966-2.1 exactly once", gs.extract_icc(fin.read_bytes()) == srgb.srgb_icc() and fin.read_bytes().count(b"ICC_PROFILE") == 1)
chk("stage 2: PIL reads the profile name", ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(fin.read_bytes() and gs.extract_icc(fin.read_bytes())))).strip() == "sRGB IEC61966-2.1")
chk("no .part files left", not list((root / 'meta').glob("*.part")))

# ---------------------------------------------------------------- settings and cleanup
s = ms.MergedSuite.__new__(ms.MergedSuite)
s.cfg = {"merged": dict(ms.MERGED_CONFIG_EXTRA["merged"]), "image_quality": {"preserve_exif": True},
         "selection_filters": {"premium_min_sharpness": 180, "min_height_ratio": 0.15, "min_box_height_px": 220}}

# --- Good crops: default drops everything but Premium
chk("defaults: Good crops off, delete-after-delivery on",
    s.cfg["merged"]["keep_good_crops"] is False and s.cfg["merged"]["delete_crops_after_delivery"] is True)
s.cfg["quality_scoring"] = {"premium_only": None}
ms.apply_crop_policy(s.cfg)
chk("default: only Premium crops are written (the engine's premium_only is switched on)", s.cfg["quality_scoring"]["premium_only"] is True)
s.cfg["merged"]["keep_good_crops"] = True
ms.apply_crop_policy(s.cfg)
chk("keep Good crops on: premium_only is switched off", s.cfg["quality_scoring"]["premium_only"] is False)
s.cfg["merged"]["keep_good_crops"] = False
ms.apply_crop_policy(s.cfg)
import json as _json                                                       # noqa: E402
_cfgdir = root / "cfg"; _cfgdir.mkdir()
for keep, want in ((None, True), (False, True), (True, False)):
    f = _cfgdir / f"s{keep}.json"
    f.write_text(_json.dumps({} if keep is None else {"merged": {"keep_good_crops": keep}}))
    loaded = ms.MergedSuite.__new__(ms.MergedSuite)._load_config(f)
    chk(f"settings file with keep_good_crops={keep}: the engine's premium_only is {want}", loaded["quality_scoring"]["premium_only"] is want)
chk("Tether Room no longer overrides the engine's save rule, guard or fast path",
    not any(n in ms.MergedSuite.__dict__ for n in ("_should_save_class", "_assess_completeness", "_is_full_frame_runner", "_cut_by_frame",
                                                   "detect_fence", "calculate_quality_score", "get_smart_crop", "_process_detection_results")))
_saved_cfg = {k: dict(v) for k, v in core.DEFAULT_CONFIG.items()}
try:
    ms.check_engine()
    core.DEFAULT_CONFIG["quality_scoring"].pop("premium_only"); core.DEFAULT_CONFIG["performance"].pop("fast_path")
    try:
        ms.check_engine(); _msg = ""
    except RuntimeError as exc:
        _msg = str(exc)
    chk("an engine without the new settings is refused with the fix spelled out",
        "premium_only" in _msg and "fast_path" in _msg and "fetch_deps.py --update" in _msg, _msg)
finally:
    for k, v in _saved_cfg.items():
        core.DEFAULT_CONFIG[k] = v
chk("an engine that has them is accepted", ms.check_engine() is None)

# --- delete after delivery
arr = np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8)
ex = piexif.dump({"0th": {}, "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:10:25 11:34:40"}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None})
def mk(name):
    p = (root / 'logic') / "work" / name; p.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(arr).save(p, "JPEG", quality=95); piexif.insert(ex, str(p)); return p
out = (root / 'logic') / "out"
c1 = mk("a.jpg")
s._faults = {}; s._finishing = False
ok = s._finish_one(c1, out, 90, None)
chk("delivered: finished copy exists with the date", ok and (out / "a.jpg").exists()
    and piexif.load(str(out / "a.jpg"))["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:25 11:34:40")
chk("delivered: native-size crop removed from the work folder", not c1.exists())
s.cfg["merged"]["delete_crops_after_delivery"] = False
c2 = mk("b.jpg"); s._finish_one(c2, out, 90, None)
chk("setting off: the crop is kept", c2.exists() and (out / "b.jpg").exists())
s.cfg["merged"]["delete_crops_after_delivery"] = True
c3 = mk("c.jpg"); s._copy_exif_without_orientation = lambda a, b: False       # metadata step fails
s._finish_one(c3, out, 90, None)
chk("metadata step failed: the crop is NOT deleted", c3.exists())
del s._copy_exif_without_orientation
((root / 'logic') / "work" / "bad.jpg").write_bytes(b"not a jpeg")
chk("unreadable crop: not deleted, reported as unreadable",
    s._finish_one((root / 'logic') / "work" / "bad.jpg", out, 90, None) is False and ((root / 'logic') / "work" / "bad.jpg").exists()
    and s._faults[(root / 'logic') / "work" / "bad.jpg"] == "unreadable")

# --- UI settings: validate, apply, persist
cfgfile = (root / 'logic') / "settings_merged.json"
cfgfile.write_text(json.dumps({"performance": {"workers": 24}, "merged": {"note_dir": "D:/x"}}))
app = ui.App.__new__(ui.App)
app.config = str(cfgfile); app.suite = s; app.pipe = None
s.cfg["merged"]["keep_good_crops"] = False
good = {"keep_good": True, "delete_after": False, "skip_cut_off": False, "premium_min_sharpness": "200", "min_height_pct": "18"}
r = app.set_settings(good)
chk("settings apply to the engine", r["ok"] and s.cfg["selection_filters"]["premium_min_sharpness"] == 200.0
    and s.cfg["selection_filters"]["min_height_ratio"] == 0.18 and s.cfg["merged"]["keep_good_crops"] is True
    and s.cfg["merged"]["delete_crops_after_delivery"] is False)
saved = json.loads(cfgfile.read_text())
chk("...and are saved without touching other keys", r["saved"] and saved["performance"] == {"workers": 24}
    and saved["merged"]["note_dir"] == "D:/x" and saved["merged"]["keep_good_crops"] is True
    and saved["selection_filters"]["min_height_ratio"] == 0.18)
chk("the cut-off guard setting is applied and saved under completeness_guard",
    s.cfg["completeness_guard"]["edge_truncation_guard"] is False and saved["completeness_guard"] == {"edge_truncation_guard": False})
chk("get_settings reports it", app.get_settings()["skip_cut_off"] is False)
chk("the UI's Good-crops switch moves the engine's premium_only with it", s.cfg["quality_scoring"]["premium_only"] is False)
app.set_settings(dict(good, keep_good=False))
chk("...and back", s.cfg["quality_scoring"]["premium_only"] is True)
app.set_settings(good)
chk("it defaults to on when the page does not send it",
    app.set_settings({k: v for k, v in good.items() if k != "skip_cut_off"})["ok"] and s.cfg["completeness_guard"]["edge_truncation_guard"] is True)
app.set_settings(good)
chk("get_settings reports them back (height as a percentage)",
    app.get_settings()["min_height_pct"] == 18.0 and app.get_settings()["premium_min_sharpness"] == 200.0)
before = dict(s.cfg["selection_filters"])
for bad, why in [(dict(good, premium_min_sharpness="abc"), "text"), (dict(good, premium_min_sharpness="-1"), "negative sharpness"),
                 (dict(good, min_height_pct="0"), "0 %"), (dict(good, min_height_pct="95"), "95 %"), (dict(good, min_height_pct=None), "missing")]:
    rr = app.set_settings(bad)
    chk(f"rejected: {why}", not rr["ok"] and rr["error"])
chk("rejected settings change nothing", s.cfg["selection_filters"] == before)
app.config = str((root / 'logic') / "nodir" / "x.json")
chk("an unwritable settings file does not stop the settings applying",
    app.set_settings(dict(good, premium_min_sharpness="210"))["ok"] and s.cfg["selection_filters"]["premium_min_sharpness"] == 210.0)

# --- previews survive deleted crops
app2 = ui.App.__new__(ui.App)
app2._sample_lock = __import__("threading").Lock(); app2._samples = []; app2._sample_names = []
app2._sampled_from = 0; app2._tiles = {}; app2._warming = True      # no rail thread
files = []
for i in range(6):
    p = (root / 'logic') / "prev" / f"{i}.jpg"; p.parent.mkdir(exist_ok=True); Image.fromarray(arr).save(p, "JPEG"); files.append(str(p))
app2.pipe = types.SimpleNamespace(lock=__import__("threading").Lock(), _queued=set(files))
app2.suite = s
chk("previews load while the crops exist", app2._load_samples() and len(app2._samples) >= 1)
n = len(app2._samples)
for f in files: Path(f).unlink()                                      # everything delivered and cleared
chk("...and keep working after the crops are deleted", app2._load_samples() and len(app2._samples) == n)
app2.suite = s
r = app2.resample()
chk("Resample with nothing left keeps the samples instead of emptying them", r["ok"] and len(app2._samples) == n)


# ---------------------------------------------------------------- new folder from the browser
base = root / "browse"; base.mkdir()
r = ui.App.make_folder(str(base), "  Finish line 1 ")
chk("New folder: created inside the open folder, name trimmed",
    r["ok"] and Path(r["path"]) == base / "Finish line 1" and (base / "Finish line 1").is_dir() and not r["existed"])
r = ui.App.make_folder(str(base), "Finish line 1")
chk("New folder: an existing name just opens it, nothing is overwritten", r["ok"] and r["existed"])
(base / "Finish line 1" / "keep.txt").write_text("x")
ui.App.make_folder(str(base), "Finish line 1")
chk("...and its contents are untouched", (base / "Finish line 1" / "keep.txt").read_text() == "x")
for name, why in [("", "empty"), ("   ", "blank"), (".", "dot"), ("..", "dot-dot"), ("a/b", "slash"), ("a\\b", "backslash"),
                  ("..\\evil", "path escape"), ("C:x", "colon"), ("a*b", "star"), ('a"b', "quote"), ("con", "reserved name"),
                  ("NUL.txt", "reserved name with extension"), ("name.", "trailing dot"), ("x" * 101, "too long")]:
    chk(f"New folder refused: {why}", not ui.App.make_folder(str(base), name)["ok"])
chk("...and none of them created anything outside or inside", sorted(p.name for p in base.iterdir()) == ["Finish line 1"]
    and not (root / "evil").exists())
chk("New folder needs a folder to be open first", not ui.App.make_folder("", "x")["ok"]
    and not ui.App.make_folder(str(base / "missing"), "x")["ok"])

# one entry the OS will not describe must not blank the listing
(base / "ok1").mkdir(); (base / "ok2").mkdir(); (base / "locked").mkdir(); (base / "a.jpg").write_bytes(b"x")
_real_is_dir = Path.is_dir
def _fussy(self):
    if self.name == "locked":
        raise PermissionError(13, "Access is denied", str(self))
    return _real_is_dir(self)
Path.is_dir = _fussy
try:
    got = ui.App.browse(str(base))
finally:
    Path.is_dir = _real_is_dir
chk("Browse: a protected entry is skipped, the rest still lists and no error is raised",
    "error" not in got and {"ok1", "ok2", "Finish line 1"} <= set(got["dirs"]) and "locked" not in got["dirs"] and got["images"] == 1)

# ---------------------------------------------------------------- stop, pause, resume where it finished
import os                                                         # noqa: E402
import time                                                       # noqa: E402


def fake_run(tag, nphotos=0, per_photo=0.05, recrop=False, root_=None, workers=8):
    """A real Pipeline whose per-photo work is a fake that records what it saw."""
    base = root_ or (root / tag)
    watch, work = base / "watch", base / "work"
    watch.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    for i in range(nphotos):
        f = watch / f"IMG_{i:04d}.jpg"
        if not f.exists():
            f.write_bytes(b"x" * (100 + i)); os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))
    su = ms.MergedSuite.__new__(ms.MergedSuite)
    su.cfg = {"merged": dict(ms.MERGED_CONFIG_EXTRA["merged"], poll_interval_sec=0.1, file_stability_checks=1),
              "performance": {"workers": workers}}
    su.output_folder = work; su.ui_mode = True; su._lut_asked = False; su._lut_name = None
    su.cropped_names = []
    su.report_rows = []; su.report_lock = threading.Lock(); su.cfg["debug"] = {"write_csv": True}

    def crop(pth):
        time.sleep(per_photo); su.cropped_names.append(pth.name)
        n = len(su.cropped_names)           # alternate between the two row shapes the engine produces
        row = ({"image": str(pth), "decision": "accept", "reason": "ok", "output_path": "x.jpg"} if n % 2
               else {"image": str(pth), "decision": "reject", "reason": "too_small", "box_height": 99})
        with su.report_lock:
            su.report_rows.append(row)
    su.process_image = crop
    return su, ms.Pipeline(su, watch, None, work, recrop=recrop), watch, work


def run_in_thread(pipe):
    t = threading.Thread(target=pipe.run, daemon=True); t.start(); return t


def wait_for(cond, secs=10.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


# 1. Stop mid-crop, then restart: carries on where it finished
su, pipe, watch, work = fake_run("resume", nphotos=120, per_photo=0.2, workers=8)
t = run_in_thread(pipe)
wait_for(lambda: pipe.crops_done >= 20)
t0 = time.time(); pipe.request_stop()
chk("Stop: reports 'stopping' straight away", pipe.stopping)
t.join(timeout=20)
chk("Stop: cropping ends within about one photo's time, not after the backlog", not t.is_alive() and time.time() - t0 < 3.0)
chk("Stop: state moves to finished, not stopping", pipe.finished.is_set() and not pipe.stopping)
first_run = set(su.cropped_names)
chk("Stop: only part of the folder was cropped", 20 <= len(first_run) < 120, len(first_run))
chk("the ledger has exactly the photos that were cropped",
    len((work / "_cropped.log").read_text().splitlines()) == len(first_run) == pipe.crops_done)

import csv as _csv                                               # noqa: E402
reports1 = sorted(work.glob("runner_crop_report_*.csv"))
chk("a crop report was saved for the run, in the work folder", len(reports1) == 1, [r.name for r in reports1])
rows1 = list(_csv.DictReader(open(reports1[0], encoding="utf-8-sig"))) if reports1 else []
chk("...with one row per cropped photo", len(rows1) == len(first_run), (len(rows1), len(first_run)))
chk("...and the columns of both row types (accepted and rejected) in the header",
    reports1 and {"output_path", "reason", "box_height", "decision"} <= set(rows1[0].keys()))
chk("...and no half-written .part file is left behind", not list(work.glob("*.part")))
su2, pipe2, _, _ = fake_run("resume", per_photo=0.01, root_=root / "resume")
t = run_in_thread(pipe2)
wait_for(lambda: pipe2.crops_done + len(first_run) >= 120)
time.sleep(0.5); pipe2.request_stop(); t.join(timeout=20)
second_run = set(su2.cropped_names)
reports2 = sorted(work.glob("runner_crop_report_*.csv"))
chk("a restart writes its own report and leaves the first one alone",
    len(reports2) == 2 and reports1[0] in reports2 and len(list(_csv.DictReader(open(reports1[0], encoding="utf-8-sig")))) == len(first_run),
    [r.name for r in reports2])
chk("restart: nothing cropped twice", not (first_run & second_run), sorted(first_run & second_run)[:3])
chk("restart: together the two runs cropped every photo exactly once", first_run | second_run == {f"IMG_{i:04d}.jpg" for i in range(120)})
chk("restart: reports how many it skipped", pipe2.skipped_known == len(first_run))

# 1b. timing and the early-skip counter that the page shows
chk("worker time per photo is measured (a 0.2 s fake photo reads as roughly 200 ms)", 120 <= pipe.avg_photo_ms <= 900, pipe.avg_photo_ms)
su.stats = {"skipped_fence_soft": 3, "skipped_early_partial": 2, "skipped_fence_score": 1, "processed_crops": 40}
chk("skipped-early counts only the fast path's own counters", pipe.skipped_early == 6, pipe.skipped_early)

# 2. a replaced file (same name, new content) and the re-crop option
f0 = watch / "IMG_0000.jpg"; f0.write_bytes(b"new picture, same name"); os.utime(f0, (1_800_000_000, 1_800_000_000))
su3, pipe3, _, _ = fake_run("resume", per_photo=0.0, root_=root / "resume")
t = run_in_thread(pipe3); wait_for(lambda: pipe3.crops_done >= 1); time.sleep(0.4); pipe3.request_stop(); t.join(timeout=10)
chk("a different photo with an old name is cropped, the rest are not", su3.cropped_names == ["IMG_0000.jpg"], su3.cropped_names)
su4, pipe4, _, _ = fake_run("resume", per_photo=0.0, root_=root / "resume", recrop=True)
t = run_in_thread(pipe4); wait_for(lambda: pipe4.crops_done >= 120); pipe4.request_stop(); t.join(timeout=20)
chk("re-crop option: every photo is cropped again", len(set(su4.cropped_names)) == 120 and pipe4.skipped_known == 0)

# 3. ledger robustness
bad = root / "ledger"; (bad / "work").mkdir(parents=True)
(bad / "work" / "_cropped.log").write_text("not a line\n12\t34\tIMG_0001.jpg\n99\t5", encoding="utf-8")   # junk + a half-written last line
su5, pipe5, _, _ = fake_run("ledger", nphotos=2, root_=bad)
chk("ledger: junk and a half-written last line are ignored, valid lines kept", pipe5._cropped_before == {"12\t34\tIMG_0001.jpg"})

# 4. pause cropping: nothing lost, nothing cropped while paused, resumes cleanly
su, pipe, watch, work = fake_run("pause", nphotos=40, per_photo=0.1, workers=4)
t = run_in_thread(pipe); wait_for(lambda: pipe.crops_done >= 4)
pipe.set_cropping(False); time.sleep(0.5)                          # in-flight ones finish
n_at_pause = pipe.crops_done; time.sleep(1.0)
chk("pause: cropping really stops (count is flat for a second)", pipe.crops_done == n_at_pause, (n_at_pause, pipe.crops_done))
chk("pause: the run is still alive", t.is_alive() and not pipe.stopping)
(watch / "late.jpg").write_bytes(b"x" * 7); time.sleep(0.6)
chk("pause: a photo arriving meanwhile is not cropped yet", "late.jpg" not in su.cropped_names)
pipe.set_cropping(True)
chk("resume: everything, including the late arrival, is cropped, none twice",
    wait_for(lambda: len(su.cropped_names) >= 41) and len(su.cropped_names) == len(set(su.cropped_names)) == 41)
pipe.request_stop(); t.join(timeout=10)

# 5. Stop while paused does not hang
su, pipe, watch, work = fake_run("pause_stop", nphotos=30, per_photo=0.1, workers=4)
t = run_in_thread(pipe); wait_for(lambda: pipe.crops_done >= 2); pipe.set_cropping(False); time.sleep(0.4)
t0 = time.time(); pipe.request_stop(); t.join(timeout=10)
chk("Stop while paused ends promptly", not t.is_alive() and time.time() - t0 < 3.0)

# 6. Stop with a delivery queue: it drains, or is skipped on request
def delivery_run(tag):
    su, pipe, watch, work = fake_run(tag, nphotos=0)
    out = root / tag / "out"; out.mkdir(); pipe.out = out
    su._lut_asked = True; su._lut_name = "x.cube"
    crops = []
    for i in range(6):
        c = work / f"c{i}.jpg"; c.write_bytes(b"x"); crops.append(c)
    pipe.batch_size = 2; pipe._queued.update(str(c) for c in crops); pipe.pending = list(crops)
    batches = []
    def deliver(take, no, o):
        time.sleep(0.6); batches.append(no); su.last_batch_failed = []; su.last_batch_faults = {}
        su.last_batch_folder = str(no); su.last_batch_count = len(take); return len(take), 0
    su.deliver_batch = deliver
    return su, pipe, batches

su, pipe, batches = delivery_run("drain")
t = run_in_thread(pipe); time.sleep(0.2); pipe.request_stop(); t.join(timeout=30)
chk("Stop: queued crops are still delivered by default (3 batches of 2)", len(batches) == 3 and not t.is_alive(), batches)
su, pipe, batches = delivery_run("skip")
t = run_in_thread(pipe); time.sleep(0.2); pipe.request_stop(); pipe.skip_delivery(); t.join(timeout=30)
chk("Stop + skip delivery: leaves after the batch in progress, the rest stay queued on disk",
    not t.is_alive() and len(batches) <= 1 and len(list((root / "skip" / "work").glob("c*.jpg"))) == 6, batches)

# 7. the App layer: Stop, Resume (same event, same destination/look/job), skip delivery
def make_app(tag):
    base = root / tag; watch, work, out = base / "watch", base / "watch_crops", base / "out"
    watch.mkdir(parents=True); out.mkdir(parents=True)
    for i in range(30):
        f = watch / f"IMG_{i:04d}.jpg"; f.write_bytes(b"x" * (50 + i)); os.utime(f, (1_700_000_000 + i,) * 2)
    su = ms.MergedSuite.__new__(ms.MergedSuite)
    su.cfg = {"merged": dict(ms.MERGED_CONFIG_EXTRA["merged"], poll_interval_sec=0.1, file_stability_checks=1),
              "performance": {"workers": 4}, "lut": {"lut_folder": str(base)}, "paths": {}, "debug": {}, "selection_filters": {"premium_min_sharpness": 180, "min_height_ratio": 0.15, "min_box_height_px": 220}}
    su._lut_asked = False; su._lut_name = None; su.ui_mode = True
    su.look_event = threading.Event(); su.cropped_names = []
    su.process_image = lambda pth: (time.sleep(0.02), su.cropped_names.append(pth.name))
    su.change_lut = lambda n: (setattr(su, "_lut_name", n), setattr(su, "_lut_asked", True), su.look_event.set(), True)[3]
    su._available_luts = lambda: []
    a = ui.App.__new__(ui.App)
    a.config = str(base / "settings.json"); a.suite = su; a.pipe = None; a.thread = None
    a.status = "ready"; a._switching = False; a.last_out = ""; a.error = ""; a.watch = a.out = ""
    a.started_at = 0.0; a._tiles = {}; a._samples = []; a._sample_names = []; a._sample_lock = threading.Lock()
    a._sampled_from = 0; a._warming = True; a.job = {"spot": "", "order": "", "desc": ""}
    a._manifest_lock = threading.Lock(); a._manifest_lines = 0; a.note_dir = ""; a.note_name = "delivery.txt"
    return a, su, watch, out

a, su, watch, out = make_app("app")
chk("cannot pause, skip or resume when nothing is running",
    not a.set_cropping(False)["ok"] and not a.skip_delivery()["ok"] and not a.resume()["ok"])
chk("start works", a.start(str(watch))["ok"] and a.status == "running")
# destination, look and job as the operator would have set them
a.pipe.out = out; su._lut_asked = True; su._lut_name = "grade.cube"
a.last_out = str(out); a.out = str(out); a.job = {"spot": "finish", "order": "8765", "desc": "10k men"}
chk("pause cropping through the App", a.set_cropping(False)["ok"] and not a.pipe.cropping_on.is_set())
chk("...and resume it", a.set_cropping(True)["ok"] and a.pipe.cropping_on.is_set())
wait_for(lambda: a.pipe.crops_done >= 5)
chk("skip delivery is refused while not stopping", not a.skip_delivery()["ok"])
a.stop()
chk("state flags: stopping is reported while winding down or already finished",
    a.pipe.stopping or a.pipe.finished.is_set())
a.thread.join(timeout=15); first = len(su.cropped_names)
chk("Stop ended the run, status is stopped", a.status == "stopped" and first < 30, first)
r = a.resume()
chk("Resume: starts a new run on the same folder", r["ok"] and a.status == "running" and a.watch == str(watch))
chk("Resume: destination, look and job details are put back without asking again",
    r["restored"] and a.pipe.out == out and su._lut_name == "grade.cube"
    and a.job == {"spot": "finish", "order": "8765", "desc": "10k men"} and su.cfg["merged"]["batch_folder_prefix"] == "finish")
wait_for(lambda: len(set(su.cropped_names)) >= 30)
st = a.state()
chk("the page state carries the timing and skip figures", isinstance(st["avg_photo_ms"], int) and st["skipped_early"] == 0
    and st["avg_photo_ms"] >= 0, (st["avg_photo_ms"], st["skipped_early"]))
chk("after a restart, Cropped counts the photos done earlier too (progress through the folder), so it matches Arrived",
    st["cropped"] == 30 == st["seen"], (st["cropped"], st["seen"]))
chk("...while the run's own count, used for rates, is only what it cropped itself",
    st["cropped_run"] == 30 - first, (st["cropped_run"], first))
a.stop(); a.thread.join(timeout=15)
chk("Resume: carries on where it stopped, every photo cropped exactly once",
    len(su.cropped_names) == len(set(su.cropped_names)) == 30, (len(su.cropped_names), len(set(su.cropped_names))))
a2, su2, w2, o2 = make_app("app3")
a2.start(str(w2)); a2.stop(); a2.thread.join(timeout=15)
r2 = a2.resume()
chk("Resume with nothing chosen before: runs, asks again for look and destination", r2["ok"] and not r2["restored"])
a2.stop(); a2.thread.join(timeout=15)

# ---------------------------------------------------------------- the look: one theme, both pages
import re                                                         # noqa: E402
import grade_ui                                                   # noqa: E402
import theme                                                      # noqa: E402

for name, page in (("Tether Room", ui.PAGE), ("Grade Room", grade_ui.PAGE)):
    chk(f"{name}: every theme placeholder was filled", "@THEME" not in page)
    chk(f"{name}: has the theme switch and sets the theme before first paint",
        'id="th-btn"' in page and "data-theme" in page.split("</head>")[0])
    used = set(re.findall(r"var\((--[a-z0-9-]+)", page))
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", theme.TOKENS))
    chk(f"{name}: every colour variable it uses is defined", not (used - defined), sorted(used - defined))

light_css, dark_css = theme.TOKENS.split(':root[data-theme="dark"]')
def tokens(block):
    return dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})\b", block))
L, D = tokens(light_css), tokens(dark_css)
names = lambda block: set(re.findall(r"(--[a-z0-9-]+)\s*:", block))
chk("the dark theme defines every colour the light one does", names(light_css) - {"--ui", "--mono", "--r", "--r-sm"} <= names(dark_css),
    sorted(names(light_css) - names(dark_css)))

def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
def contrast(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)

for tname, T in (("light", L), ("dark", D)):
    for label, fg, bg, need in [("text on page", "--ink", "--ground", 4.5), ("text on panel", "--ink", "--panel", 4.5),
                                ("secondary text on panel", "--muted", "--panel", 4.5), ("secondary text on page", "--muted", "--ground", 4.5),
                                ("small labels on panel", "--dim", "--panel", 4.5), ("small labels on page", "--dim", "--ground", 4.5),
                                ("button text on accent", "--accent-ink", "--accent", 4.5), ("button text on accent (hover)", "--accent-ink", "--accent-hi", 4.5),
                                ("accent text on panel", "--accent", "--panel", 4.5), ("error text on panel", "--bad", "--panel", 4.5),
                                ("warning text on panel", "--warn", "--panel", 4.5), ("status dot on panel", "--live", "--panel", 3.0)]:
        c = contrast(T[fg], T[bg])
        chk(f"contrast, {tname}: {label} is at least {need}:1", c >= need, f"{c:.2f}")

# ---------------------------------------------------------------- compare_reports.py
import compare_reports as cr                                      # noqa: E402

def _write_report(path, rows):
    cols = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
A = [{"image": "D:\\t\\IMG_1.JPG", "candidate_idx": 0, "decision": "accept", "quality_class": "premium", "reason": "ok"},
     {"image": "D:\\t\\IMG_2.JPG", "candidate_idx": 0, "decision": "accept", "quality_class": "premium", "reason": "ok"},
     {"image": "D:\\t\\IMG_3.JPG", "candidate_idx": 1, "decision": "reject", "reason": "too_small"}]
B_same = [dict(A[0], quality_score=81), dict(A[1], quality_score=79), A[2]]       # scores differ, crops do not
B_diff = [A[0], A[2], {"image": "D:\\other\\img_9.jpg", "candidate_idx": 0, "decision": "accept", "quality_class": "premium", "reason": "ok"}]
ra, rb, rc = root / "a.csv", root / "b.csv", root / "c.csv"
_write_report(ra, A); _write_report(rb, B_same); _write_report(rc, B_diff)
import contextlib as _ctx                                          # noqa: E402
with _ctx.redirect_stdout(io.StringIO()) as out_same:
    code_same = cr.main(["x", str(ra), str(rb)])
with _ctx.redirect_stdout(io.StringIO()) as out_diff:
    code_diff = cr.main(["x", str(ra), str(rc)])
chk("compare_reports: the same Premium crops read as IDENTICAL even if scores differ", code_same == 0 and "IDENTICAL" in out_same.getvalue(), out_same.getvalue())
chk("compare_reports: a lost and a gained crop are both listed, and the exit code says different",
    code_diff == 1 and "img_2.jpg" in out_diff.getvalue() and "img_9.jpg" in out_diff.getvalue() and "DIFFERENT" in out_diff.getvalue(), out_diff.getvalue())

print("\nFAILED:" if fails else "\nALL PASS", fails or "")
_td.cleanup()
sys.exit(1 if fails else 0)
