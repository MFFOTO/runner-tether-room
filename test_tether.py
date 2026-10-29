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
core.DEFAULT_CONFIG = {}
core.deep_merge = lambda a, b: a
core.ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
core.tqdm = None


class _Base:
    def _should_save_class(self, quality_class, score):
        return quality_class == "premium"

    def resize_final(self, img, enhance_mode="none"):
        return img

    def sharpen_image(self, img):
        return img


core.HighResRunnerSuite = _Base
sys.modules["runner_suite_core"] = core

import grade_suite as gs      # noqa: E402
import merged_suite as ms     # noqa: E402
import merged_ui as ui        # noqa: E402
import srgb                   # noqa: E402

fails = []


def chk(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)
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
def chk(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok: fails.append(label)

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
chk("default: premium saved", s._should_save_class("premium", 90))
chk("default: good NOT saved", not s._should_save_class("good", 65))
chk("default: review/reject NOT saved", not s._should_save_class("review", 50) and not s._should_save_class("reject", 10))
s.cfg["merged"]["keep_good_crops"] = True
chk("keep_good on: good saved, premium saved", s._should_save_class("good", 65) and s._should_save_class("premium", 90))
chk("keep_good on: review still follows the engine (not saved here)", not s._should_save_class("review", 50))
s.cfg["merged"]["keep_good_crops"] = False

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

print("\nFAILED:" if fails else "\nALL PASS", fails or "")
_td.cleanup()
sys.exit(1 if fails else 0)
