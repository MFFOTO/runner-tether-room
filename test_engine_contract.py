# -*- coding: utf-8 -*-
"""Tether Room on the real crop engine: does what it relies on exist, and do the two work together?

The engine (runner_suite_core.py) is fetched, not kept in this repo, so it can drift from what
Tether Room expects. This runs the engine's real per-photo loop with a fake detector and checks
the points where the two meet: the engine's settings, the Good-crops switch, the engine's EXIF
copy plus Tether Room's sRGB tag, and that every crop written is announced for delivery.
No model, GPU or real photos are needed.

    python test_engine_contract.py            (RUNNER_CORE_DIR if the engine is elsewhere)
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import tempfile
import threading
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
core_dir = Path(os.environ.get("RUNNER_CORE_DIR", HERE))
if not (core_dir / "runner_suite_core.py").exists():
    print(f"SKIP  runner_suite_core.py not found in {core_dir} (run setup.bat / fetch_deps.py, or set RUNNER_CORE_DIR)")
    sys.exit(0)
sys.path[:0] = [str(HERE), str(core_dir)]
for name, attr in (("ultralytics", "YOLO"), ("tqdm", "tqdm")):
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        setattr(m, attr, object if name == "ultralytics" else None)
        sys.modules[name] = m

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402
import piexif                           # noqa: E402
import runner_suite_core as core        # noqa: E402
import grade_suite as gs                # noqa: E402
import merged_suite as ms               # noqa: E402
import srgb                             # noqa: E402

fails = []


def chk(label, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + label + (f"   [{detail}]" if detail is not None and not ok else ""))
    if not ok:
        fails.append(label)


try:
    ms.check_engine()
    engine_ok, why = True, None
except RuntimeError as exc:
    engine_ok, why = False, str(exc)
chk("the engine has the settings Tether Room relies on (premium_only, fast_path, edge_truncation_guard)", engine_ok, why)
if not engine_ok:
    print("\nNothing else can be checked until the engine is updated.")
    sys.exit(1)
chk("the engine's EXIF copy takes the written size as its third argument",
    "size" in inspect.signature(core.HighResRunnerSuite._copy_exif_without_orientation).parameters)


class T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class Boxes:
    def __init__(self, boxes, confs): self.xyxy, self.conf, self._n = T(boxes), T(confs), len(boxes)
    def __len__(self): return self._n


def detection(boxes, confs, kps):
    return types.SimpleNamespace(boxes=Boxes(boxes, confs), keypoints=types.SimpleNamespace(data=T(kps)))


LAYOUT = [(.5, .07), (.46, .06), (.54, .06), (.42, .07), (.58, .07), (.35, .20), (.65, .20), (.30, .35), (.70, .35),
          (.30, .48), (.70, .48), (.40, .50), (.60, .50), (.40, .72), (.60, .72), (.40, .95), (.60, .95)]


def keypoints(b):
    return [[b[0] + fx * (b[2] - b[0]), b[1] + fy * (b[3] - b[1]), 0.95] for fx, fy in LAYOUT]


W, H = 2400, 1600
IMG = cv2.GaussianBlur((np.random.default_rng(3).random((H, W, 3)) * 255).astype(np.uint8), (0, 0), 0.8)
DATE = b"2026:10:25 11:34:40"


def run(settings, box, conf=0.9):
    """One photo through the engine's real loop on a MergedSuite built from `settings` (a settings file)."""
    root = Path(tempfile.mkdtemp())
    f = root / "settings.json"; f.write_text(json.dumps(settings))
    s = ms.MergedSuite.__new__(ms.MergedSuite)
    s.cfg = s._load_config(f)
    s.cfg["debug"]["write_csv"] = False
    s.stats = {}; s.stats_lock = threading.Lock(); s.report_rows = []; s.report_lock = threading.Lock()
    s.sr_lock = threading.Lock(); s._face_restorer = None; s._finishing = False
    announced = []
    s.premium_sink = announced.append
    work = root / "work"; s.output_folder = work; s.cfg["paths"]["output_folder"] = str(work); s.rejects_folder = work / "_r"
    src = root / "IMG_0001.JPG"
    cv2.imwrite(str(src), IMG, [cv2.IMWRITE_JPEG_QUALITY, 90])
    piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Make: b"NIKON CORPORATION", piexif.ImageIFD.Orientation: 1},
                               "Exif": {piexif.ExifIFD.DateTimeOriginal: DATE}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}), str(src))
    s._process_detection_results(IMG, src, [detection([box], [conf], [keypoints(box)])])
    return s, announced, sorted(str(p.relative_to(work)) for p in work.rglob("*.jpg")), work


box = (1000.0, 300.0, 1700.0, 1500.0)
low = {"merged": {}, "quality_scoring": {"classes": {"premium": 1, "good": 0, "review": 0}}, "selection_filters": {"premium_min_sharpness": 0}}

# a Premium runner: written, announced for delivery, EXIF and sRGB in place
s, announced, files, work = run(low, box)
chk("a Premium runner is written to A_Premium", len(files) == 1 and files[0].startswith("A_Premium"), files)
chk("...and announced for delivery (the engine's write completed without error)",
    len(announced) == 1 and Path(announced[0]).name == Path(files[0]).name and not s.stats.get("errors"), (announced, s.stats))
crop = work / files[0] if files else None
if crop:
    exif = piexif.load(str(crop))
    h, w = cv2.imread(str(crop)).shape[:2]
    chk("the crop carries the original capture date and camera make", exif["Exif"].get(piexif.ExifIFD.DateTimeOriginal) == DATE
        and exif["0th"].get(piexif.ImageIFD.Make) == b"NIKON CORPORATION")
    chk("...with its own pixel dimensions and no stale orientation or thumbnail",
        (exif["Exif"][piexif.ExifIFD.PixelXDimension], exif["Exif"][piexif.ExifIFD.PixelYDimension]) == (w, h)
        and piexif.ImageIFD.Orientation not in exif["0th"] and not exif["thumbnail"])
    chk("...tagged sRGB IEC61966-2.1", gs.extract_icc(crop.read_bytes()) == srgb.srgb_icc())
    chk("stage 1 keeps the crop at native size (the resize belongs to stage 2)", h > 600, (w, h))

# the Good-crops switch, through the engine's real loop
probe_score = float(re.search(r"_q(\d+)_", files[0]).group(1))
lines = {"classes": {"premium": probe_score + 3, "good": probe_score - 10, "review": 0}}
off = {"merged": {"keep_good_crops": False}, "quality_scoring": lines, "selection_filters": {"premium_min_sharpness": 0}}
on = {"merged": {"keep_good_crops": True}, "quality_scoring": lines, "selection_filters": {"premium_min_sharpness": 0}}
_, ann_off, f_off, _ = run(off, box)
_, ann_on, f_on, _ = run(on, box)
chk("keep Good crops off: a Good-class runner is not written at all", f_off == [] and ann_off == [], f_off)
chk("keep Good crops on: it is written to B_Good, and not announced for delivery",
    len(f_on) == 1 and f_on[0].startswith("B_Good") and ann_on == [], (f_on, ann_on))

# the engine's edge guard and fast path are in force through Tether Room
cut = (1500.0, 0.0, 2400.0, 1600.0)
_, _, f_cut, _ = run(low, cut)
chk("a close-up cut off by the frame is not written", f_cut == [], f_cut)
soft = {"merged": {}, "quality_scoring": {"classes": {"premium": 1, "good": 0, "review": 0}}, "selection_filters": {"premium_min_sharpness": 5000}}
s2, _, f_soft, _ = run(soft, box)
chk("a runner below the Premium sharpness line is dropped without the fence check (fast path)",
    f_soft == [] and s2.stats.get("skipped_fence_soft") == 1, s2.stats)

print("\nFAILED:" if fails else "\nALL PASS", fails or "")
sys.exit(1 if fails else 0)
