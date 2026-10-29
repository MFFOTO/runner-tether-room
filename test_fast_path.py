# -*- coding: utf-8 -*-
"""The fast path must change how long a photo takes, never what is saved.

Runs the real crop engine's per-photo loop (the fetched runner_suite_core.py) on
synthetic photos with a fake detector, once with merged.fast_path off and once
on, across randomised runners and many settings, and requires the same crops to
come out. No model, GPU or real photos are needed.

    python test_fast_path.py            (RUNNER_CORE_DIR if the engine is elsewhere)
"""

from __future__ import annotations

import copy
import itertools
import os
import random
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
import runner_suite_core as core        # noqa: E402
import merged_suite as ms               # noqa: E402

fails = []


def chk(label, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + label + (f"   [{detail}]" if detail is not None and not ok else ""))
    if not ok:
        fails.append(label)


class T:                                # stand-in for a torch tensor
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class Boxes:
    def __init__(self, boxes, confs): self.xyxy, self.conf, self._n = T(boxes), T(confs), len(boxes)
    def __len__(self): return self._n


def detection(boxes, confs, kps):
    return types.SimpleNamespace(boxes=Boxes(boxes, confs), keypoints=types.SimpleNamespace(data=T(kps)))


W, H = 2400, 1600
rng0 = np.random.default_rng(11)
IMAGES = []
for sigma in (0, 2.5, 7):               # sharp, soft, very soft: crosses the 80 and 180 sharpness lines
    base = (rng0.random((H // 8, W // 8, 3)) * 255).astype(np.uint8)
    im = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    IMAGES.append(cv2.GaussianBlur(im, (0, 0), sigma) if sigma else im)


# a crisp, fine-grained photo: the smooth ones above are soft at the small scale the
# subject-sharpness measure looks at, so only large, close runners ever pass them
IMAGES.append(cv2.GaussianBlur((np.random.default_rng(3).random((H, W, 3)) * 255).astype(np.uint8), (0, 0), 0.8))


def scenario(rnd):
    """A few runners of mixed kinds: whole and well placed, small and unsure (a low
    score ceiling), partly out of view, and touching the frame edge."""
    boxes, confs, kps = [], [], []
    for _ in range(rnd.randint(1, 4)):
        kind = rnd.choices(("healthy", "marginal", "partial", "edge"), (0.40, 0.30, 0.12, 0.18))[0]
        h = (rnd.uniform(0.16, 0.28) if kind == "marginal" else rnd.uniform(0.35, 0.9)) * H
        w = h * rnd.uniform(0.28, 0.45)
        cx = rnd.uniform(0.2 * W, 0.8 * W)
        x1 = min(max(8.0, cx - w / 2), W - w - 8.0)
        y1 = rnd.uniform(10.0, max(11.0, H - h - 10.0))
        if kind == "edge":
            side = rnd.choice(("left", "right", "top"))
            x1 = 0.0 if side == "left" else (W - w if side == "right" else x1)
            y1 = 0.0 if side == "top" else y1
        b = (x1, y1, min(W, x1 + w), min(H, y1 + h))
        visible_p = 0.30 if kind == "partial" else 0.97
        k = [[rnd.uniform(b[0], b[2]), rnd.uniform(b[1], b[3]), 0.95 if rnd.random() < visible_p else 0.05] for _ in range(17)]
        boxes.append(b); kps.append(k)
        confs.append(rnd.uniform(0.34, 0.42) if kind == "marginal" else rnd.uniform(0.5, 0.97))
    return boxes, confs, kps


def make_suite(root, fast, variant):
    s = ms.MergedSuite.__new__(ms.MergedSuite)
    s.cfg = core.deep_merge(copy.deepcopy(core.DEFAULT_CONFIG), copy.deepcopy(ms.MERGED_CONFIG_EXTRA))
    s.cfg["merged"]["fast_path"] = fast
    for section, key, value in variant:
        s.cfg[section][key] = value
    s.cfg["image_quality"]["enhance_by_class"] = {"premium": "none", "good": "none", "review": "none"}
    s.cfg["debug"]["write_csv"] = False
    s.stats = {}; s.stats_lock = threading.Lock(); s.report_rows = []; s.report_lock = threading.Lock()
    s.sr_lock = threading.Lock(); s._face_restorer = None; s._finishing = False; s.premium_sink = None
    s.output_folder = root; s.cfg["paths"]["output_folder"] = str(root); s.rejects_folder = root / "_r"
    return s


def saved(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.jpg"))


# settings combinations that change which candidates can be saved
VARIANTS = []
for premium_min, keep_good, save_review, edge, completeness, metric, premium_score in itertools.product(
        (0, 180, 600), (False, True), (False, True), (True, False), (True, False), ("subject", "box"), (73, 35)):
    VARIANTS.append([("selection_filters", "premium_min_sharpness", premium_min), ("merged", "keep_good_crops", keep_good),
                     ("quality_scoring", "save_review", save_review), ("completeness_guard", "edge_truncation_guard", edge),
                     ("completeness_guard", "enabled", completeness), ("selection_filters", "sharpness_metric", metric),
                     ("quality_scoring", "classes", {"premium": premium_score, "good": max(0, premium_score - 13), "review": 0})])

rnd = random.Random(2026)
mismatches = photos = total_saved = 0
first_bad = None
skip_counts = {}
for variant in rnd.sample(VARIANTS, 80):
    for _ in range(4):
        img = rnd.choice(IMAGES); boxes, confs, kps = scenario(rnd)
        outs = {}
        for fast in (False, True):
            root = Path(tempfile.mkdtemp()); s = make_suite(root, fast, variant)
            src = root / "photo.jpg"; cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            s._process_detection_results(img, src, [detection(boxes, confs, kps)])
            outs[fast] = [n for n in saved(root) if n != "photo.jpg"]
            if fast:
                total_saved += len(outs[fast])
                for k, v in s.stats.items():
                    if k.startswith("skipped_"):
                        skip_counts[k] = skip_counts.get(k, 0) + v
        photos += 1
        if outs[False] != outs[True]:
            mismatches += 1; first_bad = first_bad or (variant, outs)
chk(f"the same crops are saved with the fast path on and off ({photos} randomised photos, 80 setting combinations)",
    mismatches == 0, first_bad)
for rule in ("skipped_early_partial", "skipped_fence_soft", "skipped_fence_score"):
    chk(f"the scenarios exercised the rule {rule}", skip_counts.get(rule, 0) > 0, skip_counts)
print("  skips taken in the 'on' runs:", skip_counts)
chk("and crops were genuinely saved, so identical output is not just identical emptiness", total_saved >= 50, total_saved)

# the score-ceiling rule, with marginal runners only (so it is the rule that decides), several score lines
rnd2 = random.Random(77)
mism2 = n2 = 0; score_skips = saved2 = 0
for premium_score, keep_good in itertools.product((92, 84, 78, 40), (False, True)):
    variant = [("selection_filters", "premium_min_sharpness", 0), ("merged", "keep_good_crops", keep_good),
               ("quality_scoring", "classes", {"premium": premium_score, "good": premium_score - 13, "review": 0})]
    for _ in range(14):
        boxes, confs, kps = [], [], []
        for _ in range(rnd2.randint(1, 3)):
            h = rnd2.uniform(0.16, 0.34) * H; w = h * rnd2.uniform(0.28, 0.45)
            x1 = rnd2.uniform(0.2 * W, 0.7 * W); y1 = rnd2.uniform(20.0, H - h - 20.0)
            boxes.append((x1, y1, x1 + w, y1 + h)); confs.append(rnd2.uniform(0.34, 0.6))
            kps.append([[rnd2.uniform(x1, x1 + w), rnd2.uniform(y1, y1 + h), 0.95] for _ in range(17)])
        img = IMAGES[3]; outs = {}
        for fast in (False, True):
            root = Path(tempfile.mkdtemp()); s = make_suite(root, fast, variant)
            src = root / "photo.jpg"; cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            s._process_detection_results(img, src, [detection(boxes, confs, kps)])
            outs[fast] = [n for n in saved(root) if n != "photo.jpg"]
            if fast:
                score_skips += s.stats.get("skipped_fence_score", 0); saved2 += len(outs[fast])
        n2 += 1; mism2 += outs[False] != outs[True]
chk(f"score-ceiling rule: same crops with the fast path on and off ({n2} marginal-runner photos, 4 score lines)", mism2 == 0)
chk("...the rule fired often and some marginal runners were still saved", score_skips >= 10 and saved2 >= 5, (score_skips, saved2))
print(f"  score-ceiling skips: {score_skips}, crops saved in those runs: {saved2}")

# the report still says why a skipped candidate was dropped, and unmeasured fields are blank
s = make_suite(Path(tempfile.mkdtemp()), True, [("merged", "keep_good_crops", False), ("selection_filters", "premium_min_sharpness", 600)])
s.cfg["debug"]["write_csv"] = True
root = s.output_folder; src = root / "p.jpg"; cv2.imwrite(str(src), IMAGES[0])
boxes = [(900.0, 200.0, 1300.0, 1500.0)]; kps = [[[1100.0, 200.0 + i * 70, 0.95] for i in range(17)]]
s._process_detection_results(IMAGES[0], src, [detection(boxes, [0.9], kps)])
rows = s.report_rows
chk("a skipped candidate is reported with the reason, not as a crop failure",
    rows and not any(r.get("reason") == "crop_failed" for r in rows), rows)
chk("...and the sharp_box column is blank, not a misleading 0", all(r.get("sharp_box", "") in ("", None) for r in rows if "sharp_box" in r), rows)

print("\nFAILED:" if fails else "\nALL PASS", fails or "")
sys.exit(1 if fails else 0)
