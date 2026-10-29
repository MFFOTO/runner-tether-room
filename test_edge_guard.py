# -*- coding: utf-8 -*-
"""Self-test for the edge-truncation guard, run against the REAL crop engine.

The engine (runner_suite_core.py) is fetched by setup, not kept in this repo.
This test imports it from next to this file, or from the folder named in the
RUNNER_CORE_DIR environment variable, and only stubs the heavy libraries it
imports (ultralytics, tqdm) when they are not installed. No model is loaded.

    python test_edge_guard.py
"""

from __future__ import annotations

import copy
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
core_dir = Path(os.environ.get("RUNNER_CORE_DIR", HERE))
if not (core_dir / "runner_suite_core.py").exists():
    print(f"SKIP  runner_suite_core.py not found in {core_dir} (run setup.bat / fetch_deps.py, "
          f"or set RUNNER_CORE_DIR)")
    sys.exit(0)
sys.path[:0] = [str(HERE), str(core_dir)]

for name, attr in (("ultralytics", "YOLO"), ("tqdm", "tqdm")):
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        setattr(m, attr, object if name == "ultralytics" else None)
        sys.modules[name] = m

import numpy as np                    # noqa: E402
import runner_suite_core as core      # noqa: E402
import merged_suite as ms             # noqa: E402

fails = []


def chk(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        fails.append(label)


def suite(guard=True):
    s = ms.MergedSuite.__new__(ms.MergedSuite)
    s.cfg = core.deep_merge(copy.deepcopy(core.DEFAULT_CONFIG), copy.deepcopy(ms.MERGED_CONFIG_EXTRA))
    s.cfg["completeness_guard"]["edge_truncation_guard"] = guard
    return s


H, W = 4128, 6192
KP = np.array([[100, 100, 0.9]] * 17, dtype=float)       # every keypoint confident: a "complete" athlete
s = suite()

chk("guard is on by default", ms.MERGED_CONFIG_EXTRA["completeness_guard"]["edge_truncation_guard"] is True)

# the two cases the engine gets wrong on its own
cut_close_up = (4900.0, 0.0, 6192.0, 4128.0)             # big close-up cut by the right and top edge
chk("engine alone: such a close-up takes the full-frame path (the gap)",
    core.HighResRunnerSuite._is_full_frame_runner(suite(False), cut_close_up, (H, W)) is True)
chk("with the guard: it no longer takes the full-frame path, so the completeness guard applies",
    s._is_full_frame_runner(cut_close_up, (H, W)) is False)
chk("...and the guard then demotes it", s._assess_completeness(KP, cut_close_up, (H, W))["ok_for_premium"] is False)

right_only = (5100.0, 900.0, 6192.0, 3900.0)
chk("engine alone: a runner cut by the right edge is fine for Premium (the gap)",
    suite(False)._assess_completeness(KP, right_only, (H, W))["ok_for_premium"] is True)
c = s._assess_completeness(KP, right_only, (H, W))
chk("with the guard: cut by the right edge is demoted, reason truncated_side",
    c["ok_for_premium"] is False and c["reason"] == "truncated_side" and c["truncated_side"] is True)
c = s._assess_completeness(KP, (0.0, 900.0, 900.0, 3900.0), (H, W))
chk("...and so is one cut by the left edge", c["ok_for_premium"] is False and c["reason"] == "truncated_side")

# what must keep working
whole = (1000.0, 200.0, 3000.0, 4000.0)                  # tall, fully inside the frame
chk("a whole close-up runner still takes the full-frame path", s._is_full_frame_runner(whole, (H, W)) is True)
chk("a complete runner inside the frame is still Premium-eligible",
    s._assess_completeness(KP, (2000.0, 500.0, 2800.0, 3000.0), (H, W))["ok_for_premium"] is True)
chk("feet cut by the bottom edge are still fine (the engine's own rule)",
    s._assess_completeness(KP, (2000.0, 500.0, 2800.0, float(H)), (H, W))["ok_for_premium"] is True
    and not s._cut_by_frame((2000.0, 500.0, 2800.0, float(H)), (H, W), top=True))

# margin: engine's edge_margin_px (6)
chk("a box 7 px from the edge is not cut off", not s._cut_by_frame((5000.0, 900.0, W - 7.0, 3900.0), (H, W)))
chk("a box 6 px from the edge is cut off", s._cut_by_frame((5000.0, 900.0, W - 6.0, 3900.0), (H, W)))
s2 = suite(); s2.cfg["completeness_guard"]["edge_margin_px"] = 30
chk("the margin follows completeness_guard.edge_margin_px", s2._cut_by_frame((5000.0, 900.0, W - 20.0, 3900.0), (H, W)))

# switch off: the engine's own behaviour, unchanged
off = suite(False)
chk("guard off: full-frame path and completeness behave exactly like the engine",
    off._is_full_frame_runner(cut_close_up, (H, W)) is True
    and off._assess_completeness(KP, right_only, (H, W))["ok_for_premium"] is True
    and "truncated_side" not in off._assess_completeness(KP, right_only, (H, W)))

# the engine's own top-edge rule is untouched
top = (2000.0, 0.0, 2800.0, 3000.0)
c = s._assess_completeness(KP, top, (H, W))
chk("the engine's own top-edge rule still reports truncated_head", c["ok_for_premium"] is False and c["reason"] == "truncated_head")

# ---------------------------------------------------------------- the engine's real photo loop
# A synthetic textured photo and a fake detector: the engine's own accept/demote
# logic runs, so this shows what ends up in A_Premium with the guard off and on.
import tempfile                       # noqa: E402
import threading                      # noqa: E402
import cv2                            # noqa: E402

PW, PH = 3000, 2000
img = (np.random.default_rng(0).random((PH, PW, 3)) * 255).astype(np.uint8)

class T:                                                        # tiny stand-in for a torch tensor
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
class Boxes:
    def __init__(self, box): self.xyxy = T([box]); self.conf = T([0.9])
    def __len__(self): return 1
def result(box, kp_xy):
    kp = np.array([[x, y, 0.95] for x, y in kp_xy], dtype=float)
    return types.SimpleNamespace(boxes=Boxes(box),
                                 keypoints=types.SimpleNamespace(data=T([kp])))
def body_kps(x1, y1, x2, y2):                                   # 17 plausible keypoints spread over the box
    xs = np.linspace(x1 + 20, x2 - 20, 5); ys = np.linspace(y1 + 20, y2 - 20, 17)
    return [(xs[i % 5], ys[i]) for i in range(17)]

def premium_crops(box, guard):
    s = ms.MergedSuite.__new__(ms.MergedSuite)
    s.cfg = core.deep_merge(copy.deepcopy(core.DEFAULT_CONFIG), copy.deepcopy(ms.MERGED_CONFIG_EXTRA))
    s.cfg["completeness_guard"]["edge_truncation_guard"] = guard
    s.cfg["quality_scoring"]["classes"] = {"premium": 1, "good": 0, "review": 0}      # score gate out of the way: we test the guard
    s.cfg["selection_filters"]["premium_min_sharpness"] = 0
    s.cfg["selection_filters"]["min_sharpness_threshold"] = 0
    s.cfg["selection_filters"]["hard_reject_fence"] = False
    s.cfg["image_quality"]["enhance_by_class"] = {"premium": "none", "good": "none", "review": "none"}
    s.cfg["image_quality"]["target_height"] = 800
    s.cfg["debug"]["write_csv"] = False
    s.stats = {}; s.stats_lock = threading.Lock(); s.report_rows = []; s.report_lock = threading.Lock()
    s.sr_lock = threading.Lock(); s._face_restorer = None; s._finishing = False
    out = Path(tempfile.mkdtemp()); s.output_folder = out; s.cfg["paths"]["output_folder"] = str(out)
    s.rejects_folder = out / "_r"
    s.premium_sink = None
    src = out / "photo.jpg"; cv2.imwrite(str(src), img)
    s._process_detection_results(img, src, [result(box, body_kps(*box))])
    prem = list((out / "A_Premium").glob("*.jpg")) if (out / "A_Premium").exists() else []
    return len(prem)


for name, box, expect_on in [
        ("a whole runner inside the frame", (1100.0, 300.0, 1900.0, 1900.0), 1),
        ("a runner cut by the right edge", (1400.0, 500.0, 3000.0, 1700.0), 0),
        ("a big close-up cut by the right and top edge", (1500.0, 0.0, 3000.0, 2000.0), 0)]:
    chk(f"loop, guard off (the engine alone): {name} -> Premium crop", premium_crops(box, False) == 1)
    chk(f"loop, guard on: {name} -> {'Premium crop' if expect_on else 'no Premium crop'}", premium_crops(box, True) == expect_on)

print("\nFAILED:" if fails else "\nALL PASS", fails or "")
sys.exit(1 if fails else 0)
