# -*- coding: utf-8 -*-
"""Self-test for Grade Room. Run from the repo folder:

    .venv\\Scripts\\python.exe test_grade.py

Uses the real piexif when it is installed. Where it is not (a bare machine), a
tiny stand-in is used so the control flow is still exercised; the summary says
which one ran. Needs Pillow only for making test JPEGs with orientation/ICC
(ultralytics installs it)."""

from __future__ import annotations

import functools
import io
import csv
import os
import pickle
import struct
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np

REAL_PIEXIF = True
try:
    import piexif  # noqa: F401
except Exception:
    REAL_PIEXIF = False

    class _IFD:
        Orientation, Software = 274, 305

    class _EXIF:
        PixelXDimension, PixelYDimension = 40962, 40963

    fake = types.ModuleType("piexif")
    fake.ImageIFD, fake.ExifIFD = _IFD, _EXIF
    _TAG = b"FakeEx"

    def _load(data):
        if isinstance(data, str):
            data = Path(data).read_bytes()
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] in (0xE0, 0xE1):
            n = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            seg = data[pos + 4:pos + 2 + n]
            if seg[:6] == _TAG:
                return pickle.loads(seg[6:])
            pos += 2 + n
        return {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    def _dump(d):
        return _TAG + pickle.dumps(d)

    def _insert(exif, path):
        data = Path(path).read_bytes()
        seg = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
        Path(path).write_bytes(data[:2] + seg + data[2:])

    fake.load, fake.dump, fake.insert = _load, _dump, _insert
    sys.modules["piexif"] = fake

import grade_suite as gs      # noqa: E402  (after the piexif decision)
import lut as lutlib          # noqa: E402

try:
    from PIL import Image, ImageCms
except Exception:
    Image = ImageCms = None

ok_all = True


def check(label, cond, extra=""):
    extra = str(extra)
    global ok_all
    ok_all &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + label + (f"   [{extra}]" if (extra is not None and extra != "" and not cond) else ""))


def wait(cond, secs=20):
    t = time.time()
    while time.time() - t < secs:
        if cond():
            return True
        time.sleep(0.05)
    return False


def write_cube(path: Path, mapping, n=17):
    with open(path, "w") as f:
        f.write(f"LUT_3D_SIZE {n}\n")
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    o = mapping(r / (n - 1), g / (n - 1), b / (n - 1))
                    f.write("%.6f %.6f %.6f\n" % o)


def jpeg(path: Path, w=300, h=200, bgr=None, seed=0, **save_kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if bgr is not None:
        arr = np.zeros((h, w, 3), np.uint8)
        arr[:] = bgr
    else:
        arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    if save_kw and Image is not None:
        Image.fromarray(arr[..., ::-1]).save(path, "JPEG", quality=95, **save_kw)
    else:
        cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def mean_bgr(path: Path):
    return cv2.imread(str(path)).reshape(-1, 3).mean(0)


tmp = Path(tempfile.mkdtemp())
luts = tmp / "luts"
luts.mkdir()
write_cube(luts / "identity.cube", lambda r, g, b: (r, g, b))
write_cube(luts / "swap.cube", lambda r, g, b: (b, g, r))        # red <-> blue

def opts(inp, out, lut="swap.cube", **kw):
    return gs.Options(Path(inp), Path(out), lut, luts, **kw)


def run_job(o, timeout=60):
    job = gs.Job(o)
    t = threading.Thread(target=job.run, daemon=True)
    t.start()
    t.join(timeout)
    return job


# ---------------------------------------------------------------- LUT engine
print("piexif:", "real" if REAL_PIEXIF else "stand-in (install piexif for the full check)")
swap = lutlib.parse_cube(luts / "swap.cube")
ident = lutlib.parse_cube(luts / "identity.cube")
t0 = time.time()
packed_swap = gs.build_packed(swap, 1.0)
print(f"(packed table build: {time.time() - t0:.1f}s)")
rng = np.random.default_rng(3)
img = rng.integers(0, 255, (333, 517, 3), dtype=np.uint8)
dense_swap = lutlib.build_dense_lut_bgr(swap, 1.0)
check("packed LUT is bit-identical to the existing dense LUT",
      np.array_equal(gs.apply_packed(img, packed_swap), lutlib.apply_dense_bgr(img, dense_swap)))
check("strips of any height give the same result",
      np.array_equal(gs.apply_packed(img, packed_swap, rows=7), gs.apply_packed(img, packed_swap, rows=256)))
sw = gs.apply_packed(np.full((4, 4, 3), (10, 100, 200), np.uint8), packed_swap)
check("red and blue swap LUT maps BGR (10,100,200) -> (200,100,10)",
      np.abs(sw.reshape(-1, 3).astype(int) - np.array([200, 100, 10])).max() <= 2, sw[0, 0])
packed_id = gs.build_packed(ident, 1.0)
check("identity LUT leaves pixels alone (+-1)",
      np.abs(gs.apply_packed(img, packed_id).astype(int) - img).max() <= 1)
half = gs.build_packed(swap, 0.5)
mid = gs.apply_packed(np.full((2, 2, 3), (0, 0, 200), np.uint8), half)[0, 0]
check("strength 0.5 lands halfway between original and graded", abs(int(mid[0]) - 100) <= 3 and abs(int(mid[2]) - 100) <= 3, mid)

# ---------------------------------------------------------------- finishing
f = gs.finish_image(np.zeros((2000, 3000, 3), np.uint8), 4000, "none")
check("finish scales to the target height and keeps the aspect ratio (no crop)", f.shape[:2] == (4000, 6000), f.shape)
f = gs.finish_image(np.zeros((6000, 4000, 3), np.uint8), 4000, "none")
check("finish scales down as well", f.shape[:2] == (4000, 2667), f.shape)
noisy = rng.integers(0, 255, (400, 300, 3), dtype=np.uint8)
check("sharpen 'strong' changes pixels, 'none' does not",
      not np.array_equal(gs.finish_image(noisy, 400, "strong"), noisy) and np.array_equal(gs.finish_image(noisy, 400, "none"), noisy))

# ---------------------------------------------------------------- one file
src = jpeg(tmp / "one" / "a.jpg", bgr=(10, 100, 200))
o = opts(tmp / "one", tmp / "one_out")
st, _ = gs.process_file(src, tmp / "one_out" / "a.jpg", o, packed_swap)
m = mean_bgr(tmp / "one_out" / "a.jpg")
check("process_file grades and writes the file", st == "ok" and np.abs(m - np.array([200, 100, 10])).max() < 4, m)
check("no .part leftovers", not list((tmp / "one_out").glob("*.part")))
st, _ = gs.process_file(src, tmp / "one_out" / "a.jpg", o, packed_swap)
check("an existing output is skipped, not redone", st == "exists")
st, _ = gs.process_file(src, tmp / "one_out" / "a.jpg", opts(tmp / "one", tmp / "one_out", overwrite=True), packed_swap)
check("overwrite redoes it", st == "ok")

# already-graded recognition (the stamp lives in EXIF)
st, msg = gs.process_file(tmp / "one_out" / "a.jpg", tmp / "one_out2" / "a.jpg", o, packed_swap)
check("a file Grade Room already graded is recognised and skipped", st == "graded", (st, msg))
st, _ = gs.process_file(tmp / "one_out" / "a.jpg", tmp / "one_out3" / "a.jpg", opts(tmp / "x", tmp / "y", regrade=True), packed_swap)
check("regrade overrides the skip", st == "ok")

# orientation: OpenCV rotates the pixels, so the tag must not survive
if Image is not None:
    (tmp / "rot").mkdir(parents=True, exist_ok=True)
    ex = Image.Exif()
    ex[0x0112] = 6
    arr = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp / "rot" / "r.jpg", "JPEG", exif=ex, quality=95)
    st, _ = gs.process_file(tmp / "rot" / "r.jpg", tmp / "rot_out" / "r.jpg", o, packed_swap)
    h, w = cv2.imread(str(tmp / "rot_out" / "r.jpg")).shape[:2]
    check("orientation 6 source: pixels come out rotated (300 tall x 200 wide)", (h, w) == (300, 200), (h, w))
    try:
        tag = Image.open(tmp / "rot_out" / "r.jpg").getexif().get(0x0112)
    except Exception:
        tag = None
    check("the Orientation tag is gone from the output", tag is None, tag)

    # ICC profile carried through
    srgb = ImageCms.createProfile("sRGB")
    icc = ImageCms.ImageCmsProfile(srgb).tobytes()
    p = tmp / "icc" / "s.jpg"
    p.parent.mkdir()
    Image.fromarray(arr).save(p, "JPEG", icc_profile=icc, quality=95)
    seen = []
    st, detail = gs.process_file(p, tmp / "icc_out" / "s.jpg", o, packed_swap, on_profile=seen.append)
    check("an sRGB profile is copied byte for byte and not flagged",
          gs.extract_icc((tmp / "icc_out" / "s.jpg").read_bytes()) == icc and not seen)
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    p2 = tmp / "icc" / "l.jpg"
    Image.fromarray(arr).save(p2, "JPEG", icc_profile=lab, quality=95)
    seen = []
    gs.process_file(p2, tmp / "icc_out" / "l.jpg", o, packed_swap, on_profile=seen.append)
    check("a non-sRGB profile is kept and reported", gs.extract_icc((tmp / "icc_out" / "l.jpg").read_bytes()) == lab and len(seen) == 1, seen)
    big = bytes(range(256)) * 700                       # 179 KB: needs several APP2 chunks
    out_bytes = gs.insert_icc(cv2.imencode(".jpg", arr)[1].tobytes(), big)
    check("an ICC profile larger than one JPEG segment round-trips", gs.extract_icc(out_bytes) == big)
    check("and the JPEG still decodes", cv2.imdecode(np.frombuffer(out_bytes, np.uint8), 1) is not None)

# ---------------------------------------------------------------- folders
check("output inside input is refused", gs.check_folders(tmp / "in", tmp / "in" / "out") is not None)
(tmp / "in").mkdir(exist_ok=True)
check("input inside output is refused", gs.check_folders(tmp / "in", tmp) is not None)
check("input == output is refused", gs.check_folders(tmp / "in", tmp / "in") is not None)
check("a sibling output is fine", gs.check_folders(tmp / "in", tmp / "in_graded") is None)

# ---------------------------------------------------------------- a whole tree
tree = tmp / "tree"
for i, rel in enumerate(["a.jpg", "b.JPG", "sub/c.jpeg", "sub/deep/d.jpg"]):
    jpeg(tree / rel, seed=i)
(tree / "notes.txt").write_text("not a photo")
(tree / "sub" / "bad.jpg").write_bytes(b"this is not a jpeg at all")
out = tmp / "tree_graded"
job = run_job(opts(tree, out, workers=3))
s = job.snapshot()
check("batch finishes", s["status"] == "done", s["status"] + " " + s["error"])
check("4 photos graded, the corrupt one failed, the text file ignored",
      (s["done"], s["failed"], s["found"]) == (4, 1, 5), (s["done"], s["failed"], s["found"]))
check("output mirrors the input tree and names",
      all((out / r).exists() for r in ["a.jpg", "b.JPG", "sub/c.jpeg", "sub/deep/d.jpg"]) and not (out / "notes.txt").exists())
check("originals untouched", (tree / "a.jpg").stat().st_size > 0 and not list(tree.rglob("*.part")))
rep = (out / gs.REPORT_NAME).read_text(encoding="utf-8")
check("the report lists every file with its status", rep.count("\n") == 6 and "unreadable" in rep and "bad.jpg" in rep)
job2 = run_job(opts(tree, out, workers=3))
s2 = job2.snapshot()
check("running it again skips what is done (restart-safe)", (s2["done"], s2["exists"]) == (0, 4) and s2["status"] == "done", s2)

# native vs finished
job3 = run_job(opts(tree, tmp / "tree_fin", workers=2, finish=True, target_height=800, sharpen="medium"))
h, w = cv2.imread(str(tmp / "tree_fin" / "a.jpg")).shape[:2]
check("finish mode writes 800 px tall, aspect kept (300x200 -> 1200x800)", (h, w) == (800, 1200), (h, w))
h, w = cv2.imread(str(out / "a.jpg")).shape[:2]
check("native mode keeps the source size", (h, w) == (200, 300))

# ---------------------------------------------------------------- enlarged preview (server side)
import grade_ui as gui          # noqa: E402
enl = tmp / "enl"
jpeg(enl / "a.jpg", w=1600, h=1000, seed=21)
if Image is not None:
    ex = Image.Exif(); ex[0x0112] = 6                                   # stored landscape, displayed portrait
    Image.fromarray(np.random.default_rng(22).integers(0, 255, (2000, 3000, 3), dtype=np.uint8)).save(
        enl / "rot.jpg", "JPEG", quality=90, exif=ex)
else:
    jpeg(enl / "rot.jpg", w=3000, h=2000, seed=22)
app = gui.App(dict(gui.DEFAULTS, lut_folder=str(luts)))
app.prepare(str(enl))
names = [p_.name for p_ in app.samples]
ia, ir = names.index("a.jpg"), names.index("rot.jpg")
def _dims(b):
    im = cv2.imdecode(np.frombuffer(b, np.uint8), 1)
    return im.shape[1], im.shape[0]
big = app.tile("swap.cube", ia, "big", 1.0, False, 500)
check("the enlarged view is rendered at the height asked for (1600x1000 -> 800x500)", big and _dims(big) == (800, 500), big and _dims(big))
big = app.tile("swap.cube", ia, "big", 1.0, False, 5000)
check("the requested height is capped at 3000 (and never enlarges beyond the photo)", big and _dims(big)[1] == 1000, big and _dims(big))
check("without h it falls back to 1400 tall (capped by the photo)", _dims(app.tile("swap.cube", ia, "big", 1.0, True, 0))[1] == 1000)
n_before = len(app._bigbase)
app.tile("swap.cube", ia, "big", 0.5, False, 500)
check("changing the strength reuses the decoded photo", len(app._bigbase) == n_before)
org = app.tile("swap.cube", ia, "big", 1.0, True, 500)
gr = app.tile("swap.cube", ia, "big", 1.0, False, 500)
check("original and graded differ", org != gr and _dims(org) == _dims(gr))
if Image is not None:
    rot = app.tile("swap.cube", ir, "big", 1.0, True, 1000)
    check("EXIF orientation is honoured with the reduced decode (3000x2000, rotated -> 667x1000)",
          rot and _dims(rot) in ((667, 1000), (666, 1000)), rot and _dims(rot))
check("an unknown sample index gives None, not a crash", app.tile("swap.cube", 99, "big", 1.0, False, 500) is None)
try:
    app.tile("nope.cube", ia, "big", 1.0, False, 500)
    check("an unknown look on the enlarged view raises, so the page can show why", False)
except Exception:
    check("an unknown look on the enlarged view raises, so the page can show why", True)
check("thumbnails still work", app.tile("swap.cube", ia, "thumb", 1.0, False) is not None)

# ---------------------------------------------------------------- watching
wdir = tmp / "watch"
wdir.mkdir()
jpeg(wdir / "w0.jpg", seed=10)
o = opts(wdir, tmp / "watch_out", workers=2, watch=True, poll_sec=0.3)
job = gs.Job(o)
th = threading.Thread(target=job.run, daemon=True)
th.start()
check("watch mode grades what is there", wait(lambda: job.snapshot()["done"] == 1))
check("then reports 'watching'", wait(lambda: job.snapshot()["status"] == "watching"))
jpeg(wdir / "later" / "w1.jpg", seed=11)
check("and picks up a file that arrives later", wait(lambda: job.snapshot()["done"] == 2) and (tmp / "watch_out" / "later" / "w1.jpg").exists())
partial = wdir / "w2.jpg"
data = cv2.imencode(".jpg", np.random.default_rng(5).integers(0, 255, (200, 300, 3), dtype=np.uint8))[1].tobytes()
partial.write_bytes(data[: len(data) // 2])                 # a file still being written
time.sleep(0.4)
partial.write_bytes(data[: 3 * len(data) // 4])
time.sleep(0.4)
before = job.snapshot()["failed"]
partial.write_bytes(data)
check("a half-written file is not graded until its size settles", wait(lambda: job.snapshot()["done"] == 3) and job.snapshot()["failed"] == before)
job.stop.set()
th.join(10)

# ---------------------------------------------------------------- worker count
check("default is automatic (0), at most 64", gs.DEFAULT_WORKERS == 0 and gs.MAX_WORKERS == 64
      and gs.Options(input_dir=tmp, output_dir=tmp, lut="x", lut_folder=tmp).workers == 0)
_real_pf = gs.process_file
_conc = {"now": 0, "max": 0}
_conc_lock = threading.Lock()
def _slow_pf(src, dst, opts_, packed, on_profile=None, on_size=None, on_timing=None):
    with _conc_lock:
        _conc["now"] += 1
        _conc["max"] = max(_conc["max"], _conc["now"])
    time.sleep(0.12)
    with _conc_lock:
        _conc["now"] -= 1
    return "ok", ""
gs.process_file = _slow_pf
try:
    wk = tmp / "wk"
    for i in range(24):
        jpeg(wk / f"p{i}.jpg", seed=100 + i, w=40, h=30)
    _conc.update(now=0, max=0)
    j1 = run_job(opts(wk, tmp / "wk_out1", workers=2))
    check("workers=2 never runs more than 2 photos at once, and does use 2",
          _conc["max"] == 2 and j1.snapshot()["done"] == 24, dict(_conc))
    _conc.update(now=0, max=0)
    j2 = run_job(opts(wk, tmp / "wk_out2", workers=6))
    check("workers=6 runs 6 at once", _conc["max"] == 6 and j2.snapshot()["done"] == 24, dict(_conc))
    # change the count while a run is in progress
    _conc.update(now=0, max=0)
    jl = gs.Job(opts(wk, tmp / "wk_out3", workers=1, watch=True, poll_sec=0.2))
    tl = threading.Thread(target=jl.run, daemon=True); tl.start()
    check("live: starts with one worker", wait(lambda: jl.snapshot()["done"] >= 3) and _conc["max"] == 1, dict(_conc))
    _idle = gs.Job(opts(wk, tmp / "wk_idle", workers=3))      # clamping is checked on a job that is not running,
    check("live: set_workers clamps to 1..64",                  # so the cap of the running one is never really lifted
          _idle.set_workers(0) == 1 and _idle.set_workers(999) == 64)
    check("live: set_workers accepts a normal value", jl.set_workers(5) == 5)
    for i in range(24, 60):
        jpeg(wk / f"q{i}.jpg", seed=200 + i, w=40, h=30)
    check("live: raising it takes effect on a running job", wait(lambda: _conc["max"] >= 5, 20) and _conc["max"] <= 5, dict(_conc))
    check("live: the snapshot reports the count", jl.snapshot()["workers"] == 5)
    wait(lambda: jl.snapshot()["done"] >= 60, 30)
    jl.set_workers(1)
    jl.stop.set(); tl.join(10)
    check("live: the job stops cleanly", not tl.is_alive() and jl.snapshot()["status"] == "stopped")
finally:
    gs.process_file = _real_pf

# ---------------------------------------------------------------- automatic workers, single write
_real_threads, _real_mem = gs.cpu_threads, gs._avail_mem
try:
    gs.cpu_threads = lambda: 32; gs._avail_mem = lambda: None
    check("auto: one worker per CPU thread (32 threads -> 32)", gs.auto_workers() == 32 and gs.auto_workers(24) == 32)
    gs.cpu_threads = lambda: 200
    check("auto: never more than 64", gs.auto_workers() == 64)
    gs.cpu_threads = lambda: 32; gs._avail_mem = lambda: 4e9
    check("auto: fewer when 24 MP photos would not fit in 4 GB", gs.auto_workers(24) == 8 and gs.auto_workers(0) == 32, gs.auto_workers(24))
    gs._avail_mem = lambda: 1e6
    check("auto: never fewer than 1", gs.auto_workers(45) == 1)
    gs._avail_mem = lambda: 64e9
    check("auto: plenty of memory keeps every CPU thread busy", gs.auto_workers(24) == 32)
finally:
    gs.cpu_threads, gs._avail_mem = _real_threads, _real_mem

am = tmp / "automem"
for i in range(12):
    jpeg(am / f"m{i}.jpg", w=2000, h=1500, seed=300 + i)                # 3 MP each
jauto = gs.Job(opts(am, tmp / "automem_out", workers=0))
check("a job with workers=0 starts on the automatic count", jauto.auto_workers and jauto.workers == gs.auto_workers())
try:
    gs.cpu_threads = lambda: 16; gs._avail_mem = lambda: 400e6         # 400 MB free, ~93 MB a worker at 3 MP
    ja = run_job(opts(am, tmp / "automem_out", workers=0))
    sa = ja.snapshot()
    check("auto: the first batch's photo size limits the workers to what fits (400 MB free, 3 MP -> 3)",
          ja.workers == 3 and sa["done"] == 12 and sa["auto_workers"], (ja.workers, sa["done"]))
    check("...and the log says why", any("limited by free memory" in m for m in sa["log"]), sa["log"])
    jb = run_job(opts(am, tmp / "automem_out_b", workers=8))
    check("an explicit count is kept, with a memory warning in the log",
          jb.workers == 8 and any("WARNING" in m and "GB" in m for m in jb.snapshot()["log"]) and jb.snapshot()["done"] == 12)
finally:
    gs.cpu_threads, gs._avail_mem = _real_threads, _real_mem
jm = gs.Job(opts(am, tmp / "automem_out_c", workers=0))
jm.set_workers(2)
check("choosing a count by hand ends the automatic mode", jm.workers == 2 and not jm.auto_workers)
sn = gs.Job(opts(am, tmp / "automem_out_d")).snapshot()
check("the snapshot reports CPU threads and a CPU figure (or None without psutil)",
      sn["cpu_threads"] == (os.cpu_count() or 2) and (sn["cpu"] is None or 0 <= sn["cpu"] <= 100))

# EXIF and colour profile are put in place in memory, and the file is written once
base = cv2.imencode(".jpg", np.zeros((20, 30, 3), np.uint8))[1].tobytes()
exif_b = gs.exif_segment(b"Exif\x00\x00" + b"X" * 100)
icc_b = gs.icc_segments(b"P" * 200)
out_b = gs.insert_segments(base, exif_b, icc_b)
def _markers(b):
    pos, seq = 2, []
    while b[pos] == 0xFF and b[pos + 1] not in (0xDA, 0xD9):
        seq.append(b[pos + 1]); pos += 2 + struct.unpack(">H", b[pos + 2:pos + 4])[0]
    return seq
mk = _markers(out_b)
check("segments land after the JFIF header, EXIF before the profile: APP0, APP1, APP2, then the image tables",
      mk[:3] == [0xE0, 0xE1, 0xE2] and 0xDB in mk[3:], [hex(m) for m in mk])
check("the result still decodes and the profile comes back out", cv2.imdecode(np.frombuffer(out_b, np.uint8), 1) is not None
      and gs.extract_icc(out_b) == b"P" * 200)
try:
    gs.exif_segment(b"E" * 70000); big_ok = False
except ValueError:
    big_ok = True
check("...an EXIF block larger than one segment raises ValueError", big_ok)
_pi_insert = gs.piexif.insert
def _boom(*a, **k): raise AssertionError("piexif.insert must not be used any more")
gs.piexif.insert = _boom
try:
    onew = tmp / "once"; jpeg(onew / "a.jpg", seed=5); jpeg(onew / "b.jpg", seed=6)
    jo = run_job(opts(onew, tmp / "once_out", workers=2))
    so = jo.snapshot()
    check("photos are graded without any second pass over the written file", so["done"] == 2 and so["failed"] == 0, so["log"])
    ex = gs._load_exif((tmp / "once_out" / "a.jpg").read_bytes())
    check("...and still carry the stamp", gs.is_stamped(ex))
finally:
    gs.piexif.insert = _pi_insert
check("the report file handle is closed when the run ends", jo._report_fh is None and (tmp / "once_out" / gs.REPORT_NAME).exists())

# ---------------------------------------------------------------- the packed table: fast build, same bytes
def _cube3(n, dmin=(0, 0, 0), dmax=(1, 1, 1), noisy=False, seed=3):
    g = np.linspace(0, 1, n, dtype=np.float32)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    t = np.stack([R ** 0.9 * 0.9 + 0.05 * G, G ** 1.1, B * 0.8 + 0.1 * R], -1).astype(np.float32)
    if noisy:
        t = np.random.default_rng(seed).random((n, n, n, 3)).astype(np.float32) * 1.2 - 0.1   # out of range on purpose
    return {"dim": 3, "size": n, "title": "", "dmin": np.array(dmin, np.float32),
            "dmax": np.array(dmax, np.float32), "table": t}
def _cube1(n, dmin=(0, 0, 0), dmax=(1, 1, 1)):
    x = np.linspace(0, 1, n, dtype=np.float32)
    c = np.stack([x ** 0.8, np.sqrt(x), x ** 1.3], -1) * 1.1 - 0.05
    return {"dim": 1, "size": n, "title": "", "dmin": np.array(dmin, np.float32),
            "dmax": np.array(dmax, np.float32), "curve": c.astype(np.float32)}
_t0 = time.time()
_same = []
for _name, _p in [("17", _cube3(17)), ("33 noisy", _cube3(33, noisy=True)), ("33 domain", _cube3(33, (0.05, 0, 0.1), (0.9, 1, 0.95))),
                  ("2", _cube3(2, noisy=True)), ("1D", _cube1(256)), ("1D domain", _cube1(64, (0.1, 0, 0), (0.9, 1, 1)))]:
    for _s in (1.0, 0.37):
        _same.append((_name, _s, np.array_equal(gs.build_packed(_p, _s), gs.build_packed_reference(_p, _s))))
check("the fast table build gives exactly the same bytes as evaluating every colour (3D and 1D, with a domain, out-of-range values, strength 1 and 0.37)",
      all(x[2] for x in _same), [x for x in _same if not x[2]])
_p33 = _cube3(33)
_tb = time.time(); gs.build_packed(_p33, 1.0); _tb = time.time() - _tb
check("...and it takes about a second or less, not several", _tb < 3.0, f"{_tb:.2f}s")

# grading in place gives the same picture as grading into a new array
_im = np.random.default_rng(9).integers(0, 255, (321, 517, 3), dtype=np.uint8)
_pk = gs.build_packed(_cube3(17), 1.0)
_a = gs.apply_packed(_im, _pk); _b = _im.copy(); _r = gs.apply_packed(_b, _pk, inplace=True)
check("apply_packed(inplace=True) equals the normal result and returns the same array", np.array_equal(_a, _b) and _r is _b)
check("...and the normal call leaves its input untouched", np.array_equal(_im, np.random.default_rng(9).integers(0, 255, (321, 517, 3), dtype=np.uint8)))

# per-stage timing
tj = run_job(opts(am, tmp / "timing_out", workers=2))
tm = tj.snapshot()["timing"]
check("a run reports where the time goes: every stage, wall and cpu, and a waiting share",
      tm and tm["photos"] == 12 and [r["stage"] for r in tm["stages"]] == ["read", "exif", "decode", "grade", "encode", "assemble", "write"]
      and all(r["wall"] >= 0 and r["cpu"] >= 0 for r in tm["stages"]) and 0 <= tm["waiting"] <= 1, tm)
check("...and the log ends with a per-photo line", any("per photo (average)" in m for m in tj.snapshot()["log"]))
tf = run_job(opts(am, tmp / "timing_out_f", workers=2, finish=True, target_height=500))
check("with finishing on, 'finish' is a stage too", "finish" in [r["stage"] for r in tf.snapshot()["timing"]["stages"]])

# the speed test command
import contextlib
bt = tmp / "benchin"
for _i in range(6):
    jpeg(bt / f"b{_i}.jpg", w=800, h=600, seed=400 + _i)
_rep = tmp / "bench_result.txt"
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    gs.bench_main([str(bt), "--lut", "swap.cube", "--lut-folder", str(luts), "--photos", "6", "--workers", "1,2",
                   "--out", str(tmp), "--report", str(_rep)])
_txt = _buf.getvalue()
check("the speed test runs and prints its sections", all(k in _txt for k in
      ["Reading the photos", "Building the look", "One worker, one photo at a time", "Writing", "Grading with several workers",
       "What each part could deliver", "Verdict"]), _txt[-400:])
check("...saves the same text to a file", _rep.exists() and "Verdict" in _rep.read_text(encoding="utf-8"))
check("...and leaves no temporary files behind", not list(tmp.glob("_gradeoom_bench_*")))

# ---------------------------------------------------------------- small-photo warning
# jpeg_size reads the header only
sz = tmp / "sizes"
jpeg(sz / "base.jpg", w=123, h=45)
check("jpeg_size reads a baseline JPEG's header", gs.jpeg_size(sz / "base.jpg") == (123, 45))
prog = sz / "prog.jpg"
prog.write_bytes(cv2.imencode(".jpg", np.zeros((30, 200, 3), np.uint8), [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tobytes())
check("jpeg_size reads a progressive JPEG", gs.jpeg_size(prog) == (200, 30))
raw = (sz / "base.jpg").read_bytes()
big_app = b"\xff\xe1" + (65000).to_bytes(2, "big") + b"Exif\x00\x00" + b"\x00" * (65000 - 8)
(sz / "bigexif.jpg").write_bytes(raw[:2] + big_app + big_app + raw[2:])
check("jpeg_size skips large EXIF/ICC segments", gs.jpeg_size(sz / "bigexif.jpg") == (123, 45)
      and cv2.imread(str(sz / "bigexif.jpg")) is not None)
(sz / "trunc.jpg").write_bytes(raw[:6])
(sz / "notjpeg.jpg").write_bytes(b"GIF89a....")
(sz / "empty.jpg").write_bytes(b"")
check("jpeg_size gives None for truncated, non-JPEG and empty files",
      all(gs.jpeg_size(sz / n) is None for n in ["trunc.jpg", "notjpeg.jpg", "empty.jpg", "missing.jpg"]))
(sz / "huge_hdr.jpg").write_bytes(raw[:2] + b"\xff\xc0\x00\x0b\x08" + (30000).to_bytes(2, "big") * 2 + b"\x01\x11\x00" + b"\xff\xda")
check("the header is read without decoding (a 30000x30000 header is fine)", gs.jpeg_size(sz / "huge_hdr.jpg") == (30000, 30000))

# the rule: longer side strictly under the limit; orientation does not matter
check("3999 on the long side is small, 4000 is not (landscape and portrait)",
      gs.is_small(3999, 10, 4000) and gs.is_small(10, 3999, 4000)
      and not gs.is_small(4000, 10, 4000) and not gs.is_small(10, 4000, 4000) and not gs.is_small(6000, 4000, 4000))
check("0 turns the check off", not gs.is_small(10, 10, 0))

edge = tmp / "edge"
jpeg(edge / "land3999.jpg", w=3999, h=10, seed=1)
jpeg(edge / "port3999.jpg", w=10, h=3999, seed=2)
jpeg(edge / "land4000.jpg", w=4000, h=10, seed=3)
jpeg(edge / "sub" / "tiny.jpg", w=300, h=200, seed=4)
(edge / "broken.jpg").write_bytes(b"not a jpeg")
ents = gs.scan_sizes(edge)
sm = gs.summarize_sizes(ents, 4000)
check("scan_sizes lists every JPEG, with sizes and a 0x0 for the unreadable one",
      len(ents) == 5 and dict((e[0], (e[1], e[2])) for e in ents)["sub" + os.sep + "tiny.jpg"] == (300, 200)
      and dict((e[0], (e[1], e[2])) for e in ents)["broken.jpg"] == (0, 0))
check("summary: 3 small (the corrupt one is not counted), smallest 300, sorted smallest first",
      (sm["total"], sm["small"], sm["unreadable"], sm["smallest"]) == (5, 3, 1, 300)
      and sm["files"][0]["file"].endswith("tiny.jpg"), sm)
check("summary follows the limit", gs.summarize_sizes(ents, 300)["small"] == 0 and gs.summarize_sizes(ents, 0)["small"] == 0
      and gs.summarize_sizes(ents, 5000)["small"] == 4)

eo = tmp / "edge_out"
jw = run_job(opts(edge, eo, workers=2))                  # default limit 4000
sw = jw.snapshot()
check("a run counts the small photos while grading (3999 both ways and the 300 px one)",
      sw["small"] == 3 and sw["small_min"] == 300 and sw["warn_below"] == 4000, (sw["small"], sw["small_min"]))
check("the 4000 px photo is not flagged and the corrupt one fails, not 'small'",
      sw["done"] == 4 and sw["failed"] == 1)
check("the snapshot lists the small photos",
      sorted(f["w"] * f["h"] for f in sw["small_files"]) == sorted([3999 * 10, 10 * 3999, 300 * 200]))
check("the activity log names them and ends with a warning",
      any("small photo" in m for m in sw["log"]) and any("WARNING" in m for m in sw["log"]))
with open(eo / gs.REPORT_NAME, newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
hdr = rows[0]
check("the report has width, height and note columns", hdr[-3:] == ["width", "height", "note"])
byname = {r[1]: r for r in rows[1:]}
check("small photos carry their size and a note in the report",
      byname["sub" + os.sep + "tiny.jpg"][8:] == ["300", "200", "smaller than 4000px on the long side"]
      and byname["land4000.jpg"][8:] == ["4000", "10", ""], byname.get("land4000.jpg"))
check("an unreadable file has no size in the report", byname["broken.jpg"][8:10] == ["", ""])

j0 = run_job(opts(edge, tmp / "edge_out0", workers=2, warn_below=0))
check("warn_below=0 switches the warning off", j0.snapshot()["small"] == 0 and j0.snapshot()["done"] == 4)
j1 = run_job(opts(edge, tmp / "edge_out1", workers=2, warn_below=200))
check("a lower limit flags nothing here", j1.snapshot()["small"] == 0)
j2 = run_job(opts(edge, tmp / "edge_out2", workers=2, warn_below=5000, finish=True, target_height=50))
check("the check uses the source size even in finish mode", j2.snapshot()["small"] == 4 and j2.snapshot()["small_min"] == 300)

# an old report (without the new columns) is upgraded, not broken
old = tmp / "edge_old"; old.mkdir()
(old / gs.REPORT_NAME).write_text("time,source,status,detail,ms,lut,strength,finish\n2020-01-01 00:00:00,x.jpg,ok,,5,a.cube,1.00,no\n", encoding="utf-8")
run_job(opts(edge, old, workers=2))
with open(old / gs.REPORT_NAME, newline="", encoding="utf-8") as f:
    orows = list(csv.reader(f))
check("an older report gets the new columns and keeps its rows",
      orows[0][-3:] == ["width", "height", "note"] and orows[1][1] == "x.jpg" and len(orows[1]) == 11
      and all(len(r) == 11 for r in orows))

# a small photo arriving later in watch mode
wsm = tmp / "watch_small"; wsm.mkdir()
jpeg(wsm / "ok.jpg", w=4200, h=20, seed=1)
jwm = gs.Job(opts(wsm, tmp / "watch_small_out", workers=2, watch=True, poll_sec=0.3))
twm = threading.Thread(target=jwm.run, daemon=True); twm.start()
wait(lambda: jwm.snapshot()["status"] == "watching")
check("watch mode: nothing small yet", jwm.snapshot()["small"] == 0)
jpeg(wsm / "late.jpg", w=1000, h=700, seed=2)
check("watch mode: a small photo that arrives later is flagged", wait(lambda: jwm.snapshot()["small"] == 1) and jwm.snapshot()["small_min"] == 1000)
jwm.stop.set(); twm.join(10)
check("Stop ends watch mode cleanly", job.snapshot()["status"] == "stopped" and not th.is_alive())

# ---------------------------------------------------------------- errors
job = run_job(opts(tmp / "nope", tmp / "nope_out"))
check("a missing input ends in a clear error, not a crash", job.snapshot()["status"] == "error" and "does not exist" in job.snapshot()["error"])
job = run_job(gs.Options(tree, tmp / "o_err", "missing.cube", luts))
check("an unknown LUT ends in an error", job.snapshot()["status"] == "error")

# ---------------------------------------------------------------- speed
big_dir = tmp / "big"
for i in range(4):
    jpeg(big_dir / f"p{i}.jpg", w=5472, h=3648, seed=i)
t0 = time.time()
job = run_job(opts(big_dir, tmp / "big_out", workers=2), timeout=300)
el = time.time() - t0
s = job.snapshot()
print(f"      4 x 20 MP photos, 2 workers: {el:.1f}s total ({el / 4:.2f}s per photo wall, incl. one-off setup)")
check("20 MP photos grade without trouble", s["done"] == 4 and s["failed"] == 0, s)

print("\nALL PASS" if ok_all else "\nFAILURES", "(piexif: %s)" % ("real" if REAL_PIEXIF else "stand-in"))
sys.exit(0 if ok_all else 1)
