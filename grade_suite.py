# -*- coding: utf-8 -*-
"""
Grade Room engine -- applies a .cube LUT to folders of JPEGs. No cropping.

    input folder  ->  decode -> [resize + sharpen] -> LUT -> encode once  ->  output folder

* The output mirrors the input: same subfolders, same file names.
* Originals are never touched, and the output folder may not sit inside the
  input (or the other way round would re-feed it).
* Native size by default. "finish" adds Tether Room's stage-2 look: scale to a
  target height (aspect ratio kept -- nothing is trimmed), sharpen, then grade.
* Every output is stamped in its EXIF Software tag, so a file that has already
  been graded is recognised and skipped instead of being graded twice.
* The colour profile (ICC) is carried across; the .cube files assume sRGB, so a
  file tagged with anything else is reported.
* Restart-safe: files whose output already exists are skipped, and every
  output is written to a temporary name and renamed, so a killed run never
  leaves half a JPEG behind.

Depends on numpy, OpenCV and (optionally) piexif -- not on the crop engine, the
YOLO weights or torch.
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import lut as lutlib

try:
    import piexif  # type: ignore
except Exception:                                   # pragma: no cover
    piexif = None
try:
    import psutil  # type: ignore
except Exception:                                   # pragma: no cover
    psutil = None

STAMP = "GradeRoom:"
SHARPEN_AMOUNT = {"none": 0.0, "light": 0.35, "medium": 0.60, "strong": 0.90}
JPEG_EXTS = {".jpg", ".jpeg"}
REPORT_NAME = "_grade_report.csv"
DEFAULT_WORKERS = 0            # 0 = automatic: one per CPU thread, limited by free memory
DEFAULT_WARN_BELOW = 4000       # warn about photos whose longer side is under this (0 = off)
REPORT_HEADER = ["time", "source", "status", "detail", "ms", "lut", "strength", "finish",
                 "width", "height", "note"]
MAX_SMALL_LISTED = 500
MAX_WORKERS = 64
BYTES_PER_PIXEL = 11         # measured: about 250 MB per worker for a 24 MP photo
WORKER_OVERHEAD = 60e6


def cpu_threads() -> int:
    return max(1, os.cpu_count() or 2)


def _avail_mem() -> Optional[float]:
    """Bytes of memory that are free right now (None if psutil is not installed)."""
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().available)
    except Exception:
        return None


def memory_per_worker(max_mp: float) -> float:
    return max_mp * 1e6 * BYTES_PER_PIXEL + WORKER_OVERHEAD


def auto_workers(max_mp: float = 0.0) -> int:
    """One worker per CPU thread, held back if the photos are big enough that
    that many in flight would not fit in the free memory."""
    n = min(MAX_WORKERS, cpu_threads())
    avail = _avail_mem()
    if max_mp and avail:
        n = min(n, max(1, int(avail * 0.7 / memory_per_worker(max_mp))))
    return n
RECENT_SEC = 20.0            # window for the live "photos/s" figure


# ---------------------------------------------------------------------------
#  LUT: packed table, applied in row strips
# ---------------------------------------------------------------------------
def build_packed_reference(parsed: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """The straightforward (slow, memory-hungry) way to build the table: evaluate
    the LUT at all 16.7 million colours. Kept as the yardstick for build_packed."""
    dense_bgr = lutlib.build_dense_lut_bgr(parsed, strength)        # [b,g,r] -> BGR
    by_rgb = dense_bgr.transpose(2, 1, 0, 3)                        # [r,g,b] -> BGR
    bgra = np.empty(by_rgb.shape[:3] + (4,), np.uint8)
    bgra[..., :3] = by_rgb
    bgra[..., 3] = 255
    return np.ascontiguousarray(bgra).view(np.uint32).reshape(-1)


def build_packed(parsed: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """Expand a parsed LUT into one uint32 per colour, indexed by
    (r << 16 | g << 8 | b) and holding a BGRA pixel.

    Trilinear interpolation is separable, so the table is built axis by axis
    (red, then green, then blue, the same order and the same arithmetic as
    lut.apply_lut) instead of evaluating the whole LUT at 16.7 million colours.
    The result is identical, bit for bit, to build_packed_reference, in a
    fraction of the time and about a fifth of the memory."""
    strength = float(strength)
    rgb_k = np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0          # (256,)
    span = np.maximum(parsed["dmax"] - parsed["dmin"], 1e-8)
    norm = np.clip((rgb_k[:, None] - parsed["dmin"]) / span, 0.0, 1.0)        # (256,3) float32
    bgra = np.empty((256, 256, 256, 4), np.uint8)
    bgra[..., 3] = 255

    def finish(graded: np.ndarray, r0: int, r1: int) -> None:
        """graded: (r1-r0,256,256,3) in [0,1] as R,G,B -> bytes B,G,R into bgra."""
        if strength < 1.0:
            rgb = np.empty(graded.shape, np.float32)
            rgb[..., 0] = rgb_k[r0:r1, None, None]
            rgb[..., 1] = rgb_k[None, :, None]
            rgb[..., 2] = rgb_k[None, None, :]
            graded = rgb * (1.0 - strength) + graded * strength
        u8 = (graded * 255.0 + 0.5).astype(np.uint8)
        bgra[r0:r1, ..., 0] = u8[..., 2]
        bgra[r0:r1, ..., 1] = u8[..., 1]
        bgra[r0:r1, ..., 2] = u8[..., 0]

    if parsed["dim"] == 3:
        table = parsed["table"]                                                # (N,N,N,3) [r,g,b]
        n = table.shape[0]
        c = norm * (n - 1)
        c0 = np.clip(np.floor(c).astype(np.int64), 0, n - 1)
        c1 = np.clip(c0 + 1, 0, n - 1)
        f = c - c0                                                             # float64, as in lut._trilinear
        fr, fg, fb = f[:, 0], f[:, 1], f[:, 2]
        a_r = (table[c0[:, 0]] * (1 - fr)[:, None, None, None]
               + table[c1[:, 0]] * fr[:, None, None, None])                    # (256,N,N,3)
        b_g = (a_r[:, c0[:, 1]] * (1 - fg)[None, :, None, None]
               + a_r[:, c1[:, 1]] * fg[None, :, None, None])                   # (256,256,N,3)
        del a_r
        one_fb, b0, b1 = (1 - fb)[None, None, :, None], c0[:, 2], c1[:, 2]

        def slab(r0: int) -> None:
            r1 = min(256, r0 + 4)
            part = b_g[r0:r1]
            out = part[:, :, b0] * one_fb + part[:, :, b1] * fb[None, None, :, None]
            finish(np.clip(out.astype(np.float32), 0.0, 1.0), r0, r1)     # lut.apply_lut keeps float32 too

        starts = list(range(0, 256, 4))
        workers = max(1, min(8, cpu_threads()))
        if workers == 1:
            for r0 in starts:
                slab(r0)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(slab, starts))
    else:
        curve = parsed["curve"]
        xs = np.linspace(0.0, 1.0, curve.shape[0], dtype=np.float32)
        v = np.empty((256, 3), np.float32)
        for ch in range(3):
            v[:, ch] = np.interp(norm[:, ch], xs, curve[:, ch])
        v = np.clip(v, 0.0, 1.0)
        for r0 in range(0, 256, 8):
            r1 = min(256, r0 + 8)
            g = np.empty((r1 - r0, 256, 256, 3), np.float32)
            g[..., 0] = v[r0:r1, 0][:, None, None]
            g[..., 1] = v[:, 1][None, :, None]
            g[..., 2] = v[:, 2][None, None, :]
            finish(g, r0, r1)
    return bgra.view(np.uint32).reshape(-1)


def apply_packed(img: np.ndarray, packed: np.ndarray, rows: int = 256,
                 inplace: bool = False) -> np.ndarray:
    """Grade a BGR uint8 image with a build_packed() table. Works in strips so
    the temporaries stay small however large the photograph is. inplace=True
    writes the result over img instead of into a second full-size array (each
    strip's lookup indices are made before the strip is overwritten)."""
    h, w = img.shape[:2]
    out = img if inplace else np.empty_like(img)
    for y in range(0, h, rows):
        strip = img[y:y + rows]
        sh = strip.shape[0]
        a = cv2.cvtColor(strip, cv2.COLOR_BGR2BGRA)                 # bytes B,G,R,255
        idx = a.view(np.uint32).reshape(sh, w) & np.uint32(0xFFFFFF)  # r<<16|g<<8|b
        cv2.cvtColor(packed.take(idx).view(np.uint8).reshape(sh, w, 4),
                     cv2.COLOR_BGRA2BGR, dst=out[y:y + rows])
    return out


# ---------------------------------------------------------------------------
#  Finishing (optional): scale to a target height, then sharpen
# ---------------------------------------------------------------------------
def finish_image(img: np.ndarray, target_h: int, sharpen: str,
                 interpolation: str = "lanczos") -> np.ndarray:
    h, w = img.shape[:2]
    if h != target_h:
        new_w = max(1, int(round(w * target_h / float(h))))
        if h > target_h:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC if interpolation == "cubic" else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (new_w, target_h), interpolation=interp)
    amount = SHARPEN_AMOUNT.get(str(sharpen).lower(), 0.0)
    if amount > 0:
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return img


# ---------------------------------------------------------------------------
#  JPEG plumbing: colour profile
# ---------------------------------------------------------------------------
def extract_icc(data: bytes) -> Optional[bytes]:
    """The embedded ICC profile of a JPEG, reassembled from its APP2 chunks."""
    if data[:2] != b"\xff\xd8":
        return None
    pos, chunks = 2, {}
    n = len(data)
    while pos + 4 <= n and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            pos += 2
            continue
        if marker in (0xDA, 0xD9):
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        seg = data[pos + 4:pos + 2 + length]
        if marker == 0xE2 and seg[:12] == b"ICC_PROFILE\x00" and len(seg) >= 14:
            chunks[seg[12]] = seg[14:]
        pos += 2 + length
    if not chunks:
        return None
    return b"".join(chunks[k] for k in sorted(chunks))


def icc_segments(icc: bytes) -> bytes:
    """The APP2 segment(s) that carry an ICC profile."""
    step = 65535 - 2 - 14
    parts = [icc[i:i + step] for i in range(0, len(icc), step)] or [b""]
    return b"".join(
        b"\xff\xe2" + struct.pack(">H", len(c) + 16) + b"ICC_PROFILE\x00"
        + bytes([i + 1, len(parts)]) + c
        for i, c in enumerate(parts))


def exif_segment(exif: bytes) -> bytes:
    """The APP1 segment for a piexif.dump() block."""
    if len(exif) + 2 > 0xFFFF:
        raise ValueError("the EXIF block is larger than one JPEG segment")
    return b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif


def insert_segments(data: bytes, *segments: bytes) -> bytes:
    """Put ready-made segments into a JPEG right after its leading APP0/APP1
    segments, in a single pass over the bytes (one copy, not one per segment)."""
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] in (0xE0, 0xE1):
        pos += 2 + struct.unpack(">H", data[pos + 2:pos + 4])[0]
    mv = memoryview(data)
    return b"".join([mv[:pos], *segments, mv[pos:]])


def insert_icc(data: bytes, icc: bytes) -> bytes:
    """Put an ICC profile into a JPEG, after any leading APP0/APP1 segments."""
    return insert_segments(data, icc_segments(icc))


def icc_is_srgb(icc: bytes) -> bool:
    """Heuristic: sRGB profiles name themselves (as ASCII or UTF-16)."""
    return b"sRGB" in icc or b"\x00s\x00R\x00G\x00B" in icc


def icc_name(icc: bytes) -> str:
    """Best-effort human name of a profile, for the log."""
    try:
        count = struct.unpack(">I", icc[128:132])[0]
        for i in range(min(count, 100)):
            sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i:144 + 12 * i])
            if sig != b"desc":
                continue
            tag = icc[off:off + size]
            if tag[:4] == b"desc":
                n = struct.unpack(">I", tag[8:12])[0]
                return tag[12:12 + max(0, n - 1)].decode("ascii", "replace")
            if tag[:4] == b"mluc":
                roff, rlen = struct.unpack(">II", tag[20:28])
                return tag[roff:roff + rlen].decode("utf-16-be", "replace")
    except Exception:
        pass
    return "unnamed profile"


# ---------------------------------------------------------------------------
#  JPEG plumbing: EXIF
# ---------------------------------------------------------------------------
def stamp_text(lut_name: str, strength: float, finish: bool) -> str:
    return f"{STAMP} {lut_name} strength={strength:.2f}" + (" finished" if finish else "")


def is_stamped(exif_dict: Optional[dict]) -> bool:
    if not exif_dict or piexif is None:
        return False
    try:
        sw = exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")
        if isinstance(sw, str):
            sw = sw.encode("ascii", "ignore")
        return bytes(sw).startswith(STAMP.encode("ascii"))
    except Exception:
        return False


def build_exif(exif_dict: Optional[dict], stamp: str, size: Tuple[int, int]) -> Optional[bytes]:
    """The EXIF block for a graded file: everything the source had, minus

    * Orientation -- OpenCV has already rotated the pixels, so keeping the tag
      would rotate them a second time in every viewer;
    * the embedded thumbnail -- it is a picture of the ungraded original;

    plus the stamp and the true pixel dimensions."""
    if piexif is None:
        return None
    d = dict(exif_dict) if exif_dict else {}
    for k in ("0th", "Exif", "GPS", "Interop"):
        d[k] = dict(d.get(k) or {})
    d.pop("1st", None)
    d.pop("thumbnail", None)
    d["0th"].pop(piexif.ImageIFD.Orientation, None)
    d["0th"][piexif.ImageIFD.Software] = stamp.encode("ascii", "replace")
    w, h = size
    d["Exif"][piexif.ExifIFD.PixelXDimension] = int(w)
    d["Exif"][piexif.ExifIFD.PixelYDimension] = int(h)
    return piexif.dump(d)


# ---------------------------------------------------------------------------
#  One file
# ---------------------------------------------------------------------------
class Lap:
    """Wall time and CPU time of the current thread, per named stage. The gap
    between the two is time spent not running: waiting for disk, for the GIL
    or for a free CPU thread."""
    def __init__(self) -> None:
        self.w = time.perf_counter()
        self.c = time.thread_time()
        self.d: Dict[str, List[float]] = {}

    def mark(self, name: str) -> None:
        w, c = time.perf_counter(), time.thread_time()
        a = self.d.setdefault(name, [0.0, 0.0])
        a[0] += w - self.w
        a[1] += c - self.c
        self.w, self.c = w, c


STAGES = ["read", "exif", "decode", "finish", "grade", "encode", "assemble", "write"]


@dataclass
class Options:
    input_dir: Path
    output_dir: Path
    lut: str
    lut_folder: Path
    strength: float = 1.0
    finish: bool = False
    target_height: int = 4000
    sharpen: str = "strong"
    interpolation: str = "lanczos"
    quality: int = 95
    overwrite: bool = False
    regrade: bool = False
    watch: bool = False
    workers: int = DEFAULT_WORKERS      # 0 = automatic
    warn_below: int = DEFAULT_WARN_BELOW
    poll_sec: float = 3.0
    stability_checks: int = 2


def _load_exif(data: bytes) -> Optional[dict]:
    if piexif is None:
        return None
    try:
        return piexif.load(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Photo sizes (header only -- no decoding)
# ---------------------------------------------------------------------------
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(path) -> Optional[Tuple[int, int]]:
    """(width, height) read from the JPEG header, or None if it cannot be read.
    Only walks the marker segments up to the frame header, so it costs a few
    KB of reading per file. The pixel data is never decoded."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            for _ in range(4096):
                b = f.read(1)
                if not b:
                    return None
                if b != b"\xff":
                    continue
                m = f.read(1)
                while m == b"\xff":
                    m = f.read(1)
                if not m:
                    return None
                mk = m[0]
                if mk in (0x00, 0x01, 0xD8) or 0xD0 <= mk <= 0xD7:
                    continue
                if mk in (0xD9, 0xDA):          # end of image / start of scan: no frame header
                    return None
                seg = f.read(2)
                if len(seg) < 2:
                    return None
                ln = int.from_bytes(seg, "big")
                if ln < 2:
                    return None
                if mk in _SOF:
                    body = f.read(5)
                    if len(body) < 5:
                        return None
                    h = int.from_bytes(body[1:3], "big")
                    w = int.from_bytes(body[3:5], "big")
                    return (w, h) if w and h else None
                f.seek(ln - 2, 1)
    except OSError:
        return None
    return None


def is_small(w: int, h: int, warn_below: int) -> bool:
    """True when the longer side is under the threshold (0 turns the check off)."""
    return bool(warn_below) and max(w, h) < warn_below


def scan_sizes(input_dir: Path, stop: Optional[threading.Event] = None,
               progress=None) -> List[Tuple[str, int, int]]:
    """(relative path, width, height) for every JPEG under input_dir; (rel, 0, 0)
    where the header cannot be read. progress(done_so_far) is called now and then."""
    out: List[Tuple[str, int, int]] = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if stop is not None and stop.is_set():
                return out
            if Path(f).suffix.lower() not in JPEG_EXTS:
                continue
            p = Path(root) / f
            wh = jpeg_size(p) or (0, 0)
            out.append((str(p.relative_to(input_dir)), wh[0], wh[1]))
            if progress and len(out) % 50 == 0:
                progress(len(out))
    if progress:
        progress(len(out))
    return out


def summarize_sizes(entries: List[Tuple[str, int, int]], warn_below: int,
                    max_listed: int = 100) -> Dict[str, Any]:
    small = sorted((e for e in entries if e[1] and is_small(e[1], e[2], warn_below)),
                   key=lambda e: max(e[1], e[2]))
    return {
        "total": len(entries),
        "unreadable": sum(1 for e in entries if not e[1]),
        "small": len(small),
        "smallest": max(small[0][1], small[0][2]) if small else 0,
        "warn_below": warn_below,
        "files": [{"file": e[0], "w": e[1], "h": e[2]} for e in small[:max_listed]],
    }


def process_file(src: Path, dst: Path, opts: Options, packed: np.ndarray,
                 on_profile=None, on_size=None, on_timing=None) -> Tuple[str, str]:
    """Grade one file. Returns (status, detail); status is one of
    ok, exists, graded, unreadable, error."""
    if dst.exists() and not opts.overwrite:
        return "exists", ""
    lap = Lap()
    data = src.read_bytes()
    lap.mark("read")
    exif_dict = _load_exif(data)
    if not opts.regrade and is_stamped(exif_dict):
        return "graded", "already graded by Grade Room"
    lap.mark("exif")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable", "could not be decoded"
    lap.mark("decode")
    if on_size:
        on_size(img.shape[1], img.shape[0])              # the source's size, before any scaling

    if opts.finish:
        img = finish_image(img, opts.target_height, opts.sharpen, opts.interpolation)
        lap.mark("finish")
    img = apply_packed(img, packed, inplace=True)         # img is ours: no second full-size array
    lap.mark("grade")

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(opts.quality)])
    if not ok:
        return "error", "JPEG encode failed"
    payload = buf.tobytes()
    lap.mark("encode")

    # Everything is put together in memory and written once. (Inserting the EXIF and
    # the profile into the file afterwards meant reading and rewriting it twice more.)
    detail = ""
    segs: List[bytes] = []
    if piexif is not None:
        try:
            exif = build_exif(exif_dict, stamp_text(opts.lut, opts.strength, opts.finish),
                              (img.shape[1], img.shape[0]))
            if exif:
                segs.append(exif_segment(exif))
        except Exception as exc:
            detail = f"EXIF not copied ({exc})"
    icc = extract_icc(data)
    if icc:
        segs.append(icc_segments(icc))
        if not icc_is_srgb(icc):
            name = icc_name(icc)
            detail = (detail + "; " if detail else "") + f"profile {name} (not sRGB)"
            if on_profile:
                on_profile(name)
    if segs:
        payload = insert_segments(payload, *segs)
    lap.mark("assemble")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    lap.mark("write")
    if on_timing:
        on_timing(lap.d)
    return "ok", detail


# ---------------------------------------------------------------------------
#  Folders
# ---------------------------------------------------------------------------
def _inside(child: Path, parent: Path) -> bool:
    try:
        c, p = child.resolve(), parent.resolve()
    except OSError:
        return False
    return c == p or p in c.parents


def check_folders(input_dir: Path, output_dir: Path) -> Optional[str]:
    """None if the pair is usable, otherwise a sentence saying why not."""
    if not input_dir.is_dir():
        return "the input folder does not exist"
    if _inside(output_dir, input_dir):
        return ("the output folder is inside the input folder; graded files would be "
                "picked up and graded again")
    if _inside(input_dir, output_dir):
        return "the input folder is inside the output folder"
    return None


def list_sample_files(input_dir: Path, n: int = 4, cap: int = 3000) -> List[Path]:
    """A few files spread across the input, for previews. Stops listing at cap
    so a huge tree does not stall the page."""
    found: List[Path] = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in JPEG_EXTS:
                found.append(Path(root) / f)
                if len(found) >= cap:
                    break
        if len(found) >= cap:
            break
    if len(found) <= n:
        return found
    idx = np.linspace(0, len(found) - 1, n).round().astype(int)
    return [found[i] for i in sorted(set(idx.tolist()))]


# ---------------------------------------------------------------------------
#  A run
# ---------------------------------------------------------------------------
class Job:
    """One grading run: a batch over what is there, optionally followed by
    watching for more. Thread-safe counters; the UI only reads snapshot()."""

    def __init__(self, opts: Options):
        self.o = opts
        self.status = "starting"     # starting building running watching done stopped error
        self.error = ""
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.found = self.done = self.exists = self.graded = self.failed = 0
        self.started = time.time()
        self.work_started = 0.0      # when grading began (after the LUT table was built)
        self.finished = 0.0
        self.log: List[str] = []
        self.current = ""
        self.non_srgb: Set[str] = set()
        self.seen: Set[str] = set()
        self._sizes: Dict[str, Tuple[int, int]] = {}
        self._report_lock = threading.Lock()
        self.report_path = opts.output_dir / REPORT_NAME
        self._report_ready = False
        self.small = 0                           # photos under warn_below on the long side
        self.small_min = 0                       # the smallest long side seen among them
        self.small_files: List[Dict[str, Any]] = []
        # Worker count can be changed while running: threads wait at this gate.
        self._gate = threading.Condition()
        self._active = 0
        self.auto_workers = int(opts.workers) <= 0          # 0 = decide from the CPU and the photos
        self.workers = auto_workers() if self.auto_workers else max(1, min(MAX_WORKERS, int(opts.workers)))
        self._workers_checked = False
        self._lap_wall: Dict[str, float] = {}
        self._lap_cpu: Dict[str, float] = {}
        self._lapped = 0
        self._report_fh = None
        self._report_csv = None
        self._stamps: Deque[float] = deque()      # completion times, for the recent rate

    def _on_timing(self, d: Dict[str, List[float]]) -> None:
        with self.lock:
            self._lapped += 1
            for k, (w, c) in d.items():
                self._lap_wall[k] = self._lap_wall.get(k, 0.0) + w
                self._lap_cpu[k] = self._lap_cpu.get(k, 0.0) + c

    def _timing_summary(self) -> Optional[Dict[str, Any]]:
        """Average per photo (ms): wall time and CPU time for each stage. Called
        with self.lock held (from snapshot)."""
        n = self._lapped
        if n < 3:
            return None
        rows = [{"stage": k, "wall": self._lap_wall[k] / n * 1000, "cpu": self._lap_cpu.get(k, 0.0) / n * 1000}
                for k in STAGES if k in self._lap_wall]
        wall = sum(r["wall"] for r in rows)
        cpu = sum(r["cpu"] for r in rows)
        return {"photos": n, "stages": rows, "wall": wall, "cpu": cpu,
                "waiting": max(0.0, 1.0 - cpu / wall) if wall > 0 else 0.0}

    @staticmethod
    def _cpu_now() -> Optional[float]:
        """Whole-PC CPU use since the previous call (the page asks about once a second)."""
        if psutil is None:
            return None
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def _size_workers(self, fresh: List[Path]) -> None:
        """Once, from the first batch: settle the automatic worker count, and say so
        if the photos are big enough that the chosen count may not fit in memory."""
        self._workers_checked = True
        step = max(1, len(fresh) // 200)
        mp = 0.0
        for p in fresh[::step]:
            wh = jpeg_size(p)
            if wh:
                mp = max(mp, wh[0] * wh[1] / 1e6)
        if self.auto_workers:
            n = auto_workers(mp)
            if n != self.workers:
                self.workers = n
            self.note(f"workers: {self.workers} (automatic: {cpu_threads()} CPU threads"
                      + (f", limited by free memory for {mp:.0f} MP photos" if self.workers < min(MAX_WORKERS, cpu_threads()) else "")
                      + ")")
        avail = _avail_mem()
        if mp and avail and self.workers * memory_per_worker(mp) > avail:
            self.note(f"WARNING: {self.workers} workers on {mp:.0f} MP photos may need about "
                      f"{self.workers * memory_per_worker(mp) / 1e9:.1f} GB; {avail / 1e9:.1f} GB is free. "
                      f"Lower the worker count if the PC starts to swap.")

    def set_workers(self, n: int) -> int:
        n = max(1, min(MAX_WORKERS, int(n)))
        self.auto_workers = False                # an explicit choice replaces the automatic one
        with self._gate:
            old, self.workers = self.workers, n
            self._gate.notify_all()
        if n != old:
            if n > 1:
                try:
                    cv2.setNumThreads(1)
                except Exception:
                    pass
            self.note(f"workers: {old} -> {n}")
        return n

    # -- reporting -------------------------------------------------------
    def note(self, msg: str) -> None:
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
            del self.log[:-80]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            handled = self.done + self.exists + self.graded + self.failed
            end = self.finished or time.time()
            el = max(0.001, end - self.started)
            work_el = max(0.001, end - self.work_started) if self.work_started else 0.0
            rate = self.done / work_el if (self.done and work_el) else 0.0
            now = time.time()
            while self._stamps and now - self._stamps[0] > RECENT_SEC:
                self._stamps.popleft()
            if self.status in ("running", "watching") and work_el > RECENT_SEC / 2 and self._stamps:
                rate = len(self._stamps) / min(RECENT_SEC, work_el)   # what it is doing now
            left = max(0, self.found - handled)
            eta = (left / rate) if (rate and self.status in ("running", "starting")) else None
            return {
                "status": self.status, "error": self.error,
                "found": self.found, "done": self.done, "exists": self.exists,
                "graded": self.graded, "failed": self.failed, "handled": handled,
                "elapsed": el, "rate": rate, "workers": self.workers,
                "auto_workers": self.auto_workers, "cpu_threads": cpu_threads(),
                "cpu": self._cpu_now(), "timing": self._timing_summary(),
                "warn_below": self.o.warn_below, "small": self.small,
                "small_min": self.small_min, "small_files": list(self.small_files[:100]), "eta": eta, "current": self.current,
                "log": list(self.log[-14:]), "report": str(self.report_path),
                "non_srgb": sorted(self.non_srgb),
                "lut": self.o.lut, "strength": self.o.strength, "finish": self.o.finish,
                "target_height": self.o.target_height,
                "input": str(self.o.input_dir), "output": str(self.o.output_dir),
            }

    def _prepare_report(self) -> None:
        """Make sure an existing report from an older version has the current
        columns before rows are appended to it."""
        if self._report_ready:
            return
        self._report_ready = True
        if not self.report_path.exists():
            return
        with open(self.report_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows or rows[0] == REPORT_HEADER:
            return
        rows[0] = REPORT_HEADER
        rows = [r + [""] * (len(REPORT_HEADER) - len(r)) if i else r for i, r in enumerate(rows)]
        tmp = self.report_path.with_name(self.report_path.name + ".part")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        os.replace(tmp, self.report_path)

    def _report(self, rel: str, status: str, detail: str, ms: float,
                size: Optional[Tuple[int, int]] = None, note: str = "") -> None:
        try:
            with self._report_lock:
                if self._report_fh is None:
                    self._prepare_report()
                    new = not self.report_path.exists()
                    self._report_fh = open(self.report_path, "a", newline="", encoding="utf-8")
                    self._report_csv = csv.writer(self._report_fh)
                    if new:
                        self._report_csv.writerow(REPORT_HEADER)
                self._report_csv.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), rel, status, detail,
                                           f"{ms:.0f}", self.o.lut, f"{self.o.strength:.2f}",
                                           "yes" if self.o.finish else "no",
                                           size[0] if size else "", size[1] if size else "", note])
                self._report_fh.flush()
        except (OSError, ValueError):
            pass

    def _close_report(self) -> None:
        with self._report_lock:
            if self._report_fh is not None:
                try:
                    self._report_fh.close()
                except OSError:
                    pass
                self._report_fh = self._report_csv = None

    # -- discovery -------------------------------------------------------
    def _stable(self, p: Path) -> bool:
        try:
            size = p.stat().st_size
        except OSError:
            return False
        prev, streak = self._sizes.get(str(p), (-1, 0))
        streak = streak + 1 if size == prev else 1
        self._sizes[str(p)] = (size, streak)
        return streak >= self.o.stability_checks

    def _scan(self) -> Tuple[List[Path], int]:
        fresh: List[Path] = []
        unstable = 0
        for root, _dirs, files in os.walk(self.o.input_dir):
            for f in files:
                if Path(f).suffix.lower() not in JPEG_EXTS:
                    continue
                p = Path(root) / f
                if str(p) in self.seen:
                    continue
                if self._stable(p):
                    self.seen.add(str(p))
                    fresh.append(p)
                else:
                    unstable += 1
        fresh.sort()
        return fresh, unstable

    # -- work ------------------------------------------------------------
    def _one(self, packed: np.ndarray, p: Path) -> None:
        with self._gate:
            while self._active >= self.workers and not self.stop.is_set():
                self._gate.wait(0.25)
            if self.stop.is_set():
                return
            self._active += 1
        try:
            self._one_gated(packed, p)
        finally:
            with self._gate:
                self._active -= 1
                self._gate.notify()

    def _one_gated(self, packed: np.ndarray, p: Path) -> None:
        rel = p.relative_to(self.o.input_dir)
        dst = self.o.output_dir / rel
        t0 = time.time()
        with self.lock:
            self.current = str(rel)
        seen: Dict[str, int] = {}
        try:
            status, detail = process_file(p, dst, self.o, packed, self._on_profile,
                                          on_size=lambda w, h: seen.update(w=w, h=h),
                                          on_timing=self._on_timing)
        except Exception as exc:                       # keep the run alive
            status, detail = "error", f"{type(exc).__name__}: {exc}"
        ms = (time.time() - t0) * 1000
        size = (seen["w"], seen["h"]) if seen else None
        note = ""
        if size and is_small(size[0], size[1], self.o.warn_below):
            long_side = max(size)
            note = f"smaller than {self.o.warn_below}px on the long side"
            with self.lock:
                self.small += 1
                n_small = self.small
                self.small_min = long_side if not self.small_min else min(self.small_min, long_side)
                if len(self.small_files) < MAX_SMALL_LISTED:
                    self.small_files.append({"file": str(rel), "w": size[0], "h": size[1]})
            if n_small <= 3:
                self.note(f"small photo: {rel} is {size[0]}x{size[1]} "
                          f"(under {self.o.warn_below}px on the long side)")
            elif n_small == 4:
                self.note("more small photos are being counted; the report lists each one")
        with self.lock:
            if status == "ok":
                self.done += 1
                self._stamps.append(time.time())
            elif status == "exists":
                self.exists += 1
            elif status == "graded":
                self.graded += 1
            else:
                self.failed += 1
        if status in ("unreadable", "error"):
            self.note(f"FAILED {rel}: {detail}")
        if status != "exists":
            self._report(str(rel), status, detail, ms, size, note)

    def _on_profile(self, name: str) -> None:
        with self.lock:
            first = name not in self.non_srgb
            self.non_srgb.add(name)
        if first:
            self.note(f"colour profile '{name}' is not sRGB -- graded as-is, profile kept")

    def run(self) -> None:
        o = self.o
        try:
            problem = check_folders(o.input_dir, o.output_dir)
            if problem:
                raise ValueError(problem)
            o.output_dir.mkdir(parents=True, exist_ok=True)
            self.status = "building"
            self.note(f"building the lookup table for {o.lut} ...")
            parsed = lutlib.parse_cube(o.lut_folder / o.lut)
            packed = build_packed(parsed, o.strength)
            self.note(f"grading with {o.lut} at {o.strength * 100:.0f}%"
                      + (f", finished to {o.target_height}px" if o.finish else ", native size"))
            if self.workers > 1:
                try:
                    cv2.setNumThreads(1)      # parallelism comes from the workers
                except Exception:
                    pass
            self.status = "running"
            self.work_started = time.time()
            pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)   # idle threads wait at the gate
            waited = 0.0
            # Record every file's size now, so files that are already complete
            # are recognised as stable after half a second instead of a full poll.
            self._scan()
            self.stop.wait(0.5)
            try:
                while not self.stop.is_set():
                    fresh, unstable = self._scan()
                    if fresh:
                        with self.lock:
                            self.found += len(fresh)
                        self.note(f"{len(fresh)} file(s) to grade")
                        self.status = "running"
                        if not self._workers_checked:
                            self._size_workers(fresh)
                        list(pool.map(lambda q: self._one(packed, q), fresh))
                        waited = 0.0
                        continue
                    if o.watch:
                        self.status = "watching"
                        self.stop.wait(o.poll_sec)
                        continue
                    if unstable and waited < 30:      # still being copied; give it a moment
                        self.stop.wait(o.poll_sec)
                        waited += o.poll_sec
                        continue
                    if unstable:
                        self.note(f"{unstable} file(s) were still changing and were left "
                                  f"out -- run again, or use Keep watching")
                    break
            finally:
                pool.shutdown(wait=True)
            self.status = "stopped" if self.stop.is_set() else "done"
            with self.lock:
                tm = self._timing_summary()
            if tm:
                self.note("per photo (average): " + ", ".join(
                    f"{r['stage']} {r['wall']:.0f} ms" for r in tm["stages"]) +
                    f" = {tm['wall']:.0f} ms, of which {tm['cpu']:.0f} ms computing"
                    f" ({tm['waiting'] * 100:.0f}% waiting)")
            if self.small:
                self.note(f"WARNING: {self.small} photo(s) are under {o.warn_below}px on the long side "
                          f"(smallest {self.small_min}px)")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            self.note(f"stopped: {self.error}")
        finally:
            self._close_report()
            self.finished = time.time()
            self.current = ""


# ---------------------------------------------------------------------------
#  Speed test: what limits grading on this PC?
# ---------------------------------------------------------------------------
def bench_main(argv: List[str]) -> None:
    import platform
    import shutil
    import tempfile
    ap = argparse.ArgumentParser(prog="grade_suite.py bench",
                                 description="Measure what limits grading speed on this PC")
    ap.add_argument("input", help="a folder of JPEGs to test with")
    ap.add_argument("--lut", default="", help="a .cube in the LUT folder (default: the first one)")
    ap.add_argument("--lut-folder", default="luts")
    ap.add_argument("--photos", type=int, default=24, help="how many photos to test with (default 24)")
    ap.add_argument("--finish", action="store_true", help="test with finishing (scale + sharpen) on")
    ap.add_argument("--height", type=int, default=4000)
    ap.add_argument("--out", default="", help="folder for the temporary output; use the folder you really "
                                              "write to, so its drive is what gets tested")
    ap.add_argument("--workers", default="", help="worker counts to try, e.g. 1,8,32")
    ap.add_argument("--report", default="bench_grade_result.txt")
    a = ap.parse_args(argv)
    lines: List[str] = []

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    inp = Path(a.input)
    files = list_sample_files(inp, n=max(2, a.photos))
    if not files:
        print("No JPEGs found in", inp)
        sys.exit(1)
    lutdir = Path(a.lut_folder)
    lut_name = a.lut or (lutlib.list_luts(lutdir)[0].name if lutlib.list_luts(lutdir) else "")
    if not lut_name:
        print(f"No .cube files in {lutdir}")
        sys.exit(1)
    threads = cpu_threads()
    base_out = Path(a.out) if a.out else inp.parent
    tmp = Path(tempfile.mkdtemp(prefix="_gradeoom_bench_", dir=str(base_out)))
    try:
        say("Grade Room speed test")
        say(f"  python {platform.python_version()}, opencv {cv2.__version__}, piexif {'yes' if piexif else 'NO'}, "
            f"psutil {'yes' if psutil else 'NO'}")
        avail = _avail_mem()
        say(f"  {threads} CPU threads" + (f", {avail / 1e9:.1f} GB memory free" if avail else ""))
        sizes = [f.stat().st_size for f in files]
        mp = [(jpeg_size(f) or (0, 0)) for f in files]
        avg_mb = sum(sizes) / len(sizes) / 1e6
        avg_mp = sum(w * h for w, h in mp) / len(mp) / 1e6
        say(f"  {len(files)} photos from {inp}: {avg_mb:.1f} MB and {avg_mp:.1f} MP on average"
            f"{' (finishing to ' + str(a.height) + ' px)' if a.finish else ' (native size)'}")
        say(f"  temporary output on the drive of {tmp}")
        say()

        # -- disk: read (first look at the files, so this may still be partly cached) --
        t = time.perf_counter()
        for f in files:
            f.read_bytes()
        read_s = time.perf_counter() - t
        read_mbs = sum(sizes) / 1e6 / read_s
        say(f"Reading the photos:   {read_mbs:7.0f} MB/s  (one at a time; if this is high the files were probably cached already)")

        # -- one photo at a time: what does the computing cost? --
        t = time.perf_counter()
        parsed = lutlib.parse_cube(lutdir / lut_name)
        packed = build_packed(parsed, 1.0)
        say(f"Building the look:    {time.perf_counter() - t:7.1f} s   (once per run)")
        cv2.setNumThreads(1)
        o = Options(inp, tmp, lut_name, lutdir, overwrite=True, finish=a.finish, target_height=a.height)
        laps: List[Dict[str, List[float]]] = []
        out1 = tmp / "single"
        for i, f in enumerate(files):
            process_file(f, out1 / f"{i}.jpg", o, packed, on_timing=laps.append)
        n = len(laps)
        say()
        say("One worker, one photo at a time (average per photo):")
        say(f"  {'stage':10s} {'wall ms':>8s} {'cpu ms':>8s}")
        tot_w = tot_c = 0.0
        for k in STAGES:
            w = sum(l[k][0] for l in laps if k in l) / n * 1000
            c = sum(l[k][1] for l in laps if k in l) / n * 1000
            if any(k in l for l in laps):
                say(f"  {k:10s} {w:8.0f} {c:8.0f}")
                tot_w += w
                tot_c += c
        say(f"  {'total':10s} {tot_w:8.0f} {tot_c:8.0f}")
        out_sizes = [p_.stat().st_size for p_ in out1.glob("*.jpg")]
        out_mb = (sum(out_sizes) / len(out_sizes) / 1e6) if out_sizes else avg_mb

        # -- disk: write --
        blob = (out1 / "0.jpg").read_bytes() if out_sizes else b"\0" * int(avg_mb * 1e6)
        wdir = tmp / "wtest"
        wdir.mkdir()
        t = time.perf_counter()
        for i in range(len(files)):
            (wdir / f"{i}.jpg").write_bytes(blob)
        write_s = time.perf_counter() - t
        write_mbs = len(files) * len(blob) / 1e6 / write_s
        from concurrent.futures import ThreadPoolExecutor as _TPE
        t = time.perf_counter()
        with _TPE(8) as ex:
            list(ex.map(lambda i: (wdir / f"p{i}.jpg").write_bytes(blob), range(len(files))))
        write8_mbs = len(files) * len(blob) / 1e6 / (time.perf_counter() - t)
        say()
        say(f"Writing {out_mb:.1f} MB files: {write_mbs:6.0f} MB/s one at a time, {write8_mbs:.0f} MB/s with 8 at once")

        # -- the real thing at different worker counts --
        if a.workers:
            counts = sorted({max(1, min(MAX_WORKERS, int(x))) for x in a.workers.split(",") if x.strip()})
        else:
            counts = sorted({1, 2, 4, 8, 16, min(MAX_WORKERS, threads), min(MAX_WORKERS, threads * 2)})
        say()
        say("Grading with several workers at once:")
        say(f"  {'workers':>7s} {'photos/s':>9s} {'cpu %':>6s} {'speed-up':>9s}   where a photo's time goes (wall ms)")
        results: List[Tuple[int, float, float, Dict[str, float]]] = []
        first_rate = 0.0
        for w in counts:
            tasks = max(len(files), 2 * w)
            jobs = [(files[i % len(files)], tmp / f"w{w}" / f"{i}.jpg") for i in range(tasks)]
            wl: List[Dict[str, List[float]]] = []
            samples: List[float] = []
            stop = threading.Event()

            def sampler() -> None:
                if psutil is None:
                    return
                psutil.cpu_percent(None)
                while not stop.wait(0.4):
                    samples.append(psutil.cpu_percent(None))
            th = threading.Thread(target=sampler, daemon=True)
            th.start()
            t = time.perf_counter()
            with _TPE(w) as ex:
                list(ex.map(lambda j: process_file(j[0], j[1], o, packed, on_timing=wl.append), jobs))
            dt = time.perf_counter() - t
            stop.set()
            th.join()
            rate = tasks / dt
            cpu = sum(samples) / len(samples) if samples else float("nan")
            first_rate = first_rate or rate
            per = {k: sum(l[k][0] for l in wl if k in l) / len(wl) * 1000 for k in STAGES if any(k in l for l in wl)}
            results.append((w, rate, cpu, per))
            say(f"  {w:7d} {rate:9.1f} {cpu:6.0f} {rate / first_rate:8.1f}x   "
                + " ".join(f"{k} {v:.0f}" for k, v in per.items()))
            shutil.rmtree(tmp / f"w{w}", ignore_errors=True)

        best = max(results, key=lambda r: r[1])
        cpu_cap = threads / (tot_c / 1000) if tot_c else float("inf")
        read_cap = read_mbs / avg_mb
        write_cap = write8_mbs / out_mb
        say()
        say("What each part could deliver on its own:")
        say(f"  the CPU ({tot_c:.0f} ms of computing per photo on {threads} threads): up to {cpu_cap:5.1f} photos/s")
        say(f"  reading the photos ({avg_mb:.0f} MB each):                     up to {read_cap:5.1f} photos/s")
        say(f"  writing the results ({out_mb:.0f} MB each):                    up to {write_cap:5.1f} photos/s")
        say(f"  measured, best case ({best[0]} workers):                        {best[1]:5.1f} photos/s")
        say()
        limits = {"the CPU": cpu_cap, "reading the photos": read_cap, "writing the results": write_cap}
        low_name, low = min(limits.items(), key=lambda kv: kv[1])
        if best[1] >= 0.75 * low:
            if low_name == "the CPU":
                say(f"Verdict: CPU-bound. {best[1]:.1f} photos/s is close to what {threads} threads can do with this much")
                say("work per photo. More workers will not help; less work per photo will (finishing off, bicubic")
                say("instead of Lanczos, a lower JPEG quality).")
            else:
                say(f"Verdict: limited by {low_name}: about {low:.1f} photos/s is all that drive delivers. More workers")
                say("or a faster CPU will not help; a faster drive (or a different drive for the output) will.")
        else:
            say(f"Verdict: slower than every single limit above (best {best[1]:.1f} vs {low:.1f} photos/s for {low_name}).")
            say("Something else is holding it back: antivirus scanning each file, memory bandwidth, or waiting")
            say("inside Python. The wall-vs-cpu gaps above show which stage waits. Try excluding the input and")
            say("output folders from Windows Defender and run this again.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\n(saved as {a.report})")
        except OSError:
            pass


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        return bench_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description="Apply a .cube LUT to a folder of JPEGs (no cropping)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--lut", required=True, help="file name inside the LUT folder")
    ap.add_argument("--lut-folder", default="luts")
    ap.add_argument("--strength", type=float, default=1.0, help="0..1")
    ap.add_argument("--finish", action="store_true", help="scale to --height and sharpen first")
    ap.add_argument("--height", type=int, default=4000)
    ap.add_argument("--sharpen", default="strong", choices=list(SHARPEN_AMOUNT))
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--warn-below", type=int, default=DEFAULT_WARN_BELOW,
                    help="warn about photos whose longer side is under this many px (0 = off)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel photos (default 0 = one per CPU thread, limited by free memory; max {MAX_WORKERS})")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--regrade", action="store_true")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    job = Job(Options(Path(a.input), Path(a.output), a.lut, Path(a.lut_folder),
                      strength=a.strength, finish=a.finish, target_height=a.height,
                      sharpen=a.sharpen, quality=a.quality, workers=a.workers,
                      overwrite=a.overwrite, regrade=a.regrade, watch=a.watch,
                      warn_below=a.warn_below))
    t = threading.Thread(target=job.run, daemon=True)
    t.start()
    try:
        while t.is_alive():
            time.sleep(1)
            s = job.snapshot()
            print(f"\r{s['status']:9s} graded {s['done']}  skipped {s['exists'] + s['graded']}  "
                  f"failed {s['failed']}  of {s['found']}", end="", flush=True)
    except KeyboardInterrupt:
        job.stop.set()
        t.join()
    s = job.snapshot()
    print(f"\n{s['status']}: graded {s['done']}, skipped {s['exists'] + s['graded']}, "
          f"failed {s['failed']} in {s['elapsed']:.0f}s -> {a.output}")
    if s["small"]:
        print(f"WARNING: {s['small']} photo(s) are smaller than {a.warn_below}px on the long side "
              f"(smallest {s['small_min']}px); see {s['report']}")
    if s["error"]:
        print("error:", s["error"])
        sys.exit(1)


if __name__ == "__main__":
    main()
