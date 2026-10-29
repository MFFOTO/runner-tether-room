# -*- coding: utf-8 -*-
"""
Merged crop + grade pipeline.

Watches a folder for photos as they land, crops them, and once enough Premium
crops have accumulated finishes and colour-grades them into a retouched output
folder. Two folders is all the operator has to choose:

    watch folder      where the to-be-cropped photos arrive
    retouched folder  where the finished, graded crops are written

Both can come from the config, from --watch / --out, or from a prompt at
startup, and neither has to exist beforehand.

The design point carried over from the live suite: grading runs on its own
thread, so choosing or changing the LUT never pauses cropping. Crops keep being
cut while the operator is looking at previews.

Stage 1 writes crops at native size (no resize, no sharpen). Stage 2 does
resize -> sharpen -> grade -> encode in one pass, so every delivered image is
encoded exactly once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2

import grade_suite as gs            # JPEG metadata helpers (EXIF / ICC segments)
import lut as lutlib
import srgb
from runner_suite_core import (
    DEFAULT_CONFIG,
    HighResRunnerSuite,
    deep_merge,
    ensure_dir,
    tqdm,
)

MERGED_CONFIG_EXTRA: Dict[str, Any] = {
    "merged": {
        # The two folders the operator picks. Empty means "ask at startup".
        "watch_folder": "",
        "retouched_folder": "",
        # Raw crops need somewhere to live between the two stages. Kept out of
        # the retouched folder so what the operator hands over contains only
        # finished images; "" puts it beside the retouched folder.
        "work_folder": "",
        "recursive": True,              # photos may arrive in per-card subfolders
        "poll_interval_sec": 4.0,
        "file_stability_checks": 2,     # equal-size polls before a file is read
        "premium_batch_size": 100,
        "premium_batch_timeout_sec": 120,
        "batch_folder_prefix": "",      # "" -> folders are 1, 2, 3; "RACE" -> RACE1, RACE2
        # Only Premium crops are delivered. False (default): crops that are not
        # Premium are never written at all. True: Good crops are also saved, to
        # B_Good in the work folder, but still not delivered.
        "keep_good_crops": False,
        # Once a Premium crop's finished copy is written to the destination, remove
        # the native-size crop from the work folder. Crops that failed to deliver stay.
        "delete_crops_after_delivery": True,
        "note_dir": "",                 # where delivery.txt goes; "" -> with the photos
        "stop_when_idle_sec": 0,        # >0: exit after this long with nothing new (for tests)
    },
    "completeness_guard": {
        # A runner whose box touches the photo's left or right edge is cut off
        # by the frame, so it cannot be Premium. The engine only checks the top
        # edge, and its close-up ("full-frame") path skips the check entirely.
        # With this on, a box touching the left, right or top edge is judged
        # like any other runner instead of taking the close-up path, and one
        # cut at a side is demoted. Uses completeness_guard.edge_margin_px.
        "edge_truncation_guard": True,
    },
    "lut": {
        "enabled": True,
        "lut_folder": "luts",          # folder of .cube files, relative to this repo
        "selected": None,               # None -> ask on the first batch
        "strength": 1.0,
    },
}

_IMG_EXTS = {".jpg", ".jpeg"}
_EPG_SUFFIX = ".epg-data"


def _is_candidate(name: str) -> bool:
    """Plain JPEGs, plus ProLoad2's '<name>.JPG.epg-data' staged originals --
    those are plain JPEG bytes under a protective double extension, and
    cv2.imread reads by content, not by extension."""
    lower = name.lower()
    if lower.endswith(_EPG_SUFFIX):
        lower = lower[: -len(_EPG_SUFFIX)]
    return any(lower.endswith(e) for e in _IMG_EXTS)


class MergedSuite(HighResRunnerSuite):
    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    def _load_config(self, path: Path) -> Dict[str, Any]:
        cfg = deep_merge(json.loads(json.dumps(DEFAULT_CONFIG)), MERGED_CONFIG_EXTRA)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = deep_merge(cfg, json.load(f))
            except Exception as exc:
                print(f"[WARN] Could not read {path}: {exc}; using defaults.")
        else:
            print(f"[INFO] {path} not found; using defaults.")
        return cfg

    # ------------------------------------------------------------------
    # stage 1 is crop-only: the resize and sharpen belong to stage 2, where
    # they run once per delivered image right before the grade. Keeping them
    # off the per-photo path means arrivals are cropped as fast as possible.
    # ------------------------------------------------------------------
    def resize_final(self, crop: Any, enhance_mode: str = "upscale") -> Any:
        if not getattr(self, "_finishing", False):
            # Keep the aspect trim -- it only removes padding, never rescales --
            # so stage 2 receives exactly the framing it would have had.
            cp = self.cfg["crop"]
            ratio = float(cp.get("aspect_ratio", 0.666))
            h, w = crop.shape[:2]
            current = w / max(1.0, h)
            if current > ratio:
                new_w = max(1, int(round(h * ratio)))
                x0 = max(0, (w - new_w) // 2)
                return crop[:, x0:x0 + new_w]
            if current < ratio:
                new_h = max(1, int(round(w / ratio)))
                excess = h - new_h
                top_frac = float(cp.get("vertical_trim_top_fraction", 0.25))
                y0 = int(max(0, min(round(excess * top_frac), max(0, excess))))
                return crop[y0:y0 + new_h, :]
            return crop
        return super().resize_final(crop, enhance_mode=enhance_mode)

    def sharpen_image(self, image: Any) -> Any:
        if not getattr(self, "_finishing", False):
            return image
        return super().sharpen_image(image)

    # ------------------------------------------------------------------
    # Which crops belong to which photo cannot be recovered by slicing
    # report_rows: cropping runs on a thread pool, every thread appends to that
    # one shared list, and a slice taken around process_image() picks up rows
    # other threads appended meanwhile. write_output already knows the path and
    # the class at the moment it writes, so each crop announces itself here
    # instead -- exact under any number of workers.
    premium_sink: Optional[Callable[[Path], None]] = None
    _faults: Dict[Path, str] = {}          # replaced per batch in deliver_batch
    last_batch_faults: Dict[Path, str] = {}
    # Set by the Pipeline: lets a blocking wait (the look picker) notice that
    # the run is shutting down instead of holding the grader thread forever.
    stop_event: Optional[threading.Event] = None

    def _cut_by_frame(self, box: Any, img_shape: Any, top: bool = False) -> bool:
        """True if the person box runs into the photo's left or right edge (or,
        with top=True, its top edge): the athlete continues beyond the frame."""
        cg = self.cfg.get("completeness_guard", {})
        if not bool(cg.get("edge_truncation_guard", True)):
            return False
        h_img, w_img = img_shape[:2]
        margin = float(cg.get("edge_margin_px", 6))
        x1, y1, x2, _ = (float(v) for v in box[:4])
        return x1 <= margin or x2 >= w_img - margin or (top and y1 <= margin)

    def _is_full_frame_runner(self, box: Any, img_shape: Any) -> bool:
        # The close-up path bypasses the safe zone, the keypoint minimum and the
        # completeness guard. A close-up cut off by the frame must not use it.
        if self._cut_by_frame(box, img_shape, top=True):
            return False
        return super()._is_full_frame_runner(box, img_shape)

    def _assess_completeness(self, kp: Any, person_box: Any, img_shape: Any) -> Dict[str, Any]:
        comp = super()._assess_completeness(kp, person_box, img_shape)
        if self._cut_by_frame(person_box, img_shape):
            comp["truncated_side"] = True
            if comp["ok_for_premium"]:
                comp["ok_for_premium"] = False
                comp["reason"] = "truncated_side"
        return comp

    def _should_save_class(self, quality_class: str, score: float) -> bool:
        """Premium always; Good only if the operator asked to keep it. Anything
        else is dropped before it is resized, sharpened, encoded or written, so
        leaving Good crops out also saves that work."""
        if quality_class == "premium":
            return True
        keep_good = bool(self.cfg["merged"].get("keep_good_crops", False))
        if quality_class == "good":
            return keep_good
        return keep_good and super()._should_save_class(quality_class, score)

    # The engine calls this after every JPEG it writes: stage-1 crops (src is the
    # original photo), stage-2 deliveries (src is the stage-1 crop) and enhanced
    # copies. Overriding it is what makes the two promises below hold for all of
    # them, whichever path wrote the file:
    #   * the original EXIF travels with the image, capture date included. It is
    #     rebuilt rather than copied blind: Orientation goes (the pixels are
    #     already rotated), as does the thumbnail of the uncropped original, and
    #     the pixel dimensions are corrected. If a maker note is too big to fit,
    #     only that is dropped. The base version swallowed every error silently.
    #   * the file is tagged sRGB IEC61966-2.1.
    def _copy_exif_without_orientation(self, src: Path, dst: Path) -> bool:
        """True if the EXIF (when wanted) and the profile are in the file."""
        try:
            src_data = Path(src).read_bytes()
            segs: List[bytes] = []
            exif_dict = gs._load_exif(src_data)
            if gs.piexif is not None and bool(self.cfg["image_quality"].get("preserve_exif", True)):
                size = gs.jpeg_size(dst)
                if size is None:
                    raise ValueError("cannot read the size of the written JPEG")
                exif = gs.build_exif(exif_dict, None, size)
                if exif:
                    segs.append(gs.exif_segment(exif))
                    if not gs.has_capture_date(exif):
                        self._warn_once("nodate", f"{Path(dst).name} has no capture date in its EXIF "
                                                  f"(the original had none either)")
            self._check_source_colour(src_data, exif_dict, Path(src).name)
            segs.append(gs.icc_segments(srgb.srgb_icc()))
            out = gs.insert_segments(Path(dst).read_bytes(), *segs)
            tmp = Path(dst).with_name(Path(dst).name + ".part")
            tmp.write_bytes(out)
            os.replace(tmp, dst)
            return True
        except Exception as exc:
            print(f"\n[WARN] {Path(dst).name}: EXIF / colour profile not written ({exc})")
            return False

    def _warn_once(self, key: str, msg: str) -> None:
        seen = self.__dict__.setdefault("_warned", set())
        if key not in seen:
            seen.add(key)
            print(f"\n[WARN] {msg}")

    def _check_source_colour(self, data: bytes, exif_dict: Optional[dict], name: str) -> None:
        """Crops are tagged sRGB, which is only true if the camera shot sRGB.
        Say so once if the file declares otherwise."""
        icc = gs.extract_icc(data)
        if icc and not gs.icc_is_srgb(icc):
            pname = gs.icc_name(icc)
            self._warn_once("icc:" + pname, f"{name} carries the colour profile '{pname}', not sRGB. "
                                            f"Crops are tagged sRGB IEC61966-2.1 without conversion; "
                                            f"set the camera to sRGB.")
        elif gs.piexif is not None and exif_dict:
            if (exif_dict.get("Interop") or {}).get(gs.piexif.InteropIFD.InteroperabilityIndex) == b"R03":
                self._warn_once("r03", f"{name} is marked Adobe RGB by the camera. Crops are tagged "
                                       f"sRGB IEC61966-2.1 without conversion; set the camera to sRGB.")

    def write_output(self, *args, **kwargs):
        # Pass everything straight through: the base signature carries optional
        # suffix/subfolder arguments, and re-declaring them here would break the
        # moment the core gains another.
        out = super().write_output(*args, **kwargs)
        quality_class = kwargs.get("quality_class", args[6] if len(args) > 6 else None)
        if out is not None and quality_class == "premium" and self.premium_sink:
            try:
                self.premium_sink(Path(out))
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------
    # folders
    # ------------------------------------------------------------------
    def resolve_folders(self, watch: Optional[str], out: Optional[str],
                        interactive: bool) -> Tuple[Path, Path, Path]:
        m = self.cfg["merged"]
        watch_s = watch or m.get("watch_folder") or ""
        out_s = out or m.get("retouched_folder") or ""

        if interactive and not watch_s:
            watch_s = input("Folder to monitor for incoming photos: ").strip().strip('"')
        if interactive and not out_s:
            out_s = input("Folder for the finished retouched photos: ").strip().strip('"')
        if not watch_s or not out_s:
            print("[ERROR] Both a watch folder and a retouched folder are required.\n"
                  "        Pass --watch and --out, or set merged.watch_folder and\n"
                  "        merged.retouched_folder in the config.")
            sys.exit(2)

        watch_p = Path(watch_s)
        out_p = Path(out_s)
        work_s = m.get("work_folder") or ""
        work_p = Path(work_s) if work_s else out_p.parent / (out_p.name + "_work")

        if not watch_p.exists():
            print(f"[ERROR] Watch folder does not exist: {watch_p}")
            sys.exit(2)
        ensure_dir(out_p)
        ensure_dir(work_p)

        # The crop stage writes its class folders under paths.output_folder.
        self.cfg["paths"]["output_folder"] = str(work_p)
        self.output_folder = work_p
        self.rejects_folder = work_p / str(self.cfg["debug"].get("rejects_folder", "_rejects"))
        return watch_p, out_p, work_p

    def prepare_watch(self, watch: str, work: Optional[str] = None) -> Tuple[Path, Path]:
        """Set up for a run whose delivery folder is not known yet. Stage 1 can
        start immediately; stage 2 waits until the operator names a folder."""
        watch_p = Path(watch)
        if not watch_p.is_dir():
            raise ValueError(f"not a folder: {watch}")
        if work:
            work_p = Path(work)
        else:
            # A sibling of the watched folder, never inside it -- crops written
            # under the watched tree would be discovered and cropped again.
            parent = watch_p.parent if watch_p.parent != watch_p else watch_p
            work_p = parent / (watch_p.name + "_crops")
        ensure_dir(work_p)
        self.cfg["paths"]["output_folder"] = str(work_p)
        self.output_folder = work_p
        self.rejects_folder = work_p / str(self.cfg["debug"].get("rejects_folder", "_rejects"))
        return watch_p, work_p

    # ------------------------------------------------------------------
    # LUT
    # ------------------------------------------------------------------
    def _available_luts(self) -> List[Path]:
        return lutlib.list_luts(Path(str(self.cfg["lut"].get("lut_folder", ""))))

    def init_lut(self) -> None:
        self.ui_mode = False
        self.look_event = threading.Event()
        self._lut_dense = None
        self._lut_dense_name: Optional[str] = None
        self._lut_name: Optional[str] = None
        self._lut_asked = False
        if not bool(self.cfg["lut"].get("enabled", True)):
            self._lut_asked = True
            return
        available = {p.name for p in self._available_luts()}
        if not available:
            print(f"[LUT] No .cube files in {self.cfg['lut'].get('lut_folder')} -- "
                  f"delivering ungraded.")
            self._lut_asked = True
            return
        sel = self.cfg["lut"].get("selected")
        if sel and str(sel) in available:
            self._lut_name = str(sel)
            self._lut_asked = True
            print(f"[LUT] Using {self._lut_name} (set in config, no prompt).")
        elif sel:
            print(f"[LUT] Configured LUT {sel!r} is not in the LUT folder; will ask instead.")

    def ask_for_lut(self) -> None:
        """Runs on the grader thread, so cropping continues underneath it."""
        luts = self._available_luts()
        if not luts:
            self._lut_asked = True
            return
        if getattr(self, "ui_mode", False):
            # Driven by the UI: there is no terminal to read from, and reading
            # one would hit EOF and silently deliver the first batch ungraded.
            # Hold the batch instead -- cropping is on other threads and keeps
            # running -- until the operator picks a look in the page.
            print("[LUT] Waiting for a look to be chosen in the UI "
                  "(cropping continues) ...")
            while not self.look_event.wait(0.5):
                if self.stop_event is not None and self.stop_event.is_set():
                    # Shutting down before anyone chose. Leave the look
                    # unasked: the caller must not deliver this batch ungraded.
                    return
            self._lut_asked = True
            return
        print("\n" + "=" * 60)
        print("  Pick the look for this event")
        print("=" * 60)
        for i, p in enumerate(luts, 1):
            print(f"  [{i}] {p.stem}")
        print("  [0] no LUT (deliver ungraded)")
        print("\n(cropping continues while you choose)")
        while True:
            try:
                raw = input("LUT number: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[LUT] No input -- delivering ungraded.")
                self._lut_asked = True
                return
            if raw.isdigit():
                k = int(raw)
                if k == 0:
                    self._lut_name = None
                    self._lut_asked = True
                    print("[LUT] Delivering ungraded.")
                    return
                if 1 <= k <= len(luts):
                    self._lut_name = luts[k - 1].name
                    self._lut_asked = True
                    print(f"[LUT] Using {self._lut_name}.")
                    return
            print("  Please enter a number from the list.")

    def change_lut(self, name: str) -> bool:
        """Switch the look. Takes effect from the next batch; already delivered
        batches keep the look they were graded with, because re-grading a
        written JPEG would compress it a second time."""
        available = {p.name: p for p in self._available_luts()}
        if name not in available:
            print(f"[LUT] {name} is not in the LUT folder.")
            return False
        self._lut_name = name
        self._lut_asked = True
        if getattr(self, "look_event", None) is not None:
            self.look_event.set()
        print(f"[LUT] Look changed to {name} -- applies from the next batch.")
        return True

    def _grade_fn(self) -> Optional[Callable]:
        if not self._lut_name:
            return None
        if self._lut_dense is None or self._lut_dense_name != self._lut_name:
            path = Path(str(self.cfg["lut"]["lut_folder"])) / self._lut_name
            strength = float(self.cfg["lut"].get("strength", 1.0))
            print(f"[LUT] Building lookup table for {self._lut_name} ...")
            try:
                self._lut_dense = lutlib.build_dense_lut_bgr(lutlib.parse_cube(path), strength)
                self._lut_dense_name = self._lut_name
            except Exception as exc:
                print(f"[WARN] Could not load {self._lut_name} ({exc}); delivering ungraded.")
                self._lut_dense = None
                self._lut_dense_name = None
                self._lut_name = None
                return None
        dense = self._lut_dense
        return lambda img: lutlib.apply_dense_bgr(img, dense)

    # ------------------------------------------------------------------
    # stage 2: finish + grade + deliver
    # ------------------------------------------------------------------
    # Why a crop failed, recorded per path so the Pipeline can tell a bad crop
    # ("unreadable": counts against that crop) from a bad destination ("write":
    # says nothing about the crop). Threads write distinct keys.
    def _finish_one(self, src: Path, out_folder: Path, quality: int,
                    grade: Optional[Callable]) -> bool:
        img = cv2.imread(str(src))
        if img is None:
            self._faults[src] = "unreadable"
            return False
        self._finishing = True
        try:
            img = super().resize_final(img, enhance_mode="none")
            img = super().sharpen_image(img)
        finally:
            self._finishing = False
        if grade is not None:
            img = grade(img)
        dst = out_folder / src.name
        # Created here, on the first crop that is actually ready to be written,
        # so a batch in which nothing can be written leaves no empty folder.
        ensure_dir(out_folder)
        if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            self._faults[src] = "write"
            return False
        complete = self._copy_exif_without_orientation(src, dst)
        # The finished copy is in place, so the native-size crop is no longer
        # needed. If its metadata could not be written it stays, next to the
        # warning, rather than being the only good copy that gets thrown away.
        if complete and bool(self.cfg["merged"].get("delete_crops_after_delivery", True)):
            try:
                src.unlink()
            except OSError:
                pass
        return True

    def deliver_batch(self, crops: List[Path], batch_no: int, out_root: Path) -> Tuple[int, int]:
        prefix = str(self.cfg["merged"].get("batch_folder_prefix", "") or "").strip()
        # No separator: the folder name matches the delivery note row exactly,
        # so "toby11" names both.
        out_folder = out_root / (f"{prefix}{batch_no}" if prefix else str(batch_no))
        # The folder itself is made by _finish_one on the first successful write.
        quality = int(self.cfg["image_quality"].get("jpeg_quality", 95))
        grade = self._grade_fn()
        look = self._lut_name or "ungraded"
        print(f"\n[STAGE 2] Batch #{batch_no}: {len(crops)} crop(s) -> {out_folder}  ({look})")

        done = failed = 0
        t0 = time.time()
        workers = max(1, int(self.cfg["performance"].get("workers", 8)))
        workers = min(workers, 8, len(crops))    # grading is memory-hungry per image
        bar = tqdm(total=len(crops), desc=f"Batch {batch_no}") if tqdm is not None else None

        failed_paths: List[Path] = []
        self._faults = {}

        def one(p: Path) -> bool:
            try:
                ok = self._finish_one(p, out_folder, quality, grade)
            except OSError as exc:
                # Could not create the folder or write into it: the destination
                # is the suspect, not this crop.
                print(f"\n[WARN] {p.name}: {exc}")
                self._faults[p] = "write"
                ok = False
            except Exception as exc:
                print(f"\n[WARN] {p.name}: {exc}")
                self._faults[p] = "unreadable"
                ok = False
            if not ok:
                failed_paths.append(p)
            return ok

        if workers <= 1:
            for p in crops:
                ok = one(p)
                done += 1 if ok else 0
                failed += 0 if ok else 1
                if bar:
                    bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for ok in pool.map(one, crops):
                    done += 1 if ok else 0
                    failed += 0 if ok else 1
                    if bar:
                        bar.update(1)
        if bar:
            bar.close()
        secs = time.time() - t0
        print(f"[STAGE 2] Batch #{batch_no} delivered: {done} ok, {failed} failed, "
              f"{secs:.1f}s ({secs / max(1, len(crops)) * 1000:.0f} ms/crop)")
        self.last_batch_failed = list(failed_paths)
        self.last_batch_faults = dict(self._faults)
        if done == 0:
            # Nothing landed. Do not leave an empty numbered folder behind; it
            # would use up a number and never get a line in the delivery note.
            try:
                out_folder.rmdir()
            except OSError:
                pass
        self.last_batch_folder = out_folder.name
        # What the folder holds is the only count worth reporting: writes can
        # collide on a filename, and the uploader acts on the folder.
        try:
            self.last_batch_count = len(list(out_folder.glob("*.jpg")))
        except OSError:
            self.last_batch_count = done
        if self.last_batch_count != done:
            print(f"[WARN] batch #{batch_no}: {done} written but "
                  f"{self.last_batch_count} file(s) in {out_folder.name}")
        return done, failed


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------
class Pipeline:
    # A crop that cannot be read is retried this many times, then set aside.
    MAX_CROP_FAILURES = 3
    # Wait after the destination refuses a batch: base * 2^(streak-1), capped.
    BACKOFF_BASE = 5.0
    BACKOFF_MAX = 60.0

    def __init__(self, suite: MergedSuite, watch: Path, out: Optional[Path], work: Path,
                 recrop: bool = False):
        self.s = suite
        self.watch = watch
        self.out = out
        self.work = work
        m = suite.cfg["merged"]
        self.batch_size = int(m.get("premium_batch_size", 100))
        self.batch_timeout = float(m.get("premium_batch_timeout_sec", 120))
        self.poll = float(m.get("poll_interval_sec", 4.0))
        self.recursive = bool(m.get("recursive", True))
        self.stability = int(m.get("file_stability_checks", 2))
        self.stop_idle = float(m.get("stop_when_idle_sec", 0))

        self.seen: Set[str] = set()
        self._sizes: Dict[str, Tuple[int, int]] = {}
        self.pending: List[Path] = []
        # Every crop ever queued, so a resume cannot re-add something that is
        # already waiting. Duplicates inside one batch overwrite each other in
        # the output folder, silently shrinking it.
        self._queued: Set[str] = set()
        self.lock = threading.Lock()
        # Two different things: stop_requested is the operator (or Ctrl+C)
        # asking to finish; stop is set once cropping has actually wound down,
        # and is what tells the grader to flush what is left and leave.
        self.stop_requested = threading.Event()
        self.stop = threading.Event()
        self.crops_done = 0
        # Stage 1 can be paused without stopping the run (grading carries on).
        self.cropping_on = threading.Event()
        self.cropping_on.set()
        # Stop asked to leave without delivering what is still queued; those
        # crops stay in the work folder and are picked up again on the next run.
        self.abandon = threading.Event()
        self.finished = threading.Event()        # run() has fully wound down
        self._in_flight = 0                      # photos being cropped right now
        # Which photos were already cropped, kept on disk so a stopped or
        # crashed run carries on where it left off instead of starting over.
        self._ledger_path = work / self._CROP_LEDGER
        self._ledger_lock = threading.Lock()
        self._cropped_before: Set[str] = set()
        self.skipped_known = 0                   # photos skipped because an earlier run did them
        # The engine only writes its per-photo report at the end of its own batch
        # mode, which this pipeline does not use, so it is written here instead:
        # one file per run in the work folder, refreshed while running.
        self._report_path: Optional[Path] = None
        self._report_written = 0.0
        self._report_warned = False
        if recrop:
            self._reset_ledger()
        else:
            self._load_ledger()
        self.batch_no = 1
        self.batches_done = 0        # folders actually delivered this run
        self._fail_counts: Dict[str, int] = {}   # crop path -> times it has failed
        self.quarantined: List[str] = []         # crops given up on this run
        self._dest_fail_streak = 0               # consecutive batches the destination refused
        self.created = time.time()   # crops older than this are from an earlier run
        self.delivered = 0
        self.last_premium = time.time()
        self.last_arrival = time.time()
        suite.premium_sink = self._on_premium
        suite.stop_event = self.stop
        self.ui_log: List[str] = []      # short activity feed for the UI
        self.grading_now: Optional[int] = None
        self.desired_workers = max(1, int(suite.cfg["performance"].get("workers", 8)))
        self.active_workers = self.desired_workers
        # Grading can be held without touching stage 1; crops keep piling up
        # in pending and go out when it is resumed.
        self.grading_on = threading.Event()
        self.grading_on.set()
        # Called with (batch_no, folder_name, images) once a folder is closed.
        self.on_batch_done: Optional[Callable[[int, str, int], None]] = None
        # A destination change is staged here and applied between batches --
        # switching under a running batch splits it across two folders and
        # fails most of its writes.
        self.pending_out: Optional[Path] = None

    def note(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.ui_log.append(f"{stamp}  {msg}")
        if len(self.ui_log) > 60:
            del self.ui_log[:-60]

    # -- stage 1 -------------------------------------------------------
    def _stable(self, p: Path) -> bool:
        try:
            size = p.stat().st_size
        except OSError:
            return False
        key = str(p)
        prev, streak = self._sizes.get(key, (-1, 0))
        streak = streak + 1 if size == prev else 1
        self._sizes[key] = (size, streak)
        return streak >= self.stability

    # -- which photos were already cropped ------------------------------
    # One line per photo: size, modified time (ns) and path relative to the
    # watched folder. Size and time are part of the key because cameras reuse
    # file names when the counter restarts: a new IMG_0001.JPG is a new photo.
    _CROP_LEDGER = "_cropped.log"

    def _photo_key(self, p: Path) -> Optional[str]:
        try:
            st = p.stat()
            rel = os.path.normcase(str(p.relative_to(self.watch)))
        except (OSError, ValueError):
            return None
        return f"{st.st_size}\t{st.st_mtime_ns}\t{rel}"

    def _load_ledger(self) -> None:
        try:
            for line in self._ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.count("\t") >= 2 and line.split("\t", 2)[0].isdigit():   # a half-written last line is ignored
                    self._cropped_before.add(line)
        except OSError:
            pass

    def _reset_ledger(self) -> None:
        try:
            self._ledger_path.unlink()
        except OSError:
            pass

    def _record_cropped(self, p: Path) -> None:
        key = self._photo_key(p)
        if key is None:
            return
        with self._ledger_lock:
            try:
                with open(self._ledger_path, "a", encoding="utf-8") as f:
                    f.write(key + "\n")
            except OSError as exc:
                self._warn_ledger(exc)

    def _warn_ledger(self, exc: Exception) -> None:
        if not getattr(self, "_ledger_warned", False):
            self._ledger_warned = True
            print(f"[WARN] cannot record cropped photos in {self._ledger_path} ({exc}); "
                  f"a restart will crop them again")

    def _scan(self) -> List[Path]:
        it = self.watch.rglob("*") if self.recursive else self.watch.glob("*")
        fresh = []
        # Anything we write ourselves is not an arrival. If the work or
        # delivery folder ever sits inside the watched tree, crops would
        # otherwise be discovered and cropped again, forever.
        ours = [d.resolve() for d in (self.work, self.out) if d is not None]
        try:
            for p in it:
                if str(p) in self.seen or not p.is_file() or not _is_candidate(p.name):
                    continue
                try:
                    if any(o in p.resolve().parents for o in ours):
                        continue
                except OSError:
                    continue
                if self._cropped_before:
                    key = self._photo_key(p)
                    if key is not None and key in self._cropped_before:
                        self.seen.add(str(p))
                        self.skipped_known += 1
                        continue
                if self._stable(p):
                    self.seen.add(str(p))
                    fresh.append(p)
        except OSError:
            pass
        return fresh

    REPORT_EVERY_SEC = 120.0

    def write_crop_report(self) -> Optional[Path]:
        """Save what the engine decided for every photo this run (accepted or
        rejected, and why) as runner_crop_report_<date>_<time>.csv in the work
        folder. A new file per run, so a restart never overwrites an earlier one."""
        if not bool(self.s.cfg.get("debug", {}).get("write_csv", True)):
            return None
        report_lock = getattr(self.s, "report_lock", None)
        with (report_lock or threading.Lock()):
            rows = list(getattr(self.s, "report_rows", []))
        self._report_written = time.time()
        if not rows:
            return None
        if self._report_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path, n = self.work / f"runner_crop_report_{stamp}.csv", 2
            while path.exists():
                path = self.work / f"runner_crop_report_{stamp}_{n}.csv"; n += 1
            self._report_path = path
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        tmp = self._report_path.with_name(self._report_path.name + ".part")
        try:
            with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, self._report_path)
        except OSError as exc:                  # e.g. the file is open in Excel
            if not self._report_warned:
                self._report_warned = True
                print(f"[WARN] could not save {self._report_path.name} ({exc}); will try again")
            return None
        return self._report_path

    def _on_premium(self, path: Path) -> None:
        """Called by the suite from whichever worker wrote the crop."""
        key = str(path)
        with self.lock:
            if key in self._queued:
                return
            self._queued.add(key)
            self.pending.append(path)
            self.last_premium = time.time()

    def _crop_one(self, p: Path) -> None:
        # Paused: hold here, do not drop the photo. It is already marked as
        # seen, so skipping it would lose it. Stop still gets through.
        while not self.cropping_on.is_set() and not self.stop_requested.is_set():
            time.sleep(0.2)
        if self.stop_requested.is_set():
            # A large backlog is one pool.map(); without this, Stop would wait
            # for every photo in it. Skipped photos are not in the ledger, so
            # the next run crops them.
            return
        with self.lock:
            self._in_flight += 1
        try:
            self.s.process_image(p)
        except Exception as exc:
            print(f"[ERROR] crop failed for {p.name}: {exc}")
            return
        finally:
            with self.lock:
                self._in_flight -= 1
        self._record_cropped(p)
        with self.lock:
            self.crops_done += 1

    # -- stage 2 (own thread, so the LUT prompt never stalls stage 1) ---
    def _can_deliver(self) -> bool:
        """Could a batch go out right now if one were ready? Used at shutdown
        to tell "still draining" from "cannot drain, so leave"."""
        if not self.grading_on.is_set():
            return False                      # the operator paused grading
        if self.out is None:
            return False                      # no destination was ever chosen
        if not self.s._lut_asked and getattr(self.s, "ui_mode", False):
            return False                      # nobody picked a look
        return True

    def _next_free_batch_no(self) -> int:
        """One past the highest numbered folder already in the destination.

        Numbering restarts at 1 with every run, so without this a restart, or a
        second event delivered to the same place, wrote into finish1 again and
        left two note rows for the same folder."""
        prefix = str(self.s.cfg["merged"].get("batch_folder_prefix", "") or "").strip()
        pat = re.compile(r"^" + re.escape(prefix) + r"(\d+)$")
        top = 0
        try:
            for d in self.out.iterdir():
                if d.is_dir():
                    m = pat.match(d.name)
                    if m:
                        top = max(top, int(m.group(1)))
        except OSError:
            pass
        return top + 1

    def _quarantine(self, p: Path, kind: str) -> None:
        """Give up on one crop. It is moved to _failed in the work folder rather
        than deleted, so nothing is lost, but it stops being retried."""
        dest_dir = self.work / "_failed"
        try:
            dest_dir.mkdir(exist_ok=True)
            p.rename(dest_dir / p.name)
            where = f"moved to {dest_dir.name}"
        except OSError:
            where = "left where it is"
        self.quarantined.append(p.name)
        msg = (f"{p.name}: failed {self.MAX_CROP_FAILURES} times ({kind}) -- "
               f"given up, {where}")
        self.note(msg)
        print(f"[STAGE 2] {msg}")

    def _backoff(self, why: str) -> None:
        """The destination refused a batch. Wait, rather than retrying flat out,
        and never let it take the grader thread down."""
        self._dest_fail_streak += 1
        delay = min(self.BACKOFF_MAX, self.BACKOFF_BASE * 2 ** (self._dest_fail_streak - 1))
        self.note(f"{why} -- retrying in {delay:.0f}s")
        print(f"[STAGE 2] {why} -- retrying in {delay:.0f}s")
        self.stop.wait(delay)               # returns at once if we are shutting down

    def _abandon(self, n: int) -> None:
        msg = (f"stopped with {n} Premium crop(s) undelivered -- they stay in "
               f"the work folder")
        self.note(msg)
        print(f"[STOP] {msg}")

    def _grader(self) -> None:
        while True:
            stopping = self.stop.is_set()
            with self.lock:
                n = len(self.pending)
                idle = time.time() - self.last_premium
            if stopping and n == 0:
                return
            if stopping and self.abandon.is_set():
                self._abandon(n)                 # the operator chose not to wait for delivery
                return
            if stopping and n and self._dest_fail_streak >= 3:
                self._abandon(n)                 # the destination is not coming back
                return
            if stopping and not self._can_deliver():
                # Nothing more can go out: no destination, grading paused, or
                # no look chosen. Waiting here is what used to leave Stop
                # hanging for the full join timeout.
                self._abandon(n)
                return
            ready = n >= self.batch_size or (n > 0 and (idle >= self.batch_timeout or stopping))
            if not ready or not self.grading_on.is_set():
                time.sleep(0.5)
                continue
            if not self.s._lut_asked or self.out is None:
                self.s.ask_for_lut()
                if not self.s._lut_asked:
                    continue            # shutdown interrupted the wait; re-evaluate above
            if self.pending_out is not None:
                self.out = self.pending_out
                self.pending_out = None
                self._write_target(self.out)
                self.note(f"delivering to {self.out}")
            if self.out is None:            # still nowhere to deliver
                time.sleep(0.5)
                continue
            with self.lock:
                take = self.pending[:self.batch_size]
                self.pending = self.pending[self.batch_size:]
            take = [p for p in take if p.exists()]
            if not take:
                continue
            # Never reuse a folder number that already exists in the destination.
            self.batch_no = max(self.batch_no, self._next_free_batch_no())
            no = self.batch_no
            self.batch_no += 1
            self.grading_now = no
            self.note(f"batch #{no} grading with {self.s._lut_name or 'no LUT'}")
            try:
                done, _ = self.s.deliver_batch(take, no, self.out)
            except Exception as exc:
                # Anything that escapes here would end this thread for good:
                # cropping carries on, the queue grows, nothing is delivered.
                self.grading_now = None
                with self.lock:
                    self.pending = take + self.pending
                self.batch_no = no
                self._backoff(f"batch #{no} could not be delivered "
                              f"({type(exc).__name__}: {exc})")
                continue
            self.grading_now = None
            self.delivered += done
            if done:
                self.batches_done += 1
                self._dest_fail_streak = 0
            else:
                self.batch_no = no          # nothing was written: the number is unused
            # Failed crops are put back rather than lost -- a short folder is
            # worse than a slow one -- but not forever. A crop that cannot be
            # read is retried a few times and then set aside. A write failure
            # only counts against the crop if other crops in the same batch
            # were written, i.e. the destination demonstrably works.
            faults = getattr(self.s, "last_batch_faults", {})
            retry: List[Path] = []
            for rp in getattr(self.s, "last_batch_failed", []):
                if not rp.exists():
                    continue
                kind = faults.get(rp, "write")
                if kind == "unreadable" or done > 0:
                    k = self._fail_counts.get(str(rp), 0) + 1
                    self._fail_counts[str(rp)] = k
                    if k >= self.MAX_CROP_FAILURES:
                        self._quarantine(rp, kind)
                        continue
                retry.append(rp)
            if retry:
                with self.lock:
                    self.pending.extend(retry)
                    self.last_premium = time.time()
                self.note(f"batch #{no}: {len(retry)} crop(s) failed -- requeued")
                print(f"[STAGE 2] batch #{no}: {len(retry)} failed write(s) requeued")
            if not done and any(faults.get(rp, "write") == "write" for rp in retry):
                self._backoff(f"batch #{no}: the destination would not take any files")
                continue
            if not done:
                self.note(f"batch #{no}: nothing could be delivered")
                continue
            self.note(f"batch #{no} delivered — {done} crops")
            # The folder is complete and closed here: this is the one moment
            # its image count is final, so the delivery note is written now
            # and never touched again.
            if self.on_batch_done and done:
                try:
                    self.on_batch_done(no,
                                       getattr(self.s, "last_batch_folder", str(no)),
                                       int(getattr(self.s, "last_batch_count", done)))
                except Exception as exc:
                    print(f"[WARN] delivery note: {exc}")

    # -- main ----------------------------------------------------------
    # -- destination ledger --------------------------------------------
    # The work folder is named after the watch folder, so it outlives a run and
    # is shared by every later event that watches the same folder. A small
    # ledger records where its crops were being delivered, so leftovers from
    # one delivery can never be picked up by another.
    _LEDGER = "_delivery_target.txt"

    @staticmethod
    def _norm(p: Any) -> str:
        return os.path.normcase(os.path.abspath(str(p)))

    def _read_target(self) -> Optional[str]:
        try:
            txt = (self.work / self._LEDGER).read_text(encoding="utf-8").strip()
            return txt or None
        except OSError:
            return None

    def _write_target(self, out: Path) -> None:
        try:
            (self.work / self._LEDGER).write_text(str(out), encoding="utf-8")
        except OSError as exc:
            print(f"[WARN] could not record the delivery target: {exc}")

    def _set_aside(self, crop_dir: Path, files: List[Path]) -> int:
        """Move crops that belong to some other delivery into a dated subfolder
        of the work folder. Nothing is deleted; they simply stop being
        candidates for this run."""
        dest = crop_dir / time.strftime("_previous_%Y%m%d_%H%M%S")
        moved = 0
        try:
            dest.mkdir(exist_ok=True)
        except OSError as exc:
            print(f"[WARN] could not set aside earlier crops: {exc}")
            return 0
        for f in files:
            try:
                f.rename(dest / f.name)
                moved += 1
            except OSError:
                pass
        return moved

    def resume_from_disk(self) -> int:
        """Queue Premium crops that were cut but never delivered. The pending
        list lives in memory, so a restart would otherwise strand them even
        though the crops themselves are safely on disk.

        Only crops from an earlier run of *this same delivery* qualify. Crops
        the ledger says were bound for a different folder, or that have no
        ledger at all (an earlier event, or a version that did not keep one),
        are set aside rather than delivered into the wrong order. Their source
        photos, if still in the watch folder, are simply cropped again."""
        crop_dir = self.s.output_folder / self.s._quality_folder_name("premium")
        if not crop_dir.exists():
            return 0
        with self.lock:
            queued = set(self._queued)
        # A leftover is a crop from before this run began. This run's own crops
        # are in _queued, and the mtime test covers the instant between a crop
        # being written and being announced -- moving that one would lose it.
        leftovers = []
        for f in sorted(crop_dir.glob("*.jpg")):
            if str(f) in queued:
                continue
            try:
                if f.stat().st_mtime < self.created:
                    leftovers.append(f)
            except OSError:
                pass
        if self.out is None:
            return 0
        target = self._read_target()
        if leftovers and (target is None or self._norm(target) != self._norm(self.out)):
            moved = self._set_aside(crop_dir, leftovers)
            self.note(f"{moved} crop(s) from an earlier delivery set aside, not resumed")
            print(f"[RESUME] {moved} leftover crop(s) belong to another delivery "
                  f"({target or 'no record'}) -- set aside in {crop_dir.name}/_previous_*")
            leftovers = []
        self._write_target(self.out)
        already = set()
        if self.out.exists():
            already = {f.name for f in self.out.rglob("*.jpg")}
        missing = [f for f in leftovers if f.name not in already]
        if missing:
            with self.lock:
                self._queued.update(str(f) for f in missing)
                self.pending.extend(missing)
                self.last_premium = time.time()
            self.note(f"resumed {len(missing)} crop(s) not yet delivered")
            print(f"[RESUME] {len(missing)} crop(s) already cut but not delivered -- queued.")
        return len(missing)

    def request_stop(self) -> None:
        """Operator asked to stop. Cropping ends after the photos already in
        flight; the grader then delivers whatever can go out and leaves."""
        self.stop_requested.set()
        self.note("stopping -- finishing what is queued")

    def set_cropping(self, on: bool) -> bool:
        """Pause or resume stage 1. Photos already being cropped finish; the
        rest wait, and nothing is lost. Grading and delivery carry on."""
        if on:
            self.cropping_on.set()
            self.note("cropping resumed")
        else:
            self.cropping_on.clear()
            self.note("cropping paused -- grading continues")
        return on

    def skip_delivery(self) -> None:
        """While stopping: do not wait for the queued crops to be delivered.
        They stay in the work folder and are offered again on the next run."""
        self.abandon.set()
        self.note("delivery skipped -- queued crops stay in the work folder")

    @property
    def in_flight(self) -> int:
        with self.lock:
            return self._in_flight

    @property
    def stopping(self) -> bool:
        return self.stop_requested.is_set() and not self.finished.is_set()

    def set_grading(self, on: bool) -> bool:
        if on:
            self.grading_on.set()
            self.note("grading resumed")
        else:
            self.grading_on.clear()
            self.note("grading paused -- cropping continues")
        return on

    def set_workers(self, n: int) -> int:
        """Take effect between scans; the pool is swapped, not resized."""
        n = max(1, min(64, int(n)))
        self.desired_workers = n
        return n

    def run(self) -> None:
        workers = self.desired_workers
        if workers > 1:
            try:
                cv2.setNumThreads(1)     # parallelism comes from the workers
            except Exception:
                pass

        print("\n" + "=" * 72)
        print("MERGED SUITE -- crop + grade")
        print("=" * 72)
        print(f"Monitoring : {self.watch}")
        print(f"Retouched  : {self.out}")
        print(f"Work crops : {self.work}")
        print(f"Batch      : {self.batch_size} Premium crops, or {self.batch_timeout:.0f}s idle")
        print(f"Workers    : {workers} cropping (changeable while running)")
        print("=" * 72 + "\n")

        self.note(f"watching {self.watch}")
        grader = threading.Thread(target=self._grader, name="grader", daemon=True)
        grader.start()
        t0 = time.time()
        pool = ThreadPoolExecutor(max_workers=workers)
        self.active_workers = workers
        try:
            while True:
                if self.stop_requested.is_set():
                    print("\n[STOP] Stop requested -- finishing up.")
                    break
                if self.desired_workers != self.active_workers:
                    # Swap between scans, never mid-batch: a ThreadPoolExecutor
                    # cannot be resized, so the old one is drained first.
                    old_n = self.active_workers
                    pool.shutdown(wait=True)
                    pool = ThreadPoolExecutor(max_workers=self.desired_workers)
                    self.active_workers = self.desired_workers
                    self.note(f"crop workers {old_n} -> {self.active_workers}")
                    print(f"[STAGE 1] crop workers {old_n} -> {self.active_workers}")
                if not self.cropping_on.is_set():
                    time.sleep(0.3)              # paused: pick up nothing new until resumed
                    continue
                if time.time() - self._report_written >= self.REPORT_EVERY_SEC:
                    self.write_crop_report()
                fresh = self._scan()
                if fresh:
                    self.last_arrival = time.time()
                    print(f"[STAGE 1] {len(fresh)} new photo(s); cropping ...")
                    self.note(f"{len(fresh)} new photo(s) arrived")
                    list(pool.map(self._crop_one, fresh))
                    with self.lock:
                        q = len(self.pending)
                    print(f"[STAGE 1] {self.crops_done} photo(s) cropped, "
                          f"{q} Premium queued for grading")
                elif self.stop_idle and time.time() - self.last_arrival > self.stop_idle:
                    print(f"\n[STAGE 1] Nothing new for {self.stop_idle:.0f}s -- finishing up.")
                    break
                else:
                    time.sleep(self.poll)
        except KeyboardInterrupt:
            print("\n[STOP] Ctrl+C -- draining the queue ...")
        finally:
            pool.shutdown(wait=True)
            self.stop.set()
            grader.join(timeout=1800)
            report = self.write_crop_report()
            self.finished.set()
            secs = time.time() - t0
            print("\n" + "=" * 72)
            print("MERGED SUITE STOPPED")
            print("=" * 72)
            print(f"photos cropped   : {self.crops_done}")
            print(f"crops delivered  : {self.delivered}")
            print(f"batches          : {self.batches_done}")
            print(f"look             : {self.s._lut_name or 'ungraded'}")
            print(f"wall             : {secs:.1f}s "
                  f"({secs / max(1, self.crops_done):.3f}s per photo)")
            print(f"retouched folder : {self.out}")
            if report:
                print(f"crop report      : {report}")
            print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merged crop + grade pipeline")
    ap.add_argument("config", nargs="?", default="settings_merged.json")
    ap.add_argument("--watch", help="folder to monitor for incoming photos")
    ap.add_argument("--out", help="folder for the finished retouched photos")
    ap.add_argument("--lut", help="LUT filename to use (skips the prompt)")
    ap.add_argument("--batch", type=int, help="Premium crops per batch")
    ap.add_argument("--stop-idle", type=float,
                    help="exit after this many seconds with no new photos")
    ap.add_argument("--recrop", action="store_true",
                    help="crop photos again that an earlier run of this folder already cropped")
    ap.add_argument("--no-ask", action="store_true",
                    help="never prompt (fails if the folders are not configured)")
    args = ap.parse_args()

    suite = MergedSuite(args.config)
    if args.lut:
        suite.cfg["lut"]["selected"] = args.lut
    if args.batch:
        suite.cfg["merged"]["premium_batch_size"] = args.batch
    if args.stop_idle is not None:
        suite.cfg["merged"]["stop_when_idle_sec"] = args.stop_idle

    interactive = not args.no_ask and sys.stdin is not None and sys.stdin.isatty()
    watch, out, work = suite.resolve_folders(args.watch, args.out, interactive)
    suite.init_lut()
    Pipeline(suite, watch, out, work, recrop=args.recrop).run()


if __name__ == "__main__":
    main()
