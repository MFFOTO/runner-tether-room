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
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2

import lut as lutlib
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
        "note_dir": "",                 # where delivery.txt goes; "" -> with the photos
        "stop_when_idle_sec": 0,        # >0: exit after this long with nothing new (for tests)
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
            self.look_event.wait()
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
    def _finish_one(self, src: Path, out_folder: Path, quality: int,
                    grade: Optional[Callable]) -> bool:
        img = cv2.imread(str(src))
        if img is None:
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
        if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            return False
        self._copy_exif_without_orientation(src, dst)
        return True

    def deliver_batch(self, crops: List[Path], batch_no: int, out_root: Path) -> Tuple[int, int]:
        prefix = str(self.cfg["merged"].get("batch_folder_prefix", "") or "").strip()
        # No separator: the folder name matches the delivery note row exactly,
        # so "toby11" names both.
        out_folder = out_root / (f"{prefix}{batch_no}" if prefix else str(batch_no))
        ensure_dir(out_folder)
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

        def one(p: Path) -> bool:
            try:
                ok = self._finish_one(p, out_folder, quality, grade)
            except Exception as exc:
                print(f"\n[WARN] {p.name}: {exc}")
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
    def __init__(self, suite: MergedSuite, watch: Path, out: Optional[Path], work: Path):
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
        self.stop = threading.Event()
        self.crops_done = 0
        self.batch_no = 1
        self.delivered = 0
        self.last_premium = time.time()
        self.last_arrival = time.time()
        suite.premium_sink = self._on_premium
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
                if self._stable(p):
                    self.seen.add(str(p))
                    fresh.append(p)
        except OSError:
            pass
        return fresh

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
        try:
            self.s.process_image(p)
        except Exception as exc:
            print(f"[ERROR] crop failed for {p.name}: {exc}")
            return
        with self.lock:
            self.crops_done += 1

    # -- stage 2 (own thread, so the LUT prompt never stalls stage 1) ---
    def _grader(self) -> None:
        while not self.stop.is_set() or self.pending:
            with self.lock:
                n = len(self.pending)
                idle = time.time() - self.last_premium
            ready = n >= self.batch_size or (n > 0 and (idle >= self.batch_timeout or self.stop.is_set()))
            if not ready or not self.grading_on.is_set():
                time.sleep(0.5)
                continue
            if not self.s._lut_asked or self.out is None:
                self.s.ask_for_lut()
            if self.pending_out is not None:
                self.out = self.pending_out
                self.pending_out = None
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
            no = self.batch_no
            self.batch_no += 1
            self.grading_now = no
            self.note(f"batch #{no} grading with {self.s._lut_name or 'no LUT'}")
            done, _ = self.s.deliver_batch(take, no, self.out)
            self.grading_now = None
            self.delivered += done
            # Anything that failed to write is put back rather than lost; a
            # short folder is worse than a slow one.
            retry = [p for p in getattr(self.s, "last_batch_failed", []) if p.exists()]
            if retry:
                with self.lock:
                    for rp in retry:
                        self._queued.discard(str(rp))
                        self._queued.add(str(rp))
                        self.pending.append(rp)
                    self.last_premium = time.time()
                self.note(f"batch #{no}: {len(retry)} crop(s) failed -- requeued")
                print(f"[STAGE 2] batch #{no}: {len(retry)} failed write(s) requeued")
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
    def resume_from_disk(self) -> int:
        """Queue Premium crops that were cut but never delivered. The pending
        list lives in memory, so a restart would otherwise strand them even
        though the crops themselves are safely on disk."""
        crop_dir = self.s.output_folder / self.s._quality_folder_name("premium")
        if not crop_dir.exists():
            return 0
        already = set()
        if self.out is not None and self.out.exists():
            already = {f.name for f in self.out.rglob("*.jpg")}
        with self.lock:
            queued = set(self._queued)
        missing = [f for f in sorted(crop_dir.glob("*.jpg"))
                   if f.name not in already and str(f) not in queued]
        if missing:
            with self.lock:
                self._queued.update(str(f) for f in missing)
                self.pending.extend(missing)
                self.last_premium = time.time()
            self.note(f"resumed {len(missing)} crop(s) not yet delivered")
            print(f"[RESUME] {len(missing)} crop(s) already cut but not delivered -- queued.")
        return len(missing)

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
                if self.desired_workers != self.active_workers:
                    # Swap between scans, never mid-batch: a ThreadPoolExecutor
                    # cannot be resized, so the old one is drained first.
                    old_n = self.active_workers
                    pool.shutdown(wait=True)
                    pool = ThreadPoolExecutor(max_workers=self.desired_workers)
                    self.active_workers = self.desired_workers
                    self.note(f"crop workers {old_n} -> {self.active_workers}")
                    print(f"[STAGE 1] crop workers {old_n} -> {self.active_workers}")
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
            secs = time.time() - t0
            print("\n" + "=" * 72)
            print("MERGED SUITE STOPPED")
            print("=" * 72)
            print(f"photos cropped   : {self.crops_done}")
            print(f"crops delivered  : {self.delivered}")
            print(f"batches          : {self.batch_no - 1}")
            print(f"look             : {self.s._lut_name or 'ungraded'}")
            print(f"wall             : {secs:.1f}s "
                  f"({secs / max(1, self.crops_done):.3f}s per photo)")
            print(f"retouched folder : {self.out}")
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
    Pipeline(suite, watch, out, work).run()


if __name__ == "__main__":
    main()
