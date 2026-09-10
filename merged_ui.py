# -*- coding: utf-8 -*-
"""
Operator UI for the merged crop + grade pipeline.

Serves a local page on 127.0.0.1 that lets the operator:
  * browse for and pick the folder to monitor,
  * browse for and pick the folder for the finished retouched photos,
  * start and stop the run and watch both stages live,
  * pick a look from the available LUTs, previewed on this event's own crops,
  * enlarge any preview,
  * change the look at any time -- which never pauses cropping.

Deliberately a local page rather than anything hosted: this runs on a laptop at
a venue where the internet may be absent, and it has to read crops straight off
the local disk.
"""

from __future__ import annotations

import io
import json
import mimetypes
import string
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

import lut as lutlib
from merged_suite import MergedSuite, Pipeline

PORT = 8770
THUMB_H = 460
BIG_H = 1500
RAIL_H = 66
SAMPLES = 4
MANIFEST = "delivery.txt"
# Where the delivery note goes by default. Set per machine in
# settings_merged.json under merged.note_dir; "" puts it beside the photos.
DEFAULT_NOTE_DIR = ""
BOGUS_SPOT = "bogus"
BOGUS_IMAGES = "999999"


class App:
    """Everything the UI needs, and the only thing that touches the pipeline."""

    def __init__(self, config: str):
        self.config = config
        self.suite: Optional[MergedSuite] = None
        self.pipe: Optional[Pipeline] = None
        self.thread: Optional[threading.Thread] = None
        self.status = "loading"          # loading -> ready -> running -> stopped
        self.error = ""
        self.watch = ""
        self.out = ""
        self.started_at = 0.0
        self._tiles: Dict[str, bytes] = {}
        self._samples: List[np.ndarray] = []
        self._sample_names: List[str] = []
        self._sample_lock = threading.Lock()
        self._sampled_from = 0            # crops present when samples were drawn
        self._warming = False
        self.job = {"spot": "", "order": "", "desc": ""}
        self._manifest_lock = threading.Lock()
        self._manifest_lines = 0
        self.note_dir = DEFAULT_NOTE_DIR    # where the bulk loader reads it from
        self.note_name = MANIFEST
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self) -> None:
        try:
            self.suite = MergedSuite(self.config)
            self.suite.init_lut()
            configured = str(self.suite.cfg["merged"].get("note_dir", "") or "").strip()
            if configured:
                self.note_dir = configured
            self.status = "ready"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"

    # ---------------- folders ----------------
    @staticmethod
    def browse(path: str) -> Dict[str, Any]:
        """Drives when no path; otherwise subfolders, with a JPEG count so the
        operator can confirm a folder holds photos before choosing it."""
        if not path:
            drives = [f"{d}:\\" for d in string.ascii_uppercase
                      if Path(f"{d}:\\").exists()]
            return {"path": "", "parent": None, "dirs": drives, "images": 0}
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return {"path": path, "parent": str(p.parent), "dirs": [], "images": 0,
                    "error": "not a folder"}
        dirs, images = [], 0
        try:
            for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if c.is_dir():
                    dirs.append(c.name)
                elif c.suffix.lower() in (".jpg", ".jpeg"):
                    images += 1
        except OSError as exc:
            return {"path": path, "parent": str(p.parent), "dirs": [], "images": 0,
                    "error": str(exc)}
        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "dirs": dirs, "images": images}

    # ---------------- run control ----------------
    def start(self, watch: str) -> Dict[str, Any]:
        """Only the watched folder is needed to begin. Where the finished
        photos go is asked for later, when there are crops to look at."""
        if self.status == "running":
            return {"ok": False, "error": "already running"}
        if not self.suite:
            return {"ok": False, "error": "still loading the model"}
        if not watch or not Path(watch).is_dir():
            return {"ok": False, "error": "pick a folder to monitor"}
        try:
            w, work = self.suite.prepare_watch(watch)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        self.watch, self.out = str(w), ""
        self.suite.cfg["merged"]["stop_when_idle_sec"] = 0     # UI runs until stopped
        self.suite.ui_mode = True          # never prompt on stdin; wait for the page
        self.pipe = Pipeline(self.suite, w, None, work)
        self.started_at = time.time()
        self.status = "running"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return {"ok": True}

    def _run(self) -> None:
        try:
            self.pipe.run()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.status = "stopped"

    def stop(self) -> Dict[str, Any]:
        if self.pipe:
            self.pipe.stop_idle = 0.001          # makes the watcher fall through
            self.pipe.last_arrival = 0.0
        return {"ok": True}

    # ---------------- state for the UI ----------------
    def state(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "status": self.status, "error": self.error,
            "watch": self.watch, "out": self.out,
            "luts": [p.name for p in (self.suite._available_luts() if self.suite else [])],
            "look": (self.suite._lut_name if self.suite else None),
            "asked": bool(self.suite._lut_asked) if self.suite else False,
            "batch_size": int(self.suite.cfg["merged"]["premium_batch_size"]) if self.suite else 100,
            "workers": int(self.suite.cfg["performance"].get("workers", 0)) if self.suite else 0,
        }
        p = self.pipe
        if p:
            with p.lock:
                queued = len(p.pending)
                cropped = p.crops_done
            s.update({
                "cropped": cropped, "queued": queued,
                "delivered": p.delivered, "batches": p.batch_no - 1,
                "seen": len(p.seen),
                "elapsed": time.time() - self.started_at,
                "grading": bool(getattr(p, "grading_now", None)),
                "workers": int(getattr(p, "active_workers", 0)),
                "grading_on": bool(getattr(p, "grading_on").is_set()) if hasattr(p, "grading_on") else True,
                "want_workers": int(getattr(p, "desired_workers", 0)),
                "log": list(getattr(p, "ui_log", []))[-12:],
            })
        else:
            s.update({"cropped": 0, "queued": 0, "delivered": 0, "batches": 0,
                      "seen": 0, "elapsed": 0, "grading": False, "log": [],
                      "grading_on": True, "want_workers": 0})
        s["has_crops"] = bool(self._crop_dir() and any(self._crop_dir().glob("*.jpg")))
        # The one prompt: raised as soon as there are crops to judge a look on
        # and the destination is still unknown.
        s["needs_setup"] = bool(s["has_crops"] and self.status == "running"
                                and (not s["out"] or not s["look"]))
        s["prefix"] = str(self.suite.cfg["merged"].get("batch_folder_prefix", "")) if self.suite else ""
        s["job"] = dict(self.job)
        mp = self.manifest_path()
        s["manifest"] = str(mp) if mp else ""
        s["note_dir"] = self.note_dir
        s["note_name"] = self.note_name
        s["manifest_lines"] = self._manifest_lines
        return s

    def _crop_dir(self) -> Optional[Path]:
        if not self.suite or not self.suite.output_folder:
            return None
        d = self.suite.output_folder / self.suite._quality_folder_name("premium")
        return d if d.exists() else None

    # ---------------- previews ----------------
    def _load_samples(self, force: bool = False) -> bool:
        """Crops spread evenly across everything cut so far.

        Previously these were chosen on first use and cached for the whole run.
        Early on that meant four crops out of the first handful -- effectively
        consecutive frames of the same runner. Re-drawing whenever the pool has
        grown keeps the spread honest.
        """
        with self._sample_lock:
            d = self._crop_dir()
            if not d:
                return False
            files = sorted(d.glob("*.jpg"))
            if not files:
                return False
            grown = len(files) >= self._sampled_from * 2
            if self._samples and not force and not grown:
                return True
            idx = np.linspace(0, len(files) - 1, min(SAMPLES, len(files))).round().astype(int)
            picked, names = [], []
            for i in sorted(set(idx.tolist())):
                img = cv2.imread(str(files[i]))
                if img is None:
                    continue
                h, w = img.shape[:2]
                nw = max(1, round(w * THUMB_H / h))
                picked.append(cv2.resize(img, (nw, THUMB_H), interpolation=cv2.INTER_AREA))
                names.append(files[i].name)
            if not picked:
                return False
            self._samples, self._sample_names = picked, names
            self._sampled_from = len(files)
            self._tiles = {}                 # cached tiles were of the old crops
        self._warm_rail()
        return True

    def _warm_rail(self) -> None:
        """Render the rail thumbnails ahead of the picker opening, so the looks
        appear at once instead of trickling in as sixteen separate requests."""
        if self._warming:
            return
        self._warming = True

        def work():
            try:
                for name in [p.name for p in self.suite._available_luts()]:
                    self.tile(name, 0, "rail")
            except Exception:
                pass
            finally:
                self._warming = False

        threading.Thread(target=work, name="warm-rail", daemon=True).start()

    def reset_samples(self) -> None:
        with self._sample_lock:
            self._samples = []
            self._sample_names = []
            self._sampled_from = 0
            self._tiles = {}

    def tile(self, lut: str, i: int, size: str = "thumb") -> Optional[bytes]:
        if not self._load_samples() or i >= len(self._samples):
            return None
        key = f"{lut}|{i}|{size}"
        hit = self._tiles.get(key)
        if hit:
            return hit
        img = self._samples[i]
        if size == "big":
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, round(w * BIG_H / h)), BIG_H),
                             interpolation=cv2.INTER_LANCZOS4)
        elif size == "rail":
            # The rail chip renders at 22x33 css px; sending a 460px-tall JPEG
            # for it was most of the wait when the picker opened.
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, round(w * RAIL_H / h)), RAIL_H),
                             interpolation=cv2.INTER_AREA)
        if lut and lut != "__none__":
            try:
                parsed = self._parsed_cube(lut)
                # Trilinear straight onto the thumbnail. The .cube tables are
                # authored in RGB while cv2 hands us BGR, so convert around the
                # call -- feeding BGR in swaps red and blue and the result
                # looks stylised rather than obviously broken.
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                graded = lutlib.apply_lut(
                    rgb, parsed, float(self.suite.cfg["lut"].get("strength", 1.0)))
                img = cv2.cvtColor(graded, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        q = {"big": 82, "rail": 70}.get(size, 76)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None
        data = buf.tobytes()
        if len(self._tiles) > 160:
            self._tiles.clear()
        self._tiles[key] = data
        return data

    def _parsed_cube(self, name: str):
        cache = getattr(self, "_cubes", None)
        if cache is None:
            cache = self._cubes = {}
        if name not in cache:
            cache[name] = lutlib.parse_cube(
                Path(self.suite.cfg["lut"]["lut_folder"]) / name)
        return cache[name]

    def sample_caption(self, i: int) -> str:
        return self._sample_names[i] if i < len(self._sample_names) else ""

    def set_grading(self, on: bool) -> Dict[str, Any]:
        if not self.pipe:
            return {"ok": False, "error": "not running"}
        self.pipe.set_grading(bool(on))
        return {"ok": True, "grading_on": bool(on)}

    def resample(self) -> Dict[str, Any]:
        self.reset_samples()
        ok = self._load_samples(force=True)
        return {"ok": ok, "samples": list(self._sample_names)}

    # ---------------- delivery note ----------------
    def note_batch(self, batch_no: int, folder: str, images: int) -> None:
        """Append one line for a finished folder. Called once per folder, at
        the moment it closes, so the count is final when written -- the
        uploader can act on any line it sees."""
        if not self.out:
            return
        # The spot name carries the row number -- "test" becomes test1, test2,
        # ... -- and it matches the folder number, so a line identifies its
        # folder without needing a separate field.
        spot = f'{self.job["spot"]}{batch_no}' if self.job["spot"] else str(batch_no)
        line = "\t".join([spot, self.job["order"],
                           str(images), self.job["desc"]])
        path = self.manifest_path()
        if path is None:
            return
        with self._manifest_lock:
            try:
                fresh = (not path.exists()) or path.stat().st_size == 0
            except OSError:
                fresh = True
            try:
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    if fresh:
                        # Fixed leading row; the order number is the real one,
                        # the rest is literal.
                        f.write("\t".join([BOGUS_SPOT, self.job["order"],
                                            BOGUS_IMAGES, BOGUS_SPOT]) + "\n")
                    f.write(line + "\n")
                self._manifest_lines += 1      # counts real folders only
            except OSError as exc:
                print(f"[WARN] could not append to {path}: {exc}")
                return
        if self.pipe:
            self.pipe.note(f"{folder}: {images} images written to {MANIFEST}")

    def manifest_path(self) -> Optional[Path]:
        base = self.note_dir or self.out
        if not base:
            return None
        name = (self.note_name or MANIFEST).strip() or MANIFEST
        if not name.lower().endswith(".txt"):
            name += ".txt"
        return Path(base) / name

    def set_job(self, spot: str, order: str, desc: str) -> None:
        clean = {"spot": (spot or "").strip().replace("\t", " "),
                 "order": (order or "").strip().replace("\t", " "),
                 "desc": (desc or "").strip().replace("\t", " ")}
        self.job = clean

    def set_workers(self, n: Any) -> Dict[str, Any]:
        if not self.pipe:
            return {"ok": False, "error": "not running"}
        try:
            got = self.pipe.set_workers(int(n))
        except (TypeError, ValueError):
            return {"ok": False, "error": "not a number"}
        self.suite.cfg["performance"]["workers"] = got
        return {"ok": True, "workers": got}

    def set_delivery(self, lut: Optional[str], out: str, name: str = "",
                     spot: str = "", order: str = "", desc: str = "",
                     note_dir: str = "", note_name: str = "") -> Dict[str, Any]:
        """The answer to the one question the UI asks: which look, and where.
        Setting both together releases the batch that is waiting."""
        if not self.suite or not self.pipe:
            return {"ok": False, "error": "not running"}
        if not out:
            return {"ok": False, "error": "choose where to save the finished photos"}
        outp = Path(out)
        try:
            outp.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"cannot use that folder: {exc}"}
        if outp.resolve() == Path(self.watch).resolve():
            return {"ok": False, "error": "that is the folder being monitored"}
        # One name, two uses: the spot names the batch folders and opens each
        # row of the delivery note. Asking twice let them drift apart.
        clean = "".join(c for c in (spot or "").strip() if c not in '\/:*?"<>|')
        was = str(self.suite.cfg["merged"].get("batch_folder_prefix", ""))
        self.suite.cfg["merged"]["batch_folder_prefix"] = clean
        if clean != was and self.pipe is not None:
            nxt = self.pipe.batch_no
            label = f"{clean}{nxt}" if clean else str(nxt)
            self.pipe.note(f"batch folders now {label}, next ones follow")
        if self.pipe.out is None:
            self.pipe.out = outp          # first time: nothing is running yet
        elif Path(self.pipe.out) != outp:
            self.pipe.pending_out = outp  # applied at the next batch boundary
        self.out = str(outp)
        self.set_job(spot, order, desc)
        configured = str(self.suite.cfg["merged"].get("note_dir", "") or "").strip()
        nd = (note_dir or "").strip() or configured or DEFAULT_NOTE_DIR or self.out
        try:
            Path(nd).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"cannot use that note folder: {exc}"}
        self.note_dir = nd
        self.note_name = (note_name or "").strip() or MANIFEST
        self.pipe.on_batch_done = self.note_batch
        n = self.pipe.resume_from_disk()      # nothing already cut is stranded
        res = self.set_look(lut)
        if res.get("ok"):
            first = f"{clean}1" if clean else "1"
            self.pipe.note(f"delivering to {outp} as {first}, {first[:-1]}2, ...")
        return {"ok": bool(res.get("ok")), "look": res.get("look"),
                "out": self.out, "prefix": clean, "resumed": n}

    def set_look(self, name: Optional[str]) -> Dict[str, Any]:
        if not self.suite:
            return {"ok": False}
        if name is None or name == "__none__":
            self.suite._lut_name = None
            self.suite._lut_asked = True
            self.suite.look_event.set()          # release a waiting batch
            return {"ok": True, "look": None}
        ok = self.suite.change_lut(name)
        if ok and self.pipe is not None:
            self.pipe.note(f"look set to {name} — from batch #{self.pipe.batch_no}")
        return {"ok": ok, "look": self.suite._lut_name}


# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tether Room</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{--ground:#141414;--panel:#1b1b1b;--panel-2:#222;--panel-3:#282828;--line:#2b2b2b;
 --line-hi:#3a3a3a;--ink:#ededed;--muted:#8c8c8c;--dim:#666;--accent:#e0a63c;
 --accent-ink:#171208;--accent-dim:#7a5c22;--live:#5bc98a;--bad:#e0645c;
 --ui:'Archivo',system-ui,sans-serif;--mono:'IBM Plex Mono',Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.5 var(--ui);
 -webkit-font-smoothing:antialiased}
.kicker{font-family:var(--mono);font-size:10px;letter-spacing:.17em;text-transform:uppercase;
 color:var(--dim);margin:0}
button{font:inherit;font-size:13.5px;font-weight:600;cursor:pointer;border-radius:6px;
 border:1px solid var(--line-hi);padding:8px 14px;background:var(--panel-2);color:var(--ink)}
button:hover{border-color:var(--muted)}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:#f0be5e}
button.primary:hover{background:#eeb44a}
button.quiet{background:transparent;border-color:var(--line)}
button[disabled]{opacity:.4;cursor:not-allowed}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:12px 22px;
 background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:30}
.brand{font-weight:700;white-space:nowrap}.brand span{color:var(--accent)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex:none}
.dot.on{background:var(--live);animation:p 1.9s ease-in-out infinite}
.dot.err{background:var(--bad)}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
.state{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--muted)}
.spacer{flex:1}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 330px}
main{padding:22px;min-width:0}aside{border-left:1px solid var(--line);background:var(--panel);
 padding:20px 18px;display:flex;flex-direction:column;gap:20px;min-height:calc(100vh - 57px)}
.setup{display:grid;gap:14px;max-width:760px}
.fld{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px 16px}
.fld h3{margin:6px 0 0;font-size:15px}
.fld .row{display:flex;gap:10px;align-items:center;margin-top:10px}
.fld .val{flex:1;min-width:0;font-family:var(--mono);font-size:12.5px;background:var(--panel-2);
 border:1px solid var(--line);border-radius:5px;padding:8px 10px;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap;color:var(--ink)}
.fld .val.empty{color:var(--dim)}
.flow{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;margin-bottom:22px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px 17px}
.stage.active{border-color:var(--accent-dim)}
.stage h3{margin:7px 0 0;font-size:16px}.stage .sub{font-family:var(--mono);font-size:11px;
 color:var(--muted);margin-top:3px}
.nums{display:flex;gap:22px;margin-top:14px;flex-wrap:wrap}
.num .v{font-family:var(--mono);font-size:25px;font-weight:600;font-variant-numeric:tabular-nums}
.num .l{font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
 color:var(--dim);margin-top:3px}
.arrow{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;
 color:var(--dim);font-family:var(--mono);font-size:12px}.arrow .cnt{color:var(--accent);font-weight:600}
.meter{margin-top:14px}.meter .top{display:flex;justify-content:space-between;
 font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:6px}
.track{height:7px;background:var(--panel-3);border-radius:4px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:4px;transition:width .4s}
.meter.g .fill{background:var(--live)}
.card{background:var(--panel-2);border:1px solid var(--line);border-radius:9px;padding:15px 16px}
.card.look{padding:0;overflow:hidden}
.look-img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#000;max-height:250px}
.look-body{padding:14px 16px 16px}.look-body h3{margin:0;font-size:20px}
.look-body .f{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:3px}
.rate{display:flex;justify-content:space-between;font-family:var(--mono);font-size:12px;
 color:var(--muted);padding:5px 0;border-bottom:1px solid var(--line)}
.rate:last-child{border-bottom:0}.rate b{color:var(--ink);font-variant-numeric:tabular-nums}
.log{font-family:var(--mono);font-size:11px;color:var(--muted);display:flex;
 flex-direction:column-reverse;gap:5px;max-height:190px;overflow-y:auto}
.log div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scrim{position:fixed;inset:0;background:rgba(8,8,8,.82);z-index:50;display:none;padding:22px;
 overflow-y:auto}.scrim.on{display:block}
/* The folder browser opens FROM the picker, so it has to sit above it --
   equal z-index left it painting behind the sheet, invisible. */
#browse{z-index:60}
.sheet{max-width:1180px;margin:0 auto;background:var(--ground);border:1px solid var(--line-hi);
 border-radius:12px;overflow:hidden}
.sheet-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:15px 20px;
 background:var(--panel);border-bottom:1px solid var(--line)}
.sheet-head h2{margin:0;font-size:17px;flex:1;min-width:150px}
.keep{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11.5px;
 color:var(--live);background:rgba(91,201,138,.09);border:1px solid #2f5c44;border-radius:20px;
 padding:5px 12px}
.chip{display:inline-flex;align-items:center;gap:8px;background:var(--panel-2);
 border:1px solid var(--accent-dim);border-radius:20px;padding:4px 12px 4px 5px;
 font-family:var(--mono);font-size:11.5px;color:var(--accent)}
.chip .sw{width:16px;height:24px;border-radius:3px;background:#000 center/cover no-repeat;
 border:1px solid rgba(255,255,255,.12);flex:none}
.wctl{display:inline-flex;align-items:center;gap:8px}
.wctl b{font-family:var(--mono);font-size:14px;min-width:22px;text-align:center;
 font-variant-numeric:tabular-nums}
button.tiny{padding:2px 9px;font-size:15px;line-height:1.2;font-weight:600}
.dest input{font:inherit;font-family:var(--mono);font-size:12.5px;background:var(--ground);
 border:1px solid var(--line);border-radius:5px;padding:7px 10px;color:var(--ink);width:150px}
.dest input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.hintx{font-family:var(--mono);font-size:11px;color:var(--dim)}
.dest.job{border-bottom:1px solid var(--line)}
.dest.job input{flex:1;min-width:110px;width:auto}
.dest.job #jobdesc{flex:2;min-width:160px}
.dest{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 20px;
 background:var(--panel-2);border-bottom:1px solid var(--line)}
.dest .val{flex:1;min-width:180px;font-family:var(--mono);font-size:12.5px;
 background:var(--ground);border:1px solid var(--line);border-radius:5px;padding:7px 10px;
 overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dest .val.empty{color:var(--dim)}
.sheet-body{display:grid;grid-template-columns:210px minmax(0,1fr)}
.looks{border-right:1px solid var(--line);max-height:70vh;overflow-y:auto;padding:8px;
 background:var(--panel)}
.lb2{display:flex;align-items:center;gap:9px;width:100%;padding:6px 8px;margin:1px 0;
 background:none;border:1px solid transparent;border-radius:5px;text-align:left;font-size:13px;
 font-weight:400}
.lb2:hover{background:var(--panel-3)}
.lb2[aria-current=true]{background:var(--accent);color:var(--accent-ink);border-color:#f0be5e;
 font-weight:600}
.lb2 img{width:22px;height:33px;object-fit:cover;border-radius:2px;background:#000;flex:none}
.lb2 .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lb2 .cur{font-family:var(--mono);font-size:9px;text-transform:uppercase;color:var(--dim)}
.lb2[aria-current=true] .cur{color:rgba(23,18,8,.6)}
.pv{padding:18px 20px 22px}
.pv-head{display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap;margin-bottom:14px}
.pv-head h3{margin:0;font-size:23px}
.pv-actions{display:flex;gap:9px;margin-left:auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:13px}
figure{margin:0;display:flex;flex-direction:column;gap:6px}
.shot{position:relative;background:#000;border:1px solid var(--line);border-radius:4px;
 overflow:hidden;aspect-ratio:2/3;cursor:zoom-in}
.shot img{width:100%;height:100%;object-fit:cover;display:block}
.shot .z{position:absolute;right:6px;bottom:6px;font-family:var(--mono);font-size:9.5px;
 color:#fff;background:rgba(0,0,0,.6);padding:3px 7px;border-radius:3px;opacity:0;transition:.15s}
.shot:hover .z,.shot:focus-visible .z{opacity:1}
figcaption{font-family:var(--mono);font-size:10px;color:var(--dim);overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.lb{position:fixed;inset:0;background:#0b0b0b;z-index:70;display:none;flex-direction:column;
 align-items:center;justify-content:center;gap:14px;padding:20px}
.lb.on{display:flex}
.lb img{max-width:100%;max-height:calc(100vh - 110px);object-fit:contain;border-radius:4px}
.lb .meta{display:flex;gap:16px;align-items:center;font-family:var(--mono);font-size:12px;
 color:var(--muted);flex-wrap:wrap;justify-content:center}
.browser{max-width:640px;margin:0 auto;background:var(--ground);border:1px solid var(--line-hi);
 border-radius:12px;overflow:hidden}
.bhead{padding:15px 18px;background:var(--panel);border-bottom:1px solid var(--line)}
.bhead h2{margin:0 0 8px;font-size:16px}
.bpath{font-family:var(--mono);font-size:12px;color:var(--muted);word-break:break-all}
.blist{max-height:52vh;overflow-y:auto;padding:8px}
.bitem{display:flex;align-items:center;gap:9px;width:100%;padding:8px 10px;background:none;
 border:1px solid transparent;border-radius:5px;text-align:left;font-size:13.5px;font-weight:400}
.bitem:hover{background:var(--panel-2)}
.bfoot{display:flex;gap:10px;align-items:center;padding:13px 18px;background:var(--panel);
 border-top:1px solid var(--line);flex-wrap:wrap}
.bfoot .cnt{font-family:var(--mono);font-size:11.5px;color:var(--muted);flex:1}
.empty{color:var(--dim);font-family:var(--mono);font-size:12.5px;text-align:center;padding:20px 0}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
 color:var(--dim);text-align:left;font-weight:500;padding:0 10px 8px 0;border-bottom:1px solid var(--line)}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
td.mono{font-family:var(--mono);font-size:12.5px}
.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:var(--accent);
 color:var(--accent-ink);font-weight:600;padding:11px 18px;border-radius:7px;opacity:0;
 pointer-events:none;transition:.18s;z-index:90}
.toast.on{opacity:1}
@media(max-width:940px){.wrap{grid-template-columns:1fr}aside{border-left:0;
 border-top:1px solid var(--line);min-height:0}.flow{grid-template-columns:1fr}}
@media(max-width:560px){main{padding:16px}.sheet-body{grid-template-columns:1fr}
 .looks{max-height:140px;border-right:0;border-bottom:1px solid var(--line)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>

<div class="bar">
  <div class="brand">Tether<span>·</span>Room</div>
  <span class="dot" id="dot"></span><span class="state" id="statetext">loading</span>
  <span class="chip" id="lookchip" hidden><span class="sw" id="chipsw"></span>
    <b id="chipname"></b></span>
  <div class="spacer"></div>
  <span class="wctl" id="wctl" hidden>
    <span class="kicker">crop workers</span>
    <button class="tiny" id="wdown" aria-label="fewer workers">&minus;</button>
    <b id="wnow">24</b>
    <button class="tiny" id="wup" aria-label="more workers">+</button>
  </span>
  <button class="quiet" id="pausebtn" hidden>Pause grading</button>
  <button class="quiet" id="stopbtn" disabled>Stop</button>
  <button class="primary" id="lookbtn" disabled>Change look</button>
</div>

<div class="wrap">
 <main>
  <section class="setup" id="setup">
    <div class="fld">
      <p class="kicker">To begin</p><h3>Which folder should I watch?</h3>
      <p style="margin:6px 0 0;color:var(--muted);font-size:13.5px">
        Photos are cropped as they land. Once the first crops appear I&rsquo;ll ask
        which look to apply and where to save the finished photos.</p>
      <div class="row"><span class="val empty" id="watchval">no folder chosen</span>
        <button id="pickwatch">Browse&hellip;</button></div>
    </div>
    <div><button class="primary" id="startbtn" disabled>Start watching</button></div>
  </section>

  <div id="live" style="display:none">
    <div class="flow">
      <section class="stage active">
        <p class="kicker">Stage 1 · continuous</p><h3>Crop</h3>
        <div class="sub" id="w1">—</div>
        <div class="nums">
          <div class="num"><div class="v" id="n-seen">0</div><div class="l">arrived</div></div>
          <div class="num"><div class="v" id="n-crop">0</div><div class="l">cropped</div></div>
          <div class="num"><div class="v" id="n-q">0</div><div class="l">queued</div></div>
        </div>
        <div class="meter"><div class="top"><span>next batch</span><span id="qlab">0 / 100</span></div>
          <div class="track"><div class="fill" id="qfill" style="width:0"></div></div></div>
      </section>
      <div class="arrow"><span>▸</span><span class="cnt" id="bs">100</span><span>▸</span></div>
      <section class="stage" id="gstage">
        <p class="kicker">Stage 2 · per batch</p><h3>Grade</h3>
        <div class="sub" id="gsub">idle</div>
        <div class="nums">
          <div class="num"><div class="v" id="n-b">0</div><div class="l">batches</div></div>
          <div class="num"><div class="v" id="n-d">0</div><div class="l">delivered</div></div>
        </div>
      </section>
    </div>
    <p class="kicker" style="margin-bottom:10px">Folders</p>
    <table><tbody>
      <tr><td style="width:110px">Monitoring</td><td class="mono" id="t-watch">—</td></tr>
      <tr><td>Retouched</td><td class="mono" id="t-out">—</td></tr>
    </tbody></table>
  </div>
 </main>

 <aside>
  <div><p class="kicker" style="margin-bottom:9px">Applied look</p>
    <div class="card look"><img class="look-img" id="side-img" alt="" style="display:none">
      <div class="look-body"><h3 id="side-name">None yet</h3>
        <div class="f" id="side-file">pick one when the first batch lands</div></div></div></div>
  <div><p class="kicker" style="margin-bottom:9px">Throughput</p>
    <div class="card">
      <div class="rate"><span>elapsed</span><b id="r-el">—</b></div>
      <div class="rate"><span>crop rate</span><b id="r-cr">—</b></div>
      <div class="rate"><span>premium/photo</span><b id="r-y">—</b></div>
      <div class="rate"><span>queue</span><b id="r-dr">—</b></div>
      <div class="rate"><span>delivery note</span><b id="r-mf">not set</b></div>
    </div></div>
  <div><p class="kicker" style="margin-bottom:9px">Activity</p><div class="log" id="log"></div></div>
 </aside>
</div>

<div class="scrim" id="browse"><div class="browser">
  <div class="bhead"><h2 id="btitle">Choose a folder</h2><div class="bpath" id="bpath">—</div></div>
  <div class="blist" id="blist"></div>
  <div class="bfoot"><span class="cnt" id="bcnt"></span>
    <button class="quiet" id="bcancel">Cancel</button>
    <button class="primary" id="buse">Use this folder</button></div>
</div></div>

<div class="scrim" id="picker"><div class="sheet">
  <div class="sheet-head"><h2 id="ptitle">Crops are coming in</h2>
    <span class="keep"><span class="dot on"></span><span id="keep">Cropping continues</span></span>
    <button class="quiet" id="pcancel">Later</button></div>
  <div class="dest" id="destrow">
    <span class="kicker" style="flex:none">Save to</span>
    <span class="val empty" id="destval">no folder chosen</span>
    <button id="pickdest">Browse&hellip;</button>
  </div>
  <div class="dest" id="noterow">
    <span class="kicker" style="flex:none">Delivery note</span>
    <span class="val empty" id="noteval">choose a folder</span>
    <button id="picknote">Browse&hellip;</button>
    <input id="notename" type="text" placeholder="delivery.txt" autocomplete="off"
           spellcheck="false" aria-label="Delivery note filename">
  </div>
  <div class="dest job" id="jobrow">
    <span class="kicker" style="flex:none">Spot</span>
    <input id="jobspot" type="text" placeholder="e.g. finish" autocomplete="off"
           spellcheck="false" aria-label="Spot name, also used for the folder names">
    <span class="hintx" id="namehint">names the folders</span>
    <span class="kicker" style="flex:none">Order no.</span>
    <input id="joborder" type="text" placeholder="e.g. 19887" autocomplete="off"
           aria-label="Order number">
    <span class="kicker" style="flex:none">Description</span>
    <input id="jobdesc" type="text" placeholder="e.g. 10k men, wet conditions"
           autocomplete="off" aria-label="Description">
  </div>
  <div class="sheet-body">
    <div class="looks" id="looks"></div>
    <div class="pv">
      <div class="pv-head"><div><h3 id="pvname">—</h3></div>
        <div class="pv-actions"><button class="primary" id="apply">Use this look &amp; start delivering</button></div></div>
      <div class="tiles" id="tiles"></div>
      <p class="empty" id="nocrops" style="display:none">
        No crops yet — previews appear once Stage 1 has cut some.</p>
    </div>
  </div>
</div></div>

<div class="lb" id="lb"><img id="lbimg" alt="">
  <div class="meta"><span id="lbcap"></span><button class="quiet" id="lbclose">Close</button></div></div>
<div class="toast" id="toast"></div>

<script>
const $=id=>document.getElementById(id);
let ST={}, LOOKS=[], sel=null, target=null, bpath='', openedAt=0, DEST='', NOTEDIR='', asked=false, GRADING=true;
const j=(u,o)=>fetch(u,o).then(r=>r.json());
function toast(m){const t=$('toast');t.textContent=m;t.classList.add('on');
  clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove('on'),3200);}

/* ---- folder browser ---- */
function openBrowse(which){target=which;
  $('btitle').textContent = which==='watch' ? 'Folder to monitor'
    : which==='note' ? 'Where should the delivery note be written?'
    : 'Where should the finished photos go?';
  $('browse').classList.add('on');
  nav(which==='watch' ? (ST.watch||'') : which==='note' ? (NOTEDIR||DEST||ST.out||'')
                                                        : (DEST||ST.out||''));}
function nav(p){ j('/api/browse?path='+encodeURIComponent(p)).then(d=>{
  bpath=d.path; $('bpath').textContent=d.path||'This PC';
  $('bcnt').textContent = d.path ? (d.images+' JPEG'+(d.images===1?'':'s')+' here') : '';
  const L=$('blist'); L.innerHTML='';
  if(d.parent!==null&&d.path){const b=document.createElement('button');b.className='bitem';
    b.textContent='↰  ..';b.onclick=()=>nav(d.parent);L.appendChild(b);}
  if(!d.dirs.length){const e=document.createElement('div');e.className='empty';
    e.textContent='no subfolders';L.appendChild(e);}
  d.dirs.forEach(n=>{const b=document.createElement('button');b.className='bitem';
    b.textContent='▸  '+n;
    b.onclick=()=>nav(d.path?(d.path.replace(/[\\/]+$/,'')+'\\'+n):n);L.appendChild(b);});
});}
function spotHint(){
  const v=($('jobspot').value||'').trim().replace(/[^A-Za-z0-9 _.-]/g,'');
  $('namehint').textContent = v
    ? ('folders and rows: ' + v + '1, ' + v + '2, ' + v + '3\u2026')
    : 'names the folders';}
$('jobspot').addEventListener('input', spotHint);
$('wdown').onclick=()=>bumpWorkers(-4);
$('wup').onclick=()=>bumpWorkers(4);
function bumpWorkers(d){ const n=Math.max(1,Math.min(64,(ST.want_workers||ST.workers||24)+d));
  $('wnow').textContent=n;
  j('/api/workers',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({n:n})}).then(r=>{ if(r.ok) toast('Crop workers → '+r.workers);
    else toast(r.error||'could not change workers');});}
$('pickwatch').onclick=()=>openBrowse('watch');
$('pickdest').onclick=()=>openBrowse('dest');
$('picknote').onclick=()=>openBrowse('note');
$('bcancel').onclick=()=>$('browse').classList.remove('on');
$('buse').onclick=()=>{ if(!bpath){toast('Pick a folder first');return;}
  if(target==='watch'){ST.watch=bpath;$('watchval').textContent=bpath;
    $('watchval').classList.remove('empty'); $('startbtn').disabled=false;}
  else if(target==='note'){NOTEDIR=bpath;$('noteval').textContent=bpath;
    $('noteval').classList.remove('empty');}
  else {DEST=bpath;$('destval').textContent=bpath;$('destval').classList.remove('empty');}
  $('browse').classList.remove('on');};

/* ---- start / stop ---- */
$('startbtn').onclick=()=>{ $('startbtn').disabled=true;
  j('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({watch:ST.watch})}).then(r=>{
      if(!r.ok){toast(r.error||'could not start');$('startbtn').disabled=false;}
      else toast('Watching '+ST.watch);});};
$('stopbtn').onclick=()=>{j('/api/stop',{method:'POST'}).then(()=>toast('Stopping after the queue drains'));};
$('pausebtn').onclick=()=>{ const turnOn = !GRADING;
  j('/api/grading',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({on:turnOn})}).then(r=>{
      if(r.ok) toast(turnOn?'Grading resumed':'Grading paused — cropping continues');
      else toast(r.error||'could not change grading');});};

/* ---- look picker ---- */
function openPicker(){ openedAt=ST.cropped||0; $('picker').classList.add('on');
  buildLooks(); showLook(sel||LOOKS[0]);
  // Re-draw the samples across everything cropped so far, then repaint. Keeps
  // the four crops representative of the whole batch rather than of whatever
  // existed when the picker was first used.
  j('/api/resample',{method:'POST'}).then(r=>{ if(r&&r.ok){
    const bust='&v='+Date.now();
    [...$('looks').children].forEach((b,i)=>{
      b.querySelector('img').src='/api/tile?size=rail&lut='+
        encodeURIComponent(LOOKS[i])+'&i=0'+bust;});
    showLook(sel, bust);
  }}).catch(()=>{});}
$('lookbtn').onclick=openPicker;
$('pcancel').onclick=()=>$('picker').classList.remove('on');

function buildLooks(){ const L=$('looks'); if(L.dataset.n==String(LOOKS.length))return;
  L.dataset.n=String(LOOKS.length); L.innerHTML='';
  LOOKS.forEach(n=>{const b=document.createElement('button');b.className='lb2';
    b.innerHTML='<img alt="" loading="lazy" src="/api/tile?size=rail&lut='+
      encodeURIComponent(n)+'&i=0"><span class="nm"></span><span class="cur"></span>';
    b.querySelector('.nm').textContent=n.replace(/\.cube$/,'');
    b.onclick=()=>showLook(n); L.appendChild(b);});}

function showLook(n, bust){ sel=n; bust=bust||'';
  [...$('looks').children].forEach((b,i)=>{b.setAttribute('aria-current',LOOKS[i]===n);
    b.querySelector('.cur').textContent=(LOOKS[i]===ST.look)?'in use':'';});
  $('pvname').textContent=n.replace(/\.cube$/,'');
  const T=$('tiles'); T.innerHTML='';
  if(!ST.has_crops){$('nocrops').style.display='block';return;}
  $('nocrops').style.display='none';
  for(let i=0;i<4;i++){const f=document.createElement('figure');
    f.innerHTML='<div class="shot" tabindex="0" role="button"><img alt="Crop '+(i+1)+'" src="/api/tile?lut='+
      encodeURIComponent(n)+'&i='+i+bust+'"><span class="z">Enlarge</span></div><figcaption></figcaption>';
    const im=f.querySelector('img');
    im.onerror=()=>{f.remove();};
    f.querySelector('.shot').onclick=()=>{
      $('lbimg').src='/api/tile?lut='+encodeURIComponent(n)+'&i='+i+'&big=1';
      $('lbcap').textContent=n.replace(/\.cube$/,'');
      $('lb').classList.add('on');};
    T.appendChild(f);}}

$('apply').onclick=()=>{
  const dest = DEST || ST.out;
  if(!dest){toast('Choose where to save the finished photos first');return;}
  const btn=$('apply'), was=btn.textContent;
  btn.disabled=true; btn.textContent='Applying…';        // no double submits
  j('/api/deliver',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lut:sel,out:dest,
      spot:($('jobspot').value||''),order:($('joborder').value||''),
      desc:($('jobdesc').value||''),note_dir:NOTEDIR,
      note_name:($('notename').value||'')})}).then(r=>{
      btn.disabled=false; btn.textContent=was;
      if(!r.ok){toast(r.error||'could not start delivering');return;}
      const nm=sel.replace(/\.cube$/,'');
      const first=(r.prefix?r.prefix+'_1':'1');
      toast(nm + ' applied — delivering to ' + r.out + ' / ' + first
            + (r.resumed ? '  (' + r.resumed + ' crops resumed)' : ''));
      markInUse(); $('picker').classList.remove('on');});};

function markInUse(){ [...$('looks').children].forEach((b,i)=>{
  b.querySelector('.cur').textContent=(LOOKS[i]===sel)?'in use':'';});}
$('lbclose').onclick=()=>$('lb').classList.remove('on');
$('lb').onclick=e=>{if(e.target===$('lb'))$('lb').classList.remove('on');};
addEventListener('keydown',e=>{if(e.key!=='Escape')return;
  if($('lb').classList.contains('on'))$('lb').classList.remove('on');
  else if($('picker').classList.contains('on'))$('picker').classList.remove('on');
  else if($('browse').classList.contains('on'))$('browse').classList.remove('on');});

/* ---- poll ---- */
const fmt=n=>Number(n||0).toLocaleString('en-US');
function tick(){ j('/api/state').then(s=>{ ST=Object.assign({},s,{watch:ST.watch||s.watch,out:ST.out||s.out});
  LOOKS=s.luts||[];
  const running=s.status==='running';
  $('dot').className='dot'+(running?' on':(s.status==='error'?' err':''));
  $('statetext').textContent = s.status==='loading'?'loading model':s.status;
  $('stopbtn').disabled=!running; $('lookbtn').disabled=!(s.luts&&s.luts.length);
  if(running||s.status==='stopped'){$('setup').style.display='none';$('live').style.display='block';}
  $('n-seen').textContent=fmt(s.seen); $('n-crop').textContent=fmt(s.cropped);
  $('n-q').textContent=fmt(s.queued); $('n-b').textContent=fmt(s.batches);
  $('n-d').textContent=fmt(s.delivered);
  $('bs').textContent=s.batch_size;
  const q=Math.min(s.batch_size,s.queued);
  $('qlab').textContent=q+' / '+s.batch_size;
  $('qfill').style.width=(q/s.batch_size*100)+'%';
  $('w1').textContent=(s.workers||0)+' crop workers';
  $('gsub').textContent = !GRADING ? 'paused by you — queue still filling'
                        : (s.grading ? 'grading a batch' : 'idle');
  $('gstage').classList.toggle('active',!!s.grading);
  $('t-watch').textContent=s.watch||'—'; $('t-out').textContent=s.out||'—';
  const el=s.elapsed||0;
  $('r-el').textContent=el?Math.floor(el/60)+'m '+String(Math.floor(el%60)).padStart(2,'0')+'s':'—';
  $('r-cr').textContent=el&&s.cropped?(s.cropped/el*60).toFixed(0)+' photos/min':'—';
  $('r-y').textContent=s.cropped?((s.delivered+s.queued)/s.cropped).toFixed(2):'—';
  $('r-dr').textContent=s.queued>s.batch_size*2?'falling behind':'keeping up';
  const mf=$('r-mf');
  if(mf) mf.textContent = s.manifest
    ? s.manifest_lines+' folder'+(s.manifest_lines===1?'':'s')+' logged'
    : 'not set';
  if(mf && s.manifest) mf.title = s.manifest;
  if(document.activeElement !== $('notename') && !$('notename').value && s.note_name
     && s.note_name !== 'delivery.txt') $('notename').value = s.note_name;
  if(!NOTEDIR && s.note_dir){ NOTEDIR = s.note_dir;
    $('noteval').textContent = s.note_dir; $('noteval').classList.remove('empty'); }
  if(s.look){$('side-name').textContent=s.look.replace(/\.cube$/,'');
    $('side-file').textContent=s.look;
    const im=$('side-img'); if(s.has_crops){im.style.display='block';
      if(!im.dataset.l||im.dataset.l!==s.look){im.dataset.l=s.look;
        im.src='/api/tile?lut='+encodeURIComponent(s.look)+'&i=1&t='+Date.now();}}}
  const L=$('log'); L.innerHTML=''; (s.log||[]).forEach(m=>{const d=document.createElement('div');
    d.textContent=m; L.appendChild(d);});
  if($('picker').classList.contains('on')){
    const since=Math.max(0,(s.cropped||0)-openedAt);
    $('keep').textContent='Cropping continues — '+since+(since===1?' photo':' photos')+' since you opened this';}
  // always-visible confirmation of the applied look
  const chip=$('lookchip');
  if(s.look){ chip.hidden=false; $('chipname').textContent=s.look.replace(/\.cube$/,'');
    if(s.has_crops){const u="url('/api/tile?size=rail&lut="+encodeURIComponent(s.look)+"&i=0')";
      if($('chipsw').style.backgroundImage!==u) $('chipsw').style.backgroundImage=u;}
  } else chip.hidden=true;
  GRADING = s.grading_on !== false;
  const pb=$('pausebtn'); pb.hidden=!running;
  pb.textContent = GRADING ? 'Pause grading' : 'Resume grading';
  pb.style.borderColor = GRADING ? '' : 'var(--accent)';
  pb.style.color = GRADING ? '' : 'var(--accent)';
  const wc=$('wctl'); wc.hidden=!running;
  if(running){ const shown=s.want_workers||s.workers;
    if($('wnow').textContent!=String(shown)) $('wnow').textContent=shown;
    $('wnow').style.color = (s.workers===s.want_workers)?'':'var(--accent)';}
  $('ptitle').textContent = s.out ? 'Change the look' : 'Crops are coming in';
  $('apply').textContent = s.out ? 'Apply from next batch'
                                 : 'Use this look & start delivering';
  // Destination and folder name stay editable for the whole run: a name set
  // only in the first seconds could never be corrected afterwards.
  const shown = DEST || s.out || '';
  $('destval').textContent = shown || 'no folder chosen';
  $('destval').classList.toggle('empty', !shown);
  ['spot','order','desc'].forEach(k=>{
    const el=$('job'+k);
    if(document.activeElement!==el && s.job && s.job[k] && !el.value) el.value=s.job[k];});
  if(document.activeElement !== $('jobspot')) spotHint();
  // Ask once, as soon as there are crops to judge a look on.
  if(s.needs_setup && !asked && !$('picker').classList.contains('on')){
    asked=true; openPicker();}
  if(s.out) asked=true;
}).catch(()=>{}).finally(()=>setTimeout(tick,1000));}
tick();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore

    def log_message(self, *a):
        pass

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj: Any, code: int = 200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def _body(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif u.path == "/api/state":
            self._json(self.app.state())
        elif u.path == "/api/browse":
            self._json(self.app.browse(unquote(q.get("path", [""])[0])))
        elif u.path == "/api/tile":
            lut = q.get("lut", ["__none__"])[0]
            i = int(q.get("i", ["0"])[0])
            size = q.get("size", ["thumb"])[0]
            if q.get("big", ["0"])[0] == "1":
                size = "big"
            data = self.app.tile(lut, i, size)
            if data is None:
                self._send(404, "text/plain", b"no preview")
            else:
                self._send(200, "image/jpeg", data)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/start":
            b = self._body()
            self._json(self.app.start(b.get("watch", "")))
        elif u.path == "/api/stop":
            self._json(self.app.stop())
        elif u.path == "/api/look":
            self._json(self.app.set_look(self._body().get("lut")))
        elif u.path == "/api/deliver":
            b = self._body()
            self._json(self.app.set_delivery(
                b.get("lut"), b.get("out", ""), b.get("name", ""),
                b.get("spot", ""), b.get("order", ""), b.get("desc", ""),
                b.get("note_dir", ""), b.get("note_name", "")))
        elif u.path == "/api/workers":
            self._json(self.app.set_workers(self._body().get("n", 0)))
        elif u.path == "/api/grading":
            self._json(self.app.set_grading(bool(self._body().get("on", True))))
        elif u.path == "/api/resample":
            self._json(self.app.resample())
        else:
            self._send(404, "text/plain", b"not found")


def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "settings_merged.example.json"
    app = App(config)
    handler = type("Bound", (Handler,), {"app": app})
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"\n  Tether Room — {url}\n  (Ctrl+C to quit)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
