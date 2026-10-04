# -*- coding: utf-8 -*-
"""
Grade Room UI -- a local page for applying LUTs to folders of JPEGs.

Pick the input folder, pick where the graded copies go, pick a look (previewed
on your own photos, with a strength slider), press Start. Nothing is cropped and
nothing is uploaded; it works with no internet.

Serves on 127.0.0.1 only, and rejects requests that did not come from its own
page (a web page in another tab must not be able to drive it).
"""

from __future__ import annotations

import json
import os
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

import grade_suite as gs
import lut as lutlib

DEFAULTS: Dict[str, Any] = {
    "lut_folder": "luts",
    "port": 8771,
    "quality": 95,
    "target_height": 4000,
    "sharpen": "strong",
    "interpolation": "lanczos",     # lanczos | cubic (cubic is ~9x faster when upscaling)
    "warn_below": 4000,             # warn about photos whose longer side is under this (0 = off)
    "workers": 0,                   # parallel photos: 0 or "auto" = one per CPU thread (limited by free memory); 1-64 = fixed. Changeable live in the UI
}
THUMB_H = 460
RAIL_H = 66
BIG_H = 1400


def load_config(path: str) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[WARN] could not read {p}: {exc}; using defaults")
    return cfg


class App:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.lut_folder = Path(str(cfg["lut_folder"]))
        self.job: Optional[gs.Job] = None
        self.thread: Optional[threading.Thread] = None
        self.samples: List[Path] = []
        self.sample_dir = ""
        self._thumbs: Dict[int, np.ndarray] = {}
        self._tiles: Dict[str, bytes] = {}
        self._bigbase: Dict[Any, np.ndarray] = {}     # decoded + resized sample, per (index, height)
        self._big_lock = threading.Lock()             # one large render at a time
        self._cubes: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.warn_below = self._cfg_warn()
        # pre-flight scan of the chosen input folder (sizes read from the JPEG headers)
        self._pf_id = 0
        self._pf_stop = threading.Event()
        self._pf_entries: List[Any] = []
        self._pf_done = False
        self._pf_progress = 0
        self._pf_cache: Optional[Any] = None

    def _cfg_warn(self) -> int:
        try:
            n = int(self.cfg.get("warn_below", gs.DEFAULT_WARN_BELOW))
        except (TypeError, ValueError):
            n = gs.DEFAULT_WARN_BELOW
        return max(0, min(20000, n))

    # ---------------- folders ----------------
    @staticmethod
    def browse(path: str) -> Dict[str, Any]:
        if not path:
            drives = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
            if not drives:                                    # not Windows
                drives = ["/"]
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
                elif c.suffix.lower() in gs.JPEG_EXTS:
                    images += 1
        except OSError as exc:
            return {"path": path, "parent": str(p.parent), "dirs": [], "images": 0,
                    "error": str(exc)}
        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "dirs": dirs, "images": images}

    # ---------------- previews ----------------
    def prepare(self, input_dir: str) -> Dict[str, Any]:
        p = Path(input_dir)
        if not input_dir or not p.is_dir():
            return {"ok": False, "error": "that is not a folder"}
        files = gs.list_sample_files(p, n=4)
        with self._lock:
            self.samples, self.sample_dir = files, str(p)
            self._thumbs, self._tiles, self._bigbase = {}, {}, {}
        self._start_preflight(p)
        return {"ok": True, "samples": [f.name for f in files]}

    # ---------------- pre-flight: how big are the photos? ----------------
    def _start_preflight(self, folder: Path) -> None:
        self._pf_stop.set()                      # abandon a scan of the previous folder
        stop = threading.Event()
        with self._lock:
            self._pf_id += 1
            my_id = self._pf_id
            self._pf_stop = stop
            self._pf_entries, self._pf_done, self._pf_progress, self._pf_cache = [], False, 0, None

        def work():
            def prog(n: int) -> None:
                with self._lock:
                    if self._pf_id == my_id:
                        self._pf_progress = n
            try:
                entries = gs.scan_sizes(folder, stop, prog)
            except Exception:
                entries = []
            with self._lock:
                if self._pf_id == my_id and not stop.is_set():
                    self._pf_entries, self._pf_done = entries, True
                    self._pf_cache = None
        threading.Thread(target=work, name="preflight", daemon=True).start()

    def preflight(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.sample_dir:
                return None
            if not self._pf_done:
                return {"done": False, "scanned": self._pf_progress}
            key = (self._pf_id, self.warn_below)
            if self._pf_cache is None or self._pf_cache[0] != key:
                self._pf_cache = (key, gs.summarize_sizes(self._pf_entries, self.warn_below))
            out = dict(self._pf_cache[1])
        out["done"] = True
        return out

    def set_warn(self, b: Dict[str, Any]) -> Dict[str, Any]:
        try:
            n = int(b.get("warn_below"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "the size limit must be a number"}
        if not 0 <= n <= 20000:
            return {"ok": False, "error": "the size limit must be between 0 and 20000"}
        self.warn_below = n
        return {"ok": True, "warn_below": n}

    def _thumb(self, i: int) -> Optional[np.ndarray]:
        with self._lock:
            if i in self._thumbs:
                return self._thumbs[i]
            if i >= len(self.samples):
                return None
            path = self.samples[i]
        try:
            img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            return None
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * THUMB_H / h)), THUMB_H), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._thumbs[i] = img
        return img

    def _cube(self, name: str):
        with self._lock:
            hit = self._cubes.get(name)
        if hit is None:
            hit = lutlib.parse_cube(self.lut_folder / name)
            with self._lock:
                self._cubes[name] = hit
        return hit

    def _big_base(self, i: int, h: int) -> Optional[np.ndarray]:
        """The sample decoded and resized to h px tall, for the enlarged view.
        Kept, so changing look or strength does not decode the photo again."""
        key = (i, h)
        with self._lock:
            hit = self._bigbase.get(key)
            path = self.samples[i] if 0 <= i < len(self.samples) else None
        if hit is not None:
            return hit
        if path is None:
            return None
        data = path.read_bytes()
        flag = cv2.IMREAD_COLOR
        wh = gs.jpeg_size(path)
        if wh:
            # Let the JPEG decoder skip detail we would throw away anyway. Judged on the
            # shorter side so it is right whichever way the EXIF orientation turns the photo.
            for r, f in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                         (2, cv2.IMREAD_REDUCED_COLOR_2)):
                if min(wh) / r >= h:
                    flag = f
                    break
        img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        if img is None:
            return None
        ih, iw = img.shape[:2]
        if ih > h:
            img = cv2.resize(img, (max(1, round(iw * h / ih)), h), interpolation=cv2.INTER_AREA)
        with self._lock:
            if len(self._bigbase) >= 6:
                self._bigbase.clear()
            self._bigbase[key] = img
        return img

    def tile(self, lut: str, i: int, size: str, strength: float, orig: bool,
             h: int = 0) -> Optional[bytes]:
        """JPEG bytes for a preview. size 'big' is the enlarged view, h px tall
        (the window's own height); errors there propagate so the page can say why."""
        h = max(300, min(3000, int(h or BIG_H))) if size == "big" else 0
        key = f"{lut}|{i}|{size}|{h}|{strength:.2f}|{int(orig)}"
        with self._lock:
            hit = self._tiles.get(key)
        if hit:
            return hit
        if size == "big":
            with self._big_lock:
                with self._lock:
                    hit = self._tiles.get(key)
                if hit:
                    return hit
                img = self._big_base(i, h)
                if img is None:
                    return None
                if not orig and lut:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.cvtColor(lutlib.apply_lut(rgb, self._cube(lut), strength),
                                       cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                data = buf.tobytes()
                with self._lock:
                    if len(self._tiles) > 200:
                        self._tiles.clear()
                    self._tiles[key] = data
                return data
        img = self._thumb(i)
        if img is None:
            return None
        if size == "rail":
            hh, ww = img.shape[:2]
            img = cv2.resize(img, (max(1, round(ww * RAIL_H / hh)), RAIL_H), interpolation=cv2.INTER_AREA)
        if not orig and lut:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                graded = lutlib.apply_lut(rgb, self._cube(lut), strength)
                img = cv2.cvtColor(graded, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        q = {"rail": 70}.get(size, 78)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None
        data = buf.tobytes()
        with self._lock:
            if len(self._tiles) > 200:
                self._tiles.clear()
            self._tiles[key] = data
        return data

    # ---------------- run control ----------------
    def start(self, b: Dict[str, Any]) -> Dict[str, Any]:
        if self.job and self.job.status in ("starting", "building", "running", "watching"):
            return {"ok": False, "error": "a run is already in progress"}
        inp, out = str(b.get("input") or ""), str(b.get("output") or "")
        if not inp or not out:
            return {"ok": False, "error": "choose both the input and the output folder"}
        luts = {p.name for p in lutlib.list_luts(self.lut_folder)}
        lut = str(b.get("lut") or "")
        if lut not in luts:
            return {"ok": False, "error": "choose a look"}
        problem = gs.check_folders(Path(inp), Path(out))
        if problem:
            return {"ok": False, "error": problem[0].upper() + problem[1:]}
        try:
            strength = min(1.0, max(0.0, float(b.get("strength", 100)) / 100.0))
            height = int(b.get("target_height") or self.cfg["target_height"])
            quality = int(b.get("quality") or self.cfg["quality"])
            workers = int(b.get("workers") or 0)
            warn_below = int(b.get("warn_below", self.warn_below))
        except (TypeError, ValueError):
            return {"ok": False, "error": "a number field is not a number"}
        sharpen = str(b.get("sharpen") or self.cfg["sharpen"]).lower()
        if sharpen not in gs.SHARPEN_AMOUNT:
            sharpen = "strong"
        if not 500 <= height <= 20000:
            return {"ok": False, "error": "target height must be between 500 and 20000"}
        if not 50 <= quality <= 100:
            return {"ok": False, "error": "JPEG quality must be between 50 and 100"}
        if workers <= 0:
            workers = self._cfg_workers()                # 0 = automatic, settled by the run itself
        workers = min(gs.MAX_WORKERS, workers)
        if not 0 <= warn_below <= 20000:
            return {"ok": False, "error": "the size limit must be between 0 and 20000"}
        opts = gs.Options(
            input_dir=Path(inp), output_dir=Path(out), lut=lut, lut_folder=self.lut_folder,
            strength=strength, finish=bool(b.get("finish")), target_height=height,
            sharpen=sharpen, interpolation=str(self.cfg.get("interpolation", "lanczos")),
            quality=quality, overwrite=bool(b.get("overwrite")), regrade=bool(b.get("regrade")),
            watch=bool(b.get("watch")), workers=workers, warn_below=warn_below)
        self.job = gs.Job(opts)
        self.thread = threading.Thread(target=self.job.run, name="grade-job", daemon=True)
        self.thread.start()
        return {"ok": True}

    def _cfg_workers(self) -> int:
        """The configured worker count; 0 means automatic."""
        v = self.cfg.get("workers", 0)
        if isinstance(v, str) and v.strip().lower() == "auto":
            return 0
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 0
        return min(gs.MAX_WORKERS, n) if n > 0 else 0

    def set_workers(self, b: Dict[str, Any]) -> Dict[str, Any]:
        if not self.job or self.job.status not in ("starting", "building", "running", "watching"):
            return {"ok": False, "error": "nothing is running"}
        try:
            n = int(b.get("workers"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "workers must be a number"}
        if not 1 <= n <= gs.MAX_WORKERS:
            return {"ok": False, "error": f"workers must be between 1 and {gs.MAX_WORKERS}"}
        return {"ok": True, "workers": self.job.set_workers(n)}

    def stop(self) -> Dict[str, Any]:
        if self.job:
            self.job.stop.set()
        return {"ok": True}

    def reset(self) -> Dict[str, Any]:
        if self.job and self.job.status in ("starting", "building", "running", "watching"):
            return {"ok": False, "error": "stop the current run first"}
        self.job = None
        return {"ok": True}

    def state(self) -> Dict[str, Any]:
        return {
            "luts": [p.name for p in lutlib.list_luts(self.lut_folder)],
            "lut_folder": str(self.lut_folder),
            "samples": len(self.samples),
            "defaults": {
                "quality": int(self.cfg["quality"]),
                "target_height": int(self.cfg["target_height"]),
                "sharpen": str(self.cfg["sharpen"]),
                "workers": self._cfg_workers() or gs.auto_workers(),
                "workers_auto": self._cfg_workers() == 0,
                "cpu_threads": gs.cpu_threads(),
                "max_workers": gs.MAX_WORKERS,
                "warn_below": self.warn_below,
                "exif": gs.piexif is not None,
            },
            "preflight": self.preflight(),
            "job": self.job.snapshot() if self.job else None,
        }


# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grade Room</title>
<style>
:root{--ground:#141414;--panel:#1b1b1b;--panel-2:#222;--panel-3:#282828;--line:#2b2b2b;
 --line-hi:#3a3a3a;--ink:#ededed;--muted:#8c8c8c;--dim:#666;--accent:#e0a63c;
 --accent-ink:#171208;--accent-dim:#7a5c22;--live:#5bc98a;--bad:#e0645c;
 --ui:'Segoe UI',system-ui,-apple-system,sans-serif;--mono:Consolas,'SF Mono',monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.5 var(--ui);
 -webkit-font-smoothing:antialiased}
.kicker{font-family:var(--mono);font-size:10px;letter-spacing:.17em;text-transform:uppercase;color:var(--dim);margin:0}
button{font:inherit;font-size:13.5px;font-weight:600;cursor:pointer;border-radius:6px;
 border:1px solid var(--line-hi);padding:8px 14px;background:var(--panel-2);color:var(--ink)}
button:hover{border-color:var(--muted)}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:#f0be5e}
button.primary:hover{background:#eeb44a}
button.quiet{background:transparent;border-color:var(--line)}
button[disabled]{opacity:.4;cursor:not-allowed}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:12px 22px;background:var(--panel);
 border-bottom:1px solid var(--line);position:sticky;top:0;z-index:30}
.brand{font-weight:700}.brand span{color:var(--accent)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex:none}
.dot.on{background:var(--live);animation:p 1.9s ease-in-out infinite}.dot.err{background:var(--bad)}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
.state{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.spacer{flex:1}
main{max-width:1180px;margin:0 auto;padding:22px;display:grid;gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h3{margin:6px 0 0;font-size:16px}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
.row label.k{width:150px;font-size:13px;color:var(--muted);flex:none}
.val{flex:1;min-width:200px;font-family:var(--mono);font-size:12.5px;background:var(--panel-2);border:1px solid var(--line);
 border-radius:5px;padding:8px 10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.val.empty{color:var(--dim)}
.hint{font-family:var(--mono);font-size:11px;color:var(--dim);margin:8px 0 0}
.look{display:grid;grid-template-columns:230px minmax(0,1fr);margin-top:12px;border:1px solid var(--line);
 border-radius:8px;overflow:hidden;background:var(--ground)}
.looks{border-right:1px solid var(--line);max-height:430px;overflow-y:auto;padding:8px;background:var(--panel)}
.lb2{display:flex;align-items:center;gap:9px;width:100%;padding:6px 8px;margin:1px 0;background:none;border:1px solid transparent;
 border-radius:5px;text-align:left;font-size:13px;font-weight:400}
.lb2:hover{background:var(--panel-3)}
.lb2[aria-current=true]{background:var(--accent);color:var(--accent-ink);border-color:#f0be5e;font-weight:600}
.lb2 img{width:22px;height:33px;object-fit:cover;border-radius:2px;background:#000;flex:none}
.lb2 .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pv{padding:16px 18px 20px}
.pv-head{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px}
.pv-head h3{margin:0;font-size:21px;flex:1;min-width:140px}
.strength{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:12px;color:var(--muted)}
.strength input[type=range]{width:200px;accent-color:var(--accent)}
.strength b{color:var(--ink);min-width:42px;text-align:right;font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:13px}
figure{margin:0;display:flex;flex-direction:column;gap:6px}
.shot{position:relative;background:#000;border:1px solid var(--line);border-radius:4px;overflow:hidden;height:300px;
 cursor:pointer;user-select:none;-webkit-user-select:none;touch-action:none}
.shot img{width:100%;height:100%;object-fit:contain;display:block;pointer-events:none}
.shot .tag{position:absolute;left:6px;top:6px;font-family:var(--mono);font-size:9.5px;color:#fff;background:rgba(0,0,0,.65);
 padding:3px 7px;border-radius:3px;display:none}
.shot.orig .tag{display:block}
.shot .z{position:absolute;right:6px;bottom:6px;z-index:2;font-family:var(--mono);font-size:11.5px;color:#fff;
 background:rgba(0,0,0,.72);padding:6px 12px;border-radius:4px;border:1px solid rgba(255,255,255,.28);font-weight:500;cursor:zoom-in}
.shot .z:hover{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
figcaption{font-family:var(--mono);font-size:10px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.empty{color:var(--dim);font-family:var(--mono);font-size:12.5px;text-align:center;padding:30px 10px}
.opts{display:grid;gap:10px;margin-top:12px}
.opt{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:14px}
.opt input[type=checkbox]{width:17px;height:17px;accent-color:var(--accent)}
.opt small{color:var(--dim);font-family:var(--mono);font-size:11px}
input[type=number],select{font:inherit;font-family:var(--mono);font-size:12.5px;background:var(--ground);border:1px solid var(--line);
 border-radius:5px;padding:6px 9px;color:var(--ink);width:88px}
select{width:auto}
.go{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.go .sum{font-family:var(--mono);font-size:12px;color:var(--muted);flex:1;min-width:220px}
.big{font-family:var(--mono);font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}
.nums{display:flex;gap:30px;flex-wrap:wrap;margin-top:14px}
.num .l{font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--dim);margin-top:3px}
.track{height:8px;background:var(--panel-3);border-radius:4px;overflow:hidden;margin-top:18px}
.fill{height:100%;background:var(--accent);border-radius:4px;transition:width .4s}
.fill.done{background:var(--live)}.fill.bad{background:var(--bad)}
.meta{display:flex;gap:24px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:10px}
.meta b{color:var(--ink);font-weight:500}
.smalllist{max-height:190px;overflow:auto;margin-top:8px;font-family:var(--mono);font-size:11.5px;
 color:var(--ink);line-height:1.6}
.warn button{font-size:11.5px;padding:3px 9px;margin-left:8px}
.warn+.hint,.hint+.warn{margin-top:10px}
.wrow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px;font-size:13px;color:var(--muted)}
.wrow input{width:72px;background:var(--panel-2);color:var(--ink);border:1px solid var(--line-hi);border-radius:5px;padding:6px 8px;font:inherit}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:8px}
td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line)}td.mono{font-family:var(--mono);font-size:12.5px;word-break:break-all}
.log{font-family:var(--mono);font-size:11.5px;color:var(--muted);display:flex;flex-direction:column-reverse;gap:5px;
 max-height:230px;overflow-y:auto;margin-top:10px}
.log div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.warn{background:rgba(224,166,60,.09);border:1px solid var(--accent-dim);border-radius:8px;padding:10px 14px;
 font-size:13px;color:var(--accent);margin-top:12px}
.scrim{position:fixed;inset:0;background:rgba(8,8,8,.82);z-index:50;display:none;padding:22px;overflow-y:auto}.scrim.on{display:block}
.browser{max-width:640px;margin:0 auto;background:var(--ground);border:1px solid var(--line-hi);border-radius:12px;overflow:hidden}
.bhead{padding:15px 18px;background:var(--panel);border-bottom:1px solid var(--line)}
.bhead h2{margin:0 0 8px;font-size:16px}.bpath{font-family:var(--mono);font-size:12px;color:var(--muted);word-break:break-all}
.blist{max-height:52vh;overflow-y:auto;padding:8px}
.bitem{display:flex;align-items:center;gap:9px;width:100%;padding:8px 10px;background:none;border:1px solid transparent;
 border-radius:5px;text-align:left;font-size:13.5px;font-weight:400}.bitem:hover{background:var(--panel-2)}
.bfoot{display:flex;gap:10px;align-items:center;padding:13px 18px;background:var(--panel);border-top:1px solid var(--line);flex-wrap:wrap}
.bfoot .cnt{font-family:var(--mono);font-size:11.5px;color:var(--muted);flex:1}
.lb{position:fixed;inset:0;background:#0b0b0b;z-index:70;display:none;flex-direction:column;align-items:center;
 justify-content:center;gap:14px;padding:20px}.lb.on{display:flex}
.lbstage{position:relative;display:flex;justify-content:center}
.lb img{width:calc(100vw - 40px);max-width:2600px;height:calc(100vh - 110px);object-fit:contain;border-radius:4px;
 user-select:none;-webkit-user-select:none;touch-action:none}
.lbbusy{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,.78);color:#fff;
 font-family:var(--mono);font-size:12.5px;padding:9px 16px;border-radius:6px;pointer-events:none;max-width:80%;text-align:center}
.lbbusy.err{background:#5a1f1a;border:1px solid var(--bad)}
.lbtag{position:absolute;left:12px;top:12px;font-family:var(--mono);font-size:11px;color:#fff;background:rgba(0,0,0,.7);
 padding:4px 9px;border-radius:3px;display:none;pointer-events:none}
.lbx,.lbnav{position:fixed;background:rgba(30,30,30,.85);color:#fff;border:1px solid var(--line-hi);border-radius:50%;
 width:44px;height:44px;padding:0;font-size:26px;line-height:1;display:flex;align-items:center;justify-content:center;z-index:71}
.lbx{top:14px;right:18px}.lbnav{top:50%;margin-top:-22px}.lbnav.prev{left:14px}.lbnav.next{right:14px}
.lb .m{display:flex;gap:16px;align-items:center;font-family:var(--mono);font-size:12px;color:var(--muted)}
.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:var(--accent);color:var(--accent-ink);
 font-weight:600;padding:11px 18px;border-radius:7px;opacity:0;pointer-events:none;transition:.18s;z-index:90;max-width:90vw}
.toast.on{opacity:1}
@media(max-width:760px){.look{grid-template-columns:1fr}.looks{max-height:150px;border-right:0;border-bottom:1px solid var(--line)}
 .row label.k{width:100%}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="bar">
  <div class="brand">Grade<span>·</span>Room</div>
  <span class="dot" id="dot"></span><span class="state" id="statetext">ready</span>
  <div class="spacer"></div>
  <button class="quiet" id="stopbtn" hidden>Stop</button>
  <button class="primary" id="newbtn" hidden>New run</button>
</div>
<main>
 <section id="setup" style="display:grid;gap:16px">
  <div class="card">
    <p class="kicker">1 · Folders</p><h3>What to grade, and where the copies go</h3>
    <div class="row"><label class="k">Photos to grade</label><span class="val empty" id="inval">no folder chosen</span>
      <button id="pickin">Browse&hellip;</button></div>
    <div class="row"><label class="k">Save graded copies to</label><span class="val empty" id="outval">no folder chosen</span>
      <button id="pickout">Browse&hellip;</button></div>
    <p class="hint">Sub-folders and file names are kept exactly as they are. The originals are never changed.</p>
    <p class="hint" id="pfinfo" style="display:none"></p>
    <div class="warn" id="pfwarn" style="display:none"><span id="pftext"></span>
      <button class="quiet" id="pfbtn" type="button">Show which</button><div class="smalllist" id="pflist" hidden></div></div>
  </div>
  <div class="card">
    <p class="kicker">2 · Look</p><h3>Pick a look, judged on your own photos</h3>
    <div class="look">
      <div class="looks" id="looks"></div>
      <div class="pv">
        <div class="pv-head"><h3 id="pvname">&mdash;</h3>
          <label class="strength">strength
            <input type="range" id="strength" min="0" max="100" value="100"><b id="strengthv">100%</b></label></div>
        <div class="tiles" id="tiles"></div>
        <p class="empty" id="nosamples">Choose the photos folder above and previews of each look appear here.</p>
        <p class="hint" id="holdhint" style="display:none">Press and hold a photo to see the original. Double-click it, or press Enlarge, to see it larger.</p>
      </div>
    </div>
  </div>
  <div class="card">
    <p class="kicker">3 · Options</p><h3>How to write the files</h3>
    <div class="opts">
      <div class="opt"><input type="checkbox" id="finish"><label for="finish">Also finish to</label>
        <input type="number" id="height" min="500" max="20000" step="100"> <span>px tall, sharpen</span>
        <select id="sharpen"><option>none</option><option>light</option><option>medium</option><option selected>strong</option></select>
        <small>off = native size, colour only &middot; aspect ratio is always kept, nothing is cropped</small></div>
      <div class="opt"><label for="quality">JPEG quality</label><input type="number" id="quality" min="50" max="100">
        <label for="workers" style="margin-left:14px">Parallel workers</label><input type="number" id="workers" min="1" max="64">
        <small id="wauto">automatic: one per CPU thread, fewer if the photos are so large that memory would run short</small></div>
      <div class="opt"><label for="warnbelow">Warn about photos smaller than</label>
        <input type="number" id="warnbelow" min="0" max="20000" step="100"> <span>px on the long side</span>
        <small>0 = no warning; the photos are graded either way</small></div>
      <div class="opt"><input type="checkbox" id="watch"><label for="watch">Keep watching for new photos after this batch</label></div>
      <div class="opt"><input type="checkbox" id="overwrite"><label for="overwrite">Overwrite files that already exist in the output</label>
        <small>off = they are skipped, so a restart carries on where it stopped</small></div>
      <div class="opt"><input type="checkbox" id="regrade"><label for="regrade">Grade files that Grade Room has already graded</label>
        <small>off = skipped, to avoid grading twice</small></div>
    </div>
    <div class="warn" id="noexif" style="display:none">piexif is not installed, so camera EXIF data will not be copied to the graded files and they cannot be recognised as already graded.</div>
  </div>
  <div class="go"><button class="primary" id="startbtn" disabled>Start grading</button><span class="sum" id="sum"></span></div>
 </section>

 <section id="live" style="display:none;gap:16px">
  <div class="card">
    <p class="kicker" id="lk">Progress</p><div class="big" id="head">&mdash;</div>
    <div class="nums">
      <div class="num"><div class="big" id="n-done">0</div><div class="l">graded</div></div>
      <div class="num"><div class="big" id="n-skip">0</div><div class="l">skipped</div></div>
      <div class="num"><div class="big" id="n-fail">0</div><div class="l">failed</div></div>
      <div class="num"><div class="big" id="n-found">0</div><div class="l">found</div></div>
    </div>
    <div class="track"><div class="fill" id="fill" style="width:0"></div></div>
    <div class="meta"><span>rate <b id="m-rate">&mdash;</b></span><span>elapsed <b id="m-el">&mdash;</b></span>
      <span>remaining <b id="m-eta">&mdash;</b></span>
      <span>cpu <b id="m-cpu">&mdash;</b></span><span id="m-cur" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:420px"></span></div>
    <div class="hint" id="timing" style="display:none;line-height:1.7"><div id="timing1"></div><div id="timing2"></div></div>
    <div class="wrow" id="wrow"><label for="wlive">Parallel workers</label>
      <input type="number" id="wlive" min="1" max="64"><button class="quiet" id="wapply">Apply</button>
      <span class="hint" style="margin:0" id="whint">takes effect immediately; watch the rate and cpu above</span></div>
    <div class="warn" id="smallwarn" style="display:none"><span id="smalltext"></span>
      <button class="quiet" id="smallbtn" type="button">Show which</button><div class="smalllist" id="smalllist" hidden></div></div>
    <div class="warn" id="srgb" style="display:none"></div>
  </div>
  <div class="card">
    <table><tbody>
      <tr><td style="width:150px">Look</td><td class="mono" id="t-look">&mdash;</td></tr>
      <tr><td>From</td><td class="mono" id="t-in">&mdash;</td></tr>
      <tr><td>To</td><td class="mono" id="t-out">&mdash;</td></tr>
      <tr><td>Report</td><td class="mono" id="t-rep">&mdash;</td></tr>
    </tbody></table>
    <p class="kicker" style="margin-top:16px">Activity</p><div class="log" id="log"></div>
  </div>
 </section>
</main>

<div class="scrim" id="browse"><div class="browser">
  <div class="bhead"><h2 id="btitle">Choose a folder</h2><div class="bpath" id="bpath">&mdash;</div></div>
  <div class="blist" id="blist"></div>
  <div class="bfoot"><span class="cnt" id="bcnt"></span>
    <button class="quiet" id="bcancel">Cancel</button><button class="primary" id="buse">Use this folder</button></div>
</div></div>
<div class="lb" id="lb" role="dialog" aria-modal="true" aria-label="Enlarged preview">
  <button class="lbx" id="lbx" type="button" aria-label="Close">&times;</button>
  <button class="lbnav prev" id="lbprev" type="button" aria-label="Previous photo">&lsaquo;</button>
  <button class="lbnav next" id="lbnext" type="button" aria-label="Next photo">&rsaquo;</button>
  <div class="lbstage"><img id="lbimg" alt="" draggable="false"><span class="lbtag" id="lbtag">ORIGINAL</span>
    <div class="lbbusy" id="lbbusy" style="display:none"></div></div>
  <div class="m"><span id="lbcap"></span><span>hold to see the original &middot; &larr; &rarr; change photo &middot; Esc closes</span>
    <button class="quiet" id="lbclose">Close</button></div></div>
<div class="toast" id="toast"></div>

<script>
const $=id=>document.getElementById(id);
const j=(u,o)=>fetch(u,o).then(r=>r.json());
const post=(u,b)=>j(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
let ST={}, LOOKS=[], sel=null, IN='', OUT='', outTouched=false, target='', bpath='', LOADED=false, sv=100, sampled=0, tsv=0;
function toast(m){const t=$('toast');t.textContent=m;t.classList.add('on');clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove('on'),3800);}
const fmt=n=>Number(n||0).toLocaleString('en-US');
const dur=s=>{s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return (h?h+'h ':'')+(h||m?m+'m ':'')+String(s%60).padStart(2,'0')+'s';};
const lastName=n=>n.replace(/\.cube$/i,'');

/* ---- folder browser ---- */
function openBrowse(which){target=which;
  $('btitle').textContent = which==='in' ? 'Folder of photos to grade' : 'Where should the graded copies go?';
  $('browse').classList.add('on'); nav(which==='in' ? IN : (OUT||IN));}
function nav(p){ j('/api/browse?path='+encodeURIComponent(p)).then(d=>{
  bpath=d.path; $('bpath').textContent=d.path||'This PC';
  $('bcnt').textContent = d.path ? (d.images+' JPEG'+(d.images===1?'':'s')+' directly in this folder') : '';
  const L=$('blist'); L.innerHTML='';
  if(d.parent!==null&&d.path){const b=document.createElement('button');b.className='bitem';
    b.textContent='↰  ..';b.onclick=()=>nav(d.parent);L.appendChild(b);}
  if(!d.dirs.length){const e=document.createElement('div');e.className='empty';e.textContent='no subfolders';L.appendChild(e);}
  d.dirs.forEach(n=>{const b=document.createElement('button');b.className='bitem';b.textContent='▸  '+n;
    const sep=(d.path&&d.path.indexOf('/')>=0&&d.path.indexOf('\\')<0)?'/':'\\';
    b.onclick=()=>nav(d.path?(d.path.replace(/[\\/]+$/,'')+sep+n):n);L.appendChild(b);});
});}
$('pickin').onclick=()=>openBrowse('in');
$('pickout').onclick=()=>openBrowse('out');
$('bcancel').onclick=()=>$('browse').classList.remove('on');
$('buse').onclick=()=>{ if(!bpath){toast('Pick a folder first');return;}
  if(target==='in') setInput(bpath); else {OUT=bpath;outTouched=true;paint();}
  $('browse').classList.remove('on');};
function setInput(p){ IN=p;
  if(!outTouched){ const sep=(p.indexOf('/')>=0&&p.indexOf('\\')<0)?'/':'\\'; OUT=p.replace(/[\\/]+$/,'')+'_graded'; }
  paint();
  j('/api/prepare?input='+encodeURIComponent(p)).then(r=>{
    if(!r.ok){toast(r.error||'could not read that folder');return;}
    sampled=r.samples.length; LOADED=true; showLook(sel); paint();
    if(!sampled) toast('No JPEGs found in that folder (looking in sub-folders too)'); });
}

/* ---- look picker ---- */
function buildLooks(){ const L=$('looks'); if(L.dataset.n==String(LOOKS.length)+':'+(LOADED?1:0))return;
  L.dataset.n=String(LOOKS.length)+':'+(LOADED?1:0); L.innerHTML='';
  if(!LOOKS.length){ const e=document.createElement('div'); e.className='empty';
    e.textContent='No .cube files found in "'+(ST.lut_folder||'luts')+'". Put your looks there, or point lut_folder in settings_grade.json at them.';
    L.appendChild(e); return; }
  LOOKS.forEach(n=>{const b=document.createElement('button');b.className='lb2';
    b.innerHTML=(LOADED&&sampled?'<img alt="" loading="lazy" src="/api/tile?size=rail&i=0&s='+sv+'&lut='+encodeURIComponent(n)+'">':'<img alt="">')+'<span class="nm"></span>';
    b.querySelector('.nm').textContent=lastName(n); b.onclick=()=>showLook(n); L.appendChild(b);});
  markSel();}
function markSel(){ [...$('looks').children].forEach((b,i)=>b.setAttribute('aria-current',LOOKS[i]===sel)); }
function showLook(n){ if(!n) return; sel=n; markSel(); $('pvname').textContent=lastName(n);
  const T=$('tiles'); T.innerHTML='';
  const ok=LOADED&&sampled>0; $('nosamples').style.display=ok?'none':'block'; $('holdhint').style.display=ok?'block':'none';
  if(!ok){paint();return;}
  const v=Date.now(); THUMB=[];
  for(let i=0;i<sampled;i++){const f=document.createElement('figure');
    const g='/api/tile?i='+i+'&s='+sv+'&lut='+encodeURIComponent(n)+'&v='+v, o='/api/tile?orig=1&i='+i+'&lut='+encodeURIComponent(n);
    f.innerHTML='<div class="shot" tabindex="0"><img alt="Sample '+(i+1)+'" src="'+g+'"><span class="tag">ORIGINAL</span><button class="z" type="button" title="Enlarge (or double-click the photo)">Enlarge</button></div><figcaption></figcaption>';
    const shot=f.querySelector('.shot'), im=f.querySelector('img');
    im.onerror=()=>f.remove();
    const down=e=>{ if(e.target.classList.contains('z'))return; shot.classList.add('orig'); im.src=o; };
    const up=()=>{ if(!shot.classList.contains('orig'))return; shot.classList.remove('orig'); im.src='/api/tile?i='+i+'&s='+sv+'&lut='+encodeURIComponent(sel); };
    shot.addEventListener('pointerdown',down); shot.addEventListener('pointerup',up);
    shot.addEventListener('pointerleave',up); shot.addEventListener('pointercancel',up);
    THUMB[i]=g;
    const zb=f.querySelector('.z');
    zb.onclick=e=>{e.stopPropagation();lbOpen(i);};
    zb.addEventListener('mouseenter',()=>lbWarm(i));            // start rendering the large one while the cursor is on the button
    shot.addEventListener('dblclick',e=>{ if(!e.target.classList.contains('z')) lbOpen(i); });
    shot.addEventListener('keydown',e=>{ if((e.key==='Enter'||e.key===' ')&&e.target===shot){ e.preventDefault(); lbOpen(i); } });
    T.appendChild(f);}
  paint();}
$('strength').addEventListener('input',e=>{ sv=+e.target.value; $('strengthv').textContent=sv+'%';
  clearTimeout(tsv); tsv=setTimeout(()=>{ if(sel) showLook(sel); },220); paint(); });
let THUMB=[], lbIdx=0, lbTok=0, lbOpenedAt=0, lbShowingOrig=false; const lbUrl={g:'',o:''}, lbWarmed={};
const lbEl=$('lb'), lbImg=$('lbimg');
function bigUrl(i,orig){ const h=Math.max(400,Math.min(2000,Math.round(innerHeight*(devicePixelRatio||1))));
  return '/api/tile?size=big&h='+h+'&i='+i+(orig?'&orig=1':'&s='+sv)+'&lut='+encodeURIComponent(sel); }
function lbWarm(i){ const u=bigUrl(i,false); if(lbWarmed[u]) return; lbWarmed[u]=1; fetch(u).catch(()=>{}); }
function lbRelease(){ ['g','o'].forEach(k=>{ if(lbUrl[k]) URL.revokeObjectURL(lbUrl[k]); lbUrl[k]=''; }); }
function lbBusy(msg,err){ const b=$('lbbusy'); if(!msg){ b.style.display='none'; return; }
  b.textContent=msg; b.className='lbbusy'+(err?' err':''); b.style.display='block'; }
function lbFetch(url,tok){
  return fetch(url).then(r=>{
    if(!r.ok) return r.text().then(t=>{ throw new Error('HTTP '+r.status+(t?': '+t.slice(0,160):'')); });
    return r.blob(); })
  .then(b=>(tok!==lbTok)?null:URL.createObjectURL(b)); }
function lbOpen(i){
  if(!sampled) return; i=((i%sampled)+sampled)%sampled;
  lbIdx=i; const tok=++lbTok; lbRelease(); lbShowingOrig=false; $('lbtag').style.display='none';
  lbImg.src=THUMB[i]||'';                              // show what we already have at once, sharpen when the big one arrives
  $('lbcap').textContent=lastName(sel)+' · '+sv+'% · photo '+(i+1)+' of '+sampled;
  lbEl.classList.add('on'); lbOpenedAt=Date.now(); lbBusy('Loading the full-size preview…');
  lbFetch(bigUrl(i,false),tok).then(u=>{
    if(!u) return;
    lbUrl.g=u; lbImg.src=u; lbBusy('');
    return lbFetch(bigUrl(i,true),tok).then(o=>{ if(o) lbUrl.o=o; }).catch(()=>{});   // the original, ready for hold-to-compare
  }).catch(err=>{ if(tok===lbTok) lbBusy('Could not load the full-size preview ('+err.message+'). The small one is shown instead.',true); });
}
function lbClose(){ lbEl.classList.remove('on'); lbTok++; lbRelease(); lbImg.removeAttribute('src'); lbBusy(''); lbShowingOrig=false; }
lbImg.addEventListener('pointerdown',e=>{ e.preventDefault();
  if(lbUrl.o){ lbShowingOrig=true; lbImg.src=lbUrl.o; $('lbtag').style.display='block'; }
  else if(lbUrl.g) toast('The original is still loading'); });
const lbUp=()=>{ if(!lbShowingOrig) return; lbShowingOrig=false; $('lbtag').style.display='none'; lbImg.src=lbUrl.g||THUMB[lbIdx]||''; };
['pointerup','pointerleave','pointercancel'].forEach(ev=>lbImg.addEventListener(ev,lbUp));
$('lbclose').onclick=lbClose; $('lbx').onclick=lbClose;
$('lbprev').onclick=()=>lbOpen(lbIdx-1); $('lbnext').onclick=()=>lbOpen(lbIdx+1);
// a second click of a double-click lands on the backdrop right after it opens; ignore clicks that early
lbEl.onclick=e=>{ if(e.target===lbEl && Date.now()-lbOpenedAt>500) lbClose(); };
addEventListener('keydown',e=>{
  if(lbEl.classList.contains('on')){
    if(e.key==='Escape') lbClose();
    else if(e.key==='ArrowLeft') lbOpen(lbIdx-1);
    else if(e.key==='ArrowRight') lbOpen(lbIdx+1);
    return; }
  if(e.key==='Escape' && $('browse').classList.contains('on')) $('browse').classList.remove('on'); });

/* ---- setup summary ---- */
['finish','watch','overwrite','regrade'].forEach(id=>$(id).addEventListener('change',paint));
['height','quality','workers','sharpen'].forEach(id=>$(id).addEventListener('input',paint));
function paint(){
  const iv=$('inval'); iv.textContent=IN||'no folder chosen'; iv.classList.toggle('empty',!IN);
  const ov=$('outval'); ov.textContent=OUT||'no folder chosen'; ov.classList.toggle('empty',!OUT);
  const ready=IN&&OUT&&sel;
  $('startbtn').disabled=!ready;
  $('sum').textContent = ready ? ('Grade '+(sampled?'':'the ')+'photos in '+IN+' with '+lastName(sel)+' at '+sv+'%'
     + ($('finish').checked?', finished to '+$('height').value+'px':', native size')+' → '+OUT) : 'Choose the folders and a look to begin.';
}
$('startbtn').onclick=()=>{ $('startbtn').disabled=true;
  post('/api/start',{input:IN,output:OUT,lut:sel,strength:sv,finish:$('finish').checked,target_height:+$('height').value,
    sharpen:$('sharpen').value,quality:+$('quality').value,workers:($('workers').dataset.touched||!WAUTO)?+$('workers').value:0,
    overwrite:$('overwrite').checked,regrade:$('regrade').checked,watch:$('watch').checked,warn_below:+$('warnbelow').value}).then(r=>{
    if(!r.ok){toast(r.error||'could not start');paint();} });};
$('wapply').onclick=()=>post('/api/workers',{workers:+$('wlive').value}).then(r=>{
  if(!r.ok) toast(r.error||'could not change workers'); else { toast('Workers set to '+r.workers); $('wlive').dataset.dirty=''; } });
$('wlive').addEventListener('input',()=>{$('wlive').dataset.dirty='1';});
$('wlive').addEventListener('keydown',e=>{ if(e.key==='Enter') $('wapply').click(); });
$('stopbtn').onclick=()=>post('/api/stop').then(()=>toast('Stopping after the photos in progress'));
$('newbtn').onclick=()=>post('/api/reset').then(r=>{ if(!r.ok) toast(r.error||'cannot reset now'); });

/* ---- poll ---- */
const HEAD={starting:'Starting…',building:'Building the lookup table…',running:'Grading…',watching:'Watching for new photos',
  done:'Finished',stopped:'Stopped',error:'Stopped with an error'};
let WAUTO=false;
$('workers').addEventListener('input',()=>{ $('workers').dataset.touched='1'; $('wauto').style.display='none'; });
function renderTiming(jb){
  const el=$('timing'), t=jb&&jb.timing; if(!t){ el.style.display='none'; return; }
  el.style.display='block';
  $('timing1').textContent='Per photo, on average (wall time): '+t.stages.map(r=>r.stage+' '+Math.round(r.wall)+' ms').join(' · ')+'  —  '+Math.round(t.wall)+' ms in all, '+Math.round(t.cpu)+' ms of it computing';
  const w=Math.round(t.waiting*100);
  $('timing2').textContent = w>=40
    ? 'About '+w+'% of that time is spent waiting, not computing: the drive, a network share or antivirus is the limit (or there are more workers than CPU threads). More workers will not help.'
    : 'About '+(100-w)+'% of that time is spent computing: the CPU is the limit and the drive keeps up.';
}
function fillList(el,files,total,extra){ const key=JSON.stringify(files)+total; if(el.dataset.k===key) return; el.dataset.k=key;
  el.innerHTML=''; files.forEach(f=>{const d=document.createElement('div'); d.textContent=f.file+'  —  '+f.w+' × '+f.h; el.appendChild(d);});
  if(total>files.length){const d=document.createElement('div'); d.style.color='var(--muted)';
    d.textContent='… and '+fmt(total-files.length)+' more'+(extra?' ('+extra+')':''); el.appendChild(d);} }
function toggleList(btn,list){ btn.onclick=()=>{ list.hidden=!list.hidden; btn.textContent=list.hidden?'Show which':'Hide list'; }; }
toggleList($('pfbtn'),$('pflist')); toggleList($('smallbtn'),$('smalllist'));
function renderPF(pf){
  const info=$('pfinfo'), warn=$('pfwarn'); info.style.display='none'; warn.style.display='none';
  if(!IN||!pf) return;
  if(!pf.done){ info.style.display='block'; info.textContent='Checking photo sizes… '+fmt(pf.scanned)+' looked at'; return; }
  const unread=pf.unreadable?' '+fmt(pf.unreadable)+' file'+(pf.unreadable===1?' has':'s have')+' no readable size.':'';
  if(pf.small>0){ warn.style.display='block';
    $('pftext').textContent=fmt(pf.small)+' of '+fmt(pf.total)+' photo'+(pf.total===1?' is':'s are')+' smaller than '+fmt(pf.warn_below)
      +' px on the long side (smallest '+fmt(pf.smallest)+' px).'+unread+' They will still be graded.';
    fillList($('pflist'),pf.files,pf.small,'the report lists them all after a run'); }
  else if(pf.total>0 && pf.warn_below>0){ info.style.display='block';
    info.textContent='All '+fmt(pf.total)+' photos are '+fmt(pf.warn_below)+' px or larger on the long side.'+unread; }
}
function renderSmall(jb){
  const w=$('smallwarn'); if(!jb||!(jb.small>0)){ w.style.display='none'; return; }
  w.style.display='block';
  $('smalltext').textContent=fmt(jb.small)+' photo'+(jb.small===1?' is':'s are')+' smaller than '+fmt(jb.warn_below)
    +' px on the long side (smallest '+fmt(jb.small_min)+' px) — '
    +(jb.finish?'finished to '+fmt(jb.target_height)+' px tall.':'graded at their own size.');
  fillList($('smalllist'),jb.small_files,jb.small,'see the report'); }
let warnT=0;
$('warnbelow').addEventListener('input',()=>{ clearTimeout(warnT); warnT=setTimeout(()=>{
  const v=$('warnbelow').value; if(v==='') return; post('/api/warn',{warn_below:+v}).then(r=>{ if(!r.ok) toast(r.error||'invalid size'); }); },350); });
function tick(){ j('/api/state').then(s=>{ ST=s;
  if(!LOADED&&s.luts) {LOOKS=s.luts;} else if(s.luts) LOOKS=s.luts;
  if(!sel&&LOOKS.length){ sel=LOOKS[0]; }
  if(!$('height').dataset.init){ $('height').value=s.defaults.target_height; $('quality').value=s.defaults.quality;
    $('workers').value=s.defaults.workers; WAUTO=!!s.defaults.workers_auto; $('wauto').style.display=WAUTO?'':'none'; $('warnbelow').value=s.defaults.warn_below; $('sharpen').value=s.defaults.sharpen; $('height').dataset.init=1;
    $('noexif').style.display=s.defaults.exif?'none':'block'; }
  buildLooks(); if(!$('tiles').children.length&&sel) showLook(sel);
  const jb=s.job, active=jb&&['starting','building','running','watching'].includes(jb.status);
  renderPF(s.preflight); renderSmall(jb);
  document.title=(((jb&&jb.small>0)||(!jb&&s.preflight&&s.preflight.done&&s.preflight.small>0))?'(!) ':'')+'Grade Room';
  $('setup').style.display=jb?'none':'grid'; $('live').style.display=jb?'grid':'none';
  $('stopbtn').hidden=!active; $('newbtn').hidden=!(jb&&!active);
  $('dot').className='dot'+(active?' on':(jb&&jb.status==='error'?' err':''));
  $('statetext').textContent = jb?jb.status:'ready';
  if(jb){
    $('head').textContent=HEAD[jb.status]||jb.status;
    $('n-done').textContent=fmt(jb.done); $('n-skip').textContent=fmt(jb.exists+jb.graded);
    $('n-fail').textContent=fmt(jb.failed); $('n-found').textContent=fmt(jb.found);
    const pct=jb.found?Math.min(100,jb.handled/jb.found*100):0; const f=$('fill'); f.style.width=pct+'%';
    f.className='fill'+(jb.status==='done'?' done':'')+(jb.failed&&jb.status==='done'?' bad':'');
    $('m-rate').textContent=jb.rate?jb.rate.toFixed(jb.rate<10?1:0)+' photos/s':'—';
    $('m-el').textContent=dur(jb.elapsed); $('m-eta').textContent=jb.eta!=null?dur(jb.eta):'—';
    $('m-cur').textContent=jb.current?'now: '+jb.current:'';
    $('wrow').style.display=active?'flex':'none';
    $('m-cpu').textContent=(jb.cpu==null)?'—':Math.round(jb.cpu)+'%'; renderTiming(jb);
    $('whint').textContent='takes effect immediately; this PC has '+jb.cpu_threads+' CPU threads'+(jb.auto_workers?' (started on automatic)':'');
    if(active&&!$('wlive').dataset.dirty&&document.activeElement!==$('wlive')) $('wlive').value=jb.workers;
    $('t-look').textContent=lastName(jb.lut)+' at '+Math.round(jb.strength*100)+'%'+(jb.finish?' · finished':' · native size');
    $('t-in').textContent=jb.input; $('t-out').textContent=jb.output; $('t-rep').textContent=jb.report;
    const w=$('srgb'); if(jb.non_srgb&&jb.non_srgb.length){w.style.display='block';
      w.textContent='Some photos carry a colour profile that is not sRGB ('+jb.non_srgb.join(', ')+'). The looks assume sRGB, so the result may not be what you expect. The profile was kept.';}
    else w.style.display='none';
    const L=$('log'); L.innerHTML=''; (jb.log||[]).forEach(m=>{const d=document.createElement('div');d.textContent=m;L.appendChild(d);});
  }
}).catch(()=>{}).finally(()=>setTimeout(tick,1000));}
paint(); tick();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore
    port: int = 8771

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
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _json(self, obj: Any, code: int = 200):
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def _body(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _allowed(self) -> bool:
        """Only this page, on this machine. Blocks another site's page from
        posting to the local server, and DNS-rebinding tricks."""
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if self.headers.get("Host", "") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return not origin or origin in {f"http://{h}" for h in hosts}

    def do_GET(self):
        if not self._allowed():
            return self._send(403, "text/plain", b"forbidden")
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif u.path == "/api/state":
            self._json(self.app.state())
        elif u.path == "/api/browse":
            self._json(self.app.browse(unquote(q.get("path", [""])[0])))
        elif u.path == "/api/prepare":
            self._json(self.app.prepare(unquote(q.get("input", [""])[0])))
        elif u.path == "/api/tile":
            try:
                i = int(q.get("i", ["0"])[0])
                strength = min(1.0, max(0.0, float(q.get("s", ["100"])[0]) / 100.0))
            except ValueError:
                return self._send(400, "text/plain", b"bad request")
            try:
                hh = int(q.get("h", ["0"])[0])
            except ValueError:
                hh = 0
            try:
                data = self.app.tile(q.get("lut", [""])[0], i, q.get("size", ["thumb"])[0],
                                     strength, q.get("orig", ["0"])[0] == "1", hh)
            except Exception as exc:                        # enlarged view: say why, and log it
                import traceback
                traceback.print_exc()
                return self._send(500, "text/plain", f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"))
            if data is None:
                self._send(404, "text/plain", b"no preview")
            else:
                self._send(200, "image/jpeg", data)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self._allowed():
            return self._send(403, "text/plain", b"forbidden")
        u = urlparse(self.path)
        if u.path == "/api/start":
            self._json(self.app.start(self._body()))
        elif u.path == "/api/stop":
            self._json(self.app.stop())
        elif u.path == "/api/warn":
            self._json(self.app.set_warn(self._body()))
        elif u.path == "/api/workers":
            self._json(self.app.set_workers(self._body()))
        elif u.path == "/api/reset":
            self._json(self.app.reset())
        else:
            self._send(404, "text/plain", b"not found")


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "settings_grade.json")
    port = int(cfg["port"])
    app = App(cfg)
    handler = type("Bound", (Handler,), {"app": app, "port": port})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Grade Room -- {url}\n  (Ctrl+C to quit)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if app.job:
            app.job.stop.set()
        print("\nbye")


if __name__ == "__main__":
    main()
