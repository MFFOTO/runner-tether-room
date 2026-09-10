# -*- coding: utf-8 -*-
"""
Minimal .cube LUT support - vendored from D:/runner-lut-suite/lut_suite.py.

Only the parts the live pipeline actually needs (parse + apply) are kept, so
this repo has no runtime dependency on the LUT suite being present. NumPy only
-- no Pillow, no torch, no GPU.

Channel-order note, the one real trap here: the LUT suite is built around PIL
and therefore works in RGB, while runner_suite_core is built around cv2 and
works in BGR. Feeding a BGR frame to the RGB functions swaps red and blue and
quietly ruins every grade. Use build_dense_lut_bgr() / apply_dense_bgr() for
cv2 images and the plain RGB functions for anything Pillow-shaped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Parsing
# --------------------------------------------------------------------------- #
def parse_cube(path: Path) -> Dict[str, Any]:
    """Parse an Adobe .cube LUT (1D or 3D). Returns a dict with the lookup
    table already oriented as table[i_r, i_g, i_b] (3D) or curve[n,3] (1D)."""
    size: Optional[int] = None
    dim = 3
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]
    title = None
    data: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            u = s.upper()
            if u.startswith("TITLE"):
                bits = s.split(None, 1)
                title = bits[1].strip().strip('"') if len(bits) > 1 else None
                continue
            if u.startswith("LUT_3D_SIZE"):
                size, dim = int(s.split()[1]), 3
                continue
            if u.startswith("LUT_1D_SIZE"):
                size, dim = int(s.split()[1]), 1
                continue
            if u.startswith("DOMAIN_MIN"):
                dmin = [float(x) for x in s.split()[1:4]]
                continue
            if u.startswith("DOMAIN_MAX"):
                dmax = [float(x) for x in s.split()[1:4]]
                continue
            if u.startswith("LUT_3D_INPUT_RANGE") or u.startswith("LUT_1D_INPUT_RANGE"):
                v = [float(x) for x in s.split()[1:3]]
                dmin, dmax = [v[0]] * 3, [v[1]] * 3
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
    if size is None:
        raise ValueError(f"{path.name}: missing LUT_3D_SIZE / LUT_1D_SIZE")
    arr = np.asarray(data, dtype=np.float32)
    out = {"dim": dim, "size": size, "title": title,
           "dmin": np.asarray(dmin, dtype=np.float32),
           "dmax": np.asarray(dmax, dtype=np.float32)}
    if dim == 3:
        if arr.shape[0] != size ** 3:
            raise ValueError(f"{path.name}: expected {size**3} rows, got {arr.shape[0]}")
        # rows iterate red fastest -> reshape gives [b,g,r], transpose to [r,g,b]
        out["table"] = arr.reshape(size, size, size, 3).transpose(2, 1, 0, 3).copy()
    else:
        if arr.shape[0] != size:
            raise ValueError(f"{path.name}: expected {size} rows, got {arr.shape[0]}")
        out["curve"] = arr
    return out


def list_luts(folder: Path) -> List[Path]:
    """Every .cube in folder, name-sorted, case-insensitively."""
    if not folder.exists():
        return []
    return sorted(folder.glob("*.cube"), key=lambda p: p.name.lower())


# --------------------------------------------------------------------------- #
#  Application (RGB)
# --------------------------------------------------------------------------- #
def _trilinear(rgb01: np.ndarray, table: np.ndarray) -> np.ndarray:
    """rgb01: (M,3) in [0,1]; table: (N,N,N,3) indexed [i_r,i_g,i_b]."""
    n = table.shape[0]
    scale = n - 1
    c = rgb01 * scale
    c0 = np.clip(np.floor(c).astype(np.int64), 0, n - 1)
    c1 = np.clip(c0 + 1, 0, n - 1)
    f = c - c0
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    r0, g0, b0 = c0[:, 0], c0[:, 1], c0[:, 2]
    r1, g1, b1 = c1[:, 0], c1[:, 1], c1[:, 2]

    def T(ri, gi, bi):
        return table[ri, gi, bi]

    c00 = T(r0, g0, b0) * (1 - fr) + T(r1, g0, b0) * fr
    c10 = T(r0, g1, b0) * (1 - fr) + T(r1, g1, b0) * fr
    c01 = T(r0, g0, b1) * (1 - fr) + T(r1, g0, b1) * fr
    c11 = T(r0, g1, b1) * (1 - fr) + T(r1, g1, b1) * fr
    c0_ = c00 * (1 - fg) + c10 * fg
    c1_ = c01 * (1 - fg) + c11 * fg
    return c0_ * (1 - fb) + c1_ * fb


def _curve1d(rgb01: np.ndarray, curve: np.ndarray) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, curve.shape[0], dtype=np.float32)
    out = np.empty_like(rgb01)
    for ch in range(3):
        out[:, ch] = np.interp(rgb01[:, ch], xs, curve[:, ch])
    return out


def apply_lut(img_uint8: np.ndarray, lut: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """Apply a parsed LUT to an HxWx3 uint8 *RGB* image. Chunked to bound
    memory. Fine for thumbnails; for full-size frames prefer the dense path."""
    rgb = img_uint8.astype(np.float32) / 255.0
    span = np.maximum(lut["dmax"] - lut["dmin"], 1e-8)
    norm = np.clip((rgb - lut["dmin"]) / span, 0.0, 1.0)
    flat = norm.reshape(-1, 3)
    out = np.empty_like(flat)
    chunk = 1_000_000
    for i in range(0, flat.shape[0], chunk):
        blk = flat[i:i + chunk]
        out[i:i + chunk] = _trilinear(blk, lut["table"]) if lut["dim"] == 3 else _curve1d(blk, lut["curve"])
    graded = np.clip(out.reshape(rgb.shape), 0.0, 1.0)
    if strength < 1.0:
        graded = rgb * (1.0 - strength) + graded * strength
    return (graded * 255.0 + 0.5).astype(np.uint8)


def build_dense_lut(lut: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """Pre-expand a LUT into a dense 256x256x256x3 uint8 table indexed
    [r,g,b] -> RGB (one-time ~50MB / a few seconds). Applying it to an image is
    then a single array gather, far faster than interpolating every pixel."""
    r = np.arange(256, dtype=np.uint8)
    R, G, B = np.meshgrid(r, r, r, indexing="ij")
    grid = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    return apply_lut(grid, lut, strength).reshape(256, 256, 256, 3)


def apply_dense(img_uint8: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Apply a dense LUT to an RGB uint8 image: dense[r, g, b] per pixel."""
    return dense[img_uint8[..., 0], img_uint8[..., 1], img_uint8[..., 2]]


# --------------------------------------------------------------------------- #
#  Application (BGR / cv2)
# --------------------------------------------------------------------------- #
def build_dense_lut_bgr(lut: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """Dense table re-indexed for cv2: [b,g,r] -> BGR. Doing the swap once here
    keeps the per-image call a single gather with no extra channel copy."""
    dense = build_dense_lut(lut, strength)                 # [r,g,b] -> RGB
    return np.ascontiguousarray(dense.transpose(2, 1, 0, 3)[..., ::-1])


def apply_dense_bgr(img_bgr: np.ndarray, dense_bgr: np.ndarray) -> np.ndarray:
    """Apply a build_dense_lut_bgr() table to a cv2 BGR uint8 image."""
    return dense_bgr[img_bgr[..., 0], img_bgr[..., 1], img_bgr[..., 2]]
