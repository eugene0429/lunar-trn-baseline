"""Correlation primitives shared by the coarse and fine matchers.

* :func:`local_normalize` is the LVS pre-filter: subtract the local mean and divide by the
  local standard deviation over a 21 x 21 window (Johnson et al. 2023).
* :func:`correlate` computes the NCC surface of a template over a search area
  (``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``; OpenCV switches to FFT for large
  templates, which is the coarse FFT correlation of LVS).
* :func:`analyze_peak` extracts the raw peak diagnostics that LVS gates on (height, width,
  biquadratic sub-pixel fit and its shape) plus PSR and the second-peak ratio used as
  health signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def local_normalize(img: np.ndarray, win: int = 21, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """(img - local mean) / local std over ``win x win``. NaNs are treated as missing.

    Returns the normalised image (0 where undefined) and the local standard deviation.
    """
    import cv2

    a = np.asarray(img, dtype=np.float32)
    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(valid > 0, a, 0.0).astype(np.float32)
    k = (win, win)
    cnt = cv2.boxFilter(valid, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    s1 = cv2.boxFilter(a0, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.boxFilter(a0 * a0, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / np.maximum(cnt, 1.0)
        var = np.maximum(s2 / np.maximum(cnt, 1.0) - mean * mean, 0.0)
    std = np.sqrt(var)
    out = np.where((valid > 0) & (cnt >= 0.5 * win * win), (a0 - mean) / (std + eps), 0.0)
    return out.astype(np.float32), std.astype(np.float32)


def correlate(search: np.ndarray, template: np.ndarray, min_std: float = 1e-3) -> np.ndarray:
    """Zero-mean NCC surface, shape (Hs - Ht + 1, Ws - Wt + 1).

    NCC is undefined where the search window (or the template) has no variance, e.g. a
    fully shadowed map area; OpenCV returns arbitrary values (often 1.0) there, so those
    positions are set to 0.
    """
    import cv2

    s = np.ascontiguousarray(search, dtype=np.float32)
    t = np.ascontiguousarray(template, dtype=np.float32)
    if float(np.std(t)) < min_std:
        return np.zeros((s.shape[0] - t.shape[0] + 1, s.shape[1] - t.shape[1] + 1), dtype=np.float32)
    surf = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
    th, tw = t.shape
    n = float(th * tw)
    s64 = s.astype(np.float64)
    ii = cv2.integral(s64)
    ii2 = cv2.integral(s64 * s64)
    tot = ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw]
    tot2 = ii2[th:, tw:] - ii2[:-th, tw:] - ii2[th:, :-tw] + ii2[:-th, :-tw]
    var = np.maximum(tot2 / n - (tot / n) ** 2, 0.0)
    surf = np.where(var > min_std * min_std, surf, 0.0).astype(np.float32)
    return np.clip(surf, -1.0, 1.0)


@dataclass
class Peak:
    ix: int  # integer peak column in the surface
    iy: int
    dx: float  # sub-pixel offset of the fitted maximum from (ix, iy)
    dy: float
    height: float  # surface value at the integer peak
    fit_height: float  # biquadratic value at the fitted maximum
    width: float  # half-height width (px) along the flattest principal direction
    width_min: float  # same along the sharpest direction
    concave_down: bool
    psr: float  # (peak - mean) / std of the surface outside the exclusion zone
    second_ratio: float  # best value outside the exclusion zone / peak
    on_border: bool

    @property
    def x(self) -> float:
        return self.ix + self.dx

    @property
    def y(self) -> float:
        return self.iy + self.dy


def biquadratic_fit(z3: np.ndarray) -> tuple[float, float, float, np.ndarray, bool]:
    """Fit z = a + b x + c y + d x^2 + e x y + f y^2 to a 3x3 patch centred at (0, 0).

    Returns (dx, dy, value_at_max, hessian, concave_down).
    """
    ys, xs = np.mgrid[-1:2, -1:2]
    A = np.column_stack([np.ones(9), xs.ravel(), ys.ravel(), xs.ravel() ** 2, (xs * ys).ravel(), ys.ravel() ** 2])
    coef, *_ = np.linalg.lstsq(A, z3.ravel().astype(np.float64), rcond=None)
    a, b, c, d, e, f = coef
    H = np.array([[2 * d, e], [e, 2 * f]])
    concave = bool(np.all(np.linalg.eigvalsh(H) < 0))
    try:
        off = -np.linalg.solve(H, [b, c])
    except np.linalg.LinAlgError:
        off = np.array([0.0, 0.0])
        concave = False
    val = a + b * off[0] + c * off[1] + d * off[0] ** 2 + e * off[0] * off[1] + f * off[1] ** 2
    return float(off[0]), float(off[1]), float(val), H, concave


def analyze_peak(surface: np.ndarray, exclusion: int = 5) -> Peak:
    """Peak diagnostics of a correlation surface (see module docstring)."""
    s = np.asarray(surface, dtype=np.float64)
    iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    h = float(s[iy, ix])
    border = ix == 0 or iy == 0 or ix == s.shape[1] - 1 or iy == s.shape[0] - 1
    if border:
        dx = dy = 0.0
        fit = h
        H = np.full((2, 2), np.nan)
        concave = False
    else:
        dx, dy, fit, H, concave = biquadratic_fit(s[iy - 1:iy + 2, ix - 1:ix + 2])
    # Full width at half maximum of the quadratic model z = z_max + t^2 lambda / 2 along each
    # principal direction: 2 sqrt(z_max / |lambda|).
    if concave and fit > 0:
        lam = np.linalg.eigvalsh(H)  # both negative
        w = 2.0 * np.sqrt(fit / np.abs(lam))
        width, width_min = float(w.max()), float(w.min())
    else:
        width = width_min = float("inf")
    mask = np.ones_like(s, dtype=bool)
    mask[max(iy - exclusion, 0):iy + exclusion + 1, max(ix - exclusion, 0):ix + exclusion + 1] = False
    if mask.sum() > 4:
        side = s[mask]
        psr = float((h - side.mean()) / (side.std() + 1e-12))
        second = float(side.max() / h) if h > 0 else float("nan")
    else:
        psr = second = float("nan")
    return Peak(int(ix), int(iy), dx, dy, h, fit, width, width_min, concave, psr, second, bool(border))


def phase_correlation_response(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """OpenCV phase correlation of two equal-size patches: (dx, dy, response)."""
    import cv2

    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_64F)
    (dx, dy), resp = cv2.phaseCorrelate(np.asarray(a, np.float64), np.asarray(b, np.float64), win)
    return float(dx), float(dy), float(resp)
