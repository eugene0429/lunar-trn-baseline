"""V1: frame and pose consistency between a dataset and a DEM.

For a dataset image with ground-truth pose:

1. Depth check. Cast the pixel rays into the DEM and compare the hit distance with the
   dataset depth map, both as range along the ray and as z-depth. A frame mismatch
   (e.g. MOON_ME vs MOON_PA, ~0.8 km at the surface) or a wrong DEM product shows up as a
   large systematic residual or as an image-plane shift between the two depth maps; the
   shift is estimated to sub-pixel precision by Gauss-Newton registration.
2. Render check. Render the DEM from the same pose with the reconstructed Sun direction
   and compare with the image (normalised cross-correlation, phase-correlation shift, and
   a Sun-azimuth sweep that tests the Sun reconstruction itself).

All functions return raw numbers; pass/fail is left to the report script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import frames
from ..camera import PinholeCamera
from ..dem import Dem, DemSource, bilinear
from ..render import perspective_hits, render_perspective, shade_hits


def half_float_step(depth) -> np.ndarray:
    """Spacing of IEEE half floats at ``depth`` (quantisation step of the EXR depth maps)."""
    d = np.maximum(np.abs(np.asarray(depth, dtype=np.float64)), 1e-3)
    return np.exp2(np.floor(np.log2(d)) - 10.0)


def footprint_bounds(cam: PinholeCamera, T_world_cam: np.ndarray, h_lo: float = -4500.0,
                     h_hi: float = 2800.0, n_edge: int = 16) -> tuple[float, float, float, float]:
    """Stereographic bounding box of the image footprint between two terrain heights."""
    edge = np.linspace(-0.5, cam.width - 0.5, n_edge)
    cols = np.concatenate([edge, edge, np.full(n_edge, -0.5), np.full(n_edge, cam.width - 0.5)])
    rows = np.concatenate([np.full(n_edge, -0.5), np.full(n_edge, cam.height - 0.5), edge, edge])
    o, d = cam.rays_world(T_world_cam, cols, rows)
    pts = []
    for h in (h_lo, h_hi):
        r = frames.MOON_RADIUS_M + h
        b = d @ o
        c = o @ o - r * r
        disc = b * b - c
        t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
        pts.append(o + t[:, None] * d)
    pts = np.concatenate(pts)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        raise ValueError("footprint does not intersect the terrain shell")
    x, y, _ = frames.body_to_stereo(pts)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def read_frame_window(src: DemSource, cam: PinholeCamera, T_world_cam: np.ndarray,
                      sun_dir=None, margin_m: float = 300.0, shadow_reach_m: float = 0.0) -> Dem:
    """DEM window covering the footprint, extended towards the Sun by ``shadow_reach_m``."""
    xmin, xmax, ymin, ymax = footprint_bounds(cam, T_world_cam)
    xmin, xmax, ymin, ymax = xmin - margin_m, xmax + margin_m, ymin - margin_m, ymax + margin_m
    if sun_dir is not None and shadow_reach_m > 0:
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        p0 = frames.stereo_to_body(cx, cy, 0.0)
        p1 = p0 + 1000.0 * np.asarray(sun_dir)
        x1, y1, _ = frames.body_to_stereo(p1)
        vx, vy = x1 - cx, y1 - cy
        nv = math.hypot(vx, vy)
        if nv > 0:
            dx, dy = vx / nv * shadow_reach_m, vy / nv * shadow_reach_m
            xmin, xmax = min(xmin, xmin + dx), max(xmax, xmax + dx)
            ymin, ymax = min(ymin, ymin + dy), max(ymax, ymax + dy)
    return src.read_window(xmin, xmax, ymin, ymax)


def _sample(grid: np.ndarray, cols, rows) -> np.ndarray:
    return bilinear(grid, cols, rows)


def register_shift(reference: np.ndarray, moving_cols, moving_rows, moving_vals: np.ndarray,
                   n_iter: int = 15, smooth_sigma: float = 1.0, estimate_offset: bool = True) -> dict:
    """Sub-pixel shift (du, dv) and offset b with ``reference(c + du, r + dv) ~ moving + b``.

    ``reference`` is a dense image (e.g. the dataset depth map); the moving samples sit at
    pixel indices (cols, rows). Gauss-Newton with Huber weights on a lightly smoothed
    reference; returns the estimate with formal standard errors.
    """
    from scipy.ndimage import gaussian_filter

    ref = np.where(np.isfinite(reference), reference, np.nan).astype(np.float64)
    fill = np.nanmedian(ref)
    ref_f = np.where(np.isfinite(ref), ref, fill)
    ref_s = gaussian_filter(ref_f, smooth_sigma) if smooth_sigma > 0 else ref_f
    gy, gx = np.gradient(ref_s)
    invalid = gaussian_filter((~np.isfinite(ref)).astype(float), 1.0) > 1e-3
    c = np.asarray(moving_cols, dtype=np.float64).ravel()
    r = np.asarray(moving_rows, dtype=np.float64).ravel()
    m = np.asarray(moving_vals, dtype=np.float64).ravel()
    ok = np.isfinite(m)
    du = dv = 0.0
    b = float(np.nanmedian(_sample(ref_s, c, r)[ok] - m[ok])) if estimate_offset else 0.0
    cov = np.full((3, 3), np.nan)
    sigma = np.nan
    n_used = 0
    for _ in range(n_iter):
        cc, rr = c + du, r + dv
        val = _sample(ref_s, cc, rr)
        jx = _sample(gx, cc, rr)
        jy = _sample(gy, cc, rr)
        bad = _sample(invalid.astype(float), cc, rr)
        sel = ok & np.isfinite(val) & np.isfinite(jx) & (bad < 0.5)
        res = val[sel] - m[sel] - b
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        k = 1.5 * s
        wts = np.where(np.abs(res) <= k, 1.0, k / np.abs(res))
        if estimate_offset:
            # d(res)/d(b) = -1.
            J = np.column_stack([jx[sel], jy[sel], -np.ones(sel.sum())])
        else:
            J = np.column_stack([jx[sel], jy[sel]])
        A = J.T @ (J * wts[:, None])
        g = J.T @ (wts * res)
        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        du += delta[0]
        dv += delta[1]
        if estimate_offset:
            b += delta[2]
        sigma = float(np.sqrt(np.sum(wts * res**2) / max(sel.sum() - J.shape[1], 1)))
        Ainv = np.linalg.inv(A)
        cov = Ainv * sigma**2
        n_used = int(sel.sum())
        if np.max(np.abs(delta[:2])) < 1e-4:
            break
    se = np.sqrt(np.diag(cov))
    return {"du": float(du), "dv": float(dv), "offset": float(b),
            "du_se": float(se[0]), "dv_se": float(se[1]), "sigma": sigma, "n": n_used}


def robust_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    med = float(np.median(x))
    return {
        "n": int(x.size),
        "median": med,
        "mean": float(x.mean()),
        "rms": float(np.sqrt(np.mean(x**2))),
        "nmad": float(1.4826 * np.median(np.abs(x - med))),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "abs_p50": float(np.median(np.abs(x))),
    }


@dataclass
class DepthCheck:
    stats_range: dict
    stats_z: dict
    shift_range: dict
    shift_z: dict
    quant_step_median: float
    boresight_range: float
    gsd_m: float
    hit_fraction: float
    footprint_center_xy: tuple[float, float]
    footprint_center_latlon: tuple[float, float]


def depth_check(dem: Dem, cam: PinholeCamera, T_world_cam: np.ndarray, depth_map: np.ndarray,
                step: int = 2) -> tuple[DepthCheck, dict]:
    """Compare DEM ray-cast distances with the dataset depth map (see module docstring)."""
    off = step // 2
    cols, rows = cam.pixel_grid(step=step, offset=float(off))
    o, d = cam.rays_world(T_world_cam, cols, rows)
    dflat = d.reshape(-1, 3)
    t = dem.raycast(np.broadcast_to(o, dflat.shape), dflat)
    rng = t.reshape(cols.shape)
    zdep = (t * (dflat @ T_world_cam[:3, 2])).reshape(cols.shape)
    E = depth_map.astype(np.float64)
    Eg = E[rows.astype(int), cols.astype(int)]
    Eg = np.where(np.isfinite(Eg), Eg, np.nan)
    res_r = Eg - rng
    res_z = Eg - zdep
    ci, ri = cam.width // 2, cam.height // 2
    centre = E[ri - 2:ri + 2, ci - 2:ci + 2]
    bore = float(np.nanmedian(np.where(np.isfinite(centre), centre, np.nan)))
    # Boresight ground point for bookkeeping.
    oc, dc = cam.rays_world(T_world_cam, np.array([cam.width / 2 - cam.pixel_center]),
                            np.array([cam.height / 2 - cam.pixel_center]))
    tc = dem.raycast(oc[None], dc)
    pc = oc + tc[0] * dc[0]
    xc, yc, _ = frames.body_to_stereo(pc)
    lat, lon = frames.stereo_to_latlon(xc, yc)
    check = DepthCheck(
        stats_range=robust_stats(res_r),
        stats_z=robust_stats(res_z),
        shift_range=register_shift(E, cols, rows, rng),
        shift_z=register_shift(E, cols, rows, zdep),
        quant_step_median=float(np.nanmedian(half_float_step(Eg))),
        boresight_range=bore,
        gsd_m=bore / cam.fx if np.isfinite(bore) else float("nan"),
        hit_fraction=float(np.isfinite(rng).mean()),
        footprint_center_xy=(float(xc), float(yc)),
        footprint_center_latlon=(float(lat), float(lon)),
    )
    arrays = {"cols": cols, "rows": rows, "dem_range": rng, "dem_z": zdep, "data_depth": Eg,
              "res_range": res_r, "res_z": res_z}
    return check, arrays


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 10:
        return float("nan")
    x = a[m] - a[m].mean()
    y = b[m] - b[m].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def phase_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """Shift (dx, dy) of ``mov`` relative to ``ref`` by phase correlation, with response."""
    import cv2

    def prep(a):
        a = np.where(np.isfinite(a), a, np.nanmean(a)).astype(np.float64)
        return (a - a.mean()) / (a.std() + 1e-12)

    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_64F)
    (dx, dy), resp = cv2.phaseCorrelate(prep(ref), prep(mov), win)
    return float(dx), float(dy), float(resp)


def sun_from_az_el(p, az_deg: float, el_deg: float) -> np.ndarray:
    enu = frames.enu_basis(p)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.cos(el) * (np.cos(az) * enu[:, 1] + np.sin(az) * enu[:, 0]) + np.sin(el) * enu[:, 2]


def render_check(dem: Dem, cam: PinholeCamera, T_world_cam: np.ndarray, image: np.ndarray,
                 sun_dir: np.ndarray, step: int = 2, model: str = "hapke") -> tuple[dict, dict]:
    """Render with ``sun_dir`` and compare with the dataset image (NCC, phase shift)."""
    from scipy.ndimage import uniform_filter

    out = render_perspective(dem, cam, T_world_cam, sun_dir, step=step, model=model, shadows=True)
    # Box-average the image to the render sampling to suppress aliasing.
    img = uniform_filter(image.astype(np.float64), size=step) if step > 1 else image.astype(np.float64)
    ig = img[out["rows"].astype(int), out["cols"].astype(int)]
    rend = out["image"]
    rec = {"ncc": ncc(rend, ig), "shadow_fraction_render": float(np.nanmean(out["shadow"])),
           "image_dark_fraction": float(np.mean(image < 0.02))}
    dx, dy, resp = phase_shift(ig, np.where(np.isfinite(rend), rend, np.nan))
    rec.update({"shift_dx_px": dx * step, "shift_dy_px": dy * step, "shift_response": resp})
    p = T_world_cam[:3, 3]
    s_loc = np.asarray(sun_dir) @ frames.enu_basis(p)
    rec["sun_az_deg"] = float(np.rad2deg(np.arctan2(s_loc[0], s_loc[1])) % 360)
    rec["sun_el_deg"] = float(np.rad2deg(np.arcsin(np.clip(s_loc[2], -1, 1))))
    arrays = {"render": rend, "image": ig, "cols": out["cols"], "rows": out["rows"], "shadow": out["shadow"]}
    return rec, arrays


def estimate_sun(dem: Dem, cam: PinholeCamera, T_world_cam: np.ndarray, image: np.ndarray,
                 step: int = 8, az_step: float = 10.0, el_grid=(5, 10, 15, 20, 25, 30, 40, 50, 60, 75),
                 model: str = "hapke", top_k: int = 5) -> dict:
    """Sun azimuth/elevation that best explains the image (shading + cast shadows).

    Coarse grid without cast shadows, the ``top_k`` candidates re-scored with shadows, then a
    local refinement with shadows. Used only to characterise the dataset (the TRN pipeline
    takes the Sun from ephemeris).
    """
    from scipy.ndimage import uniform_filter

    hits = perspective_hits(dem, cam, T_world_cam, step=step)
    img = uniform_filter(image.astype(np.float64), size=step)
    ig = img[hits["rows"].astype(int), hits["cols"].astype(int)]
    p = T_world_cam[:3, 3]

    def score(az, el, shadows):
        r, _ = shade_hits(dem, hits, sun_from_az_el(p, az, el), model=model, shadows=shadows)
        return ncc(r, ig)

    coarse = [(score(az, el, False), az, el) for az in np.arange(0, 360, az_step) for el in el_grid]
    coarse = [c for c in coarse if np.isfinite(c[0])]
    coarse.sort(reverse=True)
    rescored = sorted(((score(az, el, True), az, el) for _, az, el in coarse[:top_k]), reverse=True)
    best = rescored[0]
    for (da, de) in ((az_step / 2, 2.5), (az_step / 4, 1.25)):
        cands = [(best[1] + a, best[2] + e) for a in (-da, 0, da) for e in (-de, 0, de) if 0 < best[2] + e < 90]
        scored = [(score(az % 360, el, True), az % 360, el) for az, el in cands]
        best = max(scored + [best])
    return {"az_deg": float(best[1]), "el_deg": float(best[2]), "ncc": float(best[0]),
            "coarse_best_no_shadow": [float(v) for v in coarse[0]]}
