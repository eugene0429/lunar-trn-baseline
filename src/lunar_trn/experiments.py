"""Shared helpers for the V2/V3 experiments: maps per test geometry, priors and scoring."""

from __future__ import annotations

import math

import numpy as np

from . import frames
from .camera import PinholeCamera
from .dem import DemSource
from .localize import LocalizationResult, MapSet
from .maps import render_maps
from .validation.v1 import footprint_bounds

COARSE_TO_GSD = 1.5  # LVS: 12 m coarse map vs ~8 m LCAM GSD at coarse start
FINE_TO_GSD = 1.0  # LVS: 6 m fine map matched to the LCAM GSD mid-interval


def fine_map_pixel(gsd_m: float) -> float:
    """Fine map pixel for a nominal camera GSD (rounded to 0.1 m)."""
    return round(FINE_TO_GSD * gsd_m, 1)


def coarse_map_pixel(gsd_m: float) -> float:
    return round(COARSE_TO_GSD * gsd_m, 1)


def build_mapset(src: DemSource, cam: PinholeCamera, T_true: np.ndarray, sun_dir, gsd_m: float,
                 prior_radius_m: float, model: str = "hapke", fine_px: float | None = None,
                 coarse_px: float | None = None, params: dict | None = None) -> MapSet:
    """Coarse/fine maps centred on the true footprint, wide enough for any prior in the radius.

    The map centre uses the truth only for convenience: its extent covers every footprint
    reachable from the prior distribution plus the search margins, and no algorithm step
    uses the map centre.
    """
    xmin, xmax, ymin, ymax = footprint_bounds(cam, T_true)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half_fp = 0.5 * max(xmax - xmin, ymax - ymin)
    fpx = fine_px or fine_map_pixel(gsd_m)
    cpx = coarse_px or coarse_map_pixel(gsd_m)
    # Footprint + prior offset + coarse search radius (prior radius treated as 3 sigma, so
    # the 3-sigma search is the radius + 300 m) + half a coarse template + 500 m.
    half = half_fp + prior_radius_m + (prior_radius_m + 300.0) + 64 * cpx + 500.0
    half = math.ceil(half / max(cpx, fpx)) * max(cpx, fpx)
    fine, coarse = render_maps(src, (cx, cy), half, [fpx, cpx], sun_dir, model=model, params=params,
                               names=["fine", "coarse"])
    return MapSet(coarse=coarse, fine=fine)


def sample_prior(T_true: np.ndarray, rng: np.random.Generator, horiz_radius_m: float = 3000.0,
                 vert_m: float = 120.0, att_sigma_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prior pose: truth + uniform-in-disk horizontal, uniform vertical and Gaussian attitude errors.

    Returns (T_prior, prior_cov 6x6, error description).
    """
    p = T_true[:3, 3]
    enu = frames.enu_basis(p)
    r = horiz_radius_m * math.sqrt(rng.uniform())
    th = rng.uniform(0, 2 * math.pi)
    dz = rng.uniform(-vert_m, vert_m)
    d_enu = np.array([r * math.cos(th), r * math.sin(th), dz])
    T = T_true.copy()
    T[:3, 3] = p + enu @ d_enu
    w = np.zeros(3)
    if att_sigma_deg > 0:
        w = rng.normal(0, math.radians(att_sigma_deg), 3)
        from .pose import _rodrigues

        T[:3, :3] = _rodrigues(w) @ T_true[:3, :3]
    # The radius is a 3-sigma bound for the localiser (LVS "+/-3 km" initial error).
    sh = horiz_radius_m / 3.0
    sv = vert_m / 3.0
    cov_enu = np.diag([sh**2, sh**2, sv**2])
    cov = np.zeros((6, 6))
    cov[:3, :3] = enu @ cov_enu @ enu.T
    sa = math.radians(max(att_sigma_deg, 0.01))
    cov[3:, 3:] = np.eye(3) * sa**2
    return T, cov, {"prior_err_enu_m": d_enu.tolist(), "prior_err_h_m": float(r), "prior_att_err_rad": w.tolist()}


def score(res: LocalizationResult, T_true: np.ndarray, map_px: float, tol_px: float = 3.0,
          tol_m: float | None = None) -> dict:
    """Errors in the local level frame at the truth and the fix / no-fix / false-fix verdict."""
    p = T_true[:3, 3]
    enu = frames.enu_basis(p)
    d = enu.T @ (res.T_world_cam[:3, 3] - p)
    eh = float(math.hypot(d[0], d[1]))
    tol = tol_m if tol_m is not None else tol_px * map_px
    from .camera import rotation_angle_deg

    att = rotation_angle_deg(T_true[:3, :3], res.T_world_cam[:3, :3])
    if res.status != "fix":
        verdict = "no_fix"
    else:
        verdict = "correct" if eh <= tol else "false_fix"
    return {"err_e_m": float(d[0]), "err_n_m": float(d[1]), "err_u_m": float(d[2]), "err_h_m": eh,
            "err_att_deg": att, "tol_m": tol, "verdict": verdict}


def annotate_landmarks(res: LocalizationResult, cam: PinholeCamera, T_true: np.ndarray, fmap) -> dict:
    """Ground-truth match error (fine-map px) of every fine landmark that produced a peak."""
    from .rectify import predicted_map_location

    lm = res.landmarks
    if not lm or "u" not in lm:
        return {}
    u = np.asarray(lm["u"], float)
    v = np.asarray(lm["v"], float)
    ok = np.isfinite(u) & np.isfinite(v)
    err = np.full(u.shape, np.nan)
    if ok.any():
        mc_t, mr_t = predicted_map_location(cam, T_true, fmap, u[ok] - cam.pixel_center, v[ok] - cam.pixel_center)
        X = np.asarray(lm["X"], float)[ok]
        x, y, _ = frames.body_to_stereo(X)
        mc, mr = fmap.xy_to_colrow(x, y)
        err[ok] = np.hypot(mc - mc_t, mr - mr_t)
    return {"match_err_px": err}


def annotate_coarse(res: LocalizationResult, cam: PinholeCamera, T_true: np.ndarray, cmap) -> list:
    """Ground-truth error (coarse-map px) of each coarse template match that has a peak."""
    from .rectify import predicted_map_location

    out = []
    for t in res.health.get("coarse", {}).get("templates", []):
        if "uv" not in t:
            out.append(float("nan"))
            continue
        uq, vq = t["uv"]
        mc_t, mr_t = predicted_map_location(cam, T_true, cmap, np.array([uq - cam.pixel_center]),
                                            np.array([vq - cam.pixel_center]))
        r0, c0 = t["tpl_rc"]
        tc = c0 + 64 - 0.5 + t["offset_px"][0]
        tr = r0 + 64 - 0.5 + t["offset_px"][1]
        out.append(float(np.hypot(tc - mc_t[0], tr - mr_t[0])))
    return out
