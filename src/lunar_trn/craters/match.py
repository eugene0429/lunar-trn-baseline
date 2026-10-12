"""Tracking-mode crater matching: catalog prediction, position-hypothesis voting, assignment.

With the attitude known and a narrow vertical prior, one (detection, catalog crater) pair
fixes a horizontal camera-position hypothesis: the camera lies on the detection ray through
the crater centre, at the prior height. Hypotheses of size-compatible pairs are voted on a
horizontal grid around the prior; the strongest peaks are verified by gated one-to-one
assignment and the pose back end (:mod:`lunar_trn.craters.localize`). Thresholds the papers
leave open are in :class:`MatchConfig` (decisions G8-G11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .. import frames
from ..validation.v1 import footprint_bounds
from .catalog import CraterCatalog
from .conics import project_rims
from .detect import Detections


@dataclass
class MatchConfig:
    search_margin_m: float = 300.0  # search radius = 3 sigma_h + margin (same rule as T-I coarse, C3)
    catalog_D_min_m: float = 0.0  # run-time catalog size floor, 0 = as built (H-c sensitivity)
    size_pad: tuple = (0.7, 1.3)  # keep candidates whose predicted D_px is within pad x the detector range
    size_ratio_vote: float = 1.6  # size compatibility of a voting pair
    vote_bin_gsd: float = 2.0  # vote grid cell, x image GSD at the prior range
    n_hypotheses: int = 3
    exclusion_bins: float = 4.0  # peak suppression radius >= this many cells
    exclusion_fov_frac: float = 0.1  # and >= this fraction of the footprint width
    gate1_px: float = 4.0  # first assignment gate max(gate1_px, gate1_rel * D_px)
    gate1_rel: float = 0.4
    gate2_px: float = 2.0  # second assignment gate (refined pose)
    gate2_rel: float = 0.2
    size_ratio_gate: float = 1.6
    ransac_px: float = 3.0  # fixed-attitude position RANSAC threshold
    sigma_min_px: float = 0.794  # per-crater sigma = max(sigma_min_px, sigma_rel * D_px), calibrated (G11)
    sigma_rel: float = 0.045
    visibility_filter: bool = False  # d/D >= c_vis tan(e) / 4 (interior shadow possible); off after calibration (G8)
    c_vis: float = 1.0
    image_margin_px: float = 0.0


@dataclass
class Candidates:
    catalog: CraterCatalog  # craters selected for this image
    pred: dict  # their prediction at the prior pose (project_rims output)
    n_window: int  # catalog craters in the search window (after catalog_D_min_m) before selection


def predict(catalog: CraterCatalog, K, R_wc, c) -> dict:
    return project_rims(K, R_wc, c, catalog.X, catalog.e1, catalog.e2, catalog.a, catalog.b, catalog.psi)


def _empty_candidates(K, R, c) -> Candidates:
    e = CraterCatalog.empty()
    return Candidates(e, predict(e, K, R, c), 0)


def select_candidates(catalog: CraterCatalog, cam, T_prior, sigma_h: float, sun_dir_world, D_px_range,
                      cfg: MatchConfig) -> Candidates:
    """Catalog craters near the prior footprint whose predicted size suits the detector."""
    T = np.asarray(T_prior, float)
    R, c = T[:3, :3], T[:3, 3]
    if len(catalog) == 0:
        return _empty_candidates(cam.K, R, c)
    reach = 3.0 * sigma_h + cfg.search_margin_m
    try:
        xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
    except ValueError:
        return _empty_candidates(cam.K, R, c)
    win = catalog.query_box(xmin - reach, xmax + reach, ymin - reach, ymax + reach)
    if cfg.catalog_D_min_m > 0:
        win = win.subset(win.D >= cfg.catalog_D_min_m)
    if len(win) == 0:
        return _empty_candidates(cam.K, R, c)
    pred = predict(win, cam.K, R, c)
    lo, hi = D_px_range
    with np.errstate(invalid="ignore"):
        keep = pred["valid"] & (pred["D_px"] >= cfg.size_pad[0] * lo) & (pred["D_px"] <= cfg.size_pad[1] * hi)
        if cfg.visibility_filter:
            s = np.asarray(sun_dir_world, float)
            s = s / np.linalg.norm(s)
            e = np.arcsin(np.clip(win.normal @ s, -1.0, 1.0))
            keep &= (e > 0) & (win.d_over_D >= cfg.c_vis * np.tan(e) / 4.0)
    idx = np.nonzero(keep)[0]
    return Candidates(win.subset(idx), {k: v[idx] for k, v in pred.items()}, len(win))


def vote_positions(det: Detections, cand: Candidates, cam, T_prior, sigma_h: float, cfg: MatchConfig) -> dict:
    """Strongest horizontal camera-position offsets (east, north at the prior) from pair votes."""
    out = {"offsets": np.zeros((0, 2)), "votes": np.zeros(0), "counts": np.zeros(0, int), "n_pairs": 0,
           "bin_m": float("nan"), "radius_m": float("nan")}
    if len(det) == 0 or len(cand.catalog) == 0:
        return out
    T = np.asarray(T_prior, float)
    R, p0 = T[:3, :3], T[:3, 3]
    east, north, up = frames.enu_basis(p0).T
    d = (np.column_stack([det.u, det.v, np.ones(len(det))]) @ np.linalg.inv(cam.K).T) @ R.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(det.D_px[:, None] / cand.pred["D_px"][None, :]))
    ii, jj = np.nonzero(lr <= np.log(cfg.size_ratio_vote))
    if ii.size == 0:
        return out
    X, di = cand.catalog.X[jj], d[ii]
    du = di @ up
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((X - p0) @ up) / np.where(du < -1e-9, du, np.nan)  # camera at the prior height on the ray
    C = X - t[:, None] * di - p0
    off = np.column_stack([C @ east, C @ north])
    radius = 3.0 * sigma_h + cfg.search_margin_m
    good = np.isfinite(off).all(axis=1) & (t > 0) & (np.hypot(off[:, 0], off[:, 1]) <= radius)
    off = off[good]
    out["n_pairs"], out["radius_m"] = int(off.shape[0]), float(radius)
    if off.shape[0] == 0:
        return out
    gsd = float(np.nanmedian(cand.pred["z"])) / cam.fx
    bin_m = max(cfg.vote_bin_gsd * gsd, 1.0)
    nb = max(int(np.ceil(2 * radius / bin_m)), 1)
    hist, xe, _ = np.histogram2d(off[:, 0], off[:, 1], bins=nb, range=[[-radius, radius], [-radius, radius]])
    sm = ndi.gaussian_filter(hist, 1.0, mode="constant")
    centres = 0.5 * (xe[:-1] + xe[1:])
    gx, gy = np.meshgrid(centres, centres, indexing="ij")
    excl = max(cfg.exclusion_bins * bin_m, cfg.exclusion_fov_frac * cam.width * gsd)
    work = sm.copy()
    offs, votes, counts = [], [], []
    for _ in range(cfg.n_hypotheses):
        k = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[k] <= 0:
            break
        o = np.array([centres[k[0]], centres[k[1]]])
        offs.append(o)
        votes.append(float(sm[k]))
        counts.append(int(np.sum(np.hypot(off[:, 0] - o[0], off[:, 1] - o[1]) <= 1.5 * bin_m)))
        work[np.hypot(gx - o[0], gy - o[1]) <= excl] = 0.0
    out.update(offsets=np.array(offs).reshape(-1, 2), votes=np.array(votes), counts=np.array(counts, int),
               bin_m=float(bin_m))
    return out


def assign(det_uv, det_D, pred_uv, pred_D, gate_px: float, gate_rel: float, size_ratio_max: float):
    """Gated one-to-one assignment (Hungarian) of detections to predicted craters."""
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    det_uv, pred_uv = np.asarray(det_uv, float).reshape(-1, 2), np.asarray(pred_uv, float).reshape(-1, 2)
    if len(det_uv) == 0 or len(pred_uv) == 0:
        return empty
    det_D, pred_D = np.asarray(det_D, float), np.asarray(pred_D, float)
    fin = np.isfinite(pred_uv).all(axis=1) & np.isfinite(pred_D) & (pred_D > 0)
    dist = np.hypot(det_uv[:, None, 0] - pred_uv[None, :, 0], det_uv[:, None, 1] - pred_uv[None, :, 1])
    gate = np.maximum(gate_px, gate_rel * det_D)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(det_D[:, None] / pred_D[None, :]))
        ok = fin[None, :] & (dist <= gate) & (lr <= np.log(size_ratio_max))
    if not ok.any():
        return empty
    rows, cols = np.nonzero(ok.any(axis=1))[0], np.nonzero(ok.any(axis=0))[0]
    sub = ok[np.ix_(rows, cols)]
    cost = np.where(sub, (dist / gate)[np.ix_(rows, cols)] + np.nan_to_num(lr[np.ix_(rows, cols)]), 1e6)
    ri, ci = linear_sum_assignment(cost)
    keep = sub[ri, ci]
    i, j = rows[ri[keep]], cols[ci[keep]]
    return i, j, dist[i, j]
