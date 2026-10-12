"""Coarse map-relative localisation (LVS coarse phase, single-image adaptation).

1. Warp the image into the coarse map frame with the prior pose.
2. Pick up to ``n_templates`` 128 x 128 (map px) templates with the strongest texture.
3. Correlate each locally normalised template over a search window sized from the prior
   position uncertainty (NCC via FFT).
4. Peak tests, then a position-only RANSAC over the template matches (attitude from the
   prior); its reprojection gate is the LVS 'geometric outlier' check.

All thresholds live in :class:`CoarseConfig`; values not published by LVS are documented in
``docs/decisions.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import frames
from .camera import PinholeCamera
from .maps import TrnMap
from .matching import analyze_peak, correlate, local_normalize, phase_correlation_response
from .pose import ransac_position
from .rectify import rectify_to_map


@dataclass
class CoarseConfig:
    template_px: int = 128  # LVS: 128 x 128 coarse template
    n_templates: int = 9  # LVS: 5 per image x 3 images; single image uses up to 9
    norm_win: int = 21  # LVS 21 x 21 mean/std filter
    search_sigma: float = 3.0  # search radius = search_sigma * prior horizontal sigma (+ margin)
    search_margin_m: float = 300.0
    max_search_px: int = 1024  # LVS FFT map size is 1024; radius clipped to (max - template)/2
    min_peak_height: float = 0.15  # NCC; calibrated, decisions.md C7
    max_peak_width_px: float = 12.0  # FWHM, coarse-map px; decisions.md C7
    geom_threshold_map_px: float = 3.0  # RANSAC reprojection gate, in coarse-map pixels
    min_inliers: int = 2
    min_template_spacing: float = 0.75  # in template sizes
    supersample: int = 2


@dataclass
class CoarseResult:
    ok: bool
    position: np.ndarray | None
    R_wc: np.ndarray
    n_templates: int
    n_inliers: int
    templates: list = field(default_factory=list)  # raw per-template diagnostics
    health: dict = field(default_factory=dict)


def _footprint_box(cam: PinholeCamera, T: np.ndarray, m: TrnMap, h: float) -> tuple[int, int, int, int]:
    """Map pixel bounding box of the image footprint at terrain height h."""
    edge = np.linspace(-0.5, cam.width - 0.5, 9)
    cols = np.concatenate([edge, edge, np.full(9, -0.5), np.full(9, cam.width - 0.5)])
    rows = np.concatenate([np.full(9, -0.5), np.full(9, cam.height - 0.5), edge, edge])
    o, d = cam.rays_world(T, cols, rows)
    r = frames.MOON_RADIUS_M + h
    b = d @ o
    c = o @ o - r * r
    t = -b - np.sqrt(np.maximum(b * b - c, 0.0))
    P = o + t[:, None] * d
    x, y, _ = frames.body_to_stereo(P)
    mc, mr = m.xy_to_colrow(x, y)
    return int(np.floor(mc.min())), int(np.floor(mr.min())), int(np.ceil(mc.max())), int(np.ceil(mr.max()))


def select_templates(rect: np.ndarray, size: int, n: int, spacing: float, stride: int | None = None):
    """Top-``n`` fully valid windows by texture score (mean local gradient energy)."""
    import cv2

    stride = stride or max(size // 8, 4)
    valid = np.isfinite(rect)
    filled = np.where(valid, rect, 0.0).astype(np.float32)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    energy = (gx * gx + gy * gy)
    H, W = rect.shape
    cands = []
    inv = (~valid).astype(np.float32)
    ii_e = cv2.integral(energy)
    ii_v = cv2.integral(inv)
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            bad = ii_v[r + size, c + size] - ii_v[r, c + size] - ii_v[r + size, c] + ii_v[r, c]
            if bad > 0:
                continue
            e = ii_e[r + size, c + size] - ii_e[r, c + size] - ii_e[r + size, c] + ii_e[r, c]
            cands.append((float(e), r, c))
    cands.sort(reverse=True)
    chosen = []
    min_d = spacing * size
    for e, r, c in cands:
        if all(math.hypot(r - r2, c - c2) >= min_d for _, r2, c2 in chosen):
            chosen.append((e, r, c))
        if len(chosen) >= n:
            break
    return chosen


def coarse_localize(image: np.ndarray, cam: PinholeCamera, T_prior: np.ndarray, prior_sigma_h_m: float,
                    cmap: TrnMap, cfg: CoarseConfig | None = None, cmap_norm: np.ndarray | None = None,
                    rng: np.random.Generator | None = None) -> CoarseResult:
    cfg = cfg or CoarseConfig()
    R = T_prior[:3, :3]
    h_ref = float(np.nanmedian(cmap.heights))
    c_lo, r_lo, c_hi, r_hi = _footprint_box(cam, T_prior, cmap, h_ref)
    rows, cols = cmap.shape
    c_lo, r_lo = max(c_lo, 0), max(r_lo, 0)
    c_hi, r_hi = min(c_hi, cols - 1), min(r_hi, rows - 1)
    health: dict = {"footprint_map_box": [c_lo, r_lo, c_hi, r_hi]}
    if c_hi - c_lo < cfg.template_px or r_hi - r_lo < cfg.template_px:
        return CoarseResult(False, None, R, 0, 0, [], dict(health, reason="footprint_outside_map"))
    alt_prior = np.linalg.norm(T_prior[:3, 3]) - frames.MOON_RADIUS_M - h_ref
    gsd_img = alt_prior / cam.fx
    ss = cfg.supersample if cmap.pixel > 1.5 * gsd_img else 1
    rect = rectify_to_map(image, cam, T_prior, cmap, c_lo, r_lo, c_hi - c_lo + 1, r_hi - r_lo + 1, supersample=ss)
    rect_n, _ = local_normalize(rect, cfg.norm_win)
    rect_n = np.where(np.isfinite(rect), rect_n, np.nan)
    if cmap_norm is None:
        cmap_norm, _ = local_normalize(cmap.image, cfg.norm_win)
    chosen = select_templates(rect, cfg.template_px, cfg.n_templates, cfg.min_template_spacing)
    radius_px = int(math.ceil((cfg.search_sigma * prior_sigma_h_m + cfg.search_margin_m) / cmap.pixel))
    radius_px = min(radius_px, (cfg.max_search_px - cfg.template_px) // 2)
    health["search_radius_px"] = radius_px
    tpl_recs = []
    uv_list, X_list = [], []
    half = cfg.template_px // 2
    for score, r, c in chosen:
        tpl = rect_n[r:r + cfg.template_px, c:c + cfg.template_px]
        mc0, mr0 = c_lo + c, r_lo + r  # template top-left in map px under the prior
        sc0, sr0 = mc0 - radius_px, mr0 - radius_px
        sc1, sr1 = mc0 + cfg.template_px + radius_px, mr0 + cfg.template_px + radius_px
        rec = {"tpl_rc": [int(mr0), int(mc0)], "texture": score}
        if sc0 < 0 or sr0 < 0 or sc1 > cols or sr1 > rows:
            # Clip the search window to the map.
            sc0c, sr0c = max(sc0, 0), max(sr0, 0)
            sc1c, sr1c = min(sc1, cols), min(sr1, rows)
            rec["search_clipped"] = True
        else:
            sc0c, sr0c, sc1c, sr1c = sc0, sr0, sc1, sr1
        search = cmap_norm[sr0c:sr1c, sc0c:sc1c]
        if search.shape[0] < cfg.template_px + 2 or search.shape[1] < cfg.template_px + 2:
            rec["reject"] = "search_outside_map"
            tpl_recs.append(rec)
            continue
        surf = correlate(search, np.nan_to_num(tpl))
        pk = analyze_peak(surf, exclusion=5)
        # Offset of the match relative to the prior prediction (map px).
        mx = sc0c + pk.x - mc0
        my = sr0c + pk.y - mr0
        # Phase correlation response between the template and the map patch at the peak.
        pr_c, pr_r = sc0c + pk.ix, sr0c + pk.iy
        mpatch = cmap_norm[pr_r:pr_r + cfg.template_px, pr_c:pr_c + cfg.template_px]
        pcr = phase_correlation_response(np.nan_to_num(tpl), mpatch)[2] if mpatch.shape == tpl.shape else float("nan")
        rec.update({"peak": pk.height, "peak_fit": pk.fit_height, "width": pk.width, "width_min": pk.width_min,
                    "concave_down": pk.concave_down, "subpx": [pk.dx, pk.dy], "psr": pk.psr,
                    "second_ratio": pk.second_ratio, "phasecorr_response": pcr, "offset_px": [mx, my],
                    "on_border": pk.on_border})
        reason = None
        if pk.on_border:
            reason = "peak_on_border"
        elif not pk.concave_down or max(abs(pk.dx), abs(pk.dy)) > 1.0:
            reason = "peak_shape"
        elif pk.height < cfg.min_peak_height:
            reason = "peak_low"
        elif pk.width > cfg.max_peak_width_px:
            reason = "peak_wide"
        rec["reject"] = reason
        tpl_recs.append(rec)
        if reason is not None:
            continue
        # Landmark: template centre. Image pixel from the prior projection of the prior map
        # point; 3D point from the matched map location.
        tc, tr = mc0 + half - 0.5, mr0 + half - 0.5
        P_prior = cmap.world_points(np.array([tc]), np.array([tr]))
        u, v, _ = cam.project(T_prior, P_prior)
        X = cmap.world_points(np.array([tc + mx]), np.array([tr + my]))
        uv_list.append([u[0] + cam.pixel_center, v[0] + cam.pixel_center])
        X_list.append(X[0])
        rec["uv"] = uv_list[-1]
    n_ok = len(uv_list)
    health.update({"n_templates": len(chosen), "n_peak_ok": n_ok})
    if n_ok < cfg.min_inliers:
        return CoarseResult(False, None, R, len(chosen), 0, tpl_recs, dict(health, reason="too_few_matches"))
    uv = np.array(uv_list)
    X = np.array(X_list)
    gsd_map_to_img = cmap.pixel / max(gsd_img, 1e-6)
    thr = cfg.geom_threshold_map_px * gsd_map_to_img
    rs = ransac_position(uv, X, cam.K, R, threshold_px=thr, rng=rng)
    k = 0
    for rec in tpl_recs:
        if rec.get("reject") is None and "uv" in rec:
            rec["geom_inlier"] = bool(rs.inliers[k])
            rec["geom_residual_px"] = float(rs.residuals_px[k])
            k += 1
    n_in = int(rs.inliers.sum())
    health.update({"n_geom_inliers": n_in, "n_geom_outliers": n_ok - n_in, "geom_threshold_img_px": thr})
    ok = rs.position is not None and n_in >= cfg.min_inliers
    return CoarseResult(ok, rs.position, R, len(chosen), n_in, tpl_recs, health)
