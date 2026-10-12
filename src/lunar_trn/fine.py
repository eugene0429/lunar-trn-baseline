"""Fine landmark matching and pose estimation (LVS fine phase, single image).

1. Select up to ``n_features`` Harris corners (5 x 5) in the image, one per subwindow.
2. Warp the image into the fine map frame with the current pose and cut a 21 x 21
   template around each feature's predicted map location.
3. Raster-scan each template over a 128 x 128 map search area (NCC), fit a biquadratic to
   the 3 x 3 peak neighbourhood, and apply the peak tests (shape, sub-pixel offset, height,
   width).
4. Keep at most ``max_landmarks`` matches (best peaks), estimate the pose with a
   position-only RANSAC followed by 6-DoF refinement, then apply the LVS adaptive
   Mahalanobis gate and refine again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import frames
from .camera import PinholeCamera
from .coarse import _footprint_box
from .maps import TrnMap
from .matching import analyze_peak, correlate, local_normalize
from .pose import mahalanobis_gate, ransac_position, refine_pose
from .rectify import rectify_to_map

REJECT_REASONS = ("peak_on_border", "peak_shape", "peak_low", "peak_wide", "geometric", "mahalanobis",
                  "outside_map", "no_template")


@dataclass
class FineConfig:
    n_features: int = 150  # LVS fine N = 150
    cell_px: int = 32  # subwindow size for feature selection (64 px on 1024 px LVS images)
    harris_block: int = 5  # LVS 5 x 5 Harris
    harris_k: float = 0.04
    template_px: int = 21  # LVS 21 x 21 template
    search_px: int = 128  # LVS 128 x 128 search area
    norm_win: int = 21
    min_peak_height: float = 0.60  # NCC; calibrated, decisions.md D4
    max_peak_width_px: float = 12.0  # FWHM in map px; calibrated, decisions.md D4
    max_landmarks: int = 100  # LVS estimator input limit
    ransac_threshold_map_px: float = 2.0
    ransac_min_px: float = 1.5
    sigma_map_px: float = 0.5  # assumed 1-sigma match noise (map px) before a-posteriori scaling
    mahalanobis_alpha: float = 1.5
    supersample: int = 2


@dataclass
class FineResult:
    ok: bool
    R_wc: np.ndarray
    position: np.ndarray
    cov: np.ndarray
    n_features: int
    n_matched: int  # passed the peak tests
    n_used: int  # entered the estimator (<= max_landmarks)
    n_inliers: int
    reproj_rms_px: float
    sigma0: float
    landmarks: dict = field(default_factory=dict)  # arrays of per-landmark raw values
    reject_counts: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)


def select_features(image: np.ndarray, cfg: FineConfig, border: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best Harris response per ``cell_px`` subwindow; returns cols, rows, responses (top N)."""
    import cv2

    img = np.asarray(image, dtype=np.float32)
    R = cv2.cornerHarris(img, cfg.harris_block, 3, cfg.harris_k)
    H, W = img.shape
    cols, rows, resp = [], [], []
    for r0 in range(border, H - border, cfg.cell_px):
        for c0 in range(border, W - border, cfg.cell_px):
            r1, c1 = min(r0 + cfg.cell_px, H - border), min(c0 + cfg.cell_px, W - border)
            if r1 <= r0 or c1 <= c0:
                continue
            blk = R[r0:r1, c0:c1]
            k = int(np.argmax(blk))
            rr, cc = np.unravel_index(k, blk.shape)
            if blk[rr, cc] > 0:
                cols.append(c0 + cc)
                rows.append(r0 + rr)
                resp.append(float(blk[rr, cc]))
    order = np.argsort(resp)[::-1][: cfg.n_features]
    return np.array(cols)[order], np.array(rows)[order], np.array(resp)[order]


def fine_localize(image: np.ndarray, cam: PinholeCamera, T_init: np.ndarray, fmap: TrnMap,
                  cfg: FineConfig | None = None, fmap_norm: np.ndarray | None = None,
                  rng: np.random.Generator | None = None, att_prior_R: np.ndarray | None = None,
                  att_prior_sigma_rad: float | None = None) -> FineResult:
    cfg = cfg or FineConfig()
    R0 = T_init[:3, :3]
    c0 = T_init[:3, 3]
    rows_m, cols_m = fmap.shape
    h_ref = float(np.nanmedian(fmap.heights))
    alt = float(np.linalg.norm(c0) - frames.MOON_RADIUS_M - h_ref)
    gsd_img = alt / cam.fx
    map_to_img = fmap.pixel / max(gsd_img, 1e-9)
    counts = {k: 0 for k in REJECT_REASONS}
    empty = FineResult(False, R0, c0, np.full((6, 6), np.nan), 0, 0, 0, 0, float("nan"), float("nan"),
                       {}, counts, {"gsd_img_m": gsd_img, "map_to_img": map_to_img})

    # Warp the footprint into the map frame once.
    b_lo_c, b_lo_r, b_hi_c, b_hi_r = _footprint_box(cam, T_init, fmap, h_ref)
    pad = cfg.template_px
    b_lo_c, b_lo_r = max(b_lo_c - pad, 0), max(b_lo_r - pad, 0)
    b_hi_c, b_hi_r = min(b_hi_c + pad, cols_m - 1), min(b_hi_r + pad, rows_m - 1)
    if b_hi_c - b_lo_c < cfg.template_px or b_hi_r - b_lo_r < cfg.template_px:
        empty.health["reason"] = "footprint_outside_map"
        return empty
    ss = cfg.supersample if map_to_img > 1.5 else 1
    rect = rectify_to_map(image, cam, T_init, fmap, b_lo_c, b_lo_r, b_hi_c - b_lo_c + 1, b_hi_r - b_lo_r + 1,
                          supersample=ss)
    rect_n, _ = local_normalize(rect, cfg.norm_win)
    if fmap_norm is None:
        fmap_norm, _ = local_normalize(fmap.image, cfg.norm_win)

    # Feature selection; keep features whose template fits inside the image (in map scale).
    border = int(np.ceil(0.5 * cfg.template_px * max(map_to_img, 1.0))) + 2
    fc, fr, fresp = select_features(image, cfg, border)
    n_feat = int(fc.size)
    if n_feat == 0:
        empty.health["reason"] = "no_features"
        return empty
    from .rectify import predicted_map_location

    mc, mr = predicted_map_location(cam, T_init, fmap, fc.astype(float), fr.astype(float))
    half_t = cfg.template_px // 2
    half_s = cfg.search_px // 2
    recs = {k: [] for k in ("feat_col", "feat_row", "harris", "peak", "peak_fit", "width", "width_min", "subpx_dx",
                            "subpx_dy", "psr", "second_ratio", "offset_dx", "offset_dy", "u", "v", "X", "status")}
    for i in range(n_feat):
        status = None
        if not (np.isfinite(mc[i]) and np.isfinite(mr[i])):
            mc[i] = mr[i] = -1e6  # ray missed the terrain shell; handled as outside_map below
        cc, rc = int(round(mc[i])), int(round(mr[i]))
        tr0, tc0 = rc - half_t - b_lo_r, cc - half_t - b_lo_c
        tpl = None
        if np.isfinite(mc[i]) and tr0 >= 0 and tc0 >= 0 and tr0 + cfg.template_px <= rect.shape[0] \
                and tc0 + cfg.template_px <= rect.shape[1]:
            raw = rect[tr0:tr0 + cfg.template_px, tc0:tc0 + cfg.template_px]
            if np.isfinite(raw).all():
                tpl = rect_n[tr0:tr0 + cfg.template_px, tc0:tc0 + cfg.template_px]
        sr0, sc0 = rc - half_s, cc - half_s
        pk = None
        if tpl is None:
            status = "no_template"
        elif sr0 < 0 or sc0 < 0 or sr0 + cfg.search_px > rows_m or sc0 + cfg.search_px > cols_m:
            status = "outside_map"
        else:
            surf = correlate(fmap_norm[sr0:sr0 + cfg.search_px, sc0:sc0 + cfg.search_px], tpl)
            pk = analyze_peak(surf, exclusion=3)
            if pk.on_border:
                status = "peak_on_border"
            elif not pk.concave_down or max(abs(pk.dx), abs(pk.dy)) > 1.0:
                status = "peak_shape"
            elif pk.height < cfg.min_peak_height:
                status = "peak_low"
            elif pk.width > cfg.max_peak_width_px:
                status = "peak_wide"
        # Landmark geometry: template centre (integer map px) seen at its prior projection.
        if pk is not None:
            mx = sc0 + pk.x + half_t - cc
            my = sr0 + pk.y + half_t - rc
            Pp = fmap.world_points(np.array([float(cc)]), np.array([float(rc)]))
            u, v, _ = cam.project(T_init, Pp)
            X = fmap.world_points(np.array([cc + mx]), np.array([rc + my]))[0]
            vals = (pk.height, pk.fit_height, pk.width, pk.width_min, pk.dx, pk.dy, pk.psr, pk.second_ratio, mx, my,
                    u[0] + cam.pixel_center, v[0] + cam.pixel_center)
        else:
            X = np.full(3, np.nan)
            vals = (np.nan,) * 12
        for k, val in zip(("peak", "peak_fit", "width", "width_min", "subpx_dx", "subpx_dy", "psr", "second_ratio",
                           "offset_dx", "offset_dy", "u", "v"), vals):
            recs[k].append(float(val))
        recs["feat_col"].append(int(fc[i]))
        recs["feat_row"].append(int(fr[i]))
        recs["harris"].append(float(fresp[i]))
        recs["X"].append(X)
        recs["status"].append(status)
        if status is not None:
            counts[status] += 1

    status = np.array([s if s is not None else "" for s in recs["status"]], dtype=object)
    good = np.nonzero(status == "")[0]
    n_matched = int(good.size)
    # Estimator input: best peaks first, at most max_landmarks.
    peaks = np.array(recs["peak"])
    good = good[np.argsort(-peaks[good])][: cfg.max_landmarks]
    n_used = int(good.size)
    landmarks = {k: (np.array(v) if k != "X" else np.array(v)) for k, v in recs.items()}
    landmarks["used"] = np.zeros(n_feat, bool)
    landmarks["used"][good] = True
    landmarks["inlier"] = np.zeros(n_feat, bool)
    health = {"gsd_img_m": gsd_img, "map_to_img": map_to_img, "n_features": n_feat}
    if n_used < 3:
        return FineResult(False, R0, c0, np.full((6, 6), np.nan), n_feat, n_matched, n_used, 0, float("nan"),
                          float("nan"), landmarks, counts, dict(health, reason="too_few_matches"))
    uv = np.column_stack([np.array(recs["u"])[good], np.array(recs["v"])[good]])
    X = np.array(recs["X"])[good]
    thr = max(cfg.ransac_threshold_map_px * map_to_img, cfg.ransac_min_px)
    rs = ransac_position(uv, X, cam.K, R0, threshold_px=thr, rng=rng, max_iter=600)
    if rs.position is None or rs.inliers.sum() < 3:
        counts["geometric"] += n_used
        return FineResult(False, R0, c0, np.full((6, 6), np.nan), n_feat, n_matched, n_used, 0, float("nan"),
                          float("nan"), landmarks, counts, dict(health, reason="ransac_failed"))
    inl = rs.inliers.copy()
    counts["geometric"] += int((~inl).sum())
    sigma = cfg.sigma_map_px * max(map_to_img, 1e-6)
    ref = refine_pose(uv[inl], X[inl], cam.K, R0, rs.position, sigma_px=sigma, att_prior_R=att_prior_R,
                      att_prior_sigma_rad=att_prior_sigma_rad)
    keep, gamma, d2 = mahalanobis_gate(ref.residuals, sigma, alpha=cfg.mahalanobis_alpha)
    idx_inl = np.nonzero(inl)[0]
    n_mahal = int((~keep).sum())
    counts["mahalanobis"] += n_mahal
    if n_mahal > 0 and keep.sum() >= 3:
        inl[idx_inl[~keep]] = False
        ref = refine_pose(uv[inl], X[inl], cam.K, ref.R_wc, ref.position, sigma_px=sigma, att_prior_R=att_prior_R,
                          att_prior_sigma_rad=att_prior_sigma_rad)
    landmarks["inlier"][good[inl]] = True
    res_all = np.full((n_feat, 2), np.nan)
    res_all[good[inl]] = ref.residuals
    landmarks["residual"] = res_all
    health.update({"ransac_threshold_px": thr, "mahalanobis_gamma": gamma, "sigma_px_assumed": sigma,
                   "ransac_inliers": int(rs.inliers.sum())})
    return FineResult(True, ref.R_wc, ref.position, ref.cov, n_feat, n_matched, n_used, int(inl.sum()),
                      ref.rms_px, ref.sigma0, landmarks, counts, health)
