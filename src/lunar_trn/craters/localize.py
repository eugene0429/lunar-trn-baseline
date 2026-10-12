"""P-G crater pattern matching: single-image localisation in tracking mode.

``localize_craters(image, K, prior_T_world_cam, prior_cov, crater_map)`` detects crater
ellipses (CNav style), predicts catalog rims through the prior pose, votes horizontal
position hypotheses, verifies the strongest ones (gated assignment, fixed-attitude position
RANSAC, perspective-centre-bias correction, the shared 6-DoF refinement with the LVS
Mahalanobis gate) and gates the fix on the inlier count. It returns the T-I
:class:`~lunar_trn.localize.LocalizationResult` with ``family = "P-G"`` and raw signals
under ``health["pg"]``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import frames
from ..camera import PinholeCamera
from ..health import scene_signals
from ..localize import LocalizationResult
from ..pose import mahalanobis_gate, project_uv, ransac_position, refine_pose
from .catalog import CraterCatalog
from .conics import project_rims
from .detect import DetectorConfig, Detections, detect_craters, pinhole_sun_fn
from .match import Candidates, MatchConfig, assign, predict, select_candidates, vote_positions


@dataclass
class CraterMap:
    """Onboard P-G map: a Sun-independent crater catalog and the Sun direction (ephemeris)."""

    catalog: CraterCatalog
    sun_dir_world: np.ndarray


@dataclass
class CraterConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    refine_passes: int = 2
    min_inliers_fix: int = 5  # decisions G12 (CNav: at least five catalog craters)
    min_match_ratio: float = 0.0  # decisions G12: inliers / catalog craters in view (SLIM-style), calibrated
    max_sigma0: float = float("inf")  # decisions G12: a-posteriori unit-weight sigma, calibrated
    cov_floor_h_px: float = 0.0  # decisions G13, image-GSD pixels per horizontal axis
    cov_floor_v_px: float = 0.0
    pixel_center: float = 0.5


def _in_image(pred: dict, cam: PinholeCamera, margin: float) -> np.ndarray:
    lo = cam.pixel_center - 0.5 + margin
    with np.errstate(invalid="ignore"):
        return (pred["valid"] & (pred["u"] - pred["A"] >= lo) & (pred["u"] + pred["A"] <= cam.width + lo - 2 * margin)
                & (pred["v"] - pred["A"] >= lo) & (pred["v"] + pred["A"] <= cam.height + lo - 2 * margin))


def _depth_guess(p0: np.ndarray, R0: np.ndarray) -> float:
    up = p0 / np.linalg.norm(p0)
    alt = np.linalg.norm(p0) - frames.MOON_RADIUS_M
    return max(alt, 1.0) / max(-(R0[:, 2] @ up), 0.2)


def _refine(det: Detections, i, cat: CraterCatalog, keep, R, c, K, R0, sigma_att, cfg: CraterConfig):
    """Bias-corrected refinement of the pairs (det[i], cat) with the LVS Mahalanobis gate."""
    sig = np.maximum(cfg.match.sigma_min_px, cfg.match.sigma_rel * det.D_px[i])
    X = cat.X
    keep = keep.copy()
    gamma, rf, pr, uvc = float("nan"), None, None, None
    for _ in range(max(cfg.refine_passes, 1)):
        if keep.sum() < 3:
            break
        pr = project_rims(K, R, c, X, cat.e1, cat.e2, cat.a, cat.b, cat.psi)
        uvc = det.uv[i] - np.nan_to_num(pr["bias"])
        rf = refine_pose(uvc[keep], X[keep], K, R, c, sigma_px=sig[keep], att_prior_R=R0,
                         att_prior_sigma_rad=sigma_att)
        R, c = rf.R_wc, rf.position
        uvp, z = project_uv(K, R, c, X)
        res = uvp - uvc
        _, gamma, _ = mahalanobis_gate(res[keep], sig[keep])
        d2 = np.sum((res / sig[:, None]) ** 2, axis=1)
        keep = np.isfinite(d2) & (z > 0) & (d2 < gamma)
    if rf is None or keep.sum() < 3:
        return None
    pr = project_rims(K, R, c, X, cat.e1, cat.e2, cat.a, cat.b, cat.psi)
    uvc = det.uv[i] - np.nan_to_num(pr["bias"])
    rf = refine_pose(uvc[keep], X[keep], K, R, c, sigma_px=sig[keep], att_prior_R=R0, att_prior_sigma_rad=sigma_att)
    R, c = rf.R_wc, rf.position
    uvp, _ = project_uv(K, R, c, X)
    return {"R": R, "c": c, "cov": rf.cov, "keep": keep, "res": uvp - uvc, "bias": pr["bias"], "rms": rf.rms_px,
            "sigma0": rf.sigma0, "gamma": gamma, "n_inliers": int(keep.sum())}


def _verify(offset, det: Detections, cand: Candidates, cam: PinholeCamera, p0, enu, R0, sigma_att,
            cfg: CraterConfig, rng) -> dict:
    m = cfg.match
    rec = {"offset_m": [float(offset[0]), float(offset[1])], "n_assigned1": 0, "n_ransac1": 0, "n_inliers": 0,
           "state": None}
    c = p0 + enu[:, 0] * offset[0] + enu[:, 1] * offset[1]
    pred = predict(cand.catalog, cam.K, R0, c)
    vis = np.nonzero(_in_image(pred, cam, m.image_margin_px))[0]
    if vis.size < 2:
        return rec
    i, jv, _ = assign(det.uv, det.D_px, np.column_stack([pred["u"][vis], pred["v"][vis]]), pred["D_px"][vis],
                      m.gate1_px, m.gate1_rel, m.size_ratio_gate)
    j = vis[jv]
    rec["n_assigned1"] = int(i.size)
    if i.size < 2:
        return rec
    rr = ransac_position(det.uv[i] - np.nan_to_num(pred["bias"][j]), cand.catalog.X[j], cam.K, R0, m.ransac_px,
                         rng=rng)
    if rr.position is None:
        return rec
    rec["n_ransac1"] = int(rr.inliers.sum())
    best = _refine(det, i, cand.catalog.subset(j), rr.inliers, R0, rr.position, cam.K, R0, sigma_att, cfg)
    if best is None:
        return rec
    best.update(i=i, j=j)
    # Second assignment from the refined pose with a tighter gate (LVS: a second fine pass).
    pred2 = predict(cand.catalog, cam.K, best["R"], best["c"])
    vis2 = np.nonzero(_in_image(pred2, cam, m.image_margin_px))[0]
    i2, jv2, _ = assign(det.uv, det.D_px, np.column_stack([pred2["u"][vis2], pred2["v"][vis2]]),
                        pred2["D_px"][vis2], m.gate2_px, m.gate2_rel, m.size_ratio_gate)
    j2 = vis2[jv2]
    if i2.size >= 3:
        rr2 = ransac_position(det.uv[i2] - np.nan_to_num(pred2["bias"][j2]), cand.catalog.X[j2], cam.K, best["R"],
                              m.ransac_px, rng=rng)
        if rr2.position is not None:
            st2 = _refine(det, i2, cand.catalog.subset(j2), rr2.inliers, best["R"], rr2.position, cam.K, R0,
                          sigma_att, cfg)
            if st2 is not None and st2["n_inliers"] >= best["n_inliers"]:
                st2.update(i=i2, j=j2)
                best = st2
    rec.update(n_inliers=best["n_inliers"], state=best)
    return rec


def _landmarks(det: Detections, best: dict | None, cand: Candidates) -> dict:
    n = len(det)
    nan = np.full(n, np.nan)
    lm = {"u": det.u, "v": det.v, "A": det.A, "B": det.B, "theta": det.theta, "D_px": det.D_px,
          "area_dark": det.area_dark, "area_bright": det.area_bright, "pair_dev_deg": det.pair_dev_deg,
          "cat_id": np.full(n, -1, np.int64), "cat_D_m": nan.copy(), "cat_d_over_D": nan.copy(),
          "cat_completeness": nan.copy(), "bias_u": nan.copy(), "bias_v": nan.copy(), "res_u": nan.copy(),
          "res_v": nan.copy(), "used": np.zeros(n, bool), "inlier": np.zeros(n, bool),
          "status": np.array(["unassigned"] * n, dtype=object)}
    if best is not None:
        i, j, keep = best["i"], best["j"], best["keep"]
        cat = cand.catalog
        lm["cat_id"][i] = cat.id[j]
        lm["cat_D_m"][i] = cat.D[j]
        lm["cat_d_over_D"][i] = cat.d_over_D[j]
        lm["cat_completeness"][i] = cat.completeness[j]
        lm["bias_u"][i], lm["bias_v"][i] = best["bias"][:, 0], best["bias"][:, 1]
        lm["res_u"][i], lm["res_v"][i] = best["res"][:, 0], best["res"][:, 1]
        lm["used"][i] = True
        lm["inlier"][i[keep]] = True
        lm["status"][i] = "gated"
        lm["status"][i[keep]] = "inlier"
    return lm


def localize_craters(image: np.ndarray, K: np.ndarray, prior_T_world_cam: np.ndarray, prior_cov: np.ndarray,
                     crater_map: CraterMap, config: CraterConfig | None = None, seed: int = 0,
                     detections: Detections | None = None) -> LocalizationResult:
    """Localise one image against a crater catalog. ``detections`` replaces the detector (tests, oracle runs)."""
    cfg = config or CraterConfig()
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    H, W = np.asarray(image).shape
    cam = PinholeCamera(K, W, H, pixel_center=cfg.pixel_center)
    T = np.asarray(prior_T_world_cam, float).copy()
    p0, R0 = T[:3, 3].copy(), T[:3, :3].copy()
    P = np.asarray(prior_cov, float)
    enu = frames.enu_basis(p0)
    Pp = enu.T @ P[:3, :3] @ enu
    sigma_h = float(np.sqrt(0.5 * (Pp[0, 0] + Pp[1, 1])))
    sigma_att = max(float(np.sqrt(np.mean(np.diag(P)[3:]))) if P.shape[0] >= 6 else 0.0, 1e-6)
    dcfg = cfg.detector
    cand = select_candidates(crater_map.catalog, cam, T, sigma_h, crater_map.sun_dir_world,
                             (dcfg.D_min_px, dcfg.D_max_px), cfg.match)
    z0 = float(np.nanmedian(cand.pred["z"])) if len(cand.catalog) else _depth_guess(p0, R0)
    gsd = z0 / cam.fx
    pg: dict = {"n_catalog_window": cand.n_window, "n_catalog_candidates": len(cand.catalog), "gsd_prior_m": gsd}
    health: dict = {"scene": scene_signals(image, gsd, None), "prior_sigma_h_m": sigma_h, "pg": pg}
    nan6 = np.full((6, 6), np.nan)

    def finish(status, T_out, cov, landmarks, stage):
        health["t_total_s"] = time.perf_counter() - t0
        health["correction_from_prior_m"] = (T_out[:3, 3] - p0).tolist()
        return LocalizationResult(status, T_out, cov, landmarks, health, stage, family="P-G")

    t1 = time.perf_counter()
    if detections is None:
        det, dh = detect_craters(image, pinhole_sun_fn(cam, T, crater_map.sun_dir_world, z0), dcfg,
                                 pixel_center=cfg.pixel_center)
    else:
        det, dh = detections, {"n_detections": len(detections), "injected": True}
    pg.update(dh)
    pg["t_detect_s"] = time.perf_counter() - t1
    if len(det) < 2:
        return finish("no_fix", T, nan6, {}, "detect")
    if len(cand.catalog) < 2:
        return finish("no_fix", T, nan6, _landmarks(det, None, cand), "match")
    t2 = time.perf_counter()
    votes = vote_positions(det, cand, cam, T, sigma_h, cfg.match)
    vv = votes["votes"]
    pg.update(vote_peak=float(vv[0]) if vv.size else 0.0, vote_second=float(vv[1]) if vv.size > 1 else 0.0,
              n_vote_pairs=votes["n_pairs"], vote_bin_m=votes["bin_m"])
    pg["vote_ratio"] = pg["vote_second"] / pg["vote_peak"] if pg["vote_peak"] > 0 else float("nan")
    hyps = [_verify(o, det, cand, cam, p0, enu, R0, sigma_att, cfg, rng) for o in votes["offsets"]]
    pg["hypotheses"] = [{**{k: v for k, v in h.items() if k != "state"}, "votes": float(vt), "count": int(ct)}
                        for h, vt, ct in zip(hyps, votes["votes"], votes["counts"])]
    pg["t_match_s"] = time.perf_counter() - t2
    good = sorted((h for h in hyps if h["state"] is not None), key=lambda h: -h["n_inliers"])
    if not good:
        return finish("no_fix", T, nan6, _landmarks(det, None, cand), "match")
    best = good[0]["state"]
    pg["hyp_inlier_margin"] = good[0]["n_inliers"] - (good[1]["n_inliers"] if len(good) > 1 else 0)
    T_est = np.eye(4)
    T_est[:3, :3], T_est[:3, 3] = best["R"], best["c"]
    pred_f = predict(cand.catalog, cam.K, best["R"], best["c"])
    with np.errstate(invalid="ignore"):
        vis_f = _in_image(pred_f, cam, 0.0) & (pred_f["D_px"] >= dcfg.D_min_px) & (pred_f["D_px"] <= dcfg.D_max_px)
    n_inl, n_ass, n_view = best["n_inliers"], int(len(best["i"])), int(vis_f.sum())
    pg.update(n_assigned=n_ass, n_inliers=n_inl, inlier_ratio=n_inl / max(n_ass, 1),
              match_ratio=n_inl / max(len(det), 1), n_catalog_view=n_view,
              catalog_match_ratio=n_inl / max(n_view, 1), reproj_rms_px=best["rms"], sigma0=best["sigma0"],
              mahalanobis_gamma=best["gamma"], reject_counts={"unassigned": len(det) - n_ass, "gated": n_ass - n_inl})
    cov = np.asarray(best["cov"], float).copy()
    if np.all(np.isfinite(cov)):
        health["cov_landmark_only"] = cov[:3, :3].tolist()
        enu_e = frames.enu_basis(best["c"])
        gsd_e = float(np.nanmedian(pred_f["z"][vis_f])) / cam.fx if vis_f.any() else gsd
        sh, sv = cfg.cov_floor_h_px * gsd_e, cfg.cov_floor_v_px * gsd_e
        cov[:3, :3] += enu_e @ np.diag([sh**2, sh**2, sv**2]) @ enu_e.T
    gate_ok = (n_inl >= cfg.min_inliers_fix and pg["catalog_match_ratio"] >= cfg.min_match_ratio
               and best["sigma0"] <= cfg.max_sigma0)
    return finish("fix" if gate_ok else "no_fix", T_est, cov, _landmarks(det, best, cand), "pose")
