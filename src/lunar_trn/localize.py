"""Single-image map-relative localisation: the integration interface (handoff section 8).

``localize(image, K, prior_T_world_cam, prior_cov, trn_map)`` runs coarse -> fine and
returns a :class:`LocalizationResult` with the status, pose, covariance, per-landmark data
and every raw health signal. No simulator dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import frames
from .camera import PinholeCamera
from .coarse import CoarseConfig, coarse_localize
from .fine import FineConfig, fine_localize
from .health import scene_signals
from .maps import TrnMap
from .matching import local_normalize


@dataclass
class MapSet:
    """Coarse and fine maps of the same area (coarse is typically 2x the fine pixel)."""

    coarse: TrnMap
    fine: TrnMap
    _norm: dict = field(default_factory=dict, repr=False)

    def normalized(self, which: str, win: int) -> np.ndarray:
        key = (which, win)
        if key not in self._norm:
            self._norm[key] = local_normalize(getattr(self, which).image, win)[0]
        return self._norm[key]


@dataclass
class LocalizerConfig:
    coarse: CoarseConfig = field(default_factory=CoarseConfig)
    fine: FineConfig = field(default_factory=FineConfig)
    fine_iterations: int = 2
    coarse_passes: int = 1  # 2 = re-run coarse from the pass-1 position with a small search radius
    coarse_refine_sigma_m: float = 100.0
    min_inliers_fix: int = 20  # Mars 2020 LVS position-update gate
    # Common-mode (map/image appearance) position error floor added to the covariance, in
    # fine-map pixels per axis; calibrated on the StereoLunar calibration split (decisions.md D7).
    cov_floor_h_px: float = 0.47
    cov_floor_v_px: float = 0.085
    skip_coarse: bool = False
    pixel_center: float = 0.5


@dataclass
class LocalizationResult:
    status: str  # "fix" or "no_fix"
    T_world_cam: np.ndarray
    cov: np.ndarray  # 6x6 (position world, attitude small-angle world)
    landmarks: dict
    health: dict
    stage: str  # last stage reached: "coarse", "fine" (T-I); "detect", "match", "pose" (P-G)
    family: str = "T-I"  # method family, handoff section 2.1


def localize(image: np.ndarray, K: np.ndarray, prior_T_world_cam: np.ndarray, prior_cov: np.ndarray,
             trn_map: MapSet, config: LocalizerConfig | None = None, seed: int = 0) -> LocalizationResult:
    cfg = config or LocalizerConfig()
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    H, W = image.shape
    cam = PinholeCamera(K, W, H, pixel_center=cfg.pixel_center)
    T = np.asarray(prior_T_world_cam, dtype=np.float64).copy()
    pos_prior = T[:3, 3].copy()
    # Horizontal prior sigma: RMS of the east and north standard deviations.
    P = np.asarray(prior_cov, float)
    enu = frames.enu_basis(pos_prior)
    Pp = enu.T @ P[:3, :3] @ enu
    sigma_h = float(np.sqrt(0.5 * (Pp[0, 0] + Pp[1, 1])))
    R_prior = T[:3, :3].copy()
    # Attitude prior (MAP pseudo-measurement in the fine refinement). A 3x3 covariance or a
    # zero attitude block means "attitude known": use a 1 microradian sigma.
    sigma_att = float(np.sqrt(np.mean(np.diag(P)[3:]))) if P.shape[0] >= 6 else 0.0
    sigma_att = max(sigma_att, 1e-6)
    h_ref = float(np.nanmedian(trn_map.fine.heights))
    gsd_img = (np.linalg.norm(pos_prior) - frames.MOON_RADIUS_M - h_ref) / cam.fx
    health: dict = {"scene": scene_signals(image, gsd_img, trn_map.fine.pixel),
                    "prior_sigma_h_m": sigma_h, "map_pixel_coarse": trn_map.coarse.pixel,
                    "map_pixel_fine": trn_map.fine.pixel}

    # Coarse.
    stage = "coarse"
    if not cfg.skip_coarse:
        cres = coarse_localize(image, cam, T, sigma_h, trn_map.coarse, cfg.coarse,
                               cmap_norm=trn_map.normalized("coarse", cfg.coarse.norm_win), rng=rng)
        health["coarse"] = dict(cres.health, ok=cres.ok, n_inliers=cres.n_inliers, templates=cres.templates)
        health["t_coarse_s"] = time.perf_counter() - t0
        if not cres.ok:
            return LocalizationResult("no_fix", T, np.full((6, 6), np.nan), {}, health, stage)
        T[:3, 3] = cres.position
        for _ in range(cfg.coarse_passes - 1):
            c2 = coarse_localize(image, cam, T, cfg.coarse_refine_sigma_m, trn_map.coarse, cfg.coarse,
                                 cmap_norm=trn_map.normalized("coarse", cfg.coarse.norm_win), rng=rng)
            health.setdefault("coarse_refine", []).append(dict(c2.health, ok=c2.ok, n_inliers=c2.n_inliers))
            if c2.ok and c2.n_inliers >= cres.n_inliers:
                T[:3, 3] = c2.position
        health["coarse"]["correction_m"] = (T[:3, 3] - pos_prior).tolist()

    # Fine (iterated: the second pass warps the image with the improved pose).
    stage = "fine"
    fres = None
    fine_hist = []
    for it in range(max(cfg.fine_iterations, 1)):
        trial = fine_localize(image, cam, T, trn_map.fine, cfg.fine,
                              fmap_norm=trn_map.normalized("fine", cfg.fine.norm_win), rng=rng,
                              att_prior_R=R_prior, att_prior_sigma_rad=sigma_att)
        fine_hist.append({"iteration": it, "ok": trial.ok, "n_inliers": trial.n_inliers,
                          "n_matched": trial.n_matched, "rms_px": trial.reproj_rms_px})
        # Keep the last successful pass; a failed re-pass does not undo a good one.
        if fres is None or (trial.ok and trial.n_inliers >= 0.8 * fres.n_inliers) or not fres.ok:
            fres = trial
        if not trial.ok:
            break
        T = np.eye(4)
        T[:3, :3] = fres.R_wc
        T[:3, 3] = fres.position
    health["t_total_s"] = time.perf_counter() - t0
    peaks = fres.landmarks.get("peak", np.array([])) if fres else np.array([])
    used = fres.landmarks.get("used", np.zeros(0, bool)) if fres else np.zeros(0, bool)
    inl = fres.landmarks.get("inlier", np.zeros(0, bool)) if fres else np.zeros(0, bool)
    health["fine"] = {
        "iterations": fine_hist,
        "n_features": fres.n_features if fres else 0,
        "n_matched": fres.n_matched if fres else 0,
        "n_used": fres.n_used if fres else 0,
        "n_inliers": fres.n_inliers if fres else 0,
        "inlier_ratio": (fres.n_inliers / fres.n_used) if fres and fres.n_used else 0.0,
        "reproj_rms_px": fres.reproj_rms_px if fres else float("nan"),
        "sigma0": fres.sigma0 if fres else float("nan"),
        "reject_counts": fres.reject_counts if fres else {},
        "peak_height_used_median": float(np.nanmedian(peaks[used])) if used.any() else float("nan"),
        "peak_height_inlier_median": float(np.nanmedian(peaks[inl])) if inl.any() else float("nan"),
        **(fres.health if fres else {}),
    }
    ok = fres is not None and fres.ok and fres.n_inliers >= cfg.min_inliers_fix
    cov = fres.cov.copy() if fres is not None else np.full((6, 6), np.nan)
    if fres is not None and np.all(np.isfinite(cov)):
        health["cov_landmark_only"] = cov[:3, :3].tolist()
        enu_est = frames.enu_basis(T[:3, 3])
        sh = cfg.cov_floor_h_px * trn_map.fine.pixel
        sv = cfg.cov_floor_v_px * trn_map.fine.pixel
        cov[:3, :3] += enu_est @ np.diag([sh**2, sh**2, sv**2]) @ enu_est.T
    health["correction_from_prior_m"] = (T[:3, 3] - pos_prior).tolist()
    return LocalizationResult("fix" if ok else "no_fix", T, cov, fres.landmarks if fres else {}, health, stage)
