"""Image and scene health signals (handoff section 7). Raw values only, no verdicts."""

from __future__ import annotations

import numpy as np


def histogram_entropy(image: np.ndarray, bins: int = 256) -> float:
    """LVS image entropy E = -sum p log2 p over the grey-level histogram (bits)."""
    a = np.asarray(image, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    hist, _ = np.histogram(np.clip(a, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    p = hist[hist > 0] / a.size
    return float(-(p * np.log2(p)).sum())


def gradient_covariance(image: np.ndarray) -> dict:
    """Covariance of pixel image gradients (Setterfield et al. 2021, Sigma_grad).

    Central differences on the image in its native units ([0, 1] for StereoLunar JPGs).
    Returns lambda_min, lambda_max and tr(Sigma^-1).
    """
    a = np.asarray(image, dtype=np.float64)
    gy, gx = np.gradient(a)
    m = np.isfinite(gx) & np.isfinite(gy)
    gx, gy = gx[m], gy[m]
    S = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    lam = np.linalg.eigvalsh(S)
    try:
        tr_inv = float(np.trace(np.linalg.inv(S)))
    except np.linalg.LinAlgError:
        tr_inv = float("inf")
    return {"grad_cov_lambda_min": float(lam[0]), "grad_cov_lambda_max": float(lam[1]), "grad_cov_tr_inv": tr_inv}


def scene_signals(image: np.ndarray, gsd_img_m: float | None = None, map_pixel_m: float | None = None) -> dict:
    out = {"entropy_bits": histogram_entropy(image)}
    out.update(gradient_covariance(image))
    a = np.asarray(image, dtype=np.float64)
    out["dark_fraction_002"] = float(np.mean(a < 0.02))
    out["dark_fraction_005"] = float(np.mean(a < 0.05))
    out["mean_intensity"] = float(np.nanmean(a))
    if gsd_img_m is not None and map_pixel_m:
        out["gsd_ratio_cam_over_map"] = float(gsd_img_m / map_pixel_m)
    return out
