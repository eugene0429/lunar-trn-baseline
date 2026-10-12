"""Ground-truth annotation of crater detections (calibration and diagnostics only).

Catalog rims are projected with the true pose; a detection corresponds to the nearest
predicted ellipse centre within max(tol_abs_px, tol_rel * D_px) whose size ratio is within
``size_ratio_max``. Errors are prediction minus detection, in pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catalog import CraterCatalog
from .conics import project_rims
from .detect import Detections


def project_catalog(catalog: CraterCatalog, cam, T_world_cam: np.ndarray, margin_px: float = 0.0) -> dict:
    T = np.asarray(T_world_cam, float)
    p = project_rims(cam.K, T[:3, :3], T[:3, 3], catalog.X, catalog.e1, catalog.e2, catalog.a, catalog.b,
                     catalog.psi)
    lo = cam.pixel_center - 0.5 + margin_px
    with np.errstate(invalid="ignore"):
        p["in_image"] = (p["valid"] & (p["u"] - p["A"] >= lo) & (p["u"] + p["A"] <= cam.width + lo - 2 * margin_px)
                         & (p["v"] - p["A"] >= lo) & (p["v"] + p["A"] <= cam.height + lo - 2 * margin_px))
    return p


def _nearest_ok(query_uv, query_D, ref_uv, ref_D, tol_abs_px, tol_rel, size_ratio_max):
    """Index into ref of the nearest compatible neighbour of each query point (-1 if none)."""
    n = len(query_uv)
    out = np.full(n, -1, np.int64)
    if n == 0 or len(ref_uv) == 0:
        return out
    k = min(8, len(ref_uv))
    dist, nb = cKDTree(ref_uv).query(query_uv, k=k)
    dist = np.asarray(dist).reshape(n, k)
    nb = np.asarray(nb).reshape(n, k)
    tol = np.maximum(tol_abs_px, tol_rel * query_D)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(ref_D[nb] / query_D[:, None]))
    ok = (dist <= tol) & (lr <= np.log(size_ratio_max))
    has = ok.any(axis=1)
    out[has] = nb[has, np.argmax(ok[has], axis=1)]
    return out


def annotate_detections(det: Detections, catalog: CraterCatalog, cam, T_true: np.ndarray, sun_fn,
                        tol_abs_px: float = 2.0, tol_rel: float = 0.25, size_ratio_max: float = 1.5) -> dict:
    n = len(det)
    nan = np.full(n, np.nan)
    out = {"cat_index": np.full(n, -1, np.int64), "cat_id": np.full(n, -1, np.int64), "err_u": nan.copy(),
           "err_v": nan.copy(), "err_antisun_px": nan.copy(), "err_perp_px": nan.copy(), "size_ratio": nan.copy(),
           "D_px_pred": nan.copy(), "anti_u": nan.copy(), "anti_v": nan.copy()}
    if n == 0 or len(catalog) == 0:
        return out
    p = project_catalog(catalog, cam, T_true)
    cand = np.nonzero(p["valid"])[0]
    if cand.size == 0:
        return out
    ref = _nearest_ok(det.uv, det.D_px, np.column_stack([p["u"][cand], p["v"][cand]]), p["D_px"][cand],
                      tol_abs_px, tol_rel, size_ratio_max)
    has = ref >= 0
    idx = cand[ref[has]]
    out["cat_index"][has] = idx
    out["cat_id"][has] = catalog.id[idx]
    out["err_u"][has] = p["u"][idx] - det.u[has]
    out["err_v"][has] = p["v"][idx] - det.v[has]
    out["size_ratio"][has] = p["D_px"][idx] / det.D_px[has]
    out["D_px_pred"][has] = p["D_px"][idx]
    anti = -sun_fn(det.u, det.v)
    out["anti_u"], out["anti_v"] = anti[:, 0], anti[:, 1]
    out["err_antisun_px"] = out["err_u"] * anti[:, 0] + out["err_v"] * anti[:, 1]
    out["err_perp_px"] = -out["err_u"] * anti[:, 1] + out["err_v"] * anti[:, 0]
    return out


def catalog_recall(det: Detections, catalog: CraterCatalog, cam, T_true: np.ndarray, sun_dir_world,
                   tol_abs_px: float = 2.0, tol_rel: float = 0.25, size_ratio_max: float = 1.5,
                   D_px_min: float = 4.0) -> dict:
    p = project_catalog(catalog, cam, T_true)
    sel = np.nonzero(p["in_image"] & (p["D_px"] >= D_px_min))[0]
    D = p["D_px"][sel]
    uv = np.column_stack([p["u"][sel], p["v"][sel]])
    detected = _nearest_ok(uv, D, det.uv, det.D_px, tol_abs_px, tol_rel, size_ratio_max) >= 0
    s = np.asarray(sun_dir_world, float)
    s = s / np.linalg.norm(s)
    return {"cat_id": catalog.id[sel], "D_px": D, "D_m": catalog.D[sel], "d_over_D": catalog.d_over_D[sel],
            "completeness": catalog.completeness[sel],
            "sun_el_deg": np.degrees(np.arcsin(np.clip(catalog.normal[sel] @ s, -1, 1))), "detected": detected}
