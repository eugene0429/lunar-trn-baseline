"""Warp a descent image into the map frame under a given pose (LVS 'image warping')."""

from __future__ import annotations

import numpy as np

from .camera import PinholeCamera
from .maps import TrnMap


def rectify_to_map(image: np.ndarray, cam: PinholeCamera, T_world_cam: np.ndarray, trn_map: TrnMap,
                   c0: int, r0: int, width: int, height: int, supersample: int = 1) -> np.ndarray:
    """Image resampled onto map pixels ``[c0, c0+width) x [r0, r0+height)``.

    Each map pixel's 3D point (map DEM height) is projected into the image with the given
    pose and the image is sampled bilinearly. Pixels that project outside the image or
    behind the camera are NaN. ``supersample > 1`` averages a k x k sub-grid per map pixel,
    which anti-aliases when the map pixel is coarser than the image GSD.
    """
    import cv2

    k = max(int(supersample), 1)
    offs = (np.arange(k) + 0.5) / k - 0.5
    cols = c0 + np.arange(width)[None, :, None, None] + offs[None, None, None, :]
    rows = r0 + np.arange(height)[:, None, None, None] + offs[None, None, :, None]
    cols, rows = np.broadcast_arrays(cols, rows)
    P = trn_map.world_points(cols.ravel(), rows.ravel())
    u, v, z = cam.project(T_world_cam, P)
    ok = np.isfinite(u) & (z > 0) & (u >= -0.5) & (v >= -0.5) & (u <= cam.width - 0.5) & (v <= cam.height - 0.5)
    # remap needs both map dimensions below SHRT_MAX; the lookup is pointwise, so any 2D shape works.
    mapx = np.where(ok, u, -10.0).astype(np.float32).reshape(height, width * k * k)
    mapy = np.where(ok, v, -10.0).astype(np.float32).reshape(height, width * k * k)
    vals = cv2.remap(np.asarray(image, dtype=np.float32), mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).ravel()
    vals = np.where(ok, vals, np.nan).reshape(height, width, k, k)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(vals, axis=(2, 3)) if k > 1 else vals[..., 0, 0]
    bad = np.isnan(vals).any(axis=(2, 3))
    return np.where(bad, np.nan, out).astype(np.float32)


def predicted_map_location(cam: PinholeCamera, T_world_cam: np.ndarray, trn_map: TrnMap,
                           cols_img, rows_img, n_iter: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Map pixel (col, row) where image pixels' rays meet the map terrain (fixed-point iteration)."""
    from . import frames

    o, d = cam.rays_world(T_world_cam, np.asarray(cols_img, float), np.asarray(rows_img, float))
    d = d.reshape(-1, 3)
    h = np.full(d.shape[0], float(np.nanmedian(trn_map.heights)))
    mc = mr = None
    for _ in range(n_iter):
        r = frames.MOON_RADIUS_M + h
        b = d @ o
        c = o @ o - r * r
        disc = b * b - c
        t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
        P = o + t[:, None] * d
        x, y, _ = frames.body_to_stereo(P)
        mc, mr = trn_map.xy_to_colrow(x, y)
        hn = trn_map.height_at_colrow(mc, mr)
        h = np.where(np.isfinite(hn), hn, h)
    return mc, mr
