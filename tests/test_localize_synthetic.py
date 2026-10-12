"""End-to-end localisation on a fully synthetic scene (no dataset needed).

A random fractal-like DEM is rendered (a) as the onboard map and (b) as a perspective
camera image with the same reflectance model. Starting 1 km off, coarse -> fine must
recover the true position to a fraction of a map pixel and report a fix.
"""

import numpy as np
import pytest

from lunar_trn import frames
from lunar_trn.camera import PinholeCamera
from lunar_trn.dem import Dem
from lunar_trn.experiments import build_mapset, sample_prior, score
from lunar_trn.localize import localize
from lunar_trn.render import render_perspective
from lunar_trn.validation.v1 import sun_from_az_el


def _fractal_dem(n=2400, pixel=5.0, cx=-20000.0, cy=30000.0, seed=0):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(f, f)
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / k**1.6
    spec[0, 0] = 0
    H = np.real(np.fft.ifft2(spec))
    H = (H - H.mean()) / H.std() * 150.0
    x0 = cx - (n - 1) / 2 * pixel
    y0 = cy + (n - 1) / 2 * pixel
    return Dem(H.astype(np.float32), x0, y0, pixel, name="synthetic")


@pytest.mark.slow
def test_localize_recovers_pose_on_synthetic_scene():
    dem = _fractal_dem()
    cx, cy = -20000.0, 30000.0
    f = 256 / np.tan(np.deg2rad(22.5))
    cam = PinholeCamera(np.array([[f, 0, 256.0], [0, f, 256.0], [0, 0, 1]]), 512, 512)
    alt = 4000.0
    p = frames.stereo_to_body(cx, cy, alt)
    down = -p / np.linalg.norm(p)
    xc = np.cross(down, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([xc, np.cross(down, xc), down])
    T[:3, 3] = p
    sun = sun_from_az_el(p, 135.0, 35.0)
    out = render_perspective(dem, cam, T, sun, step=1, model="hapke")
    img = out["image"]
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    img = np.nan_to_num(img).astype(np.float32)
    gsd = alt / f
    ms = build_mapset(dem, cam, T, sun, gsd, prior_radius_m=1000.0)
    rng = np.random.default_rng(4)
    Tp, cov, _ = sample_prior(T, rng, horiz_radius_m=1000.0, vert_m=60.0)
    res = localize(img, cam.K, Tp, cov, ms, seed=1)
    sc = score(res, T, ms.fine.pixel)
    assert res.status == "fix", res.health
    assert sc["err_h_m"] < 0.5 * ms.fine.pixel, sc
    assert abs(sc["err_u_m"]) < 3 * ms.fine.pixel
    assert res.health["fine"]["n_inliers"] >= 40
