import numpy as np
import pytest

from lunar_trn import frames
from lunar_trn.camera import PinholeCamera
from lunar_trn.dem import Dem
from lunar_trn.render import cast_shadow, reflectance, render_perspective, surface_normals


def _dem(H, pixel=5.0):
    n_r, n_c = H.shape
    return Dem(H.astype(np.float32), x0=-(n_c - 1) / 2 * pixel, y0=(n_r - 1) / 2 * pixel, pixel=pixel)


def test_normals_flat_and_tilted():
    n = 201
    flat = _dem(np.zeros((n, n)))
    x = np.array([0.0, 100.0, -230.0])
    y = np.array([0.0, -40.0, 75.0])
    nrm = surface_normals(flat, x, y)
    up = frames.stereo_to_body(x, y, 0.0)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    np.testing.assert_allclose(np.einsum("ij,ij->i", nrm, up), 1.0, atol=1e-6)

    # Plane rising towards +x (east-ish at the pole): normal tilts towards -x.
    cols = np.arange(n) * 5.0 - 500.0
    tilted = _dem(np.tile(0.5 * cols, (n, 1)))
    nt = surface_normals(tilted, np.array([0.0]), np.array([0.0]))[0]
    ex = frames.stereo_to_body(1.0, 0.0, 0.0) - frames.stereo_to_body(-1.0, 0.0, 0.0)
    ex /= np.linalg.norm(ex)
    up0 = -np.array([0.0, 0.0, 1.0])
    assert nt @ up0 == pytest.approx(1 / np.sqrt(1 + 0.25), abs=1e-6)
    assert nt @ ex == pytest.approx(-0.5 / np.sqrt(1 + 0.25), abs=1e-6)


def test_cast_shadow_behind_wall():
    n = 801
    H = np.zeros((n, n))
    H[:, 400:404] = 500.0  # N-S wall at x = 0..15 m, 500 m high
    dem = _dem(H)
    # Sun low in the east-ish map direction (+x), 20 deg elevation.
    p0 = frames.stereo_to_body(0.0, 0.0, 0.0)
    ex = frames.stereo_to_body(1.0, 0.0, 0.0) - frames.stereo_to_body(-1.0, 0.0, 0.0)
    ex /= np.linalg.norm(ex)
    up = p0 / np.linalg.norm(p0)
    el = np.deg2rad(20.0)
    sun = np.cos(el) * ex + np.sin(el) * up
    # Points west of the wall within 500/tan(20) = 1374 m are shadowed; beyond that lit.
    xs = np.array([-200.0, -1000.0, -1500.0, 200.0])
    pts = frames.stereo_to_body(xs, np.zeros_like(xs), np.zeros_like(xs))
    sh = cast_shadow(dem, pts, sun)
    assert sh.tolist() == [True, True, False, False]


def test_reflectance_models_nonnegative_and_zero_unlit():
    ci = np.array([-0.2, 0.0, 0.3, 1.0])
    for m in ("lambert", "lunar_lambert", "hapke"):
        r = reflectance(ci, np.full(4, 0.9), np.full(4, 0.5), model=m)
        assert (r >= 0).all() and r[0] == 0 and r[1] == 0 and r[3] > r[2]


def test_render_perspective_depth_on_flat_terrain():
    dem = _dem(np.full((1201, 1201), -500.0))
    f = 256 / np.tan(np.deg2rad(22.5))
    cam = PinholeCamera(np.array([[f, 0, 256], [0, f, 256], [0, 0, 1.0]]), 512, 512)
    p = frames.stereo_to_body(0.0, 0.0, 2000.0)
    down = -p / np.linalg.norm(p)
    x = np.cross(down, [1.0, 0, 0]); x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(down, x), down])
    T[:3, 3] = p
    sun = -down
    out = render_perspective(dem, cam, T, sun, step=64, shadows=False)
    centre = out["zdepth"][3:5, 3:5]
    np.testing.assert_allclose(centre, 2500.0, rtol=2e-3)
    assert np.isfinite(out["image"]).all()


def test_stereo_tangents_match_finite_differences():
    from lunar_trn.render import stereo_tangents

    x = np.array([0.0, 4.2e4, -7.0e4])
    y = np.array([0.0, -3.1e4, 6.5e4])
    u, ux, uy = stereo_tangents(x, y)
    R = frames.MOON_RADIUS_M
    np.testing.assert_allclose(u, frames.stereo_to_body(x, y, 0.0) / R, atol=1e-12)
    d = 0.5
    fx = (frames.stereo_to_body(x + d, y, 0.0) - frames.stereo_to_body(x - d, y, 0.0)) / (2 * d * R)
    fy = (frames.stereo_to_body(x, y + d, 0.0) - frames.stereo_to_body(x, y - d, 0.0)) / (2 * d * R)
    np.testing.assert_allclose(ux, fx, atol=1e-12)
    np.testing.assert_allclose(uy, fy, atol=1e-12)
