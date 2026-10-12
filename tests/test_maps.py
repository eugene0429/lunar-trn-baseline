import numpy as np
import pytest

from lunar_trn import frames
from lunar_trn.dem import Dem
from lunar_trn.maps import TrnMap, cast_shadows_grid, render_map
from lunar_trn.render import cast_shadow
from lunar_trn.validation.v1 import sun_from_az_el


def _bumpy_dem(n=601, pixel=10.0, cx=20000.0, cy=-15000.0, seed=0):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    H = gaussian_filter(rng.normal(size=(n, n)), 12) * 4000.0
    x0 = cx - (n - 1) / 2 * pixel
    y0 = cy + (n - 1) / 2 * pixel
    return Dem(H.astype(np.float32), x0, y0, pixel)


@pytest.mark.parametrize("az,el", [(30.0, 15.0), (200.0, 25.0), (300.0, 10.0)])
def test_horizon_scan_matches_raycast(az, el):
    dem = _bumpy_dem()
    cx, cy = 20000.0, -15000.0
    p = frames.stereo_to_body(cx, cy, 0.0)
    sun = sun_from_az_el(p, az, el)
    sh_grid = cast_shadows_grid(dem, sun)
    # Compare on interior nodes whose shadow casters lie inside the window.
    rng = np.random.default_rng(1)
    rows, cols = dem.shape
    r = rng.integers(150, rows - 150, 400)
    c = rng.integers(150, cols - 150, 400)
    x, y = dem.colrow_to_xy(c, r)
    pts = frames.stereo_to_body(x, y, dem.heights[r, c])
    sh_ray = cast_shadow(dem, pts, sun, offset=0.5, tol=0.05, min_step=0.5)
    agree = np.mean(sh_grid[r, c] == sh_ray)
    assert agree > 0.97, agree
    assert 0.02 < sh_ray.mean() < 0.98  # the test terrain has both lit and shadowed nodes


def test_render_map_georeference_and_save(tmp_path):
    dem = _bumpy_dem(n=401, pixel=5.0)
    cx, cy = 20000.0, -15000.0
    sun = sun_from_az_el(frames.stereo_to_body(cx, cy, 0.0), 120.0, 30.0)
    m = render_map(dem, (cx, cy), half_size_m=500.0, pixel=10.0, sun_dir=sun)
    assert m.shape == (100, 100)
    assert m.x0 == pytest.approx(cx - 500 + 5) and m.y0 == pytest.approx(cy + 500 - 5)
    np.testing.assert_allclose(m.heights[50, 50], dem.height_at(*m.colrow_to_xy(50, 50)), atol=1e-3)
    assert np.isfinite(m.image).all() and m.image.max() > 0
    m.save(tmp_path / "m.npz")
    m2 = TrnMap.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(m2.image, m.image)
    assert m2.meta["model"] == "hapke"
    P = m2.world_points(np.array([10.0]), np.array([20.0]))
    x, y, h = frames.body_to_stereo(P)
    assert (x[0], y[0]) == pytest.approx(m.colrow_to_xy(10.0, 20.0))


def test_correlate_masks_flat_windows():
    from lunar_trn.matching import analyze_peak, correlate

    rng = np.random.default_rng(0)
    search = np.zeros((80, 80), np.float32)
    search[:, 40:] = rng.normal(size=(80, 40))  # left half flat (e.g. shadow)
    tpl = search[30:51, 50:71].copy()
    surf = correlate(search, tpl)
    pk = analyze_peak(surf)
    assert (pk.ix, pk.iy) == (50, 30) and pk.height == pytest.approx(1.0, abs=1e-4)
    assert np.all(surf[:, :20] == 0.0)


def test_closed_form_normals_match_cross_product_normals():
    """shade_grid's conformal closed form must agree with render.surface_normals."""
    from lunar_trn.maps import shade_grid
    from lunar_trn.render import surface_normals

    dem = _bumpy_dem(n=121, pixel=5.0)
    sun = sun_from_az_el(frames.stereo_to_body(20000.0, -15000.0, 0.0), 70.0, 35.0)
    I, _ = shade_grid(dem, sun, model="lambert", shadows=False)
    rows, cols = dem.shape
    cc, rr = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    x, y = dem.colrow_to_xy(cc, rr)
    n = surface_normals(dem, x, y)
    ref = np.clip(n @ (sun / np.linalg.norm(sun)), 0, None)
    np.testing.assert_allclose(I, ref, atol=2e-5)


def test_resample_regular_matches_bilinear():
    from lunar_trn.maps import resample_regular

    dem = _bumpy_dem(n=201, pixel=5.0)
    x0, y0, px = dem.x0 + 12.3, dem.y0 - 7.7, 6.6
    out = resample_regular(dem, x0, y0, px, 50, 60)
    cc, rr = np.meshgrid(np.arange(60), np.arange(50))
    ref = dem.height_at(x0 + cc * px, y0 - rr * px)
    np.testing.assert_allclose(out, ref, atol=0.05)
