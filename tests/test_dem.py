import numpy as np
import pytest

from lunar_trn import frames
from lunar_trn.dem import Dem, DemSource, bilinear


def _plane_dem(h0=100.0, slope_x=0.0, slope_y=0.0, n=401, pixel=5.0, cx=0.0, cy=0.0):
    half = (n - 1) / 2 * pixel
    xs = cx - half + np.arange(n) * pixel
    ys = cy + half - np.arange(n) * pixel
    X, Y = np.meshgrid(xs, ys)
    H = h0 + slope_x * (X - cx) + slope_y * (Y - cy)
    return Dem(heights=H.astype(np.float32), x0=xs[0], y0=ys[0], pixel=pixel)


def test_bilinear_exact_on_linear_field():
    rng = np.random.default_rng(0)
    g = np.add.outer(np.arange(10) * 2.0, np.arange(12) * 3.0)  # row*2 + col*3
    col = rng.uniform(0, 11, 100)
    row = rng.uniform(0, 9, 100)
    np.testing.assert_allclose(bilinear(g, col, row), row * 2 + col * 3, atol=1e-12)
    # Last row/column are valid, outside is NaN.
    assert bilinear(g, np.array([11.0]), np.array([9.0]))[0] == pytest.approx(9 * 2 + 11 * 3)
    assert np.isnan(bilinear(g, np.array([11.01]), np.array([3.0]))[0])
    assert np.isnan(bilinear(g, np.array([-0.01]), np.array([3.0]))[0])


def test_height_at_and_grid_convention():
    dem = _plane_dem(h0=10.0, slope_x=0.01, slope_y=-0.02)
    x, y = 123.4, -56.7
    assert dem.height_at(x, y) == pytest.approx(10.0 + 0.01 * x - 0.02 * y, abs=1e-4)
    col, row = dem.xy_to_colrow(dem.x0, dem.y0)
    assert (col, row) == (0.0, 0.0)


def test_raycast_vertical_on_flat_terrain():
    dem = _plane_dem(h0=-1234.0)
    alt = 15_000.0
    o = frames.stereo_to_body(np.array([0.0, 300.0]), np.array([0.0, -200.0]), np.array([alt, alt]))
    d = -o / np.linalg.norm(o, axis=1, keepdims=True)
    t = dem.raycast(o, d)
    np.testing.assert_allclose(t, alt + 1234.0, atol=0.05)


def test_raycast_oblique_on_sloped_terrain():
    """Hits must lie on the surface and be the first crossing."""
    dem = _plane_dem(h0=0.0, slope_x=0.3, slope_y=-0.1, n=2401, pixel=10.0)
    rng = np.random.default_rng(3)
    o = np.repeat(frames.stereo_to_body(0.0, 0.0, 8000.0)[None], 200, axis=0)
    down = -o[0] / np.linalg.norm(o[0])
    # Random directions within 35 deg of nadir.
    a = np.deg2rad(rng.uniform(0, 35, 200))
    b = rng.uniform(0, 2 * np.pi, 200)
    e1 = np.cross(down, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)
    d = (np.cos(a)[:, None] * down + np.sin(a)[:, None] * (np.cos(b)[:, None] * e1 + np.sin(b)[:, None] * e2))
    t = dem.raycast(o, d, tol=0.01)
    assert np.isfinite(t).all()
    p = o + t[:, None] * d
    x, y, h = frames.body_to_stereo(p)
    np.testing.assert_allclose(h, dem.height_at(x, y), atol=0.02)


def test_raycast_finds_first_hit_behind_ridge():
    """A thin tall ridge in front of a low plain must stop the ray (no tunnelling)."""
    dem = _plane_dem(h0=0.0, n=801)
    H = dem.heights.copy()
    H[:, 450:452] = 2000.0  # ridge 10 m wide, 2 km high, east of the origin
    dem = Dem(H, dem.x0, dem.y0, dem.pixel)
    o = frames.stereo_to_body(0.0, 0.0, 3000.0)[None]
    target = frames.stereo_to_body(np.array([700.0]), np.array([0.0]), np.array([0.0]))
    d = target - o
    t = dem.raycast(o, d, tol=0.01)
    p = o + t[:, None] * (d / np.linalg.norm(d))
    x, _, h = frames.body_to_stereo(p)
    assert 245.0 <= x[0] <= 262.0  # stopped at the ridge columns
    assert h[0] > 100.0


def test_raycast_miss_returns_nan():
    dem = _plane_dem(h0=0.0, n=101)
    o = frames.stereo_to_body(0.0, 0.0, 5000.0)[None]
    up = o / np.linalg.norm(o)
    assert np.isnan(dem.raycast(o, up)[0])
    # Ray leaving the window sideways.
    side = frames.stereo_to_body(5000.0, 0.0, 5000.0)[None] - o
    assert np.isnan(dem.raycast(o, side)[0])


def test_pds_img_reader(tmp_path):
    """Synthetic PDS3 IMG with the LOLA polar label layout."""
    n = 64
    dn = (np.arange(n * n, dtype=np.int16).reshape(n, n) - 1000)
    (tmp_path / "toy.img").write_bytes(dn.astype("<i2").tobytes())
    off = (n - 1) / 2.0 + 0.0  # pixel-centred grid symmetric about the pole
    (tmp_path / "toy.lbl").write_text(
        f"""PDS_VERSION_ID = "PDS3"
    LINES                 = {n}
    LINE_SAMPLES          = {n}
    SAMPLE_TYPE           = LSB_INTEGER
    SAMPLE_BITS           = 16
    SCALING_FACTOR        = 0.5
 MAP_SCALE                    = 5 <m/pix>
 LINE_PROJECTION_OFFSET       = {off} <pix>
 SAMPLE_PROJECTION_OFFSET     = {off} <pix>
END
"""
    )
    src = DemSource(tmp_path / "toy.img")
    assert src.x0 == pytest.approx(-off * 5.0)
    assert src.y0 == pytest.approx(off * 5.0)
    win = src.read_window(-20.0, 20.0, -20.0, 20.0, margin_px=0)
    r0 = int(round((src.y0 - win.y0) / 5.0))
    c0 = int(round((win.x0 - src.x0) / 5.0))
    np.testing.assert_allclose(win.heights, dn[r0:r0 + win.shape[0], c0:c0 + win.shape[1]] * 0.5)


def test_downsample_preserves_plane_and_georeference():
    dem = _plane_dem(h0=50.0, slope_x=0.02, slope_y=0.01, n=400)
    d4 = dem.downsample(4)
    assert d4.pixel == 20.0 and d4.shape == (100, 100)
    xs = np.array([-300.0, 0.0, 412.0])
    ys = np.array([250.0, -10.0, 33.0])
    np.testing.assert_allclose(d4.height_at(xs, ys), dem.height_at(xs, ys), atol=1e-3)
