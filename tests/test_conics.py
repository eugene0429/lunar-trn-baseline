import numpy as np

from lunar_trn.craters import conics
from lunar_trn.pose import project_uv

F = 256 / np.tan(np.deg2rad(22.5))
K = np.array([[F, 0, 256.0], [0, F, 256.0], [0, 0, 1.0]])
E1 = np.array([[1.0, 0.0, 0.0]])
E2 = np.array([[0.0, 1.0, 0.0]])


def _camera(alt, tilt_deg=0.0):
    """Camera above the origin of a flat world (ground z = 0), optical axis -Z tilted about +X."""
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    t = np.deg2rad(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Rx @ R, np.array([0.0, 0.0, alt])


def _ground_point(R, c, u, v):
    d = R @ np.linalg.inv(K) @ np.array([u, v, 1.0])
    return c + (-c[2] / d[2]) * d


def test_projected_rim_points_lie_on_image_ellipse():
    R, c = _camera(3500.0, tilt_deg=30.0)
    Xc = _ground_point(R, c, 400.0, 300.0)[None, :]
    a, b, psi = 500.0, 350.0, 0.4
    p = conics.project_rims(K, R, c, Xc, E1, E2, [a], [b], [psi])
    assert p["valid"][0]
    emaj, emin = conics.rim_axes(E1, E2, np.array([psi]))
    phi = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    P = Xc + a * np.cos(phi)[:, None] * emaj + b * np.sin(phi)[:, None] * emin
    uv, _ = project_uv(K, R, c, P)
    r = conics.ellipse_residual(uv[:, 0], uv[:, 1], p["u"][0], p["v"][0], p["A"][0], p["B"][0], p["theta"][0])
    assert np.max(np.abs(r)) < 1e-6 * p["A"][0] + 1e-6


def test_conic_ellipse_round_trip():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 512, 20), rng.uniform(0, 512, 20)
    B = rng.uniform(3, 40, 20)
    A = B * rng.uniform(1.2, 2.0, 20)
    th = rng.uniform(-np.pi / 2, np.pi / 2, 20)
    u2, v2, A2, B2, th2, ok = conics.conic_to_ellipse(conics.ellipse_to_conic(u, v, A, B, th))
    assert ok.all()
    np.testing.assert_allclose(np.column_stack([u2, v2, A2, B2]), np.column_stack([u, v, A, B]), rtol=1e-9, atol=1e-7)
    dth = (th2 - th + np.pi / 2) % np.pi - np.pi / 2
    assert np.max(np.abs(dth)) < 1e-7


def test_center_bias_zero_at_nadir_and_several_px_off_axis():
    R, c = _camera(3500.0)
    p = conics.project_rims(K, R, c, np.zeros((1, 3)), E1, E2, [500.0], [500.0], [0.0])
    assert np.hypot(*p["bias"][0]) < 1e-6
    R, c = _camera(3500.0, tilt_deg=30.0)
    Xc = _ground_point(R, c, 256.0, 256.0)[None, :]
    p = conics.project_rims(K, R, c, Xc, E1, E2, [500.0], [500.0], [0.0])
    assert 2.0 < np.hypot(*p["bias"][0]) < 8.0  # ~4 px for a 1 km crater at 3.5 km, 30 deg off nadir


def test_rim_behind_camera_or_seen_from_below_is_invalid():
    R, c = _camera(3500.0)
    behind = np.array([[0.0, 0.0, 5000.0]])  # above the camera, which looks down
    p = conics.project_rims(K, R, c, behind, E1, E2, [100.0], [100.0], [0.0])
    assert not p["valid"][0] and np.isnan(p["u"][0])
    flipped = conics.project_rims(K, R, c, np.zeros((1, 3)), E2, E1, [100.0], [100.0], [0.0])  # normal -Z
    assert not flipped["valid"][0]
    empty = conics.project_rims(K, R, c, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), [], [], [])
    assert empty["u"].shape == (0,) and empty["valid"].shape == (0,)


def test_fit_ellipse_direct_recovers_parameters():
    rng = np.random.default_rng(1)
    phi = rng.uniform(0, 2 * np.pi, 60)
    u0, v0, A, B, th = 1200.0, -300.0, 80.0, 50.0, 0.7
    x = u0 + A * np.cos(phi) * np.cos(th) - B * np.sin(phi) * np.sin(th)
    y = v0 + A * np.cos(phi) * np.sin(th) + B * np.sin(phi) * np.cos(th)
    u, v, A2, B2, th2, ok = conics.fit_ellipse_direct(x + rng.normal(0, 0.2, 60), y + rng.normal(0, 0.2, 60))
    assert ok
    assert abs(u - u0) < 0.5 and abs(v - v0) < 0.5 and abs(A2 - A) < 0.5 and abs(B2 - B) < 0.5
    assert abs((th2 - th + np.pi / 2) % np.pi - np.pi / 2) < 0.02
    assert not conics.fit_ellipse_direct(np.arange(4.0), np.arange(4.0))[-1]
