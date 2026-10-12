import numpy as np
import pytest

from lunar_trn.camera import PinholeCamera, orthonormalize, rotation_angle_deg


def _cam(pc=0.5):
    f = 256 / np.tan(np.deg2rad(22.5))
    K = np.array([[f, 0, 256.0], [0, f, 256.0], [0, 0, 1]])
    return PinholeCamera(K, 512, 512, pixel_center=pc)


def _pose():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = q
    T[:3, 3] = [1e3, -2e3, -1.75e6]
    return T


@pytest.mark.parametrize("pc", [0.0, 0.5])
def test_ray_project_round_trip(pc):
    cam = _cam(pc)
    T = _pose()
    cols, rows = cam.pixel_grid(step=37)
    o, d = cam.rays_world(T, cols, rows)
    P = o + 1234.5 * d
    c2, r2, z = cam.project(T, P)
    np.testing.assert_allclose(c2, cols, atol=1e-6)
    np.testing.assert_allclose(r2, rows, atol=1e-6)
    assert (z > 0).all()


def test_symmetric_frustum_with_half_pixel_convention():
    cam = _cam(0.5)
    # With cx = W/2 and centres at +0.5, pixels 255 and 256 straddle the optical axis.
    d = cam.rays_cam(np.array([255.0, 256.0]), np.array([255.0, 255.0]))
    assert d[0, 0] == pytest.approx(-d[1, 0])
    # The image edge (col -0.5 i.e. u = 0) is at the half field of view.
    d_edge = cam.rays_cam(np.array([-0.5]), np.array([255.5]))[0]
    assert np.rad2deg(np.arctan2(-d_edge[0], d_edge[2])) == pytest.approx(22.5)


def test_orthonormalize_and_angle():
    R = _pose()[:3, :3].astype(np.float32).astype(np.float64)
    Rn = orthonormalize(R + 1e-6)
    np.testing.assert_allclose(Rn.T @ Rn, np.eye(3), atol=1e-12)
    assert rotation_angle_deg(Rn, Rn) == pytest.approx(0.0, abs=1e-5)
