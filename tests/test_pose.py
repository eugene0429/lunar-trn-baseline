import numpy as np
import pytest

from lunar_trn.pose import _rodrigues, mahalanobis_gate, project_uv, ransac_position, refine_pose


def _scene(n=60, seed=0):
    rng = np.random.default_rng(seed)
    f = 618.0
    K = np.array([[f, 0, 256.0], [0, f, 256.0], [0, 0, 1]])
    # Camera 10 km above a gently rough plane, looking down -z world.
    R = np.diag([1.0, -1.0, -1.0])  # camera x = world x, camera z = world -z
    c = np.array([100.0, -50.0, 10000.0])
    X = np.column_stack([rng.uniform(-3500, 3500, n), rng.uniform(-3500, 3500, n), rng.normal(0, 200, n)])
    uv, z = project_uv(K, R, c, X)
    keep = (uv[:, 0] > 0) & (uv[:, 0] < 512) & (uv[:, 1] > 0) & (uv[:, 1] < 512)
    return K, R, c, X[keep], uv[keep]


def test_ransac_position_with_outliers():
    K, R, c, X, uv = _scene()
    rng = np.random.default_rng(1)
    uv_n = uv + rng.normal(0, 0.3, uv.shape)
    n_out = len(uv) // 3
    uv_n[:n_out] += rng.uniform(-80, 80, (n_out, 2))
    res = ransac_position(uv_n, X, K, R, threshold_px=2.0)
    assert np.linalg.norm(res.position - c) < 5.0
    assert res.inliers[n_out:].mean() > 0.95
    assert res.inliers[:n_out].mean() < 0.1


def test_refine_pose_recovers_attitude_and_position():
    K, R, c, X, uv = _scene(n=80, seed=2)
    rng = np.random.default_rng(3)
    uv_n = uv + rng.normal(0, 0.5, uv.shape)
    R0 = _rodrigues(np.deg2rad([0.3, -0.2, 0.4])) @ R
    c0 = c + np.array([40.0, -30.0, 20.0])
    # Free attitude: tilt and horizontal position trade off; the error must be consistent
    # with the reported covariance.
    ref = refine_pose(uv_n, X, K, R0, c0, sigma_px=0.5)
    assert ref.converged
    assert ref.sigma0 == pytest.approx(1.0, abs=0.25)
    e = ref.position - c
    sd = np.sqrt(np.diag(ref.cov))
    assert np.all(np.isfinite(sd))
    assert np.all(np.abs(e) < 4 * sd[:3])
    # With the true attitude as a tight prior the position is pinned.
    ref2 = refine_pose(uv_n, X, K, R0, c0, sigma_px=0.5, att_prior_R=R, att_prior_sigma_rad=np.deg2rad(0.005))
    assert np.linalg.norm(ref2.position - c) < 5.0
    ang = np.rad2deg(np.arccos(np.clip((np.trace(ref2.R_wc.T @ R) - 1) / 2, -1, 1)))
    assert ang < 0.01


def test_mahalanobis_gate_rejects_gross_residuals():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, (100, 2))
    r[:5] = 10.0
    keep, gamma, d2 = mahalanobis_gate(r, 1.0)
    assert not keep[:5].any() and keep[5:].mean() > 0.9
    assert 2.0 < gamma < 10.0
