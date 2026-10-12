"""Camera pose from 2D-3D landmark matches.

* :func:`ransac_position`: robust camera position with the attitude held at the prior
  (2-point minimal solver on the ray constraints ``(I - d d^T)(X - c) = 0``). LVS takes
  attitude from the IMU; position-only hypotheses make RANSAC cheap and well conditioned
  for near-planar, near-nadir scenes.
* :func:`refine_pose`: 6-DoF Gauss-Newton on the reprojection error with a Huber kernel,
  returning the pose covariance (position + small-angle attitude, world frame).
* :func:`mahalanobis_gate`: the LVS adaptive gate ``d^2 < q3 + 1.5 (q3 - q1)``.

Pixel coordinates here are continuous image coordinates in the frame of ``K``
(``u = col + pixel_center``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + _skew(w)
    k = w / th
    K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _skew(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def project_uv(K: np.ndarray, R_wc: np.ndarray, c: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous pixel coordinates (N, 2) and camera z of world points X (N, 3)."""
    pc = (X - c) @ R_wc  # rows: camera-frame coordinates
    z = pc[:, 2]
    uv = np.column_stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]])
    return uv, z


def solve_position(dirs_world: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least-squares camera centre from world rays (unit) through known points."""
    n = X.shape[0]
    w = np.ones(n) if w is None else w
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(n):
        P = np.eye(3) - np.outer(dirs_world[i], dirs_world[i])
        A += w[i] * P
        b += w[i] * P @ X[i]
    return np.linalg.solve(A, b)


@dataclass
class PositionRansacResult:
    position: np.ndarray | None
    inliers: np.ndarray  # bool mask
    residuals_px: np.ndarray
    n_hypotheses: int


def ransac_position(uv: np.ndarray, X: np.ndarray, K: np.ndarray, R_wc: np.ndarray, threshold_px: float,
                    max_iter: int = 500, rng: np.random.Generator | None = None,
                    min_inliers: int = 2) -> PositionRansacResult:
    """RANSAC over point pairs for the camera position with fixed attitude ``R_wc``."""
    n = uv.shape[0]
    if n < 2:
        return PositionRansacResult(None, np.zeros(n, bool), np.full(n, np.inf), 0)
    rng = np.random.default_rng(0) if rng is None else rng
    Kinv = np.linalg.inv(K)
    d_cam = np.column_stack([uv, np.ones(n)]) @ Kinv.T
    d_w = d_cam @ R_wc.T
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    best = (-1, np.inf, None)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_iter:
        sel = rng.choice(len(pairs), size=max_iter, replace=False)
        pairs = [pairs[k] for k in sel]
    for i, j in pairs:
        # Skip near-parallel ray pairs (ill-conditioned).
        if abs(d_w[i] @ d_w[j]) > 1 - 1e-8:
            continue
        c = solve_position(d_w[[i, j]], X[[i, j]])
        res = _reproj_residual(K, R_wc, c, X, uv)
        inl = res < threshold_px
        cnt = int(inl.sum())
        score = np.sum(np.minimum(res, threshold_px))
        if cnt > best[0] or (cnt == best[0] and score < best[1]):
            best = (cnt, score, inl)
    if best[2] is None or best[0] < min_inliers:
        return PositionRansacResult(None, np.zeros(n, bool), np.full(n, np.inf), len(pairs))
    inl = best[2]
    c = solve_position(d_w[inl], X[inl])
    for _ in range(3):
        res = _reproj_residual(K, R_wc, c, X, uv)
        inl_new = res < threshold_px
        if inl_new.sum() < min_inliers:
            break
        inl = inl_new
        c = solve_position(d_w[inl], X[inl])
    res = _reproj_residual(K, R_wc, c, X, uv)
    return PositionRansacResult(c, res < threshold_px, res, len(pairs))


def _reproj_residual(K, R_wc, c, X, uv) -> np.ndarray:
    pred, z = project_uv(K, R_wc, c, X)
    r = np.linalg.norm(pred - uv, axis=1)
    return np.where(z > 0, r, np.inf)


@dataclass
class RefineResult:
    R_wc: np.ndarray
    position: np.ndarray
    cov: np.ndarray  # 6x6, order: position (3, world), attitude (3, small-angle world-frame rotation)
    residuals: np.ndarray  # (N, 2) px
    rms_px: float
    sigma0: float  # a-posteriori standard deviation of unit weight
    converged: bool


def _log_so3(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def refine_pose(uv: np.ndarray, X: np.ndarray, K: np.ndarray, R_wc: np.ndarray, c: np.ndarray,
                sigma_px: np.ndarray | float = 1.0, n_iter: int = 20, huber_px: float | None = 2.0,
                fix_attitude: bool = False, att_prior_R: np.ndarray | None = None,
                att_prior_sigma_rad: float | None = None) -> RefineResult:
    """Gauss-Newton on reprojection error over (position, attitude).

    Attitude perturbation: ``R = exp([w]x) R_wc`` with ``w`` in the world frame. With
    ``att_prior_sigma_rad`` the prior attitude ``att_prior_R`` (default: the initial ``R_wc``)
    enters as a pseudo-measurement, i.e. a MAP estimate. Near-nadir narrow-FOV views of
    near-planar terrain leave tilt and horizontal position strongly correlated, so the
    attitude prior (from the IMU in a real system) is what pins the horizontal position.
    """
    uv = np.asarray(uv, float)
    X = np.asarray(X, float)
    n = uv.shape[0]
    sig = np.broadcast_to(np.asarray(sigma_px, float), (n,)).copy()
    R = R_wc.copy()
    cc = c.copy()
    converged = False
    npar = 3 if fix_attitude else 6
    use_prior = (not fix_attitude) and att_prior_sigma_rad is not None and att_prior_sigma_rad > 0
    R_prior = R_wc.copy() if att_prior_R is None else np.asarray(att_prior_R, float)
    J_all = None
    wts = np.ones(n)
    for _ in range(n_iter):
        pc = (X - cc) @ R
        z = pc[:, 2]
        pred = np.column_stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]])
        r = (pred - uv) / sig[:, None]
        # Jacobian of projection wrt camera-frame point.
        fx, fy = K[0, 0], K[1, 1]
        Jp = np.zeros((n, 2, 3))
        Jp[:, 0, 0] = fx / z
        Jp[:, 0, 2] = -fx * pc[:, 0] / z**2
        Jp[:, 1, 1] = fy / z
        Jp[:, 1, 2] = -fy * pc[:, 1] / z**2
        # pc = R^T (X - c): d pc / d c = -R^T ; d pc / d w (world rot of camera) = R^T [X - c]x
        dpc_dc = -R.T
        J = np.zeros((n, 2, npar))
        J[:, :, :3] = Jp @ dpc_dc
        if not fix_attitude:
            for i in range(n):
                J[i, :, 3:] = Jp[i] @ (R.T @ _skew(X[i] - cc))
        J /= sig[:, None, None]
        rn = np.linalg.norm(r, axis=1)
        if huber_px is not None:
            k = huber_px
            wts = np.where(rn <= k, 1.0, k / np.maximum(rn, 1e-12))
        Jf = J.reshape(-1, npar)
        rf = r.reshape(-1)
        wf = np.repeat(wts, 2)
        if use_prior:
            r_att = _log_so3(R @ R_prior.T) / att_prior_sigma_rad
            J_att = np.zeros((3, npar))
            J_att[:, 3:] = np.eye(3) / att_prior_sigma_rad
            Jf = np.vstack([Jf, J_att])
            rf = np.concatenate([rf, r_att])
            wf = np.concatenate([wf, np.ones(3)])
        A = Jf.T @ (Jf * wf[:, None])
        g = Jf.T @ (wf * rf)
        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        cc = cc + delta[:3]
        if not fix_attitude:
            R = _rodrigues(delta[3:]) @ R
        J_all = (Jf, wf)
        if np.linalg.norm(delta[:3]) < 1e-4 and (fix_attitude or np.linalg.norm(delta[3:]) < 1e-9):
            converged = True
            break
    uv_pred, z = project_uv(K, R, cc, X)
    res = uv_pred - uv
    dof = max(2 * n - npar, 1)
    rn = np.linalg.norm(res / sig[:, None], axis=1)
    sigma0 = float(np.sqrt(np.sum(wts * rn**2) / dof))
    cov = np.full((6, 6), np.nan)
    if J_all is not None:
        Jf, wf = J_all
        A = Jf.T @ (Jf * wf[:, None])
        try:
            Ainv = np.linalg.inv(A) * max(sigma0, 1.0) ** 2
            if fix_attitude:
                cov[:3, :3] = Ainv
            else:
                cov = Ainv
        except np.linalg.LinAlgError:
            pass
    return RefineResult(R, cc, cov, res, float(np.sqrt(np.mean(np.sum(res**2, axis=1)))), sigma0, converged)


def mahalanobis_gate(residuals: np.ndarray, sigma_px: np.ndarray | float, alpha: float = 1.5) -> tuple[np.ndarray, float, np.ndarray]:
    """LVS adaptive Mahalanobis gate. Returns (keep mask, gamma, d^2)."""
    sig = np.broadcast_to(np.asarray(sigma_px, float), (residuals.shape[0],))
    d2 = np.sum((residuals / sig[:, None]) ** 2, axis=1)
    q1, q3 = np.percentile(d2, [25, 75])
    gamma = q3 + alpha * (q3 - q1)
    return d2 < gamma, float(gamma), d2
