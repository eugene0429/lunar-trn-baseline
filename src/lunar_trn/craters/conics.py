"""Crater rim conics: rim ellipses in 3D and their perspective images.

A crater rim is an ellipse on its rim plane, P(s, t) = X_c + s e1 + t e2, with semi-axes
``a >= b`` (m) and major-axis angle ``psi`` measured from ``e1`` towards ``e2``. Its image
is the conic C = H^-T C_p H^-1 with the plane-to-image homography
H = K [R^T e1, R^T e2, R^T (X_c - c)] (``R`` camera-to-world, ``c`` camera centre).

Image coordinates are continuous pixel coordinates of the ``K`` frame, as in
:func:`lunar_trn.pose.project_uv` (pixel index = u - pixel_center). The centre of an image
ellipse is not the image of the rim centre; :func:`project_rims` returns both so that
callers can correct this perspective centre bias.
"""

from __future__ import annotations

import numpy as np

from ..pose import project_uv


def _wrap_half_pi(theta):
    return (np.asarray(theta, float) + np.pi / 2) % np.pi - np.pi / 2


def plane_conic(a, b, psi) -> np.ndarray:
    """Conic matrices (N, 3, 3) of centred ellipses with semi-axes a, b and angle psi."""
    a, b, psi = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, float)) for x in (a, b, psi)))
    c, s = np.cos(psi), np.sin(psi)
    ia2, ib2 = 1.0 / a**2, 1.0 / b**2
    C = np.zeros(a.shape + (3, 3))
    C[..., 0, 0] = c * c * ia2 + s * s * ib2
    C[..., 1, 1] = s * s * ia2 + c * c * ib2
    C[..., 0, 1] = C[..., 1, 0] = c * s * (ia2 - ib2)
    C[..., 2, 2] = -1.0
    return C


def ellipse_to_conic(u, v, A, B, theta) -> np.ndarray:
    """Conic matrices (N, 3, 3) of image ellipses (centre u, v; semi-axes A, B; angle theta)."""
    Cp = plane_conic(A, B, theta)
    u, v = np.broadcast_arrays(np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float)))
    T = np.zeros(Cp.shape)
    T[..., 0, 0] = T[..., 1, 1] = T[..., 2, 2] = 1.0
    T[..., 0, 2] = -u
    T[..., 1, 2] = -v
    return np.swapaxes(T, -1, -2) @ Cp @ T


def conic_to_ellipse(C):
    """Centre (u, v), semi-axes A >= B, major-axis angle and validity of conics (N, 3, 3).

    Non-ellipses (hyperbolae, parabolae, imaginary or degenerate conics) are invalid and
    return NaN parameters.
    """
    C = np.asarray(C, float)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    M = C[..., :2, :2]
    m = C[..., :2, 2]
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] ** 2
    valid = np.isfinite(det) & (det > 0)
    sd = np.where(valid, det, 1.0)
    u = -(M[..., 1, 1] * m[..., 0] - M[..., 0, 1] * m[..., 1]) / sd
    v = -(M[..., 0, 0] * m[..., 1] - M[..., 0, 1] * m[..., 0]) / sd
    k = C[..., 2, 2] + m[..., 0] * u + m[..., 1] * v
    valid &= np.isfinite(k) & (k != 0)
    Q = M / np.where(valid, -k, 1.0)[..., None, None]
    Q = np.where(valid[..., None, None], Q, np.eye(2))
    w, V = np.linalg.eigh(Q)
    valid &= w[..., 0] > 0
    w = np.where(w > 0, w, np.nan)
    A = 1.0 / np.sqrt(w[..., 0])
    B = 1.0 / np.sqrt(w[..., 1])
    theta = _wrap_half_pi(np.arctan2(V[..., 1, 0], V[..., 0, 0]))
    nan = np.where(valid, 1.0, np.nan)
    return u * nan, v * nan, A * nan, B * nan, theta * nan, valid


def rim_axes(e1, e2, psi):
    """Major- and minor-axis unit vectors (N, 3) of rims with in-plane axes e1, e2."""
    psi = np.atleast_1d(np.asarray(psi, float))
    c, s = np.cos(psi)[:, None], np.sin(psi)[:, None]
    return c * e1 + s * e2, -s * e1 + c * e2


def project_rims(K, R_wc, c, X_c, e1, e2, a, b, psi) -> dict:
    """Image ellipses of N rims seen from a camera at ``c`` with camera-to-world rotation ``R_wc``.

    Returns ``u, v`` (ellipse centre), ``A, B, theta``, ``D_px = 2 sqrt(A B)``, ``z`` (camera
    depth of the rim centre), ``uv0`` (N, 2: image of the rim centre), ``bias`` (N, 2:
    ellipse centre minus ``uv0``) and ``valid``: an ellipse whose whole rim is in front of
    the camera and whose rim plane is seen from above (camera on the normal side).
    """
    K = np.asarray(K, float)
    R = np.asarray(R_wc, float)
    c = np.asarray(c, float)
    X_c = np.asarray(X_c, float).reshape(-1, 3)
    e1 = np.asarray(e1, float).reshape(-1, 3)
    e2 = np.asarray(e2, float).reshape(-1, 3)
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    psi = np.atleast_1d(np.asarray(psi, float))
    n = X_c.shape[0]
    if n == 0:
        z0 = np.zeros(0)
        return {"u": z0, "v": z0.copy(), "A": z0.copy(), "B": z0.copy(), "theta": z0.copy(),
                "D_px": z0.copy(), "z": z0.copy(), "uv0": np.zeros((0, 2)), "bias": np.zeros((0, 2)),
                "valid": np.zeros(0, bool)}
    cols = np.stack([e1 @ R, e2 @ R, (X_c - c) @ R], axis=-1)  # camera-frame columns R^T e1, R^T e2, R^T (X_c - c)
    H = K @ cols
    zc = (X_c - c) @ R[:, 2]
    emaj, emin = rim_axes(e1, e2, psi)
    front = (zc - np.hypot(a * (emaj @ R[:, 2]), b * (emin @ R[:, 2]))) > 0
    w = c - X_c
    above = np.einsum("ij,ij->i", np.cross(e1, e2), w) > 1e-3 * np.linalg.norm(w, axis=1)
    ok = front & above & np.isfinite(H).all(axis=(1, 2))
    Hinv = np.linalg.inv(np.where(ok[:, None, None], H, np.eye(3)))
    C = np.swapaxes(Hinv, -1, -2) @ plane_conic(a, b, psi) @ Hinv
    u, v, A, B, th, valid = conic_to_ellipse(C)
    valid &= ok
    uv0, _ = project_uv(K, R, c, X_c)
    nan = np.where(valid, 1.0, np.nan)
    u, v, A, B, th = u * nan, v * nan, A * nan, B * nan, th * nan
    return {"u": u, "v": v, "A": A, "B": B, "theta": th, "D_px": 2.0 * np.sqrt(A * B), "z": zc,
            "uv0": uv0, "bias": np.column_stack([u - uv0[:, 0], v - uv0[:, 1]]), "valid": valid}


def fit_ellipse_direct(x, y):
    """Direct least-squares ellipse fit (Fitzgibbon 1999; numerically stable form of Halir & Flusser 1998).

    Returns ``(u, v, A, B, theta, ok)``; ``ok`` is False for fewer than 5 points or when no
    ellipse solution exists.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    bad = (np.nan,) * 5 + (False,)
    if x.size < 5 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return bad
    mx, my = x.mean(), y.mean()
    sc = float(np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2)))
    if sc <= 0:
        return bad
    xn, yn = (x - mx) / sc, (y - my) / sc
    D1 = np.column_stack([xn * xn, xn * yn, yn * yn])
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return bad
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    _, V = np.linalg.eig(M)
    V = np.real(V)
    cond = 4.0 * V[0] * V[2] - V[1] ** 2
    idx = np.nonzero(cond > 0)[0]
    if idx.size == 0:
        return bad
    a1 = V[:, idx[0]]
    qa, qb, qc, qd, qe, qf = np.concatenate([a1, T @ a1])
    Cn = np.array([[qa, qb / 2, qd / 2], [qb / 2, qc, qe / 2], [qd / 2, qe / 2, qf]])
    N = np.array([[1 / sc, 0, -mx / sc], [0, 1 / sc, -my / sc], [0, 0, 1.0]])
    u, v, A, B, th, ok = conic_to_ellipse((N.T @ Cn @ N)[None])
    return float(u[0]), float(v[0]), float(A[0]), float(B[0]), float(th[0]), bool(ok[0])


def ellipse_residual(x, y, u, v, A, B, theta) -> np.ndarray:
    """Approximate signed distance of points from an ellipse, in the units of x and y."""
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = np.asarray(x, float) - u, np.asarray(y, float) - v
    xp, yp = c * dx + s * dy, -s * dx + c * dy
    return (np.sqrt((xp / A) ** 2 + (yp / B) ** 2) - 1.0) * np.sqrt(A * B)
