"""Simple physically motivated terrain rendering from a DEM.

Used for (a) the V1 overlay test (perspective render from a dataset pose) and (b) the
onboard map generator (orthographic render, see :mod:`lunar_trn.maps`).

Reflectance models
------------------
* ``lambert``: ``cos(i)``.
* ``lunar_lambert``: McEwen (1991) form ``2 L cos(i) / (cos(i) + cos(e)) + (1 - L) cos(i)``
  with ``L(g)`` from the McEwen polynomial fit for the Moon.
* ``hapke``: isotropic-multiple-scattering Hapke (1993/2012 style) with a one-term
  Henyey-Greenstein phase function, shadow-hiding opposition surge and macroscopic roughness
  omitted; parameters default to lunar-average values (``HAPKE_DEFAULT``).

Cast shadows are traced with :func:`lunar_trn.dem.raycast` towards the Sun, so the DEM window
must extend far enough in the Sun direction to contain the occluders.
"""

from __future__ import annotations

import numpy as np

from . import frames
from .camera import PinholeCamera
from .dem import Dem, bilinear

# Lunar average Hapke parameters (single scattering albedo w, HG asymmetry b, opposition
# amplitude B0 and width h). Sato et al. (2014) LROC WAC global averages are close to these.
HAPKE_DEFAULT = {"w": 0.25, "b": -0.25, "B0": 1.0, "h": 0.06}


def stereo_tangents(x, y, radius: float = frames.MOON_RADIUS_M):
    """Unit-sphere point u(x, y) and its analytic derivatives du/dx, du/dy (per map metre).

    From the inverse projection u = (4R y, 4R x, -(4R^2 - rho^2)) / (4R^2 + rho^2) in body
    (X, Y, Z) order.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    four_r2 = 4.0 * radius * radius
    rho2 = x * x + y * y
    D = four_r2 + rho2
    D2 = D * D
    u = np.stack([4 * radius * y / D, 4 * radius * x / D, -(four_r2 - rho2) / D], axis=-1)
    ux = np.stack([-8 * radius * x * y / D2, 4 * radius / D - 8 * radius * x * x / D2, 4 * four_r2 * x / D2], axis=-1)
    uy = np.stack([4 * radius / D - 8 * radius * y * y / D2, -8 * radius * x * y / D2, 4 * four_r2 * y / D2], axis=-1)
    return u, ux, uy


def surface_normals(dem: Dem, x, y) -> np.ndarray:
    """Outward unit normals (..., 3) in the body frame at stereographic points (x, y)."""
    gx, gy = dem.gradient()
    col, row = dem.xy_to_colrow(x, y)
    hx = bilinear(gx, col, row)
    hy = bilinear(gy, col, row)
    h = dem.height_at(x, y)
    # P(x, y) = (R + h) u(x, y)  =>  dP/dx = (R + h) du/dx + h_x u, same for y.
    u, ux, uy = stereo_tangents(x, y, dem.radius)
    r = (dem.radius + h)[..., None]
    tx = r * ux + hx[..., None] * u
    ty = r * uy + hy[..., None] * u
    n = np.cross(tx, ty)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def lunar_lambert_L(phase_deg) -> np.ndarray:
    """McEwen (1991) limb-darkening parameter L(g) for the Moon (valid ~0-140 deg)."""
    g = np.asarray(phase_deg, dtype=np.float64)
    L = 1.0 - 0.019 * g + 2.42e-4 * g**2 - 1.46e-6 * g**3
    return np.clip(L, 0.0, 1.0)


def reflectance(cos_i, cos_e, cos_g, model: str = "lunar_lambert", params: dict | None = None) -> np.ndarray:
    """Relative radiance factor for incidence/emission/phase cosines. Zero where cos_i <= 0."""
    cos_i = np.asarray(cos_i, dtype=np.float64)
    cos_e = np.clip(np.asarray(cos_e, dtype=np.float64), 1e-6, 1.0)
    lit = cos_i > 0
    ci = np.where(lit, cos_i, 0.0)
    if model == "lambert":
        out = ci
    elif model == "lunar_lambert":
        g = np.rad2deg(np.arccos(np.clip(cos_g, -1.0, 1.0)))
        L = lunar_lambert_L(g)
        out = 2.0 * L * ci / (ci + cos_e) + (1.0 - L) * ci
    elif model == "hapke":
        p = dict(HAPKE_DEFAULT, **(params or {}))
        out = _hapke(ci, cos_e, np.asarray(cos_g, dtype=np.float64), **p)
    else:
        raise ValueError(f"unknown reflectance model {model}")
    return np.where(lit, out, 0.0)


def _hapke(mu0, mu, cos_g, w, b, B0, h):
    g = np.arccos(np.clip(cos_g, -1.0, 1.0))
    # One-term Henyey-Greenstein; b < 0 is back-scattering with this sign convention.
    P = (1 - b**2) / (1 + 2 * b * np.cos(g) + b**2) ** 1.5
    B = B0 / (1 + np.tan(g / 2) / h)
    gamma = np.sqrt(1 - w)

    def H(x):
        return (1 + 2 * x) / (1 + 2 * gamma * x)

    return w / (4 * np.pi) * mu0 / (mu0 + mu) * ((1 + B) * P + H(mu0) * H(mu) - 1)


def cast_shadow(dem: Dem, points, sun_dir, offset: float = 1.0, tol: float = 0.25,
                min_step: float = 1.0) -> np.ndarray:
    """True where the straight path from a surface point towards the Sun hits terrain."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    s = np.broadcast_to(np.asarray(sun_dir, dtype=np.float64), p.shape)
    up = p / np.linalg.norm(p, axis=1, keepdims=True)
    t = dem.raycast(p + offset * up, s, tol=tol, min_step=min_step)
    return np.isfinite(t)


def shade_points(dem: Dem, points, view_dirs, sun_dir, model: str = "lunar_lambert",
                 shadows: bool = True, params: dict | None = None) -> dict:
    """Shade surface points. ``view_dirs`` point from the surface towards the observer.

    Returns a dict with the radiance ``I`` plus raw geometry (cos_i, cos_e, phase, shadow).
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x, y, _ = frames.body_to_stereo(p, dem.radius)
    n = surface_normals(dem, x, y)
    s = np.asarray(sun_dir, dtype=np.float64)
    s = s / np.linalg.norm(s)
    v = np.asarray(view_dirs, dtype=np.float64).reshape(-1, 3)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    cos_i = n @ s
    cos_e = np.einsum("ij,ij->i", n, v)
    cos_g = v @ s
    I = reflectance(cos_i, cos_e, cos_g, model=model, params=params)
    shadow = np.zeros(p.shape[0], dtype=bool)
    if shadows:
        cand = cos_i > 0
        if cand.any():
            shadow[cand] = cast_shadow(dem, p[cand], s)
        I = np.where(shadow, 0.0, I)
    return {"I": I, "cos_i": cos_i, "cos_e": cos_e, "cos_g": cos_g, "shadow": shadow | (cos_i <= 0)}


def perspective_hits(dem: Dem, cam: PinholeCamera, T_world_cam: np.ndarray, step: int = 1) -> dict:
    """Cast the camera rays of every ``step``-th pixel (pixel indices ``step // 2 + k step``)."""
    offset = step // 2
    cols, rows = cam.pixel_grid(step=step, offset=float(offset))
    o, d = cam.rays_world(T_world_cam, cols, rows)
    dflat = d.reshape(-1, 3)
    t = dem.raycast(np.broadcast_to(o, dflat.shape), dflat)
    hit = np.isfinite(t)
    pts = o + np.where(hit, t, 0.0)[:, None] * dflat
    x, y, _ = frames.body_to_stereo(pts[hit], dem.radius)
    normals = np.full_like(pts, np.nan)
    normals[hit] = surface_normals(dem, x, y)
    return {"cols": cols, "rows": rows, "shape": cols.shape, "t": t, "dirs": dflat, "hit": hit,
            "points": pts, "normals": normals,
            "zdepth": t * (dflat @ np.asarray(T_world_cam, dtype=np.float64)[:3, 2])}


def shade_hits(dem: Dem, hits: dict, sun_dir, model: str = "lunar_lambert", shadows: bool = True,
               params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shade precomputed perspective hits. Returns (image, shadow) shaped like the pixel grid."""
    s = np.asarray(sun_dir, dtype=np.float64)
    s = s / np.linalg.norm(s)
    hit = hits["hit"]
    n = hits["normals"][hit]
    v = -hits["dirs"][hit]
    cos_i = n @ s
    I = reflectance(cos_i, np.einsum("ij,ij->i", n, v), v @ s, model=model, params=params)
    shadow = cos_i <= 0
    if shadows:
        cand = ~shadow
        if cand.any():
            cast = cast_shadow(dem, hits["points"][hit][cand], s)
            shadow[cand] = cast
        I = np.where(shadow, 0.0, I)
    img = np.full(hit.shape[0], np.nan)
    img[hit] = I
    sh = np.zeros(hit.shape[0], dtype=bool)
    sh[hit] = shadow
    return img.reshape(hits["shape"]), sh.reshape(hits["shape"])


def render_perspective(dem: Dem, cam: PinholeCamera, T_world_cam: np.ndarray, sun_dir,
                       step: int = 1, model: str = "lunar_lambert", shadows: bool = True,
                       params: dict | None = None) -> dict:
    """Render the DEM as seen by ``cam`` at ``T_world_cam``.

    Returns a dict with ``image`` (H', W'), ``range`` (distance along the ray), ``zdepth``,
    ``cols``/``rows`` (pixel indices sampled), ``points`` (hit points) and ``shadow``.
    """
    hits = perspective_hits(dem, cam, T_world_cam, step=step)
    img, shadow = shade_hits(dem, hits, sun_dir, model=model, shadows=shadows, params=params)
    shape = hits["shape"]
    return {
        "image": img,
        "range": hits["t"].reshape(shape),
        "zdepth": hits["zdepth"].reshape(shape),
        "cols": hits["cols"],
        "rows": hits["rows"],
        "points": hits["points"].reshape(shape + (3,)),
        "shadow": shadow,
        "hits": hits,
    }
