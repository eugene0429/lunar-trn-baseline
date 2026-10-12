"""Synthetic crater terrain, catalogs and scenes for tests and controlled experiments."""

from __future__ import annotations

import numpy as np

from ..dem import Dem


def crater_height(rho, D: float, dD: float, rim_frac: float = 0.04) -> np.ndarray:
    """Radial height profile (m): parabolic bowl inside the rim crest, (r/rho)^3 ejecta outside.

    The crest is at rho = D/2 with height ``rim_frac * D``; crest-to-floor depth is ``dD * D``.
    """
    r = 0.5 * D
    rim = rim_frac * D
    rho = np.asarray(rho, float)
    inside = rim - dD * D * (1.0 - (rho / r) ** 2)
    outside = rim * (r / np.maximum(rho, r)) ** 3
    return np.where(rho <= r, inside, outside)


def place_craters(rng, n: int, extent, D_range, dD_range, margin_r: float = 3.0, spacing: float = 1.15,
                  tries_per_crater: int = 400) -> list[dict]:
    """Non-overlapping craters, log-uniform diameters, largest placed first."""
    xmin, xmax, ymin, ymax = extent
    Ds = np.sort(np.exp(rng.uniform(np.log(D_range[0]), np.log(D_range[1]), n)))[::-1]
    out: list[dict] = []
    for D in Ds:
        r = 0.5 * D
        lo_x, hi_x, lo_y, hi_y = xmin + margin_r * r, xmax - margin_r * r, ymin + margin_r * r, ymax - margin_r * r
        if lo_x >= hi_x or lo_y >= hi_y:
            continue
        for _ in range(tries_per_crater):
            x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            if all(np.hypot(x - c["x"], y - c["y"]) >= spacing * (r + 0.5 * c["D"]) for c in out):
                out.append({"x": float(x), "y": float(y), "D": float(D), "dD": float(rng.uniform(*dD_range))})
                break
    return out


def synthetic_crater_dem(n: int = 1600, pixel: float = 5.0, cx: float = -20000.0, cy: float = 30000.0,
                         n_craters: int = 40, D_range=(60.0, 1500.0), dD_range=(0.08, 0.2),
                         slope=(0.01, -0.005), noise_m: float = 0.0, seed: int = 0) -> tuple[Dem, list[dict]]:
    """Square DEM (n x n, centred at stereographic (cx, cy)) with planted bowl craters on a slope."""
    rng = np.random.default_rng(seed)
    x0 = cx - (n - 1) / 2 * pixel
    y0 = cy + (n - 1) / 2 * pixel
    xs = x0 + np.arange(n) * pixel
    ys = y0 - np.arange(n) * pixel
    X, Y = np.meshgrid(xs, ys)
    H = slope[0] * (X - cx) + slope[1] * (Y - cy)
    craters = place_craters(rng, n_craters, (xs[0], xs[-1], ys[-1], ys[0]), D_range, dD_range)
    for c in craters:
        w = int(np.ceil(2.0 * c["D"] / pixel))
        ci = int(round((c["x"] - x0) / pixel))
        ri = int(round((y0 - c["y"]) / pixel))
        sl = (slice(max(ri - w, 0), min(ri + w + 1, n)), slice(max(ci - w, 0), min(ci + w + 1, n)))
        H[sl] += crater_height(np.hypot(X[sl] - c["x"], Y[sl] - c["y"]), c["D"], c["dD"])
    if noise_m > 0:
        H += noise_m * rng.normal(size=H.shape)
    return Dem(H.astype(np.float32), x0, y0, pixel, name="synthetic_craters"), craters


from .. import frames  # noqa: E402
from ..camera import PinholeCamera  # noqa: E402
from .catalog import CraterCatalog, make_catalog, rim_frame  # noqa: E402


def flat_catalog(rng, n: int, extent, D_range, h: float = 0.0) -> CraterCatalog:
    """Circular rims on the reference sphere (height ``h``), non-overlapping, log-uniform sizes."""
    cr = place_craters(rng, n, extent, D_range, (0.1, 0.2), margin_r=0.0, spacing=1.05)
    x = np.array([c["x"] for c in cr])
    y = np.array([c["y"] for c in cr])
    D = np.array([c["D"] for c in cr])
    X = frames.stereo_to_body(x, y, np.full(x.shape, h))
    nrm = X / np.linalg.norm(X, axis=1, keepdims=True)
    e1, e2 = rim_frame(x, y, nrm)
    return make_catalog(X, nrm, e1, e2, D / 2, D / 2, np.zeros(x.size),
                        d_over_D=[c["dD"] for c in cr], completeness=np.ones(x.size))


def nadir_pose(cx: float, cy: float, alt: float) -> np.ndarray:
    """Camera-to-world pose looking straight down from ``alt`` above stereographic (cx, cy)."""
    p = frames.stereo_to_body(cx, cy, alt)
    down = -p / np.linalg.norm(p)
    xc = np.cross(down, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([xc, np.cross(down, xc), down])
    T[:3, 3] = p
    return T


def default_camera(size: int = 512) -> PinholeCamera:
    """StereoLunar-like camera: 45 deg field of view, principal point at the image centre."""
    f = size / 2 / np.tan(np.deg2rad(22.5))
    return PinholeCamera(np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]), size, size)


def render_synthetic_scene(alt: float = 4000.0, sun_az: float = 135.0, sun_el: float = 20.0, n: int = 1400,
                           n_craters: int = 150, D_range=(60.0, 600.0), seed: int = 0, size: int = 512) -> dict:
    """Hapke render of a synthetic crater DEM from a nadir camera (min-max normalised like StereoLunar)."""
    from ..render import render_perspective
    from ..validation.v1 import sun_from_az_el

    cx, cy = -20000.0, 30000.0
    dem, craters = synthetic_crater_dem(n=n, cx=cx, cy=cy, n_craters=n_craters, D_range=D_range,
                                        dD_range=(0.12, 0.2), noise_m=0.2, seed=seed)
    cam, T = default_camera(size), nadir_pose(cx, cy, alt)
    sun = sun_from_az_el(T[:3, 3], sun_az, sun_el)
    img = render_perspective(dem, cam, T, sun, step=1, model="hapke")["image"]
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    return {"dem": dem, "craters": craters, "cam": cam, "T": T, "sun": sun,
            "image": np.nan_to_num(img).astype(np.float32)}
