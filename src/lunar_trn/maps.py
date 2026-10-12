"""Onboard TRN map: orthographic appearance map rendered from a DEM.

The map is a regular grid in south polar stereographic coordinates holding

* ``image``: rendered radiance (Hapke by default) for a given Sun direction, viewed from
  the local vertical (orthographic, nadir),
* ``heights``: terrain height above the reference sphere at the pixel centres, used to
  turn matched map pixels into 3D landmarks,
* ``shadow``: cast/attached shadow mask of the render.

Cast shadows are computed with an exact 2.5D horizon scan: heights are expressed in a
Cartesian tangent-plane frame at the window centre (so Earth-curvature-like drop of the
Moon is included), the grid is rotated so that the Sun azimuth points along +columns, and a
point is lit iff ``Z - u tan(el)`` is not exceeded by any point further towards the Sun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import frames
from .dem import Dem, DemSource, bilinear
from .render import reflectance


@dataclass
class TrnMap:
    image: np.ndarray  # (rows, cols) float32
    heights: np.ndarray  # (rows, cols) float32, metres above MOON_RADIUS_M
    x0: float  # easting of column 0 centre
    y0: float  # northing of row 0 centre
    pixel: float
    sun_dir: np.ndarray
    shadow: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape

    def xy_to_colrow(self, x, y):
        return (np.asarray(x) - self.x0) / self.pixel, (self.y0 - np.asarray(y)) / self.pixel

    def colrow_to_xy(self, col, row):
        return self.x0 + np.asarray(col) * self.pixel, self.y0 - np.asarray(row) * self.pixel

    def height_at_colrow(self, col, row) -> np.ndarray:
        return bilinear(self.heights, col, row)

    def world_points(self, col, row) -> np.ndarray:
        """Body-frame 3D points of fractional map pixels (DEM height interpolated)."""
        x, y = self.colrow_to_xy(col, row)
        return frames.stereo_to_body(x, y, self.height_at_colrow(col, row))

    def crop(self, c0: int, r0: int, width: int, height: int) -> "TrnMap":
        sl_ = (slice(r0, r0 + height), slice(c0, c0 + width))
        return TrnMap(self.image[sl_], self.heights[sl_], self.x0 + c0 * self.pixel, self.y0 - r0 * self.pixel,
                      self.pixel, self.sun_dir, None if self.shadow is None else self.shadow[sl_], dict(self.meta))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, image=self.image, heights=self.heights, x0=self.x0, y0=self.y0,
                            pixel=self.pixel, sun_dir=self.sun_dir,
                            shadow=self.shadow if self.shadow is not None else np.zeros((0, 0), bool),
                            meta=np.array([repr(self.meta)]))

    @classmethod
    def load(cls, path: str | Path) -> "TrnMap":
        import ast

        with np.load(path, allow_pickle=False) as d:
            shadow = d["shadow"] if d["shadow"].size else None
            meta = ast.literal_eval(str(d["meta"][0]))
            return cls(d["image"], d["heights"], float(d["x0"]), float(d["y0"]), float(d["pixel"]),
                       d["sun_dir"], shadow, meta)


def tangent_frame(center_xy: tuple[float, float], h: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Origin and 3x3 basis (columns: map +x, map +y, up) of the tangent plane at a map point."""
    x, y = center_xy
    p = frames.stereo_to_body(x, y, h)
    up = p / np.linalg.norm(p)
    ex = frames.stereo_to_body(x + 1.0, y, h) - frames.stereo_to_body(x - 1.0, y, h)
    ex -= (ex @ up) * up
    ex /= np.linalg.norm(ex)
    ey = np.cross(up, ex)
    return p, np.column_stack([ex, ey, up])


def cast_shadows_grid(dem: Dem, sun_dir, eps: float = 0.05) -> np.ndarray:
    """Cast-shadow mask on the DEM grid for a distant Sun (exact 2.5D horizon scan)."""
    import cv2

    rows, cols = dem.shape
    xmin, xmax, ymin, ymax = dem.extent
    centre = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
    origin, B = tangent_frame(centre)
    s_t = np.asarray(sun_dir, dtype=np.float64) @ B
    horiz = math.hypot(s_t[0], s_t[1])
    if horiz < 1e-9:
        return np.zeros((rows, cols), dtype=bool)
    tan_el = s_t[2] / horiz
    if tan_el <= 0:
        return np.ones((rows, cols), dtype=bool)
    # Tangent-plane height of every DEM node (includes the sphere drop r^2 / 2R).
    Z = np.empty((rows, cols), dtype=np.float32)
    xs = dem.x0 + np.arange(cols) * dem.pixel
    for r0 in range(0, rows, 512):
        r1 = min(r0 + 512, rows)
        X, Y = np.meshgrid(xs, dem.y0 - np.arange(r0, r1) * dem.pixel)
        P = frames.stereo_to_body(X, Y, np.nan_to_num(dem.heights[r0:r1].astype(np.float64), nan=-1e5))
        Z[r0:r1] = (P - origin) @ B[:, 2]
    # Sun azimuth in the grid: +x is +col, +y is -row.
    phi = math.degrees(math.atan2(s_t[1], s_t[0]))
    diag = int(math.ceil(math.hypot(rows, cols))) + 4
    M = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), -phi, 1.0)
    # The rotation above maps the Sun direction (angle phi from +x, counter-clockwise in map
    # coordinates, i.e. clockwise in image rows) onto +col. Shift into a square canvas.
    M[0, 2] += diag / 2.0 - cols / 2.0
    M[1, 2] += diag / 2.0 - rows / 2.0
    fill = -1e7
    Zr = cv2.warpAffine(Z, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    ground_px = dem.pixel / frames.stereo_scale_factor(*centre)
    u = np.arange(diag, dtype=np.float32) * np.float32(ground_px * tan_el)
    g = Zr - u[None, :]
    # Suffix maximum excluding the point itself (points further towards the Sun, +col).
    rev = np.maximum.accumulate(g[:, ::-1], axis=1)[:, ::-1]
    suffix = np.full_like(g, -np.inf)
    suffix[:, :-1] = rev[:, 1:]
    shadow_r = (g < suffix - eps).astype(np.float32)
    Minv = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(shadow_r, Minv, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0.0)
    return back > 0.5


def shade_grid(dem: Dem, sun_dir, model: str = "hapke", params: dict | None = None,
               shadows: bool = True, chunk_rows: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic (nadir-view) radiance on the DEM grid and the shadow mask (row chunks).

    Normals use the conformal-projection closed form: with u the unit radial vector and
    a, b the unit images of the map x/y axes (a x b = u),
    ``n ~ u - (h_x a + h_y b) / q`` with ``q = (R + h) 4R / D``, ``D = 4R^2 + rho^2``.
    """
    rows, cols = dem.shape
    R = dem.radius
    s = np.asarray(sun_dir, dtype=np.float64)
    s = s / np.linalg.norm(s)
    gx, gy = dem.gradient()
    I = np.zeros((rows, cols), dtype=np.float32)
    attached = np.zeros((rows, cols), dtype=bool)
    xs = dem.x0 + np.arange(cols) * dem.pixel
    four_r2 = 4.0 * R * R
    for r0 in range(0, rows, chunk_rows):
        r1 = min(r0 + chunk_rows, rows)
        ys = dem.y0 - np.arange(r0, r1) * dem.pixel
        X, Y = np.meshgrid(xs, ys)
        rho2 = X * X + Y * Y
        D = four_r2 + rho2
        # Unit radial vector u (body X, Y, Z) and tangents a = d/dx, b = d/dy (unit).
        uX, uY, uZ = 4 * R * Y / D, 4 * R * X / D, -(four_r2 - rho2) / D
        aX, aY, aZ = -2 * X * Y / D, 1 - 2 * X * X / D, 4 * R * X / D
        bX, bY, bZ = 1 - 2 * Y * Y / D, -2 * X * Y / D, 4 * R * Y / D
        h = dem.heights[r0:r1].astype(np.float64)
        inv_q = D / ((R + h) * 4 * R)
        hx = gx[r0:r1] * inv_q
        hy = gy[r0:r1] * inv_q
        nX = uX - hx * aX - hy * bX
        nY = uY - hx * aY - hy * bY
        nZ = uZ - hx * aZ - hy * bZ
        inv_norm = 1.0 / np.sqrt(nX * nX + nY * nY + nZ * nZ)
        cos_i = (nX * s[0] + nY * s[1] + nZ * s[2]) * inv_norm
        cos_e = inv_norm  # n . u = 1 before normalisation
        cos_g = uX * s[0] + uY * s[1] + uZ * s[2]
        Ic = reflectance(cos_i, cos_e, cos_g, model=model, params=params)
        I[r0:r1] = np.where(np.isnan(Ic), 0.0, Ic)
        attached[r0:r1] = cos_i <= 0
    shadow = attached
    if shadows:
        shadow = shadow | cast_shadows_grid(dem, s)
    I[shadow] = 0.0
    return I, shadow


def resample_regular(dem: Dem, x0: float, y0: float, pixel: float, nrows: int, ncols: int) -> np.ndarray:
    """Bilinear heights of ``dem`` on a regular north-up grid (OpenCV warpAffine, NaN outside)."""
    import cv2

    a = pixel / dem.pixel
    M = np.array([[a, 0.0, (x0 - dem.x0) / dem.pixel], [0.0, a, (dem.y0 - y0) / dem.pixel]], dtype=np.float64)
    src = np.ascontiguousarray(dem.heights, dtype=np.float32)
    out = np.empty((nrows, ncols), dtype=np.float32)
    # warpAffine needs both dimensions below SHRT_MAX; tile rows if necessary.
    step = 16000
    for r0 in range(0, nrows, step):
        r1 = min(r0 + step, nrows)
        Mr = M.copy()
        Mr[1, 2] += a * r0
        out[r0:r1] = cv2.warpAffine(src, Mr, (ncols, r1 - r0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    return out


def render_map(src: DemSource | Dem, center_xy: tuple[float, float], half_size_m: float, pixel: float,
               sun_dir, model: str = "hapke", params: dict | None = None,
               shadow_reach_m: float = 12000.0, name: str = "") -> TrnMap:
    """Render one orthographic TRN map; see :func:`render_maps`."""
    return render_maps(src, center_xy, half_size_m, [pixel], sun_dir, model, params, shadow_reach_m, [name])[0]


def render_maps(src: DemSource | Dem, center_xy: tuple[float, float], half_size_m: float, pixels,
                sun_dir, model: str = "hapke", params: dict | None = None,
                shadow_reach_m: float = 12000.0, names=None) -> list[TrnMap]:
    """Render orthographic TRN maps centred at ``center_xy`` (stereographic metres).

    Each map gets its own working grid aligned with its output grid: ``work = pixel / k``
    with ``k = 2`` when that is still no finer than the DEM, else ``k = 1``. Heights on the
    working grid come from the DEM, block-mean prefiltered when the working pixel spans two
    or more DEM pixels, then bilinearly resampled. The working grid is shaded (with cast
    shadows) and box-averaged k x k to the output pixel. The DEM window is extended by
    ``shadow_reach_m`` towards the Sun so that occluders outside the map still shade it.
    """
    s = np.asarray(sun_dir, dtype=np.float64)
    s = s / np.linalg.norm(s)
    cx, cy = center_xy
    # Sun direction projected on the map plane.
    _, B = tangent_frame(center_xy)
    st = s @ B
    hz = math.hypot(st[0], st[1])
    dx, dy = (st[0] / hz * shadow_reach_m, st[1] / hz * shadow_reach_m) if hz > 1e-9 else (0.0, 0.0)
    pixels = list(pixels)
    names = list(names) if names is not None else [""] * len(pixels)
    margin = 2.0 * max(pixels) + 50.0
    xmin, xmax = cx - half_size_m - margin, cx + half_size_m + margin
    ymin, ymax = cy - half_size_m - margin, cy + half_size_m + margin
    wxmin, wxmax = min(xmin, xmin + dx), max(xmax, xmax + dx)
    wymin, wymax = min(ymin, ymin + dy), max(ymax, ymax + dy)
    dem = src.read_window(wxmin, wxmax, wymin, wymax) if isinstance(src, DemSource) else src
    out = []
    for px, nm in zip(pixels, names):
        k = 2 if px / 2.0 >= dem.pixel * 0.999 else 1
        w = px / k
        m = int(math.floor(w / dem.pixel + 1e-9))
        dem_f = dem.downsample(m) if m >= 2 else dem
        # Output grid (pixel centres) and the aligned working grid covering the DEM window.
        n = int(round(2 * half_size_m / px))
        x0 = cx - half_size_m + 0.5 * px
        y0 = cy + half_size_m - 0.5 * px
        corner_x, corner_y = x0 - 0.5 * px, y0 + 0.5 * px  # top-left corner of the output grid
        fx0, fx1, fy0, fy1 = dem_f.extent
        left = max(int(math.floor((corner_x - fx0) / w)), 0)
        top = max(int(math.floor((fy1 - corner_y) / w)), 0)
        ncw = left + n * k + max(int(math.floor((fx1 - (corner_x + n * px)) / w)), 0)
        nrw = top + n * k + max(int(math.floor(((corner_y - n * px) - fy0) / w)), 0)
        wx0 = corner_x - left * w + 0.5 * w
        wy0 = corner_y + top * w - 0.5 * w
        hw = resample_regular(dem_f, wx0, wy0, w, nrw, ncw)
        dem_w = Dem(hw, wx0, wy0, w, name=f"{dem.name}_w{w:g}", radius=dem.radius)
        I, shadow = shade_grid(dem_w, s, model=model, params=params)

        def box(a):
            a = a[top:top + n * k, left:left + n * k].astype(np.float64)
            return a.reshape(n, k, n, k).mean(axis=(1, 3))

        img = box(I).astype(np.float32)
        sh = box(shadow.astype(np.float32)) > 0.5
        h = resample_regular(dem_f, x0, y0, px, n, n)
        out.append(TrnMap(img, h, x0, y0, px, s, sh,
                          {"name": nm, "model": model, "work_pixel": float(w), "dem": dem.name,
                           "dem_prefilter": m, "center_xy": (float(cx), float(cy)),
                           "half_size_m": float(half_size_m)}))
    return out
