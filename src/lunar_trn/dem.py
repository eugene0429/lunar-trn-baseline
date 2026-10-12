"""Digital elevation model access, sampling and ray casting.

A :class:`DemSource` wraps a large polar stereographic raster on disk (GeoTIFF or PDS3 IMG)
and reads windows from it as in-memory :class:`Dem` objects. A :class:`Dem` stores heights
above the reference sphere on a regular grid in stereographic coordinates.

Grid convention: ``heights[r, c]`` is the value at pixel centre
``x = x0 + c * pixel``, ``y = y0 - r * pixel`` (row 0 is the largest northing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import frames


@dataclass
class Dem:
    """In-memory DEM window on a regular south polar stereographic grid."""

    heights: np.ndarray  # (rows, cols) float32, height above MOON_RADIUS_M in metres; NaN = no data
    x0: float  # easting of the centre of column 0 (m)
    y0: float  # northing of the centre of row 0 (m)
    pixel: float  # grid spacing (m)
    name: str = ""
    radius: float = frames.MOON_RADIUS_M
    _pyramid: list | None = field(default=None, repr=False, compare=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of pixel centres."""
        rows, cols = self.heights.shape
        return self.x0, self.x0 + (cols - 1) * self.pixel, self.y0 - (rows - 1) * self.pixel, self.y0

    def xy_to_colrow(self, x, y):
        return (np.asarray(x) - self.x0) / self.pixel, (self.y0 - np.asarray(y)) / self.pixel

    def colrow_to_xy(self, col, row):
        return self.x0 + np.asarray(col) * self.pixel, self.y0 - np.asarray(row) * self.pixel

    def height_at(self, x, y) -> np.ndarray:
        """Bilinear height at stereographic (x, y). NaN outside the grid or next to no-data."""
        col, row = self.xy_to_colrow(x, y)
        return bilinear(self.heights, col, row)

    def max_slope(self) -> float:
        """Upper bound of |grad h| (m/m) of the bilinear surface."""
        h = self.heights.astype(np.float64)
        gx = np.nanmax(np.abs(np.diff(h, axis=1))) / self.pixel
        gy = np.nanmax(np.abs(np.diff(h, axis=0))) / self.pixel
        return float(np.hypot(gx, gy))

    def height_range(self) -> tuple[float, float]:
        return float(np.nanmin(self.heights)), float(np.nanmax(self.heights))

    def gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Central-difference gradient (dh/dx_east, dh/dy_north) in map metres, per pixel."""
        h = self.heights.astype(np.float64)
        dh_drow, dh_dcol = np.gradient(h)
        return dh_dcol / self.pixel, -dh_drow / self.pixel

    def pyramid(self) -> list[dict]:
        """Max-height and max-slope pyramid used by :func:`raycast` (built once, cached).

        Level L has cells of ``2**L`` pixels; ``hmax``/``gmax`` hold the maximum over the
        3x3 cell neighbourhood, which covers every point within ``rho = (2**L - 1) * pixel``
        of any point in the centre cell, including the bilinear support.
        """
        if self._pyramid is None:
            self._pyramid = build_pyramid(self.heights, self.pixel)
        return self._pyramid

    def downsample(self, factor: int) -> "Dem":
        """Block-mean decimation by an integer factor (NaN-aware)."""
        if factor <= 1:
            return self
        rows, cols = self.heights.shape
        nr, nc = rows // factor, cols // factor
        h = self.heights[: nr * factor, : nc * factor].astype(np.float64)
        blocks = h.reshape(nr, factor, nc, factor)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(blocks, axis=(1, 3))
        half = 0.5 * (factor - 1) * self.pixel
        return Dem(heights=mean.astype(np.float32), x0=self.x0 + half, y0=self.y0 - half,
                   pixel=self.pixel * factor, name=f"{self.name}_x{factor}", radius=self.radius)

    def raycast(self, origins, dirs, **kwargs) -> np.ndarray:
        """Distance along each ray to the first terrain hit; NaN when it misses. See :func:`raycast`."""
        return raycast(self, origins, dirs, **kwargs)


def build_pyramid(heights: np.ndarray, pixel: float, min_cell: int = 2) -> list[dict]:
    from scipy.ndimage import maximum_filter

    h = np.where(np.isnan(heights), -np.inf, heights).astype(np.float32)
    # Per-pixel slope bound: largest absolute difference to the right/down neighbours,
    # attributed to both pixels of each pair.
    g = np.zeros_like(h)
    dx = np.abs(np.diff(h, axis=1)) / pixel
    dy = np.abs(np.diff(h, axis=0)) / pixel
    dx = np.where(np.isfinite(dx), dx, 0.0)
    dy = np.where(np.isfinite(dy), dy, 0.0)
    g[:, :-1] = np.maximum(g[:, :-1], dx)
    g[:, 1:] = np.maximum(g[:, 1:], dx)
    g[:-1, :] = np.maximum(g[:-1, :], dy)
    g[1:, :] = np.maximum(g[1:, :], dy)
    # A bilinear patch has |grad| <= hypot(max |dx|, max |dy|) of its corners.
    g = g * np.sqrt(2.0)
    levels = []
    cell = min_cell
    rows, cols = h.shape
    while cell < 2 * max(rows, cols):
        nr = -(-rows // cell)
        nc = -(-cols // cell)
        hp = np.full((nr * cell, nc * cell), -np.inf, dtype=np.float32)
        gp = np.zeros((nr * cell, nc * cell), dtype=np.float32)
        hp[:rows, :cols] = h
        gp[:rows, :cols] = g
        hm = hp.reshape(nr, cell, nc, cell).max(axis=(1, 3))
        gm = gp.reshape(nr, cell, nc, cell).max(axis=(1, 3))
        hm = maximum_filter(hm, size=3, mode="nearest")
        gm = maximum_filter(gm, size=3, mode="nearest")
        levels.append({"cell": cell, "rho": (cell - 1) * pixel, "hmax": hm, "gmax": gm})
        cell *= 2
    return levels


def bilinear(grid: np.ndarray, col, row) -> np.ndarray:
    """Bilinear interpolation of ``grid`` at fractional (col, row); NaN outside the grid."""
    col = np.asarray(col, dtype=np.float64)
    row = np.asarray(row, dtype=np.float64)
    rows, cols = grid.shape
    inside = (col >= 0) & (row >= 0) & (col <= cols - 1) & (row <= rows - 1)
    ci = np.clip(np.floor(np.nan_to_num(col)), 0, cols - 2).astype(np.intp)
    ri = np.clip(np.floor(np.nan_to_num(row)), 0, rows - 2).astype(np.intp)
    fc = col - ci
    fr = row - ri
    g = grid
    v = ((1 - fr) * ((1 - fc) * g[ri, ci] + fc * g[ri, ci + 1])
         + fr * ((1 - fc) * g[ri + 1, ci] + fc * g[ri + 1, ci + 1]))
    return np.where(inside, v, np.nan)


def raycast(
    dem: Dem,
    origins,
    dirs,
    tol: float = 0.02,
    max_iter: int = 600,
    safety: float = 0.9,
    min_step: float | None = None,
) -> np.ndarray:
    """First intersection of rays with the DEM surface.

    Rays are given in the Moon-fixed body frame. The clearance
    ``f(t) = |o + t d| - R - h(o + t d)`` is positive above the terrain. Every step is
    provably safe (cannot jump past the first crossing) and is the largest of:

    * a max-height step: inside the 3x3-cell neighbourhood of pyramid level L (radius
      ``rho_L``) the terrain is below ``Hmax_L``; since ``|o + t d|`` is convex in t, the ray
      height is bounded below by its tangent line, so the ray may advance while that line
      stays above ``Hmax_L`` and the horizontal travel stays within ``rho_L``;
    * a local Lipschitz step ``safety * f / (|w| + sin * G_L)`` with ``G_L`` the slope bound in
      the same neighbourhood, also limited to ``rho_L`` of horizontal travel.

    Parameters
    ----------
    origins, dirs : (N, 3) arrays; ``dirs`` need not be normalised.
    tol : clearance (m) at which a ray is declared to have hit.
    min_step : smallest step (m); defaults to ``tol / 2``. Larger values speed up grazing
        shadow rays at the cost of possibly missing occluders thinner than the step.

    Returns
    -------
    t : (N,) distance along the normalised ray direction, NaN where the ray misses the DEM
        window, leaves the valid data, or climbs out of the terrain height shell.
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    n = o.shape[0]
    hmin, hmax = dem.height_range()
    pyr = dem.pyramid()
    min_step = 0.5 * tol if min_step is None else min_step
    horiz_k = 1.01  # map metres per metre of horizontal ray travel (scale factor + margin)

    # March only inside the shell R + hmin <= |p| <= R + hmax.
    r_out = dem.radius + hmax + 1.0
    r_in = dem.radius + hmin - 1.0
    t_out_1, t_out_2 = _sphere_roots(o, d, r_out)
    t_in_1, _ = _sphere_roots(o, d, r_in)
    t_start = np.where(np.isnan(t_out_1), np.nan, np.maximum(t_out_1, 0.0))
    # A ray that reaches the inner sphere must have hit the terrain before it; otherwise it
    # leaves the shell where it crosses the outer sphere again.
    t_end = np.where(np.isfinite(t_in_1) & (t_in_1 >= 0), t_in_1, t_out_2)
    t_start = np.where(t_end < 0, np.nan, t_start)
    t = t_start.copy()
    result = np.full(n, np.nan)
    active = ~np.isnan(t)

    for _ in range(max_iter):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        ti = t[idx]
        di = d[idx]
        p = o[idx] + ti[:, None] * di
        x, y, hp = frames.body_to_stereo(p, dem.radius)
        col, row = dem.xy_to_colrow(x, y)
        ht = bilinear(dem.heights, col, row)
        f = hp - ht
        lost = np.isnan(f) | (ti > t_end[idx])
        hit = ~lost & (f < tol)
        result[idx[hit]] = ti[hit]
        up = p / np.linalg.norm(p, axis=1, keepdims=True)
        w = np.einsum("ij,ij->i", di, up)  # vertical rate of the ray height (tangent line)
        sin_r = np.sqrt(np.clip(1.0 - w**2, 0.0, 1.0)) * horiz_k
        fpos = np.maximum(np.nan_to_num(f, nan=0.0), 0.0)
        step = np.zeros_like(ti)
        ci = np.nan_to_num(col, nan=-1.0)
        ri = np.nan_to_num(row, nan=-1.0)
        for level in pyr:
            c = level["cell"]
            rho = level["rho"]
            jj = np.clip((ci // c).astype(np.intp), 0, level["hmax"].shape[1] - 1)
            ii = np.clip((ri // c).astype(np.intp), 0, level["hmax"].shape[0] - 1)
            H = level["hmax"][ii, jj]
            G = level["gmax"][ii, jj] * horiz_k
            s_horiz = np.where(sin_r > 1e-12, rho / np.maximum(sin_r, 1e-12), np.inf)
            clear = hp - H - tol
            s_height = np.where(clear > 0, np.where(w < 0, clear / np.maximum(-w, 1e-12), np.inf), 0.0)
            s_height = np.minimum(s_height, s_horiz)
            s_lip = np.minimum(safety * fpos / (np.abs(w) + sin_r * G + 1e-12), s_horiz)
            step = np.maximum(step, np.maximum(s_height, s_lip))
        # Never step beyond the shell exit by more than necessary.
        step = np.minimum(step, np.maximum(t_end[idx] - ti, 0.0) + tol)
        step = np.maximum(step, min_step)
        t[idx] = ti + step
        done = lost | hit
        active[idx[done]] = False
    return result


def _sphere_roots(o: np.ndarray, d: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Both roots t1 <= t2 of |o + t d| = radius for unit d; NaN where the line misses."""
    b = np.einsum("ij,ij->i", o, d)
    c = np.einsum("ij,ij->i", o, o) - radius * radius
    disc = b * b - c
    sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
    return -b - sq, -b + sq


class DemSource:
    """Large on-disk polar stereographic DEM read by windows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            self._init_geotiff()
        elif suffix in (".img", ".lbl"):
            self._init_pds_img()
        else:
            raise ValueError(f"unsupported DEM format: {self.path}")

    # -- GeoTIFF (PGDA products) ---------------------------------------------------------
    def _init_geotiff(self) -> None:
        import rasterio

        self._ds = rasterio.open(self.path)
        tr = self._ds.transform
        if abs(tr.b) > 0 or abs(tr.d) > 0 or abs(tr.a + tr.e) > 1e-9:
            raise ValueError("expected a north-up square-pixel GeoTIFF")
        self.pixel = float(tr.a)
        self.x0 = float(tr.c + 0.5 * tr.a)
        self.y0 = float(tr.f + 0.5 * tr.e)
        self.rows, self.cols = self._ds.height, self._ds.width
        self.scale = 1.0
        self.kind = "geotiff"
        self.name = self.path.stem

    def _read_geotiff(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        from rasterio.windows import Window

        arr = self._ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
        nodata = self._ds.nodata
        if nodata is not None and not np.isnan(nodata):
            arr[arr == nodata] = np.nan
        return arr

    # -- PDS3 IMG (LOLA GDR polar products) ----------------------------------------------
    def _init_pds_img(self) -> None:
        lbl = self.path.with_suffix(".lbl")
        if not lbl.exists():
            lbl = self.path.with_suffix(".LBL")
        text = lbl.read_text()

        def val(key: str) -> str:
            m = re.search(rf"^\s*{key}\s*=\s*\"?([^\"\n<]+)", text, re.MULTILINE)
            if not m:
                raise KeyError(key)
            return m.group(1).strip()

        self.rows = int(val("LINES"))
        self.cols = int(val("LINE_SAMPLES"))
        bits = int(val("SAMPLE_BITS"))
        stype = val("SAMPLE_TYPE")
        if bits != 16 or "LSB_INTEGER" not in stype:
            raise ValueError(f"unsupported PDS sample type {stype}/{bits}")
        self.scale = float(val("SCALING_FACTOR"))
        # LOLA polar labels give MAP_SCALE in <m/pix>.
        self.pixel = float(val("MAP_SCALE").split()[0])
        line_off = float(val("LINE_PROJECTION_OFFSET").split()[0])
        samp_off = float(val("SAMPLE_PROJECTION_OFFSET").split()[0])
        # PDS polar stereographic: x = (sample - S0 - 1) * scale, y = (1 + L0 - line) * scale,
        # with 1-based sample/line indices at pixel centres.
        self.x0 = (1 - samp_off - 1) * self.pixel
        self.y0 = (1 + line_off - 1) * self.pixel
        img = self.path.with_suffix(".img")
        if not img.exists():
            img = self.path.with_suffix(".IMG")
        self._mm = np.memmap(img, dtype="<i2", mode="r", shape=(self.rows, self.cols))
        self.kind = "pds_img"
        self.name = img.stem

    def _read_pds(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        return np.asarray(self._mm[r0:r1, c0:c1], dtype=np.float32) * np.float32(self.scale)

    # -- common ---------------------------------------------------------------------------
    def read_window(self, xmin: float, xmax: float, ymin: float, ymax: float, margin_px: int = 2) -> Dem:
        """Read the window covering [xmin, xmax] x [ymin, ymax] (stereographic metres)."""
        c0 = int(np.floor((xmin - self.x0) / self.pixel)) - margin_px
        c1 = int(np.ceil((xmax - self.x0) / self.pixel)) + margin_px + 1
        r0 = int(np.floor((self.y0 - ymax) / self.pixel)) - margin_px
        r1 = int(np.ceil((self.y0 - ymin) / self.pixel)) + margin_px + 1
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.cols), min(r1, self.rows)
        if c1 <= c0 or r1 <= r0:
            raise ValueError("window outside the DEM")
        if self.kind == "geotiff":
            h = self._read_geotiff(r0, r1, c0, c1)
        else:
            h = self._read_pds(r0, r1, c0, c1)
        return Dem(heights=h, x0=self.x0 + c0 * self.pixel, y0=self.y0 - r0 * self.pixel,
                   pixel=self.pixel, name=self.name)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x0 + (self.cols - 1) * self.pixel,
                self.y0 - (self.rows - 1) * self.pixel, self.y0)
