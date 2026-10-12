"""Crater catalog extracted from a DEM: data structure, extraction and queries.

Each crater is a rim ellipse on its rim plane (see :mod:`lunar_trn.craters.conics`) plus raw
morphology signals. The catalog does not depend on the Sun, so one catalog per DEM serves
every illumination; the matcher selects craters by apparent size at run time.

Extraction (decisions G1-G4): DEM pyramid -> scale-normalised LoG seeds -> detrended radial
profiles and crest points -> robust direct ellipse fit on the rim plane -> morphology (depth,
d/D, completeness) -> filters and de-duplication across pyramid levels.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .. import frames
from ..dem import Dem, bilinear
from ..render import stereo_tangents
from .conics import ellipse_residual, fit_ellipse_direct

_VEC = ("X", "normal", "e1", "e2")
_SCALAR = ("a", "b", "psi", "x", "y", "h", "depth", "d_over_D", "completeness", "rim_h_std",
           "fit_rms", "seed_response")
_INT = ("level", "id")
_ALL = _VEC + _SCALAR + _INT


@dataclass
class CraterCatalog:
    X: np.ndarray  # (N, 3) rim-plane centre, body frame MOON_ME (m)
    normal: np.ndarray  # (N, 3) rim-plane unit normal, pointing up
    e1: np.ndarray  # (N, 3) in-plane axis: map-easting tangent made orthogonal to the normal
    e2: np.ndarray  # (N, 3) normal x e1
    a: np.ndarray  # (N,) semi-major axis (m)
    b: np.ndarray  # (N,) semi-minor axis (m)
    psi: np.ndarray  # (N,) major-axis angle from e1 towards e2 (rad)
    x: np.ndarray  # (N,) stereographic easting of the centre (m)
    y: np.ndarray  # (N,) stereographic northing of the centre (m)
    h: np.ndarray  # (N,) centre height above the reference sphere (m)
    depth: np.ndarray  # (N,) mean crest height minus inner floor, regional plane removed (m)
    d_over_D: np.ndarray  # (N,) depth / D with D = 2 sqrt(a b)
    completeness: np.ndarray  # (N,) fraction of azimuths with an accepted crest point
    rim_h_std: np.ndarray  # (N,) crest scatter about the rim plane (m)
    fit_rms: np.ndarray  # (N,) ellipse fit residual RMS (m)
    seed_response: np.ndarray  # (N,) scale-normalised LoG response of the seed (m)
    level: np.ndarray  # (N,) DEM decimation factor of the extraction level
    id: np.ndarray  # (N,) int64 identifier
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.a.shape[0])

    @property
    def D(self) -> np.ndarray:
        """Equivalent rim diameter 2 sqrt(a b) (m)."""
        return 2.0 * np.sqrt(self.a * self.b)

    @classmethod
    def empty(cls) -> "CraterCatalog":
        kw = {k: np.zeros((0, 3)) for k in _VEC} | {k: np.zeros(0) for k in _SCALAR}
        kw |= {k: np.zeros(0, np.int64) for k in _INT}
        return cls(**kw)

    def subset(self, sel) -> "CraterCatalog":
        sel = np.asarray(sel)
        return CraterCatalog(**{k: getattr(self, k)[sel] for k in _ALL}, meta=dict(self.meta))

    def query_box(self, xmin: float, xmax: float, ymin: float, ymax: float, return_index: bool = False):
        """Craters whose centre lies in the stereographic box (inclusive)."""
        idx = np.nonzero((self.x >= xmin) & (self.x <= xmax) & (self.y >= ymin) & (self.y <= ymax))[0]
        sub = self.subset(idx)
        return (sub, idx) if return_index else sub

    @classmethod
    def concat(cls, cats, meta: dict | None = None) -> "CraterCatalog":
        cats = [c for c in cats if c is not None and len(c)]
        out = cls(**{k: np.concatenate([getattr(c, k) for c in cats]) for k in _ALL}) if cats else cls.empty()
        out.meta = dict(meta or {})
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in _ALL}, meta=np.array([repr(self.meta)]))

    @classmethod
    def load(cls, path: str | Path) -> "CraterCatalog":
        with np.load(path, allow_pickle=False) as d:
            kw = {k: d[k] for k in _ALL}
            meta = ast.literal_eval(str(d["meta"][0]))
        return cls(**kw, meta=meta)


def rim_frame(x, y, normal):
    """Polar-safe in-plane axes: the map-easting tangent made orthogonal to the rim normal."""
    normal = np.asarray(normal, float).reshape(-1, 3)
    _, ux, _ = stereo_tangents(np.atleast_1d(x), np.atleast_1d(y))
    t = ux.reshape(-1, 3)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    e1 = t - np.sum(t * normal, axis=1, keepdims=True) * normal
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(normal, e1)


def make_catalog(X, normal, e1, e2, a, b, psi, **extra) -> CraterCatalog:
    """Catalog from rim geometry; morphology fields not given in ``extra`` are NaN."""
    X = np.asarray(X, float).reshape(-1, 3)
    n = X.shape[0]
    x, y, h = frames.body_to_stereo(X)
    kw = {"X": X, "normal": np.asarray(normal, float).reshape(-1, 3), "e1": np.asarray(e1, float).reshape(-1, 3),
          "e2": np.asarray(e2, float).reshape(-1, 3), "a": np.asarray(a, float).ravel(),
          "b": np.asarray(b, float).ravel(), "psi": np.asarray(psi, float).ravel(), "x": x, "y": y, "h": h}
    for k in ("depth", "d_over_D", "completeness", "rim_h_std", "fit_rms", "seed_response"):
        kw[k] = np.asarray(extra.get(k, np.full(n, np.nan)), float).ravel()
    kw["level"] = np.asarray(extra.get("level", np.ones(n)), np.int64).ravel()
    kw["id"] = np.asarray(extra.get("id", np.arange(n)), np.int64).ravel()
    return CraterCatalog(**kw, meta=dict(extra.get("meta", {})))


@dataclass
class CatalogConfig:
    """DEM crater extraction settings (decisions G1-G4)."""

    levels: tuple = (1, 2, 4, 8, 16, 32, 64)  # DEM block-mean factors (5 m DEM -> 5 ... 320 m)
    radii_px: tuple = (5.0, 6.3, 7.9)  # seed radii per level, level pixels
    seed_sigma_frac: float = 0.45  # LoG sigma / seed radius: peak scale of rimmed bowls (not r / sqrt 2)
    seed_dD_min: float = 0.02  # LoG depth proxy 2 resp / D (resp ~ d/2 for a Gaussian bowl)
    n_azimuth: int = 48
    samples_per_radius: int = 12
    rho_max: float = 1.8  # profile extent, x seed radius
    rim_search: tuple = (0.5, 1.6)  # crest search interval, x seed radius
    inner_floor: float = 0.4  # floor = minimum inside this radius, x seed radius
    min_relief_m: float = 0.2  # crest must rise this much above the inner floor (rejects flat ground)
    detrend_annulus: tuple = (1.5, 2.5)  # regional plane fit annulus, x seed radius
    mad_k: float = 2.5  # crest outlier rejection, x 1.4826 MAD
    min_rim_points: int = 8
    size_consistency: tuple = (0.5, 1.6)  # fitted sqrt(a b) / seed radius
    centre_consistency: float = 0.5  # fitted centre offset / seed radius
    chunk: int = 20000  # seeds per vectorised profile batch
    # Filters (decisions G3)
    D_min_m: float = 50.0
    D_max_m: float = 6000.0  # 128 px at the 49 m/px top band
    completeness_min: float = 0.375  # relaxed from 0.7 by the calibration-A rule (decisions G3)
    ellipticity_max: float = 1.5
    dD_min: float = 0.04
    # De-duplication across levels (decisions G4)
    dedupe_centre: float = 0.3
    dedupe_size: float = 1.5


def level_margin_m(cfg: CatalogConfig, f: int, pixel: float) -> float:
    """Window margin that holds every profile of a seed centred in the core at level ``f``."""
    return (cfg.detrend_annulus[1] * max(cfg.radii_px) + 4.0) * f * pixel


def _crop(dem: Dem, xmin: float, xmax: float, ymin: float, ymax: float, align: int) -> Dem:
    """Sub-window of ``dem`` whose origin stays on the ``align``-pixel block grid of ``dem``."""
    rows, cols = dem.shape
    c0 = max(int(np.floor((xmin - dem.x0) / dem.pixel)) // align * align, 0)
    r0 = max(int(np.floor((dem.y0 - ymax) / dem.pixel)) // align * align, 0)
    c1 = min(int(np.ceil((xmax - dem.x0) / dem.pixel)) + 1, cols)
    r1 = min(int(np.ceil((dem.y0 - ymin) / dem.pixel)) + 1, rows)
    return Dem(dem.heights[r0:r1, c0:c1], dem.x0 + c0 * dem.pixel, dem.y0 - r0 * dem.pixel, dem.pixel,
               name=dem.name, radius=dem.radius)


def _fill_nan(h: np.ndarray):
    nan = ~np.isfinite(h)
    if not nan.any():
        return h.astype(np.float32), nan
    if nan.all():
        return np.zeros(h.shape, np.float32), nan
    idx = ndi.distance_transform_edt(nan, return_distances=False, return_indices=True)
    return h[tuple(idx)].astype(np.float32), nan


def _seeds(H: np.ndarray, pixel: float, cfg: CatalogConfig):
    radii = np.asarray(cfg.radii_px, float)
    sig = cfg.seed_sigma_frac * radii
    L = np.stack([np.float32(s * s) * ndi.gaussian_laplace(H, s) for s in sig])
    proxy = L / (radii[:, None, None] * pixel).astype(np.float32)  # 2 resp / D with D = 2 r pixel
    peak = (L == ndi.maximum_filter(L, size=(3, 5, 5), mode="nearest")) & (proxy >= cfg.seed_dD_min)
    k, i, j = np.nonzero(peak)
    r = radii[k]
    reach = cfg.detrend_annulus[1] * r + 1.0
    ok = (i >= reach) & (j >= reach) & (i < H.shape[0] - reach) & (j < H.shape[1] - reach)
    return i[ok].astype(float), j[ok].astype(float), r[ok], L[k[ok], i[ok], j[ok]].astype(float)


def _rim_candidates(H, i0, j0, r0, cfg: CatalogConfig):
    """Detrended radial profiles -> crest radius, crest height, validity per azimuth; inner floor."""
    A = cfg.n_azimuth
    th = 2.0 * np.pi * np.arange(A) / A
    rho = np.linspace(0.0, cfg.rho_max, int(round(cfg.rho_max * cfg.samples_per_radius)) + 1)
    ca, sa = np.cos(th), np.sin(th)
    dc = rho[None, None, :] * r0[:, None, None] * ca[None, :, None]  # column offsets (n, A, R)
    dr = -rho[None, None, :] * r0[:, None, None] * sa[None, :, None]  # row offsets (rows grow south)
    prof = bilinear(H, j0[:, None, None] + dc, i0[:, None, None] + dr)
    ann = np.linspace(cfg.detrend_annulus[0], cfg.detrend_annulus[1], 5)
    ac = ann[None, None, :] * r0[:, None, None] * ca[None, :, None]
    ar = -ann[None, None, :] * r0[:, None, None] * sa[None, :, None]
    ah = bilinear(H, j0[:, None, None] + ac, i0[:, None, None] + ar)
    n = i0.size
    G = np.stack([np.ones_like(ac), ac, ar], axis=-1).reshape(n, -1, 3)
    coef = np.linalg.solve(np.einsum("nki,nkj->nij", G, G),
                           np.einsum("nki,nk->ni", G, ah.reshape(n, -1))[..., None])[..., 0]
    prof = prof - (coef[:, 0, None, None] + coef[:, 1, None, None] * dc + coef[:, 2, None, None] * dr)
    lo = int(np.searchsorted(rho, cfg.rim_search[0]))
    hi = int(np.searchsorted(rho, cfg.rim_search[1], side="right"))
    seg = prof[:, :, lo:hi]
    k = np.argmax(np.nan_to_num(seg, nan=-np.inf), axis=2)
    crest = np.take_along_axis(seg, k[..., None], axis=2)[..., 0]
    inner = np.nanmin(prof[:, :, rho <= cfg.inner_floor], axis=(1, 2))
    valid = ((k > 0) & (k < seg.shape[2] - 1) & (crest - inner[:, None] > cfg.min_relief_m)
             & (crest > prof[:, :, -1]))
    return th, rho[lo + k], crest, valid, inner


def _fit_rim(dl: Dem, H, i0, j0, r0, th, crest_rho, crest, valid, inner, cfg: CatalogConfig):
    """Rim record of one seed and the reason code: "ok" or the first failed step."""
    if valid.sum() < cfg.min_rim_points:
        return None, "no_crest"
    rr = crest_rho[valid] * r0
    col = j0 + rr * np.cos(th[valid])
    row = i0 - rr * np.sin(th[valid])
    x, y = dl.colrow_to_xy(col, row)
    P = frames.stereo_to_body(x, y, bilinear(H, col, row))
    Pm = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - Pm)
    nrm = Vt[2] if Vt[2] @ Pm > 0 else -Vt[2]
    xs, ys = dl.colrow_to_xy(j0, i0)
    e1, e2 = rim_frame(np.array([xs]), np.array([ys]), nrm[None, :])
    e1, e2 = e1[0], e2[0]
    s, t = (P - Pm) @ e1, (P - Pm) @ e2
    keep = np.ones(s.size, bool)
    fit = None
    for _ in range(3):
        if keep.sum() < cfg.min_rim_points:
            return None, "few_points"
        u, v, A_, B_, psi, ok = fit_ellipse_direct(s[keep], t[keep])
        if not ok:
            return None, "fit"
        res = ellipse_residual(s, t, u, v, A_, B_, psi)
        fit = (u, v, A_, B_, psi, res, keep.copy())
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) <= cfg.mad_k * max(mad, 0.25 * dl.pixel)
        if np.array_equal(new, keep):
            break
        keep = new
    u, v, A_, B_, psi, res, keep = fit
    r_m = r0 * dl.pixel
    if not (cfg.size_consistency[0] <= np.sqrt(A_ * B_) / r_m <= cfg.size_consistency[1]):
        return None, "size"
    Xc = Pm + u * e1 + v * e2
    xc, yc, hc = frames.body_to_stereo(Xc)
    if np.hypot(xc - xs, yc - ys) > cfg.centre_consistency * r_m:
        return None, "centre"
    depth = float(np.mean(crest[valid][keep]) - inner)
    D = 2.0 * np.sqrt(A_ * B_)
    return {"X": Xc, "normal": nrm, "e1": e1, "e2": e2, "a": A_, "b": B_, "psi": psi,
            "x": float(xc), "y": float(yc), "h": float(hc), "depth": depth, "d_over_D": depth / D,
            "completeness": keep.sum() / th.size, "rim_h_std": float(np.std((P[keep] - Pm) @ nrm)),
            "fit_rms": float(np.sqrt(np.mean(res[keep] ** 2)))}, "ok"


def _dedupe(cat: CraterCatalog, cfg: CatalogConfig) -> CraterCatalog:
    if len(cat) < 2:
        return cat
    D = cat.D
    order = np.argsort(-(cat.completeness * np.abs(cat.seed_response) / D))
    tree = cKDTree(np.column_stack([cat.x, cat.y]))
    removed = np.zeros(len(cat), bool)
    for i in order:
        if removed[i]:
            continue
        for j in tree.query_ball_point([cat.x[i], cat.y[i]], cfg.dedupe_centre * D[i]):
            if j == i or removed[j]:
                continue
            dmin = min(D[i], D[j])
            if (np.hypot(cat.x[i] - cat.x[j], cat.y[i] - cat.y[j]) < cfg.dedupe_centre * dmin
                    and max(D[i], D[j]) / dmin < cfg.dedupe_size):
                removed[j] = True
    return cat.subset(~removed)


def extract_catalog(dem: Dem, cfg: CatalogConfig | None = None, core=None) -> CraterCatalog:
    """Crater catalog of a DEM window; ``core`` keeps centres in [xmin, xmax) x [ymin, ymax)."""
    cfg = cfg or CatalogConfig()
    recs: list[dict] = []
    for f in cfg.levels:
        base = dem
        if core is not None:  # only the part of the window that seeds in the core can reach
            m = level_margin_m(cfg, f, dem.pixel)
            base = _crop(dem, core[0] - m, core[1] + m, core[2] - m, core[3] + m, align=f)
        dl = base.downsample(f) if f > 1 else base
        if min(dl.shape) < 8 * max(cfg.radii_px):
            continue
        H, nan = _fill_nan(dl.heights)
        if nan.all():
            continue
        i0, j0, r0, resp = _seeds(H, dl.pixel, cfg)
        if nan.any() and i0.size:
            near = ndi.maximum_filter(nan, size=int(2 * cfg.detrend_annulus[1] * max(cfg.radii_px)) + 1)
            good = ~near[i0.astype(int), j0.astype(int)]
            i0, j0, r0, resp = i0[good], j0[good], r0[good], resp[good]
        for s0 in range(0, i0.size, cfg.chunk):
            sl = slice(s0, s0 + cfg.chunk)
            th, crest_rho, crest, valid, inner = _rim_candidates(H, i0[sl], j0[sl], r0[sl], cfg)
            for k in range(crest_rho.shape[0]):
                rec, _ = _fit_rim(dl, H, i0[sl][k], j0[sl][k], r0[sl][k], th, crest_rho[k], crest[k], valid[k],
                                  inner[k], cfg)
                if rec is not None:
                    rec["seed_response"] = float(resp[sl][k])
                    rec["level"] = int(f)
                    recs.append(rec)
    if recs:
        cat = make_catalog(np.array([r["X"] for r in recs]), np.array([r["normal"] for r in recs]),
                           np.array([r["e1"] for r in recs]), np.array([r["e2"] for r in recs]),
                           [r["a"] for r in recs], [r["b"] for r in recs], [r["psi"] for r in recs],
                           **{k: [r[k] for r in recs] for k in ("depth", "d_over_D", "completeness", "rim_h_std",
                                                                 "fit_rms", "seed_response", "level")})
    else:
        cat = CraterCatalog.empty()
    D = cat.D
    keep = (np.isfinite(D) & (D >= cfg.D_min_m) & (D <= cfg.D_max_m) & (cat.completeness >= cfg.completeness_min)
            & (cat.a / np.where(cat.b > 0, cat.b, np.nan) <= cfg.ellipticity_max) & (cat.d_over_D >= cfg.dD_min)
            & np.isfinite(cat.X).all(axis=1))
    cat = _dedupe(cat.subset(keep), cfg)
    if core is not None:
        xmin, xmax, ymin, ymax = core
        cat = cat.subset((cat.x >= xmin) & (cat.x < xmax) & (cat.y >= ymin) & (cat.y < ymax))
    cat.id = np.arange(len(cat), dtype=np.int64)
    cat.meta = {"dem": dem.name, "dem_pixel": float(dem.pixel), "config": asdict(cfg)}
    return cat


def tile_cores(extent, tile_m: float) -> list[tuple[float, float, float, float]]:
    """Half-open tile cores [x0, x1) x [y0, y1) that partition ``extent`` = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    xs = list(np.arange(xmin, xmax, tile_m)) or [xmin]
    ys = list(np.arange(ymin, ymax, tile_m)) or [ymin]
    out = []
    for x0 in xs:
        x1 = x0 + tile_m if x0 + tile_m < xmax else xmax + 1e-6
        for y0 in ys:
            y1 = y0 + tile_m if y0 + tile_m < ymax else ymax + 1e-6
            out.append((float(x0), float(x1), float(y0), float(y1)))
    return out


def explain_rim(dem: Dem, x: float, y: float, r_m: float, cfg: CatalogConfig | None = None):
    """Why a crater of radius ``r_m`` at stereographic (x, y) is (not) in the catalog.

    Runs the profile and fit steps with a seed placed exactly there, then the catalog filters.
    Returns the first failed step or "ok", and the rim record when the fit succeeded. A crater
    that is "ok" here but missing from the catalog was lost by LoG seeding or de-duplication.
    """
    cfg = cfg or CatalogConfig()
    rmin, rmax = min(cfg.radii_px), max(cfg.radii_px)
    usable = [f for f in cfg.levels if r_m / (f * dem.pixel) >= rmin]
    f = max(usable) if usable else min(cfg.levels)
    m = level_margin_m(cfg, f, dem.pixel)
    base = _crop(dem, x - m, x + m, y - m, y + m, align=f)
    dl = base.downsample(f) if f > 1 else base
    H, _ = _fill_nan(dl.heights)
    col, row = (float(v) for v in dl.xy_to_colrow(x, y))
    r0 = float(np.clip(r_m / dl.pixel, rmin, rmax))
    reach = cfg.detrend_annulus[1] * r0 + 1.0
    if not (reach <= col < H.shape[1] - reach and reach <= row < H.shape[0] - reach):
        return "edge", None
    th, crest_rho, crest, valid, inner = _rim_candidates(H, np.array([row]), np.array([col]), np.array([r0]), cfg)
    rec, reason = _fit_rim(dl, H, row, col, r0, th, crest_rho[0], crest[0], valid[0], inner[0], cfg)
    if rec is None:
        return reason, None
    D = 2.0 * np.sqrt(rec["a"] * rec["b"])
    for name, bad in (("D_min", D < cfg.D_min_m), ("D_max", D > cfg.D_max_m),
                      ("completeness", rec["completeness"] < cfg.completeness_min),
                      ("ellipticity", rec["a"] / rec["b"] > cfg.ellipticity_max),
                      ("dD", rec["d_over_D"] < cfg.dD_min)):
        if bad:
            return name, rec
    return "ok", rec
