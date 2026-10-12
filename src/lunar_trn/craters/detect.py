"""CNav-style crater detector: stable dark/bright regions paired along the Sun direction.

Follows DLR CNav (Maass 2016; Trigo et al. 2018):

* "stable connected regions of above- and below-average intensity" from a threshold sweep,
  where stability means a small centroid movement between consecutive thresholds;
* each dark region is linked to its two nearest bright regions, excluding pairs where "the
  larger area is more than ten times the size of the smaller area";
* a pair must point away from the Sun (shadowed Sun-side wall -> lit anti-Sun wall);
* the crater ellipse is the principal-components ellipse "over the union of their
  constituent image support points".

Output image coordinates are continuous coordinates u = col + pixel_center. The thresholds
the papers leave open are in :class:`DetectorConfig` (decisions G5-G7).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..pose import project_uv


@dataclass
class DetectorConfig:
    """Detector settings (decisions G5-G7); kappa, beta, pair_angle_max_deg and D_min_px are calibrated."""

    pre_smooth_px: float = 1.0  # Gaussian pre-filter against pixel noise
    sigma_local_px: float = 25.0  # high-pass: local mean and std window (Gaussian sigma)
    std_floor: float = 0.02  # local std floor (image units) so flat or black areas do not explode
    thresholds: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)  # sweep, local-sigma units
    min_area_px: int = 3
    max_area_frac: float = 0.05
    stab_move_frac: float = 0.25  # stable: centroid move <= max(1 px, frac * equivalent radius)
    stab_area_ratio: float = 0.5  # and the next-threshold region keeps >= this area fraction
    pyramid_levels: int = 3
    level_bands_px: tuple = (32.0, 64.0)  # D_px limits between pyramid levels
    pair_neighbours: int = 2  # CNav: the two nearest bright regions
    pair_area_ratio_max: float = 10.0  # CNav
    pair_dist_k: float = 1.5  # centroid distance <= k * (sum of equivalent radii)
    pair_angle_max_deg: float = 80.0  # dark->bright vs anti-Sun direction (calibrated, decisions G6)
    sun_source: str = "ephemeris"  # "ephemeris" (Sun + prior attitude) or "image" (CNav estimate)
    kappa: float = 1.0392  # semi-axis scale of the moment ellipse (calibrated, decisions G7)
    beta: float = 0.0196  # centre shift towards anti-Sun, x semi-major axis (calibrated, decisions G7)
    D_min_px: float = 12.0  # calibrated, decisions G7
    D_max_px: float = 128.0
    dedupe_centre: float = 0.3
    dedupe_size: float = 1.5


@dataclass
class Regions:
    """Stable regions of one polarity at one pyramid level (raw moments in level pixels)."""

    m0: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    mxx: np.ndarray
    mxy: np.ndarray
    myy: np.ndarray
    contrast: np.ndarray  # mean |high-pass| over the region
    threshold: np.ndarray  # sweep threshold at which the region was taken

    def __len__(self) -> int:
        return int(self.m0.size)

    @property
    def centroid(self) -> np.ndarray:
        return np.column_stack([self.mx / self.m0, self.my / self.m0])


_DET_FIELDS = ("u", "v", "A", "B", "theta", "area_dark", "area_bright", "pair_dev_deg", "contrast", "level")


@dataclass
class Detections:
    u: np.ndarray  # ellipse centre, continuous image coordinates
    v: np.ndarray
    A: np.ndarray  # semi-major axis (px)
    B: np.ndarray  # semi-minor axis (px)
    theta: np.ndarray  # major-axis angle from +u towards +v (rad)
    area_dark: np.ndarray  # shadow region area (px)
    area_bright: np.ndarray  # lit region area (px)
    pair_dev_deg: np.ndarray  # dark->bright direction minus the anti-Sun direction (deg)
    contrast: np.ndarray  # mean |high-pass| of the pair (local-sigma units)
    level: np.ndarray  # pyramid level

    def __len__(self) -> int:
        return int(self.u.shape[0])

    @property
    def D_px(self) -> np.ndarray:
        return 2.0 * np.sqrt(self.A * self.B)

    @property
    def uv(self) -> np.ndarray:
        return np.column_stack([self.u, self.v])

    def subset(self, sel) -> "Detections":
        sel = np.asarray(sel)
        return Detections(**{k: getattr(self, k)[sel] for k in _DET_FIELDS})

    @classmethod
    def empty(cls) -> "Detections":
        return cls(**{k: np.zeros(0) for k in _DET_FIELDS})

    @classmethod
    def concat(cls, parts) -> "Detections":
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in _DET_FIELDS})

    @classmethod
    def from_ellipses(cls, u, v, A, B, theta) -> "Detections":
        """Detections given only as ellipses (injected detections, oracle tests)."""
        u = np.asarray(u, float).ravel()
        nan = np.full(u.size, np.nan)
        return cls(u=u, v=np.asarray(v, float).ravel(), A=np.asarray(A, float).ravel(),
                   B=np.asarray(B, float).ravel(), theta=np.asarray(theta, float).ravel(), area_dark=nan.copy(),
                   area_bright=nan.copy(), pair_dev_deg=nan.copy(), contrast=nan.copy(), level=np.zeros(u.size))


def highpass(image: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Image minus its local mean, divided by the floored local standard deviation."""
    I = np.nan_to_num(np.asarray(image, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.pre_smooth_px > 0:
        I = ndi.gaussian_filter(I, cfg.pre_smooth_px)
    mu = ndi.gaussian_filter(I, cfg.sigma_local_px)
    var = ndi.gaussian_filter(I * I, cfg.sigma_local_px) - mu * mu
    return (I - mu) / np.maximum(np.sqrt(np.maximum(var, 0.0)), cfg.std_floor)


def stable_regions(hp: np.ndarray, polarity: int, cfg: DetectorConfig) -> Regions:
    """CNav threshold sweep: the loosest stable region of every nested chain of components."""
    s = hp if polarity > 0 else -hp
    H, W = s.shape
    max_area = cfg.max_area_frac * H * W
    lev = []
    for t in cfg.thresholds:
        n, lab = cv2.connectedComponents((s > t).astype(np.uint8), connectivity=8)
        ys, xs = np.nonzero(lab)
        lbl = lab[ys, xs]
        xf, yf = xs.astype(np.float64), ys.astype(np.float64)

        def bc(w=None, lbl=lbl, n=n):
            return np.bincount(lbl, weights=w, minlength=n).astype(np.float64)

        lev.append({"n": n, "lab": lab, "l": lbl, "ys": ys, "xs": xs, "m0": bc(), "mx": bc(xf), "my": bc(yf),
                    "mxx": bc(xf * xf), "mxy": bc(xf * yf), "myy": bc(yf * yf), "sv": bc(np.abs(s[ys, xs]))})
    nl = len(lev)
    stable = [np.zeros(L["n"], bool) for L in lev]
    parent: list = [None] * nl
    for k in range(nl - 1):
        P, C = lev[k], lev[k + 1]
        par = np.zeros(C["n"], np.int64)
        par[C["l"]] = P["lab"][C["ys"], C["xs"]]  # nested thresholds: one parent per child
        parent[k + 1] = par
        if C["n"] <= 1:
            continue
        kids = np.arange(1, C["n"])
        kids = kids[np.argsort(C["m0"][kids], kind="stable")]
        best = np.zeros(P["n"], np.int64)
        best[par[kids]] = kids  # the largest child is written last
        pk = np.nonzero(best > 0)[0]
        pk = pk[pk > 0]
        ck = best[pk]
        pa, ca = P["m0"][pk], C["m0"][ck]
        move = np.hypot(C["mx"][ck] / ca - P["mx"][pk] / pa, C["my"][ck] / ca - P["my"][pk] / pa)
        ok = ((move <= np.maximum(1.0, cfg.stab_move_frac * np.sqrt(pa / np.pi))) & (ca / pa >= cfg.stab_area_ratio)
              & (pa >= cfg.min_area_px) & (pa <= max_area))
        stable[k][pk[ok]] = True
    keys = ("m0", "mx", "my", "mxx", "mxy", "myy", "sv")
    acc: dict = {key: [] for key in keys}
    thr = []
    for k in range(nl):
        st = np.nonzero(stable[k])[0]
        if k > 0 and st.size:
            st = st[~stable[k - 1][parent[k][st]]]  # a stable parent supersedes its child
        for key in keys:
            acc[key].append(lev[k][key][st])
        thr.append(np.full(st.size, float(cfg.thresholds[k])))
    cat = {key: (np.concatenate(v) if v else np.zeros(0)) for key, v in acc.items()}
    return Regions(m0=cat["m0"], mx=cat["mx"], my=cat["my"], mxx=cat["mxx"], mxy=cat["mxy"], myy=cat["myy"],
                   contrast=cat["sv"] / np.maximum(cat["m0"], 1.0),
                   threshold=np.concatenate(thr) if thr else np.zeros(0))


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi


def pair_regions(dark: Regions, bright: Regions, cfg: DetectorConfig):
    """CNav candidate edges: each dark region to its nearest bright regions (area-ratio and distance gated)."""
    if len(dark) == 0 or len(bright) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    pd, pb = dark.centroid, bright.centroid
    k = min(cfg.pair_neighbours, len(bright))
    dist, nb = cKDTree(pb).query(pd, k=k)
    dist = np.asarray(dist, float).reshape(len(pd), k).ravel()
    j = np.asarray(nb).reshape(len(pd), k).ravel()
    i = np.repeat(np.arange(len(pd)), k)
    ad, ab = dark.m0[i], bright.m0[j]
    ok = np.maximum(ad, ab) / np.minimum(ad, ab) <= cfg.pair_area_ratio_max
    ok &= dist <= cfg.pair_dist_k * (np.sqrt(ad / np.pi) + np.sqrt(ab / np.pi))
    vec = pb[j] - pd[i]
    return i[ok], j[ok], np.arctan2(vec[ok, 1], vec[ok, 0])


def estimate_illumination(angles) -> tuple[float, float]:
    """CNav: direction minimising the angular variance inside a pi-wide window of edge directions.

    ``angles`` are dark->bright edge directions; the result ``mu`` points away from the Sun.
    Returns (nan, nan) for fewer than three edges.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan"), float("nan")
    grid = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    dev = _wrap(a[None, :] - grid[:, None])
    inw = np.abs(dev) < np.pi / 2
    cnt = inw.sum(axis=1)
    mean = np.where(cnt > 0, (dev * inw).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0, (((dev - mean[:, None]) ** 2) * inw).sum(axis=1) / np.maximum(cnt, 1), np.inf)
    var[cnt < max(3, 0.5 * a.size)] = np.inf
    g = int(np.argmin(var))
    if not np.isfinite(var[g]):
        return float("nan"), float("nan")
    return float(_wrap(grid[g] + mean[g])), float(np.sqrt(var[g]))


def pinhole_sun_fn(cam, T_world_cam: np.ndarray, sun_dir_world, z0: float):
    """Image direction towards the Sun at image points, for a camera with known attitude.

    A ground point at camera depth ``z0`` behind each pixel is moved slightly towards the
    Sun and both points are projected; this keeps the per-pixel perspective of the Sun
    direction (shadows converge to the Sun's vanishing point). Undefined -> NaN.
    """
    K = np.asarray(cam.K, float)
    T = np.asarray(T_world_cam, float)
    R, c = T[:3, :3], T[:3, 3]
    s = np.asarray(sun_dir_world, float)
    s = s / np.linalg.norm(s)
    Kinv = np.linalg.inv(K)

    def fn(u, v):
        u = np.atleast_1d(np.asarray(u, float))
        v = np.atleast_1d(np.asarray(v, float))
        d = (np.column_stack([u, v, np.ones_like(u)]) @ Kinv.T) @ R.T  # world rays with unit camera depth
        X1 = c + z0 * d
        p1, _ = project_uv(K, R, c, X1)
        p2, _ = project_uv(K, R, c, X1 + 0.01 * z0 * s)
        g = p2 - p1
        nrm = np.linalg.norm(g, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(nrm > 1e-6, g / nrm, np.nan)

    return fn


def constant_sun_fn(angle_deg: float):
    """Same image Sun direction everywhere (orthographic maps, procedural tests)."""
    vec = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
    return lambda u, v: np.tile(vec, (np.size(u), 1))


def _one_to_one(i, j, dev):
    keep, used_i, used_j = [], set(), set()
    for k in np.argsort(dev, kind="stable"):
        if i[k] in used_i or j[k] in used_j:
            continue
        used_i.add(i[k])
        used_j.add(j[k])
        keep.append(k)
    keep = np.asarray(keep, int)
    return i[keep], j[keep], dev[keep]


def _dedupe(det: Detections, cfg: DetectorConfig) -> Detections:
    if len(det) < 2:
        return det
    D = det.D_px
    tree = cKDTree(det.uv)
    removed = np.zeros(len(det), bool)
    for i in np.argsort(det.pair_dev_deg, kind="stable"):
        if removed[i]:
            continue
        for j in tree.query_ball_point(det.uv[i], cfg.dedupe_centre * D[i]):
            if j == i or removed[j]:
                continue
            dmin = min(D[i], D[j])
            if (np.hypot(*(det.uv[i] - det.uv[j])) < cfg.dedupe_centre * dmin
                    and max(D[i], D[j]) / dmin < cfg.dedupe_size):
                removed[j] = True
    return det.subset(~removed)


def detect_craters(image: np.ndarray, sun_fn, cfg: DetectorConfig | None = None,
                   pixel_center: float = 0.5) -> tuple[Detections, dict]:
    """Crater ellipses in ``image``; ``sun_fn(u, v)`` gives unit image vectors towards the Sun."""
    cfg = cfg or DetectorConfig()
    img = np.nan_to_num(np.asarray(image, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    H, W = img.shape
    n_lev = 0
    while n_lev < cfg.pyramid_levels and min(H, W) // (2**n_lev) >= 32:
        n_lev += 1
    edges = [cfg.D_min_px, *list(cfg.level_bands_px)[: max(n_lev - 1, 0)], cfg.D_max_px]
    health = {"n_dark_regions": 0, "n_bright_regions": 0, "n_candidate_edges": 0, "n_pairs": 0}
    mu, sig = float("nan"), float("nan")
    parts = []
    for lvl in range(n_lev):
        f = 2**lvl
        im = img if f == 1 else cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
        hp = highpass(im, cfg)
        dark, bright = stable_regions(hp, -1, cfg), stable_regions(hp, +1, cfg)
        health["n_dark_regions"] += len(dark)
        health["n_bright_regions"] += len(bright)
        i, j, ang = pair_regions(dark, bright, cfg)
        health["n_candidate_edges"] += int(i.size)
        if lvl == 0:
            mu, sig = estimate_illumination(ang)
        if i.size == 0:
            continue
        off = (f - 1) / 2.0 + pixel_center  # level pixel index -> full-resolution continuous coordinate
        if cfg.sun_source == "image":
            dev, lim = np.abs(_wrap(ang - mu)), sig
        else:
            cd = dark.centroid[i]
            s = sun_fn(f * cd[:, 0] + off, f * cd[:, 1] + off)
            dev, lim = np.abs(_wrap(ang - np.arctan2(-s[:, 1], -s[:, 0]))), np.deg2rad(cfg.pair_angle_max_deg)
        ok = np.isfinite(dev) & (dev <= lim)
        i, j, dev = _one_to_one(i[ok], j[ok], dev[ok])
        health["n_pairs"] += int(i.size)
        if i.size == 0:
            continue
        m0 = dark.m0[i] + bright.m0[j]
        cx = (dark.mx[i] + bright.mx[j]) / m0
        cy = (dark.my[i] + bright.my[j]) / m0
        sc2 = float(f * f)
        cxx = ((dark.mxx[i] + bright.mxx[j]) / m0 - cx * cx) * sc2 + (sc2 - 1.0) / 12.0
        cyy = ((dark.myy[i] + bright.myy[j]) / m0 - cy * cy) * sc2 + (sc2 - 1.0) / 12.0
        cxy = ((dark.mxy[i] + bright.mxy[j]) / m0 - cx * cy) * sc2
        half = 0.5 * (cxx + cyy)
        dlt = np.sqrt(np.maximum(0.25 * (cxx - cyy) ** 2 + cxy**2, 0.0))
        A = cfg.kappa * 2.0 * np.sqrt(half + dlt)
        B = cfg.kappa * 2.0 * np.sqrt(np.maximum(half - dlt, 1e-12))
        theta = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
        u, v = f * cx + off, f * cy + off
        if cfg.sun_source == "image":
            s2 = np.tile([-np.cos(mu), -np.sin(mu)], (u.size, 1))
        else:
            s2 = sun_fn(u, v)
        u = u - cfg.beta * A * s2[:, 0]  # shift towards anti-Sun: the shadow side biases the centroid
        v = v - cfg.beta * A * s2[:, 1]
        D = 2.0 * np.sqrt(A * B)
        upper = (D <= edges[lvl + 1]) if lvl == n_lev - 1 else (D < edges[lvl + 1])
        keep = np.isfinite(u) & np.isfinite(v) & (D >= edges[lvl]) & upper
        parts.append(Detections(u=u[keep], v=v[keep], A=A[keep], B=B[keep], theta=theta[keep],
                                area_dark=(dark.m0[i] * sc2)[keep], area_bright=(bright.m0[j] * sc2)[keep],
                                pair_dev_deg=np.rad2deg(dev)[keep],
                                contrast=(0.5 * (dark.contrast[i] + bright.contrast[j]))[keep],
                                level=np.full(int(keep.sum()), float(lvl))))
    det = _dedupe(Detections.concat(parts), cfg)
    centre = (W - 1) / 2.0 + pixel_center, (H - 1) / 2.0 + pixel_center
    sc = sun_fn(np.array([centre[0]]), np.array([centre[1]]))[0]
    health.update(n_detections=len(det), sun_img_angle_deg=float(np.degrees(np.arctan2(sc[1], sc[0]))),
                  sun_img_angle_est_deg=float(np.degrees(_wrap(mu + np.pi))) if np.isfinite(mu) else float("nan"),
                  sun_est_sigma_deg=float(np.degrees(sig)) if np.isfinite(sig) else float("nan"))
    q = np.percentile(det.D_px, [10, 50, 90]) if len(det) else [float("nan")] * 3
    health.update(det_D_px_p10=float(q[0]), det_D_px_p50=float(q[1]), det_D_px_p90=float(q[2]))
    return det, health
