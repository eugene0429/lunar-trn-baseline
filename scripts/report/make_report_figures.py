#!/usr/bin/env python3
"""Figures and numbers for the step-by-step pipeline report (outputs/report/).

Re-runs the coarse -> fine pipeline on one StereoLunar nadir image and records every
intermediate product (rectified images, NCC surfaces, RANSAC and Mahalanobis inputs) by
wrapping the estimator entry points; the algorithm code is not modified. Aggregate charts
come from the existing V2/V3/calibration run logs.

    .venv/bin/python scripts/report/make_report_figures.py --out outputs/report
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402

from lunar_trn import coarse as coarse_mod  # noqa: E402
from lunar_trn import fine as fine_mod  # noqa: E402
from lunar_trn import frames  # noqa: E402
from lunar_trn.coarse import _footprint_box, coarse_localize, select_templates  # noqa: E402
from lunar_trn.datasets import stereolunar as sl  # noqa: E402
from lunar_trn.dem import Dem, DemSource  # noqa: E402
from lunar_trn.experiments import build_mapset, sample_prior, score  # noqa: E402
from lunar_trn.fine import fine_localize  # noqa: E402
from lunar_trn.localize import LocalizerConfig, MapSet, localize  # noqa: E402
from lunar_trn.maps import TrnMap, cast_shadows_grid, render_maps, resample_regular, shade_grid  # noqa: E402
from lunar_trn.matching import analyze_peak, biquadratic_fit, correlate, local_normalize  # noqa: E402
from lunar_trn.rectify import predicted_map_location, rectify_to_map  # noqa: E402
from lunar_trn.render import render_perspective  # noqa: E402
from lunar_trn.validation.v1 import read_frame_window, sun_from_az_el  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/stereolunar/nadir"
DEM_PATH = ROOT / "data/dem/ldem_875s_5m.img"
NOMINAL_BANDS_M = [3500 + 3000 * k for k in range(10)]
BANDS_KM = [3.5 + 3 * k for k in range(10)]

# Report palette (figure plates are always light; matches docs/validation figures).
PLATE = "#fbfbfa"
INK = "#1b1d21"
INK2 = "#55585e"
GRID = "#e3e4e6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
YELLOW = "#eda100"
RED = "#e34948"
NEUTRAL = "#9a9ca1"

plt.rcParams.update({
    "figure.facecolor": PLATE, "axes.facecolor": PLATE, "savefig.facecolor": PLATE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "legend.frameon": False, "legend.fontsize": 8,
})


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def stretch(a, lo=1.0, hi=99.0):
    """Percentile stretch to [0, 1] (NaN kept)."""
    a = np.asarray(a, float)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return a
    p0, p1 = np.percentile(f, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-12), 0, 1)


def anaglyph(moving, ref):
    """Red = moving (warped image), cyan = reference (map). Aligned content looks grey."""
    m = np.nan_to_num(stretch(moving), nan=0.0)
    r = np.nan_to_num(stretch(ref), nan=0.0)
    return np.dstack([m, r, r])


def checker(a, b, n=8):
    a = np.nan_to_num(stretch(a), nan=0.0)
    b = np.nan_to_num(stretch(b), nan=0.0)
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((yy * n // H + xx * n // W) % 2).astype(bool)
    return np.where(mask, a, b)


def gray(ax, a, title=None, **kw):
    ax.imshow(a, cmap="gray", interpolation="nearest", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


def noaxis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, out: Path, name: str, dpi=140):
    p = out / "fig" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("  wrote", p.relative_to(out), flush=True)


def enu_of(T_true, p):
    enu = frames.enu_basis(T_true[:3, 3])
    return enu.T @ (np.asarray(p) - T_true[:3, 3])


def scalebar(ax, px_per_m, length_m, label, loc=(0.05, 0.06), color="white"):
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    w = ax.get_xlim()[1] - x0
    h = ax.get_ylim()[1] - y0
    L = length_m * px_per_m
    xa = x0 + loc[0] * w
    ya = y0 + (loc[1]) * h
    ax.plot([xa, xa + L], [ya, ya], color=color, lw=3, solid_capstyle="butt")
    ax.text(xa + L / 2, ya + 0.03 * h * np.sign(-h), label, color=color, ha="center", va="bottom", fontsize=8,
            fontweight="bold")


# --------------------------------------------------------------------------------------
# example run with captured intermediates
# --------------------------------------------------------------------------------------
class Capture:
    """Wrap estimator entry points to record their inputs and outputs."""

    def __init__(self):
        self.calls = defaultdict(list)
        self._orig = {}

    def wrap(self, module, name, tag):
        orig = getattr(module, name)
        self._orig[(module, name)] = orig

        def wrapper(*a, **k):
            out = orig(*a, **k)
            self.calls[tag].append({"args": a, "kwargs": k, "out": out})
            return out

        setattr(module, name, wrapper)

    def restore(self):
        for (module, name), orig in self._orig.items():
            setattr(module, name, orig)


def load_group(geom_id):
    fr = sl.list_frames(DATA)
    groups = [g for g in sl.assign_triplets(fr) if g[0].illumination is not None]
    return next(g for g in groups if g[0].index == geom_id)


def frame_info(f):
    cam = f.camera()
    depth = f.depth()
    c = cam.height // 2
    alt = float(np.nanmedian(depth[c - 6:c + 6, c - 6:c + 6]))
    band = int(np.argmin([abs(alt - b) for b in NOMINAL_BANDS_M]))
    return cam, depth, alt, band


def get_mapset(src, cam, T, sun, gsd_nominal, cache: Path, tag: str) -> MapSet:
    pf, pc = cache / f"{tag}_fine.npz", cache / f"{tag}_coarse.npz"
    if pf.exists() and pc.exists():
        return MapSet(coarse=TrnMap.load(pc), fine=TrnMap.load(pf))
    t0 = time.time()
    ms = build_mapset(src, cam, T, sun, gsd_nominal, 3000.0)
    ms.fine.save(pf)
    ms.coarse.save(pc)
    print(f"  rendered mapset {tag} in {time.time() - t0:.0f} s", flush=True)
    return ms


def run_example(src, geom, illum, prior_seed, cache: Path, delta_az: float = 0.0):
    g = load_group(geom)
    f = g[illum]
    cam, depth, alt, band = frame_info(f)
    T = f.T_world_cam
    img = f.image()
    gsd_nom = NOMINAL_BANDS_M[band] / cam.fx
    sun = f.sun_direction()
    if delta_az:
        az0, el0 = sl.sun_azimuth_elevation(sun, T[:3, 3])
        sun = sun_from_az_el(T[:3, 3], (az0 + delta_az) % 360.0, el0)
    ms = get_mapset(src, cam, T, sun, gsd_nom, cache, f"g{geom}_i{illum}_d{delta_az:g}")
    rng = np.random.default_rng(prior_seed)
    Tp, cov, perr = sample_prior(T, rng, 3000.0, 120.0, 0.0)
    cfg = LocalizerConfig()
    run_seed = prior_seed
    res = localize(img, cam.K, Tp, cov, ms, cfg, seed=run_seed)
    sc = score(res, T, ms.fine.pixel)

    # Replicate localize() step by step with the same RNG stream, capturing internals.
    cap = Capture()
    cap.wrap(coarse_mod, "ransac_position", "coarse_ransac")
    cap.wrap(fine_mod, "ransac_position", "fine_ransac")
    cap.wrap(fine_mod, "mahalanobis_gate", "mahal")
    cap.wrap(fine_mod, "refine_pose", "refine")
    try:
        rr = np.random.default_rng(run_seed)
        enu = frames.enu_basis(Tp[:3, 3])
        Pp = enu.T @ cov[:3, :3] @ enu
        sigma_h = float(np.sqrt(0.5 * (Pp[0, 0] + Pp[1, 1])))
        sigma_att = max(float(np.sqrt(np.mean(np.diag(cov)[3:]))), 1e-6)
        cres = coarse_localize(img, cam, Tp, sigma_h, ms.coarse, cfg.coarse,
                               cmap_norm=ms.normalized("coarse", cfg.coarse.norm_win), rng=rr)
        fines, Ts = [], []
        T1 = Tp.copy()
        if cres.ok:
            T1[:3, 3] = cres.position
            Ts.append(T1)
        Tcur = T1
        for _ in range(cfg.fine_iterations if cres.ok else 0):
            fr_ = fine_localize(img, cam, Tcur, ms.fine, cfg.fine, fmap_norm=ms.normalized("fine", cfg.fine.norm_win),
                                rng=rr, att_prior_R=Tp[:3, :3], att_prior_sigma_rad=sigma_att)
            fines.append(fr_)
            if not fr_.ok:
                break
            Tcur = np.eye(4)
            Tcur[:3, :3] = fr_.R_wc
            Tcur[:3, 3] = fr_.position
            Ts.append(Tcur)
    finally:
        cap.restore()
    same = np.allclose((Ts[-1] if Ts else Tp)[:3, 3], res.T_world_cam[:3, 3], atol=1e-6)
    print(f"  example g{geom} i{illum} daz {delta_az:g}: alt {alt:.0f} m band {band} prior err {perr['prior_err_h_m']:.0f} m -> "
          f"{res.status} err_h {sc['err_h_m']:.2f} m (replica identical: {same})", flush=True)
    return dict(delta_az=delta_az, group=g, frame=f, cam=cam, depth=depth, alt=alt, band=band, T=T, img=img, sun=sun, ms=ms,
                Tp=Tp, cov=cov, perr=perr, cfg=cfg, res=res, score=sc, cres=cres, fines=fines, Ts=Ts,
                cap=cap.calls, sigma_h=sigma_h, sigma_att=sigma_att, gsd_nom=gsd_nom, replica_ok=bool(same))


def coarse_internals(ex):
    """Recompute the coarse warping, normalisation, template choice and NCC surfaces."""
    img, cam, Tp, cmap, cfg = ex["img"], ex["cam"], ex["Tp"], ex["ms"].coarse, ex["cfg"].coarse
    h_ref = float(np.nanmedian(cmap.heights))
    c_lo, r_lo, c_hi, r_hi = _footprint_box(cam, Tp, cmap, h_ref)
    rows, cols = cmap.shape
    c_lo, r_lo = max(c_lo, 0), max(r_lo, 0)
    c_hi, r_hi = min(c_hi, cols - 1), min(r_hi, rows - 1)
    alt_prior = np.linalg.norm(Tp[:3, 3]) - frames.MOON_RADIUS_M - h_ref
    gsd_img = alt_prior / cam.fx
    ss = cfg.supersample if cmap.pixel > 1.5 * gsd_img else 1
    w, h = c_hi - c_lo + 1, r_hi - r_lo + 1
    rect = rectify_to_map(img, cam, Tp, cmap, c_lo, r_lo, w, h, supersample=ss)
    rect_true = rectify_to_map(img, cam, ex["T"], cmap, c_lo, r_lo, w, h, supersample=ss)
    rect_n, rect_std = local_normalize(rect, cfg.norm_win)
    rect_n = np.where(np.isfinite(rect), rect_n, np.nan)
    cmap_norm = ex["ms"].normalized("coarse", cfg.norm_win)
    chosen = select_templates(rect, cfg.template_px, cfg.n_templates, cfg.min_template_spacing)
    radius_px = int(math.ceil((cfg.search_sigma * ex["sigma_h"] + cfg.search_margin_m) / cmap.pixel))
    radius_px = min(radius_px, (cfg.max_search_px - cfg.template_px) // 2)
    half = cfg.template_px // 2
    tpls = []
    for score_, r, c in chosen:
        tpl = rect_n[r:r + cfg.template_px, c:c + cfg.template_px]
        mc0, mr0 = c_lo + c, r_lo + r
        sc0, sr0 = max(mc0 - radius_px, 0), max(mr0 - radius_px, 0)
        sc1 = min(mc0 + cfg.template_px + radius_px, cols)
        sr1 = min(mr0 + cfg.template_px + radius_px, rows)
        search = cmap_norm[sr0:sr1, sc0:sc1]
        surf = correlate(search, np.nan_to_num(tpl))
        pk = analyze_peak(surf, exclusion=5)
        mx, my = sc0 + pk.x - mc0, sr0 + pk.y - mr0
        # Ground-truth offset of this template (true map location of the prior-projected centre).
        tc, tr = mc0 + half - 0.5, mr0 + half - 0.5
        P_prior = cmap.world_points(np.array([tc]), np.array([tr]))
        u, v, _ = cam.project(Tp, P_prior)
        mct, mrt = predicted_map_location(cam, ex["T"], cmap, u, v)
        tpls.append(dict(score=score_, r=r, c=c, mc0=mc0, mr0=mr0, search_box=(sc0, sr0, sc1, sr1), surf=surf,
                         pk=pk, offset=(mx, my), true_offset=(float(mct[0] - tc), float(mrt[0] - tr)),
                         tpl=tpl, tpl_raw=rect[r:r + cfg.template_px, c:c + cfg.template_px]))
    import cv2

    filled = np.where(np.isfinite(rect), rect, 0.0).astype(np.float32)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    return dict(box=(c_lo, r_lo, c_hi, r_hi), rect=rect, rect_true=rect_true, rect_n=rect_n, rect_std=rect_std,
                cmap_norm=cmap_norm, chosen=chosen, radius_px=radius_px, tpls=tpls, energy=energy, ss=ss,
                gsd_img=gsd_img)


def fine_internals(ex, it=0, feature_ids=None):
    """Warp for fine iteration ``it`` and NCC surfaces of selected features."""
    img, cam, fmap, cfg = ex["img"], ex["cam"], ex["ms"].fine, ex["cfg"].fine
    T_init = ex["Ts"][it]
    h_ref = float(np.nanmedian(fmap.heights))
    rows_m, cols_m = fmap.shape
    b_lo_c, b_lo_r, b_hi_c, b_hi_r = _footprint_box(cam, T_init, fmap, h_ref)
    pad = cfg.template_px
    b_lo_c, b_lo_r = max(b_lo_c - pad, 0), max(b_lo_r - pad, 0)
    b_hi_c, b_hi_r = min(b_hi_c + pad, cols_m - 1), min(b_hi_r + pad, rows_m - 1)
    alt = float(np.linalg.norm(T_init[:3, 3]) - frames.MOON_RADIUS_M - h_ref)
    map_to_img = fmap.pixel / (alt / cam.fx)
    ss = cfg.supersample if map_to_img > 1.5 else 1
    W, H = b_hi_c - b_lo_c + 1, b_hi_r - b_lo_r + 1
    rect = rectify_to_map(img, cam, T_init, fmap, b_lo_c, b_lo_r, W, H, supersample=ss)
    rect_n, _ = local_normalize(rect, cfg.norm_win)
    fmap_norm = ex["ms"].normalized("fine", cfg.norm_win)
    lm = ex["fines"][it].landmarks
    out = dict(box=(b_lo_c, b_lo_r, b_hi_c, b_hi_r), rect=rect, rect_n=rect_n, map_to_img=map_to_img, feats={})
    if feature_ids is None:
        return out
    fc = np.asarray(lm["feat_col"], float)
    fr = np.asarray(lm["feat_row"], float)
    mc, mr = predicted_map_location(cam, T_init, fmap, fc, fr)
    ht, hs = cfg.template_px // 2, cfg.search_px // 2
    for i in feature_ids:
        if not (np.isfinite(mc[i]) and np.isfinite(mr[i])):
            continue
        cc, rc = int(round(mc[i])), int(round(mr[i]))
        tr0, tc0 = rc - ht - b_lo_r, cc - ht - b_lo_c
        sr0, sc0 = rc - hs, cc - hs
        if tr0 < 0 or tc0 < 0 or sr0 < 0 or sc0 < 0:
            continue
        tpl = rect_n[tr0:tr0 + cfg.template_px, tc0:tc0 + cfg.template_px]
        tpl_raw = rect[tr0:tr0 + cfg.template_px, tc0:tc0 + cfg.template_px]
        search_n = fmap_norm[sr0:sr0 + cfg.search_px, sc0:sc0 + cfg.search_px]
        search_raw = fmap.image[sr0:sr0 + cfg.search_px, sc0:sc0 + cfg.search_px]
        if tpl.shape != (cfg.template_px,) * 2 or search_n.shape != (cfg.search_px,) * 2:
            continue
        surf = correlate(search_n, tpl)
        pk = analyze_peak(surf, exclusion=3)
        out["feats"][i] = dict(cc=cc, rc=rc, sr0=sr0, sc0=sc0, tpl=tpl, tpl_raw=tpl_raw, search_n=search_n,
                               search_raw=search_raw, surf=surf, pk=pk)
    return out


def fine_categories(ex, it):
    """Per-landmark category for iteration ``it``: inlier / geometric / mahalanobis / peak reject."""
    fr_ = ex["fines"][it]
    lm = fr_.landmarks
    n = len(lm["feat_col"])
    status = np.array([s if s is not None else "" for s in lm["status"]], dtype=object)
    peaks = np.asarray(lm["peak"], float)
    good = np.nonzero(status == "")[0]
    good = good[np.argsort(-peaks[good])][: ex["cfg"].fine.max_landmarks]
    cat = np.where(status == "", "unused", status).astype(object)
    rs = ex["cap"]["fine_ransac"][it]["out"]
    mh = ex["cap"]["mahal"][it]
    keep, gamma, d2 = mh["out"]
    cat[good] = "inlier"
    cat[good[~rs.inliers]] = "geometric"
    idx_inl = good[rs.inliers]
    if (~keep).sum() > 0 and keep.sum() >= 3:
        cat[idx_inl[~keep]] = "mahalanobis"
    assert np.array_equal(np.nonzero(cat == "inlier")[0], np.nonzero(lm["inlier"])[0])
    d2_all = np.full(n, np.nan)
    d2_all[idx_inl] = d2
    return dict(cat=cat, good=good, d2=d2_all, gamma=gamma, rs=rs, status=status)


# --------------------------------------------------------------------------------------
# geometry helpers for drawing
# --------------------------------------------------------------------------------------
def footprint_poly(cam, T, m: TrnMap, h=None, n=12):
    """Image border projected onto the terrain sphere at height h, in map (col, row)."""
    h = float(np.nanmedian(m.heights)) if h is None else h
    e = np.linspace(-0.5, cam.width - 0.5, n)
    lo, hi = -0.5, cam.width - 0.5
    cols = np.concatenate([e, np.full(n, hi), e[::-1], np.full(n, lo)])
    rows = np.concatenate([np.full(n, lo), e, np.full(n, hi), e[::-1]])
    o, d = cam.rays_world(T, cols, rows)
    d = d.reshape(-1, 3)
    r = frames.MOON_RADIUS_M + h
    b = d @ o
    c = o @ o - r * r
    t = -b - np.sqrt(np.maximum(b * b - c, 0.0))
    P = o + t[:, None] * d
    x, y, _ = frames.body_to_stereo(P)
    return np.column_stack(m.xy_to_colrow(x, y))


def nadir_colrow(T, m: TrnMap):
    x, y, _ = frames.body_to_stereo(T[:3, 3])
    return m.xy_to_colrow(x, y)


def sun_vec_map(sun, m: TrnMap):
    from lunar_trn.maps import tangent_frame

    cx = m.x0 + 0.5 * m.shape[1] * m.pixel
    cy = m.y0 - 0.5 * m.shape[0] * m.pixel
    _, B = tangent_frame((cx, cy))
    st = np.asarray(sun) @ B
    v = np.array([st[0], -st[1]])
    return v / np.linalg.norm(v), math.degrees(math.atan2(st[2], math.hypot(st[0], st[1])))


def sun_arrow(ax, v, xy=(0.86, 0.86), L=0.1, label="Sun"):
    x, y = xy
    ax.annotate("", xy=(x + L * v[0], y - L * v[1]), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2.2, mutation_scale=14))
    ax.text(x - 0.02 * np.sign(v[0]), y + 0.02 * np.sign(v[1]), label, transform=ax.transAxes, color=YELLOW,
            fontsize=8, fontweight="bold", ha="right" if v[0] > 0 else "left", va="top" if v[1] > 0 else "bottom")


# --------------------------------------------------------------------------------------
# figures: inputs and frames
# --------------------------------------------------------------------------------------
def fig_inputs(ex, out):
    g = ex["group"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw=dict(wspace=0.06))
    names = ["Side", "Overhead", "Back"]
    for k in range(3):
        f = g[k]
        az, el = sl.sun_azimuth_elevation(f.sun_direction(), f.boresight_point())
        gray(axs[k], f.image(), vmin=0, vmax=1)
        axs[k].set_title(f"{names[k]} · az {az:.0f}°, el {el:.0f}°", fontsize=9)
        if k == 0:
            for s in axs[k].spines.values():
                s.set_visible(True)
                s.set_color(BLUE)
                s.set_linewidth(2.5)
    d = ex["depth"] / 1000.0
    im = axs[3].imshow(d, cmap="cividis")
    noaxis(axs[3])
    axs[3].set_title("EXR z-depth (km)", fontsize=9)
    cb = fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    save(fig, out, "f01_inputs")


def fig_overview(ex, src: DemSource, out):
    step = 40
    h = np.asarray(src._mm[::step, ::step], np.float32) * np.float32(src.scale)
    px = src.pixel * step
    gy, gx = np.gradient(h, px)
    az = math.radians(315)
    hs = np.clip(0.55 + 0.45 * (-(gx * math.sin(az)) + gy * math.cos(az)) / np.sqrt(1 + gx**2 + gy**2) * 3, 0, 1)
    fr = sl.list_frames(DATA)
    groups = [g for g in sl.assign_triplets(fr) if g[0].illumination is not None]
    xs, ys, alts = [], [], []
    for g in groups:
        x, y, _ = frames.body_to_stereo(g[0].boresight_point())
        xs.append(x / 1000)
        ys.append(y / 1000)
        alts.append(g[0].altitude_above_sphere() / 1000)
    ext = [src.extent[0] / 1000, src.extent[1] / 1000, src.extent[2] / 1000, src.extent[3] / 1000]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.32))
    ax = axs[0]
    ax.imshow(hs, cmap="gray", extent=ext, vmin=0, vmax=1)
    sc = ax.scatter(xs, ys, c=alts, cmap="Blues", vmin=0, vmax=32, s=14, edgecolors="#20242a", linewidths=0.4)
    x0, y0, _ = frames.body_to_stereo(ex["frame"].boresight_point())
    ax.scatter([x0 / 1000], [y0 / 1000], marker="*", s=260, color=ORANGE, edgecolors="white", linewidths=1.2, zorder=5)
    ax.set_xlabel("polar stereographic x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("LDEM_875S_5M and the 450 nadir geometries")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("camera altitude (km)")
    cb.outline.set_visible(False)
    # Zoom on the example.
    ax = axs[1]
    half = 9000.0
    dem = src.read_window(x0 - half, x0 + half, y0 - half, y0 + half)
    gy2, gx2 = np.gradient(dem.heights, dem.pixel)
    hs2 = np.clip(0.55 + 0.45 * (-(gx2 * math.sin(az)) + gy2 * math.cos(az)) / np.sqrt(1 + gx2**2 + gy2**2) * 2.5,
                  0, 1)
    e2 = [(dem.x0 - x0) / 1000, (dem.x0 + dem.shape[1] * dem.pixel - x0) / 1000,
          (dem.y0 - dem.shape[0] * dem.pixel - y0) / 1000, (dem.y0 - y0) / 1000]
    im = ax.imshow(dem.heights / 1000, cmap="cividis", extent=e2)
    ax.imshow(hs2, cmap="gray", extent=e2, alpha=0.45, vmin=0, vmax=1)
    poly = footprint_poly(ex["cam"], ex["T"], ex["ms"].fine)
    xy = np.column_stack(ex["ms"].fine.colrow_to_xy(poly[:, 0], poly[:, 1]))
    ax.plot(np.r_[xy[:, 0], xy[0, 0]] / 1000 - x0 / 1000, np.r_[xy[:, 1], xy[0, 1]] / 1000 - y0 / 1000, color=ORANGE,
            lw=2)
    ax.set_title(f"Geometry {ex['group'][0].index}: footprint at {ex['alt'] / 1000:.1f} km")
    ax.set_xlabel("x − x₀ (km)")
    ax.set_ylabel("y − y₀ (km)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("height above 1737.4 km sphere (km)")
    cb.outline.set_visible(False)
    save(fig, out, "f02_overview")
    return dict(n_geometries=len(groups), dem_relief_km=[float(np.nanmin(dem.heights) / 1000),
                                                         float(np.nanmax(dem.heights) / 1000)])


def fig_v1_render(ex, src, out):
    from lunar_trn.validation.v1 import ncc

    dem = read_frame_window(src, ex["cam"], ex["T"], ex["sun"], margin_m=300.0, shadow_reach_m=12000.0)
    r = render_perspective(dem, ex["cam"], ex["T"], ex["sun"], step=1, model="hapke")
    ren = r["image"]
    img = ex["img"]
    m = np.isfinite(ren)
    val = ncc(img, np.nan_to_num(ren), m)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw=dict(wspace=0.05))
    gray(axs[0], img, "StereoLunar image (SurRender)")
    gray(axs[1], stretch(ren, 0.5, 99.5), "Our Hapke render, GT pose")
    axs[2].imshow(anaglyph(img, ren))
    noaxis(axs[2])
    axs[2].set_title("Overlay (red image, cyan render)")
    zc = 160
    sl_ = (slice(256 - zc // 2, 256 + zc // 2), slice(256 - zc // 2, 256 + zc // 2))
    axs[3].imshow(checker(img[sl_], ren[sl_], 6), cmap="gray", interpolation="nearest")
    noaxis(axs[3])
    axs[3].set_title("Centre 160 px checkerboard")
    save(fig, out, "f03_v1_render")
    return dict(v1_render_ncc=float(val))


# --------------------------------------------------------------------------------------
# figures: map generation
# --------------------------------------------------------------------------------------
def fig_mapgen(src, out, geom=18780, illum=0):
    """Map generation steps on a shadow-rich geometry (DEM -> shading -> shadows -> map)."""
    from lunar_trn.validation.v1 import footprint_bounds

    f = load_group(geom)[illum]
    cam, _, alt, band = frame_info(f)
    T, sun = f.T_world_cam, f.sun_direction()
    px = round(NOMINAL_BANDS_M[band] / cam.fx, 1)
    xmin, xmax, ymin, ymax = footprint_bounds(cam, T, h_lo=-2500.0, h_hi=500.0)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    n = int(round(max(xmax - xmin, ymax - ymin) / px))
    half = 0.5 * n * px
    x0, y0 = cx - half + 0.5 * px, cy + half - 0.5 * px
    grid = TrnMap(np.zeros((n, n), np.float32), np.zeros((n, n), np.float32), x0, y0, px, sun)
    v, sun_el = sun_vec_map(sun, grid)
    reach = 12000.0
    dx, dy = v[0] * reach, -v[1] * reach
    m = 4 * px
    gxmin, gxmax, gymin, gymax = cx - half - m, cx + half + m, cy - half - m, cy + half + m
    dem_ext = src.read_window(min(gxmin, gxmin + dx), max(gxmax, gxmax + dx), min(gymin, gymin + dy),
                              max(gymax, gymax + dy))
    dem_loc = src.read_window(gxmin, gxmax, gymin, gymax)

    def regrid(dem, gx0, gy0, nr, nc):
        return Dem(resample_regular(dem, gx0, gy0, px, nr, nc), gx0, gy0, px, name="w", radius=dem.radius)

    e_c0 = int(math.ceil((dem_ext.x0 - x0) / px)) + 1
    e_r0 = int(math.ceil((y0 - dem_ext.y0) / px)) + 1
    e_nc = int((dem_ext.shape[1] - 1) * dem_ext.pixel / px) - 2
    e_nr = int((dem_ext.shape[0] - 1) * dem_ext.pixel / px) - 2
    dw = regrid(dem_ext, x0 + e_c0 * px, y0 - e_r0 * px, e_nr, e_nc)
    I0, _ = shade_grid(dw, sun, shadows=False)
    sh_ext = cast_shadows_grid(dw, sun)
    oc, orr = -e_c0, -e_r0
    crop = (slice(orr, orr + n), slice(oc, oc + n))
    dl = regrid(dem_loc, x0, y0, n, n)
    sh_loc = cast_shadows_grid(dl, sun)
    H = dw.heights[crop]
    shade = I0[crop]
    attached = shade <= 0
    cast = sh_ext[crop] & ~attached
    shadow = cast | attached
    final = np.where(shadow, 0.0, shade)
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.75), gridspec_kw=dict(wspace=0.07))
    im = axs[0].imshow(H / 1000, cmap="cividis")
    noaxis(axs[0])
    axs[0].set_title("① DEM height (km)")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    gray(axs[1], stretch(shade, 0.5, 99.5), "② Hapke shading only")
    mask_rgb = np.ones((n, n, 3)) * 0.93
    mask_rgb[attached] = [0.45, 0.46, 0.48]
    mask_rgb[cast] = [0.16, 0.47, 0.84]
    axs[2].imshow(mask_rgb, interpolation="nearest")
    noaxis(axs[2])
    axs[2].set_title("③ Shadow mask")
    gray(axs[3], stretch(final, 0.5, 99.5), "④ Rendered map")
    for a in axs:
        sun_arrow(a, v)
    save(fig, out, "f04_mapgen")
    # A tile inside the cast shadow: scanned with its own DEM only vs with the sunward extension.
    rr, cc_ = np.nonzero(cast)
    tn = n // 3
    tr0 = int(np.clip(np.median(rr) - tn / 2, 0, n - tn))
    tc0 = int(np.clip(np.median(cc_) - tn / 2, 0, n - tn))
    tx0, ty0 = x0 + tc0 * px, y0 - tr0 * px
    dem_t = src.read_window(tx0 - m, tx0 + tn * px + m, ty0 - tn * px - m, ty0 + m)
    sh_tile = cast_shadows_grid(regrid(dem_t, tx0, ty0, tn, tn), sun)
    tsl = (slice(tr0, tr0 + tn), slice(tc0, tc0 + tn))
    t_att = attached[tsl]
    t_ext = cast[tsl]
    t_loc = sh_tile & ~t_att
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.06))
    hsw = np.where(sh_ext, 0.12, 1.0) * stretch(I0, 0.5, 99.5)
    axs[0].imshow(hsw, cmap="gray", interpolation="nearest")
    axs[0].add_patch(Rectangle((oc, orr), n, n, fill=False, ec=ORANGE, lw=2.2))
    axs[0].add_patch(Rectangle((oc + tc0, orr + tr0), tn, tn, fill=False, ec=BLUE, lw=2.0))
    ccx, ccy = oc + n / 2, orr + n / 2
    L = reach / px
    axs[0].annotate("", xy=(ccx + v[0] * L, ccy + v[1] * L), xytext=(ccx, ccy),
                    arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2.6, mutation_scale=16))
    noaxis(axs[0])
    axs[0].set_title("DEM window: map + 12 km sunward")

    def tile_rgb(cast_mask):
        b = np.dstack([stretch(shade[tsl], 0.5, 99.5)] * 3)
        b[t_att] = [0.45, 0.46, 0.48]
        b[cast_mask] = [0.16, 0.47, 0.84]
        return b

    axs[1].imshow(tile_rgb(t_loc), interpolation="nearest")
    noaxis(axs[1])
    axs[1].set_title(f"Tile-only scan: cast {t_loc.mean() * 100:.0f}%")
    axs[2].imshow(tile_rgb(t_ext), interpolation="nearest")
    noaxis(axs[2])
    axs[2].set_title(f"With 12 km extension: cast {t_ext.mean() * 100:.0f}%")
    for a_ in axs[1:]:
        for sp in a_.spines.values():
            sp.set_visible(True)
            sp.set_color(BLUE)
            sp.set_linewidth(2)
    save(fig, out, "f05_shadow_reach")
    missed = cast & ~sh_loc
    return dict(geom=int(geom), alt_m=float(alt), map_px=float(px), shadow_frac=float(shadow.mean()),
                cast_frac=float(cast.mean()), missed_frac=float(missed.mean()), sun_el_map_deg=float(sun_el),
                image_dark_frac=float((f.image() < 0.02).mean()), tile_cast_local=float(t_loc.mean()),
                tile_cast_ext=float(t_ext.mean()), tile_km=float(tn * px / 1000))


def fig_map_vs_image(ex, src, out):
    """Each illumination: image warped with the GT pose vs our map rendered with that Sun."""
    from lunar_trn.validation.v1 import ncc

    cam, T, fmap = ex["cam"], ex["T"], ex["ms"].fine
    poly = footprint_poly(cam, T, fmap)
    c0, r0 = np.floor(poly.min(axis=0)).astype(int) + 20
    c1, r1 = np.ceil(poly.max(axis=0)).astype(int) - 20
    cxy = fmap.colrow_to_xy(0.5 * (c0 + c1), 0.5 * (r0 + r1))
    half = 0.5 * max(c1 - c0, r1 - r0) * fmap.pixel
    names = ["Side", "Overhead", "Back"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.6), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    vals = []
    for k in range(3):
        f = ex["group"][k]
        sun_k = f.sun_direction()
        mk = render_maps(src, (float(cxy[0]), float(cxy[1])), half, [fmap.pixel], sun_k)[0]
        rect = rectify_to_map(f.image(), cam, T, mk, 0, 0, mk.shape[1], mk.shape[0], supersample=1)
        val = ncc(np.nan_to_num(rect), mk.image, np.isfinite(rect))
        vals.append(float(val))
        gray(axs[0, k], stretch(rect, 0.5, 99.5), f"{names[k]} · image, GT warp")
        gray(axs[1, k], stretch(mk.image, 0.5, 99.5), f"{names[k]} · rendered map (NCC {val:.2f})")
        v, _ = sun_vec_map(sun_k, mk)
        sun_arrow(axs[1, k], v)
    save(fig, out, "f06_map_vs_image")
    return dict(map_image_ncc=vals)


# --------------------------------------------------------------------------------------
# figures: prior and coarse stage
# --------------------------------------------------------------------------------------
def fig_prior(ex, ci, out):
    cmap, fmap, cam = ex["ms"].coarse, ex["ms"].fine, ex["cam"]
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.08))
    ax = axs[0]
    gray(ax, stretch(cmap.image, 0.5, 99.5))
    pt = footprint_poly(cam, ex["T"], cmap)
    pp = footprint_poly(cam, ex["Tp"], cmap)
    ax.fill(pt[:, 0], pt[:, 1], fc="none", ec=AQUA, lw=2.2)
    ax.fill(pp[:, 0], pp[:, 1], fc="none", ec=ORANGE, lw=2.2, ls="--")
    nc_t, nr_t = nadir_colrow(ex["T"], cmap)
    nc_p, nr_p = nadir_colrow(ex["Tp"], cmap)
    ax.add_patch(Circle((nc_t, nr_t), 3000 / cmap.pixel, fill=False, ec="white", lw=1.2, ls=":"))
    ax.plot([nc_p, nc_t], [nr_p, nr_t], color="white", lw=1.2)
    ax.scatter([nc_t], [nr_t], marker="*", s=180, color=AQUA, edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter([nc_p], [nr_p], marker="o", s=60, color=ORANGE, edgecolors="white", linewidths=0.8, zorder=5)
    c_lo, r_lo, c_hi, r_hi = ci["box"]
    R = ci["radius_px"]
    ax.add_patch(Rectangle((c_lo - R, r_lo - R), c_hi - c_lo + 2 * R, r_hi - r_lo + 2 * R, fill=False, ec=YELLOW,
                           lw=1.2, ls="-."))
    ax.set_title(f"Coarse map ({cmap.pixel:.1f} m/px, {cmap.shape[1]}×{cmap.shape[0]} px)")
    from matplotlib.lines import Line2D

    hnd = [Line2D([], [], color=AQUA, lw=2, label="GT footprint ★"),
           Line2D([], [], color=ORANGE, lw=2, ls="--", label="prior footprint ●"),
           Line2D([], [], color="white", lw=1.2, ls=":", label="3 km prior radius"),
           Line2D([], [], color=YELLOW, lw=1.2, ls="-.", label="coarse search reach")]
    lg = ax.legend(handles=hnd, loc="lower left", fontsize=7, frameon=True, facecolor="#000000", framealpha=0.5,
                   labelcolor="white", edgecolor="none")
    lg.get_frame().set_linewidth(0)
    # Resolution comparison on the same 1.6 km patch around the GT nadir.
    half_m = 800.0
    for ax, m, nm in ((axs[1], cmap, "coarse"), (axs[2], fmap, "fine")):
        c, r = nadir_colrow(ex["T"], m)
        h = half_m / m.pixel
        sub = m.image[int(r - h):int(r + h), int(c - h):int(c + h)]
        gray(ax, stretch(sub, 0.5, 99.5))
        ax.set_title(f"{nm.capitalize()} map, 1.6 km patch ({m.pixel:.1f} m/px)")
    save(fig, out, "f07_prior_maps")


def fig_coarse_rectify(ex, ci, out):
    cmap = ex["ms"].coarse
    c_lo, r_lo, c_hi, r_hi = ci["box"]
    mp = cmap.image[r_lo:r_hi + 1, c_lo:c_hi + 1]
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.7), gridspec_kw=dict(wspace=0.05))
    gray(axs[0], stretch(ci["rect"], 0.5, 99.5), "Image warped, prior pose")
    gray(axs[1], stretch(mp, 0.5, 99.5), "Coarse map, same box")
    axs[2].imshow(anaglyph(ci["rect"], mp))
    noaxis(axs[2])
    axs[2].set_title("Overlay at prior")
    axs[3].imshow(anaglyph(ci["rect_true"], mp))
    noaxis(axs[3])
    axs[3].set_title("Overlay at GT (reference)")
    save(fig, out, "f08_coarse_rectify")


def fig_normalize(ex, ci, out):
    cmap = ex["ms"].coarse
    c_lo, r_lo, c_hi, r_hi = ci["box"]
    mp = cmap.image[r_lo:r_hi + 1, c_lo:c_hi + 1]
    mpn = ci["cmap_norm"][r_lo:r_hi + 1, c_lo:c_hi + 1]
    H, W = mp.shape
    s = 160
    r0, c0 = H // 2 - s // 2, W // 2 - s // 2
    sl_ = (slice(r0, r0 + s), slice(c0, c0 + s))
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.6), gridspec_kw=dict(wspace=0.05))
    gray(axs[0], ci["rect"][sl_], "Warped image (raw)")
    gray(axs[1], np.clip(ci["rect_n"][sl_], -3, 3), "Warped image, normalised")
    gray(axs[2], mp[sl_], "Map (raw)")
    gray(axs[3], np.clip(mpn[sl_], -3, 3), "Map, normalised")
    for a in axs:
        a.add_patch(Rectangle((6, 6), 21, 21, fill=False, ec=YELLOW, lw=1.5))
    save(fig, out, "f09_normalize")


def fig_templates(ex, ci, out):
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.4), gridspec_kw=dict(wspace=0.05))
    e = np.log10(ci["energy"] + 1e-6)
    e = np.where(np.isfinite(ci["rect"]), e, np.nan)
    axs[0].imshow(e, cmap="magma", vmin=np.nanpercentile(e, 5), vmax=np.nanpercentile(e, 99.5))
    noaxis(axs[0])
    axs[0].set_title("Sobel gradient energy (log)")
    gray(axs[1], stretch(ci["rect"], 0.5, 99.5), f"{len(ci['tpls'])} selected 128×128 templates (rank order)")
    for k, t in enumerate(ci["tpls"]):
        pk = t["pk"]
        rec = ex["cres"].templates[k]
        ok = rec.get("reject") is None
        inl = rec.get("geom_inlier", False)
        col = AQUA if (ok and inl) else (ORANGE if ok else RED)
        for a in axs:
            a.add_patch(Rectangle((t["c"], t["r"]), 128, 128, fill=False, ec=col, lw=1.8))
        axs[1].text(t["c"] + 4, t["r"] + 14, f"{k + 1}", color="white", fontsize=10, fontweight="bold",
                    bbox=dict(fc=col, ec="none", pad=1.2))
    save(fig, out, "f10_templates")


def fig_coarse_ncc(ex, ci, out):
    tp = ci["tpls"]
    n = len(tp)
    ncol = 4 if n > 6 else 3
    nrow = int(math.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.55 * nrow), gridspec_kw=dict(wspace=0.1, hspace=0.3))
    axs = np.atleast_2d(axs)
    R = ci["radius_px"]
    for k in range(nrow * ncol):
        ax = axs.flat[k]
        if k >= n:
            ax.axis("off")
            continue
        t = tp[k]
        rec = ex["cres"].templates[k]
        sc0, sr0, sc1, sr1 = t["search_box"]
        surf = t["surf"]
        ax.imshow(surf, cmap="magma", vmin=-0.2, vmax=1.0, interpolation="nearest")
        noaxis(ax)
        # Prior prediction (zero offset) and GT location in surface coordinates.
        zx, zy = t["mc0"] - sc0, t["mr0"] - sr0
        ax.scatter([zx], [zy], marker="+", s=90, color="white", linewidths=1.5)
        tx, ty = zx + t["true_offset"][0], zy + t["true_offset"][1]
        ax.add_patch(Circle((tx, ty), 6, fill=False, ec=AQUA, lw=1.6))
        pk = t["pk"]
        ax.scatter([pk.x], [pk.y], marker="x", s=55, color=YELLOW, linewidths=1.8)
        reason = rec.get("reject")
        verdict = ("geometric inlier" if rec.get("geom_inlier") else "geometric outlier") if reason is None else reason
        ax.set_title(f"#{k + 1}  peak {pk.height:.2f} · FWHM {pk.width:.1f} px\n{verdict}", fontsize=8.5,
                     color=INK if reason is None else RED)
        if np.isfinite(pk.x):
            ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
            hz = 14
            y0_, x0_ = max(pk.iy - hz, 0), max(pk.ix - hz, 0)
            ins.imshow(surf[y0_:pk.iy + hz + 1, x0_:pk.ix + hz + 1], cmap="magma", vmin=-0.2, vmax=1.0,
                       interpolation="nearest")
            noaxis(ins)
            for sp in ins.spines.values():
                sp.set_visible(True)
                sp.set_color("white")
    save(fig, out, "f11_coarse_ncc")


def fig_coarse_detail(ex, ci, out, k=0):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    t = ci["tpls"][k]
    cmap = ex["ms"].coarse
    sc0, sr0, sc1, sr1 = t["search_box"]
    search_raw = cmap.image[sr0:sr1, sc0:sc1]
    fig = plt.figure(figsize=(13.4, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.8, 1.25, 1.35], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    gray(ax, stretch(t["tpl_raw"], 0.5, 99.5), f"Template #{k + 1}")
    ax = fig.add_subplot(gs[1])
    gray(ax, np.clip(t["tpl"], -3, 3), "Normalised")
    ax = fig.add_subplot(gs[2])
    gray(ax, stretch(search_raw, 0.5, 99.5), "Map search window")
    zx, zy = t["mc0"] - sc0, t["mr0"] - sr0
    pk = t["pk"]
    ax.add_patch(Rectangle((zx, zy), 128, 128, fill=False, ec=ORANGE, lw=1.8, ls="--"))
    ax.add_patch(Rectangle((pk.x, pk.y), 128, 128, fill=False, ec=AQUA, lw=2.0))
    ax.annotate("", xy=(pk.x + 64, pk.y + 64), xytext=(zx + 64, zy + 64),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
    ax.text(0.02, 0.02, "‒ ‒ predicted by prior   — NCC match", transform=ax.transAxes, color="white", fontsize=7.5,
            bbox=dict(fc="#000000", alpha=0.45, ec="none", pad=2))
    ax = fig.add_subplot(gs[3], projection="3d")
    surf = t["surf"]
    st = 3
    hh, ww = (surf.shape[0] // st) * st, (surf.shape[1] // st) * st
    pooled = surf[:hh, :ww].reshape(hh // st, st, ww // st, st).max(axis=(1, 3))
    Y, X = np.mgrid[0:hh:st, 0:ww:st]
    ax.plot_surface(X, Y, pooled, cmap="magma", vmin=-0.2, vmax=1.0, linewidth=0, antialiased=True,
                    rstride=1, cstride=1)
    ax.set_zlim(-0.3, 1.0)
    ax.view_init(elev=32, azim=-58)
    ax.set_title(f"NCC surface: peak {pk.height:.2f}, PSR {pk.psr:.1f}", fontsize=9, loc="center")
    ax.set_xlabel("dx (px)", fontsize=7)
    ax.set_ylabel("dy (px)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_facecolor(PLATE)
    save(fig, out, "f12_coarse_detail")


def fig_coarse_ransac(ex, ci, out):
    cmap = ex["ms"].coarse
    px = cmap.pixel
    recs = ex["cres"].templates
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.0), gridspec_kw=dict(wspace=0.28))
    true_off = np.array([t["true_offset"] for t in ci["tpls"]]) * px
    offs = np.array([t["offset"] for t in ci["tpls"]]) * px
    cats = []
    for k, rec in enumerate(recs):
        if rec.get("reject") is not None:
            cats.append("peak")
        elif rec.get("geom_inlier"):
            cats.append("inlier")
        else:
            cats.append("outlier")
    cats = np.array(cats)
    for ax, zoom in ((axs[0], False), (axs[1], True)):
        # Map axes: +col = +x (map east), +row = -y; plot x right, y up.
        X, Y = offs[:, 0], -offs[:, 1]
        m = cats == "peak"
        ax.scatter(X[m], Y[m], marker="o", s=40, color=NEUTRAL, label="rejected by peak test", zorder=3)
        m = cats == "outlier"
        ax.scatter(X[m], Y[m], marker="X", s=70, color=ORANGE, edgecolors="white", label="geometric outlier", zorder=4)
        m = cats == "inlier"
        ax.scatter(X[m], Y[m], marker="o", s=70, color=AQUA, edgecolors="white", label="geometric inlier", zorder=5)
        ax.scatter(true_off[:, 0], -true_off[:, 1], marker="+", s=50, color=INK, linewidths=1.0, label="GT offset",
                   zorder=6)
        for k in range(len(offs)):
            ax.annotate(f"{k + 1}", (X[k], Y[k]), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK2)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlabel("map-x offset from prior prediction (m)")
        ax.set_ylabel("map-y offset (m)")
        if zoom:
            cx, cy = np.median(true_off[:, 0]), -np.median(true_off[:, 1])
            thr = ex["cfg"].coarse.geom_threshold_map_px * px
            ax.add_patch(Circle((cx, cy), thr, fill=False, ec=AQUA, lw=1, ls="--"))
            ax.set_xlim(cx - 160, cx + 160)
            ax.set_ylim(cy - 160, cy + 160)
            ax.set_title(f"Zoom ±160 m, dashed = 3 coarse px ({thr:.0f} m)")
        else:
            ax.set_title("Template offsets measured by NCC")
            ax.legend(loc="best", fontsize=7.5)
    save(fig, out, "f13_coarse_ransac")
    return dict(cats=cats.tolist())


# --------------------------------------------------------------------------------------
# figures: fine stage
# --------------------------------------------------------------------------------------
CAT_STYLE = {
    "inlier": dict(color=AQUA, marker="o", label="inlier"),
    "geometric": dict(color=ORANGE, marker="X", label="geometric outlier (RANSAC)"),
    "mahalanobis": dict(color=VIOLET, marker="^", label="Mahalanobis outlier"),
}


def fig_harris(ex, out):
    import cv2

    img = ex["img"]
    cfg = ex["cfg"].fine
    Rh = cv2.cornerHarris(img.astype(np.float32), cfg.harris_block, 3, cfg.harris_k)
    lm = ex["fines"][0].landmarks
    fc, fr = np.asarray(lm["feat_col"]), np.asarray(lm["feat_row"])
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.6), gridspec_kw=dict(wspace=0.05))
    v = np.log10(np.maximum(Rh, 0) + 1e-9)
    axs[0].imshow(v, cmap="magma", vmin=np.percentile(v, 50), vmax=np.percentile(v, 99.9))
    for k in range(0, img.shape[0] + 1, cfg.cell_px):
        axs[0].axhline(k - 0.5, color="white", lw=0.35, alpha=0.6)
        axs[0].axvline(k - 0.5, color="white", lw=0.35, alpha=0.6)
    noaxis(axs[0])
    axs[0].set_title("5×5 Harris response (log) and 32×32 px cells")
    gray(axs[1], img, f"{len(fc)} features: strongest corner per cell")
    rank = np.arange(len(fc))
    sc = axs[1].scatter(fc, fr, c=rank, cmap="Blues_r", s=26, edgecolors="white", linewidths=0.6, vmin=0,
                        vmax=len(fc) * 1.3)
    cb = fig.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.02)
    cb.set_label("rank by Harris response")
    cb.outline.set_visible(False)
    save(fig, out, "f14_harris")


def fig_fine_align(ex, out):
    """Overlay of the warped image on the fine map at the prior, after coarse and after fine."""
    fmap, cam = ex["ms"].fine, ex["cam"]
    c, r = nadir_colrow(ex["T"], fmap)
    s = 220
    c0, r0 = int(c - s / 2), int(r - s / 2)
    mp = fmap.image[r0:r0 + s, c0:c0 + s]
    poses = [("Prior pose", ex["Tp"]), ("After coarse", ex["Ts"][0]), ("After fine (final)", ex["res"].T_world_cam)]
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.6), gridspec_kw=dict(wspace=0.05))
    gray(axs[0], stretch(mp, 0.5, 99.5), f"Fine map, {s * fmap.pixel / 1000:.1f} km centre patch")
    for ax, (nm, Tq) in zip(axs[1:], poses):
        rect = rectify_to_map(ex["img"], cam, Tq, fmap, c0, r0, s, s, supersample=1)
        e = enu_of(ex["T"], Tq[:3, 3])
        ax.imshow(anaglyph(rect, mp))
        noaxis(ax)
        ax.set_title(f"{nm}: {math.hypot(e[0], e[1]):,.1f} m off")
    save(fig, out, "f15_fine_align")


def pick_landmarks(ex, cats, it=0):
    lm = ex["fines"][it].landmarks
    status = cats["status"]
    cat = cats["cat"]
    peaks = np.asarray(lm["peak"], float)
    widths = np.asarray(lm["width"], float)
    picks = {}
    inl = np.nonzero(cat == "inlier")[0]
    if inl.size:
        order = inl[np.argsort(-peaks[inl])]
        picks["inlier_a"] = int(order[min(3, len(order) - 1)])
        picks["inlier_b"] = int(order[len(order) // 2])
    for st in ("peak_low", "peak_wide", "peak_shape"):
        idx = np.nonzero(status == st)[0]
        if idx.size:
            if st == "peak_low":
                idx = idx[np.argsort(-peaks[idx])]
                picks[st] = int(idx[len(idx) // 3])
            elif st == "peak_wide":
                idx = idx[np.argsort(widths[idx])]
                picks[st] = int(idx[len(idx) // 2])
            else:
                picks[st] = int(idx[0])
    for c in ("geometric", "mahalanobis"):
        idx = np.nonzero(cat == c)[0]
        if idx.size:
            picks[c] = int(idx[np.argsort(-peaks[idx])][0])
    return picks


def fig_fine_landmark(ex, fi, i, out):
    d = fi["feats"][i]
    lm = ex["fines"][0].landmarks
    cfg = ex["cfg"].fine
    pk = d["pk"]
    img = ex["img"]
    fc, fr = int(lm["feat_col"][i]), int(lm["feat_row"][i])
    fig = plt.figure(figsize=(13.6, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.8, 1.15, 1.15, 1.1], wspace=0.18)
    ax = fig.add_subplot(gs[0])
    hw = 40
    sub = img[max(fr - hw, 0):fr + hw, max(fc - hw, 0):fc + hw]
    gray(ax, sub, "Image around feature")
    t_img = cfg.template_px * fi["map_to_img"]
    ax.add_patch(Rectangle((hw - t_img / 2, hw - t_img / 2), t_img, t_img, fill=False, ec=YELLOW, lw=1.6))
    ax.scatter([hw], [hw], marker="+", s=60, color=YELLOW)
    ax = fig.add_subplot(gs[1])
    gray(ax, np.clip(d["tpl"], -3, 3), "21×21 template")
    ax = fig.add_subplot(gs[2])
    gray(ax, stretch(d["search_raw"], 0.5, 99.5), "128×128 map search")
    ht = cfg.template_px // 2
    ax.add_patch(Rectangle((64 - ht - 0.5, 64 - ht - 0.5), 21, 21, fill=False, ec=ORANGE, lw=1.4, ls="--"))
    ax.add_patch(Rectangle((pk.x - 0.5, pk.y - 0.5), 21, 21, fill=False, ec=AQUA, lw=1.6))
    ax = fig.add_subplot(gs[3])
    ax.imshow(d["surf"], cmap="magma", vmin=-0.3, vmax=1.0, interpolation="nearest")
    noaxis(ax)
    ax.scatter([pk.x], [pk.y], marker="x", s=50, color=YELLOW, linewidths=1.6)
    ax.add_patch(Rectangle((pk.ix - 1.5, pk.iy - 1.5), 3, 3, fill=False, ec="white", lw=1.0))
    ax.set_title(f"NCC surface, peak {pk.height:.2f}")
    # 3x3 biquadratic fit, upsampled.
    ax = fig.add_subplot(gs[4])
    z3 = d["surf"][pk.iy - 1:pk.iy + 2, pk.ix - 1:pk.ix + 2].astype(float)
    dx, dy, val, Hs, concave = biquadratic_fit(z3)
    ys, xs = np.mgrid[-1:2, -1:2]
    A = np.column_stack([np.ones(9), xs.ravel(), ys.ravel(), xs.ravel() ** 2, (xs * ys).ravel(), ys.ravel() ** 2])
    coef, *_ = np.linalg.lstsq(A, z3.ravel(), rcond=None)
    gy, gx = np.mgrid[-1.5:1.5:121j, -1.5:1.5:121j]
    Z = coef[0] + coef[1] * gx + coef[2] * gy + coef[3] * gx**2 + coef[4] * gx * gy + coef[5] * gy**2
    ax.imshow(z3, cmap="magma", vmin=z3.min() - 0.05, vmax=max(val, z3.max()), extent=[-1.5, 1.5, 1.5, -1.5],
              interpolation="nearest")
    cs = ax.contour(gx, gy, Z, levels=8, colors="white", linewidths=0.6)
    ax.clabel(cs, fontsize=5.5, fmt="%.2f")
    ax.scatter([dx], [dy], marker="x", s=70, color=AQUA, linewidths=2.0)
    for (yy, xx), zz in np.ndenumerate(z3):
        ax.text(xx - 1, yy - 1 + 0.32, f"{zz:.2f}", ha="center", color="white", fontsize=6.5)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=7)
    ax.set_title(f"3×3 fit → ({dx:+.2f}, {dy:+.2f}) px")
    save(fig, out, "f16_fine_landmark")
    return dict(feature=int(i), peak=float(pk.height), fwhm=float(pk.width), fwhm_min=float(pk.width_min),
                psr=float(pk.psr), second=float(pk.second_ratio), subpx=[float(dx), float(dy)],
                fit_height=float(val))


def fig_fine_gallery(ex, fi, picks, out, name="f17_fine_gallery"):
    order = [("inlier_a", "inlier"), ("inlier_b", "inlier"), ("peak_low", "rejected: peak low"),
             ("peak_wide", "rejected: peak wide"), ("peak_shape", "rejected: shape / sub-px > 1"),
             ("geometric", "rejected: RANSAC geometry")]
    order = [(k, t) for k, t in order if k in picks and picks[k] in fi["feats"]]
    n = len(order)
    fig, axs = plt.subplots(2, n, figsize=(2.25 * n, 4.9), gridspec_kw=dict(height_ratios=[0.6, 1], wspace=0.1,
                                                                             hspace=0.28))
    for j, (key, title) in enumerate(order):
        d = fi["feats"][picks[key]]
        pk = d["pk"]
        gray(axs[0, j], np.clip(d["tpl"], -3, 3))
        axs[0, j].set_title(title, fontsize=8.5, color=INK if key.startswith("inlier") else RED)
        axs[1, j].imshow(d["surf"], cmap="magma", vmin=-0.3, vmax=1.0, interpolation="nearest")
        axs[1, j].scatter([pk.x], [pk.y], marker="x", s=36, color=YELLOW, linewidths=1.4)
        noaxis(axs[1, j])
        w = "∞" if not np.isfinite(pk.width) else f"{pk.width:.1f}"
        axs[1, j].set_title(f"peak {pk.height:.2f} · FWHM {w}", fontsize=8)
    save(fig, out, name)


def load_calibration():
    """Peak height / FWHM of calibration-split landmarks, split by GT correctness (< 1.5 px)."""
    d = ROOT / "outputs/calib/nadir/landmarks"
    hp, wp, ok = [], [], []
    for p in sorted(d.glob("*.npz")):
        with np.load(p) as z:
            keys = sorted({k.split("__")[0] for k in z.files})
            for k in keys:
                e = z[f"{k}__match_err_px"]
                h = z[f"{k}__peak"]
                w = z[f"{k}__width"]
                m = np.isfinite(e) & np.isfinite(h)
                hp.append(h[m])
                wp.append(w[m])
                ok.append(e[m] < 1.5)
    return np.concatenate(hp), np.concatenate(wp), np.concatenate(ok)


def fig_peak_gate(ex, cats, out, title="150 features, iteration 1"):
    lm = ex["fines"][0].landmarks
    peaks = np.asarray(lm["peak"], float)
    widths = np.asarray(lm["width"], float)
    status = cats["status"]
    cfg = ex["cfg"].fine
    hp, wp, ok = load_calibration()
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.1), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]
    wcl = np.where(np.isfinite(widths), np.minimum(widths, 39), 39)
    for st, col, mk, lab in (("", AQUA, "o", "passed"), ("peak_low", ORANGE, "v", "peak low"),
                             ("peak_wide", VIOLET, "s", "peak wide"), ("peak_shape", NEUTRAL, "x", "shape / sub-px")):
        m = status == st
        ax.scatter(peaks[m], wcl[m], s=26, marker=mk, color=col, label=f"{lab} ({m.sum()})",
                   edgecolors="white" if mk not in ("x",) else None, linewidths=0.5)
    ax.axvline(cfg.min_peak_height, color=INK, lw=1, ls="--")
    ax.axhline(cfg.max_peak_width_px, color=INK, lw=1, ls="--")
    ax.fill_between([cfg.min_peak_height, 1.0], 0, cfg.max_peak_width_px, color=AQUA, alpha=0.08, lw=0)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 40)
    ax.set_xlabel("NCC peak height")
    ax.set_ylabel("FWHM (fine-map px, ∞ clipped to 39)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7.2)
    ax.grid(True, color=GRID, lw=0.6)
    ax = axs[1]
    bins = np.linspace(-0.1, 1.0, 56)
    ax.hist(hp[ok], bins=bins, density=True, color=BLUE, alpha=0.55, label=f"correct ({ok.sum():,})")
    ax.hist(hp[~ok], bins=bins, density=True, color=ORANGE, alpha=0.55, label=f"wrong ({(~ok).sum():,})")
    ax.axvline(cfg.min_peak_height, color=INK, lw=1, ls="--")
    ax.text(cfg.min_peak_height + 0.01, ax.get_ylim()[1] * 0.92, "gate 0.60", fontsize=8, color=INK)
    ax.set_xlabel("NCC peak height")
    ax.set_ylabel("density")
    ax.set_title("Calibration split: peak height")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axs[2]
    wb = np.where(np.isfinite(wp), np.minimum(wp, 39.5), 39.5)
    bins = np.linspace(0, 40, 61)
    ax.hist(wb[ok], bins=bins, density=True, color=BLUE, alpha=0.55, label="correct")
    ax.hist(wb[~ok], bins=bins, density=True, color=ORANGE, alpha=0.55, label="wrong")
    ax.axvline(cfg.max_peak_width_px, color=INK, lw=1, ls="--")
    ax.text(cfg.max_peak_width_px + 0.5, ax.get_ylim()[1] * 0.92, "gate 12 px", fontsize=8, color=INK)
    ax.set_xlabel("FWHM (fine-map px, ∞ in last bin)")
    ax.set_title("Calibration split: peak width")
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, out, "f18_peak_gate")
    g = (hp >= cfg.min_peak_height) & (wp <= cfg.max_peak_width_px)
    return dict(calib_n=int(hp.size), calib_correct_frac=float(ok.mean()),
                calib_correct_pass=float(g[ok].mean()), calib_wrong_pass=float(g[~ok].mean()),
                calib_precision=float(ok[g].mean()))


def fig_fine_offsets(ex, fi, cats, out, name="f19_fine_offsets", sc=40.0):
    """Match offsets on the warped image (map frame), iteration 1."""
    lm = ex["fines"][0].landmarks
    fmap, cam = ex["ms"].fine, ex["cam"]
    fc = np.asarray(lm["feat_col"], float)
    fr = np.asarray(lm["feat_row"], float)
    mc, mr = predicted_map_location(cam, ex["Ts"][0], fmap, fc, fr)
    b_lo_c, b_lo_r = fi["box"][:2]
    x, y = mc - b_lo_c, mr - b_lo_r
    odx = np.asarray(lm["offset_dx"], float)
    ody = np.asarray(lm["offset_dy"], float)
    cat = cats["cat"]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.16))
    ax = axs[0]
    ax.imshow(stretch(fi["rect"], 0.5, 99.5), cmap="gray", alpha=0.75)
    noaxis(ax)
    m = ~np.isin(cat, list(CAT_STYLE))
    ax.scatter(x[m], y[m], s=10, color=NEUTRAL, label="rejected by peak test / unused", zorder=3)
    for c, stl in CAT_STYLE.items():
        m = cat == c
        if not m.any():
            continue
        ax.quiver(x[m], y[m], odx[m] * sc, ody[m] * sc, angles="xy", scale_units="xy", scale=1, color=stl["color"],
                  width=0.004, headwidth=3.5, zorder=4)
        ax.scatter(x[m], y[m], s=18, marker=stl["marker"], color=stl["color"], edgecolors="white", linewidths=0.5,
                   label=f"{stl['label']} ({m.sum()})", zorder=5)
    ax.set_title(f"Match offsets ×{sc:.0f} on the image warped after coarse")
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.85, frameon=True)
    ax = axs[1]
    for c, stl in CAT_STYLE.items():
        m = cat == c
        if m.any():
            ax.scatter(odx[m], -ody[m], s=26, marker=stl["marker"], color=stl["color"], edgecolors="white",
                       linewidths=0.5, label=stl["label"])
    e = enu_of(ex["T"], ex["Ts"][0][:3, 3])
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_aspect("equal")
    used = np.isin(cat, list(CAT_STYLE))
    lim = max(4, min(30, np.nanpercentile(np.abs(np.r_[odx[used], ody[used]]), 95) * 1.3))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("offset map-x (fine px)")
    ax.set_ylabel("offset map-y (fine px)")
    ax.set_title(f"Offsets cluster: coarse error {math.hypot(e[0], e[1]):.0f} m ≈ "
                 f"{math.hypot(e[0], e[1]) / fmap.pixel:.1f} px")
    save(fig, out, name)


def fig_mahalanobis(ex, cats_all, out):
    fig, axs = plt.subplots(1, len(cats_all), figsize=(6.0 * len(cats_all), 3.8), gridspec_kw=dict(wspace=0.22))
    axs = np.atleast_1d(axs)
    res = []
    for it, (ax, cats) in enumerate(zip(axs, cats_all)):
        d2 = cats["d2"]
        cat = cats["cat"]
        g = cats["gamma"]
        m_in = cat == "inlier"
        m_mh = cat == "mahalanobis"
        allv = d2[np.isfinite(d2)]
        bins = np.logspace(np.log10(max(allv.min(), 1e-4)), np.log10(max(allv.max(), g) * 1.3), 40)
        ax.hist(d2[m_in], bins=bins, color=AQUA, alpha=0.85, label=f"kept ({m_in.sum()})")
        ax.hist(d2[m_mh], bins=bins, color=VIOLET, alpha=0.9, label=f"rejected ({m_mh.sum()})")
        q1, q3 = np.percentile(allv, [25, 75])
        for q, lab in ((q1, "q1"), (q3, "q3")):
            ax.axvline(q, color=INK2, lw=0.8, ls=":")
            ax.text(q, ax.get_ylim()[1] * 0.95, lab, fontsize=7.5, color=INK2, ha="center")
        ax.axvline(g, color=RED, lw=1.4, ls="--")
        ax.text(g * 1.08, ax.get_ylim()[1] * 0.8, f"γ = q3 + 1.5(q3 − q1)\n= {g:.2f}", fontsize=8, color=RED)
        ax.set_xscale("log")
        ax.set_xlabel("d² = |r / σ|²   (σ = 0.5 map px in image px)")
        ax.set_ylabel("landmarks")
        ax.set_title(f"Iteration {it + 1}: RANSAC inliers ({np.isfinite(d2).sum()})")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.grid(True, color=GRID, lw=0.6, axis="y")
        res.append(dict(gamma=float(g), n_ransac_inliers=int(np.isfinite(d2).sum()), n_mahal=int(m_mh.sum())))
    save(fig, out, "f20_mahalanobis")
    return res


def fig_residuals(ex, out):
    fr_ = ex["fines"][-1]
    lm = fr_.landmarks
    inl = np.asarray(lm["inlier"], bool)
    res = lm["residual"][inl]
    u = np.asarray(lm["u"], float)[inl] - ex["cam"].pixel_center
    v = np.asarray(lm["v"], float)[inl] - ex["cam"].pixel_center
    sig = fr_.health["sigma_px_assumed"]
    sc = 400.0
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 5.4), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.18))
    ax = axs[0]
    ax.imshow(ex["img"], cmap="gray", alpha=0.7)
    noaxis(ax)
    ax.quiver(u, v, res[:, 0] * sc, res[:, 1] * sc, angles="xy", scale_units="xy", scale=1, color=AQUA,
              width=0.004, headwidth=3.5)
    ax.scatter(u, v, s=12, color=AQUA, edgecolors="white", linewidths=0.4)
    ax.set_title(f"Final reprojection residuals ×{sc:.0f} ({inl.sum()} inliers)")
    ax = axs[1]
    ax.scatter(res[:, 0], res[:, 1], s=26, color=AQUA, edgecolors="white", linewidths=0.5)
    for k, ls in ((1, "--"), (2, ":")):
        ax.add_patch(Circle((0, 0), k * sig, fill=False, ec=INK2, lw=1, ls=ls))
        ax.text(k * sig * 0.72, k * sig * 0.72, f"{k}σ", fontsize=8, color=INK2)
    lim = max(2.3 * sig, np.abs(res).max() * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("residual u (px)")
    ax.set_ylabel("residual v (px)")
    ax.set_title(f"RMS {fr_.reproj_rms_px:.3f} px · σ₀ {fr_.sigma0:.2f} (assumed σ {sig:.2f} px)")
    save(fig, out, "f21_residuals")


def cov_ellipse(ax, C2, n_sigma, **kw):
    w, V = np.linalg.eigh(C2)
    ang = math.degrees(math.atan2(V[1, 1], V[0, 1]))
    e = Ellipse((0, 0), 2 * n_sigma * math.sqrt(w[1]), 2 * n_sigma * math.sqrt(w[0]), angle=ang, fill=False, **kw)
    ax.add_patch(e)
    return e


def fig_pose(ex, out):
    T = ex["T"]
    pts = {"prior": enu_of(T, ex["Tp"][:3, 3]), "coarse": enu_of(T, ex["Ts"][0][:3, 3])}
    for k, Tq in enumerate(ex["Ts"][1:]):
        pts[f"fine{k + 1}"] = enu_of(T, Tq[:3, 3])
    fin = pts[f"fine{len(ex['Ts']) - 1}"]
    enu = frames.enu_basis(T[:3, 3])
    C_lm = enu.T @ np.asarray(ex["res"].health["cov_landmark_only"]) @ enu
    C_fl = enu.T @ ex["res"].cov[:3, :3] @ enu
    fpx = ex["ms"].fine.pixel
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.6), gridspec_kw=dict(wspace=0.32))
    sty = {"prior": (NEUTRAL, "o", "prior"), "coarse": (ORANGE, "s", "after coarse"),
           "fine1": (BLUE, "D", "fine iteration 1"), "fine2": (AQUA, "o", "fine iteration 2 (final)")}

    def draw(ax, keys, lim, center=(0, 0)):
        for k in keys:
            c, mk, lab = sty[k]
            p = pts[k]
            ax.scatter([p[0]], [p[1]], marker=mk, s=70, color=c, edgecolors="white", linewidths=0.8, zorder=5,
                       label=lab)
        ax.scatter([0], [0], marker="*", s=200, color=INK, zorder=6, label="ground truth")
        ax.set_xlim(center[0] - lim, center[0] + lim)
        ax.set_ylim(center[1] - lim, center[1] + lim)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlabel("east error (m)")
        ax.set_ylabel("north error (m)")

    ax = axs[0]
    ax.add_patch(Circle((0, 0), 3000, fill=True, fc="#eef0f2", ec=NEUTRAL, lw=1, ls="--", zorder=0))
    ax.annotate("", xy=pts["coarse"][:2], xytext=pts["prior"][:2],
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    draw(ax, ["prior", "coarse", "fine2"], 3400)
    ax.set_title("km scale: 3 km prior disc")
    ax.legend(loc="lower left", fontsize=7)
    ax = axs[1]
    L = max(abs(pts["coarse"][0]), abs(pts["coarse"][1])) * 1.6 + 5
    ax.annotate("", xy=pts["fine1"][:2], xytext=pts["coarse"][:2], arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    draw(ax, ["coarse", "fine1", "fine2"], L)
    ax.set_title(f"±{L:.0f} m: coarse → fine")
    ax = axs[2]
    L = max(4 * math.sqrt(np.max(np.linalg.eigvalsh(C_fl[:2, :2]))), abs(fin[0]) * 1.4, abs(fin[1]) * 1.4)
    # Fine map pixel grid.
    for g in np.arange(-math.ceil(L / fpx) * fpx, L + fpx, fpx):
        ax.axvline(g, color=GRID, lw=0.9, zorder=0)
        ax.axhline(g, color=GRID, lw=0.9, zorder=0)
    ax.set_xlim(fin[0] - L, fin[0] + L)
    ax.set_ylim(fin[1] - L, fin[1] + L)
    for ns, ls in ((1, "-"), (2, "--")):
        e1 = cov_ellipse(ax, C_lm[:2, :2], ns, ec=VIOLET, lw=1.3, ls=ls)
        e1.set_center(fin[:2])
        e2 = cov_ellipse(ax, C_fl[:2, :2], ns, ec=AQUA, lw=1.6, ls=ls)
        e2.set_center(fin[:2])
    ax.scatter([fin[0]], [fin[1]], marker="o", s=70, color=AQUA, edgecolors="white", zorder=5)
    ax.scatter([0], [0], marker="*", s=200, color=INK, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("east error (m)")
    ax.set_ylabel("north error (m)")
    ax.set_title(f"Final: 1σ/2σ, grid = fine px ({fpx:.1f} m)")
    ax.plot([], [], color=VIOLET, label="landmark noise only")
    ax.plot([], [], color=AQUA, label="+ common-mode floor")
    ax.legend(loc="lower left", fontsize=7)
    save(fig, out, "f22_pose")
    sl_ = lambda C: [float(math.sqrt(C[0, 0])), float(math.sqrt(C[1, 1])), float(math.sqrt(C[2, 2]))]  # noqa: E731
    return dict(points={k: [float(x) for x in v] for k, v in pts.items()}, sigma_lm_enu=sl_(C_lm),
                sigma_floor_enu=sl_(C_fl))


# --------------------------------------------------------------------------------------
# figures: illumination, parallax and shadow
# --------------------------------------------------------------------------------------
def fine_counts(ex):
    """Funnel numbers of the final fine iteration plus the coarse summary."""
    h = ex["res"].health
    out = {"status": ex["res"].status, "verdict": ex["score"]["verdict"], "err_h_m": ex["score"]["err_h_m"],
           "coarse_ok": bool(h["coarse"]["ok"]), "coarse_templates": int(h["coarse"].get("n_templates", 0)),
           "coarse_peak_ok": int(h["coarse"].get("n_peak_ok", 0)),
           "coarse_inliers": int(h["coarse"].get("n_geom_inliers", 0) or 0)}
    if ex["fines"]:
        f = ex["fines"][-1]
        rc = dict(f.reject_counts)
        out.update(n_features=f.n_features, n_matched=f.n_matched, n_used=f.n_used, n_inliers=f.n_inliers,
                   reject_counts=rc, rms_px=float(f.reproj_rms_px), sigma0=float(f.sigma0),
                   peak_median=float(np.nanmedian(np.asarray(f.landmarks["peak"], float))))
    return out


def fig_v3_example(ex0, exs, out):
    """Same image and prior, maps rendered with the Sun azimuth rotated by delta."""
    cam, T = ex0["cam"], ex0["T"]
    fmap0 = ex0["ms"].fine
    c, r = nadir_colrow(T, fmap0)
    s = 200
    c0, r0 = int(c - s / 2), int(r - s / 2)
    rect = rectify_to_map(ex0["img"], cam, T, fmap0, c0, r0, s, s, supersample=1)
    # Template for the comparison: centre of the patch, warped with the GT pose.
    big = rectify_to_map(ex0["img"], cam, T, fmap0, c0 - 40, r0 - 40, s + 80, s + 80, supersample=1)
    big_n, _ = local_normalize(big, 21)
    tc, tr = s // 2 + 40, s // 2 + 40
    tpl = big_n[tr - 10:tr + 11, tc - 10:tc + 11]
    n = len(exs)
    fig, axs = plt.subplots(2, n + 1, figsize=(2.55 * (n + 1), 5.6), gridspec_kw=dict(wspace=0.06, hspace=0.32))
    gray(axs[0, 0], stretch(rect, 0.5, 99.5), "Image (GT warp)")
    axs[0, 0].add_patch(Rectangle((s // 2 - 10.5, s // 2 - 10.5), 21, 21, fill=False, ec=YELLOW, lw=1.4))
    gray(axs[1, 0], np.clip(tpl, -3, 3), "21×21 template")
    stats = []
    for j, exd in enumerate(exs):
        fm = exd["ms"].fine
        sun_angle = math.degrees(math.acos(np.clip(exd["ms"].fine.sun_dir @ ex0["sun"], -1, 1)))
        fc = fine_counts(exd)
        ax = axs[0, j + 1]
        gray(ax, stretch(fm.image[r0:r0 + s, c0:c0 + s], 0.5, 99.5))
        verdict = fc["verdict"].replace("_", " ")
        inl = fc.get("n_inliers", 0) if fc["coarse_ok"] else 0
        ax.set_title(f"Δaz {exd['delta_az']:.0f}° ({sun_angle:.0f}°)\n{verdict}, {inl} inliers", fontsize=8.5,
                     color=INK if fc["verdict"] == "correct" else RED)
        v, _ = sun_vec_map(fm.sun_dir, fm)
        sun_arrow(ax, v, xy=(0.8, 0.84), L=0.13, label="")
        mn = exd["ms"].normalized("fine", 21)
        cc, rr = c0 + s // 2, r0 + s // 2
        search = mn[rr - 64:rr + 64, cc - 64:cc + 64]
        surf = correlate(search, tpl)
        pk = analyze_peak(surf, exclusion=3)
        ax = axs[1, j + 1]
        ax.imshow(surf, cmap="magma", vmin=-0.3, vmax=1.0, interpolation="nearest")
        ax.scatter([pk.x], [pk.y], marker="x", s=36, color=YELLOW, linewidths=1.4)
        ax.scatter([53.5], [53.5], marker="o", s=60, facecolors="none", edgecolors=AQUA, linewidths=1.3)
        noaxis(ax)
        ax.set_title(f"peak {pk.height:.2f}", fontsize=8.5)
        stats.append(dict(delta_az=float(exd["delta_az"]), sun_angle_deg=float(sun_angle), **fc,
                          demo_peak=float(pk.height)))
    save(fig, out, "f23_v3_example")
    return stats


def distortion_field(ex, step=4):
    """Warp error (coarse px) inside template-sized windows when warping with the prior pose."""
    import cv2

    cam, Tp, T, cmap = ex["cam"], ex["Tp"], ex["T"], ex["ms"].coarse
    h_ref = float(np.nanmedian(cmap.heights))
    c_lo, r_lo, c_hi, r_hi = _footprint_box(cam, Tp, cmap, h_ref)
    cs = np.arange(c_lo, c_hi, step, dtype=float)
    rs = np.arange(r_lo, r_hi, step, dtype=float)
    C, R = np.meshgrid(cs, rs)
    P = cmap.world_points(C.ravel(), R.ravel())
    u, v, z = cam.project(Tp, P)
    ok = (z > 0) & (u >= 0) & (v >= 0) & (u <= cam.width - 1) & (v <= cam.height - 1)
    mct, mrt = predicted_map_location(cam, T, cmap, u, v)
    dx = np.where(ok, mct - C.ravel(), np.nan).reshape(C.shape)
    dy = np.where(ok, mrt - R.ravel(), np.nan).reshape(C.shape)
    # Warp error inside a template-sized window: displacement minus its local mean.
    k = max(int(round(128 / step)), 3)
    valid = np.isfinite(dx).astype(np.float32)

    def local_mean(a):
        num = cv2.boxFilter(np.nan_to_num(a).astype(np.float32), -1, (k, k), normalize=False,
                            borderType=cv2.BORDER_CONSTANT)
        den = cv2.boxFilter(valid, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        return num / np.maximum(den, 1e-6)

    mag = np.hypot(dx - local_mean(dx), dy - local_mean(dy))
    mag[valid == 0] = np.nan
    hts = cmap.height_at_colrow(C, R)
    return dict(mag=mag, step=step, relief=float(np.nanpercentile(hts, 98) - np.nanpercentile(hts, 2)),
                p95=float(np.nanpercentile(mag, 95)), med=float(np.nanmedian(mag)))


def fig_parallax(ex_main, ex_small, ex_big, out):
    cases = [(ex_main, "9.5 km"), (ex_small, "3.5 km"), (ex_big, "3.5 km")]
    fields = [distortion_field(e) for e, _ in cases]
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 8.6), gridspec_kw=dict(wspace=0.08, hspace=0.3))
    vmax = 8
    im = None
    res = []
    for j, ((e, lab), fd) in enumerate(zip(cases, fields)):
        ax = axs[0, j]
        im = ax.imshow(fd["mag"], cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        noaxis(ax)
        L = 128 / fd["step"]
        ax.add_patch(Rectangle((2, 2), L, L, fill=False, ec=ORANGE, lw=1.6))
        ax.text(3, L + 6, "128 px template", color=ORANGE, fontsize=7.5, va="top")
        ax.set_title(f"{lab}, prior off {e['perr']['prior_err_h_m'] / 1000:.1f} km: "
                     f"p95 {fd['p95']:.1f} px", fontsize=9)
        fc = fine_counts(e)
        res.append(dict(label=lab, prior_err_m=float(e["perr"]["prior_err_h_m"]), relief_m=fd["relief"],
                        p95_px=fd["p95"], median_px=fd["med"], map_px=float(e["ms"].coarse.pixel), **fc))
    cb = fig.colorbar(im, ax=axs[0, :], fraction=0.02, pad=0.01)
    cb.set_label("warp error within a 128 px window (coarse px)")
    cb.outline.set_visible(False)
    for j, (e, lab) in enumerate(cases):
        ci = coarse_internals(e)
        t = max(ci["tpls"], key=lambda q: q["pk"].height) if ci["tpls"] else None
        ax = axs[1, j]
        if t is None:
            ax.axis("off")
            continue
        ax.imshow(t["surf"], cmap="magma", vmin=-0.2, vmax=1.0, interpolation="nearest")
        sc0, sr0 = t["search_box"][:2]
        zx, zy = t["mc0"] - sc0, t["mr0"] - sr0
        ax.add_patch(Circle((zx + t["true_offset"][0], zy + t["true_offset"][1]), 7, fill=False, ec=AQUA, lw=1.6))
        ax.scatter([t["pk"].x], [t["pk"].y], marker="x", s=50, color=YELLOW, linewidths=1.6)
        noaxis(ax)
        fc = fine_counts(e)
        ax.set_title(f"Best coarse NCC surface: peak {t['pk'].height:.2f}\n"
                     f"coarse {'ok' if fc['coarse_ok'] else 'failed'}: {fc['coarse_peak_ok']}/{fc['coarse_templates']}"
                     f" peaks pass, {fc['coarse_inliers']} geometric inliers", fontsize=8.5,
                     color=INK if fc["coarse_ok"] else RED)
    save(fig, out, "f24_parallax")
    return res


def fig_shadow_case(out, geom=16320):
    g = load_group(geom)
    runs = [json.loads(line) for line in (ROOT / "outputs/v2/nadir/runs.jsonl").open()
            if f'"geometry_id": {geom},' in line]
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 4.1), gridspec_kw=dict(wspace=0.05))
    names = ["Side", "Overhead", "Back"]
    res = []
    for k in range(3):
        img = g[k].image()
        dark = float((img < 0.02).mean())
        rr = [r for r in runs if r["map_illum"] == k and r["image_illum"] == k]
        av = np.mean([r["verdict"] == "correct" for r in rr]) if rr else float("nan")
        rgb = np.dstack([img] * 3)
        rgb[img < 0.02] = [0.16, 0.47, 0.84]
        axs[k].imshow(rgb)
        noaxis(axs[k])
        axs[k].set_title(f"{names[k]}: dark {dark * 100:.0f}% · {int(round(av * len(rr)))}/{len(rr)} correct",
                         fontsize=9, color=INK if av > 0.5 else RED)
        res.append(dict(illum=k, dark_frac=dark, availability=float(av), n=len(rr)))
    save(fig, out, "f25_shadow_case")
    return dict(geom=geom, alt_m=float(g[0].altitude_above_sphere()), per_illum=res)


# --------------------------------------------------------------------------------------
# aggregate numbers for the HTML charts
# --------------------------------------------------------------------------------------
def _runs(path: Path):
    return [json.loads(line) for line in (path / "runs.jsonl").open() if '"verdict"' in line]


def aggregate_stats():
    v2 = _runs(ROOT / "outputs/v2/nadir")
    diag = [r for r in v2 if r["map_illum"] == r["image_illum"]]
    bands = []
    for b in range(10):
        rr = [r for r in diag if r["band"] == b]
        cor = [r for r in rr if r["verdict"] == "correct"]
        coarse_ok = [r for r in rr if r["health"].get("coarse", {}).get("ok")]
        ce = [r["coarse_err_h_m"] for r in coarse_ok if "coarse_err_h_m" in r]
        fe = np.array([r["err_h_m"] for r in cor])
        bands.append(dict(band_km=BANDS_KM[b], n=len(rr), availability=len(cor) / len(rr),
                          coarse_success=len(coarse_ok) / len(rr), false_fix=sum(r["verdict"] == "false_fix" for r in rr),
                          coarse_p99_m=float(np.percentile(ce, 99)), fine_p50_m=float(np.percentile(fe, 50)),
                          fine_p95_m=float(np.percentile(fe, 95)), map_px=float(np.median([r["map_px_fine"] for r in rr])),
                          fine_p95_px=float(np.percentile(fe / np.array([r["map_px_fine"] for r in cor]), 95)),
                          inliers_median=float(np.median([r["health"]["fine"]["n_inliers"] for r in cor])),
                          t_run_median_s=float(np.median([r["t_run_s"] for r in rr]))))
    # Illumination matrix (all bands).
    mat = [[0.0] * 3 for _ in range(3)]
    ang = [[0.0] * 3 for _ in range(3)]
    for km in range(3):
        for ki in range(3):
            rr = [r for r in v2 if r["map_illum"] == km and r["image_illum"] == ki]
            mat[km][ki] = float(np.mean([r["verdict"] == "correct" for r in rr]))
            ang[km][ki] = float(np.median([r["sun_angle_map_image_deg"] for r in rr]))
    # Sun azimuth sweep.
    sweep = _runs(ROOT / "outputs/v3/nadir_sweep") + _runs(ROOT / "outputs/v3/nadir_sweep_wide")
    geoms = {r["geometry_id"] for r in sweep}
    base = [r for r in diag if r["geometry_id"] in geoms]
    groups = {"3.5 km": [0], "6.5–9.5 km": [1, 2], "12.5–30.5 km": list(range(3, 10))}
    deltas = sorted({0.0} | {float(r["map_sun_delta_az"]) for r in sweep})
    sw = {"deltas": deltas, "series": {}, "fine_p95_px": [], "sun_angle": [], "n_false_fix": 0}
    for name, bs in groups.items():
        ys = []
        for d in deltas:
            rr = base if d == 0 else [r for r in sweep if float(r["map_sun_delta_az"]) == d]
            rr = [r for r in rr if r["band"] in bs]
            ys.append(float(np.mean([r["verdict"] == "correct" for r in rr])) if rr else None)
        sw["series"][name] = ys
    for d in deltas:
        rr = base if d == 0 else [r for r in sweep if float(r["map_sun_delta_az"]) == d]
        cor = [r["err_h_m"] / r["map_px_fine"] for r in rr if r["verdict"] == "correct"]
        sw["fine_p95_px"].append(float(np.percentile(cor, 95)) if len(cor) >= 5 else None)
        sw["sun_angle"].append(float(np.median([r["sun_angle_map_image_deg"] for r in rr])))
    sw["n_runs"] = len(sweep)
    sw["n_false_fix"] = sum(r["verdict"] == "false_fix" for r in sweep)
    # Availability vs prior error and vs dark fraction (matched illumination).
    edges = [0, 500, 1000, 1500, 2000, 2500, 3000]
    pe = {"edges_km": [e / 1000 for e in edges], "series": {}}
    for name, bs in (("3.5 km", [0]), ("6.5 km", [1]), ("9.5 km", [2]), ("≥ 12.5 km", list(range(3, 10)))):
        ys = []
        for a, b in zip(edges[:-1], edges[1:]):
            rr = [r for r in diag if r["band"] in bs and a <= r["prior_err_h_m"] < b]
            ys.append(float(np.mean([r["verdict"] == "correct" for r in rr])) if rr else None)
        pe["series"][name] = ys
    dedges = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    dk = {"edges": dedges, "series": {}}
    for name, bs in (("3.5–6.5 km", [0, 1]), ("9.5–30.5 km", list(range(2, 10)))):
        ys, ns = [], []
        for a, b in zip(dedges[:-1], dedges[1:]):
            rr = [r for r in diag if r["band"] in bs and a <= r["health"]["scene"]["dark_fraction_002"] < b]
            ys.append(float(np.mean([r["verdict"] == "correct" for r in rr])) if len(rr) >= 10 else None)
            ns.append(len(rr))
        dk["series"][name] = ys
        dk.setdefault("n", {})[name] = ns
    n_all = len(v2)
    return dict(bands=bands, illum_matrix=mat, illum_angle=ang, sweep=sw, prior_err=pe, dark=dk,
                v2_runs=n_all, v2_false_fix=sum(r["verdict"] == "false_fix" for r in v2), v2_diag_runs=len(diag))


def find_seed(T, lo, hi, start=0):
    for s in range(start, start + 500):
        _, _, perr = sample_prior(T, np.random.default_rng(s), 3000.0, 120.0, 0.0)
        if lo <= perr["prior_err_h_m"] <= hi:
            return s
    raise RuntimeError("no seed found")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/report")
    ap.add_argument("--geom", type=int, default=2820)
    ap.add_argument("--illum", type=int, default=0)
    ap.add_argument("--prior-seed", type=int, default=7)
    ap.add_argument("--hard-delta-az", type=float, default=45.0)
    args = ap.parse_args(argv)
    out = args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    src = DemSource(DEM_PATH)
    stats: dict = {}
    t0 = time.time()

    ex = run_example(src, args.geom, args.illum, args.prior_seed, cache)
    h = ex["res"].health
    stats["example"] = dict(
        geometry=args.geom, illum=args.illum, prior_seed=args.prior_seed, alt_m=ex["alt"], band=ex["band"],
        band_km=BANDS_KM[ex["band"]], gsd_nominal_m=ex["gsd_nom"], map_px_fine=ex["ms"].fine.pixel,
        map_px_coarse=ex["ms"].coarse.pixel, map_shape_fine=list(ex["ms"].fine.shape),
        map_shape_coarse=list(ex["ms"].coarse.shape), prior=ex["perr"], prior_sigma_h_m=ex["sigma_h"],
        score=ex["score"], status=ex["res"].status, replica_identical=ex["replica_ok"], health=h,
        sun_azel=sl.sun_azimuth_elevation(ex["sun"], ex["frame"].boresight_point()),
        coarse_err_h_m=float(math.hypot(*enu_of(ex["T"], ex["Ts"][0][:3, 3])[:2])),
        fine_iterations=[fine_counts(dict(ex, fines=[f])) for f in ex["fines"]])
    print("figures: inputs and maps", flush=True)
    fig_inputs(ex, out)
    stats["overview"] = fig_overview(ex, src, out)
    stats["v1"] = fig_v1_render(ex, src, out)
    stats["mapgen"] = fig_mapgen(src, out)
    stats["map_vs_image"] = fig_map_vs_image(ex, src, out)
    print("figures: coarse", flush=True)
    ci = coarse_internals(ex)
    fig_prior(ex, ci, out)
    fig_coarse_rectify(ex, ci, out)
    fig_normalize(ex, ci, out)
    fig_templates(ex, ci, out)
    fig_coarse_ncc(ex, ci, out)
    fig_coarse_detail(ex, ci, out, int(np.argmax([t["pk"].height for t in ci["tpls"]])))
    fig_coarse_ransac(ex, ci, out)
    stats["coarse"] = dict(box=ci["box"], radius_px=ci["radius_px"], supersample=ci["ss"],
                           templates=[dict(rank=k + 1, peak=t["pk"].height, fwhm=t["pk"].width, psr=t["pk"].psr,
                                           second=t["pk"].second_ratio,
                                           offset_m=[t["offset"][0] * ex["ms"].coarse.pixel,
                                                     -t["offset"][1] * ex["ms"].coarse.pixel],
                                           reject=ex["cres"].templates[k].get("reject"),
                                           geom_inlier=ex["cres"].templates[k].get("geom_inlier"))
                                      for k, t in enumerate(ci["tpls"])])
    print("figures: fine", flush=True)
    cats = [fine_categories(ex, it) for it in range(len(ex["fines"]))]
    picks = pick_landmarks(ex, cats[0])
    fi = fine_internals(ex, 0, list(picks.values()))
    fig_harris(ex, out)
    fig_fine_align(ex, out)
    stats["landmark"] = fig_fine_landmark(ex, fi, picks["inlier_a"], out)
    fig_fine_offsets(ex, fi, cats[0], out)
    stats["mahalanobis"] = fig_mahalanobis(ex, cats, out)
    fig_residuals(ex, out)
    stats["pose"] = fig_pose(ex, out)

    print("figures: illumination sweep example", flush=True)
    exs = [ex] + [run_example(src, args.geom, args.illum, args.prior_seed, cache, delta_az=d) for d in (30, 45, 60, 90)]
    stats["v3_example"] = fig_v3_example(ex, exs, out)
    hard = next(e for e in exs if e["delta_az"] == args.hard_delta_az)
    if hard["fines"]:
        hc = [fine_categories(hard, it) for it in range(len(hard["fines"]))]
        hp = pick_landmarks(hard, hc[0])
        hfi = fine_internals(hard, 0, list(hp.values()))
        fig_fine_gallery(hard, hfi, hp, out)
        stats["peak_gate"] = fig_peak_gate(hard, hc[0], out, f"Map Sun +{args.hard_delta_az:.0f}°: 150 features")
        fig_fine_offsets(hard, hfi, hc[0], out, name="f19b_fine_offsets_hard", sc=12.0)
        stats["hard"] = dict(delta_az=args.hard_delta_az, iterations=[fine_counts(dict(hard, fines=[f]))
                                                                     for f in hard["fines"]],
                             final=fine_counts(hard), picks=list(hp))
    stats["main_counts"] = fine_counts(ex)

    print("figures: parallax and shadow", flush=True)
    g_low = load_group(12420)[1]
    s_small = find_seed(g_low.T_world_cam, 500, 750)
    s_big = find_seed(g_low.T_world_cam, 2400, 2700)
    ex_small = run_example(src, 12420, 1, s_small, cache)
    ex_big = run_example(src, 12420, 1, s_big, cache)
    stats["parallax"] = fig_parallax(ex, ex_small, ex_big, out)
    stats["shadow_case"] = fig_shadow_case(out)
    print("aggregate statistics", flush=True)
    stats["aggregate"] = aggregate_stats()
    stats["elapsed_s"] = time.time() - t0
    (out / "stats.json").write_text(json.dumps(jsonable(stats), indent=1, default=str))
    print(f"done in {time.time() - t0:.0f} s -> {out / 'stats.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
