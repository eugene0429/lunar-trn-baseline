#!/usr/bin/env python3
"""Figures for the P-G (crater) baseline.

    python scripts/pg_figures.py overlay --data data/stereolunar/nadir \
        --catalog data/catalog/ldem_875s_5m_craters.npz --geometries 13740 2640 4200 11220 --out docs/validation/fig
    python scripts/pg_figures.py catalog-dem --dem data/dem/ldem_875s_5m.img \
        --catalog data/catalog/ldem_875s_5m_craters.npz --box -10000 -5000 -10000 -5000 --out docs/validation/fig
    python scripts/pg_figures.py csfd --catalog data/catalog/ldem_875s_5m_craters.npz --out docs/validation/fig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from lunar_trn.craters.annotate import project_catalog  # noqa: E402
from lunar_trn.craters.catalog import CraterCatalog  # noqa: E402
from lunar_trn.craters.detect import DetectorConfig, detect_craters, pinhole_sun_fn  # noqa: E402
from lunar_trn.datasets import stereolunar as sl  # noqa: E402
from lunar_trn.validation.v1 import footprint_bounds  # noqa: E402

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (same as make_figures.py)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)


def _ellipses(ax, u, v, A, B, th, color, pc, label):
    for k in range(len(u)):
        ax.add_patch(Ellipse((u[k] - pc, v[k] - pc), 2 * A[k], 2 * B[k], angle=np.degrees(th[k]), fill=False,
                             lw=1.0, ec=color, label=label if k == 0 else None))


def cmd_overlay(args):
    cat = CraterCatalog.load(args.catalog)
    frames_ = sl.list_frames(args.data)
    groups = {g[0].index: g for g in sl.assign_triplets(frames_) if g[0].illumination is not None}
    args.out.mkdir(parents=True, exist_ok=True)
    for gid in args.geometries:
        g = groups[gid]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        for ax, f in zip(axes, g):
            cam, T, img = f.camera(), f.T_world_cam, f.image()
            xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
            sub = cat.query_box(xmin, xmax, ymin, ymax)
            p = project_catalog(sub, cam, T)
            sel = p["in_image"] & (p["D_px"] >= args.min_px)
            alt = float(np.nanmedian(f.depth()[250:262, 250:262]))
            det, h = detect_craters(img, pinhole_sun_fn(cam, T, f.sun_direction(), alt), DetectorConfig(),
                                    pixel_center=cam.pixel_center)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            _ellipses(ax, p["u"][sel], p["v"][sel], p["A"][sel], p["B"][sel], p["theta"][sel], S1, cam.pixel_center,
                      f"catalog (true pose), {int(sel.sum())}")
            _ellipses(ax, det.u, det.v, det.A, det.B, det.theta, S2, cam.pixel_center, f"detections, {len(det)}")
            ax.set_xlim(-0.5, cam.width - 0.5)
            ax.set_ylim(cam.height - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"im {f.index}  illum {f.illumination}  alt {alt / 1000:.1f} km", color=TEXT, fontsize=10,
                         loc="left")
            ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.9, fontsize=8, labelcolor=TEXT, loc="lower left")
        fig.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(args.out / f"pg_overlay_{gid}.png", dpi=110)
        plt.close(fig)


def cmd_catalog_dem(args):
    from lunar_trn.dem import DemSource

    cat = CraterCatalog.load(args.catalog)
    x0, x1, y0, y1 = args.box
    d = DemSource(args.dem).read_window(x0, x1, y0, y1)
    gy, gx = np.gradient(d.heights.astype(float), d.pixel)
    hs = np.clip(0.5 + 1.5 * (0.6 * gx - 0.6 * gy), 0, 1)
    sub = cat.query_box(x0, x1, y0, y1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs, cmap="gray", extent=(d.x0, d.x0 + d.shape[1] * d.pixel, d.y0 - d.shape[0] * d.pixel, d.y0))
    for lo, hi, col, lab in ((0, 80, S2, "D < 80 m"), (80, 200, S3, "80-200 m"), (200, 1e9, S1, ">= 200 m")):
        s = (sub.D >= lo) & (sub.D < hi)
        for k, i in enumerate(np.nonzero(s)[0]):
            ax.add_patch(Ellipse((sub.x[i], sub.y[i]), sub.D[i], sub.D[i], fill=False, lw=0.9, ec=col,
                                 label=f"{lab}, {int(s.sum())}" if k == 0 else None))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    style(ax, f"DEM catalog over hillshade ({len(sub)} craters)", "stereographic x (m)", "stereographic y (m)")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.9, fontsize=8, labelcolor=TEXT, loc="lower left")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "pg_catalog_dem.png", dpi=110)
    plt.close(fig)


def cmd_csfd(args):
    cat = CraterCatalog.load(args.catalog)
    xmin, xmax, ymin, ymax = cat.meta.get("extent", [cat.x.min(), cat.x.max(), cat.y.min(), cat.y.max()])
    area_km2 = (xmax - xmin) * (ymax - ymin) / 1e6
    D = np.sort(cat.D)[::-1] / 1000.0
    N = np.arange(1, D.size + 1) / area_km2
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(D, N, color=S1, lw=2)
    ref = N[np.searchsorted(-D, -0.2)] * (D / 0.2) ** -2.0 if D.size else D
    ax.loglog(D, ref, color=TEXT2, lw=1, ls="--")
    ax.annotate("D^-2 (equilibrium)", (D[-1], ref[-1]), textcoords="offset points", xytext=(4, 4), color=TEXT2,
                fontsize=8)
    style(ax, f"Catalog cumulative size-frequency ({D.size} craters, {area_km2:.0f} km^2)", "D (km)",
          "N(>= D) per km^2")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "pg_csfd.png", dpi=110)
    plt.close(fig)


def _annots(run_dir: Path) -> dict:
    acc: dict = {}
    for p in sorted((run_dir / "annot").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                acc.setdefault(k.split("__", 1)[1], []).append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def cmd_recall(args):
    a = _annots(args.run_dir)
    D, hit = a["rc_D_px"], a["rc_detected"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = a["rc_d_over_D"] / (np.tan(np.radians(a["rc_sun_el_deg"])) / 4.0)
    edges = np.array([4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128], float)
    centres = np.sqrt(edges[:-1] * edges[1:])

    def curve(mask):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = mask & (D >= lo) & (D < hi)
            out.append(hit[s].mean() if s.sum() >= 30 else np.nan)
        return np.array(out)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(centres, curve(np.ones_like(hit)), "-o", color=S1, lw=2, ms=5, label="all catalog craters in view")
    ax.semilogx(centres, curve(a["rc_completeness"] >= 0.7), "-o", color=S2, lw=2, ms=5,
                label="well-formed rims (completeness >= 0.7), D_min basis")
    ax.set_ylim(0, 1)
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:g}" for e in edges])
    ax.minorticks_off()
    style(ax, "Detection probability vs crater size (calibration split)", "catalog crater diameter (px)",
          "fraction detected")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out / "pg_recall.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlay")
    o.add_argument("--data", type=Path, required=True)
    o.add_argument("--catalog", required=True)
    o.add_argument("--geometries", type=int, nargs="+", required=True)
    o.add_argument("--min-px", type=float, default=6.0)
    o.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("catalog-dem")
    c.add_argument("--dem", required=True)
    c.add_argument("--catalog", required=True)
    c.add_argument("--box", type=float, nargs=4, required=True)
    c.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("csfd")
    f.add_argument("--catalog", required=True)
    f.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("recall")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    {"overlay": cmd_overlay, "catalog-dem": cmd_catalog_dem, "csfd": cmd_csfd, "recall": cmd_recall}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
