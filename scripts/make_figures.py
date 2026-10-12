#!/usr/bin/env python3
"""Figures for docs/validation from the run summaries (static PNG, light theme).

    python scripts/make_figures.py --v2 outputs/v2/nadir \
        --sweep outputs/v3/nadir_sweep outputs/v3/nadir_sweep_wide --out docs/validation/fig
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
S1 = "#2a78d6"  # categorical slot 1 (blue)
S2 = "#eb6834"  # categorical slot 2 (orange)
S3 = "#1baf7a"  # categorical slot 3 (aqua)
BANDS_KM = [3.5 + 3 * k for k in range(10)]


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def load_runs(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "runs.jsonl").open() if '"verdict"' in line]


def availability_by_band(runs, pred):
    n = defaultdict(int)
    k = defaultdict(int)
    for r in runs:
        if pred(r):
            n[r["band"]] += 1
            k[r["band"]] += r["verdict"] == "correct"
    bands = sorted(n)
    return np.array([BANDS_KM[b] for b in bands]), np.array([k[b] / n[b] for b in bands])


def fig_availability(runs, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    x1, y1 = availability_by_band(runs, lambda r: r["map_illum"] == r["image_illum"])
    x2, y2 = availability_by_band(runs, lambda r: r["map_illum"] != r["image_illum"])
    ax.plot(x1, y1, "-o", color=S1, lw=2, ms=5, label="Map Sun = image Sun")
    ax.plot(x2, y2, "-o", color=S2, lw=2, ms=5, label="Map Sun 66–134° off (dataset pairs)")
    ax.annotate(f"{y1[0]:.2f}", (x1[0], y1[0]), textcoords="offset points", xytext=(6, -10), color=TEXT, fontsize=8)
    ax.annotate(f"{y1[2]:.2f}", (x1[2], y1[2]), textcoords="offset points", xytext=(4, -12), color=TEXT, fontsize=8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(BANDS_KM)
    _style(ax, "Availability (correct fix) by altitude band, StereoLunar nadir", "Altitude band (km)",
           "Fraction of runs")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="center right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_accuracy(runs, out: Path):
    m = [r for r in runs if r["map_illum"] == r["image_illum"] and r["verdict"] == "correct"]
    by = defaultdict(list)
    for r in m:
        by[r["band"]].append(r["err_h_m"])
    bands = sorted(by)
    x = np.array([BANDS_KM[b] for b in bands])
    p50 = np.array([np.percentile(by[b], 50) for b in bands])
    p95 = np.array([np.percentile(by[b], 95) for b in bands])
    gsd = np.array([BANDS_KM[b] * 1000 / 618.04 for b in bands])
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    ax.plot(x, gsd, "--", color=TEXT2, lw=1.2, label="1 map pixel (= GSD)")
    ax.plot(x, p95, "-o", color=S1, lw=2, ms=5, label="95th percentile")
    ax.plot(x, p50, "-o", color=S3, lw=2, ms=5, label="Median")
    ax.annotate("p95 ≈ 0.82 px in every band", (x[-3], p95[-3]), textcoords="offset points", xytext=(-150, 18),
                color=TEXT, fontsize=8)
    ax.set_xticks(BANDS_KM)
    _style(ax, "Fine horizontal error of correct fixes (matched Sun)", "Altitude band (km)", "Error (m)")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_sweep(sweep_runs, base_runs, out: Path):
    """Availability vs Sun angle between map and image, low (3.5-9.5 km) vs high (>= 12.5 km) bands."""
    pts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    geoms = {r["geometry_id"] for r in sweep_runs}
    allr = [r for r in base_runs if r["map_illum"] == r["image_illum"] and r["geometry_id"] in geoms] + sweep_runs
    for r in allr:
        grp = "low" if r["band"] <= 2 else "high"
        d = round(float(r.get("map_sun_delta_az", 0.0)))
        pts[grp][d][0] += r["verdict"] == "correct"
        pts[grp][d][1] += 1
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    for grp, col, lab in (("low", S2, "3.5–9.5 km"), ("high", S1, "12.5–30.5 km")):
        ds = sorted(pts[grp])
        y = [pts[grp][d][0] / pts[grp][d][1] for d in ds]
        ax.plot(ds, y, "-o", color=col, lw=2, ms=5, label=lab)
    ax.axvspan(0, 20, color=GRID, alpha=0.6, lw=0)
    ax.text(1, 0.05, "LVS envelope ≤ 20°", color=TEXT2, fontsize=8)
    ax.set_ylim(-0.03, 1.05)
    _style(ax, "Availability vs map–image Sun azimuth offset (Sun elevation 15–20°)",
           "Map Sun azimuth offset (deg)", "Fraction of runs")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="center left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.v2)
    fig_availability(runs, args.out / "v2_availability.png")
    fig_accuracy(runs, args.out / "v2_accuracy.png")
    sweep_runs = [r for d in args.sweep if (d / "runs.jsonl").exists() for r in load_runs(d)]
    if sweep_runs:
        fig_sweep(sweep_runs, runs, args.out / "v3_sun_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
