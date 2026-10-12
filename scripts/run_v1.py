#!/usr/bin/env python3
"""Run the V1 frame/pose consistency test on a StereoLunar folder.

For every complete illumination triplet (three images with identical pose):
  * depth check against each DEM given with --dems (once per geometry),
  * render check of every image with the reconstructed generator Sun (first DEM),
  * Sun estimation from the image for every --sun-every-th geometry (first DEM).

Raw per-image records go to <out>/records.jsonl; <out>/summary.json and summary.md hold the
per-altitude-band aggregates. Example:

    python scripts/run_v1.py --data data/stereolunar/nadir \
        --dems data/dem/ldem_875s_5m.img data/dem/ldem_87s_5mpp.tif --out outputs/v1/nadir
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from lunar_trn.datasets import stereolunar as sl
from lunar_trn.dem import DemSource
from lunar_trn.validation import v1

NOMINAL_BANDS_M = [3500 + 3000 * k for k in range(10)]
_SOURCES: dict[str, DemSource] = {}


def _source(path: str) -> DemSource:
    if path not in _SOURCES:
        _SOURCES[path] = DemSource(path)
    return _SOURCES[path]


def band_of(alt_m: float) -> int:
    return int(np.argmin([abs(alt_m - b) for b in NOMINAL_BANDS_M]))


def render_dem(src: DemSource, cam, T, sun, gsd: float):
    """Window for rendering: extended towards the Sun, decimated to about GSD/2."""
    dem = v1.read_frame_window(src, cam, T, sun_dir=sun, shadow_reach_m=12000.0)
    factor = max(1, int(math.floor(gsd / 2.0 / dem.pixel)))
    return dem.downsample(factor), factor


def process_geometry(job: dict) -> list[dict]:
    frames_ = [sl.Frame(Path(job["dir"]), i, job["traj"], illumination=k, geometry_id=job["indices"][0])
               for k, i in enumerate(job["indices"])]
    f0 = frames_[0]
    cam = f0.camera(pixel_center=job["pixel_center"])
    T = f0.T_world_cam
    depth = f0.depth()
    out_geom = {"geometry_id": f0.index, "indices": job["indices"], "altitude_sphere_m": f0.altitude_above_sphere(),
                "off_nadir_deg": f0.off_nadir_deg(), "depth": {}}
    t0 = time.time()
    for dem_path in job["dems"]:
        src = _source(dem_path)
        try:
            dem = v1.read_frame_window(src, cam, T)
            chk, _ = v1.depth_check(dem, cam, T, depth, step=job["depth_step"])
            rec = chk.__dict__.copy()
        except Exception as err:  # noqa: BLE001 - report and continue
            rec = {"error": repr(err)}
        out_geom["depth"][Path(dem_path).stem] = rec
    first = out_geom["depth"][Path(job["dems"][0]).stem]
    alt = first.get("boresight_range", float("nan"))
    gsd = first.get("gsd_m", float("nan"))
    out_geom["altitude_ground_m"] = alt
    out_geom["band"] = band_of(alt) if np.isfinite(alt) else -1
    out_geom["t_depth_s"] = time.time() - t0

    records = []
    src = _source(job["dems"][0])
    for f in frames_:
        r = {"index": f.index, "illumination": f.illumination, "geometry_id": f0.index}
        try:
            sun = f.sun_direction()
            dem_r, factor = render_dem(src, cam, T, sun, gsd if np.isfinite(gsd) else 10.0)
            rc, arrays = v1.render_check(dem_r, cam, T, f.image(), sun, step=job["render_step"])
            rc["dem_decimation"] = factor
            r["render"] = rc
            if job["sun_estimate"]:
                r["sun_estimate"] = v1.estimate_sun(dem_r, cam, T, f.image(), step=max(8, job["render_step"]))
            if job.get("fig_dir"):
                _save_overlay(Path(job["fig_dir"]) / f"{f.stem}.png", arrays)
        except Exception as err:  # noqa: BLE001
            r["render"] = {"error": repr(err)}
        records.append(r)
    out_geom["t_total_s"] = time.time() - t0
    return [dict(out_geom, frames=records)]


def _save_overlay(path: Path, arrays: dict) -> None:
    import cv2

    def to8(a, pct):
        a = np.where(np.isfinite(a), a, 0.0)
        hi = np.nanpercentile(a, pct) or 1.0
        return (255 * np.clip(a / hi, 0, 1)).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack([to8(arrays["image"], 99.5), to8(arrays["render"], 99.5)]))


def summarize(geoms: list[dict], dem_names: list[str]) -> dict:
    by_band: dict[int, dict] = {}
    for b in range(len(NOMINAL_BANDS_M)):
        gs = [g for g in geoms if g["band"] == b]
        if not gs:
            continue
        row = {"nominal_alt_m": NOMINAL_BANDS_M[b], "n_geometries": len(gs),
               "gsd_m_median": float(np.median([g["depth"][dem_names[0]].get("gsd_m", np.nan) for g in gs]))}
        for name in dem_names:
            recs = [g["depth"][name] for g in gs if "stats_z" in g["depth"].get(name, {})]
            if not recs:
                continue
            med = np.array([r["stats_z"]["median"] for r in recs])
            nmad = np.array([r["stats_z"]["nmad"] for r in recs])
            rmed = np.array([r["stats_range"]["median"] for r in recs])
            du = np.array([r["shift_z"]["du"] for r in recs])
            dv = np.array([r["shift_z"]["dv"] for r in recs])
            q = np.array([r["quant_step_median"] for r in recs])
            gsd = np.array([r["gsd_m"] for r in recs])
            shift_m = np.hypot(du, dv) * gsd
            row[name] = {
                "n": len(recs),
                "z_residual_median_of_medians_m": float(np.median(med)),
                "z_residual_abs_median_max_m": float(np.max(np.abs(med))),
                "z_residual_nmad_median_m": float(np.median(nmad)),
                "range_residual_median_m": float(np.median(rmed)),
                "quant_step_m": float(np.median(q)),
                "quant_noise_std_m": float(np.median(q) / math.sqrt(12)),
                "shift_du_median_px": float(np.median(du)),
                "shift_dv_median_px": float(np.median(dv)),
                "shift_px_p95": float(np.percentile(np.hypot(du, dv), 95)),
                "shift_m_median": float(np.median(shift_m)),
                "shift_m_p95": float(np.percentile(shift_m, 95)),
            }
        frs = [fr for g in gs for fr in g["frames"] if "ncc" in fr.get("render", {})]
        for k, label in enumerate(sl.ILLUMINATION_NAMES):
            sel = [fr for fr in frs if fr["illumination"] == k]
            if sel:
                vals = np.array([fr["render"]["ncc"] for fr in sel], dtype=float)
                row[f"render_ncc_{label}_median"] = float(np.nanmedian(vals))
                row[f"render_ncc_{label}_p10"] = float(np.nanpercentile(vals, 10))
                row[f"render_ncc_{label}_n_nan"] = int(np.isnan(vals).sum())
        est = [fr for g in gs for fr in g["frames"] if "sun_estimate" in fr and "sun_az_deg" in fr.get("render", {})]
        if est:
            daz = np.array([(fr["sun_estimate"]["az_deg"] - fr["render"]["sun_az_deg"] + 180) % 360 - 180 for fr in est])
            del_ = np.array([fr["sun_estimate"]["el_deg"] - fr["render"]["sun_el_deg"] for fr in est])
            row["sun_estimate"] = {"n": len(est), "daz_median_deg": float(np.median(daz)),
                                   "daz_abs_p90_deg": float(np.percentile(np.abs(daz), 90)),
                                   "del_median_deg": float(np.median(del_)),
                                   "frac_within_20deg_az": float(np.mean(np.abs(daz) <= 20))}
        by_band[b] = row
    return by_band


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dems", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--depth-step", type=int, default=2)
    ap.add_argument("--render-step", type=int, default=2)
    ap.add_argument("--sun-every", type=int, default=3, help="estimate the Sun for every N-th geometry (0 = never)")
    ap.add_argument("--figures", type=int, default=12, help="number of geometries with saved overlays")
    ap.add_argument("--pixel-center", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true", help="rebuild the summary from records.jsonl")
    args = ap.parse_args(argv)
    dem_names = [Path(p).stem for p in args.dems]
    if args.summarize_only:
        geoms = [json.loads(line) for line in (args.out / "records.jsonl").open()]
        summary = summarize(geoms, dem_names)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    frames_ = sl.list_frames(args.data)
    groups = sl.assign_triplets(frames_)
    complete = [g for g in groups if g[0].illumination is not None]
    print(f"{len(frames_)} frames, {len(groups)} pose groups, {len(complete)} complete triplets", flush=True)
    if args.limit:
        complete = complete[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    fig_every = max(1, len(complete) // max(args.figures, 1)) if args.figures else 0
    jobs = []
    for k, g in enumerate(complete):
        jobs.append({
            "dir": str(args.data), "traj": args.data.name, "indices": [f.index for f in g],
            "dems": [str(p) for p in args.dems], "depth_step": args.depth_step, "render_step": args.render_step,
            "pixel_center": args.pixel_center,
            "sun_estimate": bool(args.sun_every) and k % args.sun_every == 0,
            "fig_dir": str(args.out / "fig") if fig_every and k % fig_every == 0 else None,
        })
    t0 = time.time()
    geoms = []
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=8) as pool, \
            (args.out / "records.jsonl").open("w") as fh:
        for n, res in enumerate(pool.imap_unordered(process_geometry, jobs), 1):
            for g in res:
                fh.write(json.dumps(g) + "\n")
                fh.flush()
                geoms.append(g)
            if n % 20 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} geometries, {time.time() - t0:.0f} s", flush=True)
    summary = summarize(geoms, dem_names)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
