#!/usr/bin/env python3
"""P-G (crater) runner for V7 / H-b on StereoLunar.

One job = one geometry (illumination triplet). Each requested image illumination j is
localised from ``--trials`` priors. The prior seed equals the one ``run_v2.py`` uses with
the map illumination set to j, so every run pairs with the T-I run of the same (geometry,
illumination, trial) and identical prior error.

    python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
        --out outputs/pg/v7_nadir --split eval --trials 6 --workers 40
    # detector calibration: detections + ground-truth annotation per image, no localisation
    python scripts/run_pg.py --data data/stereolunar/nadir --catalog data/catalog/ldem_875s_5m_craters.npz \
        --out outputs/pg/calib_detect --split calib --detect-only --workers 40
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from lunar_trn import frames
from lunar_trn.craters.annotate import annotate_detections, catalog_recall
from lunar_trn.craters.catalog import CatalogConfig, CraterCatalog, explain_rim
from lunar_trn.craters.detect import Detections, detect_craters, pinhole_sun_fn
from lunar_trn.craters.localize import CraterConfig, CraterMap, localize_craters
from lunar_trn.datasets import stereolunar as sl
from lunar_trn.dem import DemSource
from lunar_trn.experiments import fine_map_pixel, sample_prior, score
from lunar_trn.validation.v1 import footprint_bounds

NOMINAL_BANDS_M = [3500 + 3000 * k for k in range(10)]
_CAT: dict = {}


def _catalog(path: str) -> CraterCatalog:
    if path not in _CAT:
        _CAT[path] = CraterCatalog.load(path)
    return _CAT[path]


_SRC: dict = {}


def _miss_diagnosis(det, ann, cam, T, dem_path: str, cat_cfg: dict | None, D_min_px: float) -> dict:
    """Why detections without a catalog counterpart are not in the catalog (calibration only)."""
    if dem_path not in _SRC:
        _SRC[dem_path] = DemSource(dem_path)
    n = len(det)
    out = {"miss_reason": np.full(n, "", dtype="<U12"), **{k: np.full(n, np.nan) for k in
           ("miss_D_m", "miss_completeness", "miss_d_over_D", "miss_ellipticity")}}
    sel = np.nonzero((ann["cat_index"] < 0) & (det.D_px >= D_min_px))[0]
    if sel.size == 0:
        return out
    xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
    dem = _SRC[dem_path].read_window(xmin - 8000.0, xmax + 8000.0, ymin - 8000.0, ymax + 8000.0)
    o, d = cam.rays_world(T, det.u[sel] - cam.pixel_center, det.v[sel] - cam.pixel_center)
    t = dem.raycast(np.broadcast_to(o, d.shape), d)
    ccfg = CatalogConfig(**cat_cfg) if cat_cfg else CatalogConfig()
    for k, s in enumerate(sel):
        if not np.isfinite(t[k]):
            out["miss_reason"][s] = "no_hit"
            continue
        x, y, _ = frames.body_to_stereo(o + t[k] * d[k])
        r_m = 0.5 * det.D_px[s] * t[k] * float(d[k] @ T[:3, 2]) / cam.fx
        reason, rec = explain_rim(dem, float(x), float(y), r_m, ccfg)
        out["miss_reason"][s] = reason
        if rec is not None:
            out["miss_D_m"][s] = 2.0 * np.sqrt(rec["a"] * rec["b"])
            out["miss_completeness"][s] = rec["completeness"]
            out["miss_d_over_D"][s] = rec["d_over_D"]
            out["miss_ellipticity"][s] = rec["a"] / rec["b"]
    return out


def _cfg_from(overrides: dict) -> CraterConfig:
    cfg = CraterConfig()
    for key, val in overrides.items():
        part, _, name = key.partition(".")
        if name:
            setattr(getattr(cfg, part), name, val)
        else:
            setattr(cfg, part, val)
    return cfg


def _image_info(f: sl.Frame):
    cam, T = f.camera(), f.T_world_cam
    c = cam.height // 2
    alt = float(np.nanmedian(f.depth()[c - 6:c + 6, c - 6:c + 6]))
    band = int(np.argmin([abs(alt - b) for b in NOMINAL_BANDS_M]))
    return cam, T, alt, band, fine_map_pixel(NOMINAL_BANDS_M[band] / cam.fx)


def run_job(job: dict) -> list[dict]:
    d = Path(job["dir"])
    idx = job["indices"]
    cat = _catalog(job["catalog"])
    cfg = _cfg_from(job["overrides"])
    out, annots = [], {}
    for j in job["illums"]:
        f = sl.Frame(d, idx[j], job["traj"], illumination=j, geometry_id=idx[0])
        cam, T, alt, band, map_px = _image_info(f)
        img, sun = f.image(), f.sun_direction()
        xmin, xmax, ymin, ymax = footprint_bounds(cam, T)
        local = cat.query_box(xmin - 500.0, xmax + 500.0, ymin - 500.0, ymax + 500.0)
        sun_fn = pinhole_sun_fn(cam, T, sun, alt)
        base = {"geometry_id": idx[0], "image_index": idx[j], "image_illum": j, "band": band, "alt_ground_m": alt,
                "gsd_m": alt / cam.fx, "map_px_fine": map_px, "altitude_sphere_m": f.altitude_above_sphere(),
                "off_nadir_deg": f.off_nadir_deg(), "trajectory": job["traj"], "family": "P-G",
                "catalog_D_min_m": cfg.match.catalog_D_min_m}
        if job["detect_only"]:
            t1 = time.time()
            det, dh = detect_craters(img, sun_fn, cfg.detector, pixel_center=cam.pixel_center)
            kw = {"tol_rel": job["ann_tol_rel"], "size_ratio_max": job["ann_size_ratio"]}
            ann = annotate_detections(det, local, cam, T, sun_fn, **kw)
            rc = catalog_recall(det, local, cam, T, sun, **kw)
            annots[f"{idx[0]}_{j}"] = {
                **{f"det_{k}": getattr(det, k) for k in ("u", "v", "A", "B", "theta", "pair_dev_deg", "contrast",
                                                          "level", "area_dark", "area_bright")},
                **{f"ann_{k}": v for k, v in ann.items()}, **{f"rc_{k}": v for k, v in rc.items()}}
            if job.get("dem"):
                annots[f"{idx[0]}_{j}"].update(_miss_diagnosis(det, ann, cam, T, job["dem"], cat.meta.get("config"),
                                                               cfg.detector.D_min_px))
            out.append({**base, "detect": dh, "n_det": len(det), "n_det_matched": int((ann["cat_index"] >= 0).sum()),
                        "n_cat_view": int(len(rc["cat_id"])), "n_cat_detected": int(rc["detected"].sum()),
                        "t_detect_s": time.time() - t1})
            continue
        cmap = CraterMap(cat, sun)
        enu = frames.enu_basis(T[:3, 3])
        for trial in range(job["trials"]):
            k_map = j  # unpruned P-G pairs with the matched-illumination T-I run
            seed = zlib.crc32(f"{idx[0]}-{k_map}-{j}-{trial}-{job['seed']}".encode())
            rng = np.random.default_rng(seed)
            Tp, cov, perr = sample_prior(T, rng, job["prior_radius_m"], job["prior_vert_m"], job["prior_att_deg"])
            t1 = time.time()
            try:
                res = localize_craters(img, cam.K, Tp, cov, cmap, cfg, seed=seed)
            except Exception as err:  # noqa: BLE001
                out.append({**base, "trial": trial, "error": repr(err)})
                continue
            sc = score(res, T, map_px, tol_px=job["tol_px"])
            rec = {**base, "trial": trial, "status": res.status, "stage": res.stage, **perr, **sc,
                   "t_run_s": time.time() - t1, "cov_enu_diag_m2": np.diag(enu.T @ res.cov[:3, :3] @ enu).tolist(),
                   "health": res.health}
            if res.health.get("cov_landmark_only") is not None:
                lm_cov = np.asarray(res.health["cov_landmark_only"])
                rec["cov_lm_enu_diag_m2"] = np.diag(enu.T @ lm_cov @ enu).tolist()
            lm = res.landmarks
            if lm and lm["inlier"].any():
                # Ground-truth consistency of the inliers (analysis only, never fed back).
                det_all = Detections.from_ellipses(lm["u"], lm["v"], lm["A"], lm["B"], lm["theta"])
                gt = annotate_detections(det_all, local, cam, T, sun_fn)["cat_id"]
                rec["n_inliers_gt_consistent"] = int(np.sum(lm["inlier"] & (gt >= 0) & (gt == lm["cat_id"])))
            out.append(rec)
    if annots:
        p = Path(job["out"]) / "annot" / f"{idx[0]}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{f"{key}__{fld}": np.asarray(arr) for key, dct in annots.items()
                                  for fld, arr in dct.items()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--image-illums", default="0,1,2")
    ap.add_argument("--prior-radius-m", type=float, default=3000.0)
    ap.add_argument("--prior-vert-m", type=float, default=120.0)
    ap.add_argument("--prior-att-deg", type=float, default=0.0)
    ap.add_argument("--tol-px", type=float, default=3.0, help="false-fix tolerance in nominal fine-map pixels")
    ap.add_argument("--split", choices=["all", "calib", "eval"], default="all",
                    help="calib = geometries with ordinal % 4 == 1, eval = the rest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. detector.kappa=0.8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--ann-tol-rel", type=float, default=0.25)
    ap.add_argument("--ann-size-ratio", type=float, default=1.5)
    ap.add_argument("--dem", default=None, help="DEM for the catalog-miss diagnosis (--detect-only)")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, val = item.partition("=")
        overrides[key] = json.loads(val)
    groups = [g for g in sl.assign_triplets(sl.list_frames(args.data)) if g[0].illumination is not None]
    if args.split == "calib":
        groups = [g for i, g in enumerate(groups) if i % 4 == 1]
    elif args.split == "eval":
        groups = [g for i, g in enumerate(groups) if i % 4 != 1]
    groups = groups[:: max(args.every, 1)]
    if args.limit:
        groups = groups[: args.limit]
    illums = [int(x) for x in args.image_illums.split(",")]
    jobs = [{"dir": str(args.data), "traj": args.data.name, "indices": [f.index for f in g], "illums": illums,
             "catalog": args.catalog, "trials": args.trials, "prior_radius_m": args.prior_radius_m,
             "prior_vert_m": args.prior_vert_m, "prior_att_deg": args.prior_att_deg, "tol_px": args.tol_px,
             "overrides": overrides, "seed": args.seed, "detect_only": args.detect_only,
             "ann_tol_rel": args.ann_tol_rel, "ann_size_ratio": args.ann_size_ratio, "out": str(args.out), "dem": args.dem}
            for g in groups]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "crater": dataclasses.asdict(_cfg_from(overrides))},
        indent=2))
    print(f"{len(groups)} geometries, {len(jobs)} jobs", flush=True)
    _catalog(args.catalog)  # load once before forking
    t0 = time.time()
    n_runs = 0
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=8) as pool, \
            (args.out / "runs.jsonl").open("w") as fh:
        for n, recs in enumerate(pool.imap_unordered(run_job, jobs), 1):
            for r in recs:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            n_runs += len(recs)
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} jobs, {n_runs} runs, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
