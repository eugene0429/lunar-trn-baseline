#!/usr/bin/env python3
"""V2/V3 experiments: single-image coarse -> fine localisation on StereoLunar.

One job = (geometry, map illumination k). The job renders the coarse/fine maps once with
the Sun of illumination k, then localises images of the requested illuminations
(``--image-illums matched`` -> only k, ``all`` -> 0, 1, 2 for V3) from ``--trials``
random priors each. Every run is written to ``runs.jsonl`` with raw health signals and the
verdict (correct / no_fix / false_fix). ``--annotate`` adds ground-truth match errors per
landmark (for threshold calibration) to ``landmarks.npz`` files.

    python scripts/run_v2.py --data data/stereolunar/nadir --dem data/dem/ldem_875s_5m.img \
        --out outputs/v2/nadir --trials 10 --workers 24
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
from lunar_trn.datasets import stereolunar as sl
from lunar_trn.dem import DemSource
from lunar_trn.experiments import annotate_coarse, annotate_landmarks, build_mapset, sample_prior, score
from lunar_trn.localize import LocalizerConfig, localize
from lunar_trn.validation.v1 import sun_from_az_el

NOMINAL_BANDS_M = [3500 + 3000 * k for k in range(10)]
_SRC: dict[str, DemSource] = {}


def _source(p: str) -> DemSource:
    if p not in _SRC:
        _SRC[p] = DemSource(p)
    return _SRC[p]


def _cfg_from(overrides: dict) -> LocalizerConfig:
    cfg = LocalizerConfig()
    for key, val in overrides.items():
        part, _, name = key.partition(".")
        if name:
            setattr(getattr(cfg, part), name, val)
        else:
            setattr(cfg, part, val)
    return cfg


def _flat_health(h: dict) -> dict:
    """Health dict without bulky per-template lists (kept separately)."""
    out = {}
    for k, v in h.items():
        if k == "coarse":
            out[k] = {kk: vv for kk, vv in v.items() if kk != "templates"}
            tpl = v.get("templates", [])
            out["coarse_templates"] = [{kk: t.get(kk) for kk in ("peak", "width", "psr", "second_ratio",
                                                                  "phasecorr_response", "reject", "geom_inlier",
                                                                  "geom_residual_px", "texture")} for t in tpl]
        else:
            out[k] = v
    return out


def run_job(job: dict) -> list[dict]:
    t0 = time.time()
    d = Path(job["dir"])
    idx = job["indices"]
    k_map = job["map_illum"]
    f_map = sl.Frame(d, idx[k_map], job["traj"], illumination=k_map, geometry_id=idx[0])
    cam = f_map.camera()
    T = f_map.T_world_cam
    depth = f_map.depth()
    c = cam.height // 2
    alt = float(np.nanmedian(depth[c - 6:c + 6, c - 6:c + 6]))
    gsd = alt / cam.fx
    # Maps are made per nominal altitude band (an onboard map product), not per true altitude.
    band = int(np.argmin([abs(alt - b) for b in NOMINAL_BANDS_M]))
    gsd_nominal = NOMINAL_BANDS_M[band] / cam.fx
    src = _source(job["dem"])
    sun_map = f_map.sun_direction()
    d_az, d_el = job.get("delta_az", 0.0), job.get("delta_el", 0.0)
    if d_az or d_el:
        az0, el0 = sl.sun_azimuth_elevation(sun_map, T[:3, 3])
        sun_map = sun_from_az_el(T[:3, 3], (az0 + d_az) % 360.0, el0 + d_el)
    ms = build_mapset(src, cam, T, sun_map, gsd_nominal, job["prior_radius_m"])
    t_map = time.time() - t0
    cfg = _cfg_from(job["overrides"])
    out = []
    illums = [k_map] if job["image_illums"] == "matched" else [0, 1, 2]
    lm_store = {}
    for j in illums:
        f_img = sl.Frame(d, idx[j], job["traj"], illumination=j, geometry_id=idx[0])
        img = f_img.image()
        for trial in range(job["trials"]):
            seed = zlib.crc32(f"{idx[0]}-{k_map}-{j}-{trial}-{job['seed']}".encode())
            rng = np.random.default_rng(seed)
            Tp, cov, perr = sample_prior(T, rng, job["prior_radius_m"], job["prior_vert_m"], job["prior_att_deg"])
            t1 = time.time()
            try:
                res = localize(img, cam.K, Tp, cov, ms, cfg, seed=seed)
            except Exception as err:  # noqa: BLE001
                out.append({"geometry_id": idx[0], "map_illum": k_map, "image_illum": j, "trial": trial,
                            "error": repr(err)})
                continue
            sc = score(res, T, ms.fine.pixel, tol_px=job["tol_px"])
            enu = frames.enu_basis(T[:3, 3])
            corr = res.health.get("coarse", {}).get("correction_m")
            if corr is not None:
                d_c = enu.T @ (Tp[:3, 3] + np.asarray(corr) - T[:3, 3])
                sc["coarse_err_h_m"] = float(np.hypot(d_c[0], d_c[1]))
                sc["coarse_err_u_m"] = float(d_c[2])
            rec = {"geometry_id": idx[0], "image_index": idx[j], "map_illum": k_map, "image_illum": j,
                   "map_sun_delta_az": d_az, "map_sun_delta_el": d_el,
                   "sun_angle_map_image_deg": float(np.rad2deg(np.arccos(np.clip(
                       sun_map @ f_img.sun_direction(), -1, 1)))),
                   "trial": trial, "band": band, "alt_ground_m": alt, "gsd_m": gsd,
                   "map_px_fine": ms.fine.pixel, "map_px_coarse": ms.coarse.pixel,
                   "altitude_sphere_m": f_img.altitude_above_sphere(), "off_nadir_deg": f_img.off_nadir_deg(),
                   "trajectory": job["traj"], "status": res.status, "stage": res.stage, **perr, **sc,
                   "t_run_s": time.time() - t1, "cov_pos_diag": np.diag(res.cov)[:3].tolist(),
                   "health": _flat_health(res.health)}
            if job["annotate"]:
                ann = annotate_landmarks(res, cam, T, ms.fine)
                rec["coarse_template_err_px"] = annotate_coarse(res, cam, T, ms.coarse)
                if ann and res.landmarks:
                    key = f"{idx[0]}_{k_map}_{j}_{trial}"
                    lm = res.landmarks
                    lm_store[key] = {
                        "match_err_px": ann["match_err_px"],
                        **{f: np.asarray(lm[f], float) for f in ("peak", "peak_fit", "width", "width_min",
                                                                 "subpx_dx", "subpx_dy", "psr", "second_ratio",
                                                                 "harris")},
                        "status": np.array([s or "" for s in lm["status"]]),
                        "inlier": np.asarray(lm["inlier"], bool), "used": np.asarray(lm["used"], bool),
                    }
            out.append(rec)
    if lm_store:
        p = Path(job["out"]) / "landmarks" / f"{idx[0]}_{k_map}_{d_az:g}_{d_el:g}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for key, dct in lm_store.items():
            for f, arr in dct.items():
                flat[f"{key}__{f}"] = arr
        np.savez_compressed(p, **flat)
    for r in out:
        r["t_map_s"] = t_map
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--image-illums", choices=["matched", "all"], default="matched")
    ap.add_argument("--map-illums", default="0,1,2")
    ap.add_argument("--prior-radius-m", type=float, default=3000.0)
    ap.add_argument("--prior-vert-m", type=float, default=120.0)
    ap.add_argument("--prior-att-deg", type=float, default=0.0)
    ap.add_argument("--tol-px", type=float, default=3.0, help="false-fix tolerance in fine-map pixels")
    ap.add_argument("--split", choices=["all", "calib", "eval"], default="all",
                    help="calib = geometries with ordinal % 4 == 1, eval = the rest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="keep every N-th geometry after the split")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. fine.min_peak_height=0.4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sun-delta-az", default="0", help="comma list: map Sun azimuth offsets (deg) from the image Sun")
    ap.add_argument("--sun-delta-el", default="0", help="comma list: map Sun elevation offsets (deg)")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, val = item.partition("=")
        overrides[key] = json.loads(val)
    frames_ = sl.list_frames(args.data)
    groups = [g for g in sl.assign_triplets(frames_) if g[0].illumination is not None]
    if args.split == "calib":
        groups = [g for i, g in enumerate(groups) if i % 4 == 1]
    elif args.split == "eval":
        groups = [g for i, g in enumerate(groups) if i % 4 != 1]
    groups = groups[:: max(args.every, 1)]
    if args.limit:
        groups = groups[: args.limit]
    map_illums = [int(x) for x in args.map_illums.split(",")]
    jobs = [{"dir": str(args.data), "traj": args.data.name, "indices": [f.index for f in g], "map_illum": k,
             "dem": args.dem, "trials": args.trials, "image_illums": args.image_illums,
             "prior_radius_m": args.prior_radius_m, "prior_vert_m": args.prior_vert_m,
             "prior_att_deg": args.prior_att_deg, "tol_px": args.tol_px, "overrides": overrides,
             "annotate": args.annotate, "out": str(args.out), "seed": args.seed}
             | {"delta_az": float(da), "delta_el": float(de)}
            for g in groups for k in map_illums
            for da in [float(x) for x in args.sun_delta_az.split(",")]
            for de in [float(x) for x in args.sun_delta_el.split(",")]]
    # Heavier (low altitude, large maps) jobs first for better load balance.
    jobs.sort(key=lambda j: sl.Frame(Path(j["dir"]), j["indices"][0], j["traj"]).altitude_above_sphere())
    args.out.mkdir(parents=True, exist_ok=True)
    cfg_dump = dataclasses.asdict(_cfg_from(overrides))
    (args.out / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                      "localizer": cfg_dump}, indent=2))
    print(f"{len(groups)} geometries, {len(jobs)} jobs", flush=True)
    t0 = time.time()
    n_runs = 0
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=4) as pool, \
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
