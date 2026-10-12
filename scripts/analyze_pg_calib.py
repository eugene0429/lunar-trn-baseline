#!/usr/bin/env python3
"""Calibration analysis of the P-G baseline (calibration split only, decisions section G).

    python scripts/analyze_pg_calib.py detect outputs/pg/calibA0   # kappa, beta, pair angle, recall, misses
    python scripts/analyze_pg_calib.py loc outputs/pg/calibB         # fix gate and covariance floor
Writes calib_detect.json / calib_loc.json into the run directory and prints the --set flags.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

D_EDGES = [4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128]
Q_EDGES = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
# D_min_px is measured on well-formed rims (the catalog completeness before the one-time
# relaxation), so that adding partial rims for matching does not redefine the detector limit.
D_MIN_COMPLETENESS = 0.7


def load_annots(run_dir: Path) -> dict:
    """Concatenate every image's annotation arrays (keys without the image prefix)."""
    acc: dict = {}
    for p in sorted((run_dir / "annot").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                acc.setdefault(k.split("__", 1)[1], []).append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def _rate_table(x, ok, edges, min_n=30):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum()),
                     "rate": float(ok[s].mean()) if s.sum() >= min_n else float("nan")})
    return rows


def cmd_detect(run_dir: Path) -> dict:
    det_cfg = json.loads((run_dir / "config.json").read_text())["crater"]["detector"]
    k0, b0 = det_cfg["kappa"], det_cfg["beta"]
    a = load_annots(run_dir)
    m = a["ann_cat_index"] >= 0
    A0 = a["det_A"]
    D0 = 2.0 * np.sqrt(a["det_A"] * a["det_B"])
    factor = float(np.median(a["ann_size_ratio"][m]))
    kappa = k0 * factor
    A1 = A0 * factor
    beta = float(np.median((b0 * A0[m] + a["ann_err_antisun_px"][m]) / A1[m]))
    pair = float(math.ceil(np.percentile(a["det_pair_dev_deg"][m], 98.5) / 5.0) * 5.0)
    along = a["ann_err_antisun_px"] - (beta * A1 - b0 * A0)
    perp = a["ann_err_perp_px"]
    D1 = D0 * factor
    sig_rows = []
    for lo, hi in zip(D_EDGES[:-1], D_EDGES[1:]):
        s = m & (D1 >= lo) & (D1 < hi)
        if s.sum() >= 30:
            r = np.concatenate([along[s] - np.median(along[s]), perp[s] - np.median(perp[s])])
            sig_rows.append({"lo": lo, "hi": hi, "n": int(s.sum()), "sigma_px": float(1.4826 * np.median(np.abs(r)))})
    big = [r for r in sig_rows if math.sqrt(r["lo"] * r["hi"]) >= 10]
    Dc = np.array([math.sqrt(r["lo"] * r["hi"]) for r in big])
    sg = np.array([r["sigma_px"] for r in big])
    sigma_rel = float((sg * Dc).sum() / (Dc**2).sum()) if big else 0.05
    sigma_min = float(sig_rows[0]["sigma_px"]) if sig_rows else 0.5
    rec_D_all = _rate_table(a["rc_D_px"], a["rc_detected"], D_EDGES)
    well = a["rc_completeness"] >= D_MIN_COMPLETENESS
    rec_D = _rate_table(a["rc_D_px"][well], a["rc_detected"][well], D_EDGES)
    D_min = next((r["lo"] for r in rec_D if np.isfinite(r["rate"]) and r["rate"] >= 0.5), D_EDGES[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        q = a["rc_d_over_D"] / (np.tan(np.radians(a["rc_sun_el_deg"])) / 4.0)
    rec_q = _rate_table(np.nan_to_num(q, nan=-1.0), a["rc_detected"], Q_EDGES)
    low = (q < 0.75) & np.isfinite(q)
    high = (q >= 1.0) & np.isfinite(q)
    r_low = float(a["rc_detected"][low].mean()) if low.sum() else float("nan")
    r_high = float(a["rc_detected"][high].mean()) if high.sum() else float("nan")
    vis_on = bool(r_low < 0.2 and r_high >= 2 * r_low)
    c_vis = next((r["lo"] for r in rec_q if np.isfinite(r["rate"]) and r["rate"] >= 0.2), 0.0) if vis_on else 1.0
    out = {"kappa": kappa, "beta": beta, "pair_angle_max_deg": pair, "D_min_px": float(D_min),
           "sigma_min_px": sigma_min, "sigma_rel": sigma_rel, "visibility_filter": vis_on, "c_vis": float(c_vis),
           "n_detections": int(m.size), "n_matched": int(m.sum()), "size_factor": factor, "sigma_by_D": sig_rows,
           "recall_by_D": rec_D, "recall_by_D_all": rec_D_all, "recall_by_q": rec_q, "recall_q_low": r_low, "recall_q_high": r_high}
    if "miss_reason" in a:
        reasons = a["miss_reason"][a["miss_reason"] != ""]
        cnt = Counter(reasons.tolist())
        n = max(len(reasons), 1)
        out["miss_reasons"] = {k: v / n for k, v in cnt.most_common()}
        relax = {}
        for name, key, q80 in (("completeness", "miss_completeness", 20), ("dD", "miss_d_over_D", 20),
                               ("ellipticity", "miss_ellipticity", 80)):
            if cnt.get(name, 0) / n >= 0.3:
                vals = a[key][a["miss_reason"] == name]
                relax[name] = float(np.percentile(vals[np.isfinite(vals)], q80))
        if cnt.get("ok", 0) / n >= 0.3:
            relax["seed_dD_min_factor"] = 0.5
        out["miss_relax"] = relax
    out["set_flags"] = [f"--set detector.kappa={kappa:.4f}", f"--set detector.beta={beta:.4f}",
                        f"--set detector.pair_angle_max_deg={pair:g}", f"--set detector.D_min_px={D_min:g}",
                        f"--set match.sigma_min_px={sigma_min:.3f}", f"--set match.sigma_rel={sigma_rel:.4f}",
                        f"--set match.visibility_filter={json.dumps(vis_on)}", f"--set match.c_vis={c_vis:g}"]
    (run_dir / "calib_detect.json").write_text(json.dumps(out, indent=2))
    return out


def cmd_loc(run_dir: Path) -> dict:
    runs = [json.loads(line) for line in (run_dir / "runs.jsonl").open()]
    pose = [r for r in runs if r.get("stage") == "pose" and r["health"]["pg"].get("n_inliers", 0) >= 5]
    pg = lambda r, k: r["health"]["pg"].get(k)  # noqa: E731
    right = [r for r in pose if r["err_h_m"] <= r["tol_m"]]
    wrong = [r for r in pose if r["err_h_m"] > r["tol_m"]]
    s0 = np.array([pg(r, "sigma0") for r in right], float)
    s_max = float(math.ceil(np.percentile(s0, 99.5) / 0.05) * 0.05)
    w = [pg(r, "catalog_match_ratio") for r in wrong if pg(r, "sigma0") <= s_max]
    r_min = float(math.ceil((max(w) + 1e-9) * 100.0) / 100.0) if w else 0.0
    passed = [r for r in right if pg(r, "sigma0") <= s_max and pg(r, "catalog_match_ratio") >= r_min]
    retention = len(passed) / max(len(right), 1)
    eh = np.array([[r["err_e_m"] / r["gsd_m"], r["err_n_m"] / r["gsd_m"]] for r in passed])
    ev = np.array([r["err_u_m"] / r["gsd_m"] for r in passed])
    floor_h = float(np.sqrt(np.mean(eh**2))) if len(passed) else 0.0
    floor_v = float(np.sqrt(np.mean(ev**2))) if len(passed) else 0.0
    cover = []
    for r in passed:
        lm = r.get("cov_lm_enu_diag_m2")
        if lm is None:
            continue
        g = r["gsd_m"]
        sx, sy = math.sqrt(lm[0] + (floor_h * g) ** 2), math.sqrt(lm[1] + (floor_h * g) ** 2)
        cover.append(abs(r["err_e_m"]) <= 2 * sx and abs(r["err_n_m"]) <= 2 * sy)
    by_band = {}
    for b in sorted({r["band"] for r in runs if "band" in r}):
        rb = [r for r in runs if r.get("band") == b]
        ok = [r for r in rb if r.get("stage") == "pose" and pg(r, "n_inliers") >= 5 and pg(r, "sigma0") <= s_max
              and pg(r, "catalog_match_ratio") >= r_min]
        by_band[b] = {"runs": len(rb), "fix": len(ok), "correct": sum(r["err_h_m"] <= r["tol_m"] for r in ok),
                      "correct_n20": sum(r["err_h_m"] <= r["tol_m"] and pg(r, "n_inliers") >= 20 for r in ok)}
    out = {"n_min": 5, "s_max": s_max, "r_min": r_min, "retention": retention, "n_right": len(right),
           "n_wrong": len(wrong), "floor_h_px": floor_h, "floor_v_px": floor_v,
           "coverage_2sigma": float(np.mean(cover)) if cover else float("nan"), "by_band": by_band,
           "wrong_sigma0": sorted(float(pg(r, "sigma0")) for r in wrong),
           "wrong_match_ratio": sorted(float(pg(r, "catalog_match_ratio")) for r in wrong)}
    out["set_flags"] = ["--set min_inliers_fix=5", f"--set max_sigma0={s_max:g}", f"--set min_match_ratio={r_min:g}",
                        f"--set cov_floor_h_px={floor_h:.3f}", f"--set cov_floor_v_px={floor_v:.3f}"]
    (run_dir / "calib_loc.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["detect", "loc"])
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    out = cmd_detect(args.run_dir) if args.cmd == "detect" else cmd_loc(args.run_dir)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list) or k == "set_flags"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
