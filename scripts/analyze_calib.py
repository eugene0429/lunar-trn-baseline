#!/usr/bin/env python3
"""Peak-gate calibration from an annotated, gates-off run (``run_v2.py --annotate``).

Labels every fine landmark that produced a correlation peak as correct when its
ground-truth match error is below ``--correct-px`` fine-map pixels, and every coarse template
when its error is below ``--coarse-correct-px`` coarse-map pixels. Prints per-band
distributions and threshold tables (pass rates of correct vs wrong matches) and writes
``calib_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_fine(run_dir: Path, runs: list[dict]) -> dict:
    band_of = {(r["geometry_id"], r["map_illum"], r["image_illum"], r["trial"]): r["band"] for r in runs if "band" in r}
    cols = {k: [] for k in ("err", "peak", "width", "width_min", "psr", "second_ratio", "subpx", "band", "harris")}
    for p in sorted((run_dir / "landmarks").glob("*.npz")):  # names: geometry_mapillum[_daz_del]
        with np.load(p) as d:
            keys = sorted({k.split("__")[0] for k in d.files})
            for key in keys:
                g, k, j, t = (int(x) for x in key.split("_"))
                err = d[f"{key}__match_err_px"]
                pk = d[f"{key}__peak"]
                ok = np.isfinite(err) & np.isfinite(pk)
                cols["err"].append(err[ok])
                cols["peak"].append(pk[ok])
                cols["width"].append(d[f"{key}__width"][ok])
                cols["width_min"].append(d[f"{key}__width_min"][ok])
                cols["psr"].append(d[f"{key}__psr"][ok])
                cols["second_ratio"].append(d[f"{key}__second_ratio"][ok])
                cols["subpx"].append(np.hypot(d[f"{key}__subpx_dx"][ok], d[f"{key}__subpx_dy"][ok]))
                cols["harris"].append(d[f"{key}__harris"][ok])
                cols["band"].append(np.full(ok.sum(), band_of.get((g, k, j, t), -1)))
    return {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}


def pass_table(values: np.ndarray, correct: np.ndarray, thresholds, keep_above: bool) -> list[dict]:
    rows = []
    for th in thresholds:
        passed = values >= th if keep_above else values <= th
        rows.append({"threshold": float(th),
                     "correct_pass": float(passed[correct].mean()) if correct.any() else float("nan"),
                     "wrong_pass": float(passed[~correct].mean()) if (~correct).any() else float("nan"),
                     "precision": float(correct[passed].mean()) if passed.any() else float("nan")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--correct-px", type=float, default=1.5)
    ap.add_argument("--coarse-correct-px", type=float, default=2.0)
    args = ap.parse_args(argv)
    runs = [json.loads(l) for l in (args.run_dir / "runs.jsonl").open()]
    out = {}

    F = load_fine(args.run_dir, runs)
    corr = F["err"] < args.correct_px
    print(f"fine landmarks with a peak: {F['err'].size}, correct (<{args.correct_px} px): {corr.mean():.3f}")
    pct = [5, 25, 50, 75, 95, 99]
    for name in ("peak", "width", "width_min", "psr", "second_ratio", "subpx"):
        v = F[name]
        fin = np.isfinite(v)
        print(f"  {name:13s} correct {np.percentile(v[corr & fin], pct).round(3)}  wrong {np.percentile(v[~corr & fin], pct).round(3)}")
    out["fine"] = {"n": int(F["err"].size), "correct_fraction": float(corr.mean()),
                   "peak": pass_table(F["peak"], corr, np.arange(0.0, 0.95, 0.05), True),
                   "width": pass_table(np.where(np.isfinite(F["width"]), F["width"], 1e9), corr,
                                       [3, 4, 5, 6, 7, 8, 10, 12, 15], False)}
    print("  peak height gate (keep >= th): th, correct_pass, wrong_pass, precision")
    for r in out["fine"]["peak"]:
        print(f"    {r['threshold']:.2f} {r['correct_pass']:.3f} {r['wrong_pass']:.3f} {r['precision']:.3f}")
    print("  peak width gate (keep <= th):")
    for r in out["fine"]["width"]:
        print(f"    {r['threshold']:5.1f} {r['correct_pass']:.3f} {r['wrong_pass']:.3f} {r['precision']:.3f}")
    # Per band correctness and median peak.
    out["fine_by_band"] = {}
    for b in sorted(set(F["band"].tolist())):
        m = F["band"] == b
        out["fine_by_band"][int(b)] = {"n": int(m.sum()), "correct_fraction": float(corr[m].mean()),
                                       "peak_correct_median": float(np.median(F["peak"][m & corr])) if (m & corr).any() else None,
                                       "width_correct_p95": float(np.percentile(F["width"][m & corr & np.isfinite(F["width"])], 95)) if (m & corr).any() else None}
        print(f"  band {b}: n {m.sum()} correct {corr[m].mean():.3f} peak_med(correct) {out['fine_by_band'][int(b)]['peak_correct_median']} width_p95(correct) {out['fine_by_band'][int(b)]['width_correct_p95']}")

    # Coarse templates.
    pk, wd, er, bd = [], [], [], []
    for r in runs:
        errs = r.get("coarse_template_err_px")
        if not errs:
            continue
        for t, e in zip(r["health"].get("coarse_templates", []), errs):
            if t.get("peak") is None or e is None or not np.isfinite(e):
                continue
            pk.append(t["peak"]); wd.append(t["width"] if t["width"] is not None else np.inf); er.append(e); bd.append(r["band"])
    pk, wd, er, bd = map(np.asarray, (pk, wd, er, bd))
    if pk.size:
        cc = er < args.coarse_correct_px
        print(f"coarse templates: {pk.size}, correct (<{args.coarse_correct_px} px): {cc.mean():.3f}")
        print(f"  peak correct {np.percentile(pk[cc], pct).round(3)} wrong {np.percentile(pk[~cc], pct).round(3)}")
        wdf = np.where(np.isfinite(wd), wd, 1e9)
        print(f"  width correct {np.percentile(wdf[cc], pct).round(2)} wrong {np.percentile(wdf[~cc], pct).round(2)}")
        out["coarse"] = {"n": int(pk.size), "correct_fraction": float(cc.mean()),
                         "peak": pass_table(pk, cc, np.arange(0.0, 0.95, 0.05), True),
                         "width": pass_table(wdf, cc, [3, 4, 5, 6, 8, 10, 15], False)}
        print("  coarse peak gate:")
        for r in out["coarse"]["peak"]:
            print(f"    {r['threshold']:.2f} {r['correct_pass']:.3f} {r['wrong_pass']:.3f} {r['precision']:.3f}")
        print("  coarse width gate:")
        for r in out["coarse"]["width"]:
            print(f"    {r['threshold']:5.1f} {r['correct_pass']:.3f} {r['wrong_pass']:.3f} {r['precision']:.3f}")
        out["coarse_by_band"] = {int(b): {"n": int((bd == b).sum()), "correct_fraction": float(cc[bd == b].mean())}
                                 for b in sorted(set(bd.tolist()))}
        print("  by band:", {b: round(v["correct_fraction"], 3) for b, v in out["coarse_by_band"].items()})
    (args.run_dir / "calib_summary.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
