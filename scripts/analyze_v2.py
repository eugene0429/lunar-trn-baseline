#!/usr/bin/env python3
"""Aggregate run_v2.py results into the V2a/V2b/V2c (and V3) tables.

    python scripts/analyze_v2.py outputs/v2/nadir            # per band, matched illumination
    python scripts/analyze_v2.py outputs/v3/nadir --by-illum # per (map, image) illumination pair

Upper confidence bounds on rates are one-sided 95 % Clopper-Pearson (about 3/N for zero events).
Writes ``summary.json`` and ``summary.md`` next to ``runs.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

NOMINAL_BANDS_KM = [3.5 + 3 * k for k in range(10)]
ILLUM = ("side", "overhead", "back")


def cp_upper(x: int, n: int, conf: float = 0.95) -> float:
    if n == 0:
        return float("nan")
    return 1.0 if x >= n else float(beta.ppf(conf, x + 1, n - x))


def pct(a, q):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], float)
    return float(np.percentile(a, q)) if a.size else float("nan")


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    verdict = [r.get("verdict", "error") for r in runs]
    n_ok = verdict.count("correct")
    n_ff = verdict.count("false_fix")
    n_nf = verdict.count("no_fix")
    coarse_ok = [r for r in runs if r.get("health", {}).get("coarse", {}).get("ok")]
    ce = [r.get("coarse_err_h_m") for r in coarse_ok]
    corr = [r for r in runs if r.get("verdict") == "correct"]
    fixes = [r for r in runs if r.get("verdict") in ("correct", "false_fix")]
    inl = [r.get("health", {}).get("fine", {}).get("n_inliers", 0) for r in runs]
    row = {
        "n_runs": n, "n_correct": n_ok, "n_false_fix": n_ff, "n_no_fix": n_nf,
        "availability": n_ok / n if n else float("nan"),
        "false_fix_rate": n_ff / n if n else float("nan"),
        "false_fix_rate_ub95": cp_upper(n_ff, n),
        "false_fix_given_fix": n_ff / len(fixes) if fixes else float("nan"),
        "false_fix_given_fix_ub95": cp_upper(n_ff, len(fixes)) if fixes else float("nan"),
        "coarse_success": len(coarse_ok) / n if n else float("nan"),
        "coarse_err_h_p50_m": pct(ce, 50), "coarse_err_h_p95_m": pct(ce, 95), "coarse_err_h_p99_m": pct(ce, 99),
        "coarse_err_h_lt200_frac": float(np.mean([e < 200 for e in ce if e is not None])) if ce else float("nan"),
        "fine_err_h_p50_m": pct([r["err_h_m"] for r in corr], 50),
        "fine_err_h_p95_m": pct([r["err_h_m"] for r in corr], 95),
        "fine_err_h_p99_m": pct([r["err_h_m"] for r in corr], 99),
        "fine_err_h_p95_px": pct([r["err_h_m"] / r["map_px_fine"] for r in corr], 95),
        "fine_err_u_abs_p95_m": pct([abs(r["err_u_m"]) for r in corr], 95),
        "fine_err_att_p95_deg": pct([r["err_att_deg"] for r in corr], 95),
        "inliers_median": float(np.median(inl)) if inl else float("nan"),
        "inliers_ge20_frac": float(np.mean([i >= 20 for i in inl])) if inl else float("nan"),
        "map_px_fine": pct([r.get("map_px_fine") for r in runs], 50),
        "gsd_m": pct([r.get("gsd_m") for r in runs], 50),
        "max_false_fix_err_h_m": max([r["err_h_m"] for r in runs if r.get("verdict") == "false_fix"], default=None),
        "false_fix_inliers": sorted([r["health"]["fine"]["n_inliers"] for r in runs if r.get("verdict") == "false_fix"]),
        "t_run_median_s": pct([r.get("t_run_s") for r in runs], 50),
    }
    # Error-vs-gate: largest error among fixes passing the gate, and among all fine results.
    allfine = [r for r in runs if r.get("health", {}).get("fine", {}).get("n_inliers", 0) > 0]
    row["n_fine_any"] = len(allfine)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--by-illum", action="store_true")
    args = ap.parse_args(argv)
    runs = [json.loads(l) for l in (args.run_dir / "runs.jsonl").open()]
    errors = [r for r in runs if "error" in r]
    runs = [r for r in runs if "error" not in r]
    groups: dict = defaultdict(list)
    for r in runs:
        key = (r["band"], r["map_illum"], r["image_illum"]) if args.by_illum else (r["band"],)
        groups[key].append(r)
    table = {}
    for key in sorted(groups):
        table["/".join(map(str, key))] = summarize(groups[key])
    overall = summarize(runs)
    out = {"n_runs": len(runs), "n_errors": len(errors), "overall": overall, "groups": table}
    (args.run_dir / "summary.json").write_text(json.dumps(out, indent=2))

    lines = []
    if args.by_illum:
        lines.append("| 대역 (km) | 지도 조명 | 영상 조명 | 실행 | 가용성 | 오고정 | 오고정 상한95 | 거친 성공 | 정밀 오차 p95 (m) | inlier 중앙값 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, r in table.items():
            b, k, j = (int(x) for x in key.split("/"))
            lines.append(f"| {NOMINAL_BANDS_KM[b]:.1f} | {ILLUM[k]} | {ILLUM[j]} | {r['n_runs']} | {r['availability']:.3f} | "
                         f"{r['n_false_fix']} | {r['false_fix_rate_ub95']:.4f} | {r['coarse_success']:.3f} | "
                         f"{r['fine_err_h_p95_m']:.1f} | {r['inliers_median']:.0f} |")
    else:
        lines.append("| 대역 (km) | GSD (m) | 지도 px (m) | 실행 | 가용성 | 오고정 | 오고정 상한95 | 거친 성공 | 거친 오차 p99 (m) | 정밀 오차 p50/p95 (m) | p95 (px) | 수직 \\|e\\| p95 (m) | inlier 중앙값 | inlier ≥ 20 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, r in table.items():
            b = int(key)
            lines.append(f"| {NOMINAL_BANDS_KM[b]:.1f} | {r['gsd_m']:.1f} | {r['map_px_fine']:.0f} | {r['n_runs']} | "
                         f"{r['availability']:.3f} | {r['n_false_fix']} | {r['false_fix_rate_ub95']:.4f} | "
                         f"{r['coarse_success']:.3f} | {r['coarse_err_h_p99_m']:.0f} | "
                         f"{r['fine_err_h_p50_m']:.1f} / {r['fine_err_h_p95_m']:.1f} | {r['fine_err_h_p95_px']:.2f} | "
                         f"{r['fine_err_u_abs_p95_m']:.1f} | {r['inliers_median']:.0f} | {r['inliers_ge20_frac']:.3f} |")
    o = overall
    lines.append("")
    lines.append(f"전체: 실행 {o['n_runs']}, 가용성 {o['availability']:.3f}, 오고정 {o['n_false_fix']} "
                 f"(상한95 {o['false_fix_rate_ub95']:.4f}), 고정 중 오고정 {o['false_fix_given_fix']:.4f} "
                 f"(상한95 {o['false_fix_given_fix_ub95']:.4f}), 오류 {len(errors)}")
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
