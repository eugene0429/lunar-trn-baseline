#!/usr/bin/env python3
"""Build the crater catalog of a whole DEM (tiled, parallel).

Each tile core is extracted from a window enlarged by a margin that holds the profiles of
the largest crater centred in the core, so tiles do not miss or duplicate craters.

    python scripts/build_catalog.py --dem data/dem/ldem_875s_5m.img \
        --out data/catalog/ldem_875s_5m_craters.npz --tile-km 10 --workers 32
    # development run on one area
    python scripts/build_catalog.py ... --bounds -20000 -10000 20000 30000 --out /tmp/cat_dev.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from lunar_trn.craters.catalog import CatalogConfig, CraterCatalog, extract_catalog, level_margin_m, tile_cores
from lunar_trn.dem import DemSource


def run_tile(job: dict):
    src = DemSource(job["dem"])
    x0, x1, y0, y1 = job["core"]
    m = job["margin_m"]
    try:
        dem = src.read_window(x0 - m, x1 + m, y0 - m, y1 + m)
    except ValueError:
        return None
    if not np.isfinite(dem.heights).any():
        return None
    return extract_catalog(dem, CatalogConfig(**job["cfg"]), core=job["core"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile-km", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    args = ap.parse_args(argv)
    cfg = CatalogConfig()
    src = DemSource(args.dem)
    extent = tuple(args.bounds) if args.bounds else src.extent
    # The coarsest level needs the widest margin; extract_catalog crops finer levels itself.
    margin = level_margin_m(cfg, max(cfg.levels), src.pixel)
    cores = tile_cores(extent, args.tile_km * 1000.0)
    jobs = [{"dem": args.dem, "core": c, "margin_m": margin, "cfg": dataclasses.asdict(cfg)} for c in cores]
    print(f"{len(jobs)} tiles, margin {margin:.0f} m", flush=True)
    t0 = time.time()
    cats = []
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=2) as pool:
        for n, cat in enumerate(pool.imap_unordered(run_tile, jobs), 1):
            if cat is not None and len(cat):
                cats.append(cat)
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} tiles, {sum(len(c) for c in cats)} craters, {time.time() - t0:.0f} s", flush=True)
    cat = CraterCatalog.concat(cats)
    order = np.lexsort((cat.x, -cat.y))
    cat = cat.subset(order)
    cat.id = np.arange(len(cat), dtype=np.int64)
    cat.meta = {"dem": src.name, "tile_m": args.tile_km * 1000.0, "margin_m": margin, "extent": list(extent),
                "config": dataclasses.asdict(cfg), "n": len(cat), "build_s": time.time() - t0}
    cat.save(args.out)
    D = cat.D
    print(f"saved {len(cat)} craters to {args.out}")
    if len(cat):
        print("D quantiles (m) 5/50/95/99:", np.round(np.percentile(D, [5, 50, 95, 99]), 1))
        for lv in np.unique(cat.level):
            print(f"  level {lv:3d}: {int(np.sum(cat.level == lv))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
