#!/usr/bin/env python3
"""Download the LOLA south pole 5 m DEM products with parallel HTTP range requests.

Products:
    pgda87s : PGDA 5 m/pix LOLA mosaic for 87-90 S (Barker et al. 2021), GeoTIFF, MOON_ME DE421
    pds875s : PDS LOLA GDR LDEM_875S_5M (2017), int16 IMG with PDS3 label, MOON_ME DE421

Usage:
    python scripts/fetch_dem.py pgda87s pds875s
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = {
    "pgda87s": [
        "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif",
    ],
    "pds875s": [
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_875s_5m.lbl",
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_875s_5m.img",
    ],
}


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, out: Path, retries: int = 8) -> None:
    """Download bytes [start, end] (inclusive) into ``out``, resuming a partial segment."""
    for attempt in range(retries):
        have = out.stat().st_size if out.exists() else 0
        want = end - start + 1
        if have >= want:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp, out.open("ab") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as err:  # noqa: BLE001
            print(f"retry {attempt + 1} for {out.name}: {err}", file=sys.stderr)
            time.sleep(3.0 * (attempt + 1))
    if out.stat().st_size != end - start + 1:
        raise IOError(f"segment {out} incomplete")


def download(url: str, dest: Path, segments: int) -> None:
    if dest.exists():
        print(f"exists: {dest}")
        return
    size = content_length(url)
    if size < (8 << 20):
        segments = 1
    bounds = [(i * size // segments, (i + 1) * size // segments - 1) for i in range(segments)]
    parts = [dest.with_name(f"{dest.name}.part{i:02d}") for i in range(segments)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=segments) as pool:
        futs = [pool.submit(fetch_range, url, s, e, p) for (s, e), p in zip(bounds, parts)]
        for f in cf.as_completed(futs):
            f.result()
    with dest.with_suffix(dest.suffix + ".tmp").open("wb") as fh:
        for p in parts:
            fh.write(p.read_bytes())
    dest.with_suffix(dest.suffix + ".tmp").replace(dest)
    for p in parts:
        p.unlink()
    print(f"{dest.name}: {size / 1e6:.1f} MB in {time.time() - t0:.0f} s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("products", nargs="+", choices=sorted(PRODUCTS))
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "dem")
    ap.add_argument("--segments", type=int, default=12)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for prod in args.products:
        for url in PRODUCTS[prod]:
            download(url, args.out / Path(url).name, args.segments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
