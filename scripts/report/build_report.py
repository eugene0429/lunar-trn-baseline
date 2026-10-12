#!/usr/bin/env python3
"""Assemble the self-contained pipeline report from the template, figures and stats.json.

Figures referenced as ``src="fig:NAME"`` are converted to WebP and inlined as data URIs, and
``{{REPORT_DATA}}`` receives the subset of stats.json the page scripts use. Two files are
written: ``trn_pipeline_report.html`` (a full document for local viewing) and
``trn_pipeline_report.artifact.html`` (body content only, for hosts that add the skeleton).

    .venv/bin/python scripts/report/build_report.py --out outputs/report
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent


def webp_data_uri(png: Path, quality: int) -> str:
    im = Image.open(png).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "outputs/report")
    ap.add_argument("--template", type=Path, default=HERE / "report_template.html")
    ap.add_argument("--quality", type=int, default=84)
    args = ap.parse_args(argv)
    html = args.template.read_text(encoding="utf-8")
    stats = json.loads((args.out / "stats.json").read_text())
    data = {k: stats[k] for k in ("aggregate", "coarse", "main_counts", "hard")}
    html = html.replace("{{REPORT_DATA}}", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    names = sorted(set(re.findall(r'src="fig:([A-Za-z0-9_]+)"', html)))
    total = 0
    for nm in names:
        uri = webp_data_uri(args.out / "fig" / f"{nm}.png", args.quality)
        total += len(uri)
        html = html.replace(f'src="fig:{nm}"', f'src="{uri}"')
    body = html
    (args.out / "trn_pipeline_report.artifact.html").write_text(body, encoding="utf-8")
    full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    (args.out / "trn_pipeline_report.html").write_text(full, encoding="utf-8")
    print(f"{len(names)} figures inlined ({total / 1e6:.1f} MB of data URIs); "
          f"page {len(full.encode()) / 1e6:.1f} MB -> {args.out / 'trn_pipeline_report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
