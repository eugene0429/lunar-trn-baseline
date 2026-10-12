#!/usr/bin/env python3
"""Fetch a subset of a StereoLunar archive from Zenodo without downloading it whole.

Zenodo serves the archives with HTTP range support, so the zip central directory can be
read remotely and individual members extracted on demand. Per-connection throughput from
Zenodo is low (~100-300 kB/s), so members are fetched with a thread pool.

Examples:
    # List the members of nadir.zip and write the listing to data/stereolunar/
    python scripts/fetch_stereolunar.py --archive nadir --list

    # Keep the left image of the three illumination variants for every 10th stereo pair
    python scripts/fetch_stereolunar.py --archive nadir --block-stride 60
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "21133765"
URL_TEMPLATE = "https://zenodo.org/api/records/{record}/files/{name}.zip/content"
REPO_ROOT = Path(__file__).resolve().parents[1]


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file backed by HTTP range requests."""

    def __init__(self, url: str, block_size: int = 1 << 20, retries: int = 5):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.retries = retries
        self.pos = 0
        self.size = self._content_length()
        self._cache: dict[int, bytes] = {}

    def _content_length(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.headers["Content-Length"])

    def fetch(self, start: int, end: int) -> bytes:
        """Return bytes [start, end) with retries."""
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                if len(data) != end - start:
                    raise IOError(f"short read {len(data)} != {end - start}")
                return data
            except Exception as err:  # noqa: BLE001 - network errors of many kinds
                last_err = err
                time.sleep(2.0 * (attempt + 1))
        raise IOError(f"range fetch failed for {start}-{end}: {last_err}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        pos = self.pos
        while len(out) < n:
            block = pos // self.block_size
            if block not in self._cache:
                start = block * self.block_size
                end = min(start + self.block_size, self.size)
                self._cache[block] = self.fetch(start, end)
            data = self._cache[block]
            off = pos - block * self.block_size
            take = min(n - len(out), len(data) - off)
            out += data[off:off + take]
            pos += take
        b[:n] = out
        self.pos += n
        return n


def open_remote_zip(name: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    url = URL_TEMPLATE.format(record=RECORD, name=name)
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 16)), raw


def extract_member(raw: HttpRangeFile, info: zipfile.ZipInfo, out_root: Path) -> Path:
    """Extract one member with a single range request (local header + data)."""
    out_path = out_root / info.filename
    if out_path.exists() and out_path.stat().st_size == info.file_size:
        return out_path
    # Local file header is 30 bytes + name + extra; the extra length may differ from the
    # central directory copy, so read the header first.
    header = raw.fetch(info.header_offset, info.header_offset + 30)
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    data_start = info.header_offset + 30 + name_len + extra_len
    payload = raw.fetch(data_start, data_start + info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        data = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        import zlib

        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if len(data) != info.file_size:
        raise IOError(f"size mismatch for {info.filename}")
    import zlib

    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out_path)
    return out_path


INDEX_RE = re.compile(r"im_(\d+)\.(jpg|npz|exr|png)$")


def select_members(infos: list[zipfile.ZipInfo], block_stride: int, with_partner: bool):
    """Pick illumination triplets: pair blocks of six images starting at i, i % block_stride == 0.

    The ReadMe says the three illumination variants of one stereo pair are consecutive pairs,
    so the block i..i+5 (i % 6 == 0) is one stereo geometry under three suns. By default the
    even (left) image of each variant (i, i+2, i+4) is kept; ``with_partner`` keeps all six.
    """
    keep_offsets = (0, 1, 2, 3, 4, 5) if with_partner else (0, 2, 4)
    selected = []
    for info in infos:
        m = INDEX_RE.search(info.filename)
        if not m:
            continue
        idx = int(m.group(1))
        if idx % block_stride in keep_offsets:
            selected.append(info)
    return selected


def group_runs(infos: list[zipfile.ZipInfo], max_gap: int, max_run: int):
    """Group members (sorted by offset) into contiguous byte runs to cut the request count."""
    infos = sorted(infos, key=lambda i: i.header_offset)
    runs: list[list[zipfile.ZipInfo]] = []
    for info in infos:
        if runs:
            last = runs[-1]
            run_start = last[0].header_offset
            prev_end = last[-1].header_offset + 30 + len(last[-1].filename) + 256 + last[-1].compress_size
            end = info.header_offset + 30 + len(info.filename) + 256 + info.compress_size
            if info.header_offset - prev_end < max_gap and end - run_start < max_run:
                last.append(info)
                continue
        runs.append([info])
    return runs


def extract_run(raw: HttpRangeFile, run: list[zipfile.ZipInfo], out_root: Path) -> int:
    """Fetch one contiguous byte run and write every member in it. Returns compressed bytes."""
    import zlib

    todo = [i for i in run if not ((out_root / i.filename).exists()
                                   and (out_root / i.filename).stat().st_size == i.file_size)]
    if not todo:
        return sum(i.compress_size for i in run)
    start = todo[0].header_offset
    end = min(raw.size, todo[-1].header_offset + 30 + len(todo[-1].filename) + 1024 + todo[-1].compress_size)
    buf = raw.fetch(start, end)
    for info in todo:
        off = info.header_offset - start
        header = buf[off:off + 30]
        if header[:4] != b"PK\x03\x04":
            raise IOError(f"bad local header for {info.filename}")
        name_len = int.from_bytes(header[26:28], "little")
        extra_len = int.from_bytes(header[28:30], "little")
        data_start = off + 30 + name_len + extra_len
        payload = buf[data_start:data_start + info.compress_size]
        if len(payload) != info.compress_size:
            # Local extra field longer than budgeted: fetch the member on its own.
            extract_member(raw, info, out_root)
            continue
        if info.compress_type == zipfile.ZIP_STORED:
            data = payload
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(payload, -15)
        else:
            raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
        if len(data) != info.file_size or zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
            raise IOError(f"size/CRC mismatch for {info.filename}")
        out_path = out_root / info.filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(out_path)
    return sum(i.compress_size for i in run)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="nadir", choices=["nadir", "oblique", "dynamic"])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "stereolunar")
    ap.add_argument("--list", action="store_true", help="write the member listing and exit")
    ap.add_argument("--block-stride", type=int, default=60,
                    help="take the six-image pair block starting at every N-th image index (multiple of 6)")
    ap.add_argument("--with-partner", action="store_true", help="also keep the odd (right) stereo image")
    ap.add_argument("--indices-file", type=Path, default=None,
                    help="fetch exactly these image indices (one per line) instead of a block stride")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-run-mb", type=float, default=8.0)
    args = ap.parse_args(argv)

    if args.block_stride % 6 != 0:
        ap.error("--block-stride must be a multiple of 6 so that a block starts at a pair")

    zf, raw = open_remote_zip(args.archive)
    infos = zf.infolist()
    args.out.mkdir(parents=True, exist_ok=True)
    listing = args.out / f"{args.archive}_listing.tsv"
    with listing.open("w") as fh:
        fh.write("filename\tfile_size\tcompress_size\theader_offset\n")
        for info in infos:
            fh.write(f"{info.filename}\t{info.file_size}\t{info.compress_size}\t{info.header_offset}\n")
    print(f"{len(infos)} members, listing written to {listing}", flush=True)
    if args.list:
        return 0

    if args.indices_file is not None:
        wanted = {int(t) for t in args.indices_file.read_text().split()}
        selected = [i for i in infos if (m := INDEX_RE.search(i.filename)) and int(m.group(1)) in wanted]
    else:
        selected = select_members(infos, args.block_stride, args.with_partner)
    runs = group_runs(selected, max_gap=1 << 20, max_run=int(args.max_run_mb * (1 << 20)))
    total = sum(i.compress_size for i in selected)
    print(f"selected {len(selected)} members in {len(runs)} runs, {total / 1e6:.1f} MB compressed", flush=True)
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_run, raw, run, args.out): run for run in runs}
        for fut in cf.as_completed(futures):
            run = futures[fut]
            try:
                done += fut.result()
            except Exception as err:  # noqa: BLE001
                print(f"FAILED run at {run[0].filename}: {err}", file=sys.stderr, flush=True)
                continue
            rate = done / max(time.time() - t0, 1e-6)
            print(f"{run[0].filename}..{run[-1].filename}  {done / 1e6:.1f}/{total / 1e6:.1f} MB"
                  f"  {rate / 1e6:.2f} MB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
