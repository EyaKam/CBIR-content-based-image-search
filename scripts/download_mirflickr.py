"""Download the MIRFLICKR-1M files used by the project (resumable).

    python scripts/download_mirflickr.py --dry-run                # list files and sizes only
    python scripts/download_mirflickr.py                          # everything (~4.0 GB)
    python scripts/download_mirflickr.py --only ehd tags thumbs1  # a subset

Files go to data/mirflickr_raw/ (CBIR_MIRFLICKR_RAW). The server needs the descriptor zips and the
tags to search; each thumbnail zip only adds the pictures of one third of the base
(thumbs1: images 1-330 000, thumbs2: 330 001-660 000, thumbs3: 660 001-1 000 000).
Source: https://press.liacs.nl/mirflickr/mirdownload.html
"""
import argparse
import sys
import time
from pathlib import Path

import requests

BASE = "https://press.liacs.nl/mirflickr/mirflickr1m.v3b/"
FILES = {
    "ehd": "features_edgehistogram.zip",         # MPEG-7 Edge Histogram        (~624 MB)
    "htd": "features_homogeneoustexture.zip",    # MPEG-7 Homogeneous Texture   (~360 MB)
    "tags": "tags.zip",                          # keywords                     (~229 MB)
    "thumbs1": "thumbs1-33.zip",                 # 64x64 thumbnails, part 1     (~945 MB)
    "thumbs2": "thumbs34-66.zip",                # part 2                       (~932 MB)
    "thumbs3": "thumbs67-100.zip",               # part 3                       (~955 MB)
}


def remote_size(name):
    r = requests.head(BASE + name, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download(name, out, size):
    dest = out / name
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        print(f"  {name}: already complete")
        return
    if have > size:
        dest.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(BASE + name, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if have and r.status_code != 206:           # server ignored Range: start over
            have = 0
        t0, last = time.time(), 0.0
        with open(dest, "ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                have += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    rate = (have / 2**20) / max(time.time() - t0, 1e-9)
                    print(f"  {name}: {have / 2**20:7.0f} / {size / 2**20:.0f} MB", end="\r", flush=True)
    if dest.stat().st_size != size:
        sys.exit(f"{name}: size {dest.stat().st_size} != expected {size}; run the script again to resume")
    print(f"  {name}: done ({size / 2**20:.0f} MB)          ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mirflickr_raw")
    ap.add_argument("--only", nargs="+", choices=list(FILES), help="download only these files")
    ap.add_argument("--dry-run", action="store_true", help="only list the files and their sizes")
    a = ap.parse_args()

    names = [FILES[k] for k in (a.only or FILES)]
    sizes = {n: remote_size(n) for n in names}
    total = sum(sizes.values())
    for n in names:
        print(f"{n:36s} {sizes[n] / 2**20:8.0f} MB")
    print(f"{'total':36s} {total / 2**20:8.0f} MB ({total / 1e9:.1f} GB)")
    if a.dry_run:
        sys.exit(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        download(n, out, sizes[n])
    print("all files downloaded ->", out)
