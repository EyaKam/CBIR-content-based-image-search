"""Build the 1M MIRFLICKR index straight from the downloaded zip files (nothing is extracted).

    python scripts/import_mirflickr.py --raw data/mirflickr_raw --index index_1m
    python scripts/import_mirflickr.py --limit 20000 --index index_20k      # quick test

Reads, for image id i (0..999999):
    features_edgehistogram.zip       features_edgehistogram/<i//10000>/<i>.txt        -> ehd.npy (150 dims)
    features_homogeneoustexture.zip  features_homogeneoustexture/<i//10000>/<i>.txt   -> htd.npy (43 dims)
    tags.zip                         tags/<i//10000>/<i>.txt  (one tag per line)      -> tags.csv
and writes meta.json. tags.csv (id,tags) is what Logstash loads into Elasticsearch.
Thumbnail im<i+1>.jpg is served from thumbs*.zip by the server.
"""
import argparse
import csv
import json
import re
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbir.indexing import estimate_scales

ZIPS = {"ehd": ("features_edgehistogram.zip", "features_edgehistogram"),
        "htd": ("features_homogeneoustexture.zip", "features_homogeneoustexture"),
        "tags": ("tags.zip", "tags")}
CHUNK = 10000
_zips = {}


def _open(raw, key):
    if key not in _zips:
        _zips[key] = zipfile.ZipFile(Path(raw) / ZIPS[key][0])
    return _zips[key]


def _chunk(args):
    raw, start, stop, names = args
    out = {}
    for key in names:
        z, folder = _open(raw, key), ZIPS[key][1]
        rows = []
        for i in range(start, stop):
            data = z.read(f"{folder}/{i // CHUNK}/{i}.txt").decode("utf-8", "replace")
            if key == "tags":
                rows.append(" ".join(t.strip() for t in data.splitlines() if t.strip()))
            else:
                rows.append(np.array(data.split(), dtype=np.float32))
        out[key] = rows if key == "tags" else np.stack(rows)
    return start, out


def clean_tag_text(t):
    return re.sub(r"[\",\r\n]+", " ", t).strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/mirflickr_raw")
    ap.add_argument("--index", default="index_1m")
    ap.add_argument("--limit", type=int, default=1_000_000)
    ap.add_argument("--desc", nargs="+", default=["ehd", "htd"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    raw, out, n = Path(a.raw), Path(a.index), a.limit
    out.mkdir(parents=True, exist_ok=True)
    names = list(a.desc) + ["tags"]
    jobs = [(str(raw), s, min(s + CHUNK, n), names) for s in range(0, n, CHUNK)]
    t0 = time.time()
    mats = {d: [None] * len(jobs) for d in a.desc}
    tags = [None] * len(jobs)
    with Pool(a.workers) as pool:
        for done, (start, res) in enumerate(pool.imap_unordered(_chunk, jobs), 1):
            j = start // CHUNK
            for d in a.desc:
                mats[d][j] = res[d]
            tags[j] = res["tags"]
            print(f"  {done}/{len(jobs)} chunks  ({time.time() - t0:.0f}s)", flush=True)

    meta = {"image_root": str(raw), "n": n, "base": "mirflickr", "thumbs": True,
            "descriptors": list(a.desc), "dims": {}, "scales": {}, "paths": None,
            "labels": None, "tags": {}, "flickr": {}}
    for d in a.desc:
        X = np.concatenate(mats[d]).astype(np.float32)
        assert len(X) == n, (d, X.shape)
        np.save(out / f"{d}.npy", X)
        meta["dims"][d] = int(X.shape[1])
        meta["scales"][d] = estimate_scales(X)
        print(f"[index] {d}: {X.shape} scales={meta['scales'][d]}")
    with open(out / "tags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "tags"])
        i = 0
        for chunk in tags:
            for t in chunk:
                t = clean_tag_text(t)
                if t:                    # images without tags get no document (Logstash's csv
                    w.writerow([i, t])   # filter rejects empty fields)
                i += 1
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"[index] done in {time.time() - t0:.0f}s -> {out}")
