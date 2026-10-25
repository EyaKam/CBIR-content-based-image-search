"""Build an index from PRECOMPUTED descriptors (e.g. the MPEG-7 descriptors of the 1M MIRFLICKR base).

    python scripts/import_features.py --index index_mirflickr --image-root D:/mirflickr \
        --paths filelist.txt --feat scalable_color=sc.npy edge_histogram=eh.csv \
        --tags tags.csv --metric scalable_color=l1 edge_histogram=l1

* filelist.txt : one image path per line (relative to --image-root), same order as the feature rows
* .npy / .csv  : one row per image (csv: comma or whitespace separated numbers, no header)
* --tags       : optional keywords: "relative/path.jpg,tag1 tag2" lines, or the Flickr metadata CSV
Then start the server with CBIR_INDEX_DIR=index_mirflickr CBIR_IMAGE_ROOT=D:/mirflickr
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbir.indexing import estimate_scales, load_tags_file


def load_matrix(p):
    p = Path(p)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    return np.loadtxt(p, delimiter="," if "," in p.read_text(errors="ignore")[:2000] else None,
                      dtype=np.float32, ndmin=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--feat", nargs="+", required=True, help="name=file")
    ap.add_argument("--tags", default="")
    a = ap.parse_args()

    paths = [l.strip().replace("\\", "/") for l in Path(a.paths).read_text().splitlines() if l.strip()]
    out = Path(a.index)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"image_root": a.image_root, "n": len(paths), "descriptors": [], "dims": {}, "scales": {},
            "paths": paths, "labels": ["unlabeled"] * len(paths), "tags": {}, "flickr": {}}
    for spec in a.feat:
        name, f = spec.split("=", 1)
        X = load_matrix(f)
        if len(X) != len(paths):
            raise SystemExit(f"{name}: {len(X)} rows but {len(paths)} paths")
        np.save(out / f"{name}.npy", X)
        meta["descriptors"].append(name)
        meta["dims"][name] = int(X.shape[1])
        meta["scales"][name] = estimate_scales(X)
        print(f"{name}: {X.shape}")
    if a.tags:
        tags, flickr = load_tags_file(a.tags, paths)
        meta["tags"] = {str(k): v for k, v in tags.items()}
        meta["flickr"] = {str(k): v for k, v in flickr.items()}
        print(f"keywords for {len(tags)} images ({len(flickr)} with Flickr metadata)")
    (out / "meta.json").write_text(json.dumps(meta))
    print("index written to", out)
