"""Offline phase: scan the image folder, compute signatures once, save the index."""
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import config as C
from . import descriptors as D
from . import distances as DI

# columns of the Flickr metadata CSV used by the reference project (photo_metadata.csv)
FLICKR_COLUMNS = ["id", "userid", "title", "tags", "latitude", "longitude", "views", "date_taken",
                  "date_uploaded", "accuracy", "flickr_secret", "flickr_server", "flickr_farm",
                  "x", "y", "z"]


def scan_images(root):
    """Return (relative_paths, labels). Label = name of the first sub-folder."""
    root = Path(root)
    paths, labels = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in C.IMAGE_EXTS and p.is_file():
            rel = p.relative_to(root)
            paths.append(rel.as_posix())
            labels.append(rel.parts[0] if len(rel.parts) > 1 else "unlabeled")
    return paths, labels


def load_tags_file(tags_file, paths):
    """Read keywords/metadata for the indexed images.

    Two formats are accepted:
      * "relative/path.jpg,tag1 tag2 tag3"           one image per line, no header
      * the reference project's Flickr CSV (header with id, tags, flickr_secret, ...): a row is
        attached to the image whose file name (without extension) equals the row's `id`.
    Returns (tags {index: str}, flickr {index: {column: value}}).
    """
    tags, flickr = {}, {}
    lookup = {p: i for i, p in enumerate(paths)}
    stems = {}
    for i, p in enumerate(paths):
        stems.setdefault(Path(p).stem, i)
    text = Path(tags_file).read_text(encoding="utf-8", errors="replace")
    first = text.split("\n", 1)[0].lower()
    if first.startswith("id,") and "tags" in first:           # Flickr metadata CSV
        for row in csv.DictReader(text.splitlines()):
            i = stems.get(str(row.get("id", "")).strip())
            if i is None:
                continue
            flickr[i] = {k: (v or "").strip() for k, v in row.items() if k}
            tags[i] = flickr[i].get("tags", "")
    else:
        for line in text.splitlines():
            if "," in line:
                p, t = line.split(",", 1)
                i = lookup.get(p.strip())
                if i is not None:
                    tags[i] = t.strip()
    return tags, flickr


def _work(args):
    """Classic descriptors of one image (also tells us whether the file is readable)."""
    root, rel, names = args
    rgb = D.load_rgb(Path(root) / rel)
    if rgb is None:
        return None
    return D.extract(rgb, names)


def estimate_scales(X, n_pairs=3000, seed=0):
    """Mean distance between random pairs, per metric -> used to balance descriptors."""
    rng = np.random.default_rng(seed)
    n = len(X)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    out = {}
    for m in ("l1", "l2", "chi2"):
        fn = DI.METRICS[m]
        with np.errstate(invalid="ignore"):           # chi2 of vectors with negative values
            vals = [fn(X[i], X[j][None, :])[0] for i, j in zip(a[:600], b[:600]) if i != j]
            mean = float(np.nanmean(vals)) if len(vals) else 1.0
        out[m] = mean if np.isfinite(mean) and mean > 0 else 1.0
    return out


def build_index(image_root=C.IMAGE_ROOT, index_dir=C.INDEX_DIR,
                names=("hsv", "lbp", "hog"), workers=None, tags_file=C.TAGS_FILE,
                batch_size=32):
    image_root, index_dir = Path(image_root), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = scan_images(image_root)
    if not paths:
        raise SystemExit(f"No images found under {image_root}")
    print(f"[index] {len(paths)} images, {len(set(labels))} classes, descriptors={list(names)}")

    t0 = time.time()
    classic = [n for n in names if n != "vgg"]
    jobs = [(str(image_root), p, classic) for p in paths]
    if workers == 1 or len(paths) < 64:
        results = [_work(j) for j in jobs]
    else:
        with Pool(workers or os.cpu_count()) as pool:
            results = []
            for i, r in enumerate(pool.imap(_work, jobs, chunksize=16)):
                results.append(r)
                if i % 500 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)

    keep = [i for i, r in enumerate(results) if r is not None]
    if len(keep) < len(paths):
        print(f"[index] skipped {len(paths) - len(keep)} unreadable images")
    kept_paths = [paths[i] for i in keep]
    kept_labels = [labels[i] for i in keep]

    matrices = {n: np.stack([results[i][n] for i in keep]).astype(np.float32) for n in classic}

    if "vgg" in names:                       # deep features: batches, one process (TensorFlow)
        feats = []
        for s in range(0, len(kept_paths), batch_size):
            imgs = [D.load_rgb(image_root / p) for p in kept_paths[s:s + batch_size]]
            feats.append(D.vgg_batch(imgs, batch_size))
            print(f"  vgg {min(s + batch_size, len(kept_paths))}/{len(kept_paths)}", flush=True)
        matrices["vgg"] = np.concatenate(feats)

    meta = {"image_root": str(image_root), "n": len(kept_paths), "descriptors": list(names),
            "dims": {}, "scales": {}, "paths": kept_paths, "labels": kept_labels,
            "tags": {}, "flickr": {}}
    for n in names:
        X = matrices[n]
        np.save(index_dir / f"{n}.npy", X)
        meta["dims"][n] = int(X.shape[1])
        meta["scales"][n] = estimate_scales(X)
        print(f"[index] {n}: {X.shape}  scales={meta['scales'][n]}")

    if tags_file and Path(tags_file).exists():
        tags, flickr = load_tags_file(tags_file, kept_paths)
        meta["tags"] = {str(i): t for i, t in tags.items()}
        meta["flickr"] = {str(i): d for i, d in flickr.items()}
        print(f"[index] keywords loaded for {len(tags)} images ({len(flickr)} with Flickr metadata)")

    (index_dir / "meta.json").write_text(json.dumps(meta))
    print(f"[index] done in {time.time() - t0:.1f}s -> {index_dir}")
    return meta
