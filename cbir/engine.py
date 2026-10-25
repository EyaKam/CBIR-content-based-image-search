"""Online phase: query -> signature -> distances -> ranked results."""
import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from . import descriptors as D
from . import distances as DI
from .textsearch import TextIndex, analyze
from .vptree import VPTree


class _Lazy:
    """Read-only sequence computed on the fly (1M paths/labels are never stored in RAM)."""

    def __init__(self, n, fn):
        self.n, self.fn = n, fn

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.fn(int(i))


class SearchEngine:
    """One engine per base. `base` is "objects" or "mirflickr" (see config.BASES).

    Signatures: index_dir/<descriptor>.npy. Keywords and per-image metadata: Elasticsearch index
    `es_index` when `es` is given and reachable, otherwise the in-memory BM25 index (small bases).
    """

    def __init__(self, index_dir=C.INDEX_DIR, image_root=None, base="objects", es=None,
                 es_index=None):
        index_dir = Path(index_dir)
        self.base = base
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.image_root = Path(image_root or self.meta["image_root"])
        self.n = self.meta["n"]
        self.thumbs = bool(self.meta.get("thumbs"))     # 1M base: 64x64 thumbnails in zip files
        if self.meta.get("paths") is not None:
            self.paths = self.meta["paths"]
        else:
            self.paths = _Lazy(self.n, lambda i: f"thumbnails/{i // 10000 + 1}/im{i + 1}.jpg")
        if self.meta.get("labels") is not None:
            self.labels = self.meta["labels"]
            self.classes = sorted(set(self.labels))
        else:
            self.labels = _Lazy(self.n, lambda i: "unlabeled")
            self.classes = []
        self.descs = self.meta["descriptors"]
        self.scales = self.meta["scales"]
        self.flickr = self.meta.get("flickr", {})
        # small indexes live in RAM (fast); huge ones (1M images) stay memory-mapped
        self.X = {}
        for d in self.descs:
            f = index_dir / f"{d}.npy"
            self.X[d] = np.load(f, mmap_mode="r" if f.stat().st_size > 1.5e9 else None)
        self._all = np.arange(self.n)
        self._trees = {}
        self._vgg_norms = None
        self.es, self.es_index = None, es_index or base
        if es is not None and es.alive() and es.count(self.es_index) > 0:
            self.es = es
        self.text = None
        if self.es is None and self.n <= 200000:       # fallback: BM25 in process
            self.text = TextIndex([self.doc_tags(i) for i in range(self.n)])

    # ------------------------------------------------------------------ text
    def doc_tags(self, i):
        """The searchable `tags` of image i: its keywords, else its class folder name."""
        t = self.meta.get("tags", {}).get(str(i), "")
        if t:
            return t
        lab = self.labels[i]
        return "" if lab == "unlabeled" else lab.replace("_", " ").replace("-", " ")

    def text_hits(self, keywords, mode="and", kind="match"):
        """(ids, scores, total) of the images matching the keywords, best first; None if no
        query. Answered by Elasticsearch when available."""
        if kind == "fuzzy":
            if not keywords.strip():
                return None
        elif not analyze(keywords):
            return None
        if self.es is not None:
            return self.es.search_tags(self.es_index, keywords, kind, mode)
        if self.text is None:
            raise ValueError("keyword search needs Elasticsearch for this base "
                             "(start it and run the ingestion, see README)")
        ids, sc = self.text.search(keywords, kind, operator=mode)
        return ids, sc, len(ids)

    # -------------------------------------------------------------- distances
    def _setup(self, descs, weights, metrics):
        descs = [d for d in descs if d in self.descs]
        if not descs:
            raise ValueError(f"no valid descriptor, available: {self.descs}")
        w = np.array([(weights or {}).get(d, 1.0) for d in descs], dtype=np.float64)
        w = w / w.sum()
        m = [(metrics or {}).get(d, C.DEFAULT_METRIC.get(d, "l2")) for d in descs]
        return descs, w, m

    def _combined(self, qfeats, descs, w, m):
        """Return f(idxs) -> weighted sum of scale-normalised distances."""
        def f(idxs):
            full = idxs is self._all
            idxs = np.asarray(idxs)
            tot = np.zeros(len(idxs), dtype=np.float64)
            for d, wi, mi in zip(descs, w, m):
                Xs = self.X[d] if full else np.asarray(self.X[d][idxs])
                scale = self.scales[d].get(mi, self.scales[d]["l2"])
                tot += wi * DI.METRICS[mi](qfeats[d], np.asarray(Xs)) / scale
            return tot
        return f

    def _tree(self, descs, w, m):
        key = (tuple(descs), tuple(np.round(w, 4)), tuple(m))
        if key not in self._trees:
            t0 = time.time()

            def dist_db(i, idxs):
                q = {d: np.asarray(self.X[d][i]) for d in descs}
                return self._combined(q, descs, w, m)(idxs)

            self._trees[key] = VPTree(self.n, dist_db)
            print(f"[vptree] built for {key} in {time.time() - t0:.1f}s")
        return self._trees[key]

    # ----------------------------------------------------------------- cosine
    def _cosine(self, qv, i):
        x = np.asarray(self.X["vgg"][i], dtype=np.float64)
        q = np.asarray(qv, dtype=np.float64)
        return float(x @ q / (np.linalg.norm(x) * np.linalg.norm(q) + 1e-12))

    def cosine_topk(self, qvec, k, threshold=None):
        """Exact cosine ranking against all VGG vectors (what the reference project asks
        Elasticsearch for): -> [(image id, cosine)] best first, cosine >= threshold."""
        if "vgg" not in self.X:
            raise ValueError("this index has no 'vgg' descriptor (build it with --desc vgg)")
        if self._vgg_norms is None:
            self._vgg_norms = np.linalg.norm(np.asarray(self.X["vgg"]), axis=1) + 1e-12
        q = np.asarray(qvec, dtype=np.float32)
        sims = (np.asarray(self.X["vgg"]) @ q) / (self._vgg_norms * (np.linalg.norm(q) + 1e-12))
        order = np.argsort(-sims, kind="stable")[:max(k, 0)]
        return [(int(i), float(sims[i])) for i in order
                if threshold is None or sims[i] >= threshold]

    # ----------------------------------------------------------------- search
    def search(self, keywords="", kw_mode="and", combine="and", query_id=None,
               query_rgb=None, descs=("hsv", "lbp", "hog"), weights=None, metrics=None,
               k=20, method="vptree", exclude_query=False, threshold="auto",
               text_type="match"):
        """Keyword and/or image query.

        threshold: minimum cosine similarity, only used when the query is VGG-only. "auto" uses
        config.VGG_THRESHOLD (the reference project's value); None disables it.
        """
        t0 = time.time()
        kw = self.text_hits(keywords, kw_mode, text_type) if keywords else None
        kw_ids = None if kw is None else kw[0]
        has_img = query_id is not None or query_rgb is not None
        info = {"method": method, "n_dist": 0, "base": self.base,
                "text_engine": "elasticsearch" if self.es is not None else "in-process"}

        # ---- keyword-only query: ranked by BM25
        if not has_img:
            ids, sc, total = kw if kw is not None else (np.array([], dtype=np.int64),
                                                        np.array([]), 0)
            info["method"] = "text"
            rows = [self._row(int(i), 0.0, score=float(s)) for i, s in zip(ids[:k], sc[:k])]
            return self._pack(rows, t0, info, total=total)

        descs, w, m = self._setup(descs, weights, metrics)
        if query_id is not None:
            q = {d: np.asarray(self.X[d][query_id]) for d in descs}
        elif self.thumbs:
            raise ValueError("this base only has precomputed MPEG-7 signatures: query with an "
                             "image of the base (click one), not an uploaded file")
        else:
            q = D.extract(query_rgb, descs)
        f = self._combined(q, descs, w, m)
        kk = k + 1 if (exclude_query and query_id is not None) else k

        use_tree = (method == "vptree" and kw_ids is None
                    and all(x in DI.TRUE_METRICS for x in m))
        if use_tree:
            res, n_dist = self._tree(descs, w, m).knn(f, kk)
            info["n_dist"] = n_dist
        else:
            info["method"] = "brute"
            cand = self._all if kw_ids is None else np.sort(kw_ids)
            if kw_ids is not None and combine == "or":
                # union: half of the slots for the best visual matches, half for the
                # keyword matches (ranked by visual similarity), merged by distance
                half = (kk + 1) // 2
                dall = f(self._all)
                top = np.argsort(dall)[:half]
                kw_top = kw_ids[np.argsort(dall[kw_ids])[:kk - half]]
                cand = np.union1d(top, kw_top)
            dist = f(cand)
            order = np.argsort(dist)[:kk]
            res = [(float(dist[o]), int(cand[o])) for o in order]
            info["n_dist"] = len(cand)

        if exclude_query and query_id is not None:
            res = [r for r in res if r[1] != query_id][:k]

        deep_only = descs == ["vgg"]
        thr = (C.VGG_THRESHOLD if threshold == "auto" else threshold) if deep_only else None
        rows = []
        for d, i in res:
            row = self._row(i, d)
            if deep_only:
                cos = self._cosine(q["vgg"], i)
                if thr is not None and cos < thr:
                    continue
                row["score"] = round(cos + 1.0, 6)     # Elasticsearch-style score: cosine + 1
            rows.append(row)
        info["descs"] = descs
        info["metrics"] = m
        if deep_only and thr is not None:
            info["threshold"] = thr
        return self._pack(rows, t0, info)

    def _row(self, i, d, score=None):
        row = {"id": int(i), "path": self.paths[i], "label": self.labels[i],
               "distance": round(float(d), 5)}
        if score is not None:
            row["score"] = round(score, 6)
        return row

    def _enrich(self, rows):
        """Add the tags (and, for the 1M base, a readable label) from Elasticsearch."""
        if not rows:
            return
        if self.es is not None:
            try:
                docs = self.es.mget(self.es_index, [r["id"] for r in rows])
            except Exception:
                docs = {}
            for r in rows:
                d = docs.get(r["id"], {})
                t = d.get("tags", "")
                r["tags"] = " ".join(t) if isinstance(t, list) else t
                if r["label"] == "unlabeled" and r["tags"]:
                    r["label"] = " ".join(r["tags"].split()[:3])
        else:
            for r in rows:
                r["tags"] = self.doc_tags(r["id"]) if self.n <= 200000 else ""

    def _pack(self, rows, t0, info, total=None):
        self._enrich(rows)
        info["time_ms"] = round((time.time() - t0) * 1000, 2)
        info["n_results"] = len(rows)
        if total is not None:
            info["total_matches"] = int(total)
        return {"info": info, "results": rows}

    # ------------------------------------------- reference-project compatible hits
    def source(self, i, base_url=""):
        """`_source` document of image i (Flickr metadata fields when they were indexed).

        `path` is what the reference client passes to st.image(): a URL it can open (this
        server's /img/<id>); the file's path relative to the image folder is in `file`.
        """
        src = dict(self.flickr.get(str(i), {}))
        src.setdefault("tags", self.doc_tags(i))
        src.update({"image_id": int(i), "path": f"{base_url}/img/{i}", "file": self.paths[i],
                    "label": self.labels[i]})
        return src

    def es_text_response(self, query, kind, number, base_url=""):
        """Same JSON as the reference project's /search_by_text (Elasticsearch `hits`)."""
        t0 = time.time()
        kw = self.text_hits(query, "or", kind)
        ids, sc, total = kw if kw is not None else (np.array([], dtype=np.int64),
                                                     np.array([]), 0)
        hits = [{"_index": "flickrphotos", "_id": str(int(i)), "_score": float(s),
                 "_source": self.source(int(i), base_url)}
                for i, s in zip(ids[:max(number, 0)], sc[:max(number, 0)])]
        return self._es_envelope(t0, total, hits, float(sc[0]) if total else None)

    def es_image_response(self, query_rgb, number, threshold=None, base_url=""):
        """Same JSON as the reference project's /search_by_image/ (cosine + 1 scores)."""
        t0 = time.time()
        thr = C.VGG_THRESHOLD if threshold is None else threshold
        q = D.vgg_desc(query_rgb)
        found = self.cosine_topk(q, number, thr)
        hits = [{"_index": "images", "_id": str(i), "_score": cos + 1.0,
                 "_source": {"path": f"{base_url}/img/{i}", "file": self.paths[i],
                             "image_id": i, "label": self.labels[i]}} for i, cos in found]
        return self._es_envelope(t0, len(hits), hits, hits[0]["_score"] if hits else None)

    @staticmethod
    def _es_envelope(t0, total, hits, max_score):
        capped = min(total, C.MAX_TOTAL_HITS)
        return {"resulttype": {
            "took": int((time.time() - t0) * 1000), "timed_out": False,
            "hits": {"total": {"value": capped, "relation": "eq" if total <= capped else "gte"},
                     "max_score": max_score, "hits": hits}}}
