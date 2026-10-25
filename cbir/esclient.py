"""Minimal Elasticsearch client (plain HTTP): keyword search and metadata lookup.

Elasticsearch holds, for every image of a base, its id, file path, label and tags. The tags are a
`text` field: `match` (BM25) and `fuzzy` queries are answered by Elasticsearch itself.
"""
import numpy as np
import requests

MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "id": {"type": "integer"},
        "path": {"type": "keyword"},
        "label": {"type": "keyword"},
        "tags": {"type": "text"},
    }},
}
MAX_WINDOW = 10000          # Elasticsearch's default index.max_result_window


class ES:
    def __init__(self, url, timeout=10):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _req(self, method, path, timeout=None, **kw):
        return requests.request(method, f"{self.url}{path}", timeout=timeout or self.timeout, **kw)

    def alive(self):
        try:
            return self._req("GET", "/", timeout=2).ok
        except requests.RequestException:
            return False

    def count(self, index):
        try:
            r = self._req("GET", f"/{index}/_count")
            return int(r.json().get("count", 0)) if r.ok else 0
        except (requests.RequestException, ValueError):
            return 0

    def create_index(self, index, recreate=False):
        if recreate:
            self._req("DELETE", f"/{index}")
        r = self._req("PUT", f"/{index}", json=MAPPING)
        if not r.ok and "resource_already_exists" not in r.text:
            raise RuntimeError(f"cannot create index {index}: {r.text}")

    def search_tags(self, index, query, kind="match", operator="or", size=MAX_WINDOW):
        """-> (ids, scores, total) best first. kind: match (BM25) | fuzzy (one term)."""
        if kind == "fuzzy":
            q = {"fuzzy": {"tags": {"value": query}}}
        else:
            q = {"match": {"tags": {"query": query, "operator": operator}}}
        body = {"query": q, "size": min(size, MAX_WINDOW), "_source": False,
                "track_total_hits": True}
        r = self._req("POST", f"/{index}/_search", json=body)
        r.raise_for_status()
        hits = r.json()["hits"]
        ids = np.array([int(h["_id"]) for h in hits["hits"]], dtype=np.int64)
        scores = np.array([h["_score"] for h in hits["hits"]], dtype=np.float64)
        return ids, scores, int(hits["total"]["value"])

    def mget(self, index, ids):
        """{id: _source} for the given image ids (missing ids are skipped)."""
        if not len(ids):
            return {}
        r = self._req("POST", f"/{index}/_mget", json={"ids": [str(int(i)) for i in ids]})
        r.raise_for_status()
        return {int(d["_id"]): d["_source"] for d in r.json()["docs"] if d.get("found")}
