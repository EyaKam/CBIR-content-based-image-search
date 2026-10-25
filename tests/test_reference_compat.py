"""The server answers like the reference project (VGG16 + Elasticsearch) for the same queries.

The VGG network is replaced by a stub (the test index holds hand-made vectors), so what is
checked here is the search logic, the scoring, the threshold and the JSON contract.
"""
import base64
import importlib
import io
import json
import sys

import numpy as np
import pytest
from PIL import Image

N_PER_CLASS = 10
CLASSES = ["sky", "forest", "car"]
DIM = 128


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w")
    img_root, idx_dir = root / "Objects", root / "index"
    rng = np.random.default_rng(0)
    centers = unit(rng.normal(size=(3, DIM)))
    paths, labels, vecs = [], [], []
    for c, name in enumerate(CLASSES):
        (img_root / name).mkdir(parents=True)
        for k in range(N_PER_CLASS):
            p = f"{name}/{name}_{k:02d}.jpg"
            if not (name == "car" and k == 0):               # one image only exists on Flickr
                Image.new("RGB", (40, 30), (c * 80, 50, 90)).save(img_root / p)
            paths.append(p)
            labels.append(name)
            vecs.append(unit(centers[c] + 0.35 * unit(rng.normal(size=DIM))))
    X = np.stack(vecs).astype(np.float32)
    idx_dir.mkdir()
    np.save(idx_dir / "vgg.npy", X)
    from cbir.indexing import estimate_scales
    tags = {"0": "sky clouds blue", "1": "sky sunset", "20": "car red street"}
    flickr = {"20": {"id": "777", "tags": "car red street", "flickr_secret": "abc",
                     "flickr_server": "55", "flickr_farm": "6"}}
    meta = {"image_root": str(img_root), "n": len(paths), "descriptors": ["vgg"],
            "dims": {"vgg": DIM}, "scales": {"vgg": estimate_scales(X)}, "paths": paths,
            "labels": labels, "tags": tags, "flickr": flickr}
    (idx_dir / "meta.json").write_text(json.dumps(meta))
    return {"X": X, "labels": labels, "paths": paths, "img_root": img_root, "idx_dir": idx_dir,
            "query": unit(centers[0] + 0.2 * unit(rng.normal(size=DIM))).astype(np.float32)}


@pytest.fixture(scope="module")
def client(world):
    import os
    # the tests use their own tiny base: no Elasticsearch (in-process BM25) and no 1M base
    os.environ.update(CBIR_INDEX_DIR=str(world["idx_dir"]), CBIR_IMAGE_ROOT=str(world["img_root"]),
                      CBIR_WARMUP="0", CBIR_ES_URL="http://127.0.0.1:9",
                      CBIR_INDEX_1M_DIR=str(world["idx_dir"] / "no_1m_base"))
    from cbir import config
    importlib.reload(config)
    from cbir import descriptors
    qv = world["query"]
    descriptors.vgg_desc = lambda rgb: qv
    descriptors.EXTRACTORS["vgg"] = lambda rgb: qv
    sys.modules.pop("server", None)
    import server
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 200, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def es_reference(X, q, size, threshold):
    """What the reference project asks Elasticsearch: script_score = cosineSimilarity + 1.0,
    min_score = threshold + 1.0, top `size`."""
    scores = np.array([float(x @ q / (np.linalg.norm(x) * np.linalg.norm(q))) + 1.0 for x in X])
    order = np.argsort(-scores, kind="stable")[:size]
    return [(int(i), float(scores[i])) for i in order if scores[i] >= threshold + 1.0]


@pytest.mark.parametrize("number,threshold", [(5, 0.2), (30, 0.2), (30, 0.9), (3, 0.0)])
def test_search_by_image_matches_the_elasticsearch_formula(client, world, monkeypatch, number, threshold):
    from cbir import config
    monkeypatch.setattr(config, "VGG_THRESHOLD", threshold)
    r = client.post("/search_by_image/", json={"img": png_b64(), "number": number})
    assert r.status_code == 200
    hits = r.json()["resulttype"]["hits"]
    expected = es_reference(world["X"], world["query"], number, threshold)
    assert [int(h["_id"]) for h in hits["hits"]] == [i for i, _ in expected]
    assert [h["_score"] for h in hits["hits"]] == pytest.approx([s for _, s in expected])
    assert hits["total"]["value"] == len(expected)
    assert hits["max_score"] == pytest.approx(expected[0][1])
    # `path` is a URL the reference client can hand to st.image(); `file` is the relative path
    assert all(h["_source"]["path"].endswith(f"/img/{h['_id']}") and h["_source"]["path"].startswith("http")
               and h["_source"]["file"] == world["paths"][int(h["_id"])] for h in hits["hits"])


def test_default_threshold_keeps_only_the_query_class(client, world):
    hits = client.post("/search_by_image", json={"img": png_b64(), "number": 30}).json()
    labels = {h["_source"]["label"] for h in hits["resulttype"]["hits"]["hits"]}
    assert labels == {"sky"}


def test_search_by_text_match_and_fuzzy(client, world):
    r = client.post("/search_by_text", json={"tags": "sky", "type": "match", "number": 4}).json()
    hits = r["resulttype"]["hits"]
    assert hits["total"]["value"] == N_PER_CLASS and len(hits["hits"]) == 4
    # images 0 and 1 carry extra tags ("sky clouds blue", "sky sunset"): same tf, longer doc => lower
    assert all(h["_source"]["label"] == "sky" for h in hits["hits"])
    assert [h["_score"] for h in hits["hits"]] == sorted((h["_score"] for h in hits["hits"]), reverse=True)
    fz = client.post("/search_by_text/", json={"tags": "forst", "type": "fuzzy", "number": 3}).json()
    assert fz["resulttype"]["hits"]["total"]["value"] == N_PER_CLASS
    assert client.post("/search_by_text", json={"tags": "red car", "type": "fuzzy", "number": 3}
                       ).json()["resulttype"]["hits"]["total"]["value"] == 0
    assert client.post("/search_by_text", json={"tags": "x", "type": "bad", "number": 3}).status_code == 400


def test_text_hits_expose_the_flickr_fields_the_reference_client_reads(client):
    r = client.post("/search_by_text", json={"tags": "street", "type": "match", "number": 5}).json()
    h = r["resulttype"]["hits"]["hits"][0]
    for key in ("flickr_farm", "flickr_server", "flickr_secret", "id"):    # st_function.display_results
        assert key in h["_source"]
    assert {"took", "timed_out", "hits"} <= set(r["resulttype"])


def test_flickr_only_image_redirects_to_flickr(client):
    r = client.get("/img/20", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "https://farm6.staticflickr.com/55/777_abc.jpg"
    assert client.get("/img/21").status_code == 200


def test_ui_route_with_vgg_only_uses_the_same_ranking_and_threshold(client, world):
    r = client.post("/api/search", data={"image_id": "1", "descs": "vgg", "k": "30"}).json()
    ids = [x["id"] for x in r["results"]]
    expected = [i for i, _ in es_reference(world["X"], world["X"][1], 31, 0.2) if i != 1]
    assert ids == expected
    assert all("score" in x for x in r["results"]) and r["info"]["threshold"] == 0.2
    brute = client.post("/api/search", data={"image_id": "1", "descs": "vgg", "k": "30",
                                             "method": "brute"}).json()
    assert [x["id"] for x in brute["results"]] == ids            # VP-tree is exact
    nothr = client.post("/api/search", data={"image_id": "1", "descs": "vgg", "k": "30",
                                             "threshold": "-1"}).json()
    assert len(nothr["results"]) == 29


def test_ui_keyword_search_is_ranked_by_bm25(client):
    r = client.post("/api/search", data={"keywords": "sky blue", "kw_mode": "or", "k": "5"}).json()
    assert r["info"]["method"] == "text" and r["results"][0]["id"] == 0   # matches both words
    both = client.post("/api/search", data={"keywords": "sky blue", "kw_mode": "and"}).json()
    assert [x["id"] for x in both["results"]] == [0]
