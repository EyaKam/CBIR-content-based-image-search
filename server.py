"""FastAPI server.   uvicorn server:app --host 0.0.0.0 --port 8000

Two bases (config.BASES): "objects" (our own descriptors) and "mirflickr" (1M images, MPEG-7
descriptors, 64x64 thumbnails read from the zip files). A descriptor belongs to exactly one base,
so the descriptors ticked on the web page decide which base is searched.
"""
import base64
import json
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from cbir import config as C
from cbir import descriptors as D
from cbir.engine import SearchEngine
from cbir.esclient import ES
from cbir.thumbs import ThumbStore

app = FastAPI(title="CBIR - Indexation et recherche d'images par le contenu")
STATIC = Path(__file__).parent / "static"

es = ES(C.ES_URL)
engines = {}
for _name, _b in C.BASES.items():
    if (_b["index_dir"] / "meta.json").exists():
        _root = _b["image_root"] if Path(_b["image_root"]).exists() else None
        engines[_name] = SearchEngine(_b["index_dir"], image_root=_root, base=_name, es=es,
                                      es_index=_b["es_index"])
        print(f"[server] base {_name}: {engines[_name].n} images, descriptors {engines[_name].descs}, "
              f"keywords via {'Elasticsearch' if engines[_name].es else 'in-process BM25'}")
if not engines:
    raise SystemExit("No index found. Run:  python scripts/build_index.py --root <image folder>")

thumb_store = None
if "mirflickr" in engines:
    try:
        thumb_store = ThumbStore(C.MIRFLICKR_RAW, C.THUMB_ZIPS)
    except Exception as e:
        print(f"[server] MIRFLICKR thumbnails unavailable: {e}")
_thumb_cache = {}


def _warm_up():
    try:
        D.get_vgg()
    except Exception as e:                      # e.g. tensorflow missing: searches will report it
        print(f"[server] VGG16 could not be loaded: {e}")


if "vgg" in getattr(engines.get("objects"), "descs", []) and os.environ.get("CBIR_WARMUP", "1") != "0":
    threading.Thread(target=_warm_up, daemon=True).start()   # first query is not slow


def engine_of(base):
    if base not in engines:
        raise HTTPException(404, f"base '{base}' is not loaded (available: {list(engines)})")
    return engines[base]


def resolve_base(descs, base):
    """The base is given by the ticked descriptors; descriptors of two bases cannot be mixed."""
    unknown = [d for d in descs if d not in C.DESCRIPTORS]
    if unknown:
        raise HTTPException(400, f"unknown descriptor(s): {unknown}")
    used = {C.DESCRIPTORS[d]["base"] for d in descs}
    if len(used) > 1:
        raise HTTPException(400, "descriptors of different bases cannot be combined: "
                                 f"{sorted(descs)} belong to {sorted(used)}")
    if used:
        b = used.pop()
        if base and base != b:
            raise HTTPException(400, f"descriptor(s) {descs} belong to base '{b}', not '{base}'")
        return b
    return base or ("objects" if "objects" in engines else next(iter(engines)))


@app.get("/api/info")
def info():
    bases = {n: {"title": C.BASES[n]["title"], "n_images": e.n, "classes": e.classes,
                 "descriptors": e.descs, "thumbs_only": e.thumbs,
                 "keywords": "elasticsearch" if e.es is not None else "in-process"}
             for n, e in engines.items()}
    descs = {d: C.DESCRIPTORS[d] for e in engines.values() for d in e.descs if d in C.DESCRIPTORS}
    first = engines.get("objects") or next(iter(engines.values()))
    return {"bases": bases, "descriptors": descs, "default_metric": C.DEFAULT_METRIC,
            "n_images": first.n, "classes": first.classes}


@app.get("/api/browse")
def browse(label: str = "", n: int = 24, seed: Optional[int] = None, base: str = "objects"):
    eng = engine_of(base)
    rng = np.random.default_rng(seed)
    if label:
        pool = np.array([i for i, l in enumerate(eng.labels) if l == label])
    else:
        pool = None
    if pool is None:
        ids = rng.choice(eng.n, min(n, eng.n), replace=False).tolist()
    else:
        ids = rng.choice(pool, min(n, len(pool)), replace=False).tolist() if len(pool) else []
    rows = [{"id": i, "label": eng.labels[i], "path": eng.paths[i], "base": base} for i in ids]
    eng._enrich(rows)
    return rows


@app.post("/api/search")
async def search(
    keywords: str = Form(""),
    kw_mode: str = Form("and"),            # words: "and" | "or"
    combine: str = Form("and"),            # words ET/OU image: "and" | "or"
    image_id: Optional[int] = Form(None),  # query = image already in the base
    file: Optional[UploadFile] = File(None),  # query = uploaded image
    base: str = Form(""),                  # objects | mirflickr (deduced from the descriptors)
    descs: str = Form("hsv,lbp,hog"),
    weights: str = Form(""),               # JSON {"hsv": 1, "lbp": 2}
    metric: str = Form("default"),         # default | l1 | l2 | chi2
    k: int = Form(20),
    method: str = Form("vptree"),          # vptree | brute
    threshold: Optional[float] = Form(None),  # min cosine for a VGG-only query (default 0.2)
):
    k = max(1, min(k, 200))
    dl = [d.strip() for d in descs.split(",") if d.strip()]
    has_file = file is not None and bool(file.filename)
    if image_id is not None or has_file or not base:
        base = resolve_base(dl, base)       # the ticked descriptors decide the base
    eng = engine_of(base)                   # keywords only + explicit base: descriptors are ignored
    rgb = None
    if has_file:
        rgb = D.decode_rgb(await file.read())
        if rgb is None:
            raise HTTPException(400, "Unreadable image")
    if image_id is not None and not (0 <= image_id < eng.n):
        raise HTTPException(404, "Unknown image id")
    try:
        w = {k_: float(v) for k_, v in json.loads(weights).items()} if weights else None
    except Exception:
        raise HTTPException(400, "weights must be JSON")
    if metric not in ("default", "l1", "l2", "chi2"):
        raise HTTPException(400, "metric must be default, l1, l2 or chi2")
    if method not in ("vptree", "brute"):
        raise HTTPException(400, "method must be vptree or brute")
    metrics = None if metric == "default" else {d: metric for d in dl}
    try:
        return eng.search(keywords=keywords, kw_mode=kw_mode, combine=combine,
                          query_id=image_id if rgb is None else None, query_rgb=rgb,
                          descs=dl, weights=w, metrics=metrics, k=k, method=method,
                          exclude_query=image_id is not None and rgb is None,
                          threshold="auto" if threshold is None else threshold)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:                      # Elasticsearch down, etc.
        raise HTTPException(503, f"{type(e).__name__}: {e}")


def _serve_image(base, image_id, w):
    eng = engine_of(base)
    if not (0 <= image_id < eng.n):
        raise HTTPException(404)
    if eng.thumbs:                                  # 1M base: JPEG read from the zip
        if thumb_store is None:
            raise HTTPException(404, "thumbnail zips not found")
        try:
            data = thumb_store.read(image_id)
        except KeyError:
            raise HTTPException(404, "thumbnail not downloaded")
        return Response(data, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    p = eng.image_root / eng.paths[image_id]
    if not p.exists():
        fl = eng.flickr.get(str(image_id), {})
        if fl.get("flickr_farm") and fl.get("flickr_server") and fl.get("flickr_secret"):
            return RedirectResponse(
                f"https://farm{fl['flickr_farm']}.staticflickr.com/{fl['flickr_server']}/"
                f"{fl['id']}_{fl['flickr_secret']}.jpg")
        raise HTTPException(404, "file missing")
    if not w:
        return FileResponse(p)
    key = (base, image_id, w)
    if key not in _thumb_cache:
        img = cv2.imread(str(p))
        h0, w0 = img.shape[:2]
        s = w / max(w0, h0)
        if s < 1:
            img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        _thumb_cache[key] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        if len(_thumb_cache) > 5000:
            _thumb_cache.pop(next(iter(_thumb_cache)))
    return Response(_thumb_cache[key], media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/img/{base}/{image_id}")
def image_of_base(base: str, image_id: int, w: int = Query(0, ge=0, le=1600)):
    return _serve_image(base, image_id, w)


@app.get("/img/{image_id}")
def image(image_id: int, w: int = Query(0, ge=0, le=1600)):
    return _serve_image("objects", image_id, w)      # URL used by the reference client


# ---- endpoints with the same request/response contract as the reference project -----------
# (his Streamlit client can talk to this server unchanged; they search the Objects base)
@app.post("/search_by_text")
@app.post("/search_by_text/")
async def search_by_text(request: Request, content: dict = Body(...)):
    """{"tags": "sky", "type": "fuzzy" | "match", "number": 7} -> {"resulttype": <ES hits>}"""
    kind = content.get("type", "match")
    if kind not in ("fuzzy", "match"):
        raise HTTPException(400, "type must be fuzzy or match")
    try:
        number = int(content.get("number", 10))
    except (TypeError, ValueError):
        raise HTTPException(400, "number must be an integer")
    return engine_of("objects").es_text_response(
        str(content.get("tags", "")), kind, max(1, min(number, 200)),
        base_url=str(request.base_url).rstrip("/"))


@app.post("/search_by_image")
@app.post("/search_by_image/")
async def search_by_image(request: Request, content: dict = Body(...)):
    """{"img": <base64 image>, "number": 7} -> {"resulttype": <ES hits>} (score = cosine + 1)"""
    try:
        rgb = D.decode_rgb(base64.b64decode(content["img"]))
        number = int(content.get("number", 10))
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "body must be {\"img\": <base64>, \"number\": <int>}")
    if rgb is None:
        raise HTTPException(400, "Unreadable image")
    eng = engine_of("objects")
    if "vgg" not in eng.descs:
        raise HTTPException(400, "this index has no VGG16 descriptor: rebuild it with --desc vgg")
    return eng.es_image_response(rgb, max(1, min(number, 200)),
                                 base_url=str(request.base_url).rstrip("/"))


@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
