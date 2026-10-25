"""Quick end-to-end check of a RUNNING server (image search, keywords, both bases).

    python -m uvicorn server:app --port 8000      # in another terminal
    python scripts/smoke_test.py                  # [--url http://localhost:8000]

Prints one line per check and exits with code 1 if one of them fails. Checks that need a base
which is not loaded (e.g. MIRFLICKR before the level-4 setup) are skipped, not failed.
"""
import argparse
import sys

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
URL = ap.parse_args().url.rstrip("/")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


def skip(name, why):
    print(f"[skip] {name}  ({why})")


def search(**form):
    return requests.post(f"{URL}/api/search", data={k: str(v) for k, v in form.items()}, timeout=120)


try:
    info = requests.get(f"{URL}/api/info", timeout=10).json()
except Exception as e:
    sys.exit(f"Server not reachable at {URL}: {e}\nStart it with: python -m uvicorn server:app --port 8000")

bases = info["bases"]
check("server answers /api/info", bool(bases),
      ", ".join(f"{k}: {b['n_images']} images, keywords via {b['keywords']}" for k, b in bases.items()))

# ------------------------------------------------------------------ base Objects
if "objects" in bases:
    ob = bases["objects"]
    descs = [d for d in ("hsv", "lbp", "hog") if d in ob["descriptors"]]
    pick = requests.get(f"{URL}/api/browse", params={"base": "objects", "label": "butterfly", "n": 1, "seed": 1}).json()
    qid = pick[0]["id"] if pick else 0
    r = search(image_id=qid, descs=",".join(descs), k=5, base="objects")
    j = r.json() if r.ok else {}
    check("Objects: image search by id (" + "+".join(descs) + ")", r.ok and len(j.get("results", [])) == 5,
          f"{j.get('info', {}).get('method')} · {j.get('info', {}).get('time_ms')} ms · labels "
          + ", ".join(x["label"] for x in j.get("results", [])))
    if "vgg" in ob["descriptors"]:
        queries = requests.get(f"{URL}/api/browse", params={"base": "objects", "label": "butterfly", "n": 8, "seed": 3}).json()
        hits = total = 0
        for q in queries:
            labs = [x["label"] for x in search(image_id=q["id"], descs="vgg", k=5, base="objects").json().get("results", [])]
            hits, total = hits + labs.count("butterfly"), total + 5
        check("Objects: deep features (VGG16) find the same class (P@5 over 8 queries >= 0.5)",
              total > 0 and hits / total >= 0.5, f"P@5 = {hits / max(total, 1):.2f}")
    else:
        skip("Objects: VGG16", "not in the index (build with --desc hsv lbp hog vgg)")
    img = requests.get(f"{URL}/img/objects/{qid}?w=200", timeout=20)
    check("Objects: image/thumbnail served", img.ok and img.headers.get("content-type", "").startswith("image"),
          f"{len(img.content)} bytes")
    r = requests.post(f"{URL}/api/search", files={"file": ("q.jpg", img.content, "image/jpeg")},
                      data={"descs": "hsv", "k": "3", "base": "objects"}, timeout=120)
    check("Objects: search with an uploaded image", r.ok and len(r.json().get("results", [])) == 3)
    r = search(keywords="butterfly", base="objects", k=3)
    j = r.json() if r.ok else {}
    check("Objects: keyword search 'butterfly'", r.ok and j.get("info", {}).get("total_matches", 0) > 0
          and all(x["label"] == "butterfly" for x in j.get("results", [])),
          f"{j.get('info', {}).get('total_matches')} matches via {j.get('info', {}).get('text_engine')}")
    r = search(keywords="butterfly", image_id=qid, combine="and", descs=descs[0], k=5, base="objects")
    check("Objects: keywords AND image", r.ok and all(x["label"] == "butterfly" for x in r.json().get("results", [])))
    r = requests.post(f"{URL}/search_by_text", json={"tags": "buterfly", "type": "fuzzy", "number": 3}, timeout=30)
    check("Objects: fuzzy search tolerates a typo ('buterfly')",
          r.ok and r.json()["resulttype"]["hits"]["total"]["value"] > 0,
          str(r.json()["resulttype"]["hits"]["total"]["value"]) + " matches" if r.ok else r.text[:80])
else:
    skip("Objects checks", "base not loaded")

# ------------------------------------------------------------------ base MIRFLICKR
if "mirflickr" in bases:
    mb = bases["mirflickr"]
    r = search(image_id=0, descs="ehd", k=5, base="mirflickr")
    j = r.json() if r.ok else {}
    check("MIRFLICKR-1M: image search with EHD", r.ok and len(j.get("results", [])) == 5,
          f"{j.get('info', {}).get('method')} · {j.get('info', {}).get('n_dist')} distances · "
          f"{j.get('info', {}).get('time_ms')} ms (the first call builds the VP-tree)")
    if mb["keywords"] == "elasticsearch":
        r = search(keywords="sunglasses", base="mirflickr", k=3)
        j = r.json() if r.ok else {}
        check("MIRFLICKR-1M: keyword search 'sunglasses' (Elasticsearch)",
              r.ok and j.get("info", {}).get("total_matches", 0) > 0, f"{j.get('info', {}).get('total_matches')} matches")
    else:
        skip("MIRFLICKR-1M keywords", "Elasticsearch has no 'mirflickr' index (level 3/4 of the guide)")
    t = requests.get(f"{URL}/img/mirflickr/0", timeout=20)
    if t.ok:
        check("MIRFLICKR-1M: thumbnail served from the zip", t.headers.get("content-type") == "image/jpeg", f"{len(t.content)} bytes")
    else:
        skip("MIRFLICKR-1M thumbnail", "thumbs zip not downloaded (images appear grey)")
    r = requests.post(f"{URL}/api/search", files={"file": ("q.jpg", b"x", "image/jpeg")},
                      data={"descs": "ehd", "base": "mirflickr"}, timeout=30)
    check("MIRFLICKR-1M: uploaded image refused (MPEG-7 only)", r.status_code == 400)
else:
    skip("MIRFLICKR checks", "base not loaded (level 4 of the guide)")

# ------------------------------------------------------------------ both bases
if "objects" in bases and "mirflickr" in bases:
    r = search(image_id=1, descs="hog,ehd", k=3)
    check("Mixing descriptors of two bases (HOG + EHD) is refused", r.status_code == 400, r.json().get("detail", "")[:70])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
