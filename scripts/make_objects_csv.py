"""Write index/objects.csv (id,path,label,tags) from the Objects index: the file Logstash loads
into the Elasticsearch index `objects`.

    python scripts/make_objects_csv.py --index index
"""
import argparse
import csv
import json
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    a = ap.parse_args()
    out = Path(a.index)
    meta = json.loads((out / "meta.json").read_text())
    tags = meta.get("tags", {})
    with open(out / "objects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "label", "tags"])
        for i, (p, lab) in enumerate(zip(meta["paths"], meta["labels"])):
            t = tags.get(str(i)) or ("" if lab == "unlabeled" else lab.replace("_", " ").replace("-", " "))
            w.writerow([i, p, lab, t])
    print(f"{len(meta['paths'])} rows -> {out / 'objects.csv'}")
