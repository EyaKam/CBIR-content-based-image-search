"""Quantitative evaluation: precision/recall/mAP per descriptor + brute-force vs VP-tree timing.

    python scripts/evaluate.py --queries 200 --out results
Relevant = same class as the query (leave-one-out: the query itself is excluded).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbir import config as C
from cbir.engine import SearchEngine

KS = (5, 10, 20, 50)


def pr_at_k(labels, qlabel, ranked_ids, class_size):
    rel = np.array([labels[i] == qlabel for i in ranked_ids], dtype=float)
    out = {}
    for k in KS:
        hits = rel[:k].sum()
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = hits / max(class_size - 1, 1)
    # average precision over the whole ranking
    cum = np.cumsum(rel)
    prec = cum / (np.arange(len(rel)) + 1)
    out["AP"] = float((prec * rel).sum() / max(rel.sum(), 1))
    return out


def run(eng, qids, descs, metrics=None, method="brute", k=100):
    sizes = {c: eng.labels.count(c) for c in eng.classes}
    agg, t_tot, nd = {}, 0.0, 0
    for q in qids:
        r = eng.search(query_id=int(q), descs=descs, metrics=metrics, k=k,
                       method=method, exclude_query=True, threshold=None)
        t_tot += r["info"]["time_ms"]
        nd += r["info"]["n_dist"]
        ids = [x["id"] for x in r["results"]]
        for key, v in pr_at_k(eng.labels, eng.labels[q], ids, sizes[eng.labels[q]]).items():
            agg.setdefault(key, []).append(v)
    row = {key: float(np.mean(v)) for key, v in agg.items()}
    row["ms/query"] = t_tot / len(qids)
    row["dist_evals"] = nd / len(qids)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(C.INDEX_DIR))
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    eng = SearchEngine(a.index)
    rng = np.random.default_rng(0)
    qids = rng.choice(eng.n, min(a.queries, eng.n), replace=False)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    avail = eng.descs

    # 1) descriptors alone and combined, with their default metric
    configs = [[d] for d in avail]
    if len(avail) > 1:
        configs.append(list(avail))
    rows = []
    for descs in configs:
        row = {"config": "+".join(descs), **run(eng, qids, descs)}
        rows.append(row)
        print(row["config"], {k: round(v, 3) for k, v in row.items() if k != "config"})

    # 2) metric comparison for each single descriptor
    metric_rows = []
    for d in avail:
        for m in ("l1", "l2", "chi2"):
            row = {"config": f"{d}/{m}", **run(eng, qids, [d], metrics={d: m})}
            metric_rows.append(row)
            print(row["config"], round(row["P@10"], 3), round(row["AP"], 3))

    # 3) brute force vs VP-tree (same results, fewer distance evaluations?)
    speed = []
    for descs in configs:
        b = run(eng, qids[:50], descs, method="brute", k=10)
        t0 = time.time()
        eng._tree(*eng._setup(descs, None, None))          # build time
        build = time.time() - t0
        v = run(eng, qids[:50], descs, method="vptree", k=10)
        # exactness check on a few queries
        same = 0
        for q in qids[:20]:
            r1 = eng.search(query_id=int(q), descs=descs, k=10, method="brute", threshold=None)["results"]
            r2 = eng.search(query_id=int(q), descs=descs, k=10, method="vptree", threshold=None)["results"]
            same += int({x["id"] for x in r1} == {x["id"] for x in r2})
        speed.append({"config": "+".join(descs), "brute_ms": b["ms/query"], "vptree_ms": v["ms/query"],
                      "brute_evals": b["dist_evals"], "vptree_evals": v["dist_evals"],
                      "build_s": build, "identical_top10": f"{same}/20"})
        print(speed[-1])

    for name, data in (("descriptors", rows), ("metrics", metric_rows), ("speed", speed)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    # precision/recall curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4.5))
    for r in rows:
        plt.plot([r[f"R@{k}"] for k in KS], [r[f"P@{k}"] for k in KS], "o-", label=r["config"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=.3)
    plt.legend()
    plt.title("Precision / recall (k = 5, 10, 20, 50)")
    plt.tight_layout()
    plt.savefig(out / "precision_recall.png", dpi=150)
    print(f"\nresults written to {out}/")
