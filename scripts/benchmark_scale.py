"""How does the VP-tree behave as N and the dimension grow?  (synthetic clustered vectors, L2)

    python scripts/benchmark_scale.py --dims 59 512 --sizes 10000 100000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbir.vptree import VPTree
from cbir import distances as DI

ap = argparse.ArgumentParser()
ap.add_argument("--dims", type=int, nargs="+", default=[16, 59, 512])
ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000])
ap.add_argument("--queries", type=int, default=30)
a = ap.parse_args()

rng = np.random.default_rng(0)
print(f"{'N':>8} {'dim':>5} {'build s':>8} {'brute ms':>9} {'vptree ms':>10} {'evals/N':>8} {'exact':>6}")
for dim in a.dims:
    for n in a.sizes:
        centers = rng.normal(0, 1, (50, dim))
        X = (centers[rng.integers(0, 50, n)] + rng.normal(0, 0.35, (n, dim))).astype(np.float32)
        t0 = time.time()
        tree = VPTree(n, lambda i, idx: DI.l2(X[i], X[idx]))
        build = time.time() - t0
        tb = tv = ev = 0.0
        ok = 0
        for q in rng.choice(n, a.queries, replace=False):
            qv = X[q] + rng.normal(0, 0.05, dim).astype(np.float32)
            t0 = time.time(); d = DI.l2(qv, X); bi = set(np.argsort(d)[:10].tolist()); tb += time.time() - t0
            t0 = time.time(); r, ne = tree.knn(lambda idx: DI.l2(qv, X[idx]), 10); tv += time.time() - t0
            ev += ne; ok += bi == {i for _, i in r}
        print(f"{n:>8} {dim:>5} {build:>8.1f} {tb / a.queries * 1e3:>9.2f} {tv / a.queries * 1e3:>10.2f} "
              f"{ev / a.queries / n:>8.3f} {ok:>3}/{a.queries}")
