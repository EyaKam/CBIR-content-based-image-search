import numpy as np
from cbir import distances as DI
from cbir.vptree import VPTree


def test_vptree_matches_brute_force():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 20)).astype(np.float32)
    tree = VPTree(len(X), lambda i, idx: DI.l2(X[i], X[idx]), leaf_size=16)
    for _ in range(25):
        q = rng.normal(size=20).astype(np.float32)
        exact = np.argsort(DI.l2(q, X))[:10]
        res, n_eval = tree.knn(lambda idx: DI.l2(q, X[idx]), 10)
        assert [i for _, i in res] == exact.tolist()
        assert n_eval <= len(X)  # unclustered 20-D data: pruning may be nil (curse of dimensionality)


def test_chi2_is_symmetric_and_zero_on_self():
    rng = np.random.default_rng(2)
    X = rng.random((50, 30)).astype(np.float32)
    X /= X.sum(1, keepdims=True)
    assert np.allclose(DI.chi2(X[0], X[:1]), 0, atol=1e-4)
    assert np.allclose(DI.chi2(X[0], X[1:2]), DI.chi2(X[1], X[0:1]))
