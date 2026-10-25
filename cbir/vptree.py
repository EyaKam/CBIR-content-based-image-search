"""Vantage-Point tree for exact k-NN search in any *metric* space.

The tree only needs a function  dist(i, idxs) -> distances between item i and the
items `idxs`  (vectorised with numpy), so it works for a single descriptor as well
as for a weighted combination of several descriptors.

Pruning relies on the triangle inequality: this is why it must only be used with
true metrics (L1, L2, sqrt-chi2) and not with KL or histogram intersection.
"""
import heapq

import numpy as np


class _Node:
    __slots__ = ("vp", "mu", "inner", "outer", "items")

    def __init__(self):
        self.vp = None      # vantage point (item index)
        self.mu = 0.0       # median distance to the vantage point
        self.inner = None   # items with d <= mu
        self.outer = None   # items with d > mu
        self.items = None   # leaf bucket (np.ndarray of item indices)


class VPTree:
    def __init__(self, n_items, dist_db, leaf_size=40, seed=0):
        """dist_db(i, idxs): distances from database item i to database items idxs."""
        self.n = n_items
        self.dist_db = dist_db
        self.leaf_size = leaf_size
        self.rng = np.random.default_rng(seed)
        self.n_nodes = 0
        self.root = self._build(np.arange(n_items))

    # ------------------------------------------------------------------ build
    def _build(self, idxs):
        node = _Node()
        self.n_nodes += 1
        if len(idxs) <= self.leaf_size:
            node.items = idxs
            return node
        # pick a random vantage point, remove it from the set
        p = int(self.rng.integers(len(idxs)))
        vp = idxs[p]
        rest = np.delete(idxs, p)
        d = self.dist_db(vp, rest)
        mu = float(np.median(d))
        inner_mask = d <= mu
        # guard: all distances equal -> would recurse forever
        if inner_mask.all() or not inner_mask.any():
            node.items = idxs
            return node
        node.vp, node.mu = int(vp), mu
        node.inner = self._build(rest[inner_mask])
        node.outer = self._build(rest[~inner_mask])
        return node

    # ----------------------------------------------------------------- search
    def knn(self, dist_q, k):
        """dist_q(idxs): distances from the query to database items idxs.

        Returns (list of (distance, idx) sorted ascending, number of distance evals).
        """
        heap = []              # max-heap via negative distance: (-d, idx)
        counter = [0]
        tau = [np.inf]

        def push(ds, ids):
            counter[0] += len(ids)
            for d, i in zip(ds.tolist(), ids.tolist()):
                if len(heap) < k:
                    heapq.heappush(heap, (-d, i))
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, i))
            if len(heap) == k:
                tau[0] = -heap[0][0]

        def visit(node):
            if node.items is not None:
                push(dist_q(node.items), node.items)
                return
            d = float(dist_q(np.array([node.vp]))[0])
            counter[0] += 1
            if len(heap) < k:
                heapq.heappush(heap, (-d, node.vp))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, node.vp))
            if len(heap) == k:
                tau[0] = -heap[0][0]
            if d <= node.mu:
                visit(node.inner)
                if d + tau[0] >= node.mu:
                    visit(node.outer)
            else:
                visit(node.outer)
                if d - tau[0] <= node.mu:
                    visit(node.inner)

        visit(self.root)
        res = sorted((-nd, i) for nd, i in heap)
        return res, counter[0]
