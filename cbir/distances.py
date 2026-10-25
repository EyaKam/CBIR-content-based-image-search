"""Vectorised distances: one query vector against a whole matrix of signatures."""
import numpy as np

EPS = 1e-10


def l1(q, X):
    return np.abs(X - q).sum(axis=1)


def l2(q, X):
    d = X - q
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def chi2(q, X):
    """sqrt of the chi-squared distance used in the course.

    sqrt(chi2) is a true metric (the triangle inequality holds), so it is safe
    for the VP-tree, whereas chi2 itself is not.
    """
    num = (X - q) ** 2
    den = X + q + EPS
    return np.sqrt(0.5 * (num / den).sum(axis=1))


def intersection(q, X):
    """1 - histogram intersection (for L1-normalised histograms). Not a metric."""
    return 1.0 - np.minimum(X, q).sum(axis=1)


def kl(q, X):
    """Kullback-Leibler divergence (asymmetric, NOT a metric)."""
    P = np.clip(q, EPS, None)
    Q = np.clip(X, EPS, None)
    return (P * np.log2(P / Q)).sum(axis=1)


METRICS = {"l1": l1, "l2": l2, "chi2": chi2, "intersection": intersection, "kl": kl}
TRUE_METRICS = {"l1", "l2", "chi2"}
