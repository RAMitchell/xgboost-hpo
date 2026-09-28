"""Frozen feature encoding, normal scores and historical kernel likelihood."""

from statistics import NormalDist

import numpy as np

FEATURES = [
    "log_learning_rate",
    "depth_active",
    "depth",
    "leaves_active",
    "log_leaves",
    "lossguide",
    "log_min_child_weight",
    "subsample",
    "column_tree",
    "column_level",
    "column_node",
    "lambda_active",
    "log_lambda",
    "alpha_active",
    "log_alpha",
    "gamma_active",
    "log_gamma",
    "log_max_bin",
]


def normal_rank_scores(y):
    """Normal scores of average ranks; ties receive the same score.

    NormalDist is part of Python's standard library. NumPy is the only
    third-party dependency, including for the GP linear algebra.
    """
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    quantiles = ((np.cumsum(counts) - 0.5 * counts) / len(y))[inverse]
    inverse_cdf = NormalDist().inv_cdf
    return np.fromiter((inverse_cdf(float(p)) for p in quantiles), float, len(y))


def encode(configs):
    out = []
    for cf in configs:
        p = cf["params"]
        d = p["max_depth"]
        l = p["max_leaves"]
        x = [
            np.log(p["learning_rate"] / 0.01) / np.log(30),
            float(d > 0),
            (d - 2) / 10 if d else 0.0,
            float(l > 0),
            np.log(l / 8) / np.log(32) if l else 0.0,
            float(p["grow_policy"] == "lossguide"),
            np.log(p["min_child_weight"] / 1e-5) / np.log(1e7),
            (p["subsample"] - 0.5) / 0.5,
        ]
        x.extend((1 - p["colsample_by" + k]) / 0.5 for k in ["tree", "level", "node"])
        for name, hi in [("reg_lambda", 100), ("reg_alpha", 10), ("gamma", 10)]:
            v = p[name]
            x.extend(
                [float(v > 0), np.log(v / 1e-5) / np.log(hi / 1e-5) if v > 0 else 0.0]
            )
        x.append(np.log2(p["max_bin"] / 128) / 2)
        out.append(x)
    a = np.asarray(out)
    assert a.shape == (len(configs), 18) and a.min() > -1e-10 and a.max() < 1 + 1e-10
    return np.clip(a, 0, 1)


def kernel(theta, d, gradient=False):
    n = d.shape[-1]
    q = d / np.exp(2 * theta[:n])
    r = np.sqrt(5 * q.sum(axis=2))
    e = np.exp(-r)
    amp = np.exp(theta[n])
    K = amp * (1 + r + r * r / 3) * e
    if not gradient:
        return K
    g = amp * 5 / 3 * (1 + r[:, :, None]) * e[:, :, None] * q
    return K, np.concatenate([g, K[:, :, None]], axis=2)


def objective(theta, groups):
    from scipy.linalg import cho_solve

    value = 0.0
    grad = np.zeros_like(theta)
    for d, second, w in groups:
        K, g = kernel(theta, d, True)
        n = len(K)
        noise = np.exp(theta[-1])
        L = np.linalg.cholesky(K + (noise + 1e-10) * np.eye(n))
        inv = cho_solve((L, True), np.eye(n), check_finite=False)
        value += (
            w
            * 0.5
            * (
                2 * np.log(np.diag(L)).sum()
                + np.sum(inv * second)
                + n * np.log(2 * np.pi)
            )
            / n
        )
        Q = 0.5 * (inv - inv @ second @ inv) / n
        grad += w * np.r_[np.einsum("ij,ijk->k", Q, g), noise * np.trace(Q)]
    return float(value), grad
