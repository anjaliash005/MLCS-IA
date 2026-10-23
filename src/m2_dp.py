"""Module 2 - central differential privacy for ML.

* DP gradient descent (DP-SGD of Abadi et al., 2016, run full-batch): per-
  example gradients are clipped to L2 norm C, summed, and Gaussian noise
  N(0, sigma^2 C^2 I) is added. Under add/remove adjacency each step is
  1/(2 sigma^2)-zCDP; T steps compose to rho = T/(2 sigma^2) (Bun & Steinke,
  2016), converted to (eps, delta)-DP with eps = rho + 2 sqrt(rho ln(1/delta)).
  Full-batch training gets no subsampling amplification, so the reported eps
  is a conservative (upper-bound) accounting.
* DP k-means (DPLloyd, Su et al., 2016) - differentially private
  *unsupervised* learning (syllabus 2.4).
* Hypothesis-testing bound on membership advantage for an (eps, delta)-DP
  mechanism (Kairouz et al., 2015): TPR - FPR <= (e^eps - 1 + 2 delta)/(e^eps + 1).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from common import NumpyMLP

DELTA = 1e-5


# ---------------- accountant ------------------------------------------------
def zcdp_to_eps(rho: float, delta: float = DELTA) -> float:
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def eps_for(sigma: float, steps: int, delta: float = DELTA) -> float:
    return zcdp_to_eps(steps / (2 * sigma ** 2), delta)


def sigma_for(eps: float, steps: int, delta: float = DELTA) -> float:
    L = np.log(1 / delta)
    rho = (np.sqrt(L + eps) - np.sqrt(L)) ** 2
    return float(np.sqrt(steps / (2 * rho)))


def mia_advantage_bound(eps: float, delta: float = DELTA) -> float:
    return min(1.0, (np.exp(eps) - 1 + 2 * delta) / (np.exp(eps) + 1))


# ---------------- DP gradient descent ---------------------------------------
def dp_gd_recipe(eps: float, hidden: int, steps: int = 200, lr: float = 0.5,
                 clip: float = 1.0, delta: float = DELTA):
    """Returns recipe(X, y, rng) -> NumpyMLP. eps=np.inf -> non-private GD
    with the same architecture, step count and learning rate (no clipping)."""
    private = np.isfinite(eps)
    sigma = sigma_for(eps, steps, delta) if private else 0.0

    def recipe(X, y, rng):
        m = NumpyMLP(X.shape[1], hidden, rng)
        theta = m.get()
        n = len(X)
        for _ in range(steps):
            m.set(theta)
            G = m.per_example_grads(X, y)
            if private:
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                G = G / np.maximum(1.0, norms / clip)
                g = (G.sum(0) + rng.normal(0, sigma * clip, G.shape[1])) / n
            else:
                g = G.mean(0)
            theta = theta - lr * g
        m.set(theta)
        m.sigma, m.eps = sigma, eps
        return m

    recipe.sigma = sigma
    return recipe


# ---------------- DP k-means (DPLloyd) --------------------------------------
def dp_kmeans(X: np.ndarray, k: int, eps: float, iters: int,
              rng: np.random.Generator) -> np.ndarray:
    """X must lie in [-1, 1]^d. Each iteration spends eps/iters, split equally
    between noisy counts (L1 sensitivity 1) and noisy sums (L1 sensitivity d)."""
    n, d = X.shape
    C = rng.uniform(-0.5, 0.5, (k, d))   # data-independent initialisation
    e_it = eps / iters
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            pts = X[lab == j]
            cnt = len(pts) + rng.laplace(0, 1 / (e_it / 2))
            s = pts.sum(0) + rng.laplace(0, d / (e_it / 2), d)
            if cnt > 1:
                C[j] = np.clip(s / cnt, -1, 1)
    return C


def sse(X, C) -> float:
    return float(((X[:, None, :] - C[None]) ** 2).sum(-1).min(1).sum())


def dp_kmeans_experiment(X, k=2, eps_list=(0.1, 0.25, 0.5, 1, 2, 4, 8),
                         iters=5, trials=30, seed=0):
    rng = np.random.default_rng(seed)
    base = sse(X, KMeans(k, n_init=10, random_state=seed).fit(X).cluster_centers_)
    rows = []
    for e in eps_list:
        r = [sse(X, dp_kmeans(X, k, e, iters, rng)) / base for _ in range(trials)]
        rows.append({"eps": e, "sse_ratio_mean": np.mean(r), "sse_ratio_std": np.std(r)})
    return rows, base
