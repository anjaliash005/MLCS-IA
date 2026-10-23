"""Module 5 - compressive privacy (Kung, 2017).

Clients send a k-dimensional projection z = W^T x instead of the raw record.
  * PCA  - unsupervised, keeps maximum variance (and hence reconstructability).
  * DCA  - Discriminant Component Analysis: maximises
           w^T (S_B + rho' I) w / w^T (S_bar + rho I) w for the utility label,
           where S_bar = S_W + S_B is the centred scatter.
  * DUCA - Differential Utility/Cost Analysis: maximises utility discriminant
           power while penalising discriminant power for a *privacy* label:
           w^T (S_B^u + rho' I) w / w^T (S_B^p + S_W^u + rho I) w.
Privacy is measured three ways: (i) attacker's balanced accuracy at inferring
the privacy attribute from z, (ii) linear reconstruction error of x from z,
(iii) membership inference against the model trained on z.
Utility (syllabus 5.4) is the accuracy of the classifier trained on z.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _scatter(X, y):
    mu = X.mean(0)
    Sw = np.zeros((X.shape[1],) * 2)
    Sb = np.zeros_like(Sw)
    for c in np.unique(y):
        Xc = X[y == c]
        mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc)
        Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    n = len(X)
    return Sw / n, Sb / n


def pca_proj(X, k):
    Xc = X - X.mean(0)
    vals, vecs = eigh(Xc.T @ Xc / len(X))
    return vecs[:, ::-1][:, :k]


def dca_proj(X, y, k, rho=1e-2, rho_p=1e-3):
    Sw, Sb = _scatter(X, y)
    I = np.eye(X.shape[1])
    vals, vecs = eigh(Sb + rho_p * I, Sw + Sb + rho * I)
    return vecs[:, ::-1][:, :k]


def duca_proj(X, y, s, k, rho=1e-2, rho_p=1e-3, gamma=5.0):
    Sw_u, Sb_u = _scatter(X, y)
    _, Sb_p = _scatter(X, s)
    I = np.eye(X.shape[1])
    vals, vecs = eigh(Sb_u + rho_p * I, gamma * Sb_p + Sw_u + rho * I)
    return vecs[:, ::-1][:, :k]


class Projected:
    """Wraps a model trained on z = (x - mu) W so it answers raw-x queries."""
    def __init__(self, W, mu, model):
        self.W, self.mu, self.model = W, mu, model

    def predict_proba(self, X):
        return self.model.predict_proba((X - self.mu) @ self.W)


def fit_projection(method: str, k: int, X, y, s=None):
    """Learned by the data owner on its own training data, then published to
    clients. Shadow models reuse the published W (the attacker knows it)."""
    if method == "PCA":
        return pca_proj(X, k)
    if method == "DCA":
        return dca_proj(X, y, k)
    if method == "DUCA":
        return duca_proj(X, y, s, k)
    raise ValueError(method)


def projected_recipe(W, mu, base_recipe):
    def recipe(X, y, rng):
        return Projected(W, mu, base_recipe((X - mu) @ W, y, rng))
    return recipe
