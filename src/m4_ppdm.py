"""Module 4 - privacy-preserving data mining.

* 4.3 Protecting privacy while publishing data: Mondrian multidimensional
  k-anonymity (LeFevre et al., 2006) on the quasi-identifiers
  (Age, Pregnancies, BMI). Each QI value is replaced by its equivalence-class
  mean before the model is trained.
* 4.4 Beyond k-anonymity: the same partitioner with an additional distinct
  l-diversity constraint on the sensitive attribute (diabetes outcome), plus
  homogeneity-attack exposure and t-closeness measurement.
* 4.2 Modifying the input: additive random perturbation of features
  (Agrawal & Srikant, 2000) - no formal guarantee.
* 4.5 Modifying the data-mining output: restricting the prediction API to
  rounded scores or labels only (implemented in attacks.api_output).
"""
from __future__ import annotations

import numpy as np

PIMA_QI = ["Age", "Pregnancies", "BMI"]


def mondrian(Q: np.ndarray, s: np.ndarray, k: int, l: int = 1) -> list:
    """Q: (n, q) quasi-identifiers already scaled to the same public range.
    s: sensitive attribute. Returns a list of index arrays (equivalence classes)."""
    def ok(idx):
        return len(idx) >= k and len(np.unique(s[idx])) >= l

    out, stack = [], [np.arange(len(Q))]
    while stack:
        idx = stack.pop()
        spans = Q[idx].max(0) - Q[idx].min(0)
        done = True
        for dim in np.argsort(-spans):
            if spans[dim] == 0:
                break
            v = Q[idx, dim]
            med = np.median(v)
            left, right = idx[v <= med], idx[v > med]
            if len(right) == 0:            # many ties at the median
                left, right = idx[v < med], idx[v >= med]
            if len(left) and len(right) and ok(left) and ok(right):
                stack += [left, right]
                done = False
                break
        if done:
            out.append(idx)
    return out


def anonymise(Q: np.ndarray, classes: list) -> np.ndarray:
    G = Q.copy()
    for idx in classes:
        G[idx] = Q[idx].mean(0)
    return G


def anonymity_report(Q: np.ndarray, s: np.ndarray, classes: list,
                     domain_width: float = 2.0) -> dict:
    sizes = np.array([len(c) for c in classes])
    ncp = np.mean([((Q[c].max(0) - Q[c].min(0)) / domain_width).mean()
                   for c in classes for _ in c])
    homog = sum(len(c) for c in classes if len(np.unique(s[c])) == 1) / len(s)
    p_all = s.mean()
    # t-closeness for a binary attribute: total-variation distance between the
    # class distribution of the sensitive value and the overall distribution.
    t = max(abs(s[c].mean() - p_all) for c in classes)
    return {"n_classes": len(classes), "min_size": int(sizes.min()),
            "avg_size": float(sizes.mean()), "ncp": float(ncp),
            "homogeneous_frac": float(homog), "t_closeness": float(t),
            "min_distinct_l": int(min(len(np.unique(s[c])) for c in classes))}


def kanon_recipe(k: int, l: int, qi_cols: list, base_recipe):
    def recipe(X, y, rng):
        classes = mondrian(X[:, qi_cols], y, k, l)
        Xa = X.copy()
        Xa[:, qi_cols] = anonymise(X[:, qi_cols], classes)
        return base_recipe(Xa, y, rng)
    return recipe


def input_noise_recipe(sigma: float, base_recipe):
    def recipe(X, y, rng):
        return base_recipe(X + rng.normal(0, sigma, X.shape), y, rng)
    return recipe
