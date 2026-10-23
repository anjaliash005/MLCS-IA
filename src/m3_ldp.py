"""Module 3 - local differential privacy.

Every patient perturbs their own record on-device before it reaches the
hospital, so the collector never sees raw values.

Numeric mechanisms on x in [-1, 1] (all unbiased):
  * Laplace           - x + Lap(2/eps)
  * Duchi et al. 2013 - binary output +/-(e^eps+1)/(e^eps-1)
  * Piecewise (PM)    - Wang et al. 2019 (syllabus 3.3)
  * Hybrid (HM)       - PM with prob 1 - e^{-eps/2} (eps > 0.61), else Duchi
Categorical: binary randomised response; optimised unary encoding (OUE,
Wang et al. 2017) as a frequency oracle for LDP Naive Bayes (syllabus 3.4,
following Yilmaz et al. 2019).
"""
from __future__ import annotations

import numpy as np


# ---------------- numeric mechanisms ----------------------------------------
def laplace(x, eps, rng):
    return x + rng.laplace(0, 2 / eps, np.shape(x))


def duchi(x, eps, rng):
    e = np.exp(eps)
    b = (e + 1) / (e - 1)
    p = (e - 1) / (2 * e + 2) * x + 0.5
    return np.where(rng.random(np.shape(x)) < p, b, -b)


def piecewise(x, eps, rng):
    x = np.asarray(x, float)
    e2 = np.exp(eps / 2)
    C = (e2 + 1) / (e2 - 1)
    l = (C + 1) / 2 * x - (C - 1) / 2
    r = l + C - 1
    inside = rng.random(x.shape) < e2 / (e2 + 1)
    u_in = l + rng.random(x.shape) * (r - l)
    # outside: uniform over [-C, l) U (r, C]; total length C + 1
    left_len = l + C
    t = rng.random(x.shape) * (C + 1)
    u_out = np.where(t < left_len, -C + t, r + (t - left_len))
    return np.where(inside, u_in, u_out)


def hybrid(x, eps, rng):
    if eps <= 0.61:
        return duchi(x, eps, rng)
    use_pm = rng.random(np.shape(x)) < 1 - np.exp(-eps / 2)
    return np.where(use_pm, piecewise(x, eps, rng), duchi(x, eps, rng))


MECHANISMS = {"Laplace": laplace, "Duchi": duchi, "Piecewise": piecewise, "Hybrid": hybrid}


def theoretical_var(mech: str, x, eps):
    x = np.asarray(x, float)
    if mech == "Laplace":
        return np.full_like(x, 8 / eps ** 2)
    if mech == "Duchi":
        return ((np.exp(eps) + 1) / (np.exp(eps) - 1)) ** 2 - x ** 2
    if mech == "Piecewise":
        e2 = np.exp(eps / 2)
        return x ** 2 / (e2 - 1) + (e2 + 3) / (3 * (e2 - 1) ** 2)
    if mech == "Hybrid":
        if eps <= 0.61:
            return theoretical_var("Duchi", x, eps)
        a = 1 - np.exp(-eps / 2)
        # mixture of two unbiased estimators: variance is the weighted mix
        return a * theoretical_var("Piecewise", x, eps) + (1 - a) * theoretical_var("Duchi", x, eps)
    raise ValueError(mech)


def rr_binary(y, eps, rng):
    keep = rng.random(len(y)) < np.exp(eps) / (1 + np.exp(eps))
    return np.where(keep, y, 1 - y)


def mean_estimation_experiment(eps_list, n=20000, trials=40, seed=0):
    """Correctness check: empirical variance of each mechanism vs theory."""
    rng = np.random.default_rng(seed)
    x = np.clip(rng.beta(2, 5, n) * 2 - 1, -1, 1)
    rows = []
    for mech, f in MECHANISMS.items():
        for e in eps_list:
            errs, var_emp = [], []
            for _ in range(trials):
                out = f(x, e, rng)
                errs.append((out.mean() - x.mean()) ** 2)
                var_emp.append(np.mean((out - x) ** 2))
            rows.append({"mech": mech, "eps": e, "mse_mean": np.mean(errs),
                         "per_record_var_emp": np.mean(var_emp),
                         "per_record_var_theory": float(theoretical_var(mech, x, e).mean())})
    return rows


# ---------------- LDP training-data collection -----------------------------
def ldp_feature_recipe(eps: float, base_recipe, mech: str = "Piecewise"):
    """Each record's d features and its label are perturbed on-device with the
    budget split equally over the d+1 attributes (sequential composition), so
    every record is eps-LDP. The downstream model is post-processing, hence
    also eps-DP w.r.t. each training record."""
    f = MECHANISMS[mech]

    def recipe(X, y, rng):
        e_attr = eps / (X.shape[1] + 1)
        Xp = np.clip(f(X, e_attr, rng), -10, 10)   # clip extreme reports for stability
        yp = rr_binary(y, e_attr, rng)
        if len(np.unique(yp)) < 2:
            yp[0] = 1 - yp[0]
        return base_recipe(Xp, yp, rng)

    return recipe


# ---------------- LDP Naive Bayes -------------------------------------------
# Public clinical cut points (raw units). Missing values fall in the bin that
# contains the clinically typical value; SkinThickness / Insulin missingness
# gets its own category because it is common and informative.
PIMA_BINS = {
    "Pregnancies": [1, 3, 6],
    "Glucose": [100, 140, 160],
    "BloodPressure": [60, 80, 90],
    "SkinThickness": [20, 30, 40],
    "Insulin": [50, 150, 300],
    "BMI": [25, 30, 35],
    "DiabetesPedigreeFunction": [0.25, 0.5, 1.0],
    "Age": [30, 40, 50],
}
TYPICAL = {"Glucose": 110, "BloodPressure": 72, "BMI": 30}


def discretise_pima(raw) -> np.ndarray:
    cols = []
    for c, cuts in PIMA_BINS.items():
        v = raw[c].to_numpy(float)
        miss = np.isnan(v)
        b = np.digitize(np.nan_to_num(v, nan=TYPICAL.get(c, 0)), cuts)
        if c in ("SkinThickness", "Insulin"):
            b = np.where(miss, len(cuts) + 1, b)
        cols.append(b)
    return np.column_stack(cols).astype(int)


def n_bins(Xd_col_index: int) -> int:
    c = list(PIMA_BINS)[Xd_col_index]
    return len(PIMA_BINS[c]) + 1 + (1 if c in ("SkinThickness", "Insulin") else 0)


class CategoricalNB:
    def __init__(self, log_prior, log_cond):
        self.log_prior, self.log_cond = log_prior, log_cond  # cond[j]: (2, B_j)

    def predict_proba(self, Xd):
        Xd = np.asarray(Xd, int)
        s = np.tile(self.log_prior, (len(Xd), 1))
        for j, lc in enumerate(self.log_cond):
            s += lc[:, Xd[:, j]].T
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(1, keepdims=True)


def _nb_from_freq(freqs, floor=1e-3):
    """freqs[j]: estimated joint frequency table (2, B_j), may be noisy/negative."""
    prior = np.mean([np.clip(f, 0, None).sum(1) for f in freqs], axis=0)
    prior = np.clip(prior, floor, None); prior /= prior.sum()
    cond = []
    for f in freqs:
        f = np.clip(f, floor, None)
        cond.append(np.log(f / f.sum(1, keepdims=True)))
    return CategoricalNB(np.log(prior), cond)


def nb_recipe(eps: float):
    """eps=inf -> ordinary categorical NB (Laplace smoothing). Otherwise each
    user samples one feature j uniformly and reports the joint value
    (class, bin_j) through OUE with the full eps (Yilmaz et al. 2019's
    'feature sampling' variant; better utility than splitting eps over d)."""
    def recipe(Xd, y, rng):
        n, d = Xd.shape
        freqs = []
        if not np.isfinite(eps):
            for j in range(d):
                B = n_bins(j)
                t = np.ones((2, B))
                np.add.at(t, (y, Xd[:, j]), 1)
                freqs.append(t / t.sum())
            return _nb_from_freq(freqs)
        p, q = 0.5, 1 / (np.exp(eps) + 1)
        feat = rng.integers(0, d, n)
        for j in range(d):
            B = n_bins(j)
            users = np.where(feat == j)[0]
            D = 2 * B
            bits = rng.random((len(users), D)) < q
            true_idx = y[users] * B + Xd[users, j]
            bits[np.arange(len(users)), true_idx] = rng.random(len(users)) < p
            est = (bits.sum(0) - len(users) * q) / (p - q)
            freqs.append((est / max(len(users), 1)).reshape(2, B))
        return _nb_from_freq(freqs)

    return recipe
