"""Shared utilities: dataset loading, data-independent preprocessing, splits,
metrics and a small NumPy MLP that supports per-example gradients (used by the
DP-SGD experiments in m2_dp.py).

Author: Anjali Ashtankar (16014223012, B1) - IA-2, Security and Privacy for ML.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# numpy 2.0 + macOS Accelerate emits spurious matmul warnings; results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")

PIMA_FILE = os.path.join(DATA_RAW, "pima-indians-diabetes.data.csv")
PIMA_SHA256 = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"
PIMA_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Public, data-independent bounds (clinical plausibility, rounded outward from
# the ranges published with the dataset). Fixing them a priori means scaling
# does not consume privacy budget, which the DP / LDP guarantees depend on.
PIMA_BOUNDS = {
    "Pregnancies": (0, 17), "Glucose": (40, 200), "BloodPressure": (20, 125),
    "SkinThickness": (5, 100), "Insulin": (10, 850), "BMI": (15, 68),
    "DiabetesPedigreeFunction": (0.05, 2.5), "Age": (21, 81),
}
# In this dataset a value of 0 means "not measured" for these columns.
PIMA_ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
# Columns with substantial missingness get an explicit missing-indicator.
PIMA_INDICATORS = ["SkinThickness", "Insulin"]


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # scaled to [-1, 1], data-independent transform
    y: np.ndarray            # binary utility label
    feature_names: list
    raw: pd.DataFrame        # cleaned, unscaled table (NaN = missing)
    privacy_attr: np.ndarray | None = None   # sensitive attribute for M5
    privacy_attr_name: str | None = None


def scale_to_unit(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip(values, lo, hi)
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def load_pima() -> Dataset:
    if sha256(PIMA_FILE) != PIMA_SHA256:
        raise RuntimeError("Pima file hash mismatch - dataset changed.")
    df = pd.read_csv(PIMA_FILE, header=None, names=PIMA_COLUMNS)
    raw = df.copy()
    for c in PIMA_ZERO_MISSING:
        raw.loc[raw[c] == 0, c] = np.nan
    cols, names = [], []
    for c, (lo, hi) in PIMA_BOUNDS.items():
        v = raw[c].to_numpy(dtype=float)
        s = scale_to_unit(np.nan_to_num(v, nan=(lo + hi) / 2.0), lo, hi)
        s[np.isnan(v)] = 0.0        # missing -> centre of public range
        cols.append(s)
        names.append(c)
    for c in PIMA_INDICATORS:
        cols.append(np.where(raw[c].isna(), 1.0, -1.0))
        names.append(f"{c}_missing")
    X = np.column_stack(cols)
    y = raw["Outcome"].to_numpy(dtype=int)
    age = raw["Age"].to_numpy()
    return Dataset("Pima", X, y, names, raw,
                   privacy_attr=(age >= 35).astype(int),
                   privacy_attr_name="Age >= 35")


def load_breast_cancer_ds() -> Dataset:
    b = load_breast_cancer()
    raw = pd.DataFrame(b.data, columns=b.feature_names)
    # Feature min/max are published in the dataset documentation (DESCR), so
    # they are treated as public metadata for scaling.
    lo, hi = b.data.min(axis=0), b.data.max(axis=0)
    X = np.column_stack([scale_to_unit(b.data[:, j], lo[j], hi[j])
                         for j in range(b.data.shape[1])])
    # sklearn encodes malignant = 0; use malignant = 1 as the positive class.
    y = (b.target == 0).astype(int)
    raw["Malignant"] = y
    return Dataset("BreastCancer", X, y, list(b.feature_names), raw)


def bc_file_info() -> tuple[str, str]:
    import sklearn.datasets as skd
    p = os.path.join(os.path.dirname(skd.__file__), "data", "breast_cancer.csv")
    return p, sha256(p)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
@dataclass
class Split:
    target_in: np.ndarray    # members of the target model
    target_out: np.ndarray   # non-members (also the utility test set)
    shadow_pool: np.ndarray  # attacker's auxiliary data, same distribution


def stratified_perm(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutation that interleaves the classes so any prefix is stratified."""
    idx = [rng.permutation(np.where(y == c)[0]) for c in np.unique(y)]
    keys = np.concatenate([(np.arange(len(i)) + rng.random()) / len(i) for i in idx])
    order = np.argsort(keys, kind="stable")
    return np.concatenate(idx)[order]


def make_split(y: np.ndarray, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    perm = stratified_perm(y, rng)
    q = len(y) // 4
    return Split(perm[:q], perm[q:2 * q], perm[2 * q:])


def shadow_splits(pool: np.ndarray, y: np.ndarray, n_shadows: int,
                  size: int, seed: int):
    """Each shadow model trains on `size` records from the pool and uses
    another disjoint `size` records as its non-members."""
    rng = np.random.default_rng(seed + 10_000)
    out = []
    for _ in range(n_shadows):
        perm = pool[stratified_perm(y[pool], rng)]
        out.append((perm[:size], perm[size:2 * size]))
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def utility_metrics(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "auc": auc}


# --------------------------------------------------------------------------
# NumPy MLP with per-example gradients (one hidden tanh layer; hidden=0 gives
# logistic regression). Used for both the non-private and DP-SGD variants so
# the comparison isolates the effect of clipping + noise.
# --------------------------------------------------------------------------
class NumpyMLP:
    def __init__(self, d: int, hidden: int, rng: np.random.Generator):
        self.d, self.h = d, hidden
        if hidden > 0:
            self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, hidden))
            self.b1 = np.zeros(hidden)
            self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden)
        else:
            self.w2 = np.zeros(d)
        self.b2 = 0.0

    # parameter vector helpers
    def get(self) -> np.ndarray:
        parts = [self.W1.ravel(), self.b1] if self.h else []
        return np.concatenate(parts + [self.w2, [self.b2]])

    def set(self, theta: np.ndarray):
        i = 0
        if self.h:
            self.W1 = theta[i:i + self.d * self.h].reshape(self.d, self.h); i += self.d * self.h
            self.b1 = theta[i:i + self.h]; i += self.h
        k = self.h if self.h else self.d
        self.w2 = theta[i:i + k]; i += k
        self.b2 = float(theta[i])

    def _forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1) if self.h else X
        z = a1 @ self.w2 + self.b2
        return a1, 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, X):
        _, p = self._forward(X)
        return np.column_stack([1 - p, p])

    def per_example_grads(self, X, y) -> np.ndarray:
        """Gradient of the logistic loss for every example, shape (n, P)."""
        a1, p = self._forward(X)
        dz = p - y
        g_w2 = dz[:, None] * a1
        g_b2 = dz[:, None]
        if not self.h:
            return np.hstack([g_w2, g_b2])
        dh = (dz[:, None] * self.w2[None, :]) * (1 - a1 ** 2)
        g_W1 = np.einsum("nd,nh->ndh", X, dh).reshape(len(X), -1)
        return np.hstack([g_W1, dh, g_w2, g_b2])
