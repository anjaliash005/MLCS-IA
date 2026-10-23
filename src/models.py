"""Module 1 - target-model complexity / regularisation matrix (IA-1 sec. 10.1).

Every recipe has the signature recipe(X, y, rng) -> fitted model, so the same
function trains the target and all of the attacker's shadow models.
"""
from __future__ import annotations

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _seed(rng):
    return int(rng.integers(0, 2 ** 31 - 1))


def logreg(X, y, rng):                       # Model A: low complexity, moderate L2
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def mlp_small(X, y, rng):                    # Model B: medium complexity, low reg
    return MLPClassifier((32,), alpha=1e-4, max_iter=1000,
                         random_state=_seed(rng)).fit(X, y)


def mlp_large(X, y, rng):                    # Model C: high complexity, ~no reg (BASELINE)
    return MLPClassifier((128, 128), alpha=1e-6, max_iter=1000, tol=0,
                         n_iter_no_change=1000, random_state=_seed(rng)).fit(X, y)


def mlp_large_reg(X, y, rng):                # Model D: high complexity, high reg
    return MLPClassifier((128, 128), alpha=3.0, max_iter=1000, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=30,
                         random_state=_seed(rng)).fit(X, y)


def random_forest(X, y, rng):                # Model E: fully grown trees
    return RandomForestClassifier(200, random_state=_seed(rng), n_jobs=1).fit(X, y)


M1_MODELS = {
    "A: LogReg (L2)": logreg,
    "B: MLP-32": mlp_small,
    "C: MLP-128x2 (baseline)": mlp_large,
    "D: MLP-128x2 + L2 + early stop": mlp_large_reg,
    "E: Random forest (deep)": random_forest,
}
BASELINE = "C: MLP-128x2 (baseline)"
