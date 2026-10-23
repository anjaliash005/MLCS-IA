"""Module 1 - membership inference attacks (black-box).

Two attackers are evaluated against every target model:
  * Shadow-model attack (Shokri et al., 2017) - the IA-1 design. The attacker
    trains shadow models with the *same training recipe* as the target
    (including any defence: Kerckhoffs' principle), labels their outputs as
    member / non-member and fits a logistic-regression attack classifier.
  * Loss-threshold attack (Yeom et al., 2018) - shadow-free; a record is
    called a member when the model's loss on it is low. Its decision
    threshold is calibrated on the shadow models.

`output_mode` models the prediction API (Module 4.5, output modification):
  "full"    - full probability vector
  "rounded" - probabilities rounded to one decimal place
  "label"   - only the predicted class (label-only attacker)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import shadow_splits, utility_metrics

EPS = 1e-6


def api_output(model, X, mode: str) -> np.ndarray:
    p = model.predict_proba(X)
    if mode == "rounded":
        p = np.round(p, 1)
    elif mode == "label":
        p = np.eye(2)[(p[:, 1] >= 0.5).astype(int)]
    return p


def attack_features(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    pt = np.clip(probs[np.arange(len(y)), y], EPS, 1 - EPS)
    pc = np.clip(probs, EPS, 1)
    loss = -np.log(pt)
    entropy = -(pc * np.log(pc)).sum(1)
    correct = (probs.argmax(1) == y).astype(float)
    logit = np.log(pt) - np.log(1 - pt)
    return np.column_stack([pt, loss, entropy, probs.max(1), correct, logit, y])


def loss_score(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    pt = np.clip(probs[np.arange(len(y)), y], EPS, 1 - EPS)
    return np.log(pt)  # higher = more member-like


def _binary_metrics(y_true, y_pred) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"acc": (tp + tn) / len(y_true), "prec": prec, "rec": rec, "f1": f1,
            "adv": rec - fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _max_advantage(y_true, score) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def _best_threshold(y_true, score) -> float:
    s = np.unique(score)
    cands = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]]) if len(s) > 1 else np.array([s[0]])
    accs = [((score >= t).astype(int) == y_true).mean() for t in cands]
    return float(cands[int(np.argmax(accs))])


def run_mia(recipe, X, y, split, seed: int, n_shadows: int = 8,
            output_modes=("full",), return_details: bool = False) -> dict:
    """Train target + shadows with `recipe(X, y, rng) -> model` once, then
    attack the target through each prediction-API mode. Returns {mode: metrics}."""
    rng = np.random.default_rng(seed)
    tin = split.target_in
    target = recipe(X[tin], y[tin], rng)
    shadows = [(recipe(X[s_in], y[s_in], rng), s_in, s_out) for s_in, s_out in
               shadow_splits(split.shadow_pool, y, n_shadows, len(tin), seed)]
    return {m: _attack(target, shadows, X, y, split, m, return_details)
            for m in output_modes}


def _attack(target, shadows, X, y, split, output_mode, return_details):
    tin, tout = split.target_in, split.target_out
    # ---- shadow models -> attack training set
    feats, labels, lscores = [], [], []
    for sm, s_in, s_out in shadows:
        for idx, lab in ((s_in, 1), (s_out, 0)):
            p = api_output(sm, X[idx], output_mode)
            feats.append(attack_features(p, y[idx]))
            lscores.append(loss_score(p, y[idx]))
            labels.append(np.full(len(idx), lab))
    F, L, LS = np.vstack(feats), np.concatenate(labels), np.concatenate(lscores)
    attack = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    attack.fit(F, L)
    thr = _best_threshold(L, LS)

    # ---- evaluate on the target's true members / non-members (balanced)
    idx = np.concatenate([tin, tout])
    m_true = np.concatenate([np.ones(len(tin), int), np.zeros(len(tout), int)])
    p_t = api_output(target, X[idx], output_mode)
    s_shadow = attack.predict_proba(attack_features(p_t, y[idx]))[:, 1]
    s_loss = loss_score(p_t, y[idx])

    res = {}
    for name, score, pred in (("sh", s_shadow, (s_shadow >= 0.5).astype(int)),
                              ("ls", s_loss, (s_loss >= thr).astype(int))):
        auc = roc_auc_score(m_true, score) if len(np.unique(score)) > 1 else 0.5
        bm = _binary_metrics(m_true, pred)
        res.update({f"{name}_auc": auc, f"{name}_acc": bm["acc"],
                    f"{name}_prec": bm["prec"], f"{name}_rec": bm["rec"],
                    f"{name}_f1": bm["f1"], f"{name}_adv": bm["adv"],
                    f"{name}_advmax": _max_advantage(m_true, score)})

    # ---- utility of the target model (non-members are unseen test data)
    u_test = utility_metrics(target, X[tout], y[tout])
    u_train = utility_metrics(target, X[tin], y[tin])
    res.update({"test_acc": u_test["acc"], "test_f1": u_test["f1"],
                "test_auc": u_test["auc"], "train_acc": u_train["acc"],
                "gap": u_train["acc"] - u_test["acc"]})
    if return_details:
        res["_details"] = {"m_true": m_true, "s_shadow": s_shadow,
                           "s_loss": s_loss, "target": target}
    return res
