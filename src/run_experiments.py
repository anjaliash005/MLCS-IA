"""Runs every experiment for IA-2 and writes CSV/JSON results to results/.

    .venv/bin/python src/run_experiments.py            # full run (10 seeds)
    .venv/bin/python src/run_experiments.py --quick    # 2 seeds, smoke test
"""
from __future__ import annotations

import os

# One BLAS thread per worker; parallelism comes from joblib across seeds.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import m2_dp
import m3_ldp
import m4_ppdm
import m5_compressive as m5
from attacks import run_mia
from common import (DATA_PROCESSED, PIMA_FILE, RESULTS, bc_file_info,
                    load_breast_cancer_ds, load_pima, make_split, sha256)
from models import BASELINE, M1_MODELS, logreg, mlp_large, mlp_large_reg

TABLES = os.path.join(RESULTS, "tables")
N_SHADOWS = 8


# ---------------------------------------------------------------------------
# Configuration list. make(ds, split) -> (recipe, X_used, extras)
# ---------------------------------------------------------------------------
def plain(recipe):
    return lambda ds, sp: (recipe, ds.X, {})


def cp_make(method, k, base):
    def make(ds, sp):
        tin = sp.target_in
        Xi, yi = ds.X[tin], ds.y[tin]
        s = ds.privacy_attr[tin] if ds.privacy_attr is not None else None
        W = m5.fit_projection(method, k, Xi, yi, s)
        mu = Xi.mean(0)
        extras = cp_privacy_measures(ds, sp, W, mu)
        return m5.projected_recipe(W, mu, base), ds.X, extras
    return make


def cp_privacy_measures(ds, sp, W, mu):
    """Attacker holds the shadow pool (with labels for the privacy attribute)
    and sees the compressed vectors z of the target's training records."""
    proj = (lambda X: X) if W is None else (lambda X: (X - mu) @ W)
    Zp, Zt = proj(ds.X[sp.shadow_pool]), proj(ds.X[sp.target_in])
    out = {}
    rec = LinearRegression().fit(Zp, ds.X[sp.shadow_pool])
    err = ds.X[sp.target_in] - rec.predict(Zt)
    ref = ds.X[sp.target_in] - ds.X[sp.shadow_pool].mean(0)
    out["recon_nrmse"] = float(np.sqrt((err ** 2).mean() / (ref ** 2).mean()))
    if ds.privacy_attr is not None:
        rf = RandomForestClassifier(200, min_samples_leaf=3, random_state=0)
        rf.fit(Zp, ds.privacy_attr[sp.shadow_pool])
        out["attr_bacc"] = float(balanced_accuracy_score(ds.privacy_attr[sp.target_in], rf.predict(Zt)))
    return out


def kanon_make(k, l, base):
    def make(ds, sp):
        qi = [ds.feature_names.index(c) for c in m4_ppdm.PIMA_QI]
        Xi, yi = ds.X[sp.target_in], ds.y[sp.target_in]
        rep = m4_ppdm.anonymity_report(Xi[:, qi], yi, m4_ppdm.mondrian(Xi[:, qi], yi, k, l))
        return m4_ppdm.kanon_recipe(k, l, qi, base), ds.X, rep
    return make


def nb_make(eps):
    def make(ds, sp):
        return m3_ldp.nb_recipe(eps), m3_ldp.discretise_pima(ds.raw), {}
    return make


DP_EPS = [0.5, 1, 2, 4, 8]
DP_LR = dict(hidden=0, steps=200, lr=1.0)
DP_MLP = dict(hidden=128, steps=200, lr=0.5)
NP_LONG = dict(hidden=128, steps=2000, lr=2.0)


def configs_for(name: str) -> list:
    C = []
    add = lambda module, cfg, family, param, make, modes=("full",): C.append(
        dict(module=module, config=cfg, family=family, param=param, make=make, modes=modes))

    for cfg, r in M1_MODELS.items():
        modes = ("full", "rounded", "label") if cfg == BASELINE else ("full",)
        add("M1", cfg, "M1 model matrix", np.nan, plain(r), modes)

    eps_list = DP_EPS if name == "Pima" else [1, 4]
    add("M2", "NumPy MLP-128, long training (non-private)", "NP-MLP long", np.inf,
        plain(m2_dp.dp_gd_recipe(np.inf, **NP_LONG)))
    for e in eps_list + [np.inf]:
        add("M2", f"DP-GD LogReg eps={e}", "DP-LR", e, plain(m2_dp.dp_gd_recipe(e, **DP_LR)))
        add("M2", f"DP-GD MLP-128 eps={e}", "DP-MLP", e, plain(m2_dp.dp_gd_recipe(e, **DP_MLP)))

    if name == "Pima":
        for e in [1, 4, 8, 16]:
            add("M3", f"LDP-Piecewise LogReg eps={e}", "LDP-PM-LR", e,
                plain(m3_ldp.ldp_feature_recipe(e, logreg, "Piecewise")))
        for e in [1, 2, 4, 8, np.inf]:
            add("M3", f"LDP Naive Bayes eps={e}", "LDP-NB", e, nb_make(e))
        for k in [5, 10, 25]:
            add("M4", f"Mondrian k={k}", "k-anon", k, kanon_make(k, 1, mlp_large))
        add("M4", "Mondrian k=10, l=2", "k-anon+l", 10, kanon_make(10, 2, mlp_large))

    for s in [0.25, 0.5]:
        add("M4", f"Input noise sigma={s}", "input-noise", s,
            plain(m4_ppdm.input_noise_recipe(s, mlp_large)))

    cp = ([("PCA", 2), ("PCA", 4), ("DCA", 1), ("DCA", 2), ("DCA", 4), ("DUCA", 2)]
          if name == "Pima" else [("PCA", 2), ("PCA", 5), ("DCA", 1), ("DCA", 5)])
    for meth, k in cp:
        add("M5", f"{meth} k={k}", meth, k, cp_make(meth, k, mlp_large))
    return C


def run_job(ds, cfg, seed):
    sp = make_split(ds.y, seed)
    t = time.time()
    recipe, X, extras = cfg["make"](ds, sp)
    res = run_mia(recipe, X, ds.y, sp, seed, N_SHADOWS, cfg["modes"])
    rows = []
    for mode, r in res.items():
        row = {"dataset": ds.name, "module": cfg["module"], "config": cfg["config"],
               "family": cfg["family"], "param": cfg["param"], "mode": mode, "seed": seed}
        row.update(r)
        row.update(extras)
        row["seconds"] = time.time() - t
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Stand-alone (non-MIA) experiments
# ---------------------------------------------------------------------------
def standalone(pima, bc, seeds):
    out = {}
    # M2: composition - eps as a function of steps for several noise levels
    out["m2_accountant"] = pd.DataFrame(
        [{"sigma": s, "steps": T, "eps": m2_dp.eps_for(s, T)}
         for s in [5, 10, 20, 40, 80] for T in [1, 10, 50, 100, 200, 500, 1000, 2000]])
    out["m2_sigma_used"] = pd.DataFrame(
        [{"eps": e, "steps": 200, "sigma": m2_dp.sigma_for(e, 200),
          "adv_bound": m2_dp.mia_advantage_bound(e)} for e in DP_EPS])
    # M2: DP k-means (unsupervised)
    rows, base = m2_dp.dp_kmeans_experiment(pima.X, k=2)
    out["m2_dp_kmeans"] = pd.DataFrame(rows).assign(nonprivate_sse=base)
    # M3: LDP mean-estimation correctness check
    out["m3_mean_estimation"] = pd.DataFrame(
        m3_ldp.mean_estimation_experiment([0.25, 0.5, 1, 2, 4, 8]))
    # M3: LDP-NB utility vs number of reporting users (bootstrap simulation)
    Xd = m3_ldp.discretise_pima(pima.raw)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sp = make_split(pima.y, seed)
        train = np.concatenate([sp.target_in, sp.shadow_pool])   # 576 records
        test = sp.target_out
        for n in [576, 5000, 50000]:
            idx = train if n == 576 else rng.choice(train, n, replace=True)
            for e in [0.5, 1, 2, 4, 8, np.inf]:
                m = m3_ldp.nb_recipe(e)(Xd[idx], pima.y[idx], rng)
                acc = (m.predict_proba(Xd[test]).argmax(1) == pima.y[test]).mean()
                rows.append({"seed": seed, "n_users": n, "eps": e, "acc": acc})
    out["m3_ldp_nb_scaling"] = pd.DataFrame(rows)
    # M4: anonymity / information loss on the full Pima table
    qi = [pima.feature_names.index(c) for c in m4_ppdm.PIMA_QI]
    rows = []
    for l in [1, 2]:
        for k in [2, 5, 10, 25, 50]:
            cl = m4_ppdm.mondrian(pima.X[:, qi], pima.y, k, l)
            rows.append({"k": k, "l": l, **m4_ppdm.anonymity_report(pima.X[:, qi], pima.y, cl)})
    out["m4_anonymity"] = pd.DataFrame(rows)
    # M5: utility / privacy as a function of the number of components
    rows = []
    for ds in (pima, bc):
        d = ds.X.shape[1]
        for seed in seeds:
            sp = make_split(ds.y, seed)
            tin, tout = sp.target_in, sp.target_out
            s = ds.privacy_attr[tin] if ds.privacy_attr is not None else None
            methods = ["PCA", "DCA"] + (["DUCA"] if s is not None else [])
            for meth in methods + ["Raw"]:
                for k in ([d] if meth == "Raw" else range(1, d + 1)):
                    if meth == "Raw":
                        W, mu = None, None
                        Zi, Zo = ds.X[tin], ds.X[tout]
                    else:
                        W = m5.fit_projection(meth, k, ds.X[tin], ds.y[tin], s)
                        mu = ds.X[tin].mean(0)
                        Zi, Zo = (ds.X[tin] - mu) @ W, (ds.X[tout] - mu) @ W
                    clf = LogisticRegression(max_iter=2000).fit(Zi, ds.y[tin])
                    row = {"dataset": ds.name, "method": meth, "k": k, "seed": seed,
                           "utility_acc": (clf.predict(Zo) == ds.y[tout]).mean()}
                    row.update(cp_privacy_measures(ds, sp, W, mu))
                    rows.append(row)
    out["m5_components"] = pd.DataFrame(rows)
    return out


def roc_details(pima):
    """Per-record attack scores (seed 0) for ROC plots."""
    sp = make_split(pima.y, 0)
    rows = {}
    for label, recipe in [("C: baseline MLP", mlp_large),
                          ("D: regularised MLP", mlp_large_reg),
                          ("E: random forest", M1_MODELS["E: Random forest (deep)"]),
                          ("DP-MLP eps=1", m2_dp.dp_gd_recipe(1, **DP_MLP))]:
        r = run_mia(recipe, pima.X, pima.y, sp, 0, N_SHADOWS, ("full",), True)["full"]
        d = r["_details"]
        rows[label] = {"m_true": d["m_true"].tolist(), "s_shadow": d["s_shadow"].tolist(),
                       "s_loss": d["s_loss"].tolist()}
    return rows


# ---------------------------------------------------------------------------
def save_processed(pima, bc):
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    paths = {}
    p = os.path.join(DATA_PROCESSED, "pima_scaled.csv")
    pd.DataFrame(pima.X, columns=pima.feature_names).assign(Outcome=pima.y).to_csv(p, index=False)
    paths["pima_scaled"] = p
    p = os.path.join(DATA_PROCESSED, "pima_discretised.csv")
    pd.DataFrame(m3_ldp.discretise_pima(pima.raw), columns=list(m3_ldp.PIMA_BINS)).assign(
        Outcome=pima.y).to_csv(p, index=False)
    paths["pima_discretised"] = p
    p = os.path.join(DATA_PROCESSED, "breast_cancer_scaled.csv")
    pd.DataFrame(bc.X, columns=bc.feature_names).assign(Malignant=bc.y).to_csv(p, index=False)
    paths["breast_cancer_scaled"] = p
    raw = pima.raw
    bc_path, bc_hash = bc_file_info()
    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "pima": {
            "file": os.path.relpath(PIMA_FILE, os.path.dirname(DATA_PROCESSED)),
            "sha256": sha256(PIMA_FILE),
            "rows_original": int(len(raw)), "columns_original": 9,
            "class_counts": {str(k): int(v) for k, v in raw["Outcome"].value_counts().items()},
            "missing_after_zero_to_nan": {c: int(raw[c].isna().sum()) for c in raw.columns},
            "final_shape_scaled": list(pima.X.shape),
        },
        "breast_cancer": {"file": bc_path, "sha256": bc_hash,
                          "rows": int(bc.X.shape[0]), "features": int(bc.X.shape[1]),
                          "class_counts": {"malignant": int(bc.y.sum()),
                                           "benign": int((1 - bc.y).sum())}},
        "processed_files": {k: {"path": os.path.relpath(v, os.path.dirname(DATA_PROCESSED)),
                                "sha256": sha256(v)} for k, v in paths.items()},
    }
    with open(os.path.join(TABLES, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(2 if args.quick else 10))
    os.makedirs(TABLES, exist_ok=True)

    pima, bc = load_pima(), load_breast_cancer_ds()
    save_processed(pima, bc)
    t0 = time.time()
    for ds in (pima, bc):
        jobs = [(cfg, s) for cfg in configs_for(ds.name) for s in seeds]
        print(f"{ds.name}: {len(jobs)} jobs", flush=True)
        out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_job)(ds, c, s) for c, s in jobs)
        df = pd.DataFrame([r for rows in out for r in rows])
        df.to_csv(os.path.join(TABLES, f"mia_runs_{ds.name}.csv"), index=False)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    for k, df in standalone(pima, bc, seeds).items():
        df.to_csv(os.path.join(TABLES, f"{k}.csv"), index=False)
    with open(os.path.join(TABLES, "roc_seed0.json"), "w") as f:
        json.dump(roc_details(pima), f)
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
