"""IA-1 preliminary implementation: baseline membership inference attack on
the Pima diabetes task (single split, seed 0, shadow-model attack, models A-D).

    .venv/bin/python src/run_ia1_baseline.py

Writes results/ia1/baseline_seed0.csv and results/ia1/baseline_log.txt.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import numpy as np
import pandas as pd

from attacks import run_mia
from common import PIMA_BOUNDS, RESULTS, load_pima, make_split, sha256, PIMA_FILE
from models import logreg, mlp_large, mlp_large_reg, mlp_small

OUT = os.path.join(RESULTS, "ia1")
SEED = 0
MODELS = {
    "A: LogReg (L2)": logreg,
    "B: MLP-32": mlp_small,
    "C: MLP-128x2 (baseline)": mlp_large,
    "D: MLP-128x2 + L2 + early stop": mlp_large_reg,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    ds = load_pima()
    raw = ds.raw
    log("=== IA-1 baseline: membership inference on Pima (seed 0) ===")
    log(f"raw file sha256 : {sha256(PIMA_FILE)}")
    log(f"records         : {len(raw)}  (Outcome 0/1 = {(raw.Outcome == 0).sum()}/{(raw.Outcome == 1).sum()})")
    log("zero -> missing : " + ", ".join(f"{c}={int(raw[c].isna().sum())}"
                                         for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]))
    log(f"features        : {ds.X.shape[1]} (8 scaled to [-1,1] with public bounds + 2 missing indicators)")
    log("public bounds   : " + "; ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in PIMA_BOUNDS.items()))
    sp = make_split(ds.y, SEED)
    log(f"split           : target members {len(sp.target_in)}, non-members {len(sp.target_out)}, "
        f"shadow pool {len(sp.shadow_pool)}")
    log(f"positive rate   : members {ds.y[sp.target_in].mean():.3f}, non-members {ds.y[sp.target_out].mean():.3f}, "
        f"pool {ds.y[sp.shadow_pool].mean():.3f}")
    log("attack          : 8 shadow models (same recipe as target) -> LR attack classifier")
    log()
    rows = []
    for name, recipe in MODELS.items():
        t = time.time()
        r = run_mia(recipe, ds.X, ds.y, sp, SEED)["full"]
        r = {k: v for k, v in r.items() if not k.startswith("ls_")}
        rows.append({"model": name, **r, "seconds": time.time() - t})
        log(f"[{name}]  {time.time() - t:5.1f}s")
        log(f"   utility : train acc {r['train_acc']:.3f} | test acc {r['test_acc']:.3f} | test F1 {r['test_f1']:.3f} "
            f"| test AUC {r['test_auc']:.3f} | gap {r['gap']:.3f}")
        log(f"   attack  : AUC {r['sh_auc']:.3f} | acc {r['sh_acc']:.3f} | prec {r['sh_prec']:.3f} | "
            f"rec {r['sh_rec']:.3f} | F1 {r['sh_f1']:.3f} | adv {r['sh_adv']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "baseline_seed0.csv"), index=False)
    log()
    log("saved results/ia1/baseline_seed0.csv")
    with open(os.path.join(OUT, "baseline_log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
