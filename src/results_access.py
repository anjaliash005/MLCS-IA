"""Read-only access to aggregated results for the report/manuscript builders.
Every number printed in the PDFs comes from these CSV/JSON files."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import RESULTS
from models import BASELINE

TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")


class Results:
    def __init__(self):
        self.summ = {ds: pd.read_csv(os.path.join(TABLES, f"summary_{ds}.csv"))
                     for ds in ("Pima", "BreastCancer")}
        self.stats = {ds: pd.read_csv(os.path.join(TABLES, f"stats_{ds}.csv"))
                      for ds in ("Pima", "BreastCancer")}
        self.runs = {ds: pd.read_csv(os.path.join(TABLES, f"mia_runs_{ds}.csv"))
                     for ds in ("Pima", "BreastCancer")}
        rd = lambda n: pd.read_csv(os.path.join(TABLES, n))
        self.sigma = rd("m2_sigma_used.csv")
        self.kmeans = rd("m2_dp_kmeans.csv")
        self.mean_est = rd("m3_mean_estimation.csv")
        self.nb_scaling = rd("m3_ldp_nb_scaling.csv")
        self.anon = rd("m4_anonymity.csv")
        self.comp = rd("m5_components.csv")
        self.prov = json.load(open(os.path.join(TABLES, "provenance.json")))
        self.n_seeds = int(self.summ["Pima"].n_seeds.max())
        self.runtime_s = self.runs["Pima"].groupby("seed").seconds.sum().sum() + \
            self.runs["BreastCancer"].groupby("seed").seconds.sum().sum()

    def row(self, ds, config, mode="full"):
        s = self.summ[ds]
        r = s[(s.config == config) & (s["mode"] == mode)]
        if r.empty:
            raise KeyError((ds, config, mode))
        return r.iloc[0]

    def v(self, ds, config, metric, mode="full"):
        return float(self.row(ds, config, mode)[metric])

    def base(self, ds, metric, mode="full"):
        return self.v(ds, BASELINE, metric, mode)

    def stat(self, ds, config, mode="full"):
        s = self.stats[ds]
        r = s[(s.config == config) & (s["mode"] == mode)]
        return r.iloc[0] if len(r) else None

    def worst_auc(self, ds, config, mode="full"):
        r = self.row(ds, config, mode)
        return float(max(r.sh_auc, r.ls_auc))

    def comp_mean(self, ds, method, k, metric):
        c = self.comp
        f = c[(c.dataset == ds) & (c.method == method) & (c.k == k)]
        return float(f[metric].mean())

    def nb(self, n, eps):
        f = self.nb_scaling
        e = np.inf if eps == "inf" else eps
        return float(f[(f.n_users == n) & (f.eps == e)].acc.mean())
