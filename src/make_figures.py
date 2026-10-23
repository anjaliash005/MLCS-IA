"""Aggregates per-seed results into summary/statistics tables and draws all
figures used in the report.  Run after run_experiments.py.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.metrics import roc_curve

from common import RESULTS
from m2_dp import mia_advantage_bound
from models import BASELINE

TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")

# Reference categorical palette (fixed order) + text tokens
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6, "figure.dpi": 150, "savefig.bbox": "tight",
    "text.color": INK,
})

METRICS = ["test_acc", "test_f1", "test_auc", "gap", "sh_auc", "sh_acc", "sh_prec",
           "sh_rec", "sh_f1", "sh_adv", "sh_advmax", "ls_auc", "ls_adv", "ls_advmax"]


def random_adv_floor(n=192, trials=2000, seed=0):
    """95th percentile of max(TPR-FPR) for a random attacker on n vs n records."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n), np.zeros(n)]
    v = []
    for _ in range(trials):
        fpr, tpr, _ = roc_curve(y, rng.random(2 * n))
        v.append((tpr - fpr).max())
    return float(np.percentile(v, 95))


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "module", "config", "family", "param", "mode"]
    extra = [c for c in ["ncp", "homogeneous_frac", "t_closeness", "avg_size",
                         "recon_nrmse", "attr_bacc"] if c in df]
    g = df.groupby(keys, dropna=False, sort=False)
    out = g[METRICS + extra].mean()
    sd = g[METRICS + extra].std(ddof=1).add_suffix("_sd")
    out = out.join(sd)
    out["n_seeds"] = g.size()
    return out.reset_index()


def paired_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Each configuration vs the baseline (Model C, full API) on the same seeds."""
    base = df[(df.config == BASELINE) & (df["mode"] == "full")].set_index("seed")
    rows = []
    for (cfg, mode), g in df.groupby(["config", "mode"], sort=False):
        if cfg == BASELINE and mode == "full":
            continue
        g = g.set_index("seed").loc[base.index.intersection(g.set_index("seed").index)]
        row = {"config": cfg, "mode": mode, "module": g.module.iloc[0]}
        for m in ["sh_auc", "ls_auc", "test_acc"]:
            d = g[m] - base.loc[g.index, m]
            n = len(d)
            ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            try:
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = np.nan
            row.update({f"d_{m}": d.mean(), f"d_{m}_ci": ci, f"p_{m}": p})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.6,
                weight="bold", color=INK)
        off = 6.0 + 2.6 * title.count("\n")
        ax.text(x + w / 2, y + h - off, body, ha="center", va="top", fontsize=7.2,
                color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    box(1, 40, 18, 17, "Public datasets", "Pima Diabetes (768)\nBreast Cancer WDBC (569)\nSHA-256 verified", INK2)
    box(23, 40, 18, 17, "Public-bounds\npreprocessing", "zero->missing, clip,\nscale to [-1,1]\n(no privacy cost)", INK2)
    box(45, 40, 22, 17, "Stratified split / seed", "target members 25%\nnon-members 25%\nattacker shadow pool 50%", INK2)
    for x0, x1 in ((19.8, 22.2), (41.8, 44.2)):
        arrow(x0, 48.5, x1, 48.5)

    defs = [("M1  Attack + models", "shadow-model MIA target\nLogReg, MLP, RF matrix\nregularisation levels", S[3]),
            ("M2  Central DP", "DP-GD (zCDP accountant)\nLogReg / MLP, eps 0.5-8\nDP k-means", S[0]),
            ("M3  Local DP", "Piecewise / Duchi / Hybrid\nrandomised response\nLDP Naive Bayes (OUE)", S[1]),
            ("M4  PPDM", "Mondrian k-anonymity\n+ l-diversity, t-closeness\ninput noise, output limits", S[2]),
            ("M5  Compressive", "PCA / DCA / DUCA\nk-dim projections\nutility vs privacy task", S[6])]
    xs = [1, 21, 41, 61, 81]
    for (t, b, c), x in zip(defs, xs):
        box(x, 18, 17.5, 15, t, b, c)
        arrow(56, 39.2, x + 8.75, 33.8)

    box(23, 0.5, 26, 13, "Target + 8 shadow models", "same recipe incl. defence\n(Kerckhoffs' principle)\nAPI: full / rounded / label", S[7])
    box(55, 0.5, 26, 13, "Evaluation (10 seeds)", "utility: acc, F1, AUC, gap\nMIA: AUC, prec, rec, F1, adv\npaired Wilcoxon vs baseline", INK2)
    for x in xs:
        arrow(x + 8.75, 17.2, 36, 14.2)
    arrow(49.8, 7, 54.2, 7)
    ax.text(82.5, 7, "Shadow-model attack\n(Shokri 2017) +\nloss-threshold attack\n(Yeom 2018)", fontsize=7.2,
            color=INK2, va="center")
    save(fig, "fig01_architecture.png")


def fig_m1(summ, roc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    for i, ds in enumerate(["Pima", "BreastCancer"]):
        s = summ[(summ.dataset == ds) & (summ.module == "M1") & (summ["mode"] == "full")]
        ax.errorbar(s.gap, s.sh_auc, xerr=s.gap_sd, yerr=s.sh_auc_sd, fmt="o", color=S[i],
                    ms=7, capsize=2, lw=1, label=ds, mec="white", mew=1.2)
        for _, r in s.iterrows():
            ax.annotate(r.config.split(":")[0], (r.gap, r.sh_auc), xytext=(5, 3),
                        textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.99, 0.5, "random guess ", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("generalisation gap (train acc - test acc)")
    ax.set_ylabel("shadow-attack ROC-AUC")
    ax.set_title("Overfitting drives membership leakage")
    ax.legend(loc="upper left")
    ax = axes[1]
    for i, (lab, d) in enumerate(roc.items()):
        fpr, tpr, _ = roc_curve(d["m_true"], d["s_shadow"])
        ax.plot(fpr, tpr, color=S[i], label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Shadow-attack ROC (Pima, seed 0)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig02_m1_overfitting_roc.png")


def fig_m2(summ, sig, acct, km):
    s = summ[(summ.dataset == "Pima") & (summ.module == "M2")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    eps_pos = {0.5: 0, 1: 1, 2: 2, 4: 3, 8: 4, np.inf: 5.3}
    ticks = ["0.5", "1", "2", "4", "8", "∞"]
    for i, fam in enumerate(["DP-LR", "DP-MLP"]):
        f = s[s.family == fam].copy()
        f["x"] = f.param.map(eps_pos)
        f = f.sort_values("x")
        axes[0].errorbar(f.x, f.test_acc, yerr=f.test_acc_sd, color=S[i], marker="o", capsize=2,
                         label=fam.replace("DP-", "DP-GD "))
        axes[1].errorbar(f.x, f.sh_auc, yerr=f.sh_auc_sd, color=S[i], marker="o", capsize=2,
                         label=fam.replace("DP-", "DP-GD "))
    npl = s[s.family == "NP-MLP long"]
    base = summ[(summ.dataset == "Pima") & (summ.config == BASELINE) & (summ["mode"] == "full")]
    for ax, m in ((axes[0], "test_acc"), (axes[1], "sh_auc")):
        ax.axhline(float(npl[m].iloc[0]), color=S[3], ls=":", lw=1.5, label="NumPy MLP-128, long (non-DP)")
        ax.axhline(float(base[m].iloc[0]), color=S[7], ls="--", lw=1.2, label="Model C baseline (non-DP)")
        ax.set_xticks(list(eps_pos.values())); ax.set_xticklabels(ticks)
        ax.set_xlabel("privacy budget ε (δ = 1e-5)")
    axes[0].set_ylabel("test accuracy"); axes[0].set_title("Utility vs ε (Pima)")
    axes[1].set_ylabel("shadow-attack ROC-AUC"); axes[1].set_title("Membership leakage vs ε")
    axes[1].axhline(0.5, color=MUTED, lw=1)
    axes[0].legend(fontsize=7, loc="lower right")
    ax = axes[2]
    e = np.linspace(0.05, 8, 200)
    ax.plot(e, [mia_advantage_bound(x) for x in e], color=INK2, lw=1.5, label="DP bound (e^ε-1+2δ)/(e^ε+1)")
    for i, fam in enumerate(["DP-LR", "DP-MLP"]):
        f = s[(s.family == fam) & np.isfinite(s.param)]
        ax.scatter(f.param, f[["sh_advmax", "ls_advmax"]].max(1), color=S[i], zorder=3,
                   s=30, edgecolor="white", label=f"{fam} empirical max adv.")
    fl = random_adv_floor()
    ax.axhspan(0, fl, color=GRID, alpha=0.7, lw=0)
    ax.text(7.9, fl + 0.01, f"random-score floor ({fl:.2f})", ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("ε"); ax.set_ylabel("membership advantage (TPR - FPR)")
    ax.set_title("Empirical advantage is far below the bound")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout(); save(fig, "fig03_m2_dp_tradeoff.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for i, sg in enumerate(sorted(acct.sigma.unique())):
        a = acct[acct.sigma == sg]
        axes[0].plot(a.steps, a.eps, color=S[i], marker="o", ms=4, label=f"σ = {sg:g}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("training steps T (composition)"); axes[0].set_ylabel("ε (δ = 1e-5)")
    axes[0].set_title("Sequential composition via zCDP"); axes[0].legend(fontsize=7)
    axes[1].errorbar(km.eps, km.sse_ratio_mean, yerr=km.sse_ratio_std, color=S[0], marker="o", capsize=2)
    axes[1].axhline(1, color=MUTED, ls="--", lw=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("ε"); axes[1].set_ylabel("SSE / non-private SSE")
    axes[1].set_title("DP k-means (unsupervised, Pima, k = 2)")
    fig.tight_layout(); save(fig, "fig04_m2_composition_kmeans.png")


def fig_m3(summ, me, nbs):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    ax = axes[0]
    for i, m in enumerate(["Laplace", "Duchi", "Piecewise", "Hybrid"]):
        f = me[me.mech == m]
        ax.plot(f.eps, f.per_record_var_theory, color=S[i], lw=1.2, ls="--")
        ax.plot(f.eps, f.per_record_var_emp, color=S[i], marker="o", ls="none", label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    ax.set_xlabel("ε (per value)"); ax.set_ylabel("per-report variance")
    ax.set_title("LDP mechanisms: empirical (dots) vs theory")
    ax.legend(fontsize=7)
    ax = axes[1]
    for i, n in enumerate(sorted(nbs.n_users.unique())):
        f = nbs[nbs.n_users == n].groupby("eps").acc.agg(["mean", "std"]).reset_index()
        x = f.eps.replace(np.inf, 16)
        ax.errorbar(x, f["mean"], yerr=f["std"], color=S[i], marker="o", capsize=2,
                    label=f"n = {n:,}" + (" (real)" if n == 576 else " (bootstrap sim.)"))
    ax.axhline(500 / 768, color=MUTED, ls="--", lw=1)
    ax.text(0.55, 500 / 768 + 0.005, "majority class", fontsize=7, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "∞"])
    ax.set_xlabel("ε per user"); ax.set_ylabel("test accuracy")
    ax.set_title("LDP Naive Bayes needs many users"); ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    s = summ[(summ.dataset == "Pima") & (summ.module == "M3")]
    for i, fam in enumerate(["LDP-PM-LR", "LDP-NB"]):
        f = s[s.family == fam].copy(); f["x"] = f.param.replace(np.inf, 32)
        f = f.sort_values("x")
        ax.errorbar(f.x, f.test_acc, yerr=f.test_acc_sd, color=S[i], marker="o", capsize=2,
                    label=f"{fam} accuracy")
        ax.plot(f.x, f.sh_auc, color=S[i], marker="s", ls=":", label=f"{fam} attack AUC")
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "∞"])
    ax.set_xlabel("ε per user"); ax.set_title("LDP-trained models (n = 192)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout(); save(fig, "fig05_m3_ldp.png")


def fig_m4(summ, anon):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ax = axes[0]
    for i, l in enumerate([1, 2]):
        f = anon[anon.l == l]
        ax.plot(f.k, f.ncp, color=S[i], marker="o", label=f"NCP info loss, l = {l}")
        ax.plot(f.k, f.homogeneous_frac, color=S[i], marker="s", ls=":", label=f"homogeneous classes, l = {l}")
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 25, 50]); ax.set_xticklabels(["2", "5", "10", "25", "50"])
    ax.set_xlabel("k"); ax.set_ylabel("fraction")
    ax.set_title("k-anonymity: loss vs homogeneity exposure"); ax.legend(fontsize=7)
    ax = axes[1]
    s = summ[(summ.dataset == "Pima") & (summ.module == "M4") & (summ.family != "input-noise")]
    base = summ[(summ.dataset == "Pima") & (summ.config == BASELINE)]
    rows = pd.concat([base, s])
    labels = [("Model C, full API" if (r.config == BASELINE and r["mode"] == "full") else
               f"Model C, {r['mode']} API" if r.config == BASELINE else r.config) for _, r in rows.iterrows()]
    y = np.arange(len(rows))
    ax.barh(y - 0.2, rows.sh_auc, 0.38, color=S[0], label="shadow attack")
    ax.barh(y + 0.2, rows.ls_auc, 0.38, color=S[1], label="loss / label attack")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5); ax.invert_yaxis()
    ax.axvline(0.5, color=MUTED, lw=1); ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("attack ROC-AUC"); ax.set_title("PPDM defences vs MIA (Pima)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax = axes[2]
    ax.barh(y, rows.test_acc, 0.6, color=S[2], xerr=rows.test_acc_sd, error_kw=dict(lw=1, ecolor=INK2))
    ax.set_yticks(y); ax.set_yticklabels([""] * len(y)); ax.invert_yaxis()
    ax.set_xlim(0.6, 0.8); ax.set_xlabel("test accuracy"); ax.set_title("Utility")
    fig.tight_layout(); save(fig, "fig06_m4_ppdm.png")


def fig_m5(comp):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    p = comp[comp.dataset == "Pima"]
    raw = p[p.method == "Raw"]
    for i, m in enumerate(["PCA", "DCA", "DUCA"]):
        f = p[p.method == m].groupby("k")[["utility_acc", "attr_bacc", "recon_nrmse"]].mean()
        axes[0].plot(f.index, f.utility_acc, color=S[[0, 1, 6][i]], marker="o", label=m)
        axes[1].plot(f.index, f.attr_bacc, color=S[[0, 1, 6][i]], marker="o", label=m)
        axes[2].plot(f.index, f.recon_nrmse, color=S[[0, 1, 6][i]], marker="o", label=m)
    for ax, m, lab in ((axes[0], "utility_acc", "raw data"), (axes[1], "attr_bacc", "raw data"),
                       (axes[2], "recon_nrmse", None)):
        if lab:
            ax.axhline(raw[m].mean(), color=MUTED, ls="--", lw=1)
            ax.text(0.99, raw[m].mean(), "raw features ", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_xlabel("components k (of 10)")
    axes[0].set_ylabel("diabetes accuracy"); axes[0].set_title("Utility task (5.4)")
    axes[1].set_ylabel("balanced accuracy"); axes[1].set_title("Privacy task: infer Age ≥ 35")
    axes[1].axhline(0.5, color=MUTED, lw=1)
    axes[2].set_ylabel("normalised RMSE"); axes[2].set_title("Reconstruction error of x from z")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "fig07_m5_compressive.png")


def fig_frontier(summ):
    s = summ[(summ.dataset == "Pima") & (summ["mode"].isin(["full", "label", "rounded"]))].copy()
    s["att"] = s[["sh_auc", "ls_auc"]].max(1)
    colors = {"M1": S[3], "M2": S[0], "M3": S[1], "M4": S[2], "M5": S[6]}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for mod, c in colors.items():
        f = s[s.module == mod]
        ax.scatter(f.test_acc, f.att, color=c, s=36, edgecolor="white", lw=1, label=mod, zorder=3)
    for _, r in s.iterrows():
        if r.config == BASELINE or r.config.startswith(("DP-GD MLP-128 eps=1", "DCA k=1", "DCA k=4", "Mondrian k=25",
                                                        "E:", "DP-GD LogReg eps=4", "D:", "Input noise sigma=0.5")):
            lab = r.config if r["mode"] == "full" else f"C, {r['mode']} API"
            lab = lab.replace(" (baseline)", "").replace(" + L2 + early stop", " + L2/early stop")
            off = {"D:": (-8, -11), "DP-GD LogReg eps=4": (4, 6), "Input noise sigma=0.5": (-30, 7)}
            xy = next((v for k, v in off.items() if lab.startswith(k)), (5, 2))
            ax.annotate(lab, (r.test_acc, r.att), xytext=xy, textcoords="offset points",
                        fontsize=6.8, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("target test accuracy (utility) →")
    ax.set_ylabel("worst-case attack ROC-AUC (leakage) →")
    ax.set_title("Privacy–utility landscape, all configurations (Pima, mean of 10 seeds)")
    ax.legend(title="module", fontsize=7.5, title_fontsize=7.5, loc="upper left")
    fig.tight_layout(); save(fig, "fig08_frontier.png")


def main():
    summs = []
    for ds in ["Pima", "BreastCancer"]:
        df = pd.read_csv(os.path.join(TABLES, f"mia_runs_{ds}.csv"))
        sm = summarise(df)
        sm.to_csv(os.path.join(TABLES, f"summary_{ds}.csv"), index=False)
        paired_stats(df).to_csv(os.path.join(TABLES, f"stats_{ds}.csv"), index=False)
        summs.append(sm)
    summ = pd.concat(summs)
    roc = json.load(open(os.path.join(TABLES, "roc_seed0.json")))
    fig_architecture()
    fig_m1(summ, roc)
    fig_m2(summ, pd.read_csv(os.path.join(TABLES, "m2_sigma_used.csv")),
           pd.read_csv(os.path.join(TABLES, "m2_accountant.csv")),
           pd.read_csv(os.path.join(TABLES, "m2_dp_kmeans.csv")))
    fig_m3(summ, pd.read_csv(os.path.join(TABLES, "m3_mean_estimation.csv")),
           pd.read_csv(os.path.join(TABLES, "m3_ldp_nb_scaling.csv")))
    fig_m4(summ, pd.read_csv(os.path.join(TABLES, "m4_anonymity.csv")))
    fig_m5(pd.read_csv(os.path.join(TABLES, "m5_components.csv")))
    fig_frontier(summ)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
