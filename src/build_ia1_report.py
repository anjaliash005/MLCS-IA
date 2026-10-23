"""Builds reports/IA1_Report_Anjali_Ashtankar_16014223012.pdf
(Problem Formulation and Preliminary Implementation).

Run after run_ia1_baseline.py and run_experiments.py (for the mechanism checks).
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
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from sklearn.metrics import roc_curve

from common import PIMA_BOUNDS, RESULTS, ROOT, load_pima
from m2_dp import mia_advantage_bound, sigma_for
from make_figures import GRID, INK, INK2, MUTED, S
from pdfkit import P, bullets, f3, figure, strip_dates, styles, table

AUTHOR, ROLL, BATCH = "Anjali Ashtankar", "16014223012", "B1"
TITLE = "Membership Inference Attack on Machine Learning Models: A Healthcare Privacy Use Case"
SUB = "Problem formulation, five-module mapping and preliminary implementation"
IA1 = os.path.join(RESULTS, "ia1")
TABLES = os.path.join(RESULTS, "tables")
OUT = os.path.join(ROOT, "reports", f"IA1_Report_{AUTHOR.replace(' ', '_')}_{ROLL}.pdf")

st = styles()
W = A4[0] - 4.0 * cm
H1 = lambda t: Paragraph(t, st["h1"])
H2 = lambda t: Paragraph(t, st["h2"])
H3 = lambda t: Paragraph(t, st["h3"])
B = lambda t: P(t, st["body"])
fig = lambda name, cap, w=W: figure(os.path.join(IA1, name), w, cap, st)

base = pd.read_csv(os.path.join(IA1, "baseline_seed0.csv")).set_index("model")
C, D = "C: MLP-128x2 (baseline)", "D: MLP-128x2 + L2 + early stop"
bC, bD, bA = base.loc[C], base.loc[D], base.loc["A: LogReg (L2)"]
log_text = open(os.path.join(IA1, "baseline_log.txt")).read()
prov = json.load(open(os.path.join(TABLES, "provenance.json")))
pima = load_pima()
raw = pima.raw


# ----------------------------------------------------------------------------
# IA-1 figures
# ----------------------------------------------------------------------------
def save(f, name):
    f.savefig(os.path.join(IA1, name), dpi=200, bbox_inches="tight")
    plt.close(f)


def fig_workflow():
    f, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, t, b, c, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white",
                                    ec=c, lw=1.6, ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h - 2, t, ha="center", va="top", fontsize=7.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 2 - 3.4 * (1 + t.count("\n")), b, ha="center", va="top", fontsize=6.8,
                color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    steps = [("1. Public\ndata", "Pima (768)\nhash-verified"), ("2. Pre-\nprocess", "zero -> missing\npublic bounds\n[-1, 1]"),
             ("3. Split", "members 192\nnon-members 192\nshadow pool 384"),
             ("4. Train\ntarget", "models A-D\n(complexity x\nregularisation)"),
             ("5. Shadow\nmodels", "8 x same recipe\nmember / non-\nmember outputs"),
             ("6. Attack\nmodel", "logistic\nregression on\nconfidences"),
             ("7. Evaluate", "utility + attack\nAUC, prec, rec,\nF1, advantage")]
    for i, (t, b) in enumerate(steps):
        x = 1 + i * 14.1
        box(x, 23, 12.4, 19, t, b, S[7] if i in (4, 5) else INK2)
        if i:
            arrow(x - 1.6, 32, x - 0.1, 32)
    mods = [("M2 central DP", "DP-GD + accountant", S[0]), ("M3 local DP", "client-side noise", S[1]),
            ("M4 PPDM", "k-anon., noise, output", S[2]), ("M5 compressive", "PCA / DCA", S[6])]
    for i, (t, b, c) in enumerate(mods):
        x = 2 + i * 21
        box(x, 4, 18, 10.5, t, b, c, dashed=True)
        arrow(x + 9, 15.3, 50, 22.2)
    ax.text(99, 9, "dashed = defences\nprototyped in IA-1,\nattacked in IA-2", ha="right", va="center",
            fontsize=7, color=INK2)
    save(f, "ia1_fig1_workflow.png")


def fig_baseline():
    roc = json.load(open(os.path.join(TABLES, "roc_seed0.json")))
    f, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    names = list(base.index)
    y = np.arange(len(names))
    ax.barh(y - 0.2, base.gap, 0.38, color=S[3], label="generalisation gap")
    ax.barh(y + 0.2, base.sh_auc - 0.5, 0.38, color=S[0], label="attack AUC - 0.5")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("value"); ax.set_title("Gap and leakage go together (seed 0)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    for i, (lab, key) in enumerate([("C: baseline MLP-128x2", "C: baseline MLP"), ("D: regularised MLP", "D: regularised MLP")]):
        d = roc[key]
        fpr, tpr, _ = roc_curve(d["m_true"], d["s_shadow"])
        ax.plot(fpr, tpr, color=S[i], label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Shadow-attack ROC (seed 0)"); ax.legend(fontsize=7.5, loc="lower right")
    f.tight_layout(); save(f, "ia1_fig2_baseline.png")


def fig_mechanisms():
    me = pd.read_csv(os.path.join(TABLES, "m3_mean_estimation.csv"))
    acct = pd.read_csv(os.path.join(TABLES, "m2_accountant.csv"))
    an = pd.read_csv(os.path.join(TABLES, "m4_anonymity.csv"))
    comp = pd.read_csv(os.path.join(TABLES, "m5_components.csv"))
    f, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    ax = axes[0]
    for i, sg in enumerate([10, 20, 40, 80]):
        a = acct[acct.sigma == sg]
        ax.plot(a.steps, a.eps, color=S[i], marker="o", ms=3, label=f"σ = {sg}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("steps T"); ax.set_ylabel("ε (δ = 1e-5)")
    ax.set_title("M2: DP accountant"); ax.legend(fontsize=6.5)
    ax = axes[1]
    for i, m in enumerate(["Laplace", "Duchi", "Piecewise"]):
        g = me[me.mech == m]
        ax.plot(g.eps, g.per_record_var_theory, color=S[i], lw=1.2, ls="--")
        ax.plot(g.eps, g.per_record_var_emp, color=S[i], marker="o", ls="none", ms=5, label=m)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.25, 1, 4]); ax.set_xticklabels(["0.25", "1", "4"])
    ax.set_xlabel("ε"); ax.set_ylabel("per-report variance"); ax.set_title("M3: LDP check (dots = empirical)")
    ax.legend(fontsize=6.5)
    ax = axes[2]
    g = an[an.l == 1]
    ax.plot(g.k, g.ncp, color=S[2], marker="o", label="info loss (NCP)")
    ax.plot(g.k, g.homogeneous_frac, color=S[7], marker="s", ls=":", label="homogeneous classes")
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 25, 50]); ax.set_xticklabels(["2", "5", "10", "25", "50"])
    ax.set_xlabel("k"); ax.set_title("M4: Mondrian k-anonymity"); ax.legend(fontsize=6.5)
    ax = axes[3]
    p = comp[(comp.dataset == "Pima") & (comp.seed == 0)]
    for i, m in enumerate(["PCA", "DCA"]):
        g = p[p.method == m]
        ax.plot(g.k, g.utility_acc, color=S[[0, 1][i]], marker="o", ms=4, label=f"{m} utility acc")
    ax.axhline(p[p.method == "Raw"].utility_acc.iloc[0], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("components k"); ax.set_title("M5: utility of projections"); ax.legend(fontsize=6.5)
    f.tight_layout(); save(f, "ia1_fig3_mechanisms.png")


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------
def title_page():
    s = [Spacer(1, 2.2 * cm), Paragraph("Security and Privacy for Machine Learning", st["subtitle"]),
         Paragraph("IA-1: Problem Formulation and Preliminary Implementation", st["subtitle"]),
         Spacer(1, 1.0 * cm), Paragraph(TITLE, st["title"]), Paragraph(SUB, st["subtitle"]),
         Spacer(1, 1.6 * cm)]
    s.append(table([["Field", "Details"], ["Student", AUTHOR], ["Roll number", ROLL], ["Batch", BATCH],
                    ["Assessment", "IA-1 (20 marks) - individual research-based assessment"],
                    ["Dataset", "Pima Indians Diabetes (public benchmark)"],
                    ["Code & data", '<link href="https://github.com/anjaliash005/MLCS-IA" color="#1c5cab">https://github.com/anjaliash005/MLCS-IA</link>'],
                    ["Entry point", "`src/run_ia1_baseline.py` and modules in `src/` (see Section 4)"]],
                   [4.2 * cm, W - 4.2 * cm], st, font_size=9))
    s.append(Spacer(1, 1.2 * cm))
    s.append(P("*Builds on the Module 1 research-foundation document. All numbers are produced by the code "
               "in `src/`; experiments use only public data and locally trained models.*", st["cap"]))
    s.append(PageBreak())
    return s


def abstract():
    return [H1("Abstract"),
            B("A machine-learning model trained on patient records can reveal whether a particular person's record "
              "was in its training set. This is a **membership inference attack (MIA)**, and in healthcare, "
              "membership alone can disclose a diagnosis. This IA-1 report formulates the problem for a "
              "diabetes-risk classifier, maps it to all five course modules, prepares the Pima Indians Diabetes "
              "dataset with privacy-safe preprocessing, and implements a black-box shadow-model attack as the "
              "baseline."),
            B(f"On the first split, the unregularised MLP memorises its training data (train accuracy "
              f"{bC.train_acc:.2f}, test {bC.test_acc:.2f}) and the attack identifies members with ROC-AUC "
              f"**{bC.sh_auc:.3f}**, against {bA.sh_auc:.3f} for logistic regression. Prototypes of the defence "
              f"mechanisms for Modules 2-5 (DP accountant, LDP mechanisms, Mondrian k-anonymity, PCA/DCA "
              f"projections) were implemented and checked. IA-2 will attack each defence with the same harness and "
              f"compare privacy against utility over repeated runs.")]


def stage1():
    s = [H1("1. Problem Definition and Module Mapping")]
    s.append(H2("1.1 Significance"))
    s.append(B("Hospitals increasingly train predictive models on patient data and expose them through prediction "
               "services. Protecting the database is not enough: the released model itself can leak information "
               "about its training records. If a model is trained on patients enrolled in a diabetes programme, "
               "learning that a person's record was in the training set can reveal that they took part in the "
               "programme. MIAs are among the most studied privacy attacks on ML (Shokri et al., 2017), and they "
               "are a practical way to *measure* how much a model leaks."))
    s.append(H2("1.2 Problem statement"))
    s.append(B("*Given black-box access to a diabetes-risk classifier and auxiliary data from the same population, "
               "how accurately can an attacker decide whether a specific patient's record was used for training, "
               "and which privacy-preserving techniques from the course reduce this leakage at an acceptable cost "
               "in predictive utility?*"))
    s.append(H2("1.3 Objectives"))
    s += bullets([
        "**O1** Build a reproducible black-box MIA (shadow models + attack classifier) against a healthcare classifier.",
        "**O2** Measure how leakage depends on model complexity and regularisation (baseline matrix A-D).",
        "**O3** Map the problem to all five modules and choose one defence family from each.",
        "**O4** Prepare a public dataset with preprocessing that does not itself leak private information.",
        "**O5** Prototype and check the defence mechanisms, and plan their evaluation for IA-2."], st)
    s.append(H2("1.4 Scope"))
    s += bullets([
        "**In scope:** binary tabular classification, black-box access to confidence scores, attackers with "
        "auxiliary same-distribution data, defences from Modules 1-5, utility and privacy metrics.",
        "**Out of scope:** white-box or gradient access, attacks on real services or real people, large neural "
        "networks and unstructured data, legal compliance analysis.",
        "**Expected outcome:** a quantified baseline leakage, a working attack harness, working prototypes of the "
        "defence mechanisms, and an evaluation plan that compares defences on one common privacy metric."], st)
    s.append(H2("1.5 Module-wise concept mapping"))
    rows = [["Module", "Relevant syllabus topics", "Relevance to this problem", "Planned treatment"],
            ["**M1** Privacy considerations in ML", "1.1 privacy in the AI era; 1.2 threats and attacks; 1.3 PPML; "
             "1.4 PPDM overview", "Defines the threat (MIA), the healthcare harm, and model choice / regularisation "
             "as the first defence", "Implemented: attack + model matrix (IA-1)"],
            ["**M2** Differential privacy for ML", "2.1 mechanisms and properties; 2.2 DP in ML; 2.3 DP supervised; "
             "2.4 DP unsupervised", "DP bounds any single record's influence on the model, which directly limits "
             "membership inference", "Accountant prototyped (IA-1); DP-GD evaluated against the attack (IA-2); "
             "DP clustering as a demonstration"],
            ["**M3** Local DP for ML", "3.1 LDP mechanisms; 3.2 advanced LDP; 3.3 Piecewise mechanism; 3.4 LDP "
             "Naive Bayes", "Patients perturb their own records before the hospital sees them, so the collector "
             "need not be trusted", "Mechanisms implemented and checked against theory (IA-1); LDP-trained models "
             "attacked (IA-2)"],
            ["**M4** Privacy-preserving data mining", "4.1 privacy in data mining; 4.2 modifying input; 4.3 "
             "publishing data; 4.4 beyond k-anonymity; 4.5 modifying output", "Anonymising the training table and "
             "restricting API outputs are common real-world responses to leakage", "Mondrian k-anonymity "
             "prototyped (IA-1); input noise, l-diversity and output restriction attacked (IA-2); 4.1 analytical"],
            ["**M5** Compressive privacy for ML", "5.1 compressive privacy; 5.2 mechanisms; 5.3 CP for ML; 5.4 "
             "utility-task accuracy", "Releasing low-dimensional projections instead of raw records can keep the "
             "diagnosis signal while hiding other attributes", "PCA/DCA prototyped with utility vs k (IA-1); "
             "privacy task and MIA evaluated (IA-2)"]]
    s.append(table(rows, [3.0 * cm, 4.0 * cm, 5.4 * cm, W - 12.4 * cm], st, font_size=7))
    return s


def stage2():
    s = [H1("2. Conceptual Understanding")]
    s.append(H2("2.1 Concept notes"))
    notes = [
        ("Membership inference (M1)", "An attacker observes the model's output M(x) for a record x and decides "
         "whether x was in the training set D. Shadow models trained by the attacker on similar data produce "
         "outputs with known membership, which are used to train an attack classifier.",
         "Attacker has same-distribution data and knows the training recipe; black-box access to scores.",
         "Black-box, realistic, quantitative (AUC, advantage).",
         "Average-case metric; a weak attack can underestimate risk; needs balanced evaluation."),
        ("Overfitting and regularisation (M1)", "Models that fit training data much better than unseen data "
         "give members higher confidence, so the train-test gap predicts leakage (Yeom et al., 2018). L2 penalties, "
         "early stopping and simpler models reduce memorisation.",
         "Leakage is driven mainly by memorisation.", "Cheap, often improves accuracy too.",
         "No formal guarantee; strong attackers may still succeed."),
        ("Differential privacy (M2)", "A mechanism is (ε, δ)-DP if adding or removing one record changes the "
         "output distribution by at most a factor e^ε (plus δ). DP-SGD/DP-GD clips each record's gradient and adds "
         "Gaussian noise; composition tracks the total ε over training steps.",
         "Trusted curator; neighbouring datasets differ by one record.",
         "Provable bound on any attacker's advantage: TPR - FPR <= (e^ε - 1 + 2δ)/(e^ε + 1).",
         "Utility loss at small ε; ε is hard to interpret; hyperparameter tuning costs privacy."),
        ("Local differential privacy (M3)", "Each user randomises their own data before sending it (Laplace, "
         "Duchi, Piecewise mechanisms; randomised response; unary encoding for frequencies). Classifiers such as "
         "Naive Bayes can be built from the perturbed reports.",
         "Untrusted collector; many users.", "No trust in the hospital needed; protects data at collection.",
         "Much noisier than central DP; needs large user numbers."),
        ("k-anonymity and beyond (M4)", "A released table is k-anonymous if every combination of "
         "quasi-identifiers (age, pregnancies, BMI) is shared by at least k records. l-diversity also requires "
         "varied sensitive values in each group. Other PPDM options perturb inputs or restrict outputs.",
         "Attacker links records through quasi-identifiers.", "Intuitive and widely used for data publishing.",
         "No formal guarantee; homogeneity and background-knowledge attacks; information loss."),
        ("Compressive privacy (M5)", "Clients release z = W<super>T</super>x with k << d. PCA keeps maximum "
         "variance; DCA (Discriminant Component Analysis) keeps the directions that separate the utility label, "
         "so the utility task stays accurate while other information (privacy task, reconstruction) is lost.",
         "Projection is public; utility label known when W is learned.",
         "Reduces what is released; can target a specific privacy attribute.",
         "No formal guarantee; protects released data, not necessarily the trained model.")]
    rows = [["Concept", "Principle", "Assumptions", "Advantages", "Limitations"]]
    rows += [[f"**{a}**", b, c, d, e] for a, b, c, d, e in notes]
    s.append(table(rows, [2.6 * cm, 5.6 * cm, 2.9 * cm, 2.9 * cm, W - 14.0 * cm], st, font_size=6.8))
    s.append(H2("2.2 How the concepts address the problem"))
    s.append(B("The attack from Module 1 is used as a **common measuring instrument**. Every other module offers a "
               "different place to intervene: during training (central DP), at data collection (LDP), on the "
               "released data or API (PPDM), or on the shared representation (compressive privacy). Measuring "
               "each one with the same attack and the same utility metrics makes them directly comparable, and "
               "shows which defences give formal guarantees and which only reduce empirical leakage."))
    s.append(H2("2.3 Treatment of each concept"))
    s.append(table([["Treatment", "Concepts"],
                    ["Implemented (IA-1)", "Shadow-model MIA; model matrix A-D; public-bounds preprocessing; zCDP "
                     "accountant; Laplace / Duchi / Piecewise / Hybrid mechanisms; Mondrian k-anonymity; PCA and "
                     "DCA projections"],
                    ["Demonstrated (IA-1)", "LDP mechanism variance vs theory; ε vs training steps (composition); "
                     "k-anonymity information loss vs k; projection utility vs k"],
                    ["Experimentally evaluated (IA-2)", "Each defence attacked by the MIA over repeated splits: DP-GD, "
                     "LDP training and LDP Naive Bayes, k-anonymity / l-diversity, input noise, output restriction, "
                     "PCA / DCA / DUCA"],
                    ["Discussed analytically", "Privacy complications in the AI era (1.1); PPDM overview (1.4, 4.1); "
                     "knowledge distillation as an alternative defence"]],
                   [4.0 * cm, W - 4.0 * cm], st, font_size=7.6))
    s.append(H2("2.4 Methodology and workflow"))
    s.append(fig("ia1_fig1_workflow.png", "Figure 1. Proposed workflow. Solid boxes form the IA-1 baseline "
                 "pipeline; dashed boxes are defences prototyped in IA-1 and attacked in IA-2."))
    s.append(H2("2.5 Justification for concept selection"))
    s += bullets([
        "**Shadow-model MIA** is the reference black-box attack and matches the IA-1 threat model (prediction API "
        "returning confidence scores).",
        "**DP-GD** is the standard formal defence for training and gives a bound that can be compared with the "
        "measured attack success.",
        "**Piecewise mechanism and LDP Naive Bayes** are the LDP methods named in the syllabus (3.3, 3.4) that "
        "suit numeric and categorical clinical data.",
        "**Mondrian k-anonymity** suits numeric quasi-identifiers such as age and BMI; l-diversity covers 4.4.",
        "**DCA** is the compressive-privacy method designed for a known utility label (diabetes), with PCA as "
        "the unsupervised comparison."], st)
    return s


def stage3():
    pp = prov["pima"]
    miss = pp["missing_after_zero_to_nan"]
    s = [PageBreak(), H1("3. Dataset Identification and Preliminary Preparation")]
    s.append(H2("3.1 Dataset description and suitability"))
    s.append(B("The **Pima Indians Diabetes** dataset contains 768 records of female patients aged 21 or older of "
               "Pima heritage, with eight clinical measurements and a binary outcome (diabetes onset within five "
               "years). It is suitable because (i) it is a real healthcare prediction task matching the use case, "
               "(ii) it is public and widely used, so no private data is put at risk, (iii) it is small enough "
               "that models can overfit, which makes membership leakage measurable, and (iv) its age, pregnancies "
               "and BMI columns act as quasi-identifiers for the k-anonymity work."))
    s.append(table([["Item", "Details"],
        ["Source", "jbrownlee/Datasets GitHub mirror of the UCI ML Repository file "
                   "(raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)"],
        ["Original publisher", "National Institute of Diabetes and Digestive and Kidney Diseases; Smith et al. (1988); "
                               "donated to UCI in 1990"],
        ["Collection method", "Clinical examination: 2-hour oral glucose tolerance test, diastolic blood pressure, "
                              "triceps skinfold, 2-hour serum insulin, BMI, diabetes pedigree function, age"],
        ["Access date", f"{prov['generated_utc'][:10]} (UTC), HTTPS download"],
        ["Licence / usage terms", "Not stated by the mirror (no licence file). Kaggle's 'uciml' redistribution lists "
                                  "CC0. Treated as unknown; used only for non-commercial academic benchmarking"],
        ["Version / integrity", f"Mirror commit 357ed4a (2018); SHA-256 {pp['sha256']}"],
        ["Size", f"{pp['rows_original']} rows x 9 columns; outcome 0/1 = {pp['class_counts']['0']}/{pp['class_counts']['1']}"],
        ["Limitations", "Small; single population and sex; zeros used for missing values; ethical concerns about "
                        "continued benchmark use of data from an Indigenous community"]],
        [3.4 * cm, W - 3.4 * cm], st, font_size=7.4))
    s.append(H2("3.2 Feature profile"))
    rows = [["Feature", "Unit", "Public bound", "Missing (zeros)", "Median (observed)"]]
    units = {"Pregnancies": "count", "Glucose": "mg/dL (2-h OGTT)", "BloodPressure": "mm Hg (diastolic)",
             "SkinThickness": "mm (triceps)", "Insulin": "µU/mL (2-h)", "BMI": "kg/m²",
             "DiabetesPedigreeFunction": "score", "Age": "years"}
    for c, (lo, hi) in PIMA_BOUNDS.items():
        rows.append([c, units[c], f"{lo}-{hi}", str(int(raw[c].isna().sum())), f"{raw[c].median():g}"])
    s.append(table(rows, [4.4 * cm, 3.6 * cm, 2.6 * cm, 2.8 * cm, W - 13.4 * cm], st, font_size=7.4))
    s.append(H2("3.3 Preliminary preparation log"))
    s.append(table([["Step", "Operation", "Result"],
        ["1", "Download and record SHA-256; verify the hash on every load", "Pipeline stops if the file changes"],
        ["2", "Check shape and labels", f"768 x 9, no duplicate header, outcome 0/1 = 500/268"],
        ["3", "Recode physiologically impossible zeros as missing (Glucose, BloodPressure, SkinThickness, Insulin, BMI)",
         f"Missing: Glucose {miss['Glucose']}, BP {miss['BloodPressure']}, Skin {miss['SkinThickness']}, "
         f"Insulin {miss['Insulin']}, BMI {miss['BMI']}"],
        ["4", "Clip to fixed public clinical bounds and scale to [-1, 1]",
         "Data-independent scaling: no statistics of private records are used, so later DP/LDP guarantees hold"],
        ["5", "Impute missing values with the centre of the public range; add missing indicators for "
              "SkinThickness and Insulin", "10 model features; no rows dropped"],
        ["6", "Stratified split: 25% target members, 25% non-members (test set), 50% attacker shadow pool",
         "192 / 192 / 384 records, positive rate 0.349 in each part"]],
        [1.0 * cm, 8.4 * cm, W - 9.4 * cm], st, font_size=7.2))
    s.append(B("**Why not standard scaling?** Fitting a scaler or median imputer on the training records would "
               "use private statistics. Any DP or LDP guarantee would then have to account for them. Fixing the "
               "bounds from clinical plausibility avoids this at negligible cost."))
    return s


def stage4():
    s = [PageBreak(), H1("4. Initial Implementation")]
    s.append(H2("4.1 Tools and implementation choices"))
    s.append(B("Python 3.9 with NumPy, pandas, scikit-learn (models, metrics), SciPy and Matplotlib, pinned in "
               "`requirements.lock.txt`. Each training procedure is written as a *recipe* function "
               "`recipe(X, y, rng) -> model`. The attacker trains its shadow models with exactly the same recipe "
               "as the target, which is the strongest realistic assumption (Kerckhoffs' principle) and lets any "
               "future defence be dropped in unchanged."))
    s.append(table([["Component", "File", "Inputs -> outputs"],
        ["Data loading and preprocessing", "`src/common.py`", "raw CSV -> scaled feature matrix X, labels y, splits"],
        ["Target models A-D", "`src/models.py`", "(X, y) -> fitted classifier with predict_proba"],
        ["Shadow-model attack", "`src/attacks.py`", "recipe, data, split -> attack scores and metrics"],
        ["Baseline experiment", "`src/run_ia1_baseline.py`", "-> `results/ia1/baseline_seed0.csv`, `baseline_log.txt`"],
        ["Mechanism prototypes", "`src/m2_dp.py`, `src/m3_ldp.py`, `src/m4_ppdm.py`, `src/m5_compressive.py`",
         "accountant, LDP mechanisms, Mondrian partitioner, PCA/DCA projections"]],
        [4.2 * cm, 6.0 * cm, W - 10.2 * cm], st, font_size=7.4))
    s.append(H3("Attack procedure"))
    s += bullets([
        "Train the target on the 192 member records.",
        "Train 8 shadow models, each on 192 records from the attacker's pool, with 192 other pool records as its "
        "non-members.",
        "For every shadow record, compute features from the model output: true-class probability, loss, entropy, "
        "maximum probability, correctness, logit and label. Label each record member (1) or non-member (0).",
        "Fit a standardised logistic-regression attack classifier on these labelled shadow outputs.",
        "Score the target's 192 members and 192 non-members and compute the attack metrics."], st)
    s.append(H2("4.2 Baseline results (single split, seed 0)"))
    rows = [["Model", "Train acc", "Test acc", "Test F1", "Gap", "Attack AUC", "Attack acc", "Prec", "Rec", "F1", "Adv"]]
    for m, r in base.iterrows():
        rows.append([m, f3(r.train_acc), f3(r.test_acc), f3(r.test_f1), f3(r.gap), f3(r.sh_auc), f3(r.sh_acc),
                     f3(r.sh_prec), f3(r.sh_rec), f3(r.sh_f1), f3(r.sh_adv)])
    s.append(KeepTogether([P("**Table 1.** Baseline utility and membership-inference results on Pima (seed 0).", st["cap"]),
                           table(rows, [4.2 * cm] + [(W - 4.2 * cm) / 10] * 10, st, font_size=6.8)]))
    s.append(Spacer(1, 6))
    s.append(fig("ia1_fig2_baseline.png", "Figure 2. Left: generalisation gap and attack AUC above chance for models "
                 "A-D. Right: ROC curve of the shadow attack against the baseline and the regularised MLP."))
    s.append(H2("4.3 Prototype checks of the defence mechanisms"))
    eps4 = sigma_for(4, 200)
    s.append(B(f"Before attacking the defences in IA-2, each mechanism was checked in isolation (Fig. 3). "
               f"**M2:** the zCDP accountant reproduces sequential composition (ε grows with the number of noisy "
               f"steps). For 200 steps, ε = 4 needs noise multiplier σ = {eps4:.1f}, and the advantage bound at ε = 4 "
               f"is {mia_advantage_bound(4):.3f}. **M3:** the empirical variance of the Laplace, Duchi and Piecewise "
               f"mechanisms matches the theoretical formulas, which confirms the implementations. **M4:** Mondrian "
               f"partitioning of the full table shows information loss rising with k while the share of records in "
               f"diagnosis-homogeneous groups falls. **M5:** a single DCA component already matches the accuracy of "
               f"the raw features, while PCA needs several components."))
    s.append(fig("ia1_fig3_mechanisms.png", "Figure 3. Mechanism-level checks: DP accounting, LDP mechanism variance "
                 "(dashed = theory), k-anonymity trade-off, and utility of PCA/DCA projections (seed 0)."))
    s.append(H2("4.4 Execution instructions"))
    s.append(P("`git clone https://github.com/anjaliash005/MLCS-IA.git && cd MLCS-IA`", st["code"]))
    s.append(P("`python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt`", st["code"]))
    s.append(P("`.venv/bin/python src/run_ia1_baseline.py`", st["code"]))
    s.append(H3("Output log"))
    for line in log_text.strip().split("\n"):
        if line.startswith("public bounds"):
            line = "public bounds   : (see Section 3.2)"
        s.append(P(f"`{line.replace(' ', '&nbsp;').replace('<', '&lt;').replace('>', '&gt;')}`" if line else "&nbsp;",
                   st["code"]))
    return s


def stage5():
    s = [H1("5. Preliminary Evaluation and Documentation")]
    s.append(H2("5.1 Initial observations"))
    s += bullets([
        f"**Leakage is real and measurable.** The unregularised MLP (C) fits its training set perfectly (train "
        f"accuracy {bC.train_acc:.2f}) and the attack reaches AUC {bC.sh_auc:.3f}, recall {bC.sh_rec:.2f} and "
        f"advantage {bC.sh_adv:.2f}.",
        f"**Leakage tracks overfitting.** Models with small train-test gaps (A, B, D) give attack AUCs between "
        f"{base.drop(C).sh_auc.min():.2f} and {base.drop(C).sh_auc.max():.2f}, close to random guessing.",
        f"**Higher capacity did not buy accuracy.** Logistic regression (A) has test accuracy {bA.test_acc:.3f} vs "
        f"{bC.test_acc:.3f} for model C, so on this small dataset the leaky model is also the less useful one.",
        f"**Regularisation needs care.** On this split the heavily regularised model D collapses towards the majority "
        f"class (test F1 {bD.test_f1:.3f}): it leaks nothing, but it is barely useful. One split is not enough to "
        f"judge a defence, which motivates repeated runs and tuning in IA-2."], st)
    s.append(H2("5.2 Evaluation measures"))
    s.append(table([["Aspect", "Measures"],
        ["Utility", "Test accuracy, F1-score, ROC-AUC on held-out non-members; generalisation gap"],
        ["Privacy (attack)", "Attack ROC-AUC, accuracy, precision, recall, F1, membership advantage (TPR - FPR)"],
        ["Formal privacy", "ε (and δ) for DP / LDP configurations; comparison of measured advantage with the DP bound"],
        ["Data release", "Information loss (NCP), homogeneity exposure, l-diversity (M4); attribute-inference accuracy "
                         "and reconstruction error (M5)"],
        ["Statistics", "Mean ± standard deviation over repeated random splits; paired tests against the baseline"]],
        [3.4 * cm, W - 3.4 * cm], st, font_size=7.6))
    s.append(H2("5.3 Known limitations"))
    s += bullets([
        "Single split: results can vary noticeably between splits on 768 records.",
        "Only one attacker (shadow model + logistic regression); stronger attacks may find more leakage.",
        "One small dataset from one population; external validity is limited.",
        "Defence mechanisms have only been checked in isolation, not yet attacked."], st)
    s.append(H2("5.4 Remaining work for IA-2"))
    s.append(table([["Area", "Planned work"],
        ["Statistics", "Repeat over 10 seeds; report mean ± SD; paired Wilcoxon tests against the baseline"],
        ["Attack", "Add a loss-threshold attack and rounded / label-only prediction APIs"],
        ["M2", "Train DP-GD logistic regression and MLP at ε = 0.5-8; compare measured advantage with the DP bound; DP k-means demo"],
        ["M3", "Train on Piecewise-perturbed records; LDP Naive Bayes; study how many users LDP needs"],
        ["M4", "Attack models trained on k-anonymous data; add l-diversity, input noise and output restriction"],
        ["M5", "Measure attribute inference, reconstruction and MIA for PCA, DCA and privacy-aware DUCA"],
        ["Generalisation", "Replicate on a second public medical dataset (Breast Cancer Wisconsin)"],
        ["Documentation", "Final technical report and research-manuscript draft"]],
        [3.0 * cm, W - 3.0 * cm], st, font_size=7.6))
    s.append(H2("5.5 Individual-contribution statement"))
    s.append(B(f"This IA-1 is my individual work ({AUTHOR}, {ROLL}, {BATCH}). I contributed:"))
    s += bullets([
        "Problem formulation, objectives, scope and the five-module concept mapping.",
        "Dataset selection, provenance record and the privacy-safe preprocessing design (zeros as missing, public "
        "clinical bounds, missing indicators).",
        "Implementation of the shadow-model attack harness, the model matrix and the baseline experiment.",
        "Prototype implementations and checks of the DP accountant, LDP mechanisms, Mondrian k-anonymity and "
        "PCA/DCA projections.",
        "Analysis of the preliminary results and the evaluation plan for IA-2."], st)
    s.append(H1("References"))
    refs = [
        "R. Shokri, M. Stronati, C. Song and V. Shmatikov, \"Membership inference attacks against machine learning models,\" *IEEE S&amp;P*, 2017.",
        "S. Yeom, I. Giacomelli, M. Fredrikson and S. Jha, \"Privacy risk in machine learning: Analyzing the connection to overfitting,\" *IEEE CSF*, 2018.",
        "C. Dwork and A. Roth, \"The algorithmic foundations of differential privacy,\" *Found. Trends Theor. Comput. Sci.*, 2014.",
        "M. Abadi *et al.*, \"Deep learning with differential privacy,\" *ACM CCS*, 2016.",
        "M. Bun and T. Steinke, \"Concentrated differential privacy,\" *TCC*, 2016.",
        "N. Wang *et al.*, \"Collecting and analyzing multidimensional data with local differential privacy,\" *IEEE ICDE*, 2019.",
        "E. Yilmaz, M. Al-Rubaie and J. M. Chang, \"Locally differentially private naive Bayes classification,\" arXiv:1905.01039, 2019.",
        "K. LeFevre, D. J. DeWitt and R. Ramakrishnan, \"Mondrian multidimensional k-anonymity,\" *IEEE ICDE*, 2006.",
        "A. Machanavajjhala *et al.*, \"l-diversity: Privacy beyond k-anonymity,\" *ACM TKDD*, 2007.",
        "S. Y. Kung, \"Compressive privacy: From information/estimation theory to machine learning,\" *IEEE Signal Processing Magazine*, 2017.",
        "J. M. Chang, D. Zhuang and G. D. Samaraweera, *Privacy-Preserving Machine Learning*. Manning, 2023.",
        "J. W. Smith *et al.*, \"Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,\" *Proc. SCAMC*, 1988.",
    ]
    s += [P(f"[{i}] {r}", st["ref"]) for i, r in enumerate(refs, 1)]
    return s


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColorRGB(0.35, 0.35, 0.33)
    if doc.page > 1:
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, f"IA-1 | Security and Privacy for ML | {AUTHOR} ({ROLL}, {BATCH})")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    fig_workflow(); fig_baseline(); fig_mechanisms()
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title=TITLE, author=f"{AUTHOR} ({ROLL})",
                            subject="IA-1 Security and Privacy for Machine Learning")
    doc.build(title_page() + abstract() + stage1() + stage2() + stage3() + stage4() + stage5(),
              onFirstPage=on_page, onLaterPages=on_page)
    strip_dates(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
