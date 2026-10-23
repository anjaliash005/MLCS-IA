"""Builds reports/IA2_Technical_Report_Anjali_Ashtankar_16014223012.pdf
from the result tables and figures. Run after make_figures.py."""
from __future__ import annotations

import os

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer)

from common import ROOT
from m2_dp import mia_advantage_bound, DELTA
from models import BASELINE
from pdfkit import P, bullets, f3, figure, pfmt, strip_dates, styles, table
from results_access import FIGS, Results

AUTHOR = "Anjali Ashtankar"
ROLL = "16014223012"
BATCH = "B1"
TITLE = ("Membership Inference as a Common Yardstick: Evaluating Five Families of "
         "Privacy-Preserving Defences for Healthcare Machine Learning")
OUT = os.path.join(ROOT, "reports", f"IA2_Technical_Report_{AUTHOR.replace(' ', '_')}_{ROLL}.pdf")

R = Results()
st = styles()
W = A4[0] - 4.0 * cm
fig = lambda name, cap, w=W: figure(os.path.join(FIGS, name), w, cap, st)
H1 = lambda t: Paragraph(t, st["h1"])
H2 = lambda t: Paragraph(t, st["h2"])
H3 = lambda t: Paragraph(t, st["h3"])
B = lambda t: P(t, st["body"])


def pct(x):
    return f"{100 * x:.1f}%"


def ms(ds, cfg, m, mode="full", nd=3):
    r = R.row(ds, cfg, mode)
    return f3(r[m], r[m + "_sd"], nd)


# ----------------------------------------------------------------------------
# Key numbers used in the narrative
# ----------------------------------------------------------------------------
P_ = "Pima"
C = BASELINE
D = "D: MLP-128x2 + L2 + early stop"
E = "E: Random forest (deep)"
NPL = "NumPy MLP-128, long training (non-private)"
b_auc, b_acc, b_gap = R.base(P_, "sh_auc"), R.base(P_, "test_acc"), R.base(P_, "gap")
b_ls = R.base(P_, "ls_auc")
lab_auc = R.worst_auc(P_, C, "label")
rnd_auc = R.worst_auc(P_, C, "rounded")
d_auc, d_acc = R.v(P_, D, "sh_auc"), R.v(P_, D, "test_acc")
e_auc = R.v(P_, E, "sh_auc")
_s0 = R.runs[P_]
ia1_auc = float(_s0[(_s0.seed == 0) & (_s0.config == C) & (_s0["mode"] == "full")].sh_auc.iloc[0])
A_ = "A: LogReg (L2)"
a_acc, a_auc = R.v(P_, A_, "test_acc"), R.worst_auc(P_, A_)
lrinf_acc = R.v(P_, "DP-GD LogReg eps=inf", "test_acc")
dp4_acc, dp4_auc = R.v(P_, "DP-GD MLP-128 eps=4", "test_acc"), R.worst_auc(P_, "DP-GD MLP-128 eps=4")
dp1_acc = R.v(P_, "DP-GD MLP-128 eps=1", "test_acc")
dplr4_acc, dplr4_auc = R.v(P_, "DP-GD LogReg eps=4", "test_acc"), R.worst_auc(P_, "DP-GD LogReg eps=4")
dpinf_auc = R.worst_auc(P_, "DP-GD MLP-128 eps=inf")
npl_auc, npl_acc = R.v(P_, NPL, "sh_auc"), R.v(P_, NPL, "test_acc")
nb4_acc = R.v(P_, "LDP Naive Bayes eps=4", "test_acc")
nbinf_acc = R.v(P_, "LDP Naive Bayes eps=inf", "test_acc")
pm16_acc = R.v(P_, "LDP-Piecewise LogReg eps=16", "test_acc")
k25_auc, k25_acc = R.worst_auc(P_, "Mondrian k=25"), R.v(P_, "Mondrian k=25", "test_acc")
k5_auc = R.worst_auc(P_, "Mondrian k=5")
noise5_auc, noise5_acc = R.worst_auc(P_, "Input noise sigma=0.5"), R.v(P_, "Input noise sigma=0.5", "test_acc")
dca1_acc, dca1_auc = R.v(P_, "DCA k=1", "test_acc"), R.worst_auc(P_, "DCA k=1")
an = R.anon
homog_k5_l1 = float(an[(an.k == 5) & (an.l == 1)].homogeneous_frac.iloc[0])
homog_k10_l1 = float(an[(an.k == 10) & (an.l == 1)].homogeneous_frac.iloc[0])
ncp_k10_l1 = float(an[(an.k == 10) & (an.l == 1)].ncp.iloc[0])
ncp_k10_l2 = float(an[(an.k == 10) & (an.l == 2)].ncp.iloc[0])
raw_attr = R.comp_mean(P_, "Raw", 10, "attr_bacc")
raw_util = R.comp_mean(P_, "Raw", 10, "utility_acc")
pca1_attr, dca1_attr, duca1_attr = (R.comp_mean(P_, m, 1, "attr_bacc") for m in ("PCA", "DCA", "DUCA"))
pca1_util, dca1_util, duca1_util = (R.comp_mean(P_, m, 1, "utility_acc") for m in ("PCA", "DCA", "DUCA"))
pca2_rec, dca2_rec = R.comp_mean(P_, "PCA", 2, "recon_nrmse"), R.comp_mean(P_, "DCA", 2, "recon_nrmse")
sig = R.sigma.set_index("eps")
NS = R.n_seeds
prov = R.prov
bc_b_auc, bc_b_acc = R.base("BreastCancer", "sh_auc"), R.base("BreastCancer", "test_acc")
bc_e_auc = R.v("BreastCancer", E, "sh_auc")


def word_change(new, old, up="increased", down="reduced"):
    return up if new > old else down


# ----------------------------------------------------------------------------
def title_page():
    s = [Spacer(1, 2.2 * cm),
         Paragraph("Security and Privacy for Machine Learning", st["subtitle"]),
         Paragraph("IA-2: Implementation, Evaluation and Technical Outcomes", st["subtitle"]),
         Spacer(1, 1.0 * cm), Paragraph(TITLE, st["title"]), Spacer(1, 0.6 * cm),
         Paragraph("Final Technical Report (individual submission)", st["subtitle"]),
         Spacer(1, 1.6 * cm)]
    s.append(table([["Field", "Details"],
                    ["Student", AUTHOR], ["Roll number", ROLL], ["Batch", BATCH],
                    ["Assessment", "IA-2 (20 marks) - continuation of IA-1 research foundation"],
                    ["IA-1 topic", "Membership Inference Attack on ML Models: A Healthcare Privacy Use Case"],
                    ["Datasets", "Pima Indians Diabetes (primary); Breast Cancer Wisconsin Diagnostic (replication)"],
                    ["Code & data", '<link href="https://github.com/anjaliash005/MLCS-IA" color="#1c5cab">https://github.com/anjaliash005/MLCS-IA</link>' + " (src/, data/, results/) - see Appendix A"]],
                   [4.2 * cm, W - 4.2 * cm], st, font_size=9))
    s.append(Spacer(1, 1.2 * cm))
    s.append(P("*All results in this report were produced by the code in `src/` and are read "
               "directly from `results/tables/`; no value was typed in by hand. Experiments use only "
               "public datasets and locally trained models - no real person, service or "
               "organisation was attacked.*", st["cap"]))
    s.append(PageBreak())
    return s


def abstract():
    return [H1("Abstract"),
            B(f"Machine-learning models trained on medical records can reveal whether a particular patient's "
              f"record was part of the training set. IA-1 framed this as a black-box **membership inference "
              f"attack (MIA)** on a diabetes-risk classifier. IA-2 turns that attack into a **common "
              f"yardstick** and uses it to evaluate one defence family from every course module: "
              f"regularisation and model choice (Module 1), central differential privacy via DP gradient "
              f"descent with a zCDP accountant (Module 2), local differential privacy with the Piecewise "
              f"mechanism and LDP Naive Bayes (Module 3), Mondrian k-anonymity, l-diversity, input "
              f"randomisation and output restriction (Module 4), and compressive privacy with PCA/DCA/DUCA "
              f"projections (Module 5). Each configuration is trained as a target model plus eight shadow "
              f"models over {NS} random splits, attacked with a shadow-model attack and a loss-threshold attack, "
              f"and compared with the overfitting baseline using paired Wilcoxon tests."),
            B(f"On the Pima data the baseline MLP leaks clearly (shadow-attack AUC {b_auc:.3f}, train-test gap "
              f"{b_gap:.3f}). Strong regularisation removes most of the leakage (AUC {d_auc:.3f}) with no significant "
              f"change in accuracy, and a plain logistic regression is both the most accurate and essentially leak-free "
              f"model. DP-GD at ε = 4 holds the worst-case attack AUC at {dp4_auc:.3f} with "
              f"{pct(dp4_acc)} test accuracy (baseline {pct(b_acc)}), and the attacker's measured advantage stays "
              f"well below the theoretical DP bound. Restricting the API to labels only "
              f"lowers the attack AUC only to {lab_auc:.3f}, and k-anonymity (k = 25) only to {k25_auc:.3f}, "
              f"so output hiding and data generalisation are not substitutes for formal guarantees. LDP gives "
              f"the strongest trust model but needs far more users than this dataset has. Compressive privacy "
              f"with one DCA component keeps {pct(dca1_util)} utility-task accuracy while cutting the "
              f"attacker's balanced accuracy at inferring age from {raw_attr:.2f} (raw data) to {dca1_attr:.2f}. "
              f"The trends replicate on the Breast Cancer dataset."),
            P("**Keywords** - membership inference, differential privacy, local differential privacy, "
              "k-anonymity, compressive privacy, privacy-utility trade-off, healthcare ML.", st["body"])]


def stage1():
    s = [H1("1. Problem Refinement and Objectives")]
    s.append(H2("1.1 Summary of IA-1"))
    s.append(B("IA-1 (the Module 1 research-foundation document and the IA-1 report) defined a black-box membership "
               "inference attack against a diabetes-risk classifier on the Pima data. It mapped the problem to all "
               "five modules, prepared the data with public-bounds preprocessing, and ran a baseline shadow-model "
               f"attack on a single split (seed 0) for models A-D. The unregularised MLP leaked clearly there "
               f"(attack AUC {ia1_auc:.3f}). "
               "The DP accountant, LDP mechanisms, Mondrian partitioner and PCA/DCA projections were prototyped and "
               "checked in isolation but not yet attacked. On that split the regularised model D also collapsed "
               "towards the majority class, which showed that one split is not enough to judge a defence."))
    s.append(H2("1.2 Revised problem statement"))
    s.append(B("*How much training-set membership does a black-box attacker learn from a healthcare "
               "classifier, and which privacy-preserving techniques from the five course modules reduce that "
               "leakage, at what cost in predictive utility and under which trust assumptions?*"))
    s.append(B("The refinement keeps the IA-1 attack unchanged but uses it as the **measurement instrument** "
               "for every module. This gives the five modules one shared, quantitative privacy metric "
               "instead of five unrelated demonstrations."))
    s.append(H2("1.3 Research questions"))
    s.append(table([["RQ", "Question", "Module"],
                    ["RQ1", "How strongly does membership leakage depend on model complexity and overfitting?", "M1"],
                    ["RQ2", "How does the central DP budget ε trade utility against measured leakage, and how "
                            "does measured leakage compare with the theoretical DP bound?", "M2"],
                    ["RQ3", "Can patients protect themselves with LDP before data collection, and what data "
                            "volume does that require for useful accuracy?", "M3"],
                    ["RQ4", "Do classical PPDM techniques (k-anonymity, l-diversity, input noise, output "
                            "restriction) mitigate MIA, and what attacks do they leave open?", "M4"],
                    ["RQ5", "Can compressive projections keep the utility task accurate while suppressing a "
                            "privacy task and reconstruction?", "M5"],
                    ["RQ6", "Across all modules, which defences give the best privacy-utility balance for this "
                            "use case?", "All"]],
                   [1.2 * cm, W - 2.8 * cm, 1.6 * cm], st))
    s.append(H2("1.4 Final objectives and technical requirements"))
    s += bullets([
        "**O1** Implement a reproducible MIA pipeline (target + 8 shadow models, two attackers, three API modes).",
        "**O2** Implement at least one technique per module *from first principles* (DP-GD + accountant, LDP "
        "mechanisms + OUE, Mondrian + l-diversity, DCA/DUCA) so that each mechanism is auditable.",
        f"**O3** Evaluate utility (accuracy, F1, ROC-AUC, generalisation gap) and privacy (attack AUC, accuracy, "
        f"precision, recall, F1, advantage) for every configuration over {NS} seeds.",
        "**O4** Test significance against the baseline with paired tests and report mean ± SD.",
        "**O5** Replicate on a second public medical dataset to check external validity.",
        "**O6** Document provenance, preprocessing, ethics and reproducibility to the IA-2 checklist.",
        "**Requirements:** data-independent preprocessing (needed for valid DP/LDP guarantees); "
        "Kerckhoffs-style attacker (shadow models use the defended training recipe); fixed seeds; runtime "
        "under one hour on a laptop."], st)
    s.append(H2("1.5 Changes since IA-1 and their justification"))
    s.append(table([["Aspect", "IA-1 plan", "IA-2 change", "Reason"],
        ["Scope", "MIA + defences discussed (Module 1 focus)", "One defence family per module implemented and "
         "evaluated against the same attack", "IA brief requires integration of all five modules"],
        ["Research question", "Single RQ on attack effectiveness and regularisation", "RQ1-RQ6 (Sec. 1.3)",
         "Each module needs a testable question"],
        ["Attacker", "Shadow models + logistic-regression attack", "Added loss-threshold attack, rounded and "
         "label-only APIs", "IA-1 limitation: one simple attack may underestimate risk"],
        ["Statistics", "Single split", f"{NS} seeds, mean ± SD, paired Wilcoxon, 95% CI",
         "IA-1 Sec. 10 called for repeated runs"],
        ["Data", "Pima only", "Pima + Breast Cancer Wisconsin replication", "External-validity limitation"],
        ["Preprocessing", "Public-bounds scaling, zeros as missing, missing indicators", "Kept; added public "
         "clinical bins for LDP Naive Bayes", "Bins must be data-independent for the LDP guarantee to hold"],
        ["Defences M2-M5", "Mechanisms prototyped and checked in isolation (accountant, LDP variance, Mondrian, "
         "PCA/DCA)", "Each defence built into a training recipe and attacked; DP-GD in NumPy with per-example "
         "clipping; DUCA, l-diversity, input noise and output restriction added", "IA-1 left the defences "
         "unattacked; own DP-GD keeps the accounting transparent (the foundation document had planned "
         "TensorFlow Privacy)"],
        ["Model matrix", "Models A-D", "A-E (added deep random forest)", "RF is a known high-leakage model"],
        ["Knowledge distillation", "Candidate defence", "Not implemented (discussed in Sec. 4.10)",
         "No formal guarantee; time spent on formal (DP/LDP) defences instead"]],
        [2.3 * cm, 3.6 * cm, 4.6 * cm, W - 10.5 * cm], st, font_size=7))
    s.append(H2("1.6 Scope and extended threat model"))
    s.append(table([["Component", "Assumption in IA-2"],
        ["Target", "Binary classifier trained on 25% of the records (members); another 25% are non-members"],
        ["Attacker knowledge", "50% of the records from the same distribution (shadow pool), the training "
         "recipe and any published transform (Kerckhoffs' principle)"],
        ["Access", "Black-box queries: full probabilities, probabilities rounded to 0.1, or label only"],
        ["Goal (MIA)", "Decide whether a known record was in the target training set"],
        ["Goal (M4)", "Link a released k-anonymous record to a person through quasi-identifiers and learn the "
         "diagnosis (homogeneity attack)"],
        ["Goal (M5)", "From compressed vectors z, infer a sensitive attribute (age >= 35) or reconstruct x"],
        ["Trust (M3)", "Data collector untrusted: each patient perturbs their own record"],
        ["Safety", "Public data, locally trained models; no real system queried, no re-identification"]],
        [3.4 * cm, W - 3.4 * cm], st))
    return s


def stage2():
    s = [PageBreak(), H1("2. Integration of Course Concepts")]
    s.append(B("Figure 1 shows the end-to-end pipeline. Every defence is expressed as a *training recipe* "
               "`recipe(X, y, rng) -> model`. The same function trains the target and all eight shadow models, "
               "so the attacker always faces the defended system and no defence gets credit from a mismatched "
               "attacker."))
    s.append(fig("fig01_architecture.png", "Figure 1. IA-2 architecture: shared data pipeline, one defence "
                 "family per module, a common attack harness and paired evaluation.", W * 0.98))
    s.append(H2("2.1 Final module-mapping table"))
    rows = [["Syllabus topic", "How it is used in this work", "Treatment", "Evidence"],
        ["**1.1** Privacy complications in the AI era", "Healthcare use case: membership itself reveals diagnosis/"
         "treatment participation", "Analytical", "Sec. 1, IA-1"],
        ["**1.2** Threats & attacks on ML", "Shadow-model MIA (Shokri) + loss-threshold MIA (Yeom); label-only "
         "and rounded API variants", "Implemented", "`attacks.py`, Tab. 3, Fig. 2"],
        ["**1.3** Privacy-preserving ML", "Model-complexity / regularisation matrix A-E as the first line of defence",
         "Experimental", "`models.py`, Tab. 3"],
        ["**1.4** PPDM overview", "Bridges to Module 4; positions the defence families by where they act "
         "(input, training, output)", "Analytical", "Sec. 2.2"],
        ["**2.1** DP mechanism & properties", "Gaussian mechanism on clipped gradients; sequential composition "
         "(zCDP), post-processing, DP-to-advantage bound", "Implemented + demo", "`m2_dp.py`, Fig. 3-4"],
        ["**2.2** Applying DP in ML", "DP gradient descent with per-example clipping", "Implemented", "Tab. 4"],
        ["**2.3** DP supervised learning", "DP logistic regression and DP MLP-128, ε ∈ {0.5,...,8}",
         "Experimental", "Tab. 4, Fig. 3"],
        ["**2.4** DP unsupervised learning", "DP k-means (DPLloyd) on Pima features", "Controlled demo", "Fig. 4"],
        ["**3.1** LDP & mechanisms", "Laplace, Duchi, randomised response; variance checked against theory",
         "Implemented + correctness check", "`m3_ldp.py`, Fig. 5"],
        ["**3.2** Advanced LDP", "Hybrid mechanism; optimised unary encoding (OUE) frequency oracle",
         "Implemented", "Fig. 5"],
        ["**3.3** Piecewise mechanism", "Piecewise mechanism for numeric features before training",
         "Experimental", "Tab. 5"],
        ["**3.4** LDP Naive Bayes", "Feature-sampling LDP-NB over public clinical bins", "Experimental",
         "Tab. 5, Fig. 5"],
        ["**4.1** Privacy in data processing & mining", "Defences grouped by pipeline stage", "Analytical",
         "Sec. 2.2"],
        ["**4.2** Modifying the input", "Additive random perturbation (σ = 0.25, 0.5)", "Experimental", "Tab. 6"],
        ["**4.3** Publishing data", "Mondrian multidimensional k-anonymity on Age, Pregnancies, BMI",
         "Implemented", "`m4_ppdm.py`, Tab. 6"],
        ["**4.4** Beyond k-anonymity", "Distinct l-diversity constraint; homogeneity exposure; t-closeness",
         "Implemented", "Tab. 7, Fig. 6"],
        ["**4.5** Modifying the mining output", "API returns rounded scores or labels only", "Experimental",
         "Tab. 6, Fig. 6"],
        ["**5.1** Compressive privacy", "Clients release k-dim projection z = W<super>T</super>x, not x", "Implemented",
         "`m5_compressive.py`"],
        ["**5.2** CP mechanisms", "PCA (unsupervised), DCA (utility-supervised), DUCA (utility vs privacy label)",
         "Implemented", "Fig. 7"],
        ["**5.3** CP for ML applications", "Diabetes classifier trained on z; MIA, attribute inference and "
         "reconstruction attacks on z", "Experimental", "Tab. 8"],
        ["**5.4** Accuracy of the utility task", "Utility accuracy vs k for each projection", "Experimental",
         "Fig. 7"]]
    s.append(table(rows, [3.6 * cm, 6.4 * cm, 2.6 * cm, W - 12.6 * cm], st, font_size=7))
    s.append(H2("2.2 How the modules fit together, and where they do not apply directly"))
    s.append(B("The modules protect different parts of the pipeline and assume different trust models. "
               "**Input-side** techniques (LDP, input noise, k-anonymity, compressive projection) change what the "
               "data holder receives or releases. **Training-side** techniques (regularisation, DP-GD) limit how "
               "much the model memorises. **Output-side** techniques (rounding, label-only) limit what the API "
               "reveals. Measuring all of them with one attack makes this difference visible: only the formal "
               "methods (DP, LDP) bound leakage for *every* attacker, while the others reduce leakage only against "
               "the attacks that were tried."))
    s.append(B("**Where a module is not directly applicable, it is handled as follows.**"))
    s += bullets([
        "**Module 1.4 / 4.1** are overview topics. They have no algorithm of their own and are covered analytically "
        "through the input/training/output grouping above.",
        "**Module 2.4 (unsupervised DP).** Membership inference here targets a supervised classifier, so DP "
        "clustering cannot be attacked in the same harness. It is shown as a controlled demonstration "
        "(DP k-means cost vs ε, Fig. 4) instead.",
        "**Module 3 on the second dataset.** WDBC has 30 continuous features and no public clinical bounds. "
        "Splitting a per-patient budget over 31 attributes leaves under 0.3 per attribute even at ε = 8, so LDP "
        "training is not meaningful at n ≈ 142. LDP is therefore evaluated on Pima only, plus a bootstrap "
        "simulation that asks how many users would be needed.",
        "**Module 4 (k-anonymity) on the second dataset.** WDBC contains only cell-nucleus measurements and no "
        "demographic quasi-identifiers, so there is nothing to generalise. k-anonymity is evaluated on Pima "
        "(QIs: Age, Pregnancies, BMI). Input noise and output restriction still apply to both datasets.",
        "**Module 5 vs the MIA threat.** Compressive privacy protects the *released data* rather than the trained "
        "model, so its primary metrics are attribute inference and reconstruction. MIA against a model trained on "
        "z is reported as well, but a low-dimensional input does not by itself stop an over-parameterised model "
        "from memorising. DUCA needs a privacy label and is therefore Pima-only (Age >= 35)."], st)
    return s


def stage3():
    pp = prov["pima"]
    miss = pp["missing_after_zero_to_nan"]
    s = [PageBreak(), H1("3. Dataset Finalisation and Individual Implementation")]
    s.append(H2("3.1 Dataset provenance record"))
    s.append(table([["Item", "Pima Indians Diabetes (primary)", "Breast Cancer Wisconsin Diagnostic (replication)"],
        ["Source / URL", "Public mirror: raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes"
         ".data.csv (mirror of the UCI ML Repository version)",
         "Copy bundled with scikit-learn 1.6.1 (`sklearn.datasets.load_breast_cancer`); original: UCI ML "
         "Repository, DOI 10.24432/C5DW2B"],
        ["Original author / publisher", "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK); "
         "Smith et al. (1988); donated to UCI by V. Sigillito (JHU APL), 1990",
         "W. H. Wolberg, W. N. Street, O. L. Mangasarian, University of Wisconsin (1995)"],
        ["Collection method (original)", "Clinical measurements (2-h OGTT glucose, BP, skinfold, insulin, BMI, "
         "pedigree function, age) for female patients >= 21 years of Pima heritage; outcome = diabetes within "
         "5 years", "30 features computed from digitised images of fine-needle aspirates of breast masses; "
         "diagnosis malignant/benign"],
        ["Collection date (this work)", f"Downloaded {prov['generated_utc'][:10]} (UTC) via HTTPS",
         f"Loaded locally {prov['generated_utc'][:10]} from the installed package"],
        ["Licence / usage terms", "The mirror has no licence file (GitHub API reports none). The UCI page is no longer "
         "online; a widely used redistribution (Kaggle 'uciml') lists CC0. **Exact licence of the mirror: unknown.** "
         "Used for non-commercial academic benchmarking only.", "CC BY 4.0 (UCI)."],
        ["Version / hash", f"Mirror commit 357ed4a (10 Mar 2018); SHA-256 {pp['sha256'][:32]}...",
         f"SHA-256 {prov['breast_cancer']['sha256'][:32]}..."],
        ["Original size", f"{pp['rows_original']} rows x 9 columns; class 0/1 = "
         f"{pp['class_counts']['0']}/{pp['class_counts']['1']}",
         f"{prov['breast_cancer']['rows']} rows x 30 features; malignant/benign = "
         f"{prov['breast_cancer']['class_counts']['malignant']}/{prov['breast_cancer']['class_counts']['benign']}"],
        ["Final size", f"{pp['final_shape_scaled'][0]} rows x {pp['final_shape_scaled'][1]} features "
         "(8 scaled + 2 missing indicators) + label; no rows dropped", "569 x 30 scaled + label; no rows dropped"],
        ["Access restrictions & safeguards", "Public, de-identified. No linkage with other data, no "
         "re-identification attempted; MIA run only on local models", "Public, de-identified, no demographics"],
        ["Known limitations", "Small (768); one population and sex only; zeros encode missing values; ethical "
         "concerns about secondary use of data from an Indigenous community (Sec. 6)",
         "Small (569); derived image features; not representative of clinical populations"]],
        [3.0 * cm, (W - 3.0 * cm) / 2, (W - 3.0 * cm) / 2], st, font_size=7))
    s.append(H2("3.2 Preprocessing log (Pima)"))
    s.append(table([["Step", "Operation", "Effect"],
        ["1", "Verify SHA-256 of the raw file on every load", "Pipeline aborts if the file changes"],
        ["2", "Zero -> missing for Glucose, BloodPressure, SkinThickness, Insulin, BMI",
         f"Missing: Glucose {miss['Glucose']}, BP {miss['BloodPressure']}, Skin {miss['SkinThickness']}, "
         f"Insulin {miss['Insulin']}, BMI {miss['BMI']}"],
        ["3", "Clip each feature to fixed public bounds (e.g. Glucose 40-200, Age 21-81) and scale to [-1, 1]",
         "Data-independent: costs no privacy budget, so DP/LDP guarantees remain valid"],
        ["4", "Missing values -> centre of the public range (0 after scaling)", "No data-derived imputation"],
        ["5", "Add missing indicators for SkinThickness and Insulin", "Keeps the information that a test "
         "was not performed; 10 model features in total"],
        ["6", "Discretise with public clinical cut points (e.g. BMI 25/30/35; Age 30/40/50) for LDP-NB",
         "4-5 bins per feature; written to data/processed/pima_discretised.csv"],
        ["7", f"Stratified split per seed (seeds 0-{NS - 1}): 192 members / 192 non-members / 384 shadow pool",
         "Non-members double as the utility test set"]],
        [1.0 * cm, 8.6 * cm, W - 9.6 * cm], st, font_size=7.2))
    s.append(B("For WDBC, features are scaled to [-1, 1] using the minimum and maximum values published in the "
               "dataset documentation (treated as public metadata), and the label is recoded so that "
               "malignant = 1. Processed files and their hashes are listed in "
               "`results/tables/provenance.json`."))
    s.append(H2("3.3 Implementation"))
    s.append(table([["File", "Content (written for this IA)"],
        ["`src/common.py`", "Loading with hash check, public-bounds scaling, stratified splits, utility metrics, "
         "NumPy MLP with per-example gradients"],
        ["`src/attacks.py`", "Shadow-model attack, loss-threshold attack, API modes (full / rounded / label), "
         "attack metrics including advantage"],
        ["`src/models.py`", "Model matrix A-E (complexity x regularisation)"],
        ["`src/m2_dp.py`", "DP-GD, zCDP accountant (σ <-> ε), DP-advantage bound, DPLloyd k-means"],
        ["`src/m3_ldp.py`", "Laplace / Duchi / Piecewise / Hybrid mechanisms, randomised response, OUE, LDP Naive Bayes"],
        ["`src/m4_ppdm.py`", "Mondrian k-anonymity with l-diversity, NCP / homogeneity / t-closeness, input noise"],
        ["`src/m5_compressive.py`", "PCA, DCA, DUCA projections; projected-model wrapper"],
        ["`src/run_experiments.py`", f"All configurations x {NS} seeds in parallel; standalone experiments; provenance"],
        ["`src/make_figures.py`", "Summary tables, paired statistics, all figures"],
        ["`src/build_report.py`, `src/build_manuscript.py`", "This report and the IEEE-style manuscript"]],
        [4.6 * cm, W - 4.6 * cm], st, font_size=7.4))
    s.append(H3("Key algorithms"))
    s.append(B(f"**DP-GD and accounting.** For T = 200 full-batch steps, each per-example gradient is clipped to L2 norm "
               f"C = 1, the clipped gradients are summed, and N(0, σ²C²I) noise is added. One step is "
               f"1/(2σ²)-zCDP under add/remove adjacency, so T steps give ρ = T/(2σ²) and ε = ρ + 2√(ρ ln(1/δ)) "
               f"with δ = 10<super>-5</super>. Inverting this gives the noise used: σ = {sig.loc[1, 'sigma']:.1f} for ε = 1, "
               f"{sig.loc[4, 'sigma']:.1f} for ε = 4 and {sig.loc[8, 'sigma']:.1f} for ε = 8. Without "
               "subsampling there is no privacy amplification, so the stated ε is conservative."))
    s.append(B("**Piecewise mechanism.** For x ∈ [-1, 1] and budget ε, with C = (e^(ε/2)+1)/(e^(ε/2)-1), the report "
               "is drawn uniformly from [l(x), r(x)] with probability e^(ε/2)/(e^(ε/2)+1), and otherwise uniformly "
               "from the rest of [-C, C]. Here l(x) = (C+1)x/2 - (C-1)/2 and r(x) = l(x) + C - 1. The output is "
               "unbiased. Each record's 10 features and its label share ε equally (sequential composition)."))
    s.append(B("**LDP Naive Bayes.** Each patient samples one feature j and reports the joint value "
               "(class, bin_j) through OUE (p = 1/2, q = 1/(e^ε+1)) with the full ε. The aggregator debiases the "
               "counts, clips negatives and forms the class prior and class-conditional tables."))
    s.append(B("**Mondrian with l-diversity.** The partitioner recursively median-splits the QI with the widest "
               "range and accepts a split only if both halves have >= k records and >= l distinct outcomes. QI "
               "values are then replaced by their equivalence-class mean."))
    s.append(B("**DCA / DUCA.** DCA solves the generalised eigenproblem (S_B + ρ'I)w = λ(S_W + S_B + ρI)w "
               "for the utility label. DUCA uses (S_B^u + ρ'I)w = λ(γS_B^p + S_W^u + ρI)w, which penalises "
               "directions that separate the privacy attribute (γ = 5, ρ = 10<super>-2</super>, ρ' = 10<super>-3</super>)."))
    s.append(H2("3.4 Execution instructions"))
    s.append(P("`git clone https://github.com/anjaliash005/MLCS-IA.git && cd MLCS-IA`", st["code"]))
    s.append(P("`python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt`", st["code"]))
    s.append(P("`.venv/bin/python src/run_experiments.py        # add --quick for a 2-seed smoke test`", st["code"]))
    s.append(P("`.venv/bin/python src/make_figures.py && .venv/bin/python src/build_report.py`", st["code"]))
    n_jobs = sum(len(R.runs[d].drop_duplicates(["config", "seed"])) for d in R.runs)
    s.append(B(f"The full run performs {n_jobs} target/attack evaluations "
               f"(9 models each) and took roughly {R.runtime_s / 3600 / 8:.1f} h wall-clock on 8 worker processes "
               "(10-core laptop CPU)."))
    s.append(H2("3.5 Individual-contribution statement"))
    s.append(B(f"This IA-2 is my individual work ({AUTHOR}, {ROLL}). Building on my IA-1 foundation document, "
               "I contributed:"))
    s += bullets([
        "**Problem refinement:** reframed the IA-1 attack as a common yardstick for all five modules and wrote RQ1-RQ6.",
        "**Dataset work:** sourced and hashed both datasets; designed the data-independent preprocessing (public "
        "clinical bounds, zero-as-missing handling, missing indicators, clinical bins) needed for valid DP/LDP claims.",
        "**Implementation:** wrote the attack harness, DP-GD with its zCDP accountant, the four LDP mechanisms and "
        "LDP-NB, Mondrian with l-diversity, and DCA/DUCA from first principles in NumPy (scikit-learn is used only "
        "for standard non-private models and metrics).",
        f"**Experimental design:** Kerckhoffs-style shadow models, three API modes, {NS}-seed paired design, "
        "advantage noise floor, DP-bound comparison, bootstrap user-scaling simulation, second-dataset replication.",
        "**Analysis and documentation:** statistical testing, figures, this report, and the manuscript draft."], st)
    return s


def m1_table():
    rows = [["Model", "Train acc", "Test acc", "Gap", "Shadow AUC", "Loss AUC", "Prec", "Rec", "F1", "Adv"]]
    for cfg in [c for c in R.summ[P_].config.unique() if c[:2] in ("A:", "B:", "C:", "D:", "E:")]:
        r = R.row(P_, cfg)
        rows.append([cfg.replace(" (baseline)", " **(baseline)**"), f3(r.test_acc + r.gap), f3(r.test_acc, r.test_acc_sd),
                     f3(r.gap), f3(r.sh_auc, r.sh_auc_sd), f3(r.ls_auc), f3(r.sh_prec), f3(r.sh_rec),
                     f3(r.sh_f1), f3(r.sh_adv)])
    return table(rows, [4.3 * cm, 1.2 * cm, 2.0 * cm, 1.1 * cm, 2.0 * cm, 1.3 * cm, 1.1 * cm, 1.1 * cm,
                        1.1 * cm, W - 15.2 * cm], st, font_size=6.8)


def stage4():
    s = [PageBreak(), H1("4. Evaluation, Comparison and Critical Analysis")]
    s.append(H2("4.1 Metrics"))
    s += bullets([
        "**Utility:** test accuracy, F1 and ROC-AUC on the held-out non-members; generalisation gap = train acc - test acc.",
        "**Attack ROC-AUC:** threshold-free probability that a random member scores above a random non-member "
        "(0.5 = random guessing). *Worst-case AUC* is the larger of the shadow and loss attacks.",
        "**Attack accuracy / precision / recall / F1** at the threshold calibrated on the shadow models "
        "(balanced 192 vs 192 evaluation).",
        "**Advantage** = TPR - FPR (Yeom et al.). *Max advantage* is the best threshold in hindsight; it is compared "
        "with the DP bound (e^ε - 1 + 2δ)/(e^ε + 1).",
        f"**Statistics:** mean ± SD over {NS} seeds; paired differences vs the baseline with 95% t-intervals and "
        "two-sided Wilcoxon signed-rank p-values."], st)

    s.append(H2("4.2 Module 1 - model complexity and overfitting (RQ1)"))
    s.append(KeepTogether([P("**Table 3.** Model matrix on Pima (mean ± SD where shown; attack metrics are for the shadow attack).",
                             st["cap"]), m1_table()]))
    s.append(Spacer(1, 6))
    s.append(B(f"Leakage follows the generalisation gap closely (Fig. 2). The unregularised MLP-128x2 (baseline C) memorises "
               f"its 192 training records (gap {b_gap:.3f}), and the shadow attack separates members from non-members "
               f"with AUC {b_auc:.3f}. The deep random forest leaks even more (AUC {e_auc:.3f}) despite better test "
               f"accuracy, so **high accuracy does not imply low leakage**. Adding L2 and early stopping to the same "
               f"architecture (model D) brings the AUC down to {d_auc:.3f} with test accuracy {pct(d_acc)} versus "
               f"{pct(b_acc)} (difference not significant, p = {pfmt(R.stat(P_, D).p_test_acc)}). On this small table "
               f"the simplest model, logistic regression (A), is both the most accurate ({pct(a_acc)}) and essentially "
               f"leak-free (AUC {a_auc:.3f}): capacity beyond what the data supports buys memorisation, not accuracy. "
               "Regularisation and model choice are therefore an effective first line of defence, but they come "
               "with no guarantee."))
    s.append(fig("fig02_m1_overfitting_roc.png", "Figure 2. Left: attack AUC vs generalisation gap for models A-E on "
                 "both datasets (error bars = SD over seeds). Right: shadow-attack ROC curves on one Pima split."))

    s.append(H2("4.3 Module 2 - central differential privacy (RQ2)"))
    rows = [["Configuration", "σ", "Test acc", "Gap", "Shadow AUC", "Loss AUC", "Max adv.", "DP bound"]]
    for fam, h in (("DP-GD LogReg", "LR"), ("DP-GD MLP-128", "MLP")):
        for e in [0.5, 1, 2, 4, 8, "inf"]:
            cfg = f"{fam} eps={e}"
            r = R.row(P_, cfg)
            ee = np.inf if e == "inf" else e
            rows.append([cfg.replace("eps=", "ε = ").replace("inf", "∞ (non-private, T = 200)"),
                         f"{sig.loc[ee, 'sigma']:.1f}" if e != "inf" else "0",
                         f3(r.test_acc, r.test_acc_sd), f3(r.gap), f3(r.sh_auc, r.sh_auc_sd), f3(r.ls_auc),
                         f3(max(r.sh_advmax, r.ls_advmax)),
                         f3(mia_advantage_bound(ee)) if e != "inf" else "–"])
    r = R.row(P_, NPL)
    rows.append(["NumPy MLP-128, long training (T = 2000, non-private)", "0", f3(r.test_acc, r.test_acc_sd),
                 f3(r.gap), f3(r.sh_auc, r.sh_auc_sd), f3(r.ls_auc), f3(max(r.sh_advmax, r.ls_advmax)), "–"])
    s.append(KeepTogether([P("**Table 4.** DP gradient descent on Pima (δ = 10<super>-5</super>, C = 1, T = 200).", st["cap"]),
                           table(rows, [5.4 * cm, 1.1 * cm, 2.3 * cm, 1.2 * cm, 2.3 * cm, 1.4 * cm, 1.5 * cm,
                                        W - 15.2 * cm], st, font_size=6.8)]))
    s.append(Spacer(1, 6))
    s.append(B(f"Utility rises with ε as expected (Fig. 3). At ε = 4 the DP-MLP reaches {pct(dp4_acc)} accuracy (DP "
               f"logistic regression {pct(dplr4_acc)}), compared with {pct(b_acc)} for the non-private baseline, while "
               f"the worst-case attack AUC stays at {dp4_auc:.3f}. Below ε = 1 the noise dominates: the DP-MLP falls to "
               f"{pct(dp1_acc)} at ε = 1. The **ablation** separates the two causes of the improvement. The same "
               f"NumPy MLP trained without noise for 2000 steps leaks (AUC {npl_auc:.3f}), but trained for only 200 "
               f"steps it already leaks little (AUC {dpinf_auc:.3f}). Most of the *empirical* reduction against these "
               f"attacks therefore comes from the short training schedule that DP forces. What DP adds is a "
               f"**provable bound that holds for every attacker**, including stronger ones than those tested here."))
    s.append(B("Across all ε, the measured maximum advantage stays inside the random-score noise floor (grey band, "
               "Fig. 3 right) and far below the theoretical bound. This is consistent with DP, and it also shows that "
               "the bound is loose for average-case attackers at moderate ε. Fig. 4 shows the two DP properties used: "
               "sequential composition (ε grows with the number of noisy steps for fixed σ) and an unsupervised "
               "application, DP k-means. Its clustering cost approaches the non-private cost as ε grows "
               f"(SSE ratio {R.kmeans.set_index('eps').loc[8, 'sse_ratio_mean']:.2f} at ε = 8, "
               f"{R.kmeans.set_index('eps').loc[0.5, 'sse_ratio_mean']:.2f} at ε = 0.5)."))
    s.append(fig("fig03_m2_dp_tradeoff.png", "Figure 3. DP-GD on Pima: utility (left) and attack AUC (centre) vs ε; "
                 "right: empirical maximum advantage vs the (ε, δ)-DP bound, with the 95th-percentile advantage of a "
                 "random attacker shown as a grey band."))
    s.append(fig("fig04_m2_composition_kmeans.png", "Figure 4. Left: zCDP composition - ε after T Gaussian steps for "
                 "several noise multipliers. Right: DP k-means (DPLloyd, 5 iterations) cost relative to non-private "
                 "k-means (30 trials).", W * 0.85))

    s.append(H2("4.4 Module 3 - local differential privacy (RQ3)"))
    rows = [["Configuration", "Test acc", "Test F1", "Shadow AUC", "Loss AUC", "Max adv."]]
    for cfg in [c for c in R.summ[P_].config.unique() if c.startswith(("LDP-", "LDP "))]:
        r = R.row(P_, cfg)
        rows.append([cfg.replace("eps=", "ε = ").replace("inf", "∞ (non-private)"), f3(r.test_acc, r.test_acc_sd),
                     f3(r.test_f1), f3(r.sh_auc, r.sh_auc_sd), f3(r.ls_auc), f3(max(r.sh_advmax, r.ls_advmax))])
    s.append(KeepTogether([P("**Table 5.** Models trained on locally privatised Pima records (n = 192 per model; "
                             "ε is the per-patient budget).", st["cap"]),
                           table(rows, [5.6 * cm, 2.4 * cm, 1.6 * cm, 2.4 * cm, 1.8 * cm, W - 13.8 * cm], st,
                                 font_size=7)]))
    s.append(Spacer(1, 6))
    s.append(B("The correctness check (Fig. 5, left) confirms that all four mechanisms are unbiased and that their "
               "empirical per-report variance matches theory. Duchi has lower variance at small ε (≤ 0.5) and Piecewise "
               "wins from ε = 1 upward, which reproduces the crossover reported by Wang et al. (2019). Hybrid matches "
               "Duchi at small ε and lies between the two at larger ε here, because it minimises *worst-case* variance "
               "while this test input is skewed towards the ends of the range. "
               f"LDP gives the strongest trust model, since the hospital never sees raw values, and every "
               f"LDP-trained model shows attack AUC close to 0.5. The cost is utility at this data size. Piecewise "
               f"perturbation of all 11 attributes reaches only {pct(pm16_acc)} at ε = 16, because each attribute "
               f"gets ε/11. LDP Naive Bayes, where each user reports one sampled feature with the full ε, reaches "
               f"{pct(nb4_acc)} at ε = 4 versus {pct(nbinf_acc)} non-privately. The bootstrap simulation (Fig. 5, "
               f"centre) shows what scale is needed: with 50,000 reporting users LDP-NB at ε = 1 reaches "
               f"{pct(R.nb(50000, 1))}, compared with {pct(R.nb(576, 1))} using the 576 real records. **LDP is the "
               f"right tool for population-scale collection, but not for a single small clinic.**"))
    s.append(fig("fig05_m3_ldp.png", "Figure 5. Left: per-report variance of the LDP mechanisms (dots = empirical, "
                 "dashed = theory). Centre: LDP Naive Bayes accuracy vs ε for the real 576 records and bootstrap "
                 "simulations of larger user bases. Right: accuracy and attack AUC of LDP-trained models."))

    s.append(H2("4.5 Module 4 - privacy-preserving data mining (RQ4)"))
    rows = [["Configuration", "Test acc", "Gap", "Shadow AUC", "Loss/label AUC", "Adv", "NCP", "Homog."]]
    for cfg, mode, lab in [(C, "full", "Model C, full probabilities"), (C, "rounded", "Model C, rounded API (4.5)"),
                           (C, "label", "Model C, label-only API (4.5)")] + \
            [(c, "full", c) for c in R.summ[P_].config.unique() if c.startswith(("Mondrian", "Input noise"))]:
        r = R.row(P_, cfg, mode)
        rows.append([lab.replace("sigma", "σ"), f3(r.test_acc, r.test_acc_sd), f3(r.gap), f3(r.sh_auc, r.sh_auc_sd),
                     f3(r.ls_auc), f3(r.sh_adv), f3(r.get("ncp", np.nan)), f3(r.get("homogeneous_frac", np.nan))])
    s.append(KeepTogether([P("**Table 6.** PPDM defences on Pima. NCP = normalised certainty penalty (information loss) "
                             "of the anonymised training set; Homog. = share of records in equivalence classes with "
                             "a single outcome.", st["cap"]),
                           table(rows, [4.6 * cm, 2.2 * cm, 1.2 * cm, 2.2 * cm, 2.0 * cm, 1.2 * cm, 1.2 * cm,
                                        W - 14.6 * cm], st, font_size=6.9)]))
    s.append(Spacer(1, 6))
    rows = [["k", "l", "Classes", "Avg size", "NCP", "Homogeneous", "t (max TV dist.)"]]
    for _, r in R.anon.iterrows():
        rows.append([int(r.k), int(r.l), int(r.n_classes), f"{r.avg_size:.1f}", f3(r.ncp), pct(r.homogeneous_frac),
                     f3(r.t_closeness)])
    s.append(KeepTogether([P("**Table 7.** Anonymity analysis of the full Pima table (QIs: Age, Pregnancies, BMI; "
                             "sensitive attribute: diabetes outcome).", st["cap"]),
                           table(rows, [1.2 * cm, 1.2 * cm, 2 * cm, 2 * cm, 2 * cm, 2.6 * cm, W - 11 * cm], st,
                                 font_size=7.2)]))
    s.append(Spacer(1, 6))
    s.append(B(f"**Output restriction (4.5)** helps less than expected. Rounding scores lowers the worst-case AUC to "
               f"{rnd_auc:.3f}, and returning labels only lowers it to {lab_auc:.3f} (baseline {max(b_auc, b_ls):.3f}). "
               "An overfit model is simply *correct* far more often on its members, and a label-only attacker "
               "exploits that, as Choquette-Choo et al. (2021) predict. **k-anonymity (4.3)** reduces the attack AUC "
               f"from {b_auc:.3f} to {k5_auc:.3f} (k = 5) and {k25_auc:.3f} (k = 25) as generalisation blurs the "
               "training records, but the model still overfits the remaining features. It also has a privacy flaw of "
               f"its own: at k = 5, {pct(homog_k5_l1)} of records (k = 10: {pct(homog_k10_l1)}) sit in classes whose "
               "members all share the same diagnosis, so anyone who knows a patient's age, pregnancies and BMI learns "
               "the outcome (a homogeneity attack). **Adding l = 2 diversity (4.4)** drives this exposure to 0% for "
               f"every k, at a small extra information loss (NCP {ncp_k10_l1:.3f} -> {ncp_k10_l2:.3f} at k = 10). "
               f"**Input randomisation (4.2)** with σ = 0.5 acts as a strong regulariser (AUC {noise5_auc:.3f}, accuracy "
               f"{pct(noise5_acc)}) but carries no formal guarantee."))
    s.append(fig("fig06_m4_ppdm.png", "Figure 6. Left: information loss and homogeneity exposure vs k, with and without "
                 "l-diversity. Centre/right: attack AUC and utility for the PPDM defences on Pima."))

    s.append(H2("4.6 Module 5 - compressive privacy (RQ5, syllabus 5.4)"))
    rows = [["Projection", "Utility acc (MLP)", "Shadow AUC", "Loss AUC", "Age-inference bal. acc", "Recon. NRMSE"]]
    for cfg in [c for c in R.summ[P_].config.unique() if c.startswith(("PCA", "DCA", "DUCA"))]:
        r = R.row(P_, cfg)
        rows.append([cfg, f3(r.test_acc, r.test_acc_sd), f3(r.sh_auc, r.sh_auc_sd), f3(r.ls_auc),
                     f3(r.attr_bacc), f3(r.recon_nrmse)])
    rows.append(["Raw features (reference, logistic regression)", f3(raw_util), "–", "–", f3(raw_attr), "0.000"])
    s.append(KeepTogether([P("**Table 8.** Compressive privacy on Pima. Attribute inference: random forest trained on "
                             "the attacker's pool to predict Age >= 35 from z. Reconstruction: linear regression from z "
                             "to x (NRMSE 1 = no better than the mean).", st["cap"]),
                           table(rows, [5.4 * cm, 2.6 * cm, 2.2 * cm, 1.7 * cm, 2.8 * cm, W - 14.7 * cm], st,
                                 font_size=7)]))
    s.append(Spacer(1, 6))
    s.append(B(f"Fig. 7 shows that the choice of projection matters more than its dimension. **PCA** keeps the "
               f"directions of largest variance, so it stays reconstructable (NRMSE {pca2_rec:.2f} at k = 2) and leaks "
               f"age (balanced accuracy {pca1_attr:.2f} even at k = 1), while its utility at small k is poor "
               f"({pct(pca1_util)} at k = 1). **DCA** puts the diabetes signal into its first component: one "
               f"number per patient gives {pct(dca1_util)} accuracy (raw data: {pct(raw_util)}) while age inference "
               f"falls to {dca1_attr:.2f} and reconstruction is almost impossible (NRMSE {dca2_rec:.2f} at k = 2). "
               f"**DUCA** additionally penalises the age direction and pushes age inference close to chance "
               f"({duca1_attr:.2f} at k = 1) for a small utility cost ({pct(duca1_util)}). On the MIA axis, an "
               f"over-parameterised MLP on z still memorises once z has enough dimensions (DCA/DUCA k >= 2, PCA k >= 4). "
               f"With k = 1 (DCA) the worst-case AUC is "
               f"{dca1_auc:.3f} at {pct(dca1_acc)} accuracy. So compressive privacy protects the released data, and "
               f"has to be combined with training-side measures to protect the model."))
    s.append(fig("fig07_m5_compressive.png", "Figure 7. Compressive privacy on Pima vs number of components k (mean "
                 "over seeds): utility-task accuracy (logistic regression on z), privacy-task balanced accuracy, and "
                 "reconstruction error."))
    s += cross_module()
    s += replication()
    s += discussion()
    return s


def cross_module():
    s = [H2("4.7 Cross-module comparison (RQ6)")]
    picks = [(C, "full"), (D, "full"), (C, "label"), ("DP-GD LogReg eps=4", "full"), ("DP-GD MLP-128 eps=4", "full"),
             ("DP-GD MLP-128 eps=1", "full"), ("LDP Naive Bayes eps=4", "full"), ("LDP-Piecewise LogReg eps=16", "full"),
             ("Mondrian k=25", "full"), ("Mondrian k=10, l=2", "full"), ("Input noise sigma=0.5", "full"),
             ("DCA k=1", "full"), ("DUCA k=2", "full")]
    rows = [["Mod.", "Configuration", "Test acc", "Worst AUC", "Δ shadow AUC [95% CI]", "p", "Δ acc [95% CI]", "p",
             "Formal guarantee"]]
    guar = {"M1": "none", "M2": "(ε, δ)-DP", "M3": "ε-LDP per record", "M4": "k-anon. / l-div. (syntactic)",
            "M5": "none (info-theoretic intuition)"}
    for cfg, mode in picks:
        r = R.row(P_, cfg, mode)
        lab = cfg if mode == "full" else f"{cfg.split(' (')[0]}, {mode} API"
        if cfg == C and mode == "full":
            rows.append(["M1", "Baseline C (reference)", f3(r.test_acc), f3(max(r.sh_auc, r.ls_auc)), "–", "–", "–", "–",
                         "none"])
            continue
        t = R.stat(P_, cfg, mode)
        mod = "M4" if mode != "full" else r.module
        g = "none" if (mode != "full" or cfg.startswith("Input noise")) else guar[mod]
        rows.append([mod, lab.replace("eps=", "ε = ").replace("sigma", "σ"), f3(r.test_acc),
                     f3(max(r.sh_auc, r.ls_auc)),
                     f"{t.d_sh_auc:+.3f} [±{t.d_sh_auc_ci:.3f}]", pfmt(t.p_sh_auc),
                     f"{t.d_test_acc:+.3f} [±{t.d_test_acc_ci:.3f}]", pfmt(t.p_test_acc), g])
    s.append(KeepTogether([P(f"**Table 9.** Selected configurations vs baseline C on Pima (paired over {NS} seeds; "
                             "Wilcoxon signed-rank p-values; Δ = configuration - baseline).", st["cap"]),
                           table(rows, [1.0 * cm, 4.0 * cm, 1.3 * cm, 1.4 * cm, 2.6 * cm, 1.1 * cm, 2.6 * cm, 1.1 * cm,
                                        W - 15.1 * cm], st, font_size=6.5)]))
    s.append(Spacer(1, 6))
    s.append(fig("fig08_frontier.png", "Figure 8. Privacy-utility landscape of every configuration on Pima. Lower-right "
                 "is better (high accuracy, attack AUC near 0.5).", W * 0.82))
    s.append(B(f"Fig. 8 brings all modules together. The lower-right region (high accuracy, attack AUC near 0.5) is "
               f"occupied by the **simple logistic regression (A)**, **DP-GD at ε = 4-8** and **DCA with k = 1**. The "
               f"regularised MLP (D) and input noise also reach chance-level leakage at slightly lower accuracy, and "
               f"the LDP models reach it at a larger accuracy cost. Of these, only DP-GD and LDP come with a formal "
               f"guarantee. Output restriction and k-anonymity leave the overfit model in the upper half. For this use "
               f"case the recommended configuration is **DP-GD logistic regression at ε ≈ 4**. It reaches "
               f"{pct(dplr4_acc)} accuracy, only {100 * (lrinf_acc - dplr4_acc):.1f} points below its non-private "
               f"counterpart and {100 * (dplr4_acc - b_acc):.1f} points above the leaky baseline, with attack AUC "
               f"≈ {dplr4_auc:.2f}, and its guarantee also covers attackers stronger than the ones tested. If records "
               "are released rather than models, DCA/DUCA projections are the preferred release format."))
    s.append(H3("Results recorded for the IA-1 table (Section 13 of IA-1)"))
    rows = [["Model", "Test accuracy", "MIA accuracy", "MIA precision", "MIA recall", "MIA F1", "MIA ROC-AUC"]]
    for lab, cfg in (("Baseline (C)", C), ("Regularised (D)", D), ("DP model (DP-GD MLP, ε = 4)", "DP-GD MLP-128 eps=4")):
        r = R.row(P_, cfg)
        rows.append([lab, f3(r.test_acc), f3(r.sh_acc), f3(r.sh_prec), f3(r.sh_rec), f3(r.sh_f1), f3(r.sh_auc)])
    s.append(table(rows, [4.4 * cm] + [(W - 4.4 * cm) / 6] * 6, st, font_size=7.4))
    return s


def replication():
    s = [H2("4.8 Replication on Breast Cancer Wisconsin")]
    bc = R.summ["BreastCancer"]
    rows = [["Configuration", "Test acc", "Gap", "Shadow AUC", "Loss AUC", "Recon. NRMSE"]]
    for _, r in bc.iterrows():
        lab = r.config if r["mode"] == "full" else f"{r.config.split(' (')[0]}, {r['mode']} API"
        rows.append([lab.replace("eps=", "ε = ").replace("sigma", "σ").replace("inf", "∞"),
                     f3(r.test_acc, r.test_acc_sd), f3(r.gap), f3(r.sh_auc, r.sh_auc_sd), f3(r.ls_auc),
                     f3(r.get("recon_nrmse", np.nan))])
    s.append(KeepTogether([P("**Table 10.** All configurations on Breast Cancer Wisconsin (Diagnostic).", st["cap"]),
                           table(rows, [6.0 * cm, 2.6 * cm, 1.4 * cm, 2.6 * cm, 1.8 * cm, W - 14.4 * cm], st,
                                 font_size=6.8)]))
    s.append(Spacer(1, 6))
    s.append(B(f"WDBC is an easier and better-separated task (baseline accuracy {pct(bc_b_acc)}), so even the "
               f"over-parameterised MLP generalises well and leaks much less (AUC {bc_b_auc:.3f}). The deep random "
               f"forest remains the leakiest model (AUC {bc_e_auc:.3f}). The qualitative ordering is the same as on "
               "Pima: leakage tracks the generalisation gap, DP-GD and regularisation pull the attack to chance, and "
               f"one DCA component keeps {pct(R.comp_mean('BreastCancer', 'DCA', 1, 'utility_acc'))} accuracy while "
               f"making reconstruction much harder (NRMSE {R.comp_mean('BreastCancer', 'DCA', 1, 'recon_nrmse'):.2f} vs "
               f"{R.comp_mean('BreastCancer', 'PCA', 2, 'recon_nrmse'):.2f} for PCA with k = 2). **Membership risk is therefore a property of "
               "the model-data pair, not of the dataset alone.** A model that leaks badly on one small table can be "
               "reasonably safe on another."))
    return s


def discussion():
    s = [H2("4.9 Answers to the research questions")]
    s.append(table([["RQ", "Finding"],
        ["RQ1", f"Leakage tracks the generalisation gap (baseline AUC {b_auc:.2f} at gap {b_gap:.2f}); regularisation "
                f"cuts it to {d_auc:.2f} with no significant accuracy change, and logistic regression is both the most "
                f"accurate and least leaky model."],
        ["RQ2", f"DP-GD at ε = 4 keeps {pct(dp4_acc)} accuracy with attack AUC {dp4_auc:.2f}; the measured advantage stays "
                f"far below the DP bound. The ablation shows that the short schedule DP requires causes much of the "
                f"empirical gain; DP adds the guarantee."],
        ["RQ3", f"LDP removes trust in the collector and suppresses MIA, but at n = 192 utility is limited "
                f"(LDP-NB {pct(nb4_acc)} at ε = 4); useful LDP-NB needs on the order of 10<super>4</super> or more users."],
        ["RQ4", f"Output restriction and k-anonymity only partially reduce MIA (label-only {lab_auc:.2f}, k = 25 "
                f"{k25_auc:.2f}); k-anonymity leaves homogeneity exposure that l-diversity removes."],
        ["RQ5", f"One DCA component keeps {pct(dca1_util)} utility while age inference drops from {raw_attr:.2f} to "
                f"{dca1_attr:.2f} and reconstruction becomes infeasible; DUCA goes further for privacy."],
        ["RQ6", "Formal training-side protection (DP-GD) is the best overall defence for model release; compressive "
                "projections are the best defence for data release; syntactic and output-side measures are useful "
                "only as complements."]], [1.2 * cm, W - 1.2 * cm], st, font_size=7.6))
    s.append(H2("4.10 Limitations and threats to validity"))
    s += bullets([
        "**Internal validity.** The defended recipes were tuned lightly (e.g. DP learning rate, number of steps) and "
        "not by private hyperparameter search. Tuning on the data would itself consume privacy budget, so the "
        "reported ε excludes it.",
        "**Construct validity.** Attack AUC measures *average* leakage. Recent work (Carlini et al., 2022) argues for "
        "TPR at very low FPR, but with 192 members per run such low-FPR estimates are too noisy to report reliably. "
        "The attackers used are standard but not the strongest known (e.g. LiRA with many shadow models).",
        "**Conclusion validity.** With 10 seeds the smallest attainable two-sided Wilcoxon p-value is about 0.002. "
        "Small differences (|Δ AUC| < 0.02) should be read as 'no detectable difference', not as equivalence.",
        "**External validity.** Two small tabular datasets; Pima covers one population and one sex. Results may not "
        "transfer to large clinical datasets, images or language models.",
        "**DP accounting.** Full-batch DP-GD gives no subsampling amplification, so ε is conservative, and "
        "add/remove adjacency is assumed. Public-bounds preprocessing is assumed to be truly public.",
        "**LDP scaling.** The user-scaling curves use bootstrap resampling of 576 records. They show the statistical "
        "trend, not real population diversity.",
        "**Not implemented.** Knowledge distillation (an IA-1 candidate) and DP hyperparameter selection were left out "
        "of scope."], st)
    s.append(H2("4.11 Proposed improvements"))
    s += bullets([
        "Add Poisson-subsampled DP-SGD with an RDP/PRV accountant to get privacy amplification and tighter ε.",
        "Evaluate LiRA and label-only augmentation attacks, and report TPR at FPR = 1% on a larger dataset "
        "(e.g. MIMIC-derived tabular data under a credentialed licence).",
        "Combine defences: DCA projection followed by DP-GD training, to protect both the released data and the model.",
        "Run a distillation (PATE-style) baseline to complete the IA-1 defence list."], st)
    return s


def stage5():
    s = [PageBreak(), H1("5. Conclusion")]
    s.append(B("This work turns the IA-1 membership inference attack into a common, quantitative yardstick and uses it "
               "to evaluate one defence family from each of the five course modules on a healthcare prediction task. "
               "Four conclusions follow. (i) Leakage is driven by memorisation: model choice and regularisation "
               "matter as much as any dedicated privacy technique. (ii) Central DP gives the best combination of "
               "utility, measured privacy and a provable guarantee for model release, provided ε is moderate (about 4). "
               "(iii) LDP and compressive privacy suit settings where the data holder or the data recipient is not "
               "trusted. LDP needs large user bases; DCA/DUCA can make released data nearly useless for attackers "
               "while keeping it useful for diagnosis. (iv) Syntactic anonymity and output restriction are weak "
               "against MIA and must be backed by l-diversity and formal methods. All code, data hashes and results "
               "are provided for reproduction, and an IEEE-style manuscript draft built on these results accompanies "
               "this report."))
    s.append(H1("6. Ethics and Responsible Data Use"))
    s += bullets([
        "Only public, de-identified datasets were used. No real healthcare service was queried, and no attempt was "
        "made to identify any individual.",
        "The Pima data come from the Akimel O'odham (Pima) community of the Gila River Indian Community. Its "
        "continued use as an ML benchmark has been criticised because of consent for secondary use and the risk of "
        "stigmatising a community. Here it is used only as a methodological benchmark, chosen for continuity with "
        "IA-1. No claims about the community are made, and WDBC (CC BY 4.0) is included so the conclusions do not "
        "depend on this dataset alone.",
        "The attack code is a measurement tool for evaluating defences. Releasing such tools follows standard practice "
        "in privacy research (e.g. TensorFlow Privacy's membership-inference tests)."], st)
    return s


REFS = [
    "R. Shokri, M. Stronati, C. Song and V. Shmatikov, \"Membership inference attacks against machine learning models,\" in *Proc. IEEE Symp. Security and Privacy*, 2017, pp. 3-18.",
    "S. Yeom, I. Giacomelli, M. Fredrikson and S. Jha, \"Privacy risk in machine learning: Analyzing the connection to overfitting,\" in *Proc. IEEE CSF*, 2018, pp. 268-282.",
    "C. A. Choquette-Choo, F. Tramèr, N. Carlini and N. Papernot, \"Label-only membership inference attacks,\" in *Proc. ICML*, 2021.",
    "N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis and F. Tramèr, \"Membership inference attacks from first principles,\" in *Proc. IEEE Symp. Security and Privacy*, 2022.",
    "C. Dwork and A. Roth, \"The algorithmic foundations of differential privacy,\" *Foundations and Trends in Theoretical Computer Science*, vol. 9, no. 3-4, 2014.",
    "M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar and L. Zhang, \"Deep learning with differential privacy,\" in *Proc. ACM CCS*, 2016, pp. 308-318.",
    "M. Bun and T. Steinke, \"Concentrated differential privacy: Simplifications, extensions, and lower bounds,\" in *Proc. TCC*, 2016, pp. 635-658.",
    "P. Kairouz, S. Oh and P. Viswanath, \"The composition theorem for differential privacy,\" in *Proc. ICML*, 2015.",
    "D. Su, J. Cao, N. Li, E. Bertino and H. Jin, \"Differentially private k-means clustering,\" in *Proc. ACM CODASPY*, 2016.",
    "J. C. Duchi, M. I. Jordan and M. J. Wainwright, \"Local privacy and statistical minimax rates,\" in *Proc. IEEE FOCS*, 2013.",
    "N. Wang, X. Xiao, Y. Yang, J. Zhao, S. C. Hui, H. Shin, J. Shin and G. Yu, \"Collecting and analyzing multidimensional data with local differential privacy,\" in *Proc. IEEE ICDE*, 2019.",
    "T. Wang, J. Blocki, N. Li and S. Jha, \"Locally differentially private protocols for frequency estimation,\" in *Proc. USENIX Security*, 2017.",
    "E. Yilmaz, M. Al-Rubaie and J. M. Chang, \"Locally differentially private naive Bayes classification,\" arXiv:1905.01039, 2019.",
    "L. Sweeney, \"k-anonymity: A model for protecting privacy,\" *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems*, vol. 10, no. 5, 2002.",
    "K. LeFevre, D. J. DeWitt and R. Ramakrishnan, \"Mondrian multidimensional k-anonymity,\" in *Proc. IEEE ICDE*, 2006.",
    "A. Machanavajjhala, D. Kifer, J. Gehrke and M. Venkitasubramaniam, \"l-diversity: Privacy beyond k-anonymity,\" *ACM TKDD*, vol. 1, no. 1, 2007.",
    "N. Li, T. Li and S. Venkatasubramanian, \"t-closeness: Privacy beyond k-anonymity and l-diversity,\" in *Proc. IEEE ICDE*, 2007.",
    "R. Agrawal and R. Srikant, \"Privacy-preserving data mining,\" in *Proc. ACM SIGMOD*, 2000.",
    "S. Y. Kung, \"Compressive privacy: From information/estimation theory to machine learning,\" *IEEE Signal Processing Magazine*, vol. 34, no. 1, pp. 94-112, 2017.",
    "S. Y. Kung, T. Chanyaswad, J. M. Chang and P. Wu, \"Collaborative PCA/DCA learning methods for compressive privacy,\" *ACM Trans. Embedded Computing Systems*, vol. 16, no. 3, 2017.",
    "J. M. Chang, D. Zhuang and G. D. Samaraweera, *Privacy-Preserving Machine Learning*. Manning, 2023.",
    "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler and R. S. Johannes, \"Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,\" in *Proc. Symp. Computer Applications in Medical Care*, 1988, pp. 261-265.",
    "W. N. Street, W. H. Wolberg and O. L. Mangasarian, \"Nuclear feature extraction for breast tumor diagnosis,\" in *Proc. SPIE Biomedical Image Processing and Biomedical Visualization*, vol. 1905, 1993.",
    "F. Pedregosa *et al.*, \"Scikit-learn: Machine learning in Python,\" *JMLR*, vol. 12, pp. 2825-2830, 2011.",
]


def references():
    s = [H1("7. References")]
    s += [P(f"[{i}] {r}", st["ref"]) for i, r in enumerate(REFS, 1)]
    return s


def appendix():
    env = prov["environment"]
    s = [PageBreak(), H1("Appendix A - Reproducibility")]
    s.append(table([["Item", "Value"],
        ["Repository", '<link href="https://github.com/anjaliash005/MLCS-IA" color="#1c5cab">https://github.com/anjaliash005/MLCS-IA</link>'],
        ["Python / platform", f"{env['python']} / {env['platform']}"],
        ["Key libraries", f"numpy {env['numpy']}, scikit-learn {env['sklearn']}, scipy 1.13.1, pandas 2.3.3, "
                          "matplotlib 3.9.4, reportlab 4.2.5 (pinned in requirements.lock.txt)"],
        ["Seeds", f"0-{NS - 1} (split, model initialisation, DP noise and LDP noise all derive from the seed)"],
        ["Shadow models", "8 per target, each trained on 192 pool records with 192 disjoint pool records as non-members"],
        ["Raw data hash (Pima)", prov["pima"]["sha256"]],
        ["Raw data hash (WDBC)", prov["breast_cancer"]["sha256"]]] +
        [[f"Processed: {k}", f"{v['path']} - {v['sha256'][:40]}..."] for k, v in prov["processed_files"].items()],
        [3.6 * cm, W - 3.6 * cm], st, font_size=7.2))
    s.append(B("Outputs: per-seed results `results/tables/mia_runs_*.csv`; aggregates `summary_*.csv`; paired tests "
               "`stats_*.csv`; standalone experiments `m2_*.csv`, `m3_*.csv`, `m4_anonymity.csv`, `m5_components.csv`; "
               "figures `results/figures/`; run log `results/run_log.txt`."))
    s.append(H1("Appendix B - Full Pima summary"))
    rows = [["Configuration", "API", "Test acc", "Test F1", "Shadow AUC", "Shadow acc", "Loss AUC", "Loss adv"]]
    for _, r in R.summ[P_].iterrows():
        rows.append([r.config.replace("eps=", "ε=").replace("sigma", "σ"), r["mode"], f3(r.test_acc), f3(r.test_f1),
                     f3(r.sh_auc), f3(r.sh_acc), f3(r.ls_auc), f3(r.ls_adv)])
    s.append(table(rows, [6.0 * cm, 1.4 * cm] + [(W - 7.4 * cm) / 6] * 6, st, font_size=6.6))
    return s


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColorRGB(0.35, 0.35, 0.33)
    if doc.page > 1:
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, f"IA-2 | Security and Privacy for ML | {AUTHOR} ({ROLL}, {BATCH})")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title=TITLE, author=f"{AUTHOR} ({ROLL})",
                            subject="IA-2 Security and Privacy for Machine Learning")
    story = title_page() + abstract() + stage1() + stage2() + stage3() + stage4() + stage5() + references() + appendix()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    strip_dates(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
