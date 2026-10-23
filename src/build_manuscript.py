"""Builds an IEEE-style two-column research manuscript draft from the same
results: reports/IA2_Manuscript_Draft_Anjali_Ashtankar_16014223012.pdf"""
from __future__ import annotations

import os

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer)

import build_report as br
from common import ROOT
from pdfkit import P, f3, figure, pfmt, strip_dates, styles, table

R = br.R
OUT = os.path.join(ROOT, "reports", f"IA2_Manuscript_Draft_{br.AUTHOR.replace(' ', '_')}_{br.ROLL}.pdf")
st = styles(base_size=9.4, leading=11.6)
st["h1"] = ParagraphStyle("mh1", parent=st["h1"], fontName="Times-Bold", fontSize=10, leading=12,
                          alignment=TA_CENTER, spaceBefore=8, spaceAfter=4, textColor=st["body"].textColor)
st["h2"] = ParagraphStyle("mh2", parent=st["h2"], fontName="Times-Italic", fontSize=9.6, leading=11.5,
                          spaceBefore=5, spaceAfter=2)
MARGIN, GAP = 1.6 * cm, 0.6 * cm
COLW = (A4[0] - 2 * MARGIN - GAP) / 2
B = lambda t: P(t, st["body"])
H1 = lambda t: Paragraph(t, st["h1"])
H2 = lambda t: Paragraph(t, st["h2"])
fig = lambda n, cap: figure(os.path.join(br.FIGS, n), COLW, cap, st)
NS, P_, C = br.NS, br.P_, br.C


def header():
    t = ParagraphStyle("t", fontName="Times-Roman", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=8)
    a = ParagraphStyle("a", fontName="Times-Roman", fontSize=10.5, leading=13, alignment=TA_CENTER)
    return [Paragraph(br.TITLE, t),
            Paragraph(f"{br.AUTHOR}<br/><i>Roll No. {br.ROLL}, Batch {br.BATCH} - Security and Privacy for Machine "
                      "Learning (IA-2 manuscript draft)</i><br/><font size=9>Code and data: <link href='https://github.com/anjaliash005/MLCS-IA' color='#1c5cab'>https://github.com/anjaliash005/MLCS-IA</link></font>", a),
            Spacer(1, 10)]


def body():
    s = []
    s.append(P(f"<b><i>Abstract</i>—Membership inference attacks (MIAs) reveal whether a record was used to train a "
               f"model, which in healthcare can disclose a diagnosis. We use a black-box MIA as a common yardstick to "
               f"compare five families of privacy-preserving techniques on a diabetes-risk task: regularisation, "
               f"central differential privacy (DP-GD with a zCDP accountant), local differential privacy (Piecewise "
               f"mechanism and LDP Naive Bayes), privacy-preserving data mining (Mondrian k-anonymity, l-diversity, "
               f"input noise, output restriction) and compressive privacy (PCA/DCA/DUCA). Across {NS} paired splits "
               f"with Kerckhoffs-style shadow models, an overfit MLP leaks membership with attack AUC {br.b_auc:.2f}. "
               f"DP-GD at ε = 4 keeps {100 * br.dp4_acc:.1f}% accuracy at AUC {br.dp4_auc:.2f}, while label-only "
               f"outputs and k-anonymity reduce AUC only to {br.lab_auc:.2f} and {br.k25_auc:.2f}. A single DCA "
               f"component keeps {100 * br.dca1_util:.1f}% accuracy while reducing age inference from "
               f"{br.raw_attr:.2f} to {br.dca1_attr:.2f} balanced accuracy. The results replicate on a second medical "
               f"dataset.</b>", st["body"]))
    s.append(P("<b><i>Index Terms</i>—membership inference, differential privacy, local differential privacy, "
               "k-anonymity, compressive privacy, healthcare machine learning.</b>", st["body"]))

    s.append(H1("I. INTRODUCTION"))
    s.append(B("Releasing a trained model is not the same as releasing nothing about its training data. Shokri et al. [1] "
               "showed that black-box outputs reveal training-set membership, and Yeom et al. [2] linked this leakage to "
               "overfitting. Many defences have been proposed, but they come from separate communities (differential "
               "privacy [5], local DP [10], data anonymisation [14], compressive privacy [19]; see [21] for a unified treatment) and are "
               "usually evaluated "
               "with incompatible metrics. A practitioner choosing among them for a clinical model has no common "
               "basis for comparison."))
    s.append(B("**Contributions.** (1) A reproducible harness in which every defence is a training recipe attacked by "
               "shadow models trained with the *same* recipe. (2) First-principles implementations of DP-GD with zCDP "
               "accounting, four LDP mechanisms with OUE-based LDP Naive Bayes, Mondrian k-anonymity with l-diversity, "
               "and DCA/DUCA projections. (3) A paired, multi-seed comparison on two public medical datasets, "
               "including an ablation that separates the effect of DP noise from that of shortened training, and a "
               "comparison of measured advantage with the theoretical DP bound."))

    s.append(H1("II. BACKGROUND AND RELATED WORK"))
    s.append(B("*Membership inference.* Shadow-model attacks [1] train a meta-classifier on outputs of models with known "
               "membership; loss-threshold attacks [2] need no shadows; label-only attacks [3] work even when scores are "
               "hidden; LiRA [4] targets the low-FPR regime. *Differential privacy.* DP-SGD [6] clips per-example "
               "gradients and adds Gaussian noise; zCDP [7] composes Gaussian steps tightly; a mechanism that is "
               "(ε, δ)-DP limits any membership test to TPR − FPR ≤ (e^ε − 1 + 2δ)/(e^ε + 1) [8]. *Local DP* removes "
               "trust in the collector: Duchi's mechanism [10], the Piecewise and Hybrid mechanisms [11], and frequency "
               "oracles such as OUE [12] that enable LDP Naive Bayes [13]. *Syntactic anonymity*: k-anonymity [14] via "
               "Mondrian [15], l-diversity [16] and t-closeness [17]; random input perturbation [18]. *Compressive "
               "privacy* [19], [20] releases low-dimensional projections designed to keep a utility task and suppress "
               "a privacy task."))

    s.append(H1("III. THREAT MODEL AND METHOD"))
    s.append(B("A hospital trains a binary classifier on n = 192 records (members); another 192 are non-members and "
               "serve as the test set. The attacker holds 384 records from the same distribution, knows the training "
               "recipe and any published transform, and queries the model as a black box. The API returns full "
               "probabilities, probabilities rounded to 0.1, or labels only. For each configuration the attacker "
               "trains 8 shadow models with the defended recipe, extracts features (true-class probability, loss, "
               "entropy, maximum probability, correctness, logit, label), and fits a logistic-regression attack. A "
               "loss-threshold attack calibrated on the shadows is run alongside it. We report attack ROC-AUC, "
               "calibrated advantage and maximum advantage."))
    s.append(B(f"*Defences.* **M1:** logistic regression, MLP-32, MLP-128x2 without regularisation (baseline), the same "
               f"with L2 = 3 and early stopping, and a deep random forest. **M2:** full-batch DP-GD (T = 200, C = 1, "
               f"δ = 10<super>-5</super>), where ε = ρ + 2√(ρ ln(1/δ)) and ρ = T/(2σ²); σ = {br.sig.loc[4, 'sigma']:.1f} gives ε = 4. "
               f"**M3:** each patient perturbs 10 features and the label with ε/11 each (Piecewise + randomised "
               f"response), or reports one sampled (class, bin) pair through OUE for LDP-NB. **M4:** Mondrian on Age, "
               f"Pregnancies and BMI with k ∈ {{5, 10, 25}} and optional l = 2; Gaussian input noise; output rounding "
               f"and label-only APIs. **M5:** PCA, DCA (S_B + ρ'I vs S_W + S_B + ρI) and DUCA (adds γS_B of the "
               f"privacy label, Age >= 35, to the denominator). An unsupervised DP k-means demonstration [9] is reported in "
               f"the companion technical report."))
    s.append(B("*Data.* Pima Indians Diabetes (768 records, 268 positive) [22] and Breast Cancer Wisconsin Diagnostic "
               "(569 records) [23]. Code, data and results are available at https://github.com/anjaliash005/MLCS-IA. Everything is implemented in NumPy, with scikit-learn [24] used for the non-private "
               "models and metrics. All scaling uses fixed public bounds, so preprocessing consumes no privacy budget. "
               f"Results are means over seeds 0-{NS - 1}; paired Wilcoxon tests compare each configuration with the "
               "baseline."))

    s.append(H1("IV. RESULTS"))
    picks = [(C, "full", "Baseline MLP-128x2"), (br.D, "full", "+L2 + early stop"), (br.E, "full", "Deep RF"),
             (C, "label", "Baseline, label-only API"), ("DP-GD MLP-128 eps=1", "full", "DP-GD MLP ε=1"),
             ("DP-GD MLP-128 eps=4", "full", "DP-GD MLP ε=4"), ("DP-GD LogReg eps=4", "full", "DP-GD LR ε=4"),
             ("LDP Naive Bayes eps=4", "full", "LDP-NB ε=4"), ("Mondrian k=25", "full", "Mondrian k=25"),
             ("Mondrian k=10, l=2", "full", "Mondrian k=10, l=2"), ("Input noise sigma=0.5", "full", "Input noise σ=0.5"),
             ("DCA k=1", "full", "DCA k=1"), ("DUCA k=2", "full", "DUCA k=2")]
    picks.insert(1, ("A: LogReg (L2)", "full", "Logistic regression"))
    rows = [["Configuration", "Acc", "AUC*", "p (AUC)"]]
    for cfg, mode, lab in picks:
        r = R.row(P_, cfg, mode)
        t = R.stat(P_, cfg, mode) if not (cfg == C and mode == "full") else None
        rows.append([lab, f3(r.test_acc), f3(max(r.sh_auc, r.ls_auc)), pfmt(t.p_sh_auc) if t is not None else "ref."])
    s.append(KeepTogether([P(f"<b>TABLE I.</b> Pima results (mean of {NS} seeds). AUC* = worst-case attack AUC; p = "
                             "Wilcoxon vs baseline, shadow AUC.", st["cap"]),
                           table(rows, [3.9 * cm, 1.3 * cm, 1.3 * cm, COLW - 6.5 * cm], st, font_size=7)]))
    s.append(Spacer(1, 4))
    s.append(B(f"*Overfitting (M1).* The baseline reaches train-test gap {br.b_gap:.2f} and attack AUC {br.b_auc:.2f}; "
               f"the deep random forest leaks even more ({br.e_auc:.2f}) despite higher accuracy. Regularisation lowers "
               f"AUC to {br.d_auc:.2f} without a significant accuracy change, and plain logistic regression is both the "
               f"most accurate ({100 * br.a_acc:.1f}%) and essentially leak-free ({br.a_auc:.2f})."))
    s.append(B(f"*Central DP (M2).* Accuracy rises with ε, from {100 * br.dp1_acc:.1f}% at ε = 1 to "
               f"{100 * br.dp4_acc:.1f}% at ε = 4, while attack AUC stays near 0.5. The ablation shows that the same "
               f"network trained without noise for 2000 steps leaks (AUC {br.npl_auc:.2f}), but for 200 steps it does "
               f"not ({br.dpinf_auc:.2f}). Measured maximum advantage stays within the random-score noise floor and "
               f"far below the bound of [8]."))
    s.append(fig("fig08_frontier.png", "Fig. 1. Privacy-utility landscape of all configurations on Pima (lower right is better)."))
    s.append(B(f"*Local DP (M3).* All mechanisms match their theoretical variance; Piecewise beats Duchi from ε = 1 upward, "
               f"as in [11]. LDP-trained models are near chance for MIA, but utility is data-limited: LDP-NB reaches "
               f"{100 * br.nb4_acc:.1f}% at ε = 4 (non-private {100 * br.nbinf_acc:.1f}%). A bootstrap simulation "
               f"with 50,000 users gives {100 * R.nb(50000, 1):.1f}% at ε = 1."))
    s.append(B(f"*PPDM (M4).* Label-only outputs leave AUC at {br.lab_auc:.2f} because an overfit model is right far "
               f"more often on its members [3]. k-anonymity with k = 25 gives AUC {br.k25_auc:.2f}. Without l-diversity, "
               f"{100 * br.homog_k10_l1:.1f}% of records at k = 10 fall in diagnosis-homogeneous classes; l = 2 removes "
               f"this exposure for a small increase in NCP ({br.ncp_k10_l1:.2f} to {br.ncp_k10_l2:.2f})."))
    s.append(B(f"*Compressive privacy (M5).* PCA retains reconstructable, age-revealing variance (age balanced accuracy "
               f"{br.pca1_attr:.2f} at k = 1). DCA concentrates the diagnostic signal: one component yields "
               f"{100 * br.dca1_util:.1f}% accuracy (raw {100 * br.raw_util:.1f}%), age inference {br.dca1_attr:.2f} "
               f"and reconstruction NRMSE {br.dca2_rec:.2f} at k = 2. DUCA lowers age inference to {br.duca1_attr:.2f}. "
               f"Models trained on z can still memorise once z has enough dimensions (DCA k >= 2, PCA k >= 4), so CP protects released data rather than models."))
    rows = [["Projection", "Utility acc", "Age bal. acc", "Recon. NRMSE"]]
    for m, k in [("Raw", 10), ("PCA", 1), ("PCA", 4), ("DCA", 1), ("DCA", 2), ("DUCA", 1), ("DUCA", 2)]:
        rows.append([("Raw features" if m == "Raw" else f"{m}, k = {k}"), f3(R.comp_mean(P_, m, k, "utility_acc")),
                     f3(R.comp_mean(P_, m, k, "attr_bacc")), f3(R.comp_mean(P_, m, k, "recon_nrmse"))])
    s.append(KeepTogether([P("<b>TABLE II.</b> Compressive privacy on Pima: utility (logistic regression on z), "
                             "attribute-inference attacker (Age >= 35) and linear reconstruction of x.", st["cap"]),
                           table(rows, [2.6 * cm, 1.9 * cm, 1.9 * cm, COLW - 6.4 * cm], st, font_size=7)]))
    s.append(Spacer(1, 4))
    s.append(fig("fig02_m1_overfitting_roc.png", "Fig. 2. Left: attack AUC vs generalisation gap (models A-E, both "
                 "datasets). Right: shadow-attack ROC curves on one Pima split."))
    s.append(B(f"*Replication.* On WDBC the baseline generalises well (accuracy {100 * br.bc_b_acc:.1f}%, AUC "
               f"{br.bc_b_auc:.2f}), while the deep random forest remains leaky ({br.bc_e_auc:.2f}). The ordering of "
               f"defences is unchanged."))

    s.append(H1("V. DISCUSSION AND LIMITATIONS"))
    s.append(B("On small tabular data, matching model capacity to the data is the cheapest defence. Beyond that, formal "
               "training-side protection is the most robust choice for model release. Compressive projections "
               "are the most effective for data release. Syntactic and output-side measures should be complements, "
               "not replacements. Our study is limited to two small tabular datasets, to average-case attackers (AUC "
               "rather than TPR at low FPR [4]) and to full-batch DP-GD without subsampling amplification, so the "
               "reported ε is conservative. Hyperparameters of the private models were not selected privately."))
    s.append(B("*Ethics.* Only public, de-identified data and locally trained models were used. The Pima data originate "
               "from an Indigenous community and their benchmark use has been criticised, so no claims about the "
               "community are made and a second dataset is included."))
    s.append(H1("VI. CONCLUSION"))
    s.append(B("Using one attack as the yardstick turns a collection of privacy techniques into comparable options. In "
               "our healthcare setting, moderate-ε DP-GD offers the best guarantee-utility balance for releasing "
               "models, DCA/DUCA offer the best balance for releasing data, and LDP becomes attractive once user bases "
               "reach tens of thousands. Future work will add subsampled DP-SGD with RDP accounting, LiRA-style attacks "
               "and combined DCA + DP pipelines on larger clinical datasets."))
    s.append(H1("REFERENCES"))
    ref = ParagraphStyle("mref", parent=st["ref"], fontSize=7.6, leading=9.2)
    s += [P(f"[{i}] {r}", ref) for i, r in enumerate(br.REFS, 1)]
    return s


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.7 * cm,
                          bottomMargin=1.7 * cm, title=br.TITLE, author=f"{br.AUTHOR} ({br.ROLL})")
    H = A4[1] - 3.4 * cm
    head_h = 5.2 * cm
    first = [Frame(MARGIN, 1.7 * cm + H - head_h, A4[0] - 2 * MARGIN, head_h, id="head"),
             Frame(MARGIN, 1.7 * cm, COLW, H - head_h, id="c1"),
             Frame(MARGIN + COLW + GAP, 1.7 * cm, COLW, H - head_h, id="c2")]
    later = [Frame(MARGIN, 1.7 * cm, COLW, H, id="l1"), Frame(MARGIN + COLW + GAP, 1.7 * cm, COLW, H, id="l2")]

    def foot(c, d):
        c.saveState(); c.setFont("Times-Roman", 8)
        c.drawCentredString(A4[0] / 2, 1.0 * cm, f"{d.page}")
        c.restoreState()

    doc.addPageTemplates([PageTemplate("first", first, onPage=foot), PageTemplate("later", later, onPage=foot)])
    story = header() + [NextPageTemplate("later"), FrameBreak()] + body()
    doc.build(story)
    strip_dates(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
