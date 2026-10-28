"""Gate 2 report from the committed result files: non-inferiority of the retrained models vs the paper (rule: docs/PLAN.md section 6).

Verdict rows:
  site   : per-fold differences vs the paper's own fold models re-scored (tests/fixtures/paper_cv_xent_models_rescored.csv).
  gene   : per (config, set) differences vs the paper's GlobalSummary (tests/fixtures/paper_global_summary.csv).
  gene/p : per-fold paired differences, our fold-f model - the paper's fold-f model through the same pipeline
           (docs/results/gene_level_paper_folds.csv); required once more than one fold has gene-level results.
Diagnostics (printed, not in the verdict):
  gene/f : per-fold differences vs the GlobalSummary (a single model, paper fold 6), for our models and the paper's.
Exit code 0 only if every verdict row is non-inferior.
"""

import sys
from pathlib import Path

import pandas as pd

from pymiraw.evaluation.noninferiority import CONFIDENCE, MARGINS, non_inferiority

root = Path(__file__).resolve().parents[1]
site_paper = pd.read_csv(root / "tests/fixtures/paper_cv_xent_models_rescored.csv").set_index("fold")
gene_paper = pd.read_csv(root / "tests/fixtures/paper_global_summary.csv")
cv = pd.read_csv(root / "docs/results/cv_paper_regime.csv")
gene = pd.read_csv(root / "docs/results/gene_level_retrained.csv")
if "fold" not in gene:  # files written before the fold column existed are all fold 6
    gene["fold"] = 6
paper_folds_file = root / "docs/results/gene_level_paper_folds.csv"  # the paper's own fold models, same pipeline
paper_folds = pd.read_csv(paper_folds_file) if paper_folds_file.exists() else None


def vs_global_summary(df):
    df = df.merge(gene_paper[["config", "set", "Accuracy", "F1"]], on=["config", "set"], suffixes=("", "_paper"))
    return df.assign(dAccuracy=df.Accuracy - df.Accuracy_paper, dF1=df.F1 - df.F1_paper)


rows, diagnostics, missing = [], [], []
for o, df in cv[~cv.collapsed].groupby("orientation"):
    f = df.groupby("fold")[["Accuracy", "AUC", "F1"]].mean()
    for m in ("Accuracy", "AUC"):
        rows.append((o, "site", m, non_inferiority(f[m] - site_paper[m], m)))
for o, df in gene.groupby("orientation"):
    d = vs_global_summary(df)
    cell = d.groupby(["config", "set"])[["dAccuracy", "dF1"]].mean()  # average over folds and seeds
    rows.append((o, "gene", "Accuracy", non_inferiority(cell.dAccuracy, "Accuracy")))
    rows.append((o, "gene", "F1", non_inferiority(cell.dF1, "F1")))
    if df.fold.nunique() < 2:
        continue
    if paper_folds is None:
        missing.append(o)
    else:  # one paired value per fold: mean over the shared (config, set) cells of ours fold f - paper fold f
        mine = df.groupby(["fold", "config", "set"])[["Accuracy", "F1"]].mean()  # average over seeds
        ref = paper_folds.groupby(["fold", "config", "set"])[["Accuracy", "F1"]].mean()
        p = (mine - ref).dropna().groupby("fold").mean()
        rows.append((o, "gene/p", "Accuracy", non_inferiority(p.Accuracy, "Accuracy")))
        rows.append((o, "gene/p", "F1", non_inferiority(p.F1, "F1")))
    per_fold = d.groupby("fold")[["dAccuracy", "dF1"]].mean()
    diagnostics.append((o, per_fold))
if paper_folds is not None:
    diagnostics.append(("paper folds", vs_global_summary(paper_folds).groupby("fold")[["dAccuracy", "dF1"]].mean()))

header = f"{'orientation':12s} {'level':6s} {'metric':9s} {'n':>3s} {'mean':>8s} {'lower':>8s} {'limit':>8s} {'slack':>8s}  verdict"


def line(o, lvl, m, v):
    fragile = "  (fragile)" if v.non_inferior and v.slack < 0.001 else ""
    return f"{o:12s} {lvl:6s} {m:9s} {v.n:3d} {v.mean:+8.4f} {v.lower:+8.4f} {-v.margin:+8.4f} {v.slack:+8.4f}  {'PASS' if v.non_inferior else 'FAIL'}{fragile}"


print(
    f"Gate 2 non-inferiority: one-sided {CONFIDENCE:.0%} lower bound of (ours - paper) must exceed -margin; margins {MARGINS}\n"
)
print(header)
ok = not missing
for r in rows:
    ok &= r[3].non_inferior
    print(line(*r))
for o in missing:
    print(f"{o:12s} gene/p    -- missing: {paper_folds_file.relative_to(root)} (paper fold models) -> FAIL")
if diagnostics:
    print("\nDiagnostics, not in the verdict: per-fold differences vs the GlobalSummary (one model, paper fold 6)")
    print(header)
    for o, pf in diagnostics:
        for m in ("Accuracy", "F1"):
            print(line(o, "gene/f", m, non_inferiority(pf[f"d{m}"], m)))
        print(f"{'':12s} per-fold dAcc {pf.dAccuracy.round(3).to_dict()}")
print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
