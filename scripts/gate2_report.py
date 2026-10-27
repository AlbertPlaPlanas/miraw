"""Gate 2 report from the committed result files: non-inferiority of the retrained models vs the paper.

Site level : per-fold differences vs the paper's own fold models re-scored (tests/fixtures/paper_cv_xent_models_rescored.csv).
Gene level : per (config, set) differences vs the paper's GlobalSummary (tests/fixtures/paper_global_summary.csv).
Exit code 0 only if every row is non-inferior.
"""

import sys
from pathlib import Path

import pandas as pd

from pymiraw.evaluation.noninferiority import CONFIDENCE, MARGINS, non_inferiority

root = Path(__file__).resolve().parents[1]
site_paper = pd.read_csv(root / "tests/fixtures/paper_cv_xent_models_rescored.csv").set_index("fold")
gene_paper = pd.read_csv(root / "tests/fixtures/paper_global_summary.csv")
cv = pd.read_csv(root / "docs/results/cv_paper_regime.csv")
gene = pd.read_csv(root / "docs/results/gene_level_fold6_retrained.csv")

rows = []
for o, df in cv[~cv.collapsed].groupby("orientation"):
    f = df.groupby("fold")[["Accuracy", "AUC", "F1"]].mean()
    for m in ("Accuracy", "AUC"):
        rows.append((o, "site", m, non_inferiority(f[m] - site_paper[m], m)))
gene = gene.merge(gene_paper[["config", "set", "Accuracy", "F1"]], on=["config", "set"], suffixes=("", "_paper"))
for o, df in gene.groupby("orientation"):
    cell = (
        df.assign(dAccuracy=df.Accuracy - df.Accuracy_paper, dF1=df.F1 - df.F1_paper)
        .groupby(["config", "set"])[["dAccuracy", "dF1"]]
        .mean()
    )
    rows.append((o, "gene", "Accuracy", non_inferiority(cell.dAccuracy, "Accuracy")))
    rows.append((o, "gene", "F1", non_inferiority(cell.dF1, "F1")))

print(
    f"Gate 2 non-inferiority: one-sided {CONFIDENCE:.0%} lower bound of (ours - paper) must exceed -margin; margins {MARGINS}\n"
)
print(
    f"{'orientation':12s} {'level':5s} {'metric':9s} {'n':>3s} {'mean':>8s} {'lower':>8s} {'limit':>8s} {'slack':>8s}  verdict"
)
ok = True
for o, lvl, m, v in rows:
    ok &= v.non_inferior
    print(
        f"{o:12s} {lvl:5s} {m:9s} {v.n:3d} {v.mean:+8.4f} {v.lower:+8.4f} {-v.margin:+8.4f} {v.slack:+8.4f}  {'PASS' if v.non_inferior else 'FAIL'}{'  (fragile)' if v.non_inferior and v.slack < 0.001 else ''}"
    )
print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
