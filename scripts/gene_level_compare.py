"""Gene-level comparison of retrained fold models with the paper (same pipeline, same balanced sets).

Usage: python scripts/gene_level_compare.py runs [out.csv]   (optionally writes the per-run rows; reads runs/gene_<orientation>_f6_s<seed>/<config>_<set>/metrics.json)
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from miraw import resources

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
paper = resources.paper_global_summary()
rows = []
for d in sorted(root.glob("gene_*_f*_s*")):
    if not d.is_dir():
        continue
    _, orient, fold, seed = d.name.split("_")  # gene_<orientation>_f<fold>_s<seed>
    for m in d.glob("*/metrics.json"):
        cfg, _, s = m.parent.name.rpartition("_")  # e.g. Personalized_0_9_7_NF_0 -> cfg Personalized_0_9_7_NF
        finder, _, flt = cfg.rpartition("_")
        if finder.startswith("Personalized_"):
            finder = finder.replace("_", ";")  # Personalized_0_9_7 -> Personalized;0;9;7
        rows.append(
            {
                "orientation": orient,
                "fold": int(fold[1:]),
                "seed": seed,
                "config": f"{finder}|{flt}",
                "set": int(s),
                **json.loads(m.read_text()),
            }
        )
ours = pd.DataFrame(rows)
if len(sys.argv) > 2:  # persist the per-run rows (the committed docs/results files are produced this way)
    out, is_paper = Path(sys.argv[2]), ours.orientation == "paper"  # gene_paper_f<f>_s0 = the paper's own fold models
    keys = ["orientation", "fold", "seed", "config", "set"]
    ours[~is_paper].sort_values(keys).to_csv(out, index=False)
    if is_paper.any():
        ours[is_paper].sort_values(keys).to_csv(out.with_name("gene_level_paper_folds.csv"), index=False)
ours = ours.merge(paper[["config", "set", "Accuracy", "F1"]], on=["config", "set"], suffixes=("", "_paper"))
ours["dAcc"], ours["dF1"] = ours.Accuracy - ours.Accuracy_paper, ours.F1 - ours.F1_paper
print(
    f"{len(ours)} runs: {ours.groupby('orientation').seed.nunique().to_dict()} seeds, sets {sorted(ours.set.unique())}, configs {sorted(ours.config.unique())}\n"
)
g = (
    ours.groupby(["orientation", "config"])[["Accuracy", "Accuracy_paper", "dAcc", "F1", "F1_paper", "dF1"]]
    .mean()
    .round(3)
)
print(g.to_string(), "\n")
for o, df in ours.groupby("orientation"):
    cell = df.groupby(["config", "set"])[["dAcc", "dF1"]].mean()  # average over seeds -> one value per (config, set)
    print(
        f"{o:12s} mean dAcc {cell.dAcc.mean():+.4f}  mean dF1 {cell.dF1.mean():+.4f}  "
        f"dF1>=0 in {(cell.dF1 >= 0).sum()}/{len(cell)} cells; one-sided Wilcoxon p(worse than paper in F1) = {wilcoxon(cell.dF1, alternative='less').pvalue:.3f}"
    )
a = ours.pivot_table(index=["seed", "config", "set"], columns="orientation", values="F1")
if {"natural", "old-default"} <= set(a.columns):
    x = (a["natural"] - a["old-default"]).dropna()
    print(
        f"\nnatural - old-default gene-level F1: mean {x.mean():+.4f}, better in {(x > 0).sum()}/{len(x)}, one-sided Wilcoxon p = {wilcoxon(x, alternative='greater').pvalue:.4f}"
    )
