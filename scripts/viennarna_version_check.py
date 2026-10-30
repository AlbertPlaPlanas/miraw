"""Do the per-pair site-count differences vs the paper come from the ViennaRNA version?

Runs the first N pairs of a balanced test set through the pipeline with two folding backends (the in-process
2.7.x bindings and the 2.2.5 command-line tools from the miraw-ref environment) and compares PosSites /
NegSites / Prediction with the paper's recorded targetPredictionOutput.csv.

Usage: python scripts/viennarna_version_check.py <test_set.csv> <paper_targetPredictionOutput.csv> <weights.bin> [N]
"""

import sys

import pandas as pd

from miraw.prediction.pipeline import predict_file, results_frame

test_csv, paper_csv, weights = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
paper = pd.read_csv(paper_csv, sep="\t", index_col=False).head(n)
base = dict(model=weights, finder="miRAW", filters=[], max_site_length=40, seed_alignment_offset=5, cache_dir=None)

variants = {
    "bindings 2.7.2": dict(backend="bindings"),
    "cli 2.2.5": dict(backend="cli", backend_args={"bin_dir": "/opt/conda/envs/miraw-ref/bin"}),
}
for name, extra in variants.items():
    df = results_frame(predict_file(test_csv, {**base, **extra}, workers=8, limit=n))
    same_counts = ((df.PosSites.values == paper.PosSites.values) & (df.NegSites.values == paper.NegSites.values)).mean()
    same_pos = (df.PosSites.values == paper.PosSites.values).mean()
    same_call = ((df.Prediction > 0).values == (paper.Prediction > 0).values).mean()
    maxdiff = (df.Prediction.values - paper.Prediction.values).__abs__().max()
    print(
        f"{name:15s} same Pos+Neg counts {same_counts:.1%} | same PosSites {same_pos:.1%} | same gene call {same_call:.1%} | max |score diff| {maxdiff:.5f}"
    )
