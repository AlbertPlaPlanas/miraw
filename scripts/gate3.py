"""Gate 3: the original weights on the paper's full unbalanced test set (17,043 pairs: 16,495 targets, 548 non-targets).

Rule (docs/PLAN.md section 6): the AUC of the default configuration (miRAW-6-1:10, no filter) is at least the
paper's on the same pairs. All 10 configurations are run and compared with the paper's results as well.

Steps (default: all):
  build   rebuild the test file data/original/Data/TestData/fullTestSet/unbalancedTestSplit.csv, which the paper
          archive does not ship: the pairs of the paper's results (Results/Evaluate_14K_FULL/<config>/
          targetPredictionOutput.csv) with their sequences from the original repository's
          TestData/Data/fullUnifiedFile.csv. Checks: every pair is found exactly once with the paper's label, and
          the sequences equal those of the 10 balanced test sets.
  run     predict every configuration in chunks (resumable: finished chunks are kept) into runs/gate3/<config>/.
  report  AUC, accuracy, F1, ... of pyMIRAW and of the paper on the same pairs -> docs/results/gate3_comparison.csv.
          Exit code 0 only if the gate passes.

Usage: python scripts/gate3.py [steps...] [--unified PATH] [--workers 14] [--configs miRAW|NF ...]
Needs data/original (`pymiraw data unpack-archive PLOSComb.7z`) and, for `build`, fullUnifiedFile.csv (688 MB) from
https://bitbucket.org/bipous/miraw_-dl4mirnatargeting_code (TestData/Data/; get it with git, see docs/data.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pymiraw.config import PredictConfig
from pymiraw.evaluation.metrics import gene_level_metrics, roc_auc
from pymiraw.evaluation.parity import CONFIGS, DISPLAY, FILTERS
from pymiraw.prediction.pipeline import predict_pairs, read_unified, results_frame
from pymiraw.storage import ORIGINAL_DIR, cache_dir

ROOT = Path(__file__).resolve().parents[1]
PAPER = ORIGINAL_DIR / "Results/Evaluate_14K_FULL"
TEST_FILE = ORIGINAL_DIR / "Data/TestData/fullTestSet/unbalancedTestSplit.csv"
BALANCED = ORIGINAL_DIR / "Data/TestData/balanced10"
UNIFIED = cache_dir() / "raw/original/miraw_code/repo/TestData/Data/fullUnifiedFile.csv"
OUT = ROOT / "runs/gate3"
RESULT = ROOT / "docs/results/gate3_comparison.csv"
DEFAULT = "miRAW|NF"
CHUNK = 1000
# the paper's result folders per configuration; two runs were split in two parts by the authors
PAPER_DIRS = {
    "miRAW|NF": ["miRAW_NF", "miRAW_NF_last"],
    "miRAW|AE": ["miRAW_AccessibilityEnergy10"],
    "Personalized;0;9;7|NF": ["Personalized_9_7_NF"],
    "Personalized;0;9;7|AE": ["Personalized_9_7_AccessibilityEnergy10"],
    "Personalized;1;9;7|NF": ["Personalized_1_9_7_NF_front", "Personalized_1_9_7_NF_end"],
    "Personalized;1;9;7|AE": ["Personalized_1_9_7_AccessibilityEnergy10"],
    "PITA|NF": ["pita_NF"],
    "PITA|AE": ["pita_AccessibilityEnergy10"],
    "TS|NF": ["ts_NF"],
    "TS|AE": ["ts_AccessibilityEnergy10"],
}
KEY = ["miRNA", "GeneId"]


def paper_results(config: str) -> pd.DataFrame:
    files = [PAPER / d / "targetPredictionOutput.csv" for d in PAPER_DIRS[config]]
    return pd.concat([pd.read_csv(f, sep="\t", index_col=False) for f in files], ignore_index=True)


def pair_set(df: pd.DataFrame) -> set:
    return set(zip(df.miRNA, df.GeneId, strict=True))


def build(args) -> None:
    pairs = paper_results("miRAW|AE")[["GeneName", "GeneId", "miRNA", "RealClass"]]  # one complete run, input order
    for config in PAPER_DIRS:
        other = paper_results(config)
        if len(other) != len(pairs) or pair_set(other) != pair_set(pairs):
            sys.exit(f"the paper's {config} results cover other pairs")
    print(f"paper pairs: {len(pairs)} ({int(pairs.RealClass.sum())} targets)")
    full = pd.read_csv(args.unified, sep="\t", dtype=str, keep_default_na=False)
    full = full.merge(pairs.rename(columns={"GeneId": "EnsemblId"})[["miRNA", "EnsemblId"]], on=["miRNA", "EnsemblId"])
    dup = full.duplicated(["miRNA", "EnsemblId"], keep=False)
    if dup.any():  # the same pair more than once: keep it only if every copy is identical
        same = full[dup].groupby(["miRNA", "EnsemblId"]).nunique().le(1).all(axis=1)
        if not same.all():
            sys.exit(f"{int((~same).sum())} pairs have different sequences or labels in {args.unified}")
        full = full.drop_duplicates(["miRNA", "EnsemblId"])
    test = pairs.rename(columns={"GeneId": "EnsemblId"}).merge(full, on=["miRNA", "EnsemblId"], how="left")
    missing = int(test.Mature_mirna_transcript.isna().sum())
    if missing:
        sys.exit(f"{missing} of the paper's pairs are not in {args.unified}")
    wrong = int((test.Positive_Negative.astype(int) != test.RealClass).sum())
    if wrong:
        sys.exit(f"{wrong} pairs have a different label in {args.unified}")
    cols = ["miRNA", "gene_name", "EnsemblId", "Positive_Negative", "Mature_mirna_transcript", "3UTR_transcript"]
    test = test[cols]
    seqs = test.set_index(["miRNA", "EnsemblId"])[["Mature_mirna_transcript", "3UTR_transcript"]]
    checked = 0
    for f in sorted(BALANCED.glob("randomLeveragedTestSplit_*.csv")):
        b = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False).set_index(["miRNA", "EnsemblId"])
        b = b[["Mature_mirna_transcript", "3UTR_transcript"]]
        common = b.index.intersection(seqs.index)
        if not (b.loc[common] == seqs.loc[common]).all().all():
            sys.exit(f"sequences differ from {f.name}")
        checked += len(common)
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(TEST_FILE, sep="\t", index=False)
    print(f"wrote {TEST_FILE}: {len(test)} pairs; {checked} balanced-set rows have identical sequences")


def run(args) -> None:
    pairs = read_unified(TEST_FILE)
    order = sorted(args.configs, key=lambda c: c != DEFAULT)  # the gate's configuration first
    for config in order:
        finder, flt = config.split("|")
        folder = OUT / config.replace(";", "_").replace("|", "_")
        folder.mkdir(parents=True, exist_ok=True)
        cfg = PredictConfig(
            finder=finder,
            filters=FILTERS[flt],
            cache_dir=str(cache_dir() / "foldcache"),
            sites_dir=str(OUT / "sites"),
        )
        for k, start in enumerate(range(0, len(pairs), CHUNK)):
            f = folder / f"chunk_{k:02d}.tsv"
            if f.exists():
                continue
            res = predict_pairs(pairs.iloc[start : start + CHUNK], cfg, workers=args.workers)
            tmp = f.with_suffix(".tmp")
            results_frame(res).to_csv(tmp, sep="\t", index=False)
            tmp.replace(f)
            print(f"{config}: chunk {k} done", flush=True)


def ours(config: str) -> pd.DataFrame | None:
    folder = OUT / config.replace(";", "_").replace("|", "_")
    n = -(-len(read_unified(TEST_FILE)) // CHUNK)
    files = sorted(folder.glob("chunk_*.tsv"))
    if len(files) < n:
        return None
    return pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)


def summary(df: pd.DataFrame) -> dict:
    m = gene_level_metrics(df.RealClass.astype(int), df.Prediction)
    return {"AUC": roc_auc(df.RealClass.astype(int), df.Prediction), **m}


def report(args) -> int:
    rows = []
    for config in PAPER_DIRS:
        mine = ours(config)
        if mine is None:
            print(f"{config}: not run (or not finished)")
            continue
        paper = paper_results(config)
        both = mine.merge(paper[[*KEY, "Prediction"]], on=KEY, suffixes=("", "_paper"))
        if len(both) != len(paper):
            sys.exit(f"{config}: {len(paper) - len(both)} pairs could not be matched with the paper's results")
        a, b = summary(mine), summary(paper)
        row = {"config": config, "pairs": len(both)}
        for k in ("AUC", "Accuracy", "F1", "Recall", "Specificity"):
            row |= {f"{k}_ours": a[k], f"{k}_paper": b[k], f"{k}_diff": a[k] - b[k]}
        row["same_call"] = float(((both.Prediction > 0) == (both.Prediction_paper > 0)).mean())
        rows.append(row)
    if not rows:
        print("nothing to report")
        return 1
    df = pd.DataFrame(rows).set_index("config")
    df.index = [f"{DISPLAY[c.split('|')[0]]} {c.split('|')[1]}" for c in df.index]
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(RESULT)
    cols = [c for c in df.columns if c.startswith(("AUC", "Accuracy_d", "F1_d"))] + ["same_call"]
    print(df[cols].round(4).to_string())
    gate = df.loc[f"{DISPLAY['miRAW']} NF"]
    ok = gate.AUC_ours >= gate.AUC_paper
    print(
        f"\nGate 3 ({DISPLAY['miRAW']} NF): AUC {gate.AUC_ours:.4f} vs the paper's {gate.AUC_paper:.4f} "
        f"(diff {gate.AUC_diff:+.4f}) -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="*", default=["build", "run", "report"])
    p.add_argument("--unified", type=Path, default=UNIFIED)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--configs", nargs="+", default=[f"{f}|{flt}" for f, flt in CONFIGS])
    args = p.parse_args()
    code = 0
    for step in args.steps:
        if step == "build":
            build(args)
        elif step == "run":
            run(args)
        elif step == "report":
            code = report(args)
        else:
            p.error(f"unknown step {step!r}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
