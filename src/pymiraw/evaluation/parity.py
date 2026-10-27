"""Gate 1: run the pipeline on the paper's 10 balanced test sets x 10 configurations and compare
with the published GlobalSummary (per set, then mean). Always uses the original weights by default."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pymiraw.evaluation.metrics import gene_level_metrics
from pymiraw.prediction.pipeline import predict_file, results_frame

PAPER_SUMMARY = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "paper_global_summary.csv"
CONFIGS = [
    (f, flt) for f in ("miRAW", "Personalized;0;9;7", "Personalized;1;9;7", "PITA", "TS") for flt in ("NF", "AE")
]
FILTERS = {"NF": [], "AE": ["AccessibilityEnergy;10"]}
DISPLAY = {
    "miRAW": "miRAW-6-1:10",
    "Personalized;0;9;7": "miRAW-7-1:10",
    "Personalized;1;9;7": "miRAW-7-2:10",
    "PITA": "miRAW-PITA",
    "TS": "miRAW-TS",
}


def run_parity(
    test_dir: str | Path,
    model: str,
    out_dir: str | Path,
    sets=range(10),
    configs=CONFIGS,
    backend="bindings",
    cache_dir=None,
    workers=None,
    max_site_length=40,
    seed_alignment_offset=5,
):
    """Returns a DataFrame with our metrics per (config, set). Resumable: finished runs are reused."""
    out_dir = Path(out_dir)
    rows = []
    for k in sets:
        for finder, flt in configs:
            tag = f"{finder.replace(';', '_')}_{flt}_{k}"
            f = out_dir / tag / "metrics.json"
            if not f.exists():
                cfg = dict(
                    model=model,
                    finder=finder,
                    filters=FILTERS[flt],
                    backend=backend,
                    cache_dir=cache_dir,
                    max_site_length=max_site_length,
                    seed_alignment_offset=seed_alignment_offset,
                )
                res = results_frame(predict_file(Path(test_dir) / f"randomLeveragedTestSplit_{k}.csv", cfg, workers))
                f.parent.mkdir(parents=True, exist_ok=True)
                res.to_csv(f.parent / "targetPredictionOutput.csv", sep="\t", index=False)
                f.write_text(json.dumps(gene_level_metrics(res.RealClass.astype(int), res.Prediction)))
            rows.append({"config": f"{finder}|{flt}", "set": k, **json.loads(f.read_text())})
    return pd.DataFrame(rows)


def compare_with_paper(ours: pd.DataFrame, tolerance: float = 0.03) -> pd.DataFrame:
    """Mean Accuracy/F1 per config vs the paper; ``ok`` if both are within ``tolerance``."""
    paper = pd.read_csv(PAPER_SUMMARY)
    paper = paper[paper.set.isin(ours.set.unique())]
    cols = ["Accuracy", "F1"]
    a = ours.groupby("config")[cols].mean().add_suffix("_ours")
    b = paper.groupby("config")[cols].mean().add_suffix("_paper")
    m = a.join(b)
    for c in cols:
        m[f"{c}_diff"] = m[f"{c}_ours"] - m[f"{c}_paper"]
    m["ok"] = (m["Accuracy_diff"].abs() <= tolerance) & (m["F1_diff"].abs() <= tolerance)
    return m
