"""Re-run the acceptance gates from scratch and compare every number with the committed results (docs/results/).

Run this after any change that could touch results (refactors, dependency updates). Steps (default: all):

  gate0  bundled original model on CV fold 6 = the paper's 0.9351428571; the 9 paper fold models re-scored =
         pymiraw/resources/paper_cv_xent_models_rescored.csv.
  gate1  `pymiraw parity` with the original weights, 10 sets x 10 configs, into a fresh sites folder (site
         finders and accessibility filter recomputed; ViennaRNA window folds come from the fold cache):
         per-run TP/FP/TN/FN must equal docs/results/gate1_all_sets.csv, and the gate must pass.
  gene   the gene-level Gate 2 runs with the committed model files (retrained fold models and the paper's
         fold models), same sets/configs as the committed results: TP/FP/TN/FN must be identical.
  site   retrain the paper regime (folds 0-8 x seeds 0-2, both orientations) and compare with
         docs/results/cv_paper_regime.csv. Training on a different torch build or device need not be
         bit-identical; differences are reported, then `gene-new` re-runs the gene level with the new models.
  gene-new  the gene-level runs with the models retrained by `site` (only needed if they differ).
  report Gate 2 report (scripts/gate2_report.py) on the re-run results.

Usage: python scripts/verify_gates.py [steps...] [--out runs/verify] [--workers 15]
Needs data/original (pymiraw data unpack-archive PLOSComb.7z) and the committed models in runs/train_paperRegime_*.
Exit code 0 only if every comparison and gate passes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pymiraw import resources
from pymiraw.config import TrainConfig
from pymiraw.evaluation.metrics import roc_auc
from pymiraw.evaluation.parity import CONFIGS, compare_with_paper, run_parity
from pymiraw.models.artifact import load_artifact
from pymiraw.storage import ORIGINAL_DIR, original_weights_path
from pymiraw.training.cv import run_cv
from pymiraw.training.data import load_csv

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs/results"
TEST_DIR = ORIGINAL_DIR / "Data/TestData/balanced10"
FOLDS_DIR = ORIGINAL_DIR / "Data/CrossVal/RelaxedPW60_14K_35K/binary"
PAPER_FOLDS = ORIGINAL_DIR / "ResultsCV/VeryDeepXENT14K_e10"  # Set<f>/bestModel.bin: the paper's CV fold models
FOLD_CACHE = Path.home() / ".cache/pymiraw/foldcache"
COUNTS = ["TP", "FP", "TN", "FN"]
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


def gene_runs(models_dir: Path) -> list[tuple[str, Path, range, str | None]]:
    """(output name, model file, sets, config filter) of every committed gene-level result."""
    runs = [(f"gene_old-default_f{f}_s0", models_dir / f"old-default_f{f}_s0.pt", range(10), None) for f in range(9)]
    runs += [(f"gene_old-default_f6_s{s}", models_dir / f"old-default_f6_s{s}.pt", range(10), "NF") for s in (1, 2)]
    nat = Path(str(models_dir).replace("old-default", "natural"))
    runs += [(f"gene_natural_f6_s{s}", nat / f"natural_f6_s{s}.pt", range(2), "NF") for s in range(3)]
    return runs


def parity(model: Path | str, out: Path, sets, config_filter, args) -> pd.DataFrame:
    cfgs = [c for c in CONFIGS if not config_filter or config_filter in f"{c[0]}|{c[1]}"]
    sites = args.out / "sites"
    return run_parity(TEST_DIR, model, out, sets, cfgs, "bindings", FOLD_CACHE, args.workers, sites_dir=sites)


def same_counts(new: pd.DataFrame, old: pd.DataFrame, keys: list[str], what: str) -> None:
    m = old.merge(new, on=keys, how="left", suffixes=("_old", "_new"), indicator=True)
    missing = int((m._merge != "both").sum())
    differ = int(sum((m[f"{c}_old"] != m[f"{c}_new"]).sum() for c in COUNTS)) if not missing else -1
    summary = f"{what}: {len(old)} runs, {missing} missing, {differ} count differences"
    check(missing == 0 and differ == 0 and len(new) == len(old), summary)


def gate0(args) -> None:
    print("Gate 0")
    model, _, _ = load_artifact(original_weights_path())
    x, y = load_csv(FOLDS_DIR / "set_6_test.csv")
    with torch.no_grad():
        acc = float(((model.site_score(torch.from_numpy(x)).numpy() > 0) == y).mean())
    check(abs(acc - 0.9351428571) < 1e-10, f"bundled model, CV fold 6 accuracy {acc:.10f} (paper 0.9351428571)")
    ref = resources.paper_cv_models().set_index("fold")
    worst = 0.0
    for f in range(9):
        net, _, _ = load_artifact(PAPER_FOLDS / f"Set{f}/bestModel.bin")
        x, y = load_csv(FOLDS_DIR / f"set_{f}_test.csv")
        with torch.no_grad():
            s = net.site_score(torch.from_numpy(x)).numpy()
        worst = max(worst, abs(((s > 0) == y).mean() - ref.Accuracy[f]), abs(roc_auc(y, s) - ref.AUC[f]))
    check(worst < 1e-9, f"paper fold models 0-8 re-scored = the committed baseline (max |diff| {worst:.1e})")


def gate1(args) -> None:
    print("Gate 1 (fresh sites folder; takes a while)")
    ours = parity("original", args.out / "gate1", range(10), None, args)
    same_counts(ours, pd.read_csv(COMMITTED / "gate1_all_sets.csv"), ["config", "set"], "Gate 1 per-run counts")
    cmp = compare_with_paper(ours)
    cmp.to_csv(args.out / "gate1_comparison.csv")
    worst = cmp[["Accuracy_diff", "F1_diff"]].abs().max().max()
    check(bool(cmp.ok.all()), f"Gate 1 verdict: all 10 configs within 0.02 (largest |diff| {worst:.4f})")


def gene(args, models_dir: Path, results: Path, prefix: str = "") -> None:
    """Gene-level runs of ``models_dir``'s fold models plus the paper's fold models (run once, in ``args.out``)."""
    base = args.out / prefix if prefix else args.out
    for name, model, sets, flt in gene_runs(models_dir):
        print(f"  {name} ({model.name})")
        parity(model, base / name, sets, flt, args)
    for f in range(9):
        name = f"gene_paper_f{f}_s0"
        if f != 6:  # the paper's fold-6 model is the original weights: its runs are Gate 1's
            print(f"  {name}")
            parity(PAPER_FOLDS / f"Set{f}/bestModel.bin", args.out / name, range(10), None, args)
        source = args.out / "gate1" if f == 6 else args.out / name
        if (base / name) != source and not (base / name).exists():
            (base / name).symlink_to(source.resolve())
    results.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, ROOT / "scripts/gene_level_compare.py", base, results / "gene_level_retrained.csv"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    keys = ["orientation", "fold", "seed", "config", "set"]
    files = ["gene_level_paper_folds.csv"] + ([] if prefix else ["gene_level_retrained.csv"])
    for f in files:  # with models retrained by `site`, our rows are new results: nothing to compare them with
        same_counts(pd.read_csv(results / f), pd.read_csv(COMMITTED / f), keys, f"{prefix or 'committed models'}: {f}")


def site(args, results: Path) -> None:
    print("Site level: retraining (paper regime)")
    frames = []
    for o in ("old-default", "natural"):
        df = run_cv(FOLDS_DIR, TrainConfig.paper(site_orientation=o), range(9), range(3), args.out / f"train_{o}")
        frames.append(df.assign(orientation=o))
    new = pd.concat(frames)[pd.read_csv(COMMITTED / "cv_paper_regime.csv").columns]
    num = new.select_dtypes("number").columns.drop(["fold", "seed"])
    new[num] = new[num].round(5)
    results.mkdir(parents=True, exist_ok=True)
    new.to_csv(results / "cv_paper_regime.csv", index=False)
    old = pd.read_csv(COMMITTED / "cv_paper_regime.csv")
    m = old.merge(new, on=["orientation", "fold", "seed"], suffixes=("_old", "_new"))
    same = int((m[[f"{c}_old" for c in COUNTS]].to_numpy() == m[[f"{c}_new" for c in COUNTS]].to_numpy()).all(1).sum())
    d_acc = float(np.abs(m.Accuracy_old - m.Accuracy_new).max())
    print(f"  {same}/{len(m)} retrained runs have identical confusion counts; max |dAccuracy| {d_acc:.5f}")


def retrained_identical(results: Path) -> bool:
    """Did retraining (step ``site``) reproduce the committed site-level confusion counts exactly?"""
    new, old = pd.read_csv(results / "cv_paper_regime.csv"), pd.read_csv(COMMITTED / "cv_paper_regime.csv")
    m = old.merge(new, on=["orientation", "fold", "seed"], suffixes=("_old", "_new"))
    return len(m) == len(old) and all((m[f"{c}_old"] == m[f"{c}_new"]).all() for c in COUNTS)


def report(results: Path) -> None:
    print(f"Gate 2 report on {results}")
    for f in ("cv_paper_regime.csv", "gene_level_retrained.csv", "gene_level_paper_folds.csv"):
        if not (results / f).exists():
            shutil.copy(COMMITTED / f, results / f)
            print(f"  ({f}: committed file, not re-run)")
    r = subprocess.run([sys.executable, ROOT / "scripts/gate2_report.py", results], capture_output=True, text=True)
    (results / "gate2_report.txt").write_text(r.stdout)
    print("  " + "\n  ".join(r.stdout.strip().splitlines()[-1:]))
    check(r.returncode == 0, f"Gate 2 verdict ({results.name})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="*", default=["gate0", "gate1", "gene", "site", "gene-new", "report"])
    p.add_argument("--out", type=Path, default=ROOT / "runs/verify")
    p.add_argument("--workers", type=int, default=15)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    committed_models = ROOT / "runs/train_paperRegime_old-default"
    for step in args.steps:
        if step == "gate0":
            gate0(args)
        elif step == "gate1":
            gate1(args)
        elif step == "gene":
            print("Gene level, committed models")
            gene(args, committed_models, args.out / "results")
        elif step == "site":
            site(args, args.out / "results_retrained")
        elif step == "gene-new":
            if retrained_identical(args.out / "results_retrained"):
                print("Gene level with retrained models: skipped (retraining reproduced the committed models' counts)")
                continue
            print("Gene level, models retrained by `site`")
            gene(args, args.out / "train_old-default", args.out / "results_retrained", prefix="retrained")
        elif step == "report":
            report(args.out / "results")
            if (args.out / "results_retrained/cv_paper_regime.csv").exists():
                report(args.out / "results_retrained")
        else:
            p.error(f"unknown step {step!r}")
    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} FAILED: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
