# pyMIRAW progress log

Read this first in a new session (see CLAUDE.md). Update it after every meaningful step.
Plan and decisions: [PLAN.md](PLAN.md).

## Status (2026-10-27)

| Phase | State |
|---|---|
| 0 Bootstrap (env, skeleton, S3, hooks, CI) | **done**, `conda-lock.yml` generated (linux-64); not done: Java reference env (`environment-reference.yml`) not created; only the original weights are mirrored to S3 (the 1 GB archive is not) |
| 1 Primitives (folding, encoding, MLP, DL4J import) | **done**. Gate 0 passed |
| 2 Candidate sites and filters | **done** for Default/Personalized/PITA/TargetScan + AccessibilityEnergy. SpecificZones/ReservedZones filters are not ported (unused by the paper configs) |
| 3 Prediction and evaluation | **done**; Gate 1 passed on set 0 (all 10 configs), sets 1-9 running |
| 4 Training stack | **done**; Gate 2 at statistical parity (old-default kept as orientation); not promoted |
| 5 Hardening and docs | not started (`docs/extending.md`, `docs/data.md`, lockfile, Java oracle comparison) |

## Gate results so far

* **Gate 0 (imported paper model vs paper CV):** PASSED, exactly. Fold 6 test accuracy
  0.9351428571 equals the paper's `BestModelEval.txt` value to all digits.
* **Gate 1 (pipeline vs paper, original weights, ViennaRNA 2.7.2 bindings):** partial. 64 pairs of
  balanced set 0 with `miRAW` NF: gene-level call identical for 64/64, max |score diff| 1.8e-4.
  Per-pair PosSites/NegSites equal for only 69 % of pairs (differences of a few sites, mostly
  NegSites); not yet explained, suspected ViennaRNA 2.2 vs 2.7 energy differences near dG=0 or
  edge windows. The site finder reproduced 99.4 % of recorded sites exactly in the early prototype.
  Full 10 configs x 10 sets run: see "Running jobs".
* **Gate 2 (retrained model):** statistical parity, not promoted. Site level: natural 0.9249 / 0.9766, old-default 0.9192 / 0.9752 vs
  paper models 0.9210 / 0.9765. Gene level (fold 6, NF configs, sets 0-1): old-default +0.003 F1, natural -0.002 F1 vs paper. Shuffled
  training stays below (0.88-0.90 / 0.954-0.964). AE configs and the other sets are still to run.
* **Orientation experiment (`old-default` vs `natural`):** DECIDED: keep `old-default`. `natural` is significantly better at site
  level (paper regime: AUC +0.0015, acc +0.0057, p <= 0.002) but not at gene level (F1 -0.006 vs old-default, p = 0.76).

## Running jobs (started 2026-10-27, on the dev machine; outputs are in the gitignored `runs/`)

* Gate 1: `pymiraw parity <balanced10 dir> --output-dir runs/parity --cache-dir ~/.cache/pymiraw/foldcache --sets 0-9 --workers 14`
  (log `runs/parity.log`; resumable: rerun the same command; results land in `runs/parity/<config>_<set>/`;
  final comparison in `runs/parity/comparison.csv`; exit code 0 only if every config is within 0.03).
* Gate 2 / orientation: `pymiraw train <folds dir> --orientation {old-default,natural} --folds 0-8 --seeds 3`
  (logs `runs/train_<orientation>.log`, results `runs/train_<orientation>/cv_<orientation>.csv`).

## Where things are

* Code: `src/pymiraw/` (folding, sites, encoding, models, training, prediction, evaluation, cli).
* Tests: `tests/unit`, `tests/functional` (88 pass; tests needing the original weights skip without S3).
* Fixtures (committed): `tests/fixtures/` (paper pairs + expected output, 500 CV test rows,
  `paper_global_summary.csv` per-set paper metrics, `paper_cv_xent_best.csv` paper CV scores).
* Heavy data: S3 `s3://digitalrnd-drf-projects-raw-ireland/spatial_omics/dev/pyMIRAW/`; currently only
  `raw/original/miraw_data/NetworkModel/bestModel.bin` (in `data/manifest.yaml`).
* Not on S3 yet and needed to rerun benchmarks/training on another machine: the paper archive
  `PLOSComb.7z` (https://bitbucket.org/bipous/miraw_data, raw/master/PLOSComb.7z, 1 GB). It holds the
  10 balanced test sets (`Data/TestData/balanced10`), CV folds (`Data/CrossVal/RelaxedPW60_14K_35K/binary`)
  and `Results/`. Now on S3 and in the manifest as `paper-archive` (`pymiraw data pull paper-archive`).
* Original Java code (for reading): https://bitbucket.org/bipous/miraw_-dl4mirnatargeting_code

## Findings worth remembering

* Published results use miRNA nt 2-10 for the `Default` ("6-1:10") CSSM, not 1-10 (prototype matched 99.4 % vs 65.5 %).
* No autoencoder pre-training; dropout 0.0; paper training = 10 epochs, no validation early stopping,
  "last" models collapsed to all-negative (so the best epoch by training loss was used).
* DL4J `XENT` on a 2-way softmax = 2 x NLL (doubles the effective learning rate); implemented as such.
* `pymiraw/models/__init__.py` must import `mlp`, otherwise the model registry is empty.
* The bundled `TestData/ExperimentResults` of the original repo come from an old dev run and are NOT a valid oracle.
* `pip install` into the base conda env can break `snowflake-connector-python` (needs `filelock<4`); use the `pymiraw` env.

## Next steps

1. Read the Gate 1 comparison when `runs/parity/comparison.csv` exists; investigate site-count differences if any config is outside +-0.03.
2. Read the CV results; if accuracy < 0.921, tune (epochs/patience/lr/batch) before concluding Gate 2; then decide the orientation experiment per PLAN.md decision 8.
3. Generate `conda-lock.yml`; `environment-reference.yml` and the Java oracle comparison.
4. Commit (nothing has been committed yet by the assistant; review `git status`), push, check GitHub push protection is on.

## Log (newest first)

* 2026-10-27 (23:00): **Gate 2, gene level (retrained fold-6 models, 3 seeds each, balanced sets 0-1, 5 unfiltered configs,
  same pipeline/folding as the paper numbers; rows in `docs/results/gene_level_fold6_retrained.csv`, script
  `scripts/gene_level_compare.py`).** Mean difference vs the paper's GlobalSummary: **old-default dAcc +0.007, dF1 +0.003**
  (F1 >= paper in 6/10 config-set cells; one-sided p(worse) 0.90; best on `miRAW` NF: +0.013 F1); **natural dAcc -0.003, dF1 -0.002**
  (3/10 cells; p(worse) 0.12). natural - old-default gene-level F1: -0.0056, better in 16/30, one-sided p = 0.76.
  **Verdict per PLAN decision 8: `old-default` stays the default orientation for new models** (natural's site-level
  advantage does not carry to gene level). Neither is significantly worse than the paper at gene level.
  **Caveats:** only fold 6, unfiltered (NF) configs and sets 0-1; AE configs and sets 2-9 not yet run for retrained models;
  site-level, `old-default` is nominally below the paper's means (acc 0.9192 vs 0.9210, AUC 0.9752 vs 0.9765; not significant),
  so the strict "mean >= paper" wording of Gate 2 is not literally met by either orientation. Retrained models are therefore
  NOT promoted: `--original-weights` stays the default. Retrained fold-6 artefacts are in `runs/train_paperRegime_*/` (not in git).

* 2026-10-27 (21:40): **Gate 2, site level, paper regime** (`--no-shuffle --val-fraction 0 --max-epochs 10`, folds 0-8 x 3
  seeds, XENT; raw rows in `docs/results/cv_paper_regime.csv`). Baseline = paper fold models re-scored (acc 0.9210 / AUC 0.9765):
  old-default acc 0.9192 AUC 0.9752 (-0.002/-0.0014 vs paper, one-sided p(worse) 0.33/0.08: not significantly worse);
  **natural acc 0.9249 AUC 0.9766 (+0.004/+0.0001, >= paper accuracy in 8/9 folds)**. No run collapsed.
  **natural vs old-default** (27 paired runs): AUC +0.0015, acc +0.0057, F1 +0.0062; one-sided Wilcoxon p = 0.002 / 0.001 / 0.001;
  better in 19-20/27. So the *site-level* part of decision 8 is met by `natural` (significant AND not below the paper).
  **Still open before `natural` becomes the default for new models:** the gene-level part (mean F1 over the 10 configs x
  balanced sets, paired, through the same pipeline) with retrained models, and the Gate 1 benchmark itself (sets 1-9).
  The original weights remain the default until then. Note: AUC margin over the paper is tiny (+0.0001); treat as parity, not improvement.

* 2026-10-27 (21:00): **Correction + root cause for Gate 2.** (1) The paper's recorded CV AUC (0.958) used coarse 0.05
  thresholds. Re-scoring the paper's 9 fold models with our metric code gives acc 0.9210 / AUC 0.9765
  (`tests/fixtures/paper_cv_xent_models_rescored.csv`); that is the Gate 2 baseline now (PLAN.md updated). All earlier
  "AUC at parity" statements below are superseded. (2) Retrained, shuffled, all data, 20 epochs: acc 0.903 / AUC 0.964,
  below the baseline in all 9 folds. (3) Cause: the released train files alternate labels 1,0,1,0... and DL4J read them
  in file order without shuffling. Unshuffled training (10 epochs, all data) on folds 6 and 0, 2 seeds: acc 0.91-0.935,
  AUC 0.973-0.980 (paper models 0.935/0.981 and 0.902/0.977); shuffled: ~0.90/0.965. (4) Added `--no-shuffle`
  (and fixed a bug where the CLI did not pass it on; regression test added). Running now: full CV in the paper regime
  (`--no-shuffle --val-fraction 0 --max-epochs 10`, 9 folds x 3 seeds, both orientations) ->
  `runs/train_paperRegime_{old-default,natural}/cv_*.csv`. Gate 2 and the orientation verdict are decided from these.

* 2026-10-27 (20:00): **Gate 1, set 0 complete (all 10 configs):** every config within 0.005 accuracy and 0.005 F1
  of the paper (tolerance 0.03); `miRAW-6-1:10 AE` identical to the paper (440/209/339/108). Other configs differ by
  1-5 pairs per cell (e.g. 7-1:10 AE 385/150/398/163 vs 380/149/399/168). Confirms: accessibility filter = remove
  if dG_open > +10, all four finders, original weights. Sets 1-9 still running (`runs/parity/`, warm cache).

* 2026-10-27 (latest): **Gate 1, first full run:** `miRAW` NF on balanced set 0, original weights, ViennaRNA 2.7.2
  bindings: TP/FP/TN/FN = 511/350/198/37 vs paper 510/349/199/38 (accuracy 0.64690 vs 0.64690, F1 0.7253 vs 0.7249).
  Within one pair per cell. (Run took ~2 h while sharing the machine with training; fold cache now warm.)
  **Retraining, site-level CV, folds 0-8 x 3 seeds, val_fraction 0.1, 10 epochs, checkpoint on val loss:**
  old-default acc 0.884 / AUC 0.954; natural acc 0.888 / AUC 0.956; paper acc 0.921 / AUC 0.958; none collapsed.
  natural vs old-default: mean AUC +0.0022 (p=0.20), acc +0.0040 (p=0.14), better in 16/27 -> not significant, so
  per decision 8 `old-default` stays. **Gate 2 NOT met** (accuracy 0.884 < 0.921; AUC 0.954 < 0.958).
  Diagnostic (fold 0 and 6, 2 seeds, all train data, per-epoch test AUC): AUC reaches 0.962-0.971 at 10-30 epochs
  (paper 0.949 / 0.968), accuracy stays ~0.89-0.91 at threshold score>0 (paper 0.902 / 0.935). So ranking quality
  is at parity; the accuracy gap looks like holdout + short training + threshold effects. Added `--val-fraction 0`
  (train on all data, keep lowest-training-loss epoch = paper regime). Running: 20 epochs, 9 folds x 3 seeds ->
  `runs/train_allData_e20/cv_old-default.csv`.

* 2026-10-27 (later): README is now marked work-in-progress and path-based (no S3 mention). Added
  `pymiraw data unpack-archive PLOSComb.7z [--dest]` (default `data/original/`, gitignored) and made
  `data/original/NetworkModel/bestModel.bin` the first place the original weights are looked up
  (order: `$PYMIRAW_ORIGINAL_WEIGHTS`, local path, S3 mirror). Verified on the real archive (10 test
  sets, 20 fold files, weights byte-identical). Generated `conda-lock.yml`. 86 unit tests pass.
  GPU: training already uses the T4 (Lightning `auto`); the parity benchmark is bound by ViennaRNA
  folding on CPU and cannot use the GPU. Machine load while both jobs run was only ~8/16 cores.

* 2026-10-27: Implemented phases 0-4 (see Status). 88 tests pass. Original-weights flag
  (`--original-weights`, default) + S3 fetch with sha256 check. Secret guards in place.
