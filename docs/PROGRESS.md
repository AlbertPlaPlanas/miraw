# pyMIRAW progress log

Read this first in a new session (see CLAUDE.md). Update it after every meaningful step.
Plan and decisions: [PLAN.md](PLAN.md).

## Status (2026-10-27)

| Phase | State |
|---|---|
| 0 Bootstrap (env, skeleton, S3, hooks, CI) | **done**, except: lockfile (`conda-lock.yml`) not generated; Java reference env (`environment-reference.yml`) not created; only the original weights are mirrored to S3 (the 1 GB archive is not) |
| 1 Primitives (folding, encoding, MLP, DL4J import) | **done**. Gate 0 passed |
| 2 Candidate sites and filters | **done** for Default/Personalized/PITA/TargetScan + AccessibilityEnergy. SpecificZones/ReservedZones filters are not ported (unused by the paper configs) |
| 3 Prediction and evaluation | **done**; Gate 1 benchmark **running** (see below) |
| 4 Training stack | **done**; Gate 2 and the orientation experiment **running / not concluded** |
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
* **Gate 2 (retrained model):** single smoke run, fold 6, seed 0, XENT, 10 epochs, checkpoint on
  validation loss: accuracy 0.891, AUC 0.9595. Paper mean over folds 0-8: accuracy 0.921, AUC 0.958.
  Accuracy is below the paper, so tuning may be needed. The 3-seed, 9-fold runs are in progress.
* **Orientation experiment (`old-default` vs `natural`):** not concluded (runs in progress).

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

* 2026-10-27: Implemented phases 0-4 (see Status). 88 tests pass. Original-weights flag
  (`--original-weights`, default) + S3 fetch with sha256 check. Secret guards in place.
