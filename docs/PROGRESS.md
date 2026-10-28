# pyMIRAW progress log

Read this first in a new session (see CLAUDE.md). Update it after every meaningful step.
Plan and decisions: [PLAN.md](PLAN.md).

## Status (2026-10-28)

| Phase | State |
|---|---|
| 0 Bootstrap (env, skeleton, S3, hooks, CI) | **done**, `conda-lock.yml` generated (linux-64); not done: Java reference env (`environment-reference.yml`) not created; only the original weights are mirrored to S3 (the 1 GB archive is not) |
| 1 Primitives (folding, encoding, MLP, DL4J import) | **done**. Gate 0 passed |
| 2 Candidate sites and filters | **done** for Default/Personalized/PITA/TargetScan + AccessibilityEnergy. SpecificZones/ReservedZones filters are not ported (unused by the paper configs) |
| 3 Prediction and evaluation | **done; Gate 1 PASSED** (100/100 runs, all means within 0.005) |
| 4 Training stack | **done**; Gate 2 non-inferiority PASS for old-default on site level (folds 0-8) and gene level (fold 6, all 10 sets, 10 configs); not promoted |
| 5 Hardening and docs | `docs/extending.md`, `docs/data.md`, lockfile, README done; not done: Java `miRAW.jar` oracle run, profiling, v1.0.0 tag, Gate 2 with other folds' models and AE configs for seeds 1-2 |

## Gate results so far

* **Gate 0 (imported paper model vs paper CV):** PASSED, exactly. Fold 6 test accuracy
  0.9351428571 equals the paper's `BestModelEval.txt` value to all digits.
* **Gate 1 (pipeline vs paper, original weights, ViennaRNA 2.7.2 bindings):** partial. 64 pairs of
  balanced set 0 with `miRAW` NF: gene-level call identical for 64/64, max |score diff| 1.8e-4.
  Per-pair PosSites/NegSites equal for only 69 % of pairs (differences of a few sites, mostly
  NegSites); not yet explained, suspected ViennaRNA 2.2 vs 2.7 energy differences near dG=0 or
  edge windows. The site finder reproduced 99.4 % of recorded sites exactly in the early prototype.
  Full 10 configs x 10 sets run: see "Running jobs".
* **Gate 2 (retrained model):** PASSES the agreed non-inferiority rule on the partial evidence (see Log 23:30), not promoted. Site level: natural 0.9249 / 0.9766, old-default 0.9192 / 0.9752 vs
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

* 2026-10-28 (14:30): **Licence is MIT** (owner decision: they are the author of the original miRAW and relicense it). `LICENSE` was already MIT;
  fixed the remaining GPL-3.0 mentions in `pyproject.toml`, `README.md`, `docs/data.md`, `data/manifest.yaml`. The upstream Bitbucket repos still say GPL until the
  owner changes them there. Third-party data (e.g. TarBase) keeps its own terms.

* 2026-10-28 (11:40): Fixed `python -m pymiraw.cli`: the `__main__` guard sat above the `parity`/`train` commands, so they were
  not registered via `-m` (the installed `pymiraw` entry point was fine). **Running (Gate 2, other folds):** `runs/gene_folds.sh`
  = `pymiraw parity ... --model-path --model runs/train_paperRegime_old-default/old-default_f{f}_s0.pt --sets 0-9 --sites-dir runs/sites`
  for f in 0-5,7,8 (seed 0, all 10 configs); progress in `runs/gene_folds.progress`, logs `runs/gene_folds_f*.log`, results in
  `runs/gene_old-default_f{f}_s0/`. Sites are folded once (fold 0) and reused by the later folds. Next: add a fold column to
  `scripts/gene_level_compare.py` / `gate2_report.py`, rerun the report, update README TL;DR.

* 2026-10-28 (11:30): **Saved candidate sites (`--sites-dir`).** New `src/pymiraw/sites/store.py` (`SiteStore`): the sites found with
  ViennaRNA are written per pair as parquet shards (`sites-<id>.parquet`, append-only, atomic) and loaded on later runs, so only the
  network/filters run again. Key = sha1(backend name+version, finder class+all settings, miRNA, 3'UTR); pairs without sites are stored
  as marker rows (distinct from "never computed"). `predict_file` looks every pair up in the main process, passes saved sites to the
  workers, computes only the misses and saves them afterwards. Options: `--sites-dir DIR` on `predict`/`evaluate`/`parity`
  (`sites_dir:` in YAML), `--require-sites` (LookupError instead of folding). Scores, filter results and filter energies are not
  stored (the accessibility filter still folds its 200 nt window). Tests: `tests/unit/test_site_store.py` (round trip, empty pairs,
  copies, key sensitivity, second run does 0 folding calls); 105 unit tests pass in the scratch venv (the 1 failure there is
  `py7zr` missing in it); ruff clean. **Verified with real ViennaRNA 2.7.2** (pip `ViennaRNA` in the scratch venv) and the original weights on
  `tests/fixtures/paper_pairs.tsv` (6 pairs): `pymiraw evaluate ... --original-weights --sites-dir S` run 1 folds and writes one shard,
  run 2 loads it (4.7 s -> 2.5 s, mostly start-up), and `targetPredictionOutput.csv` and `targetSites.csv` are byte-identical to a run without
  `--sites-dir`; `--require-sites` with another finder raises. Timing, balanced set 0, first 100 pairs, `miRAW` NF, 8 workers: cold store 103.6 s, warm store 2.9 s, outputs identical. README now has a "Status" section (done / remaining) to keep current (rule added to CLAUDE.md). Docs: `docs/data.md`.

* 2026-10-28 (10:20): **Gate 2 gene level, `old-default` retrained fold-6 models on all 10 balanced sets.** Finished runs
  (`runs/gene_full.sh`): seed 0 on all 10 configs (NF and AE), seeds 1-2 on the 5 NF configs, 100 (config, set) cells.
  Mean over cells (ours - paper): **accuracy +0.0059, F1 +0.0047**; F1 >= paper in 70/100 cells; mean per config within
  -0.002..+0.023 F1 (largest: `miRAW` NF +0.023 accuracy / +0.013 F1). Non-inferiority (`scripts/gate2_report.py`, rows rewritten
  by `scripts/gene_level_compare.py runs docs/results/gene_level_fold6_retrained.csv`, now 230 runs): **OVERALL PASS**, all 8 rows;
  old-default gene accuracy lower bound +0.0036, F1 +0.0020 (slack 0.012-0.014). Fragile passes unchanged: old-default site
  accuracy (slack 0.0001) and AUC (0.0005). `natural` remains at sets 0-1 NF only (orientation decision unchanged).
  Still not promoted: original weights stay the default. This machine has no `pymiraw` conda env (analysis ran in a scratch venv with
  pandas/scipy, `PYTHONPATH=src`); recreate with `conda env create -f environment.yml` before running anything heavy.

* 2026-10-28 (02:30): **GATE 1 PASSED: all 100 runs (10 configurations x 10 balanced test sets), original weights, ViennaRNA 2.7.2
  bindings, vs the paper's GlobalSummary** (`docs/results/gate1_comparison.csv`, per-run rows `docs/results/gate1_all_sets.csv`,
  reproduce with `pymiraw parity <balanced10 dir>`). Mean differences (ours - paper), accuracy / F1: miRAW-6-1:10 NF -0.000/-0.000,
  AE +0.000/+0.001; 7-1:10 NF -0.001/-0.000, AE -0.001/-0.001; 7-2:10 NF -0.001/-0.001, AE -0.001/-0.002; PITA NF +0.001/+0.001,
  AE -0.001/-0.002; TS NF +0.001/+0.001, AE -0.003/-0.005. **Largest mean difference 0.003 accuracy / 0.005 F1 (tolerance 0.03).**
  Four single cells differ by > 0.01 (TS AE set 3 -0.031; Personalized;1;9;7 AE set 2 -0.018; PITA AE set 6 -0.016; Personalized;0;9;7
  AE set 2 -0.011); all four have unreliable references: the original per-run files are truncated (PITA AE set 6: 459 of 1096 pairs,
  positives only; Personalized AE set 2: 103 and 147 rows; TS AE set 3: 814 rows, and its TP/FN equal ours exactly) or the summary row
  has a non-integer count (397.6). With ViennaRNA 2.2.5 the port is bit-exact (see 01:00 entry). Next: Gate 2 gene level on all
  sets/configs for the retrained models (running).

* 2026-10-27 (resume): Gate 1 verified alive at 23:54: parent pid 12086 up 5 h, 14 workers all running at ~42 % CPU, fold cache still being written; sets 0-6 complete (70 of 100 runs, set 6 finished 23:46, ~17 min per set), set 7 folding, so ETA is about 1 h; `runs/parity.log` is empty (0 bytes), so watch `ls runs/parity | wc -l` instead; no `comparison.csv` yet. Wrote `docs/data.md`. Next: when `comparison.csv` appears, copy it to `docs/results/`, record the full 10-set result here.

* 2026-10-28 (02:00): **Gate 1 interim, test sets 0-5 complete (60 of 100 runs; sets 6-9 still running).** Mean over the 6 sets,
  every config within **0.0046 accuracy / 0.0080 F1** of the paper's GlobalSummary (tolerance 0.03): all 10 configs PASS
  (`docs/results/gate1_interim_comparison.csv`; per-run rows `docs/results/gate1_interim_sets0-5.csv`). Largest mean
  differences are on `miRAW-TS AE` (-0.0046 / -0.0080), the rest below 0.003. **Three single (config, set) cells exceed 0.01**
  (TS AE set 3: -0.031 acc / -0.061 F1; Personalized;1;9;7 AE set 2: -0.018; Personalized;0;9;7 AE set 2: -0.011). Checked against the
  original per-run output files (`Results/Evaluate_14K_XENT/<config>_<set>/targetPredictionOutput.csv`): for TS AE set 3 the original run
  gives TP=193, FN=355, **identical to pyMIRAW** (the summary spreadsheet row, TP 226 / FN 322, disagrees with its own run file, and that
  run is incomplete: 814 of 1096 pairs); the two Personalized AE set-2 run files are truncated (103 and 147 rows) and their summary row has
  a non-integer TP (397.6), so they cannot be checked. Conclusion: the outliers come from the reference data, not from the port.

* 2026-10-28 (01:00): **The port is bit-exact with the original ViennaRNA version.** Created `environment-reference.yml`
  (`pymiraw-ref`: ViennaRNA 2.2.5 + OpenJDK 8). First 64 pairs of balanced set 0, `miRAW` NF, original weights
  (`scripts/viennarna_version_check.py`): with the 2.2.5 CLI backend **100 % of pairs have identical PosSites/NegSites and
  identical scores (max |diff| 0.00000)** vs the paper's recorded output; with the 2.7.2 bindings 68.8 % identical counts
  (76.6 % identical PosSites), gene call identical in 100 %, max |score diff| 0.00018. So the residual differences in the
  benchmark come only from ViennaRNA 2.2 -> 2.7 energy changes, not from porting errors. New test
  `test_exact_match_with_the_original_viennarna_version` (marker `reference`, skips without `pymiraw-ref`). 110 tests pass; lint clean.
  Not done: running the original `miRAW.jar` itself as an oracle (the paper's recorded outputs serve as the oracle instead).

* 2026-10-28 (00:10): Added training smoke tests (learning, reproducibility, checkpoint round-trip, collapse guard, and a
  test-only conv model trained through the same stack = the extension seam), `docs/extending.md`, README status table.
  Tests: 109 pass with the original weights, 105 pass + 4 skip without (CI scenario; weight tests now skip immediately instead of
  waiting for S3 credential retries); ruff check and format clean. Pytest warnings from third-party deprecations are filtered
  in `pyproject.toml`. Gate 1 benchmark still running (see `ls runs/parity | wc -l`, 100 = done).

* 2026-10-27 (23:30): **Gate 2 rule changed to non-inferiority at 99 % one-sided confidence** (owner decision; margins
  0.01 accuracy/F1, 0.005 AUC are my proposal, adjustable constants in `pymiraw/evaluation/noninferiority.py`).
  `scripts/gate2_report.py` on the committed results (folds 0-8 site level; fold-6 models, 5 NF configs, sets 0-1 gene level):
  **PASS for both orientations on all 8 rows.** Fragile passes (slack < 0.001): old-default site accuracy (lower bound
  -0.0099 vs limit -0.0100) and old-default site AUC (-0.0045 vs -0.0050, slack 0.0005). natural gene accuracy slack 0.0011.
  Passing on this partial evidence does not promote the model; remaining evidence to collect for the final Gate 2:
  filtered (AE) configs and all 10 sets at gene level, other folds' models. The original weights stay the default.

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
