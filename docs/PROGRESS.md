# pyMIRAW progress log

Read this first in a new session (see CLAUDE.md). Update it after every meaningful step.
Plan and decisions: [PLAN.md](PLAN.md).

## Status (2026-10-29)

| Phase | State |
|---|---|
| 0 Bootstrap (env, skeleton, S3, hooks, CI) | **done**, `conda-lock.yml` generated (linux-64); not done: Java reference env (`environment-reference.yml`) not created; only the original weights are mirrored to S3 (the 1 GB archive is not) |
| 1 Primitives (folding, encoding, MLP, DL4J import) | **done**. Gate 0 passed |
| 2 Candidate sites and filters | **done** for Default/Personalized/PITA/TargetScan + AccessibilityEnergy. SpecificZones/ReservedZones filters are not ported (unused by the paper configs) |
| 3 Prediction and evaluation | **done; Gate 1 PASSED** (100/100 runs, all config means within 0.005; tolerance now 0.02); **Gate 3 PASSED** (full unbalanced set, AUC 0.8225 vs 0.8228, tolerance 0.005) |
| 4 Training stack | **done; Gate 2 PASSED** (rank-based bound, margins acc/F1 0.02, AUC 0.005; set post hoc, see Log 22:00); fragile rows: paired gene accuracy (slack 0.0005), site AUC (0.0009). Not promoted (owner decision needed) |
| 5 Hardening and docs | `docs/extending.md`, `docs/data.md`, lockfile, README, high-level API + bundled weights + tutorial notebook (branch `restructure-api`, all gates re-verified identical) done; not done: Java `miRAW.jar` oracle run, profiling, v1.0.0 tag, AE configs for seeds 1-2 of fold 6 |

## Gate results so far

* **Gate 0 (imported paper model vs paper CV):** PASSED, exactly. Fold 6 test accuracy
  0.9351428571 equals the paper's `BestModelEval.txt` value to all digits.
* **Gate 1 (pipeline vs paper, original weights, ViennaRNA 2.7.2 bindings):** PASSED, 100/100 runs, all config means within
  0.003 accuracy / 0.005 F1 of the GlobalSummary (Log 02:30); bit-exact with ViennaRNA 2.2.5 (Log 01:00).
* **Gate 2 (retrained model, `old-default`):** **PASSED** under the rule as of 2026-10-28 22:00 (rank-based bound, 99 %, margins
  accuracy/F1 0.02, AUC 0.005; both changes post hoc, owner decisions). Lower bounds: site acc -0.0109, site AUC -0.0041 (fragile, slack
  0.0009); gene per cell acc -0.0013 / F1 -0.0035; gene paired per fold acc -0.0195 (**fragile, slack 0.0005**) / F1 -0.0044. Under the
  original margins (0.01) it fails on site acc and paired gene acc. Not promoted; the original weights stay the default.
* **Gate 3 (full unbalanced test set, 17,043 pairs, original weights, ViennaRNA 2.7.2):** **PASSED** (rule: AUC >= the paper's
  - 0.005, the same AUC tolerance as Gate 2, for consistency with the other gates). miRAW-6-1:10 NF AUC 0.8225 vs 0.8228
  (-0.0003); all 10 configurations within -0.0003..+0.0013 AUC and 0.0055 accuracy/F1, same call for 99.4-99.95 % of pairs.
  With ViennaRNA 2.2.5 the differing pairs equal the paper's and the AUC is the paper's exactly.
* **Orientation experiment (`old-default` vs `natural`):** DECIDED: keep `old-default`. `natural` is significantly better at site
  level (paper regime: AUC +0.0015, acc +0.0057, p <= 0.002) but not at gene level (F1 -0.006 vs old-default, p = 0.76).

## Running jobs (started 2026-10-27, on the dev machine; outputs are in the gitignored `runs/`)

* Gate 1: `pymiraw parity <balanced10 dir> --output-dir runs/parity --cache-dir ~/.cache/pymiraw/foldcache --sets 0-9 --workers 14`
  (log `runs/parity.log`; resumable: rerun the same command; results land in `runs/parity/<config>_<set>/`;
  final comparison in `runs/parity/comparison.csv`; exit code 0 only if every config is within the tolerance, 0.02 since 2026-10-28).
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

1. Merge branch `gate3` (Gate 3 passed) into `main`.
2. **Test the installation in a truly fresh environment**: a machine or container with no ViennaRNA, no PyTorch and none
   of the other dependencies preinstalled. Run the README's install command as written
   (`pip install "pymiraw @ git+https://github.com/AlbertPlaPlanas/pyMiraw.git"`), then `pymiraw --help`, the README
   quick start, `pytest` and the tutorial notebook. The 2026-10-28 check used a new venv but with CPU PyTorch installed
   first; also check a default (CUDA) PyTorch install, macOS and Windows (ViennaRNA wheels), and Python 3.11-3.13.
3. **Publish on PyPI**: the package is ready (branch `pypi-package`); what is left needs the owner's accounts.
   Follow `docs/releasing.md`: add the pending trusted publishers on PyPI and TestPyPI, create the GitHub environments
   `pypi` and `testpypi`, run the publish workflow manually (TestPyPI and the 3-OS smoke test), then tag `v0.2.0`.
   After publishing: `pip install pymiraw` in a fresh environment (item 2; the workflow's smoke test already covers a
   clean install of the wheel on Linux, macOS and Windows).
4. Owner decision on promoting a retrained model (Gate 2 passed; the original weights stay the default until then).
5. Run the original `miRAW.jar` side by side (Java oracle, `environment-reference.yml`), also to compare speed.
6. SpecificZones and ReservedZones filters (unused by the paper); profiling; the `v1.0.0` release.

## Log (newest first)

* 2026-10-30: **Publish workflow, macOS and Windows smoke tests failed** (`RuntimeError: An attempt has been made to
  start a new process before the current process has finished its bootstrapping phase`). Worker processes there are
  spawned, and a spawned worker re-runs the main script; `scripts/smoke_installed.py` had its code at the top level, so
  each worker tried to start workers of its own. Fixed: the script's code is under `if __name__ == "__main__":`; the
  same guard added to `src/pymiraw/__main__.py` (`python -m pymiraw ... --workers N` had the same problem; the installed
  `pymiraw` command did not, pip's launcher has the guard). The Linux smoke job also runs `--spawn` now, so the spawned
  path is tested on every OS. Reproduced locally with spawned workers (old script fails, fixed one passes; `python -m
  pymiraw evaluate --workers 2` passes). README: a note that scripts on macOS/Windows need the guard.

* 2026-10-29 (23:55 UTC): **PyPI package ready (branch `pypi-package`).** `pyproject.toml`: version read from
  `src/pymiraw/__init__.py`, SPDX licence + licence file, author, project URLs, classifiers (Linux/macOS/Windows,
  Python 3.11-3.13; ViennaRNA 2.7.2 has wheels for 3.10-3.14 on all three), keywords, explicit sdist contents (package,
  tests, README, LICENSE, CITATION.cff, and the 2 repository files the tests read; the earlier sdist also shipped a local
  `.hypothesis/` cache, now gitignored). PyPI page via `hatch-fancy-pypi-readme`: relative links/images -> absolute GitHub
  URLs, anchors -> the GitHub page, Mermaid -> a link; checked with `twine check --strict` and `readme_renderer[md]`
  (10 images, no relative links). New `.github/workflows/publish.yml` (tag -> PyPI, manual -> TestPyPI, trusted
  publishing) with a smoke test of the installed wheel on 3 OS x Python 3.11/3.13 (`scripts/smoke_installed.py`:
  predictions equal known values, workers, training). Locally: a fresh venv with the wheel passes the smoke test; the
  sdist's own tests pass (138 + the fixed subprocess test; 3 skip without the paper data). Fixes: the paper-data folder
  of an installed package is now `~/.cache/pymiraw/original` (was inside Python's `lib/`), override `PYMIRAW_DATA_DIR`;
  a test that started a subprocess relied on `PYTHONPATH`. Install instructions now `pip install pymiraw`; PyPI badge.
  Release steps: `docs/releasing.md`.

* 2026-10-29 (23:30 UTC): **CI fix: ViennaRNA could not load on the GitHub runner** (`ImportError: Failed to load ViennaRNA
  RNAlib Python wrapper`; 12 tests). Cause: the Bioconda `viennarna` 2.7.2 build links `libopenblas.so.0` but does not declare
  it; with PyTorch the solver picks MKL BLAS, so the conda env has no OpenBLAS. The dev machine supplied it from the OS
  (`libopenblas0-pthread`), the runner does not. Unnoticed until the weights were bundled, because the tests that run
  ViennaRNA used to skip in CI. Fix: `libopenblas` added to `environment.yml`; `conda-lock.yml` updated with
  `--update libopenblas` (adds libopenblas 0.3.34; 11 dev/AWS tools got patch bumps; numpy, PyTorch and ViennaRNA unchanged);
  a CI step imports ViennaRNA's compiled module so a missing library is named. Checked: an env installed from the lock file
  resolves `libopenblas.so.0` inside the env, and the fast tests pass (141 passed, 1 skipped).
* 2026-10-29 (22:30 UTC): **Gate 3 PASSED; all 10 configurations done.** Gate 3 rule now has an AUC tolerance of 0.005 (owner
  decision): the same AUC margin as Gate 2, so that Gate 3, like Gates 1 and 2, allows for ViennaRNA version differences
  (`scripts/gate3.py` uses `MARGINS["AUC"]`). miRAW-6-1:10 NF: AUC 0.8225 vs the paper's 0.8228 (diff -0.0003) -> PASS. A small
  test with the paper's ViennaRNA 2.2.5 (the 260 pairs that differ under 2.7.2) gave results identical to the paper's.
  All configurations (AUC ours / paper / diff; accuracy and F1 diffs):

  | Config | AUC ours | AUC paper | AUC diff | Acc diff | F1 diff | same call |
  |---|---|---|---|---|---|---|
  | 6-1:10 NF | 0.8225 | 0.8228 | -0.0003 | +0.0002 | +0.0001 | 99.95 % |
  | 6-1:10 AE | 0.7928 | 0.7918 | +0.0010 | +0.0022 | +0.0014 | 99.75 % |
  | 7-1:10 NF | 0.8135 | 0.8133 | +0.0002 | +0.0023 | +0.0013 | 99.72 % |
  | 7-1:10 AE | 0.7840 | 0.7836 | +0.0004 | +0.0049 | +0.0035 | 99.39 % |
  | 7-2:10 NF | 0.8191 | 0.8185 | +0.0007 | +0.0024 | +0.0014 | 99.72 % |
  | 7-2:10 AE | 0.7922 | 0.7909 | +0.0013 | +0.0055 | +0.0038 | 99.37 % |
  | PITA NF | 0.8043 | 0.8038 | +0.0005 | +0.0011 | +0.0008 | 99.87 % |
  | PITA AE | 0.7711 | 0.7705 | +0.0006 | +0.0015 | +0.0015 | 99.82 % |
  | TS NF | 0.7923 | 0.7914 | +0.0009 | +0.0009 | +0.0007 | 99.88 % |
  | TS AE | 0.7621 | 0.7615 | +0.0006 | +0.0016 | +0.0018 | 99.81 % |

  Run time: about 12 h (08:52-20:38 UTC) for the 10 configurations on 14 workers (miRAW NF 8.5 h, mostly cold folding; the rest reuse the
  fold cache and saved sites).

* 2026-10-29 (18:30 UTC): **Gate 3 explained: with ViennaRNA 2.2.5 pyMIRAW reproduces the paper exactly.** The 260 pairs whose
  miRAW NF predictions differed under ViennaRNA 2.7.2 were re-run with ViennaRNA 2.2.5 (`--backend cli`, Bioconda 2.2.5, 1 h 25 min
  on 4 workers): 260/260 same call, same site counts and same prediction (max |diff| 1.8e-6 = rounding of the paper's printed
  values). Full-set AUC with those 260 from 2.2.5: 0.822805 = the paper's 0.822805. So the -0.0003 under 2.7.2 is entirely the
  ViennaRNA energy-parameter change, as in Gate 1. Per-pair values: `docs/results/gate3_differing_pairs_vienna225.csv`.
  Verdict under the rule as written: PASS with ViennaRNA 2.2.5 (exact), FAIL by 0.0003 with the default 2.7.2; the owner
  decides how the rule treats the default backend.

* 2026-10-29 (17:55 UTC): **Gate 3, default configuration: FAIL by 0.0003 under the rule as written (AUC >= the paper's).**
  miRAW-6-1:10 NF on the full unbalanced set (17,043 pairs, original weights, ViennaRNA 2.7.2 bindings): AUC 0.8225 vs the
  paper's 0.8228 (diff -0.0003); accuracy +0.0002, F1 +0.0001; same call for 99.95 % of pairs. 16,783 predictions identical to
  the paper's (< 1e-6); 260 differ (246 targets, 14 non-targets), 8 of them change the call. With the paper's scores for just
  those 14 non-targets the AUC would be 0.8233. Same pattern as Gate 1 (ViennaRNA 2.7.2 vs the paper's 2.2.5 energies).
  Rule NOT changed (owner's call). Next: re-run the 260 differing pairs with ViennaRNA 2.2.5 (`--backend cli`) to test that
  explanation. The other 9 configurations are still running (`runs/gate3/gate3.log`). `docs/results/gate3_comparison.csv` holds
  the miRAW NF row so far.

* 2026-10-29 (09:30): Gate 3 running. miRAW NF chunk 0 (1,000 pairs, all targets: the file lists the 548 non-targets
  last) took 29 min, so miRAW NF needs ~8.5 h (verdict ~17:30); same target/non-target call as the paper for 99.9 % of
  those pairs. The AUC needs the final chunk (the non-targets).

* 2026-10-29 (09:00): **Gate 3 started (branch `gate3`).** The paper archive has the paper's per-pair results on the full
  unbalanced test set (`Results/Evaluate_14K_FULL`, all 10 configs, 17,043 pairs: 16,495 targets, 548 non-targets; miRAW NF and
  7-2:10 NF were run by the authors in two parts each) but not the input file. Rebuilt it (`scripts/gate3.py build`) from those
  pairs plus the sequences in the original repository's `TestData/Data/fullUnifiedFile.csv` (688 MB, 305,008 pairs, sha256
  997e689f...; fetched with a sparse git clone, the Bitbucket raw endpoint fails on it): all 17,043 pairs found with the paper's
  label; sequences identical to the balanced sets where they overlap. The full set's 548 negatives are the balanced sets'
  negatives; only 564 of its positives are in the balanced sets. Paper AUCs on this set (from their outputs): miRAW NF 0.8228
  (the gate's baseline), AE 0.7918, 7-1:10 NF 0.8133 / AE 0.7836, 7-2:10 NF 0.8185 / AE 0.7909, PITA NF 0.8038 / AE 0.7705,
  TS NF 0.7914 / AE 0.7615. Running: `scripts/gate3.py run report --workers 14` (log `runs/gate3/gate3.log`, resumable;
  chunks in `runs/gate3/<config>/`); estimated 5-6 h for miRAW NF (about 15,900 pairs not in the fold cache), ~3 h more for the rest.
  Also fixed: `pymiraw data unpack-archive` never extracted the paper's 9 CV fold models (needed by Gate 2 / `verify_gates.py`);
  they now unpack to `data/original/Results/ResultsCV/` (the archive's layout), with the full-set results.

* 2026-10-29 (08:30): README section "Running on CPU or GPU". Prediction always runs on CPU cores (ViennaRNA is CPU-only;
  the network stays on the CPU in each worker); training follows `accelerator` (`auto` = GPU if PyTorch finds one). New
  `pymiraw train --accelerator`. Measured, CV fold 6 full training (14,000 sites, paper regime): GPU (T4) 14 s, CPU
  (8 threads) 21 s; test accuracy 0.9231 (GPU, = the committed results) vs 0.9286 (CPU), AUC 0.978 both.
  `CUDA_VISIBLE_DEVICES=""` with `auto` gives the CPU result exactly. `make train-cv` time corrected to 20-30 min.

* 2026-10-29 (08:00): **README revised twice for the new API; tutorial predicts from a pandas DataFrame.**
  * README section 2 reorganised (install -> quick start -> notebook -> input -> output -> options -> ViennaRNA -> training ->
    reproducing the paper -> developers); every Python block and CLI example was run in a fresh folder (outputs match the
    comments), all 19 internal links and 15 file links resolve.
  * Fixes found by the revision: `pymiraw train` now defaults to the paper regime (`--no-shuffle`, `--val-fraction 0`), as
    `pymiraw.train` already did; `pymiraw.load_config` exported and `pymiraw.<submodule>` (e.g. `pymiraw.config`) works in a
    fresh interpreter; `scripts/verify_gates.py` works on a fresh clone (retrains first, then uses the retrained models when
    the committed `runs/train_paperRegime_*` are absent); `docs/data.md` (label is the first column of the CV files; bundled
    data and its licence; output names) and `docs/extending.md` updated. 141 tests pass.
  * Notebook section 1.4: a three-pair DataFrame (let-7a vs a 3'UTR with a let-7 site: 0.9996 target; miR-21 vs the same
    3'UTR: -0.98; let-7a vs a 3'UTR without a site: -0.89).

* 2026-10-29 (02:30): **Branch `restructure-api` verified: every gate re-run from scratch on the refactored code gives the
  committed numbers exactly.** `scripts/verify_gates.py` (all steps, `runs/verify/`, ~4 h on this machine): Gate 0 identical;
  Gate 1 from an empty sites folder 100/100 runs with identical TP/FP/TN/FN, PASS (largest config diff 0.0049); Gate 2 gene
  level, all 19 model runs (committed retrained models + paper fold models): 1030 + 900 runs, 0 count differences; site level:
  retraining reproduces all 54 runs (27 old-default weights bit-identical); Gate 2 report byte-identical to the committed one,
  OVERALL PASS. Tutorial notebook executed with outputs (172 s on 16 cores; `scripts/build_tutorial_notebook.py`,
  `scripts/run_notebook.py`). README logo (`docs/logo.png`, 960x540).
* 2026-10-28 (23:30): **Branch `restructure-api`: developer-friendly API, self-contained package.**
  * Original weights bundled as `src/pymiraw/resources/miraw_original.pt` (1.75 MB, bit-exact conversion of `NetworkModel/bestModel.bin`
    by `scripts/bundle_original_weights.py`, sha256 pinned in `pymiraw.resources`); `original_weights_path()` = `$PYMIRAW_ORIGINAL_WEIGHTS`
    or the bundled file, no S3. Paper reference tables moved from `tests/fixtures/` to `pymiraw/resources/`; example data bundled
    (`example_pairs.tsv`, 24 pairs; `example_sites.tsv.gz`, 5000 fold-6 sites decoded to sequences, re-encoding exact; `scripts/make_example_data.py`).
  * New `pymiraw.api` (lazy top-level exports): `predict`, `predict_pair`, `evaluate`, `load_model`, `train`, `cross_validate`,
    `encode_sites`; `Model` (network + encoder + meta), `Predictions` (`.genes`, `.sites`, `.metrics()`, `.save()`); inputs as file,
    DataFrame or `(mirna, utr)` tuples; progress bar. `TrainConfig` moved to `pymiraw.config` (+ `TrainConfig.paper()`);
    `PredictConfig` rejects unknown keys, accepts `Path`s; `backend_args` replaced by `vienna_bin`. CLI is a thin layer over the API;
    `python -m pymiraw`. Scoring always single-threaded (bit-identical across paths). Lightning and ViennaRNA are core dependencies.
  * Equivalence: the 11 CLI runs (example pairs x 10 configs + paper pairs) before/after the refactor give **byte-identical**
    outputs; Gate 0 re-run: fold-6 accuracy 0.9351428571, paper fold models re-scored identical. 137 tests pass.
  * Self-contained check: the wheel (1.8 MB, includes `resources/`) installed with pip into a new venv (torch 2.15 CPU,
    ViennaRNA 2.7.2 from PyPI), run outside the repo with no data: predictions on the 24 example pairs identical to the
    pre-refactor baseline; demo training AUC 0.9272 (same as the source tree on CPU); `pymiraw --help` works.
  * Retraining (`verify_gates.py site`): the 27 old-default models (folds 0-8 x seeds 0-2, paper regime) are **bit-identical**
    to the committed `runs/train_paperRegime_old-default/*.pt` (max |dw| = 0); all 54 runs (both orientations) have the
    committed confusion counts (`runs/verify/results_retrained/cv_paper_regime.csv`).
  * Custom architectures and spawned workers (macOS/Windows): each worker now imports the module of the model's class (a
    spawned worker imports only pyMIRAW, so even a class in an importable module failed before); a class that exists only in
    a notebook/`__main__` is predicted in-process with a warning when workers are not forked. Tests: real spawn run, fallback.
    140 tests pass.
  * Tutorial notebook `notebooks/pymiraw_tutorial.ipynb` (quick start, configuration, training/custom weights/architectures).

* 2026-10-28 (23:00): README figures and citation. New: a pipeline flow diagram (Mermaid), two charts (light/dark SVGs in
  `docs/figures/`, drawn from the committed result files by `scripts/make_readme_figures.py`; byte-reproducible; needs matplotlib,
  not a package dependency): Gate 1 per-configuration differences vs the ±0.02 tolerance, and Gate 2 per-fold site/gene accuracy
  of pyMIRAW vs the paper's own fold models. Citation section + `CITATION.cff` (preferred citation: Pla, Zhong & Rayner 2018, PLoS
  Comput Biol 14(7): e1006185, doi 10.1371/journal.pcbi.1006185; checked on journals.plos.org). Colours: dataviz reference palette
  slots 1-2 (its documented validation; the validator script needs Node.js, not installed here). **Speed vs the original: not
  claimed**: the original `miRAW.jar` has not been timed on this machine (part of the open oracle-run item), so there is no
  like-for-like number.

* 2026-10-28 (22:30): README restructured into three sections with an index: (1) comparison with the original miRAW (results,
  why retraining differs, feature table), (2) usage (install, data, CLI, Python library), (3) status (gates table, done, remaining).
  The two Python library examples were run on `tests/fixtures/paper_pairs.tsv` with the original weights (TMEM136 / miR-129-1-3p:
  Prediction 0.997 with AccessibilityEnergy;10).

* 2026-10-28 (22:15): README: new section "How close is pyMIRAW to the original miRAW?": identical with the original checkpoints
  (bit-exact with ViennaRNA 2.2.5; within 0.005 with 2.7.2), comparable when training from scratch (Gate 2 with the 0.02 margins, post hoc,
  fragile paired gene accuracy). Explanations given (plausible, not tested individually): different RNG/initialisation and unrecorded paper
  seeds; DL4J/ND4J CPU vs PyTorch/cuBLAS GPU float32 summation order amplified by SGD (2,800 updates); framework details (Nesterov
  formulation, how the best-epoch training loss is measured); gene-level amplification of small site-level calibration shifts.

* 2026-10-28 (22:00): **Margins: 0.02 for every non-AUC gate criterion (owner decision, post hoc). Gate 2 now PASSES; Gate 1 still passes.**
  Gate 2 accuracy/F1 margins 0.01 -> 0.02 (`MARGINS` in `pymiraw.evaluation.noninferiority`), AUC stays 0.005; Gate 1 tolerance
  0.03 -> 0.02 (`pymiraw parity --tolerance` default, `evaluation/parity.py`). Gate 1 at 0.02: PASS (largest config mean difference
  0.0049, `docs/results/gate1_comparison.csv`). `scripts/gate2_report.py`: **OVERALL PASS** (exit 0); fragile rows: paired gene accuracy
  per fold (lower -0.0195 vs -0.02, slack 0.0005) and old-default site AUC (slack 0.0009). Diagnostic: the paper's own fold models vs the
  GlobalSummary would still fail the unpaired per-fold accuracy check (-0.0262). Options considered and rejected: per-fold tests (stricter:
  folds 2, 5, 8 fail), pooling the 900 (fold, config, set) cells as independent units (passes, +0.0005, but counts each model 100 times),
  two-sample comparison of the fold models (-0.018 at margin 0.01). The margin was chosen after seeing the results; 0.02 is passed with
  0.0005 to spare. 111 unit tests pass. Promotion of a retrained model is still an owner decision; the original weights stay the default.

* 2026-10-28 (21:40): **Gate 2 test is now rank-based (owner: wins must not penalise us); result: OVERALL FAIL on 2 rows.**
  `pymiraw.evaluation.noninferiority`: `METHOD = "rank"` = exact one-sided 99 % Wilcoxon signed-rank lower bound of the Hodges-Lehmann
  estimate (a large win counts by rank, not size; needs n >= 7 at 99 %); the t-bound stays available as `method="t"`. Post hoc (decided
  after the paired results), recorded as such in PLAN.md section 6. Tests: 3 new (n = 9 bound = 4th smallest Walsh average; a huge win
  gives the same bound as a modest one under rank but a lower one under t; minimum n); 111 unit tests pass. `scripts/gate2_report.py`
  (old-default): site acc HL -0.0016, lower **-0.0109 FAIL** (was -0.0099 PASS under t: the exact rank bound at n = 9 is stricter); site AUC
  -0.0041 PASS (fragile); gene per cell acc -0.0013 / F1 -0.0035 PASS; **gene/p acc HL +0.0015, lower -0.0195 FAIL** (was -0.0229);
  gene/p F1 HL +0.0066, lower -0.0044 PASS. The paired accuracy failure is not caused by the large wins: folds 2 and 8 are ~0.027 below
  the paper's same-fold models, and 3 of 9 folds are below -0.01. **Model improvements: owner rule "no tuning with things the paper did
  not do".** A 3-seed ensemble per fold was tried and withdrawn for that reason (site level on the CV test folds it gave acc 0.942 /
  AUC 0.986 vs single seeds 0.919 / 0.975; the gene-level run was stopped after part of fold 0 and the code was removed, not committed).
  Within the paper's own recipe (XENT, 10 epochs, unshuffled, lowest-training-loss epoch) there is no knob left to change without tuning on
  the gate's test sets, and the paper's own fold models fail the analogous per-fold check by more (diagnostic: lower -0.0262 acc).
  Original weights stay the default.

* 2026-10-28 (21:15): **Gate 2 final, under the paired rule: OVERALL FAIL, only on paired gene-level accuracy.** The paper's 9 fold models ran
  through the same pipeline on all 10 sets x 10 configs (`runs/gene_paper_folds.sh`, 20:15-21:05, all exit 0; fold 6 = `runs/parity`). Rows:
  `docs/results/gene_level_paper_folds.csv` (900 runs; `gene_level_retrained.csv` regenerated, same rows, now sorted by fold).
  `scripts/gate2_report.py`: `gene/p` accuracy mean **+0.0052**, lower bound **-0.0229** (limit -0.01) FAIL; `gene/p` F1 mean **+0.0047**,
  lower **-0.0048** PASS; every other verdict row PASS (unchanged). Paired per fold, ours - paper fold f, dAcc / dF1: f0 -0.006/+0.010,
  f1 +0.024/+0.013, f2 -0.028/-0.014, f3 +0.065/+0.016, f4 +0.021/+0.008, f5 -0.011/+0.007, f6 +0.007/+0.006, f7 +0.002/+0.007,
  f8 -0.027/-0.009. Ours beat the paper's fold model in F1 in 7/9 folds. **Why pairing did not cancel the swings:** they are model-specific,
  not fold-data-specific (per-fold gene accuracy, ours vs paper's same fold: r = -0.14), so pairing adds the paper's variation instead of
  removing it: SD of the paired per-fold dAcc 0.029 vs 0.016 for ours vs the GlobalSummary. The paper's fold models vary more than ours
  (per-fold gene accuracy SD 0.022 vs 0.016; e.g. paper fold 3 gene specificity is 0.21 below ours, fold 3 is 0.046 below the GlobalSummary).
  Diagnostic: the paper's own fold models vs the GlobalSummary would fail the unpaired per-fold rule too (acc mean -0.0051, lower -0.0264;
  F1 -0.0075, lower -0.0194). With n = 9 and a between-model SD of 0.02-0.03, the 99 % bound sits ~0.02-0.03 below the mean, so the
  0.01 accuracy margin cannot be met by any single-model-per-fold comparison, the paper's own models included. **Rule not changed after
  seeing the results**; any change (e.g. more seeds per fold, a different margin, or F1 only at gene level) is the owner's decision and
  would be post hoc. No practical consequence now: the original weights are the default either way.

* 2026-10-28 (20:28): **Gate 2 rule decided (owner), before the paired results were seen:** the gene-level per-fold verdict row is now
  **paired** (`gene/p`: our fold f - the paper's fold f, same cells and pipeline, n = 9, same margins/confidence). The unpaired per-fold row
  (vs the GlobalSummary = one model) moves to the diagnostics, together with the same row for the paper's own fold models. The per-cell
  rows vs the GlobalSummary stay in the verdict (absolute anchor, with Gate 1). PLAN.md section 6 updated; `scripts/gate2_report.py`
  restructured (verdict rows vs diagnostics; OVERALL fails while `docs/results/gene_level_paper_folds.csv` is missing). Smoke-tested on
  partial data (paper folds 0, 6 and part of 1); the retrained rows file regenerates identically. Waiting for paper folds 2-5, 7, 8.

* 2026-10-28 (20:15): **Running: paired Gate 2 baseline (the paper's own fold models).** The archive holds all 9 paper CV models
  (`PLOSComb/Results/ResultsCV/VeryDeepXENT14K_e10/Set{0-8}/bestModel.bin`; extracted to the gitignored `data/original/ResultsCV/`).
  `Set6/bestModel.bin` is byte-identical (sha256 416b1f56...) to `NetworkModel/bestModel.bin`, so the default weights = paper fold 6 and
  `runs/parity` (Gate 1) is its run (symlinked as `runs/gene_paper_f6_s0`). `runs/gene_paper_folds.sh` runs folds 0-5,7,8 on all 10 sets x
  10 configs (same pipeline, saved sites, ~12 min per fold); progress `runs/gene_paper_folds.progress`. Scripts: `gene_level_compare.py`
  now writes `orientation == paper` rows to `docs/results/gene_level_paper_folds.csv`; `gate2_report.py` adds a paired row `gene/p`
  (per fold: ours fold f - paper fold f over the same cells) and a diagnostic (not in the verdict): the paper's fold models vs the GlobalSummary
  with the same unpaired per-fold rule. Why: gene-level per-fold swings follow each model's site recall/specificity balance (Spearman 0.95 with
  gene recall), not its site accuracy (-0.63 with gene accuracy), so the unpaired per-fold row mostly measures model-to-model variation.

* 2026-10-28 (17:15): **Gate 2 gene level on all 9 folds: the per-fold check FAILS.** All folds' retrained `old-default` models (seed 0; fold 6: 3 seeds)
  were run through the pipeline on all 10 sets x 10 configs (`runs/gene_folds.sh`, 8 folds in 3.5 h incl. warm-up; each warm fold 6-13 min thanks to the saved
  sites + ΔG_open). Results: `docs/results/gene_level_retrained.csv` (1030 runs, replaces `gene_level_fold6_retrained.csv`; `gene_level_compare.py` now globs all folds).
  `scripts/gate2_report.py`: the per-cell rows still pass (accuracy lower bound -0.0016, F1 -0.0050; n = 100, optimistic: cells share pairs), but the new
  per-fold row **fails**: mean dAcc +0.0003 / dF1 -0.0028 over 9 folds, lower bounds -0.0151 / -0.0123 vs limit -0.01. Per fold (dAcc / dF1 vs the paper's published
  numbers): f0 -0.017/-0.018, f1 -0.003/+0.003, f2 +0.005/-0.003, f3 +0.020/-0.004, f4 +0.016/+0.003, f5 -0.008/-0.007, f6 +0.007/+0.005, f7 +0.011/+0.012,
  f8 -0.029/-0.016. Site-level metrics of the same models are tight (acc 0.909-0.925, AUC 0.972-0.979). So the means match the paper but gene-level accuracy
  varies by fold by up to ±0.03 (fold 3/4 gain on NF configs through recall, fold 0/8 lose). **Caveat on the baseline:** the paper's GlobalSummary comes from ONE model
  (fold 6, `bestModel.bin`), so a per-fold comparison also measures model-to-model variation, not only our training stack. A paired baseline (the paper's own
  fold-f model through the same gene-level pipeline) would separate the two; not yet run (needs `Results/ResultsCV/Set*/bestModel.bin` from the archive). The
  rule is NOT being changed to make it pass; whether the per-fold row belongs in Gate 2 is the owner's decision. Original weights stay the default.

* 2026-10-28 (15:50): **Everything model-independent is now precomputed and reused.** (1) `--sites-dir` also stores the accessibility
  energies ΔG_open (`opening-*.parquet`, key = backend/version + UTR hash + start + end): they depend only on UTR and position, so they are shared
  across finders, models and AE thresholds (the filter was the bottleneck of the Gate 2 fold runs: ~7 min per AE run even with saved sites).
  (2) The site store reads only the pairs of the current run and builds site objects lazily. Timing, 100 pairs of set 0, `miRAW` + AE;10, 8 workers:
  cold 49 s, warm 3.3 s, warm with AE;5 3.1 s (before: every AE run refolded). Tests: 108 unit tests pass. The Gate 2 fold job was restarted on
  the new code (finished runs are kept). Remaining model-independent cost: none (window co-folds = sites, UTR/open-gap folds = ΔG_open).

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
