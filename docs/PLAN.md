# pyMIRAW — Python reimplementation plan

Port of miRAW (Pla, Zhong & Rayner, *PLoS Comput Biol* 2018, [PMC6067737](https://pmc.ncbi.nlm.nih.gov/articles/PMC6067737/);
Java/DL4J code at [bipous/miraw_-dl4mirnatargeting_code](https://bitbucket.org/bipous/miraw_-dl4mirnatargeting_code/src/master/))
to a maintainable Python package on PyTorch.

**Scope:** a faithful port of the existing pipeline and its MLP. No new architectures yet, but the
encoder/model interfaces are designed so CNNs, RNNs or Transformers can be added later as plug-ins
without touching the pipeline.

**Non-negotiable acceptance rule:** pyMIRAW must be *at least as good* as the original on the paper's
own benchmark (see [§6 Acceptance gates](#6-acceptance-gates-at-least-as-good)).

---

## 1. What the original does (from reading the code + paper data)

Core package is `no.uio.dl4miRNA` (~15 k LOC Java; the rest of the repo is experiments/scratch).

| Stage | Java class(es) | Behaviour to reproduce |
|---|---|---|
| Unified dataset | `DCP/dataConstruction.buildUnifiedFile` | TarBase v7 + miRTarBase 6.1 + Ensembl 87 3'UTRs + miRBase r21 → TSV `miRNA, gene_name, EnsemblId, Positive_Negative, Mature_mirna_transcript, 3UTR_transcript` |
| Candidate site search (CSSM) | `SiteFinder/*CandidateSiteFinder` | Slide a window (30 or 40 nt) over the 3'UTR **from 3' to 5', step 1**. Co-fold each window with the miRNA via `RNAcofold` (`site&miRNA`). Keep it if ΔG < max ΔG (0), the structure shows a real duplex, and ≥ *k* pairs fall in the seed region. Variants: `Default` (paper "miRAW-6-1:10"; config type `miRAW` falls through to it), `Personalized;start;size;minBinds` (paper "7-1:10" = `0;9;7`, "7-2:10" = `1;9;7`), `PITA`, `TargetScan` |
| Overlap resolution | `CandidateSiteFinder.getCandidateSites` | `alignAndExtend` re-centres every site so the seed starts at `siteLength − seedAlignmentOffset`, padding with `L`. Then keep the lower-ΔG site when two consecutive sites start < `winStep` (5) apart |
| A-posteriori filters | `SiteFilters/*` | `AccessibilityEnergy;T`: ΔG_open = ΔG(RNAfold with site positions constrained unpaired) − ΔG(RNAfold free), computed on a 200-nt window around the site. Site removed if ΔG_open > T. Applied **only to sites the network called positive**; removed sites get score 0. Also `SpecificZones` and `ReservedZones` |
| Encoding | `DCP/Binarization`, `DL/CandidatePrediction` | One-hot C,A,G,U(T) = `1000,0100,0010,0001`; `L`/space/unknown = `0000`. miRNA padded to 30 nt. **Site is encoded reversed** (documented bug in `Binarization` line ~189, "patched" in `CandidatePrediction`). Flat vector = `[miRNA(120) ‖ site(4·L)]` |
| Network | `DL/MLModelLearner` | Dense ReLU MLP → softmax(2). Paper model `bestModel.bin`: **280 → 400 → 350 → 300 → 150 → 100 → 100 → 50 → 50 → 2**, BinaryXENT loss, Nesterov momentum 0.9, LR 0.005, batch 50, Xavier init, early stopping on validation loss |
| Site score | `CandidatePrediction.doPrediction` | `p(pos) − p(neg)` ∈ [−1, 1]. Positive if > 0 |
| Gene score | `DL/GenePrediction.doPrediction` | max over site scores (0 if there are no sites). Outputs `targetPredictionOutput.csv` (per pair) and `positiveTargetSites.csv` (per site) |
| Training data | `DCP/RobustDataset`, `dataConstruction` | Positives: CLASH/CLIP sites cross-referenced with TarBase/miRTarBase plus TargetScan conserved sites, aligned to 40 nt (30 + 5 + 5). Negatives: candidate sites (minBindsSeed = 5) in validated-negative pairs. Relaxed pairwise CV (14 000 train / 3 500 test, 10 sets) |

### Reference material available (all public, already downloaded to the session scratchpad)

* [`bipous/miraw_data`](https://bitbucket.org/bipous/miraw_data) → `PLOSComb.7z` (1.0 GB packed, 3.3 GB unpacked). Contains:
  * `NetworkModel/bestModel.bin`: **the paper's trained XENT model** (DL4J zip: `configuration.json` + `coefficients.bin`).
  * `ConfFiles/`: every training and testing config used in the paper.
  * `Data/ValidTargetSites/`: 33 142 positive and 32 284 negative training sites, plus their binarized form.
  * `Data/CrossVal/RelaxedPW60_14K_35K/`: the exact 10 CV folds.
  * `Data/TestData/balanced10/`: the 10 balanced test sets (548+/548−; the negatives are the same in every set).
  * `Data/DIANA DataSet/DIANADataSet.zip`: unified file, train/test splits.
  * `Results/`: per-run `targetPredictionOutput.csv`, per-site `positiveTargetSites.csv` / `allTargetSites.csv`, and `GlobalSummary.xls` (the numbers we must match).
* [`bipous/miraw_dl4mirna_binaries`](https://bitbucket.org/bipous/miraw_dl4mirna_binaries): `miRAW.jar` / `miRAWGPU.jar`, to run the original as a reference oracle.
* Original repo `TestData/`: `shortUnifiedFile.csv` (5 pairs), `goodModel.bin` and their expected outputs. `src/test/.../*NGTest.java` holds about 12 hand-checked site-finder cases each for Default, PITA and TargetScan.

Verified during planning:
* `coefficients.bin` parses in pure Python: 207 002 float32 values for `goodModel`, with nothing left over. DL4J weights can therefore be imported into PyTorch without Java.
* S3 read/write works with the credentials described in `.secrets.md`.
* bioconda provides ViennaRNA **2.2.5** (close to what miRAW was built against) and **2.7.2**.
* **A 100-line Python prototype of the `Default` CSSM** (ViennaRNA 2.7.2 bindings) was compared
  against the published `positiveTargetSites.csv` (XENT, balanced set 0). Over 120 pairs spanning
  all UTR lengths it reproduced **2 302 / 2 315 (99.4 %) recorded sites exactly**: start, end, seed
  pairs and ΔG. The 0.6 % residue is consistent with ViennaRNA 2.2 → 2.7 drift.
  * Cost: about 18 CPU-seconds per pair. A full balanced set takes about 20 minutes on 16 cores.
    The 1-nt windows are shared by all CSSMs, so the fold cache makes the 10-config benchmark
    cost little more than one config.

### Resolved decisions (2026-10-27)

Evidence comes from the code, the saved models and the CV logs in `Results/ResultsCV/`.

1. **No autoencoder pre-training.**
   * Every training path (`MLModelLearner.createCustomModel`) builds a fresh Xavier-initialized
     `MultiLayerNetwork` with `pretrain(false)` on every layer and never loads external weights.
   * The only unsupervised layer type in the repo is an RBM in the unused `no.uio.future`
     package, which is also `pretrain(false)`.
   * `bestModel.bin` stores `pretrain: false` and plain dense layers.
   * The paper's description is not reproduced.
2. **Dropout = 0.0.**
   * `MLExecution` defaults dropout to 0 when `ANN.Dropout` is absent, and none of the paper
     configs set it. The 0.2 is only the constructor default, which gets overwritten.
   * The CV logs print `Dropout rate= 0.0`, and the model JSON agrees.
   * The port's MLP keeps a `dropout` parameter (default 0.0) for future experiments.
3. **Paper training regime, as it actually ran** (and one more finding, 2026-10-27: the training files
   are strictly alternating 1,0,1,0,... and DL4J read them **in file order, never shuffled**, so every
   batch of 50 was balanced and identical each epoch. Shuffling instead costs about 0.02 accuracy and
   0.012 AUC; `pymiraw train --no-shuffle --val-fraction 0` reproduces the paper regime):
   * XENT, Nesterov 0.9, LR 0.005, batch 50, **max 10 epochs**;
   * no validation-based early stopping;
   * the "best" model is the epoch with the lowest *training* score (the last minibatch), at
     epochs 4–7;
   * the "last" model **collapsed to predicting all-negative in every fold** (accuracy 0.5);
   * `bestModel.bin` is byte-identical to CV `Set6/bestModel.bin` (the paper's "fold 7",
     1-based);
   * XENT CV over the 9 evaluated folds (`BestModelEval.txt`): mean accuracy **0.921**, mean
     AUC **0.958**.
   * Consequence: `train` checkpoints on **validation** loss by default. The `compat` regime is
     available for reproduction only.
4. **Seed windows: the code is authoritative, and the paper labels are display names only.**
   * The original intent was "6-1:10 = ≥ 6 pairs in miRNA positions 1–10". The published results,
     however, were produced with ≥ 6 pairs in **nt 2–10**: `seedRegionStart = 1`, size 9. The
     nt 1–10 variant is a commented-out older setting.
   * Prototype evidence: nt 2–10 reproduces 99.4 % of recorded sites, nt 1–10 only 65.5 %.
   * Likewise `Personalized;0;9;7` covers nt 1–9 and `Personalized;1;9;7` covers nt 2–10.
   * pyMIRAW keeps the paper names (`miraw-6-1:10`, …) but documents the real windows. A true
     nt 1–10 variant can be added later as a separate, explicitly named config.
5. **ΔG_open threshold.** The code semantics are kept: remove a site if ΔG_open > +10, as in
   `AccessibilityEnergy;10`. Gate 1's AE configs confirm or refute this empirically.
6. **Tolerance for folding-related differences is ±0.03** (accuracy and F1 per config), so the
   ViennaRNA **2.7.2 bindings are the default backend**. 2.2.5 stays available for exact
   reproduction.
7. **No dataset rebuild from raw sources in v1** (see §7).
   * The released archive contains everything both gates need: unified file and splits
     (`DIANADataSet.zip`), training sites, CV folds and test sets.
   * "At least as good" is measured on those released test sets, so a rebuilt dataset would
     break comparability.
   * The random splits could not be regenerated anyway, and TarBase v7 requires DIANA
     registration.
   * The released files are imported to parquet instead. The site-generation code is still
     ported so new training sets can be derived from a unified file later.
8. **Site encoding orientation: implement `old-default` first, then test `natural`.**
   * **Step 1 (Phases 1–3):** implement the reversed encoding exactly as the original, behind
     `site_orientation: old-default`.
     * It is the default for v1 and the only option for imported DL4J models.
     * Gates 0 and 1 are run with it.
   * **Step 2 (Phase 4):** add `site_orientation: natural` and retrain with each orientation under
     the same protocol: released CV folds, the same ≥ 5 seeds, the same hyper-parameters, the
     same pipeline and folding backend.
     * `natural` becomes the default for *newly trained* models only if it passes Gate 2 **and**
       is better on both primary metrics:
       * mean site-level CV AUC, paired per fold and seed;
       * mean gene-level F1 across the 10 configs × 10 balanced sets, paired, with a one-sided
         Wilcoxon p < 0.05.
     * A tie or a non-significant difference keeps `old-default` (no churn).
     * Results are recorded in `docs/parity.md` either way.
   * **Outcome (2026-10-27): `old-default` kept.** In the paper regime `natural` is significantly better at site
     level (AUC +0.0015, one-sided p <= 0.002) but not at gene level (F1 -0.006, p = 0.76), so the plan's rule
     (both metrics must win) keeps `old-default`. Details in `docs/PROGRESS.md`; rerun for position-aware models.
   * **Expectation (written before the experiment):** for the MLP, orientation is only a fixed permutation of input features,
     which a fully connected network can absorb, so a difference within seed noise is the likely
     outcome. The experiment still validates the `natural` path, which will matter for
     position-aware architectures (CNN/RNN/Transformer). Those should rerun this comparison when
     they are added.
9. **Bugs that the port reproduces in `compat` mode** (and only fixes behind an explicit flag):
   * the reversed site encoding (`site_orientation: old-default`, see decision 8);
   * `Default` finder never scans position 0 (`i > 0`), and both finders skip the trailing sub-window;
   * `Default` rejects ΔG == 0 while `Personalized` rejects ΔG ≥ 0;
   * the `Personalized;…` factory parsing quirks;
   * `getPairAt`'s opening-bracket branch starts at `pos − 1` (unreachable in practice, but tested).

---

## 2. Target architecture

```
pyMiraw/
├── environment.yml              # main dev/run env (conda-forge + bioconda, ViennaRNA 2.7.2, PyTorch CUDA)
├── environment-reference.yml    # parity env: ViennaRNA 2.2.5 CLI + OpenJDK 8 to run the original miRAW.jar
├── conda-lock.yml               # fully pinned lock for linux-64 (generated)
├── pyproject.toml               # hatchling build; ruff, mypy, pytest, coverage config
├── Makefile                     # one-word entry points: env, test, lint, parity, train, ...
├── .pre-commit-config.yaml      # ruff, ruff-format, mypy, gitleaks (blocks committed secrets)
├── .github/workflows/ci.yml     # free GitHub-hosted runner: lint + unit + fast functional (~5 min)
├── configs/
│   ├── paper/                   # YAML twins of the paper's .properties (10 CSSM×filter configs, training)
│   └── examples/
├── data/manifest.yaml           # every heavy artefact: S3 URI + sha256 + size + provenance
├── scripts/with-s3-creds.sh     # exports creds from the gitignored .secrets.md, then exec "$@"
├── src/pymiraw/
│   ├── cli.py                   # Typer app: `pymiraw <command>`
│   ├── config.py                # pydantic v2 models; YAML loader + legacy .properties loader
│   ├── storage.py               # fsspec/s3fs: pull/push/verify against manifest, local cache
│   ├── io/                      # fasta, unified TSV/parquet, legacy CSV writers (byte-compatible outputs)
│   ├── folding/                 # FoldingBackend protocol
│   │   ├── vienna_bindings.py   #   default: in-process `RNA` module (fast)
│   │   ├── vienna_cli.py        #   RNAcofold/RNAfold subprocess (any version, used for 2.2.5 parity)
│   │   └── cache.py             #   on-disk memo keyed by (seqs, constraint, backend, version)
│   ├── sites/
│   │   ├── structure.py         # dot-bracket utils: binding start, pair lookup, duplex check, WC/wobble
│   │   ├── candidate.py         # CandidateSite dataclass
│   │   ├── align.py             # alignAndExtend
│   │   ├── finders/             # default.py, personalized.py, pita.py, targetscan.py + registry
│   │   └── filters/             # accessibility.py, zones.py + registry
│   ├── encoding/
│   │   ├── base.py              # Encoder protocol: (mirna, site) -> Tensor; declares output shape
│   │   └── onehot.py            # OneHotFlat (DL4J-compatible, site_orientation="old-default"|"natural")
│   │                            # OneHotSequence ((L,4) channels; for future CNN/RNN/Transformer)
│   ├── models/
│   │   ├── base.py              # SiteClassifier(nn.Module): forward(batch) -> logits (B,2)
│   │   ├── registry.py          # @register_model("mlp"); build_model(cfg)
│   │   ├── mlp.py               # MirawMLP(layer_sizes, dropout, activation)
│   │   └── dl4j_import.py       # DL4J zip -> MirawMLP state_dict + metadata
│   ├── training/
│   │   ├── data.py              # Dataset/DataModule over parquet site tables
│   │   ├── module.py            # LightningModule, generic over any registered model + encoder
│   │   ├── losses.py            # "xent" (DL4J BinaryXENT on softmax), "nll" (cross-entropy)
│   │   └── cv.py                # released folds loader + relaxed-pairwise CV generator (seeded)
│   ├── datasets/                # unified build, negative/positive site generation
│   ├── prediction/
│   │   ├── pipeline.py          # find → (encode, score) batched → filter → aggregate
│   │   └── aggregate.py         # max / counts / removed-sites logic
│   └── evaluation/
│       ├── metrics.py           # TP/FP/TN/FN, acc, precision, recall, NPV, specificity, F1, ROC-AUC
│       └── report.py            # GlobalSummary-style table + paper comparison
├── tests/
│   ├── unit/  functional/  acceptance/
│   ├── fixtures/                # < 1 MB golden files, committed
│   └── conftest.py              # markers: slow, s3, gpu, reference (oracle env), acceptance
└── docs/                        # PLAN.md, parity.md (live results), data.md, extending.md
```

### Design choices

* **PyTorch + Lightning.** The MLP needs nothing fancy. Lightning provides checkpointing, early
  stopping, seeding, GPU/CPU switching and loggers for free, and is where future architectures will
  benefit most.
* **Folding is the real cost, not the network.** The original spawns one `RNAcofold` process per
  1-nt window. pyMIRAW uses the in-process ViennaRNA bindings, a process pool (16 cores here) and an
  on-disk cache. Expect orders-of-magnitude faster site finding.
* **Plug-in seams for future architectures:** `Encoder` (declares output shape) + `SiteClassifier`
  (logits in, registered by name) + YAML `model: {name: mlp, ...}`. Adding a CNN means one module,
  one `@register_model`, one config, and zero pipeline changes. A test-only dummy model in the
  test suite proves the seam stays open.
* **Two numerical modes:** `compat` (bit-faithful to Java, including bugs; default for the paper
  model) and `fixed` (bugs corrected; only used by newly trained models, recorded in model
  metadata). Each model artefact stores its encoder and mode, so the wrong one can never be used.
* **Original-weights flag (implemented).** Any prediction command can be run with the original
  miRAW paper weights, so results are always comparable with the published ones:
  * CLI: `pymiraw predict|evaluate <input> --original-weights`; YAML: `original_weights: true`
    or `model: original`. It overrides any other `model` value, and **it is also the default
    when no model is given**, so pyMIRAW can never silently run with an untested model.
  * The weights are `bestModel.bin` of the paper, a DL4J file that is converted to PyTorch at load
    time (no Java). They are not stored in git (3 MB, over the 1 MB hook limit); they sit at
    `raw/original/miraw_data/NetworkModel/bestModel.bin` in the S3 bucket, are listed with their
    sha256 in `data/manifest.yaml`, and are fetched into `~/.cache/pymiraw` on first use
    (`pymiraw data pull|verify original-weights`).
  * Offline or without S3 access, set `PYMIRAW_ORIGINAL_WEIGHTS=/path/to/bestModel.bin`.
  * The original weights imply the `old-default` site orientation (the importer sets it), and
    the run summary prints `weights: original miRAW` so it is visible in every log.
  * Tests that need the weights are marked `data` and skip cleanly when they are unavailable.
* **Site orientation flag:** `encoding.site_orientation` takes `old-default` or `natural` (see
  decision 8 in §1).
  * `old-default` (alias `reversed`) is the original behaviour. The site is one-hot encoded
    3'→5', i.e. reversed, as the paper model expects.
  * `natural` encodes the site 5'→3', as written.
  * The value is saved in every model artefact and checked at load time. Running a model with the
    wrong orientation is a hard error, not a silent accuracy loss.
* **Configs:** YAML validated by pydantic. `pymiraw ... --legacy-properties file.properties` also
  accepts the original configs unchanged.
* **Outputs:** parquet internally. Legacy TSVs (`targetPredictionOutput.csv`, `positiveTargetSites.csv`)
  are written byte-compatible so old analysis scripts keep working.
* **CLI** mirrors the original verbs: `data pull|push|verify`, `dataset build-unified|sites|cv`,
  `sites find`, `model import-dl4j`, `train`, `predict`, `evaluate`, `report`, `parity`.
  `predict`, `evaluate` and `parity` accept `--original-weights`.

---

## 3. Conda environment (development **and** execution)

The `defaults` channel is not reachable from this machine, so both env files use `nodefaults`.

`environment.yml` (main env, `pymiraw`):

```yaml
name: pymiraw
channels: [conda-forge, bioconda, nodefaults]
dependencies:
  - python=3.12
  - viennarna=2.7.2          # provides the `RNA` Python module + CLI tools
  - pytorch>=2.5             # no build string: CUDA build on the T4 box, CPU build on CI runners
  - lightning>=2.4
  - numpy, pandas, pyarrow, scikit-learn, scipy
  - pydantic>=2, pyyaml, typer, rich, joblib, tqdm
  - fsspec, s3fs, boto3
  - py7zr, xlrd              # reading the original archive / GlobalSummary.xls
  # dev
  - pytest, pytest-cov, pytest-xdist, hypothesis, moto
  - ruff, mypy, pre-commit, conda-lock
  - pip
  - pip:
      - -e .                 # the package itself, editable
```

`environment-reference.yml` (parity oracle env, `pymiraw-ref`): `viennarna=2.2.5`, `openjdk=8`,
`python=3.12`, `pandas`. It is used only to run the original `miRAW.jar` and to give pyMIRAW's CLI
folding backend the exact ViennaRNA version.

Setup (also wrapped as `make env` / `make env-ref`):

```bash
conda env create -f environment.yml            # or: conda-lock install -n pymiraw conda-lock.yml
conda activate pymiraw
pre-commit install
make test                                      # unit + fast functional
# parity oracle (only needed for Gate 1 / regenerating golden files)
conda env create -f environment-reference.yml
```

Lock refresh: `conda-lock -f environment.yml -p linux-64` (commit `conda-lock.yml`). The lock pins
the workstation (CUDA) environment for reproducible local and acceptance runs. CI solves the CPU
variant from `environment.yml` and caches it (see §5, CI/CD).

---

## 4. Data management (S3)

Root: `s3://digitalrnd-drf-projects-raw-ireland/spatial_omics/dev/pyMIRAW/` (currently empty).

```
pyMIRAW/
├── raw/original/miraw_data/PLOSComb.7z          # + unpacked/ (selected folders)
├── raw/original/miraw_binaries/miRAW.jar
├── raw/original/miraw_code/TestData/            # fullUnifiedFile.csv (657 MB) etc.
├── raw/external/                                # Ensembl 87 3'UTR, miRBase r21, miRTarBase 6.1,
│                                                #   TarBase v7*, TargetScan 7.1, CLASH, CLIP
├── processed/v{N}/                              # unified.parquet, sites/*.parquet, cv/*.parquet
├── models/{name}/{version}/                     # model.pt, model.yaml (encoder, mode, metrics), source .bin
├── reference/                                   # golden outputs from the Java oracle
└── runs/{yyyymmdd}_{name}/                      # predictions, metrics, logs of acceptance runs
```

* `data/manifest.yaml` in git lists every object: URI, sha256, size, provenance and licence note.
  `pymiraw data pull <key>` downloads into `~/.cache/pymiraw`, verifies the checksum and is
  idempotent. `pymiraw data push` uploads and updates the manifest.
* Only fixtures under 1 MB go in git (`tests/fixtures/`).
* \*TarBase v7 needs registration with DIANA, so keep it in the private bucket only and never in git.
* **Credentials:** never in code, config or git. `.secrets.md` is gitignored (done during planning).
  `scripts/with-s3-creds.sh` reads it, unsets the web-identity role variables as the file instructs,
  and runs the command (`make data-pull` uses it). The library itself only uses the standard boto3
  credential chain.
* **Secret guard (implemented 2026-10-27)** has three layers:
  1. pre-commit hooks: `scripts/check_secrets.py`, gitleaks, detect-private-key,
     detect-aws-credentials, and check-added-large-files (> 1 MB). The custom check blocks
     credential files, the literal values from the local `.secrets.md`/`.env`, and key patterns,
     and it scans the staged content.
  2. `.github/workflows/secrets.yml` re-runs the hooks on all files and gitleaks on the full
     history, catching `--no-verify` commits and clones without hooks.
  3. GitHub push protection.

  Every new clone must run `pre-commit install` once.
  *Recommendation:* the credential is a long-lived IAM user key in plain text. Plan to rotate it and
  move it to a named profile or a secrets manager.

---

## 5. Testing strategy

| Layer | Runs | Examples |
|---|---|---|
| **Unit** (`tests/unit`, < 30 s, no network, ViennaRNA stubbed where possible) | every commit, CI | Encoding: bit-exact against sampled rows of `allTrainingSitesBinarized.txt`, including the reversed site and `L` padding. Structure utils: binding start, `getPairAt`, duplex check, WC/wobble tables. `alignAndExtend` edge cases (start < 0, end > len). Overlap resolution with a fake folding backend. Each finder on the ported Java NGTest cases. Filters with a stubbed ΔG. Gene aggregation (max, removed sites → 0, counts, no-site pairs). Metrics against scikit-learn. Config and legacy `.properties` parsing. Model registry plus a dummy plug-in model. DL4J import shapes and layout. S3 storage against `moto` |
| **Property** (hypothesis) | CI | Encoded length is invariant. Aligned sites always have length L. Kept sites are never closer than `winStep`. Scores stay in [−1, 1]. Encoder output is deterministic |
| **Functional** (`tests/functional`, < 5 min) | CI | (a) The imported `bestModel.bin` / `goodModel.bin` reproduces recorded **site-level** predictions from `positiveTargetSites.csv` to ≤ 1e-5, with no folding needed. (b) End-to-end on `shortUnifiedFile.csv` + `goodModel` → `targetPredictionOutput.csv` / `targetSites.csv` (exact with the 2.2.5 CLI backend, tolerance-checked with 2.7.2). (c) Training smoke test: tiny data, 2 epochs, loss decreases, checkpoint round-trip gives identical outputs. (d) CLI smoke tests via Typer's `CliRunner` |
| **Reference** (`-m reference`) | on demand, needs `pymiraw-ref` | Runs the original `miRAW.jar` on sampled pairs and diffs candidate sites, ΔG and predictions against pyMIRAW |
| **Acceptance** (`make parity`, hours, 16 CPUs) | before each release / model promotion | Full Gate 1 / Gate 2 benchmarks below. Writes `docs/parity.md` and `runs/…` on S3 |

Coverage target: ≥ 90 % on `src/pymiraw` excluding CLI glue.

### CI/CD: zero cost, zero infrastructure

Constraint: this is an individual, unpaid project, so there must be no paid services and no
runners to manage.

* **GitHub Actions on GitHub-hosted `ubuntu-latest` runners only.** There is nothing to provision or
  maintain; it is one workflow file. The repo is **public**, so standard runners have no minute
  limit. Even if it went private again, GitHub Free gives 2 000 min/month with a $0 spending limit,
  so it would never bill.
* **GitHub secret scanning + push protection** (free for public repos) is the server-side last line
  of defence. Check it is enabled in *Settings → Code security*.
* **Keep each run at about 5 minutes** for fast feedback:
  * one job: ruff → mypy → gitleaks → `pytest -m "not slow and not s3 and not reference and not acceptance"`;
  * CPU only;
  * the conda env comes from `mamba-org/setup-micromamba` with its built-in cache, keyed on the
    hash of `environment.yml`;
  * triggers are pull requests and pushes to `main`;
  * `concurrency: cancel-in-progress` drops superseded runs;
  * `paths-ignore` skips docs-only changes.
* **One env file works for GPU and CI.** `environment.yml` pins `pytorch` without a build string.
  conda-forge picks the CUDA build on the T4 workstation and the CPU build on CI runners (no
  `__cuda` virtual package), so no CI-specific env file needs maintaining.
* **CI never touches S3 and holds no secrets.** Every CI test runs on small committed fixtures.
  Tests that need S3, the Java oracle, a GPU or hours of CPU are marked and run only locally
  (`make test-all`, `make parity`). The workstation is the free "heavy runner".
* **pre-commit** (ruff, mypy, gitleaks) runs the same checks locally before each commit, so CI is a
  safety net rather than the main gate.
* **No CD service.** A "release" is a git tag. `make release` builds the wheel, and when a model is
  promoted it is pushed to `s3://…/models/` by `pymiraw data push`. GitHub Releases (free) can hold
  the changelog. No package registry or Docker registry is needed.

---

## 6. Acceptance gates ("at least as good")

Benchmark = the paper's 10 balanced test sets × 10 configurations (miRAW 6-1:10, 7-1:10, 7-2:10,
PITA, TS; each with no filter (NF) and with the accessibility filter (AE)). The baseline is
`Results/GlobalSummary.xls`. Mean results:

| Config | Acc | F1 | | Config | Acc | F1 |
|---|---|---|---|---|---|---|
| 6-1:10 AE | 0.705 | 0.729 | | 6-1:10 NF | 0.649 | 0.727 |
| 7-1:10 AE | 0.712 | 0.707 | | 7-1:10 NF | 0.690 | 0.741 |
| 7-2:10 AE | 0.721 | 0.722 | | 7-2:10 NF | 0.688 | 0.744 |
| PITA AE | 0.657 | 0.569 | | PITA NF | **0.729** | 0.731 |
| TS AE | 0.647 | 0.532 | | TS NF | 0.718 | 0.714 |

Site-level CV (paper, XENT): accuracy 0.92, AUC 0.96.

* **Gate 0: network fidelity.** The imported DL4J model reproduces recorded site predictions to
  ≤ 1e-5 (max abs) on every site in the shipped result files.
* **Gate 1: pipeline fidelity (paper model, port only).** Always run with `--original-weights`.
  * With the default ViennaRNA 2.7.2 bindings backend: mean accuracy and F1 within **±0.03** of
    GlobalSummary for every config (folding tolerance agreed 2026-10-27), and the drift reported
    in `docs/parity.md`.
  * With the 2.2.5 CLI backend: TP/FP/TN/FN equal GlobalSummary per set and per config. Fewer than
    0.5 % of pairs may differ, each explained.
  * This gate proves the port is correct.
* **Gate 2: retrained model (pyMIRAW training stack): non-inferiority.** Rule agreed 2026-10-27, replacing
  the earlier "mean at least as high as the paper". Folding is held fixed so the gate measures only the
  network. A retrained model passes if, for every row below, the **one-sided 99 % lower confidence bound
  of the paired difference (ours - paper) is above -margin**, with margins 0.01 for
  accuracy and F1 and 0.005 for AUC (`pymiraw.evaluation.noninferiority`: `CONFIDENCE`, `MARGINS`, `METHOD`).
  Bound (changed 2026-10-28, owner decision, **after** the paired results were seen): exact Wilcoxon signed-rank
  bound of the Hodges-Lehmann estimate, so that large wins count by rank and do not widen the bound. Until then
  it was the t-based bound of the mean (`method="t"`), under which large wins inflate the SD.
  Retrained models may only use what the paper's training did (owner rule, 2026-10-28): no ensembling,
  threshold calibration or other additions to pass the gate.
  * Site level: per-fold differences over folds 0-8 (seed-averaged), 9 paired values, versus the paper's
    own fold models **re-scored with our metric code** (`tests/fixtures/paper_cv_xent_models_rescored.csv`;
    acc 0.9210, AUC 0.9765). The paper's *recorded* AUC (0.958) used coarse thresholds and is not the
    baseline. Metrics: accuracy, AUC. Retraining uses the paper regime (`--no-shuffle --val-fraction 0`).
  * Gene level: per (config x balanced test set) differences, run through the same pipeline and folding
    backend as the paper numbers (`tests/fixtures/paper_global_summary.csv`). Metrics: accuracy, F1.
    The final check must cover all 10 configs (NF and AE) and all 10 sets.
  * Gene level, per fold (added 2026-10-28, owner decision, fixed **before** the paired results were seen):
    one paired value per fold 0-8 = mean over the same (config x set) cells of (our fold-f model - the
    paper's own fold-f model, `PLOSComb/Results/ResultsCV/VeryDeepXENT14K_e10/Set<f>/bestModel.bin`), both run
    through the same pipeline, folding backend and saved sites. Metrics: accuracy, F1. Same margins and
    confidence. Reason: the GlobalSummary comes from one model (paper fold 6 = `NetworkModel/bestModel.bin`,
    byte-identical) and gene-level results swing by up to ±0.03 between fold models, so an unpaired per-fold
    comparison against it measures model-to-model variation rather than the training stack. The unpaired
    per-fold row (each of our fold models vs the GlobalSummary) and the same row for the paper's own fold
    models are still printed by the report, as diagnostics outside the verdict. The per-cell row above, together
    with Gate 1, remains the absolute anchor to the published numbers.
  * `scripts/gate2_report.py` evaluates the rule from the committed result files and exits non-zero on failure.
  * Limits: the cells of the gene-level check share pairs across configs, so its bounds are optimistic;
    passes with slack below 0.001 are flagged as fragile. The margins were chosen after seeing the data.
  * Passing does **not** promote a model automatically: the original weights (`--original-weights`,
    `model: original`) remain the default unless the owner decides otherwise, so pyMIRAW is never worse than miRAW.
* **Gate 3: full unbalanced test set** (`Evaluate_14K_FULL`): AUC ≥ the paper's (> 0.75) for the
  default model.

---

## 7. Phased roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **0. Bootstrap** | Repo skeleton, `pyproject`, both env files + lock, Makefile, pre-commit, CI. Original archive, binaries and test data mirrored to S3 with `manifest.yaml`. Released data (unified file, splits, training sites, CV folds, test sets, results) imported to parquet under `processed/v1/`. `pymiraw-ref` env runs `miRAW.jar` on `shortUnifiedFile.csv` and stores golden outputs in `s3://…/reference/` | `make test` green in CI. `pymiraw data verify` passes. The Java oracle reproduces `TestData/ExperimentResults/genePredict` |
| **1. Primitives** | `folding/` (both backends + cache), `sites/structure.py`, `encoding/onehot.py`, `models/mlp.py`, `models/dl4j_import.py`, `model import-dl4j` CLI | **Gate 0** |
| **2. Candidate sites & filters** | All four finders, `alignAndExtend`, overlap resolution, accessibility/zone filters, `sites find` CLI, parallel execution | Ported NGTests pass. Candidate sites (positions, ΔG, transcript) match Java oracle on ≥ 200 sampled pairs per finder (2.2.5) |
| **3. Gene prediction & evaluation** | `prediction/`, `evaluation/`, `predict`, `evaluate`, `report`, `parity` CLI, legacy output writers, YAML twins of all paper configs | **Gate 1** and **Gate 3** (paper model) |
| **4. Training stack** | Lightning module, losses (xent/nll), early stopping, released-fold loader, seeded relaxed-pairwise CV generator, training-site generation (negatives; positives from aligned CLASH/CLIP + TargetScan), `train` CLI, model artefact format. **Orientation experiment:** `old-default` vs `natural`, same seeds, folds and pipeline (§1, decision 8) | Reproduces paper CV on released folds. **Gate 2** decides promotion. The orientation verdict is recorded in `docs/parity.md` |
| **5. Hardening & docs** | `docs/extending.md` (how to add an architecture, with the dummy model as a worked example), `docs/data.md`, performance profiling, release tag `v1.0.0` | All gates green, docs reviewed |

Phases 1 → 3 are the critical path. Phase 4 can start once Phase 1 lands, since the released training
sites are already encoded.

**Out of scope for v1 (backlog):** rebuilding the unified dataset from raw TarBase, miRTarBase,
Ensembl and miRBase (`dataset build-unified`). See decision 7 in §1. Revisit it only when a newer
TarBase or Ensembl release is wanted. It would be a new benchmark, not a parity target.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ViennaRNA 2.2 → 2.7 energy changes alter candidate sets and therefore results | Pluggable folding backend. Gate 1 is run on 2.2.5 for exactness and drift is measured on 2.7.2. The backend and version are stored in every run's metadata and in the fold cache key |
| DL4J flattened-parameter layout (`f` vs `c` order, W/b interleaving) | Gate 0 checks it against recorded predictions, so a wrong layout fails immediately |
| CV randomness makes the paper's training irreproducible | Use the released folds and the released model. Seed everything in new CV generation |
| Ambiguities between paper and code (§1) | Resolved: code, saved model and logs are the source of truth, and the seed window was verified empirically (§1, decisions 1–6) |
| XENT training is unstable (the original "last" models collapsed) | Checkpoint on validation loss, seed sweeps, and a collapse detector (constant predictions) that fails the run |
| Folding cost on 300 k pairs | Bindings + process pool + persistent cache. Full-dataset runs are batch jobs that write to S3 |
| Licence of third-party data (TarBase) | Private bucket only. The manifest records the licence. Nothing third-party goes in git |
| Secrets leakage | `.gitignore`, gitleaks in pre-commit and CI, a wrapper script instead of hard-coded keys |
| `miRAW.jar` may not run on modern Java | The reference env pins OpenJDK 8 |
| Public repo: anything committed is published | A pre-commit secret guard (implemented: `scripts/check_secrets.py` + gitleaks + private-key and large-file hooks), the same checks in CI, and GitHub push protection. Heavy or licensed data lives only in S3 |

---

## 9. Open questions

None at the moment. Everything from the first draft was resolved on 2026-10-27, including the
encoding orientation strategy (§1, "Resolved decisions").

## 10. Exclusions

Exclude plans and information explaining reimplementation decisions from the repository.

## 11. Code Style
Include docstrings in all functions
Be generous with comments so people understand what is going on

## 12. Distribution
Package should be installable using pypi and easy to use

## 10. Working agreement: keep a progress log

Work happens across many sessions and machines. The state of the work is therefore recorded in git:

* [`docs/PROGRESS.md`](PROGRESS.md) is the single source of truth for status, results, running jobs,
  findings and next steps. [`CLAUDE.md`](../CLAUDE.md) at the repo root tells every new Claude Code
  session to read PROGRESS.md and this plan first.
* **The assistant must update PROGRESS.md after every meaningful step** (feature done, test or
  benchmark result, decision, blocker) and before ending a session, recording numbers and exact
  commands. A session that changes the state without updating PROGRESS.md is incomplete.
* PROGRESS.md never contains credentials or private data. This plan changes only when a decision changes.
