# pyMIRAW

> **Work in progress.** The pipeline and the training stack are implemented and tested. With the original
> weights the pipeline reproduces the paper's benchmark (all 10 configurations x 10 balanced test sets, every
> mean within 0.003 accuracy / 0.005 F1 of the published values). Retrained models are being validated and are
> not the default. Current status: [docs/PROGRESS.md](docs/PROGRESS.md).

Python/PyTorch reimplementation of **miRAW** ([Pla, Zhong & Rayner, PLoS Comput Biol 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6067737/)),
a deep-learning miRNA-target predictor. The original is Java/DL4J
([bitbucket.org/bipous](https://bitbucket.org/bipous/miraw_-dl4mirnatargeting_code)). The design,
decisions and acceptance gates are in [docs/PLAN.md](docs/PLAN.md).

## Setup

```bash
conda env create -f environment.yml        # or: conda-lock install -n pymiraw conda-lock.yml
conda activate pymiraw
pip install -e . --no-deps
pre-commit install                         # secret guards, once per clone
make test
```

## Get the original miRAW data and weights

The original weights, the paper's test sets and the cross-validation folds come from the authors'
data archive. Download it once and unpack the parts pyMIRAW uses into `data/original/`:

```bash
curl -L -o PLOSComb.7z https://bitbucket.org/bipous/miraw_data/raw/master/PLOSComb.7z   # ~1 GB
pymiraw data unpack-archive PLOSComb.7z
```

This creates:

```
data/original/NetworkModel/bestModel.bin                         # original miRAW weights
data/original/Data/TestData/balanced10/                          # the paper's 10 balanced test sets
data/original/Data/CrossVal/RelaxedPW60_14K_35K/binary/          # released CV folds
data/original/Results/GlobalSummary.xls                          # published results
```

To keep the weights elsewhere, set `PYMIRAW_ORIGINAL_WEIGHTS=/path/to/bestModel.bin`.

## Run with the original miRAW weights

```bash
pymiraw evaluate pairs.tsv --original-weights --finder miRAW --output-dir runs --name my_run
```

`--original-weights` is also the default when no model is given. The input is the miRAW *unified
file*, a TSV with the columns `miRNA, gene_name, EnsemblId, [Positive_Negative,]
Mature_mirna_transcript, 3UTR_transcript` (an example is `tests/fixtures/paper_pairs.tsv`).
Site finders: `miRAW` (paper 6-1:10), `Personalized;0;9;7` (7-1:10), `Personalized;1;9;7` (7-2:10),
`PITA`, `TS`. Filter: `--filter AccessibilityEnergy;10`.

## Exact reproduction with the original ViennaRNA

The default folding backend is ViennaRNA 2.7 (fast, in-process). The paper's numbers were produced with
an older ViennaRNA whose energies differ slightly. For an exact match use the 2.2.5 reference environment:

```bash
conda env create -f environment-reference.yml                         # -> /opt/conda/envs/pymiraw-ref
pymiraw evaluate pairs.tsv --original-weights --backend cli --vienna-bin /opt/conda/envs/pymiraw-ref/bin
```

`--vienna-bin` is the folder holding the 2.2.5 `RNAcofold`/`RNAfold` tools (omit it to use the tools on the PATH).
The same option exists for `pymiraw parity`. The command-line backend is much slower than the default.

## Reproduce the paper benchmark, retrain

```bash
pymiraw parity data/original/Data/TestData/balanced10 --output-dir runs/parity     # hours, uses all cores
pymiraw train  data/original/Data/CrossVal/RelaxedPW60_14K_35K/binary --orientation old-default --seeds 3 \
    --no-shuffle --val-fraction 0 --max-epochs 10     # the original training regime
```

Adding a new architecture (CNN, RNN, Transformer): see [docs/extending.md](docs/extending.md).

## Status

What it takes to bring the original miRAW to Python, and where that stands (details and numbers:
[docs/PROGRESS.md](docs/PROGRESS.md)). Last updated 2026-10-28.

**Done**

| Part | State |
|---|---|
| Original weights (DL4J `bestModel.bin`) imported to PyTorch | exact: fold-6 accuracy 0.935143 as in the paper |
| Candidate-site finders (miRAW, 7-1:10, 7-2:10, PITA, TargetScan), accessibility filter, encoding, gene scoring | ported |
| Pipeline vs paper, all 10 test sets x 10 configurations | **passed**: every configuration within 0.003 accuracy / 0.005 F1 of the published mean (tolerance 0.03) |
| Pipeline with the original ViennaRNA 2.2.5 (`environment-reference.yml`) | exact: identical site counts and scores on 64 pairs |
| Saved candidate sites (`--sites-dir`) | done: 100 pairs take 104 s cold and 3 s warm, identical output |
| Training stack, orientation experiment | done; retrained models are non-inferior to the paper's (site level folds 0-8; gene level fold 6, all 10 sets and configs) but are **not** the default |
| Docs: extending, data, lockfile | done |

**Remaining**

1. Gate 2 with the other folds' retrained models, and the filtered (AE) configs for more than one seed.
2. Run the original `miRAW.jar` as an oracle on sampled pairs (the paper's recorded outputs stand in for it today).
3. Gate 3: the full unbalanced test set (`Evaluate_14K_FULL`).
4. Not ported: the `SpecificZones` / `ReservedZones` filters (unused by the paper's configs) and rebuilding the dataset from raw TarBase/miRTarBase/Ensembl (out of scope for v1).
5. Profiling, packaging for PyPI and the `v1.0.0` release.

The retrained models only reproduce the paper when trained like the original: the released training
files alternate labels and the original read them in file order, never shuffled (`--no-shuffle`).
`--original-weights` stays the default.

## Commands

| | |
|---|---|
| `pymiraw predict / evaluate` | gene-level predictions (and TP/FP/TN/FN, accuracy, F1 when labels are given) |
| `pymiraw parity <balanced10 dir>` | compare with the paper's published table |
| `pymiraw train <folds dir>` | cross-validate on the released folds; `--orientation old-default\|natural`, `--no-shuffle --val-fraction 0` = paper regime |
| `make gate2` | non-inferiority report of the retrained models vs the paper |
| `pymiraw model import-dl4j a.bin b.pt` | convert a DL4J model |
| `--sites-dir DIR` (on predict/evaluate/parity) | save the ViennaRNA binding sites on the first run and reuse them later; `--require-sites` fails instead of folding |
| `pymiraw data unpack-archive` | unpack the original weights/data from `PLOSComb.7z` |

Credentials are never stored in the repository: `.secrets.md` is gitignored and guarded by
pre-commit hooks and CI.

License: GPL-3.0 (the original miRAW license).
