# pyMIRAW

> **Work in progress.** The pipeline and the training stack are implemented and tested. Parity with the
> paper is confirmed on the first balanced test set (all 10 configurations within 0.005 of the published
> accuracy and F1) and on the first six of ten test sets; the rest is still running, so treat the results as preliminary.
> Current status: [docs/PROGRESS.md](docs/PROGRESS.md).

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

## Status of the parity checks (see docs/PROGRESS.md for details)

| Check | State |
|---|---|
| Imported original weights reproduce the paper's fold-6 accuracy | exact (0.935143) |
| Pipeline vs paper, balanced test set 0, 10 configurations | all within 0.005 accuracy/F1 (tolerance 0.03) |
| Pipeline vs paper with the original ViennaRNA version (2.2.5, `environment-reference.yml`) | exact: identical site counts and scores on 64 pairs |
| Pipeline vs paper, test sets 0-5, mean of 10 configurations | all within 0.005 accuracy / 0.008 F1 of the published values (sets 6-9 running) |
| Retrained model vs the paper's own fold models (site level, folds 0-8) | non-inferior at 99 % (accuracy 0.919-0.925, AUC 0.975-0.977 vs 0.921 / 0.9765) |
| Retrained model at gene level | non-inferior on the partial evidence (fold 6, unfiltered configs, 2 test sets) |

The retrained models only reproduce the paper when trained like the original: the released training
files alternate labels and the original read them in file order, never shuffled (`--no-shuffle`).
Retrained models are **not** the default; `--original-weights` is.

## Commands

| | |
|---|---|
| `pymiraw predict / evaluate` | gene-level predictions (and TP/FP/TN/FN, accuracy, F1 when labels are given) |
| `pymiraw parity <balanced10 dir>` | compare with the paper's published table |
| `pymiraw train <folds dir>` | cross-validate on the released folds; `--orientation old-default\|natural`, `--no-shuffle --val-fraction 0` = paper regime |
| `make gate2` | non-inferiority report of the retrained models vs the paper |
| `pymiraw model import-dl4j a.bin b.pt` | convert a DL4J model |
| `pymiraw data unpack-archive` | unpack the original weights/data from `PLOSComb.7z` |

Credentials are never stored in the repository: `.secrets.md` is gitignored and guarded by
pre-commit hooks and CI.

License: GPL-3.0 (the original miRAW license).
