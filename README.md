# pyMIRAW

Python/PyTorch reimplementation of **miRAW** ([Pla, Zhong & Rayner, PLoS Comput Biol 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6067737/)),
a deep-learning miRNA-target predictor. The original is Java/DL4J
([bitbucket.org/bipous](https://bitbucket.org/bipous/miraw_-dl4mirnatargeting_code)). The full design,
decisions and acceptance gates are in [docs/PLAN.md](docs/PLAN.md); current status in [docs/PROGRESS.md](docs/PROGRESS.md).

## Setup

```bash
conda env create -f environment.yml     # or: make env
conda activate pymiraw
pip install -e . --no-deps              # or: make install
pre-commit install                      # secret guards, once per clone
make test
```

## Run with the original miRAW weights

```bash
# the original paper weights are fetched from S3 on first use and checksum-verified
scripts/with-s3-creds.sh pymiraw evaluate pairs.tsv --original-weights --finder miRAW
# offline: PYMIRAW_ORIGINAL_WEIGHTS=/path/to/bestModel.bin pymiraw evaluate ...
```

`--original-weights` is also the default when no model is given. Input is the miRAW *unified file*
(TSV: `miRNA, gene_name, EnsemblId, [Positive_Negative,] Mature_mirna_transcript, 3UTR_transcript`).
Finders: `miRAW` (paper 6-1:10), `Personalized;0;9;7` (7-1:10), `Personalized;1;9;7` (7-2:10), `PITA`, `TS`;
filter: `--filter AccessibilityEnergy;10`.

## Commands

| | |
|---|---|
| `pymiraw predict / evaluate` | gene-level predictions (and TP/FP/TN/FN, accuracy, F1 with labels) |
| `pymiraw parity <balanced10 dir>` | reproduce the paper's benchmark (Gate 1) |
| `pymiraw train <folds dir>` | cross-validate a model on the released folds; `--orientation old-default\|natural` |
| `pymiraw model import-dl4j a.bin b.pt` | convert a DL4J model |
| `pymiraw data pull\|verify` | fetch/verify heavy data from S3 |

Heavy data lives on S3 (`data/manifest.yaml`), never in git. Credentials are never stored in the
repository (`.secrets.md` is gitignored and guarded by pre-commit hooks and CI).

License: GPL-3.0 (the original miRAW license).
