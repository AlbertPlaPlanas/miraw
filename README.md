# pyMIRAW

Python/PyTorch reimplementation of **miRAW** ([Pla, Zhong & Rayner, PLoS Comput Biol 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6067737/)),
a deep-learning miRNA-target predictor. The original is Java/DL4J
([bitbucket.org/bipous](https://bitbucket.org/bipous/miraw_-dl4mirnatargeting_code)).

> **Work in progress.** The prediction pipeline and the training stack are implemented and tested. With the original
> weights pyMIRAW reproduces the paper's benchmark; retrained models give comparable results and are not the default.
> See [3. Status](#3-status).

## Contents

1. [How pyMIRAW compares to the original miRAW](#1-how-pymiraw-compares-to-the-original-miraw)
   * [Results](#results)
   * [ViennaRNA version and exact reproduction](#viennarna-version-and-exact-reproduction)
   * [Why retraining does not give the same weights](#why-retraining-does-not-give-the-same-weights)
   * [What is the same and what differs](#what-is-the-same-and-what-differs)
2. [Using pyMIRAW](#2-using-pymiraw)
   * [Installation](#installation)
   * [Original data and weights](#original-data-and-weights)
   * [Command line](#command-line)
   * [Python library](#python-library)
3. [Status](#3-status)

---

## 1. How pyMIRAW compares to the original miRAW

### Results

* **With the original checkpoints (`--original-weights`, the default): identical results.** With the ViennaRNA version
  the paper used (2.2.5) the port is bit-exact: same candidate sites, same site scores (max difference 0.00000) and same
  gene calls. With the default ViennaRNA 2.7.2 the network is still identical, but ViennaRNA's energy parameters changed
  between versions, so a few borderline sites are found differently; all 10 configurations stay within 0.005 accuracy and
  F1 of the published results over the paper's 10 balanced test sets (Gate 1).
* **Training from scratch (same recipe, same folds): comparable results, not identical ones.** Retrained models match the
  paper's on average (site-level accuracy -0.002 and AUC -0.001 versus the paper's own fold models; gene-level accuracy and
  F1 within 0.005) and pass the non-inferiority test of Gate 2 (99 % one-sided bound within 0.02 accuracy/F1 and 0.005 AUC).
  Individual models, ours and the paper's alike, differ by up to about ±0.03 in gene-level accuracy from fold to fold.

The gates and their exact rules are in [docs/PLAN.md](docs/PLAN.md) (section 6); the result files are in
[docs/results/](docs/results/).

### ViennaRNA version and exact reproduction

The original miRAW ran the ViennaRNA 2.2 command-line tools; pyMIRAW defaults to the faster ViennaRNA 2.7.2 Python
bindings. ViennaRNA's energy parameters and algorithms changed between these versions, so a few candidate sites whose
binding energy is close to the cut-off are found (or not) differently. With the original weights this changes the gene
call for very few pairs (all 10 configurations stay within 0.005 accuracy and F1 of the paper). With ViennaRNA 2.2.5 the
port is bit-exact. To reproduce the paper exactly, use the reference environment and the command-line backend:

```bash
conda env create -f environment-reference.yml                         # ViennaRNA 2.2.5 -> /opt/conda/envs/pymiraw-ref
pymiraw evaluate pairs.tsv --backend cli --vienna-bin /opt/conda/envs/pymiraw-ref/bin
```

The 2.2.5 tools are much slower than the default bindings. How to select other ViennaRNA versions:
[Choosing the ViennaRNA version](#command-line).

### Why retraining does not give the same weights

The training recipe is the same as the original's (Nesterov SGD, momentum 0.9, learning rate 0.005, batch 50, XENT loss,
Xavier initialisation, 10 epochs, file order). The remaining differences are plausibly due to:

1. **Random initialisation.** DL4J/ND4J and PyTorch use different random number generators, and the paper's seeds are not
   recorded, so the networks start from different Xavier draws and converge to different (equally good) minima.
2. **Numerical backend.** The original was trained with DL4J on ND4J (Java, CPU BLAS); pyMIRAW trains with PyTorch, here on a
   GPU (cuBLAS, non-deterministic kernels). Different float32 summation orders give differences of ~1e-7 per step, which
   SGD amplifies over the 2,800 updates of training (280 per epoch) into different final weights.
3. **Framework details.** Small implementation differences, for example in how the Nesterov update is formulated and how the
   training loss used to pick the best epoch is measured (DL4J scores the network after the epoch, Lightning averages the
   mini-batch losses during it), can select a different epoch.
4. **Amplification at gene level.** A gene is predicted as a target if one of its many candidate sites scores positive, so
   a small shift in a model's site-level decision point changes gene-level specificity a lot. This is why retrained models
   agree closely with the paper at site level but vary more at gene level, as the paper's own fold models do.

### What is the same and what differs

| | original miRAW | pyMIRAW |
|---|---|---|
| Network | DL4J MLP 280 → 400-350-300-150-100-100-50-50 → 2 | same architecture in PyTorch; DL4J `.bin` checkpoints load directly |
| Site finders | miRAW (6-1:10), Personalized, PITA, TargetScan | same; `miRAW` = `Default` with miRNA nt 2-10, as in the published runs |
| Filters | AccessibilityEnergy, SpecificZones, ReservedZones | AccessibilityEnergy (the one the paper used); the other two are not ported |
| RNA folding | ViennaRNA 2.2 command-line tools | ViennaRNA 2.7 Python bindings by default; 2.2.5 command-line tools for exact reproduction |
| Input / output | unified TSV, `targetPredictionOutput.csv`, `targetSites.csv` | same files; YAML configs, and legacy `.properties` files are read |
| Training | DL4J, CPU | PyTorch Lightning, CPU or GPU; any registered architecture ([docs/extending.md](docs/extending.md)) |
| Speed-ups | – | parallel workers; ViennaRNA results (sites and accessibility energies) can be saved and reused (`--sites-dir`) |

---

## 2. Using pyMIRAW

### Installation

```bash
conda env create -f environment.yml        # or: conda-lock install -n pymiraw conda-lock.yml
conda activate pymiraw
pip install -e . --no-deps
pre-commit install                         # secret guards, once per clone (contributors)
make test
```

### Original data and weights

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

### Command line

**Predict / evaluate.** The input is the miRAW *unified file*, a TSV with the columns `miRNA, gene_name, EnsemblId,
[Positive_Negative,] Mature_mirna_transcript, 3UTR_transcript` (example: `tests/fixtures/paper_pairs.tsv`).

```bash
pymiraw predict  pairs.tsv --output-dir runs --name my_run                     # original weights (default)
pymiraw evaluate pairs.tsv --finder miRAW --filter "AccessibilityEnergy;10" --name my_eval   # needs Positive_Negative
pymiraw predict  pairs.tsv --model my_model.pt                                 # a retrained model
pymiraw predict  -c run.yaml                                                   # all options from a YAML (or legacy .properties) file
```

Each run writes `runs/<name>/targetPredictionOutput.csv` (one row per pair), `targetSites.csv` (every candidate
site and its score), `config.yaml` and, for `evaluate`, `metrics.json` (TP/FP/TN/FN, accuracy, F1, ...).

Scores range from -1 to +1. A site's score is p(target) - p(no target) from the network's two outputs, so a score
above 0 means p(target) > 0.5. A pair's `Prediction` is the highest score among its sites, and **the pair is predicted
as a target when `Prediction > 0`**. Pairs without candidate sites, and positive sites removed by a filter, score 0
(not a target). This is the same convention as the original miRAW.

* Site finders (`--finder`): `miRAW` (paper 6-1:10), `Personalized;0;9;7` (7-1:10), `Personalized;1;9;7` (7-2:10), `PITA`, `TS`.
* Filter (`--filter`, repeatable): `AccessibilityEnergy;10` (removes positive sites whose opening energy is above +10).
* `--workers N` (default: all cores); `--sites-dir DIR` saves the ViennaRNA results on the first run and reuses them on
  later runs with any model (`--require-sites` fails instead of folding).

**Choosing the ViennaRNA version.** pyMIRAW folds RNA with ViennaRNA through one of two backends (`--backend`, or
`backend:` in a YAML config):

* `bindings` (default): the ViennaRNA Python bindings (`import RNA`) of the active environment, in-process and fast.
  The version is whatever is installed there (2.7.2 in `environment.yml`); to use another one, install it in the
  environment (e.g. `conda install -c bioconda viennarna=2.6.4`, or `pip install ViennaRNA==<version>`).
* `cli`: the `RNAcofold` and `RNAfold` command-line tools of **any** installed ViennaRNA version, run as subprocesses
  (much slower). `--vienna-bin DIR` is the folder holding the tools; without it the tools on the `PATH` are used.

```bash
pymiraw evaluate pairs.tsv --backend cli --vienna-bin /path/to/viennarna/bin     # any version's tools
pymiraw evaluate pairs.tsv --backend cli                                          # the RNAcofold on the PATH
```

The same options exist for `pymiraw parity`. The version in use is recorded in the saved results: `--sites-dir` and
`--cache-dir` key everything by backend and version, so results from different versions are never mixed. For the
ViennaRNA version that reproduces the paper exactly, see
[ViennaRNA version and exact reproduction](#viennarna-version-and-exact-reproduction).

**Benchmark and training.**

```bash
pymiraw parity data/original/Data/TestData/balanced10 --output-dir runs/parity     # Gate 1, hours, uses all cores
pymiraw train  data/original/Data/CrossVal/RelaxedPW60_14K_35K/binary --seeds 3 \
    --no-shuffle --val-fraction 0 --max-epochs 10                                 # the original training regime
make gate2                                                                        # Gate 2 report from docs/results/
```

| Command | Purpose |
|---|---|
| `pymiraw predict` / `evaluate` | gene-level predictions (and metrics when labels are given) |
| `pymiraw parity <balanced10 dir>` | all 10 configurations x 10 test sets, compared with the paper's published table |
| `pymiraw train <folds dir>` | cross-validation on the released folds; `--orientation old-default\|natural` |
| `pymiraw model import-dl4j a.bin b.pt` | convert a DL4J checkpoint to a pyMIRAW artefact |
| `pymiraw data unpack-archive` | unpack the original weights and data from `PLOSComb.7z` |

`pymiraw <command> --help` lists every option.

### Python library

Run the pipeline on a unified file:

```python
from pymiraw.prediction.pipeline import predict_file, results_frame
from pymiraw.storage import original_weights_path

cfg = {
    "model": str(original_weights_path()),     # or a .pt artefact from `pymiraw train`
    "finder": "miRAW",
    "filters": ["AccessibilityEnergy;10"],     # [] for no filter
    "backend": "bindings",
    "max_site_length": 40,
    "seed_alignment_offset": 5,
    # optional: "sites_dir": "runs/sites" to save and reuse the ViennaRNA results
}
results = predict_file("pairs.tsv", cfg, workers=4)
df = results_frame(results)    # GeneName, miRNA, Prediction, HighestPredVal, PosSites, NegSites, RemovedSites, ...
```

Predict a single miRNA-3'UTR pair without files:

```python
from pymiraw.folding.base import FoldService
from pymiraw.folding.vienna_cli import make_backend
from pymiraw.models.artifact import load_artifact
from pymiraw.prediction.pipeline import Predictor
from pymiraw.sites.finders.registry import get_site_finder
from pymiraw.storage import original_weights_path

model, encoder, _ = load_artifact(original_weights_path())   # .bin (DL4J) or .pt
predictor = Predictor(model, encoder, get_site_finder("miRAW"), FoldService(make_backend("bindings")))
res = predictor.predict_pair(mirna_sequence, utr_sequence)
# other ViennaRNA: make_backend("cli", bin_dir="/path/to/viennarna/bin"); folding temperature: make_backend("bindings", temperature=25)
res.prediction           # highest site score, in [-1, +1]; > 0 means target
[(s.start, s.end, s.score) for s in res.sites]               # candidate sites and their scores
```

New architectures (CNN, RNN, Transformer) plug into the same training and prediction code: see
[docs/extending.md](docs/extending.md).

---

## 3. Status

**TL;DR (2026-10-28):** the port is functionally complete. With the original weights it reproduces the paper's benchmark
(all 10 configurations x 10 test sets within 0.005 of the published means; bit-exact with ViennaRNA 2.2.5). Retrained
models pass Gate 2 (non-inferiority within 0.02 accuracy/F1 and 0.005 AUC) but are not the default.

| Gate | Result |
|---|---|
| Gate 0: imported paper model vs the paper's CV | passed (identical fold-6 test accuracy, 0.9351428571) |
| Gate 1: pipeline with the original weights vs the published benchmark | passed (100/100 runs) |
| Gate 2: retrained models vs the paper's models | passed; not promoted to default |
| Gate 3: full unbalanced test set | not run yet |

**Done:** folding backends (ViennaRNA 2.7 bindings, 2.2.5 command line), all site finders used in the paper,
the accessibility filter, the MLP and DL4J checkpoint import, the prediction pipeline and CLI, evaluation and the
benchmark runner, the training stack (paper regime, any registered architecture), saved ViennaRNA results, docs
([docs/data.md](docs/data.md), [docs/extending.md](docs/extending.md)) and the conda lockfile.

**Remaining:** a decision on promoting a retrained model, the `miRAW.jar` oracle run, Gate 3, the SpecificZones and
ReservedZones filters (unused by the paper), profiling, PyPI packaging and the `v1.0.0` release.

Detailed log, numbers and commands: [docs/PROGRESS.md](docs/PROGRESS.md). Design and decisions: [docs/PLAN.md](docs/PLAN.md).

---

Credentials are never stored in the repository: `.secrets.md` is gitignored and guarded by pre-commit hooks and CI.

License: MIT (see [LICENSE](LICENSE)).
