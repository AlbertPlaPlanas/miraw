"""Write notebooks/pymiraw_tutorial.ipynb from the cells below (no outputs; then run scripts/run_notebook.py).

Usage: python scripts/build_tutorial_notebook.py notebooks/pymiraw_tutorial.ipynb
"""

import sys

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# pyMIRAW tutorial: predict miRNA targets, configure runs, train your own models

pyMIRAW is a Python port of **miRAW** (Pla, Zhong and Rayner, *PLoS Comput Biol* 2018,
[doi:10.1371/journal.pcbi.1006185](https://doi.org/10.1371/journal.pcbi.1006185)). Given a mature miRNA and a
gene's 3'UTR, it finds candidate binding sites, scores each one with a deep neural network and says whether the
gene is a target. **The original miRAW weights come with the package**, so there is nothing to download or set up.

The notebook has three parts:

1. **[Quick start](#1.-Quick-start)**: predictions in a few lines from a file, a pandas table or two sequences; no
   programming experience needed.
2. **[Configuration](#2.-Configuration)**: site finders, filters, ViennaRNA versions, saving work between runs, the command line.
3. **[Training and custom models](#3.-Training-and-custom-models)**: train in the paper's regime, save and reuse your own
   weights, change the training settings, plug in a new architecture.

If you use pyMIRAW, please cite the original miRAW paper (see the end of this notebook).
""")

md("""
## Setup

If pyMIRAW is not installed yet, run this once (it installs pyMIRAW with everything it needs, including ViennaRNA
and the original weights):

```
pip install pymiraw
```

In a notebook you can type `%pip install ...` in a cell instead. Then import it:
""")

code("""
import pymiraw

print("pyMIRAW", pymiraw.__version__)
""")

md("""
## 1. Quick start

Run each cell with **Shift + Enter**. You only need the next three cells to get predictions.

### 1.1 Predict a few example pairs

pyMIRAW ships 24 miRNA-gene pairs from the paper's test data (12 real targets, 12 non-targets) so you can try it
straight away. `pymiraw.predict` does everything: it finds candidate sites, scores them with the original miRAW model
and gives a verdict per pair. A progress bar shows how far it is; all CPU cores are used automatically.
""")

code("""
pairs = pymiraw.example_pairs()
predictions = pymiraw.predict(pairs)
predictions
""")

md("""
### 1.2 Read the results

`predictions.genes` is a table with one row per miRNA-gene pair:

* **`Prediction`** is between **-1 and +1**: the score of the gene's best binding site (the network's
  probability of "target" minus its probability of "no target").
* **`IsTarget`** is `True` when `Prediction > 0`. This is miRAW's call.
* `PosSites` / `NegSites`: how many candidate sites scored above / below zero. `RealClass` is the known answer (1 = target),
  when the input has one.
""")

code("""
predictions.genes[["miRNA", "GeneName", "Prediction", "IsTarget", "RealClass", "PosSites", "NegSites"]]
""")

md("""
The pairs have known answers, so you can see how well the predictions match them:
""")

code("""
m = predictions.metrics()
print(f"accuracy {m['Accuracy']:.2f}, F1 {m['F1']:.2f}  (TP {m['TP']}, FP {m['FP']}, TN {m['TN']}, FN {m['FN']})")
""")

md("""
Every candidate site, with its position in the 3'UTR and its score, is in `predictions.sites`:
""")

code("""
predictions.sites.head()
""")

md("""
### 1.3 Your own data

Put your pairs in a tab-separated file with these columns (the original miRAW "unified file" format):

| column | content |
|---|---|
| `miRNA` | miRNA name, e.g. `hsa-miR-25-3p` |
| `gene_name`, `EnsemblId` | gene name and id (any text) |
| `Mature_mirna_transcript` | mature miRNA sequence |
| `3UTR_transcript` | the gene's 3'UTR sequence |
| `Positive_Negative` | optional: 1 = known target, 0 = not a target |

Then give the file name to `predict` and save the results in a folder. Here we first write the example pairs to a file,
to show the whole round trip:
""")

code("""
from pathlib import Path

work = Path("pymiraw_tutorial_output")
work.mkdir(exist_ok=True)
pairs.to_csv(work / "my_pairs.tsv", sep="\\t", index=False)

predictions = pymiraw.predict(work / "my_pairs.tsv")
folder = predictions.save(work / "my_results")
sorted(p.name for p in folder.iterdir())
""")

md("""
`targetPredictionOutput.csv` (one line per pair) and `targetSites.csv` (one line per site) are the same files the
original miRAW wrote; they open in Excel.

### 1.4 Predict from a pandas table

If your pairs are already in Python, put them in a pandas DataFrame and pass it to `predict` directly, no file needed.
Only two columns are required: **`mirna_seq`** (the mature miRNA) and **`utr`** (the 3'UTR). `miRNA`, `gene_name`,
`EnsemblId` and `label` (1/0) are optional; a DataFrame with the unified-file columns above works as well.

This example has three pairs: let-7a against a 3'UTR fragment that contains a perfect let-7 site, miR-21 against the
same fragment, and let-7a against a 3'UTR without a let-7 site.
""")

code("""
import pandas as pd

let7a = "UGAGGUAGUAGGUUGUAUAGUU"  # hsa-let-7a-5p
mir21 = "UAGCUUAUCAGACUGAUGUUGA"  # hsa-miR-21-5p
utr_a = "UAGCUGAGCGGCGAACCACUAGAAAAGGUUAACUAUACAACCUACUACCUCACAGACCCCGGAGCCCAGCCGUCACGAUUGU"  # has a let-7 site
utr_b = "GCGCGAUUACGCGGCAUAUCGCGCCGAUUAGCGCGAUCGAUCGGCGCUAUCGAUCGCGGCAUAGCUAGCGC"  # no let-7 site

my_pairs = pd.DataFrame({
    "miRNA": ["hsa-let-7a-5p", "hsa-miR-21-5p", "hsa-let-7a-5p"],
    "gene_name": ["geneA", "geneA", "geneB"],
    "mirna_seq": [let7a, mir21, let7a],
    "utr": [utr_a, utr_a, utr_b],
})
my_pairs
""")

code("""
table_predictions = pymiraw.predict(my_pairs)
table_predictions.genes[["miRNA", "GeneName", "Prediction", "IsTarget", "PosSites", "NegSites"]]
""")

md("""
Only let-7a on geneA is called a target: its site pairs perfectly with the let-7 seed. miR-21 does not bind that
fragment, and geneB's 3'UTR has candidate sites, but they all score below zero. A plain list of `(mirna_seq, utr)`
tuples works too: `pymiraw.predict([(let7a, utr_a), (mir21, utr_a)])`.

### 1.5 A single miRNA and 3'UTR

For one pair, paste the two sequences:
""")

code("""
mirna = "CAUUGCACUUGUCUCGGUCUGA"  # hsa-miR-25-3p
utr = pairs["3UTR_transcript"][0]   # a 3'UTR (any RNA or DNA letters)

result = pymiraw.predict_pair(mirna, utr)
print(f"prediction {result.prediction:+.3f} -> {'target' if result.is_target else 'not a target'}, "
      f"{len(result.sites)} candidate sites")
""")

md("""
### How long does it take?

Almost all the time goes to RNA folding (ViennaRNA), and it grows with the length of the 3'UTRs. On a 16-core machine
the 24 example pairs (short 3'UTRs) take about 15 seconds per site finder, and one of the paper's test sets (about
1,100 pairs) takes one to two minutes per configuration. Fewer cores take proportionally longer. Two settings make
repeated runs much faster (section 2.3).
""")

md("""
## 2. Configuration

### 2.1 Site finders and filters

The paper evaluated five ways of finding candidate sites, each with and without an accessibility filter. They are
selected with `finder=` and `filters=`:

| `finder=` | paper name | seed rule |
|---|---|---|
| `"miRAW"` (default) | miRAW-6-1:10 | at least 6 pairs in miRNA nt 2-10 |
| `"Personalized;0;9;7"` | miRAW-7-1:10 | at least 7 pairs in nt 1-10 |
| `"Personalized;1;9;7"` | miRAW-7-2:10 | at least 7 pairs in nt 2-10 |
| `"PITA"` | miRAW-PITA | PITA's seed rules |
| `"TargetScan"` | miRAW-TS | TargetScan's seed rules |

`filters="AccessibilityEnergy;10"` (the paper's "AE" configurations) drops positive sites whose opening energy is above
10 kcal/mol. Removed sites count as 0.
""")

code("""
pita_ae = pymiraw.predict(pairs, finder="PITA", filters="AccessibilityEnergy;10")
print(pita_ae)
print({k: round(v, 3) for k, v in pita_ae.metrics().items() if k in ("Accuracy", "F1")})
""")

md("""
### 2.2 A full configuration object, YAML files and the command line

All the settings of a run are in one `PredictConfig`. Its defaults are the paper's `miRAW-6-1:10` configuration with the
original weights. You can keep a configuration in a YAML file and use it from Python or from the command line.
Unknown or misspelled settings are reported as errors.
""")

code("""
import yaml

config = pymiraw.PredictConfig(finder="Personalized;1;9;7", filters=["AccessibilityEnergy;10"])
(work / "my_config.yaml").write_text(yaml.safe_dump(config.model_dump()))
print((work / "my_config.yaml").read_text())
""")

code("""
from pymiraw.config import load_config

same = pymiraw.predict(pairs, config=load_config(work / "my_config.yaml"), progress=False)
print(same)
""")

md("""
The same run from a terminal (`pymiraw --help` lists every command and option; `evaluate` also writes `metrics.json`):

```
pymiraw evaluate pymiraw_tutorial_output/my_pairs.tsv --config pymiraw_tutorial_output/my_config.yaml --name cli_run
pymiraw predict my_pairs.tsv --finder PITA --filter "AccessibilityEnergy;10" --output-dir results
```

The original miRAW `.properties` configuration files are accepted by `--config` too.
""")

md("""
From a notebook, `!` runs a terminal command; `python -m pymiraw` is the same as `pymiraw`:
""")

code("""
import sys

!{sys.executable} -m pymiraw evaluate {work}/my_pairs.tsv --config {work}/my_config.yaml --output-dir {work} --name cli_run 2>/dev/null
""")

md("""
### 2.3 Faster repeated runs

* `cache_dir=` keeps the RNA folds of every 3'UTR window on disk. A second run on the same pairs (any finder) skips
  most of the folding.
* `sites_dir=` saves the candidate sites (and the accessibility energies) of each finder. Re-running the same finder
  with another model or filter threshold then needs no ViennaRNA at all; this is how the benchmarks compare many models.
* `workers=` sets the number of processes (default: one per CPU core).
""")

code("""
import time

for attempt in ("first run", "second run (sites loaded from disk)"):
    t = time.time()
    pymiraw.predict(pairs, sites_dir=work / "sites", progress=False)
    print(f"{attempt}: {time.time() - t:.1f} s")
""")

md("""
### 2.4 Choosing the ViennaRNA version

Candidate sites come from RNA folding, so the ViennaRNA version slightly changes the results:

* `backend="bindings"` (default): the ViennaRNA Python package installed with pyMIRAW (2.7). With the original weights it
  reproduces the paper's benchmark within 0.005 accuracy.
* `backend="cli"`: the ViennaRNA command-line programs (`RNAcofold`, `RNAfold`) of any installed version, from the PATH
  or from `vienna_bin=`. With ViennaRNA **2.2.5**, the version miRAW used, the published results are reproduced exactly.
  One way to get it: `conda create -n vienna225 -c bioconda viennarna=2.2.5`.
""")

code("""
import shutil

vienna_bin = None  # e.g. "/opt/conda/envs/vienna225/bin"
if vienna_bin or shutil.which("RNAcofold"):
    exact = pymiraw.predict(pairs.head(4), backend="cli", vienna_bin=vienna_bin, progress=False)
    print(exact.genes[["GeneName", "Prediction"]])
else:
    print("No ViennaRNA command-line tools found: set vienna_bin to their folder to try the 'cli' backend.")
""")

md("""
## 3. Training and custom models

### 3.1 Training data

The network scores **sites**: a miRNA sequence plus a 40-nucleotide piece of 3'UTR, labelled 1 (real binding site) or
0. pyMIRAW ships 5,000 such sites sampled from the paper's cross-validation fold 6 (4,000 for training, 1,000 for
testing). The full paper data (about 14,000 sites per fold) comes with the paper archive, see section 3.6.
""")

code("""
train_sites = pymiraw.example_sites("train")
test_sites = pymiraw.example_sites("test")
print(len(train_sites), "training sites,", len(test_sites), "test sites")
train_sites.head()
""")

md("""
### 3.2 Train a model the way the paper did

`pymiraw.train` uses the paper's training regime by default (`TrainConfig.paper()`): the paper's 9-layer network,
Nesterov SGD with learning rate 0.005, batches of 50, XENT loss, 10 epochs in file order, keeping the epoch with the
lowest training loss. With a test set it also reports test metrics. On 4,000 sites this takes a few seconds; a GPU is
used when there is one.
""")

code("""
model = pymiraw.train(train_sites, test_sites)
print(model)
print({k: round(v, 3) for k, v in model.meta["metrics"].items() if k in ("Accuracy", "AUC", "F1")})
""")

md("""
### 3.3 Save it, load it, predict with your own weights

A model is saved as one `.pt` file holding the weights, the architecture and the site encoding it was trained with,
so it cannot be loaded with the wrong encoding by mistake. Give the file (or the model object) to `predict` wherever
the original weights were used, here and on the command line (`pymiraw predict ... --model my_model.pt`).
""")

code("""
path = model.save(work / "my_model.pt")

mine = pymiraw.predict(pairs, path, progress=False)
original = pymiraw.predict(pairs, progress=False)  # the original miRAW weights
for name, p in (("my model (4,000 sites)", mine), ("original miRAW", original)):
    print(f"{name:24s} accuracy {p.metrics()['Accuracy']:.2f}")
""")

md("""
A model trained on 4,000 sites is weaker than the paper's (trained on 14,000). `pymiraw.load_model` opens any model
file, including the original miRAW DL4J files (`bestModel.bin`), and scores site sequences directly:
""")

code("""
loaded = pymiraw.load_model(path)
site_seqs = list(test_sites.site_seq[:3])
mirna_seq = test_sites.mirna_seq[0]
print("my model:", loaded.score_sites(mirna_seq, site_seqs))
print("original:", pymiraw.load_model("original").score_sites(mirna_seq, site_seqs))
""")

md("""
### 3.4 Other training settings

Every field of `TrainConfig` can be changed, by keyword or with a full configuration. Examples: the `natural` site
orientation (the site encoded 5'->3'; the original encoded it reversed), the plain negative log-likelihood loss, early
stopping on a validation split, or a smaller network. Keep in mind that only the defaults reproduce the paper's
models: the gates in `docs/PLAN.md` test those.
""")

code("""
from dataclasses import asdict

custom = pymiraw.TrainConfig(
    loss="nll",                      # instead of the paper's XENT
    site_orientation="natural",      # site encoded 5'->3'
    val_fraction=0.1, patience=3,    # keep the best epoch on a 10 % validation split, stop after 3 bad epochs
    max_epochs=30, shuffle=True,
    model_params={"layer_sizes": (280, 200, 100, 50, 2)},
)
small = pymiraw.train(train_sites, test_sites, config=custom)
print({k: round(v, 3) for k, v in small.meta["metrics"].items() if k in ("Accuracy", "AUC")},
      "after", small.meta["metrics"]["epochs_run"], "epochs")
""")

md("""
### 3.5 A new network architecture

Any PyTorch module that maps the encoded input (280 values: 30 miRNA + 40 site nucleotides, one-hot) to two logits can
be trained, saved and used for prediction. Register it under a name and refer to it in `TrainConfig(model=...)`;
`docs/extending.md` has the details. The network below is a toy (weaker than the paper's MLP); it shows the
mechanics. On macOS and Windows, a class defined in a notebook cell is predicted in one
process (pyMIRAW warns you); put it in a `.py` file and import it to use all CPU cores.
""")

code("""
import torch
from torch import nn


@pymiraw.register_model("tutorial-cnn")
class SiteCNN(pymiraw.SiteClassifier):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv1d(4, channels, kernel_size=5, padding=2)
        self.head = nn.Linear(channels, 2)

    def forward(self, x):
        seq = x.view(len(x), 70, 4).transpose(1, 2)  # (batch, 4 nucleotides, 70 positions)
        return self.head(torch.relu(self.conv(seq)).amax(dim=2))

    def config(self):  # what is needed to rebuild it from a saved file
        return {"name": "tutorial-cnn", "channels": self.channels}


cnn = pymiraw.train(train_sites, test_sites, model="tutorial-cnn", model_params={"channels": 32}, max_epochs=20)
print({k: round(v, 3) for k, v in cnn.meta["metrics"].items() if k in ("Accuracy", "AUC")})
cnn_predictions = pymiraw.predict(pairs, cnn, progress=False)  # an in-memory model works too
print(cnn_predictions)
""")

md("""
### 3.6 The full paper data: cross-validation and benchmarks

The paper archive (`PLOSComb.7z`, 1 GB, from [bitbucket.org/bipous/miraw_data](https://bitbucket.org/bipous/miraw_data))
holds the 10 cross-validation folds and the 10 balanced test sets. Unpack it once with
`pymiraw data unpack-archive PLOSComb.7z` (it needs `pip install "pymiraw[data]"`). Then:

* `pymiraw.cross_validate(folds_dir)` trains one model per fold in the paper's regime (15-20 s per fold, GPU or CPU);
* `pymiraw parity <balanced10 folder>` reruns the paper's 10 x 10 benchmark (Gate 1 in the README).

The next cell runs only when the data is there.
""")

code("""
from pymiraw.storage import ORIGINAL_DIR

folds_dir = ORIGINAL_DIR / "Data/CrossVal/RelaxedPW60_14K_35K/binary"
if folds_dir.is_dir():
    cv = pymiraw.cross_validate(folds_dir, folds=[6], out_dir=work / "cv")  # one fold here; folds 0-8 in the paper
    print(cv[["fold", "seed", "Accuracy", "AUC"]])
else:
    print(f"paper data not found in {ORIGINAL_DIR}: unpack PLOSComb.7z to run cross-validation")
""")

md("""
## Citation

pyMIRAW reimplements the method of the miRAW paper. If you use it, please cite:

> Pla A, Zhong X, Rayner S (2018). miRAW: A deep learning-based approach to predict microRNA targets by analyzing whole
> microRNA transcripts. *PLoS Computational Biology* 14(7): e1006185.
> [doi:10.1371/journal.pcbi.1006185](https://doi.org/10.1371/journal.pcbi.1006185)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1], len(cells), "cells")
