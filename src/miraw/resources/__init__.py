"""Files shipped inside the package: the original miRAW weights, the paper's reference results and small
example datasets. They are found relative to the installed package, so they work from a wheel as well as
from a source checkout.

* ``miraw_original.pt``: the miRAW paper model (``NetworkModel/bestModel.bin`` of the paper archive, a DL4J
  file) converted to the miRAW artefact format. The conversion is bit-exact (``tests`` check it against the
  DL4J file whenever ``data/original`` is present); ``ORIGINAL_WEIGHTS_SHA256`` pins the bundled file.
* ``paper_global_summary.csv``: the paper's ``Results/GlobalSummary.xls`` per (configuration, test set).
* ``paper_cv_xent_models_rescored.csv``: the paper's 9 XENT cross-validation fold models re-scored with
  miRAW's metric code (the Gate 2 site-level baseline).
* ``example_pairs.tsv`` and ``example_sites.tsv.gz``: built by ``scripts/make_example_data.py``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd

ORIGINAL_WEIGHTS_SHA256 = "ea9cd196f22010881368d935e3d6961485761f582888f98569abe3658e36be3b"  # miraw_original.pt
SOURCE_DL4J_SHA256 = "416b1f564562cbe611e72eb30339fa0b1b716da1791367be09248dee1ec955f5"  # NetworkModel/bestModel.bin


def path(name: str) -> Path:
    """Filesystem path of a bundled resource."""
    return Path(str(files(__package__) / name))


def original_weights() -> Path:
    return path("miraw_original.pt")


def paper_global_summary() -> pd.DataFrame:
    return pd.read_csv(path("paper_global_summary.csv"))


def paper_cv_models() -> pd.DataFrame:
    return pd.read_csv(path("paper_cv_xent_models_rescored.csv"))


def example_pairs() -> pd.DataFrame:
    """24 labelled miRNA-gene pairs (12 targets, 12 non-targets) from the paper's balanced test set 0.

    Columns as in a unified input file: ``miRNA, gene_name, EnsemblId, Positive_Negative,
    Mature_mirna_transcript, 3UTR_transcript``.
    """
    return pd.read_csv(path("example_pairs.tsv"), sep="\t", dtype=str, keep_default_na=False)


def example_sites(split: str | None = None) -> pd.DataFrame:
    """Labelled site-level rows (``mirna_seq, site_seq, label``) from the released CV fold 6.

    ``split="train"`` (4000 rows) or ``"test"`` (1000 rows); ``None`` returns both with a ``split`` column.
    """
    d = pd.read_csv(path("example_sites.tsv.gz"), sep="\t")
    return d if split is None else d[d.split == split].drop(columns="split").reset_index(drop=True)
