"""Build the small example datasets bundled in ``miraw/resources/`` from the paper data (data/original/).

* ``example_pairs.tsv``: 24 labelled miRNA-gene pairs (12 targets, 12 non-targets) with short 3'UTRs, from
  the paper's balanced test set 0, so a first prediction runs in seconds.
* ``example_sites.tsv.gz``: labelled site-level training rows (miRNA and site sequences) sampled from the
  released CV fold 6 (train and test split). The released rows are one-hot features; they are decoded back
  to sequences here and the script checks that re-encoding them gives the released features exactly.

Usage: python scripts/make_example_data.py      (needs `miraw data unpack-archive PLOSComb.7z` first)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from miraw.encoding.onehot import OneHotFlat
from miraw.training.data import MIRNA_LEN, SITE_LEN, load_csv

root = Path(__file__).resolve().parents[1]
original = root / "data/original/Data"
out = root / "src/miraw/resources"
NT = np.array(list("CAGU"))


def decode(block: np.ndarray, n: int) -> str:
    """One-hot rows (n x 4) -> sequence; all-zero rows are gaps ('L'), trailing ones are padding."""
    rows = block.reshape(n, 4)
    seq = np.where(rows.any(1), NT[rows.argmax(1)], "L")
    return "".join(seq).rstrip("L")


def example_pairs() -> pd.DataFrame:
    d = pd.read_csv(original / "TestData/balanced10/randomLeveragedTestSplit_0.csv", sep="\t", dtype=str)
    short = d[d["3UTR_transcript"].str.len().between(200, 1200)]
    picked = [short[short.Positive_Negative == c].sample(12, random_state=0) for c in ("1", "0")]
    return pd.concat(picked).sort_index()


def example_sites(n_train: int = 4000, n_test: int = 1000) -> pd.DataFrame:
    enc = OneHotFlat(MIRNA_LEN, SITE_LEN, "old-default")  # the released features use the original orientation
    parts = []
    for split, n in (("train", n_train), ("test", n_test)):
        x, y = load_csv(original / f"CrossVal/RelaxedPW60_14K_35K/binary/set_6_{split}.csv")
        idx = np.sort(np.random.default_rng(0).choice(len(y), n, replace=False))
        mirnas = [decode(r[: 4 * MIRNA_LEN], MIRNA_LEN) for r in x[idx]]
        sites = [decode(r[4 * MIRNA_LEN :], SITE_LEN)[::-1] for r in x[idx]]  # stored 3'->5': reverse back
        again = np.stack([enc.encode(m, s) for m, s in zip(mirnas, sites, strict=True)])
        if not np.array_equal(again, x[idx]):
            raise AssertionError(f"decoding the {split} rows does not round-trip")
        parts.append(pd.DataFrame({"split": split, "mirna_seq": mirnas, "site_seq": sites, "label": y[idx]}))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    pairs = example_pairs()
    pairs.to_csv(out / "example_pairs.tsv", sep="\t", index=False)
    sites = example_sites()
    sites.to_csv(out / "example_sites.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{len(pairs)} pairs, {len(sites)} sites (round-trip exact) -> {out}")
