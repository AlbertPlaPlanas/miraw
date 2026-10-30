"""One-hot encoding of (miRNA, site) pairs, compatible with the original miRAW/DL4J inputs.

Each nucleotide becomes 4 values in the order C, A, G, U (T is U). Gaps ``L``, spaces and
unknown characters are all-zero. The flat input vector is ``[miRNA(4*mirna_len) | site(4*site_len)]``.

``site_orientation``:
  * ``old-default`` (alias ``reversed``): the site is encoded 3'->5' (reversed), which is what
    the original code did (a documented bug in ``Binarization`` that was "patched" by reversing
    the site again in ``CandidatePrediction``). Required by every model trained by miRAW.
  * ``natural``: the site is encoded 5'->3' as written.
"""

from __future__ import annotations

import numpy as np

ORIENTATIONS = ("old-default", "natural")
_ALIASES = {"reversed": "old-default"}

_LUT = np.zeros(256, dtype=np.int8) - 1  # -1 = all-zero column
for _i, _chars in enumerate(("Cc", "Aa", "Gg", "UuTt")):
    for _c in _chars:
        _LUT[ord(_c)] = _i


def normalize_orientation(value: str) -> str:
    value = _ALIASES.get(value, value)
    if value not in ORIENTATIONS:
        raise ValueError(f"site_orientation must be one of {ORIENTATIONS} (or 'reversed'), got {value!r}")
    return value


def one_hot(seq: str, max_len: int) -> np.ndarray:
    """(max_len, 4) float32 one-hot of the first ``max_len`` characters of ``seq``."""
    out = np.zeros((max_len, 4), dtype=np.float32)
    codes = np.frombuffer(seq[:max_len].encode("latin-1", errors="replace"), dtype=np.uint8)
    idx = _LUT[codes]
    rows = np.nonzero(idx >= 0)[0]
    out[rows, idx[rows]] = 1.0
    return out


class OneHotFlat:
    """Flat one-hot encoder producing the 4*(mirna_len+site_len) input of the miRAW MLP."""

    def __init__(self, mirna_len: int = 30, site_len: int = 40, site_orientation: str = "old-default"):
        self.mirna_len = mirna_len
        self.site_len = site_len
        self.site_orientation = normalize_orientation(site_orientation)

    @property
    def output_dim(self) -> int:
        return 4 * (self.mirna_len + self.site_len)

    def encode_site(self, site: str) -> np.ndarray:
        if self.site_orientation == "old-default":
            site = site[::-1]
        return one_hot(site, self.site_len)

    def encode(self, mirna: str, site: str) -> np.ndarray:
        return np.concatenate([one_hot(mirna, self.mirna_len).ravel(), self.encode_site(site).ravel()])

    def encode_batch(self, mirna: str, sites: list[str]) -> np.ndarray:
        """Encode many sites against one miRNA -> (n_sites, output_dim)."""
        if not sites:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        m = one_hot(mirna, self.mirna_len).ravel()
        s = np.stack([self.encode_site(x).ravel() for x in sites])
        return np.concatenate([np.broadcast_to(m, (len(sites), m.size)), s], axis=1)

    def config(self) -> dict:
        return {
            "name": "onehot_flat",
            "mirna_len": self.mirna_len,
            "site_len": self.site_len,
            "site_orientation": self.site_orientation,
        }
