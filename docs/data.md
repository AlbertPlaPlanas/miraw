# Data

pyMIRAW needs three kinds of data. None of the heavy files are stored in git.

| What | Where it comes from | Used by |
|---|---|---|
| Original miRAW weights (`bestModel.bin`, 3 MB) | `PLOSComb.7z` (`NetworkModel/`) | `--original-weights` (the default) |
| Paper test sets and CV folds | `PLOSComb.7z` (`Data/TestData/balanced10`, `Data/CrossVal/RelaxedPW60_14K_35K/binary`) | `pymiraw parity`, `pymiraw train` |
| Published results (`GlobalSummary.xls`) | `PLOSComb.7z` (`Results/`) | parity comparison |

## Getting the data

The archive is published by the miRAW authors at
<https://bitbucket.org/bipous/miraw_data> (about 1 GB):

```bash
curl -L -o PLOSComb.7z https://bitbucket.org/bipous/miraw_data/raw/master/PLOSComb.7z
pymiraw data unpack-archive PLOSComb.7z          # unpacks the parts pyMIRAW uses into data/original/
```

`data/original/` is gitignored. To keep the weights elsewhere, set
`PYMIRAW_ORIGINAL_WEIGHTS=/path/to/bestModel.bin`.

The weights are looked up in this order: `$PYMIRAW_ORIGINAL_WEIGHTS`,
`data/original/NetworkModel/bestModel.bin`, then the artefact mirror listed in
`data/manifest.yaml` (sha256 verified; requires credentials for the mirror).

## File formats

* **Unified file** (input of `predict` / `evaluate`): TSV with columns
  `miRNA, gene_name, EnsemblId, [Positive_Negative,] Mature_mirna_transcript, 3UTR_transcript`.
  An example is `tests/fixtures/paper_pairs.tsv`.
* **CV fold files**: the paper's binarized site tables (one-hot miRNA followed by the one-hot site,
  label last). Train files alternate labels 1,0,1,0,... which matters for training order
  (see `pymiraw train --no-shuffle`).
* **Outputs**: `targetPredictionOutput.csv` (per pair) and `positiveTargetSites.csv` (per site),
  written in the original column layout so old analysis scripts keep working.

## Caches and run output

* Folding results are memoised in `~/.cache/pymiraw/foldcache` (`--cache-dir`), keyed by sequences,
  constraint, backend and ViennaRNA version. It is safe to delete; the first run after that is slow.
* `runs/` holds benchmark and training output and is gitignored.

## Licences

The archive and weights are GPL-3.0, copyright the miRAW authors. Third-party sources behind the
paper's dataset (for example TarBase) have their own terms; pyMIRAW does not redistribute them.
