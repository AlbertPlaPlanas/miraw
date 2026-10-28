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
* **Saved candidate sites** (`--sites-dir DIR`): the binding sites found with ViennaRNA are saved per pair as
  parquet shards in `DIR` and loaded on later runs, so only the network and filters run again. Use it on
  `predict`, `evaluate` and `parity`, or `sites_dir:` in a YAML config:

  ```bash
  pymiraw evaluate pairs.tsv --finder miRAW --sites-dir sites/   # first run: folds and saves
  pymiraw evaluate pairs.tsv --finder miRAW --sites-dir sites/ --model my_model.pt   # no ViennaRNA
  pymiraw evaluate pairs.tsv --sites-dir sites/ --require-sites  # error if a pair is not saved
  ```

  A pair is reused only if the miRNA, the 3'UTR, all finder settings and the folding backend and version
  are identical (they are hashed into its key); otherwise it is computed and added as a new shard. The
  directory can be copied between machines. The accessibility energies ΔG_open are saved too (`opening-*.parquet`; they depend only on the UTR and
  the site position, so they are shared by all finders, models and thresholds). Model scores are not stored. Unlike `--cache-dir`, which memoises raw window
  folds, this stores the final candidate sites and is independent of the folding library at load time.
* `runs/` holds benchmark and training output and is gitignored.

## Licences

The miRAW code, weights and archive are released under the MIT licence by their author (see `LICENSE`). Third-party sources behind the
paper's dataset (for example TarBase) have their own terms; pyMIRAW does not redistribute them.
