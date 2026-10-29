# Data

Predicting needs no data beyond the package: the original weights and small examples are bundled. The heavy paper
data is only needed for the benchmarks and full-size training, and is never stored in git.

| What | Where it comes from | Used by |
|---|---|---|
| Original miRAW weights | bundled: `src/pymiraw/resources/miraw_original.pt` (1.75 MB, a bit-exact conversion of `PLOSComb.7z` `NetworkModel/bestModel.bin`) | `model="original"` / `--original-weights` (the default) |
| Example pairs and training sites | bundled: `example_pairs.tsv` (24 pairs from balanced test set 0), `example_sites.tsv.gz` (5,000 sites from CV fold 6), built by `scripts/make_example_data.py` | `pymiraw.example_pairs()`, `pymiraw.example_sites()`, tutorial, tests |
| Published results | bundled: `paper_global_summary.csv` (from `Results/GlobalSummary.xls`), `paper_cv_xent_models_rescored.csv` | Gate 1 and Gate 2 comparisons |
| Paper test sets, CV folds, the paper's fold models | `PLOSComb.7z` (`Data/TestData/balanced10`, `Data/CrossVal/RelaxedPW60_14K_35K/binary`, `ResultsCV/`) | `pymiraw parity`, `pymiraw train`, `scripts/verify_gates.py` |

## Getting the data

The archive is published by the miRAW authors at
<https://bitbucket.org/bipous/miraw_data> (about 1 GB):

```bash
curl -L -o PLOSComb.7z https://bitbucket.org/bipous/miraw_data/raw/master/PLOSComb.7z
pymiraw data unpack-archive PLOSComb.7z          # unpacks the parts pyMIRAW uses into data/original/
```

`data/original/` is gitignored.

The original weights are `$PYMIRAW_ORIGINAL_WEIGHTS` if set (a `.pt` or DL4J `.bin` file), else the bundled copy;
nothing is downloaded. The project's S3 mirror (`data/manifest.yaml`, sha256 verified, credentials required) is only an
archive of record for the maintainers.

## File formats

* **Unified file** (input of `predict` / `evaluate`): TSV with columns
  `miRNA, gene_name, EnsemblId, [Positive_Negative,] Mature_mirna_transcript, 3UTR_transcript`.
  An example is `src/pymiraw/resources/example_pairs.tsv`. The Python API also takes a pandas DataFrame
  (`mirna_seq`, `utr`, optional `miRNA`, `gene_name`, `EnsemblId`, `label`) or `(mirna_seq, utr)` tuples.
* **CV fold files**: the paper's binarized site tables, headerless CSV: the label (1/0) first, then the one-hot miRNA
  (30 nt x 4) and the one-hot site (40 nt x 4, stored 3'->5' as the original encoded it). Train files alternate labels
  1,0,1,0,... which matters for training order (the paper regime trains in file order: `TrainConfig.paper()`, the
  default of `pymiraw.train` and `pymiraw train`).
* **Outputs**: `targetPredictionOutput.csv` (per pair) and `targetSites.csv` (per candidate site), plus `config.yaml`
  and, for `evaluate`, `metrics.json`. The per-pair file keeps the original column layout.

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

The miRAW code, weights and archive are released under the MIT licence by their author (see `LICENSE`). pyMIRAW bundles
the weights and two small samples of the archive (24 test pairs and 5,000 training sites: sequences and labels). The
third-party sources behind the paper's dataset (for example TarBase) have their own terms; pyMIRAW does not
redistribute those databases.
