"""``pymiraw`` command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

app = typer.Typer(help="pyMIRAW: miRNA target prediction (Python port of miRAW).", no_args_is_help=True)
data_app = typer.Typer(help="Download / verify / upload heavy data on S3.", no_args_is_help=True)
model_app = typer.Typer(help="Model utilities.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")

ORIGINAL_HELP = "Use the original miRAW paper weights (fetched from S3 on first use)."


def _run(
    cfg_file,
    input_file,
    model,
    original_weights,
    finder,
    filters,
    output_dir,
    name,
    workers,
    backend,
    limit,
    evaluate,
    vienna_bin=None,
    sites_dir=None,
    require_sites=False,
):
    from pymiraw.config import PredictConfig, load_config, resolve_model_path
    from pymiraw.evaluation.metrics import gene_level_metrics
    from pymiraw.prediction.pipeline import predict_file, results_frame, sites_frame

    base = load_config(cfg_file).model_dump() if cfg_file else {}
    overrides = {
        "input_file": input_file,
        "model": model,
        "finder": finder,
        "output_dir": output_dir,
        "experiment_name": name,
        "workers": workers,
        "backend": backend,
        "filters": list(filters) if filters else None,
        "original_weights": True if original_weights else None,
        "vienna_bin": vienna_bin,
        "sites_dir": sites_dir,
        "require_sites": True if require_sites else None,
    }
    base.update({k: v for k, v in overrides.items() if v is not None})
    cfg = PredictConfig(**base)
    out = Path(cfg.output_dir) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = {
        "model": resolve_model_path(cfg.model),
        "finder": cfg.finder,
        "filters": cfg.filters,
        "backend": cfg.backend,
        "backend_args": {"bin_dir": cfg.vienna_bin} if cfg.vienna_bin else {},
        "max_site_length": cfg.max_site_length,
        "seed_alignment_offset": cfg.seed_alignment_offset,
        "cache_dir": cfg.cache_dir,
        "sites_dir": cfg.sites_dir,
        "require_sites": cfg.require_sites,
    }
    results = predict_file(cfg.input_file, run_cfg, cfg.workers, limit)
    df = results_frame(results)
    df.to_csv(out / "targetPredictionOutput.csv", sep="\t", index=False)
    sites_frame(results).to_csv(out / "targetSites.csv", sep="\t", index=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump()))
    typer.echo(f"{len(df)} pairs -> {out}  (weights: {'original miRAW' if cfg.uses_original_weights else cfg.model})")
    if evaluate and df["RealClass"].notna().all():
        m = gene_level_metrics(df["RealClass"].astype(int), df["Prediction"])
        (out / "metrics.json").write_text(json.dumps(m, indent=2))
        typer.echo(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


@app.command()
def predict(
    input_file: str = typer.Argument(
        None, help="Unified TSV (miRNA, gene_name, EnsemblId, Mature_mirna_transcript, 3UTR_transcript)."
    ),
    config: str = typer.Option(None, "--config", "-c", help="YAML or legacy .properties run config."),
    original_weights: bool = typer.Option(False, "--original-weights", help=ORIGINAL_HELP),
    model: str = typer.Option(None, help="Model artefact (.pt) or DL4J .bin. Default: original weights."),
    finder: str = typer.Option(None, help="miRAW | Personalized;1;9;7 | PITA | TS"),
    filter: list[str] = typer.Option(None, "--filter", help="e.g. AccessibilityEnergy;10 (repeatable)"),
    output_dir: str = typer.Option(None),
    name: str = typer.Option(None),
    workers: int = typer.Option(None),
    backend: str = typer.Option(None, help="bindings | cli"),
    limit: int = typer.Option(None, help="Only the first N pairs."),
    vienna_bin: str = typer.Option(
        None, "--vienna-bin", help="Folder with RNAcofold/RNAfold for --backend cli (e.g. the pymiraw-ref env bin)."
    ),
    sites_dir: str = typer.Option(
        None,
        "--sites-dir",
        help="Folder of saved candidate sites: reuse them (no ViennaRNA) and save newly computed ones.",
    ),
    require_sites: bool = typer.Option(
        False, "--require-sites", help="With --sites-dir: fail if a pair has no saved sites."
    ),
):
    """Predict miRNA-gene targets."""
    _run(
        config,
        input_file,
        model,
        original_weights,
        finder,
        filter,
        output_dir,
        name,
        workers,
        backend,
        limit,
        False,
        vienna_bin,
        sites_dir,
        require_sites,
    )


@app.command()
def evaluate(
    input_file: str = typer.Argument(None, help="Unified TSV including the Positive_Negative column."),
    config: str = typer.Option(None, "--config", "-c"),
    original_weights: bool = typer.Option(False, "--original-weights", help=ORIGINAL_HELP),
    model: str = typer.Option(None),
    finder: str = typer.Option(None),
    filter: list[str] = typer.Option(None, "--filter"),
    output_dir: str = typer.Option(None),
    name: str = typer.Option(None),
    workers: int = typer.Option(None),
    backend: str = typer.Option(None),
    limit: int = typer.Option(None),
    vienna_bin: str = typer.Option(
        None, "--vienna-bin", help="Folder with RNAcofold/RNAfold for --backend cli (e.g. the pymiraw-ref env bin)."
    ),
    sites_dir: str = typer.Option(
        None,
        "--sites-dir",
        help="Folder of saved candidate sites: reuse them (no ViennaRNA) and save newly computed ones.",
    ),
    require_sites: bool = typer.Option(
        False, "--require-sites", help="With --sites-dir: fail if a pair has no saved sites."
    ),
):
    """Predict and compare against the Positive_Negative labels (TP/FP/TN/FN, accuracy, F1...)."""
    _run(
        config,
        input_file,
        model,
        original_weights,
        finder,
        filter,
        output_dir,
        name,
        workers,
        backend,
        limit,
        True,
        vienna_bin,
        sites_dir,
        require_sites,
    )


@data_app.command("pull")
def data_pull(key: str = typer.Argument("original-weights"), force: bool = False):
    from pymiraw import storage

    typer.echo(storage.pull(key, force=force))


@data_app.command("unpack-archive")
def data_unpack(
    archive: str = typer.Argument(..., help="PLOSComb.7z from bitbucket.org/bipous/miraw_data"),
    dest: str = typer.Option(None),
):
    """Unpack the original weights, balanced test sets and CV folds into data/original/."""
    from pymiraw import storage

    out = storage.unpack_archive(archive, dest or storage.ORIGINAL_DIR)
    typer.echo(f"unpacked to {out}")


@data_app.command("verify")
def data_verify(key: str = typer.Argument("original-weights")):
    from pymiraw import storage

    ok = storage.verify(key)
    typer.echo(f"{key}: {'OK' if ok else 'missing or corrupt'}")
    raise typer.Exit(0 if ok else 1)


@model_app.command("import-dl4j")
def model_import(src: str, dest: str):
    """Convert a DL4J ``.bin`` into a pyMIRAW ``.pt`` artefact (keeps the old-default encoding)."""
    from pymiraw.models.artifact import load_artifact, save_artifact

    model, enc, meta = load_artifact(src)
    save_artifact(dest, model, enc, {**meta, "imported_from": Path(src).name})
    typer.echo(f"wrote {dest}")


@app.command()
def parity(
    test_dir: str = typer.Argument(..., help="Folder with randomLeveragedTestSplit_{0..9}.csv (paper balanced sets)."),
    output_dir: str = typer.Option("runs/parity"),
    original_weights: bool = typer.Option(True, "--original-weights/--model-path", help=ORIGINAL_HELP),
    model: str = typer.Option(None, help="Used only with --model-path."),
    sets: str = typer.Option("0-9", help="e.g. 0-9, 0, 0-2"),
    config_filter: str = typer.Option(None, "--configs", help="Substring filter, e.g. 'miRAW|NF'."),
    backend: str = typer.Option("bindings"),
    cache_dir: str = typer.Option(None),
    workers: int = typer.Option(None),
    sites_dir: str = typer.Option(None, "--sites-dir", help="Folder of saved candidate sites (reused and extended)."),
    tolerance: float = typer.Option(0.02, help="Allowed |mean diff| for accuracy and F1."),
    vienna_bin: str = typer.Option(
        None, "--vienna-bin", help="Folder with RNAcofold/RNAfold for --backend cli (e.g. the pymiraw-ref env bin)."
    ),
):
    """Gate 1: reproduce the paper's GlobalSummary with the original weights."""
    from pymiraw.config import resolve_model_path
    from pymiraw.evaluation.parity import CONFIGS, DISPLAY, compare_with_paper, run_parity

    lo, _, hi = sets.partition("-")
    ks = range(int(lo), int(hi or lo) + 1)
    cfgs = [c for c in CONFIGS if not config_filter or config_filter in f"{c[0]}|{c[1]}"]
    weights = resolve_model_path("original" if original_weights else model)
    ours = run_parity(
        test_dir,
        weights,
        output_dir,
        ks,
        cfgs,
        backend,
        cache_dir,
        workers,
        backend_args={"bin_dir": vienna_bin} if vienna_bin else None,
        sites_dir=sites_dir,
    )
    cmp = compare_with_paper(ours, tolerance)
    cmp.index = [f"{DISPLAY[i.split('|')[0]]} {i.split('|')[1]}" for i in cmp.index]
    typer.echo(cmp.round(3).to_string())
    cmp.to_csv(Path(output_dir) / "comparison.csv")
    raise typer.Exit(0 if cmp["ok"].all() else 1)


@app.command()
def train(
    folds_dir: str = typer.Argument(..., help="Folder with set_{k}_train.csv / set_{k}_test.csv (released CV folds)."),
    output_dir: str = typer.Option("runs/train"),
    orientation: str = typer.Option("old-default", help="old-default | natural"),
    loss: str = typer.Option("xent", help="xent | nll"),
    folds: str = typer.Option("0-8"),
    seeds: int = typer.Option(1, help="Number of seeds (0..n-1)."),
    max_epochs: int = typer.Option(10),
    patience: int = typer.Option(0),
    shuffle: bool = typer.Option(
        True, "--shuffle/--no-shuffle", help="--no-shuffle = file order every epoch (paper regime)."
    ),
    val_fraction: float = typer.Option(
        0.1, help="0 = train on all data, keep the lowest-training-loss epoch (paper regime)."
    ),
):
    """Cross-validate a model on the released folds and compare with the paper (Gate 2, site level)."""
    from pymiraw.training.cv import compare_with_paper, run_cv
    from pymiraw.training.trainer import TrainConfig

    lo, _, hi = folds.partition("-")
    cfg = TrainConfig(
        loss=loss,
        site_orientation=orientation,
        max_epochs=max_epochs,
        patience=patience,
        val_fraction=val_fraction,
        shuffle=shuffle,
    )
    df = run_cv(folds_dir, cfg, range(int(lo), int(hi or lo) + 1), range(seeds), output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / f"cv_{orientation}.csv", index=False)  # one file per orientation per output_dir
    typer.echo(df.round(4).to_string(index=False))
    typer.echo(json.dumps(compare_with_paper(df), indent=2, default=float))


if __name__ == "__main__":
    app()
