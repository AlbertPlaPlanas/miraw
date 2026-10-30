"""Run configurations: prediction (YAML or the original ``.properties`` files) and training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ORIGINAL_MODEL_ALIASES = {"original", "original-weights", "miraw-paper-xent"}


class PredictConfig(BaseModel):
    """Everything a prediction run needs; the defaults are the paper's ``miRAW-6-1:10`` configuration.

    ``model`` is a path to a ``.pt`` artefact or a DL4J ``.bin``, or the literal ``original`` to
    use the original miRAW paper weights (``original_weights: true`` is the same switch).
    ``finder``: ``miRAW`` | ``Personalized;<start>;<size>;<min seed pairs>`` | ``PITA`` | ``TargetScan``.
    ``filters``: e.g. ``["AccessibilityEnergy;10"]``. ``backend``: ``bindings`` (ViennaRNA Python package)
    or ``cli`` (ViennaRNA command-line tools, in ``vienna_bin`` or on the PATH). Unknown keys are errors.
    ``experiment_name``, ``input_file`` and ``output_dir`` are only used by the command line.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "run"
    input_file: str | None = None
    output_dir: str = "runs"
    model: str = "original"
    original_weights: bool = False
    finder: str = "miRAW"
    filters: list[str] = Field(default_factory=list)
    max_site_length: int = 40
    seed_alignment_offset: int = 5
    backend: str = "bindings"
    vienna_bin: str | None = None  # folder with RNAcofold/RNAfold for backend "cli" (e.g. the 2.2.5 reference env)
    cache_dir: str | None = None
    sites_dir: str | None = None  # directory of saved candidate sites: load what exists, save what is computed
    require_sites: bool = False  # with sites_dir: fail instead of running ViennaRNA for pairs not saved
    workers: int | None = None

    @field_validator("input_file", "output_dir", "model", "vienna_bin", "cache_dir", "sites_dir", mode="before")
    @classmethod
    def _paths_as_text(cls, v):
        return os.fspath(v) if isinstance(v, os.PathLike) else v

    @model_validator(mode="after")
    def _original_flag(self):
        if self.original_weights:
            self.model = "original"
        return self

    @property
    def uses_original_weights(self) -> bool:
        return self.model in ORIGINAL_MODEL_ALIASES


def resolve_model_path(model: str | Path) -> str:
    """Turn a model reference (``original`` or a file path) into a loadable file path."""
    if str(model) in ORIGINAL_MODEL_ALIASES:
        from miraw.storage import original_weights_path

        return str(original_weights_path())
    if not Path(model).is_file():
        raise FileNotFoundError(f"model file not found: {model}")
    return str(model)


@dataclass
class TrainConfig:
    """Training settings. The defaults are the training stack's; the paper's regime is
    ``TrainConfig.paper()`` (10 epochs in file order, no validation split, best epoch by training loss)."""

    model: str = "mlp"
    model_params: dict = field(default_factory=dict)
    loss: str = "xent"
    lr: float = 0.005
    momentum: float = 0.9
    batch_size: int = 50
    max_epochs: int = 10
    patience: int = 0  # 0 = no early stopping
    val_fraction: float = 0.1
    shuffle: bool = True  # False = file order every epoch, as the original (DL4J) trained
    site_orientation: str = "old-default"
    seed: int = 0
    accelerator: str = "auto"  # Lightning: "auto" (a CUDA or Apple GPU if present, else CPU), "cpu", "gpu", "mps"

    @classmethod
    def paper(cls, **overrides) -> TrainConfig:
        """The regime of the paper's models (what Gate 2 tests): XENT, Nesterov SGD lr 0.005, batch 50."""
        return cls(**{"max_epochs": 10, "val_fraction": 0.0, "shuffle": False, **overrides})


def load_properties(path: str | Path) -> dict[str, str]:
    """Parse a Java ``.properties`` file (``key=value`` or ``key: value``)."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        for sep in ("=", ":"):
            if sep in line:
                k, v = line.split(sep, 1)
                out[k.strip()] = v.strip()
                break
    return out


def from_legacy_properties(path: str | Path, input_override: str | None = None) -> PredictConfig:
    """Read an original GenePrediction ``.properties`` file (Filter.0, Filter.1, ... supported)."""
    p = load_properties(path)
    filters, i = [], 0
    while f"Filter.{i}" in p:
        filters.append(p[f"Filter.{i}"])
        i += 1
    return PredictConfig(
        experiment_name=p.get("ExperimentName", "run"),
        input_file=input_override or p["UnifiedFile"],
        output_dir=p.get("ExperimentFolder", "runs"),
        model="original" if p.get("DLModel", "").endswith("bestModel.bin") else p.get("DLModel", "original"),
        finder=p.get("CandidateSiteFinder.Type", "Default"),
        filters=filters,
        max_site_length=int(p.get("MaxSiteLength", 30)),
        seed_alignment_offset=int(p.get("SeedAlignementOffset", 10)),
    )


def load_config(path: str | Path) -> PredictConfig:
    path = Path(path)
    if path.suffix == ".properties":
        return from_legacy_properties(path)
    return PredictConfig(**yaml.safe_load(path.read_text()))
