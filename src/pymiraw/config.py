"""Run configuration (YAML or the original ``.properties`` files) and model resolution."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

ORIGINAL_MODEL_ALIASES = {"original", "original-weights", "miraw-paper-xent"}


class PredictConfig(BaseModel):
    """What the ``predict`` / ``evaluate`` commands need.

    ``model`` is a path to a ``.pt`` artefact or a DL4J ``.bin``, or the literal ``original`` to
    use the original miRAW paper weights (``original_weights: true`` is the same switch).
    """

    experiment_name: str = "run"
    input_file: str
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
    workers: int | None = None

    @model_validator(mode="after")
    def _original_flag(self):
        if self.original_weights:
            self.model = "original"
        return self

    @property
    def uses_original_weights(self) -> bool:
        return self.model in ORIGINAL_MODEL_ALIASES


def resolve_model_path(model: str) -> str:
    """Turn a config ``model`` value into a loadable file path (fetching original weights)."""
    if model in ORIGINAL_MODEL_ALIASES:
        from pymiraw.storage import original_weights_path

        return str(original_weights_path())
    if not Path(model).is_file():
        raise FileNotFoundError(f"model file not found: {model}")
    return model


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
