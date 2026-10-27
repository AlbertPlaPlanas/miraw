"""Importing this package registers the built-in architectures."""

from pymiraw.models import mlp  # noqa: F401  (registers "mlp")
from pymiraw.models.base import SiteClassifier, build_model, register_model, registered_models

__all__ = ["SiteClassifier", "build_model", "register_model", "registered_models"]
