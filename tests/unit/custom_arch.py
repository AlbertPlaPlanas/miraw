"""A custom architecture in an importable module (used by test_api's spawned-worker test)."""

import torch

from miraw import SiteClassifier, register_model


@register_model("test-spawn-linear")
class SpawnLinear(SiteClassifier):
    """A tiny trainable network: one linear layer over the 280 encoded inputs."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(280, 2)

    def forward(self, x):
        return self.lin(x)

    def config(self):
        return {"name": "test-spawn-linear"}
