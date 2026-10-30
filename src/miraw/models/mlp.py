"""The miRAW multilayer perceptron: dense ReLU layers and a 2-way output layer."""

from __future__ import annotations

from torch import nn

from miraw.models.base import SiteClassifier, register_model

# Paper model (bestModel.bin): 280 -> 400-350-300-150-100-100-50-50 -> 2
PAPER_LAYER_SIZES = (280, 400, 350, 300, 150, 100, 100, 50, 50, 2)


@register_model("mlp")
class MirawMLP(SiteClassifier):
    def __init__(self, layer_sizes=PAPER_LAYER_SIZES, dropout: float = 0.0):
        super().__init__()
        sizes = list(layer_sizes)
        if len(sizes) < 3 or sizes[-1] != 2:
            raise ValueError("layer_sizes must have >= 3 entries and end in 2")
        self.layer_sizes = sizes
        self.dropout = dropout
        layers: list[nn.Module] = []
        for n_in, n_out in zip(sizes[:-2], sizes[1:-1], strict=True):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier (Glorot) normal weights and zero biases, as DL4J's ``WeightInit.XAVIER``."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def config(self) -> dict:
        return {"name": "mlp", "layer_sizes": self.layer_sizes, "dropout": self.dropout}
