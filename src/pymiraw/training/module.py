"""Lightning module that trains any registered :class:`SiteClassifier` (not MLP-specific)."""

from __future__ import annotations

import lightning as L
import torch

from pymiraw.models.base import SiteClassifier
from pymiraw.training.losses import get_loss


class SiteModule(L.LightningModule):
    def __init__(self, model: SiteClassifier, loss: str = "xent", lr: float = 0.005, momentum: float = 0.9):
        super().__init__()
        self.model, self.loss_name, self.lr, self.momentum = model, loss, lr, momentum
        self.loss_fn = get_loss(loss)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.loss_fn(logits, y), on_epoch=True, prog_bar=False)
        self.log("val_acc", (logits.argmax(1) == y).float().mean(), on_epoch=True)

    def configure_optimizers(self):
        # Nesterov momentum SGD, as the original (DL4J Updater.NESTEROVS, momentum 0.9)
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True)
