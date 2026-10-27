# Adding a new architecture (CNN, RNN, Transformer, ...)

pyMIRAW separates **what goes in** (an encoder), **what predicts** (a model) and **how it is trained**
(the Lightning module). A new architecture is one class and one registration; the pipeline, training,
cross-validation, artefact format and CLI do not change.

## 1. Write the model

A model maps a batch of encoded (miRNA, site) pairs to logits `(B, 2)` (class 1 = target). Subclass
`SiteClassifier`, register it by name, and return its constructor arguments from `config()` so it can be
saved and rebuilt:

```python
import torch
from pymiraw.models.base import SiteClassifier, register_model


@register_model("conv1d")
class Conv1D(SiteClassifier):
    def __init__(self, in_dim: int = 280, channels: int = 16):
        super().__init__()
        self.in_dim, self.channels = in_dim, channels
        self.conv = torch.nn.Conv1d(1, channels, kernel_size=8, stride=4)
        self.head = torch.nn.Linear(channels * ((in_dim - 8) // 4 + 1), 2)

    def forward(self, x):                      # x: (B, in_dim)
        return self.head(torch.relu(self.conv(x.unsqueeze(1))).flatten(1))

    def config(self) -> dict:                  # stored in the artefact; rebuilds the model
        return {"name": "conv1d", "in_dim": self.in_dim, "channels": self.channels}
```

Put it in `src/pymiraw/models/` and import it from `src/pymiraw/models/__init__.py`. Without that import
the registry does not know about it (this bit us once: an empty registry).

## 2. Train it

```python
from pymiraw.training.trainer import TrainConfig, train_fold
cfg = TrainConfig(model="conv1d", model_params={"channels": 32}, loss="xent",
                  site_orientation="natural", val_fraction=0.0, shuffle=False, max_epochs=10)
net, metrics, encoder = train_fold(cfg, x_train, y_train, x_test, y_test, workdir)
```

`tests/functional/test_training_smoke.py::test_new_architecture_trains_through_the_same_stack` is a
working example. Cross-validation over the released folds is `pymiraw.training.cv.run_cv`.

## 3. Different input encoding

`OneHotFlat` produces the flat `[miRNA | site]` vector the MLP expects. A position-aware model usually
wants a `(length, 4)` sequence instead. Add an encoder next to `src/pymiraw/encoding/onehot.py` (the
building block `one_hot(seq, max_len)` already returns `(max_len, 4)`), register it in
`models/artifact.py:build_encoder`, and make sure `config()` carries everything needed to rebuild it.
The encoder is saved **inside** every artefact, so a model can never be run with the wrong encoding.

**Site orientation.** `site_orientation="old-default"` (reversed, as the original miRAW) or `"natural"`.
For the MLP the choice is only a permutation of input features. A position-aware model can behave
differently, so **rerun the orientation comparison** (`pymiraw train --orientation ...`, then
`scripts/gate2_report.py`) when adding one. The decision rule is in `docs/PLAN.md` (decision 8).

## 4. Check it against the paper

A model is only a candidate to replace the original weights if it passes Gate 2
(`docs/PLAN.md`, section 6): one-sided 99 % non-inferiority to the paper's models at site level and at
gene level, with margins 0.01 (accuracy, F1) and 0.005 (AUC). Train in the paper regime
(`--no-shuffle --val-fraction 0`) first for a like-for-like comparison, and use `make gate2`.
Run `pymiraw predict --model your_model.pt` to use it; the default stays `--original-weights`.
