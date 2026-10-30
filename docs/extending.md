# Adding a new architecture (CNN, RNN, Transformer, ...)

miRAW separates **what goes in** (an encoder), **what predicts** (a model) and **how it is trained**
(the Lightning module). A new architecture is one class and one registration; the pipeline, training,
cross-validation, artefact format and CLI do not change.

## 1. Write the model

A model maps a batch of encoded (miRNA, site) pairs to logits `(B, 2)` (class 1 = target). Subclass
`SiteClassifier`, register it by name, and return its constructor arguments from `config()` so it can be
saved and rebuilt:

```python
import torch
from miraw import SiteClassifier, register_model


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

The registration must have run before a saved model of this type is loaded. For a model that is part of
miRAW, put it in `src/miraw/models/` and import it from `src/miraw/models/__init__.py` (without that
import the registry is empty; this bit us once). In your own code, import the module that defines it first.

Prediction runs in worker processes, and each one rebuilds the model from its saved file. Each worker
imports the module that defines the model's class, so a class in any importable module works in parallel
everywhere. A class defined in a notebook cell or a script's `__main__` cannot be imported. On Linux it still
works, because the workers are forked copies of the notebook. Where workers are spawned (macOS, Windows),
miRAW predicts in the notebook's own process instead, with a warning: the results are the same, only slower.

## 2. Train it

```python
import miraw

model = miraw.train(train_sites, test_sites, model="conv1d", model_params={"channels": 32})
model.meta["metrics"]               # test-set metrics
model.save("conv1d.pt")             # then: miraw.predict(pairs, "conv1d.pt")
```

`train_sites` is a DataFrame (`mirna_seq`, `site_seq`, `label`) or encoded `(x, y)` arrays. The other
`TrainConfig` fields keep the paper's regime unless overridden (`miraw.TrainConfig.paper()`).
Cross-validation over the released folds: `miraw.cross_validate(folds_dir, miraw.TrainConfig.paper(model="conv1d"))`.
Working examples: the tutorial notebook (section 3.5) and
`tests/functional/test_training_smoke.py::test_new_architecture_trains_through_the_same_stack`.

## 3. Different input encoding

`OneHotFlat` produces the flat `[miRNA | site]` vector the MLP expects. A position-aware model usually
wants a `(length, 4)` sequence instead. Add an encoder next to `src/miraw/encoding/onehot.py` (the
building block `one_hot(seq, max_len)` already returns `(max_len, 4)`), register it in
`models/artifact.py:build_encoder`, and make sure `config()` carries everything needed to rebuild it.
The encoder is saved **inside** every artefact, so a model can never be run with the wrong encoding.

**Site orientation.** `site_orientation="old-default"` (reversed, as the original miRAW) or `"natural"`.
For the MLP the choice is only a permutation of input features. A position-aware model can behave
differently, so **rerun the orientation comparison** (`miraw train --orientation ...`, then
`scripts/gate2_report.py`) when adding one. The decision rule is in `docs/PLAN.md` (decision 8).

## 4. Check it against the paper

A model is only a candidate to replace the original weights if it passes Gate 2
(`docs/PLAN.md`, section 6): one-sided 99 % non-inferiority to the paper's models at site level and at
gene level, with margins 0.02 (accuracy, F1) and 0.005 (AUC). Train in the paper regime first (`TrainConfig.paper()`,
the default of `miraw.train` and `miraw train`) for a like-for-like comparison, and use `make gate2`.
Run `miraw predict --model your_model.pt` to use it; the default stays `--original-weights`.
