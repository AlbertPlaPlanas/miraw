"""Training smoke tests on tiny synthetic data (CPU, seconds): learning, checkpoint round-trip and
the plug-in seam for new architectures."""

import numpy as np
import pytest
import torch

from miraw.models.artifact import load_artifact, save_artifact
from miraw.models.base import SiteClassifier, register_model
from miraw.training.trainer import CollapsedModelError, TrainConfig, train_fold

torch.set_num_threads(2)  # tiny models: avoid thread oversubscription on busy or shared machines
IN_DIM = 280  # 30 nt miRNA + 40 nt site, one-hot


def make_data(n, seed):
    """Separable toy problem: the label is whether a few input features are on."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, IN_DIM)) < 0.25).astype(np.float32)
    y = (x[:, :8].sum(1) > 2).astype(np.int64)
    return x, y


def cfg(**kw):
    base = dict(
        model="mlp", model_params={"layer_sizes": (IN_DIM, 32, 16, 2)}, max_epochs=8, batch_size=50,
        accelerator="cpu", val_fraction=0.0, shuffle=False, seed=0,
    )  # fmt: skip
    return TrainConfig(**{**base, **kw})


@pytest.fixture(scope="module")
def data():
    xtr, ytr = make_data(1500, 0)
    xte, yte = make_data(500, 1)
    return xtr, ytr, xte, yte


def test_training_learns_and_is_reproducible(data, tmp_path):
    xtr, ytr, xte, yte = data
    _, m1, _ = train_fold(cfg(), xtr, ytr, xte, yte, tmp_path / "a")
    _, m2, _ = train_fold(cfg(), xtr, ytr, xte, yte, tmp_path / "b")
    assert m1["Accuracy"] > 0.85 and m1["AUC"] > 0.9
    assert m1["Accuracy"] == m2["Accuracy"] and m1["AUC"] == pytest.approx(m2["AUC"])  # same seed, same run


def test_validation_mode_checkpoints_on_val_loss(data, tmp_path):
    xtr, ytr, xte, yte = data
    _, m, _ = train_fold(cfg(val_fraction=0.2, shuffle=True), xtr, ytr, xte, yte, tmp_path)
    assert m["Accuracy"] > 0.85


def test_natural_orientation_is_recorded_in_the_artifact(data, tmp_path):
    xtr, ytr, xte, yte = data
    net, _, enc = train_fold(cfg(site_orientation="natural", max_epochs=2), xtr, ytr, xte, yte, tmp_path)
    save_artifact(tmp_path / "m.pt", net, enc, {"k": 1})
    net2, enc2, meta = load_artifact(tmp_path / "m.pt")
    assert enc2.site_orientation == "natural" and enc2.site_len == 40 and meta == {"k": 1}
    x = torch.from_numpy(xte[:20])
    assert torch.allclose(net.site_score(x), net2.site_score(x))  # checkpoint round-trip is exact


def test_constant_labels_are_rejected_as_collapsed(data, tmp_path):
    xtr, _, xte, _ = data
    ones_tr, ones_te = np.ones(len(xtr), np.int64), np.ones(len(xte), np.int64)
    with pytest.raises(CollapsedModelError):
        train_fold(cfg(max_epochs=3), xtr, ones_tr, xte, ones_te, tmp_path)


def test_new_architecture_trains_through_the_same_stack(data, tmp_path):
    """The extension seam: a registered model with a different structure needs no pipeline changes."""

    @register_model("test-conv")
    class TinyConv(SiteClassifier):
        def __init__(self, in_dim=IN_DIM):
            super().__init__()
            self.conv = torch.nn.Conv1d(1, 4, kernel_size=8, stride=4)
            self.head = torch.nn.Linear(4 * ((in_dim - 8) // 4 + 1), 2)

        def forward(self, x):
            return self.head(torch.relu(self.conv(x.unsqueeze(1))).flatten(1))

        def config(self):
            return {"name": "test-conv"}

    xtr, ytr, xte, yte = data
    net, m, enc = train_fold(cfg(model="test-conv", model_params={}, max_epochs=8), xtr, ytr, xte, yte, tmp_path)
    assert isinstance(net, TinyConv) and m["Accuracy"] > 0.8
    save_artifact(tmp_path / "c.pt", net, enc)
    net2, _, _ = load_artifact(tmp_path / "c.pt")
    assert isinstance(net2, TinyConv)
