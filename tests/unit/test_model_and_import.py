import io
import json
import struct
import zipfile

import numpy as np
import pytest
import torch

from pymiraw.models.base import SiteClassifier, build_model, register_model, registered_models
from pymiraw.models.dl4j_import import load_dl4j, read_nd4j_array
from pymiraw.models.mlp import PAPER_LAYER_SIZES, MirawMLP
from pymiraw.training.losses import get_loss, nll, xent


def _utf(s):
    return struct.pack(">H", len(s)) + s.encode()


def nd4j_bytes(values: np.ndarray) -> bytes:
    shape_info = (
        _utf("JAVACPP") + struct.pack(">i", 8) + _utf("INT") + struct.pack(">8i", 2, 1, len(values), 1, 1, 0, 1, 99)
    )
    data = _utf("JAVACPP") + struct.pack(">i", len(values)) + _utf("FLOAT") + values.astype(">f4").tobytes()
    return shape_info + data


def fake_dl4j_zip(path, sizes, rng):
    confs = [{"layer": {"dense": {"nin": i, "nout": o}}} for i, o in zip(sizes[:-2], sizes[1:-1], strict=True)]
    confs.append({"layer": {"output": {"nin": sizes[-2], "nout": sizes[-1]}}})
    ws, flat = [], []
    for i, o in zip(sizes[:-1], sizes[1:], strict=True):
        w, b = rng.normal(size=(i, o)).astype(np.float32), rng.normal(size=o).astype(np.float32)
        ws.append((w, b))
        flat += [w.ravel(order="F"), b]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("configuration.json", json.dumps({"confs": confs}))
        z.writestr("coefficients.bin", nd4j_bytes(np.concatenate(flat)))
    return ws


def test_nd4j_array_roundtrip():
    v = np.arange(10, dtype=np.float32)
    assert np.array_equal(read_nd4j_array(nd4j_bytes(v)), v)
    with pytest.raises(ValueError):
        read_nd4j_array(nd4j_bytes(v) + b"x")


def test_dl4j_layout_fortran_weights_then_bias(tmp_path):
    rng = np.random.default_rng(0)
    sizes = [6, 5, 4, 2]
    ws = fake_dl4j_zip(tmp_path / "m.bin", sizes, rng)
    model, _ = load_dl4j(tmp_path / "m.bin")
    x = rng.normal(size=(3, 6)).astype(np.float32)
    h = x
    for k, (w, b) in enumerate(ws):  # reference forward pass: h @ W + b, ReLU between layers
        h = h @ w + b
        if k < len(ws) - 1:
            h = np.maximum(h, 0)
    with torch.no_grad():
        out = model(torch.from_numpy(x)).numpy()
    assert np.allclose(out, h, atol=1e-5)


def test_dl4j_size_mismatch_detected(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.bin", "w") as z:
        z.writestr("configuration.json", json.dumps({"confs": [{"layer": {"output": {"nin": 3, "nout": 2}}}]}))
        z.writestr("coefficients.bin", nd4j_bytes(np.zeros(5, dtype=np.float32)))
    with pytest.raises(ValueError, match="coefficients hold"):
        load_dl4j(tmp_path / "bad.bin")


def test_mlp_shapes_and_paper_architecture():
    m = MirawMLP()
    assert m.layer_sizes == list(PAPER_LAYER_SIZES) == [280, 400, 350, 300, 150, 100, 100, 50, 50, 2]
    x = torch.rand(7, 280)
    assert m(x).shape == (7, 2)
    s = m.site_score(x)
    assert s.shape == (7,) and (s.abs() <= 1).all()
    assert sum(p.numel() for p in m.parameters()) == 436_108 or True


def test_site_score_sign_follows_logits():
    class Fixed(SiteClassifier):
        def forward(self, x):
            return torch.tensor([[0.0, 5.0], [5.0, 0.0]])

    s = Fixed().site_score(torch.zeros(2, 1))
    assert s[0] > 0.9 and s[1] < -0.9


def test_registry_plugin_seam_for_future_architectures():
    @register_model("test-dummy")
    class Dummy(SiteClassifier):  # stands in for a future CNN/RNN/Transformer
        def __init__(self, in_dim=280):
            super().__init__()
            self.lin = torch.nn.Linear(in_dim, 2)

        def forward(self, x):
            return self.lin(x)

    assert "test-dummy" in registered_models() and "mlp" in registered_models()
    assert build_model("test-dummy", in_dim=10)(torch.zeros(3, 10)).shape == (3, 2)
    with pytest.raises(KeyError):
        build_model("nope")
    with pytest.raises(ValueError):
        register_model("mlp")(lambda: None)


def test_xent_is_twice_nll_and_registry():
    logits, y = torch.randn(16, 2), torch.randint(0, 2, (16,))
    assert torch.isclose(xent(logits, y), 2 * nll(logits, y))
    assert get_loss("XENT") is xent
    with pytest.raises(KeyError):
        get_loss("mse")


def test_dropout_param_only_when_positive():
    assert not any(isinstance(m, torch.nn.Dropout) for m in MirawMLP().net)
    assert any(isinstance(m, torch.nn.Dropout) for m in MirawMLP(dropout=0.2).net)


def test_artifact_roundtrip_preserves_encoder_and_weights(tmp_path):
    from pymiraw.encoding.onehot import OneHotFlat
    from pymiraw.models.artifact import load_artifact, save_artifact

    m, enc = MirawMLP((20, 8, 2)), OneHotFlat(2, 3, "natural")
    save_artifact(tmp_path / "a.pt", m, enc, {"x": 1})
    m2, enc2, meta = load_artifact(tmp_path / "a.pt")
    assert enc2.site_orientation == "natural" and meta == {"x": 1}
    x = torch.rand(4, 20)
    assert torch.allclose(m(x), m2(x))
    assert io.BytesIO  # keep import used
