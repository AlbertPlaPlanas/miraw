"""Import models saved by DL4J (``ModelSerializer`` zip files) into PyTorch, without Java.

A DL4J zip holds ``configuration.json`` (layer sizes) and ``coefficients.bin`` (all parameters
as one flat ND4J array). For dense layers the flat array is, layer after layer, ``W`` (shape
``nIn x nOut``, Fortran order) followed by ``b`` (``nOut``).
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

from miraw.models.mlp import MirawMLP

_DTYPES = {"FLOAT": ">f4", "DOUBLE": ">f8", "INT": ">i4", "LONG": ">i8", "HALF": ">f2"}


def _read_utf(f: io.BytesIO) -> str:
    (n,) = struct.unpack(">H", f.read(2))
    return f.read(n).decode()


def _read_buffer(f: io.BytesIO) -> np.ndarray:
    alloc = _read_utf(f)
    if alloc in ("HEAP", "DIRECT", "JAVACPP"):
        (length,) = struct.unpack(">i", f.read(4))
    else:
        (length,) = struct.unpack(">q", f.read(8))
    dtype = _read_utf(f)
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported ND4J data type {dtype!r}")
    dt = np.dtype(_DTYPES[dtype])
    return np.frombuffer(f.read(length * dt.itemsize), dtype=dt)


def read_nd4j_array(data: bytes) -> np.ndarray:
    """Parse a serialized ND4J array (shape-info buffer followed by data buffer) to 1-D float32."""
    f = io.BytesIO(data)
    _read_buffer(f)  # shape info (rank, shape, strides, ...)
    values = _read_buffer(f)
    if f.read(1):
        raise ValueError("trailing bytes after ND4J array")
    return values.astype(np.float32)


def layer_sizes_from_config(config: dict) -> list[int]:
    sizes: list[int] = []
    for conf in config["confs"]:
        (layer,) = conf["layer"].values()
        n_in, n_out = int(layer["nin"]), int(layer["nout"])
        if sizes and sizes[-1] != n_in:
            raise ValueError("non-sequential layer sizes in DL4J configuration")
        if not sizes:
            sizes.append(n_in)
        sizes.append(n_out)
    return sizes


def load_dl4j(path: str | Path) -> tuple[MirawMLP, dict]:
    """Load a DL4J dense network. Returns the PyTorch model and the DL4J configuration."""
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("configuration.json"))
        flat = read_nd4j_array(z.read("coefficients.bin"))
    sizes = layer_sizes_from_config(config)
    expected = sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:], strict=True))
    if flat.size != expected:
        raise ValueError(f"coefficients hold {flat.size} values, configuration needs {expected}")
    model = MirawMLP(sizes)
    state, pos = {}, 0
    linear_idx = [i for i, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)]
    for idx, (n_in, n_out) in zip(linear_idx, zip(sizes[:-1], sizes[1:], strict=True), strict=True):
        w = flat[pos : pos + n_in * n_out].reshape((n_in, n_out), order="F")
        pos += n_in * n_out
        b = flat[pos : pos + n_out]
        pos += n_out
        state[f"net.{idx}.weight"] = torch.from_numpy(np.ascontiguousarray(w.T))
        state[f"net.{idx}.bias"] = torch.from_numpy(b.copy())
    model.load_state_dict(state)
    return model.eval(), config
