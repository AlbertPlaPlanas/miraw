"""Convert the paper's DL4J model (data/original/NetworkModel/bestModel.bin) into the bundled
``src/miraw/resources/miraw_original.pt`` and check the conversion is bit-exact.

Usage: python scripts/bundle_original_weights.py      (needs `miraw data unpack-archive PLOSComb.7z` first)
Then update ORIGINAL_WEIGHTS_SHA256 in src/miraw/resources/__init__.py with the printed value.
"""

import torch

from miraw import resources, storage
from miraw.models.artifact import load_artifact, save_artifact

src = storage.ORIGINAL_DL4J
if storage.sha256_file(src) != resources.SOURCE_DL4J_SHA256:
    raise SystemExit(f"{src} is not the paper's bestModel.bin (sha256 mismatch)")
model, encoder, _ = load_artifact(src)
meta = {
    "source": "dl4j",
    "file": "bestModel.bin",
    "source_sha256": resources.SOURCE_DL4J_SHA256,
    "description": "miRAW paper model (Pla, Zhong and Rayner, PLoS Comput Biol 2018), PLOSComb/NetworkModel",
}
dest = resources.original_weights()
save_artifact(dest, model, encoder, meta)

again, enc2, meta2 = load_artifact(dest)
a, b = model.state_dict(), again.state_dict()
assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "weights differ after conversion"
assert enc2.config() == encoder.config() and meta2 == meta
print(f"wrote {dest} ({dest.stat().st_size} bytes), sha256 {storage.sha256_file(dest)}")
