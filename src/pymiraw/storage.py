"""Where the original weights are, and the manifest-driven download/upload of the heavy paper data from/to S3
(developer use; prediction never needs S3), with sha256 verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifest.yaml"
ORIGINAL_WEIGHTS_KEY = "original-weights"
ORIGINAL_WEIGHTS_ENV = "PYMIRAW_ORIGINAL_WEIGHTS"  # use another copy of the original weights (.pt or DL4J .bin)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKOUT = (REPO_ROOT / "pyproject.toml").is_file()  # running from a clone, not from an installed wheel
DATA_DIR_ENV = "PYMIRAW_DATA_DIR"  # where the paper data is unpacked (overrides the default below)


def _original_dir() -> Path:
    """The unpacked paper data: ``$PYMIRAW_DATA_DIR``, else ``data/original`` in a source checkout, else in the
    user cache."""
    if os.environ.get(DATA_DIR_ENV):
        return Path(os.environ[DATA_DIR_ENV])
    if SOURCE_CHECKOUT:
        return REPO_ROOT / "data" / "original"
    return Path(os.environ.get("PYMIRAW_CACHE", Path.home() / ".cache" / "pymiraw")) / "original"


ORIGINAL_DIR = _original_dir()  # where `pymiraw data unpack-archive` puts the paper data
ORIGINAL_DL4J = ORIGINAL_DIR / "NetworkModel" / "bestModel.bin"  # the DL4J source of the bundled weights
ARCHIVE_URL = "https://bitbucket.org/bipous/miraw_data/raw/master/PLOSComb.7z"
UNPACK_PATTERNS = (
    "PLOSComb/NetworkModel/bestModel.bin",
    "PLOSComb/Data/TestData/balanced10/",
    "PLOSComb/Data/CrossVal/RelaxedPW60_14K_35K/binary/",
    "PLOSComb/Results/GlobalSummary.xls",
)
UNPACK_SUFFIXES = (  # (prefix, suffix): only some files of these folders
    ("PLOSComb/Results/ResultsCV/VeryDeepXENT14K_e10/", "/bestModel.bin"),  # the paper's 9 CV fold models (Gate 2)
    ("PLOSComb/Results/Evaluate_14K_FULL/", "/targetPredictionOutput.csv"),  # paper results, full test set (Gate 3)
)


def cache_dir() -> Path:
    return Path(os.environ.get("PYMIRAW_CACHE", Path.home() / ".cache" / "pymiraw"))


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict:
    return yaml.safe_load(Path(path).read_text())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path(key: str, manifest: dict | None = None) -> Path:
    entry = (manifest or load_manifest())["files"][key]
    return cache_dir() / entry["path"]


def verify(key: str, manifest: dict | None = None) -> bool:
    manifest = manifest or load_manifest()
    p = local_path(key, manifest)
    return p.is_file() and sha256_file(p) == manifest["files"][key]["sha256"]


def pull(key: str, manifest: dict | None = None, force: bool = False) -> Path:
    """Download ``key`` into the cache (idempotent) and verify its checksum."""
    import fsspec

    manifest = manifest or load_manifest()
    entry = manifest["files"][key]
    dest = local_path(key, manifest)
    if not force and verify(key, manifest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    uri = f"{manifest['root'].rstrip('/')}/{entry['path']}"
    try:
        with fsspec.open(uri, "rb") as src, open(dest, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    except (PermissionError, OSError) as e:
        raise RuntimeError(
            f"cannot read {uri}: {e}\nOn the dev machine run the command through scripts/with-s3-creds.sh; "
            "elsewhere download PLOSComb.7z and run `pymiraw data unpack-archive`."
        ) from e
    if sha256_file(dest) != entry["sha256"]:
        dest.unlink()
        raise RuntimeError(f"checksum mismatch for {key}; download discarded")
    return dest


def push(key: str, src: str | Path, manifest: dict | None = None) -> str:
    import fsspec

    manifest = manifest or load_manifest()
    uri = f"{manifest['root'].rstrip('/')}/{manifest['files'][key]['path']}"
    with open(src, "rb") as f, fsspec.open(uri, "wb") as out:
        while chunk := f.read(1 << 20):
            out.write(chunk)
    return uri


def unpack_archive(archive: str | Path, dest: str | Path = ORIGINAL_DIR) -> Path:
    """Extract the parts of the paper archive (PLOSComb.7z) that pyMIRAW uses into ``dest``."""
    import py7zr

    with py7zr.SevenZipFile(archive) as z:
        targets = [
            i.filename
            for i in z.list()
            if i.filename.startswith(UNPACK_PATTERNS)
            or any(i.filename.startswith(a) and i.filename.endswith(b) for a, b in UNPACK_SUFFIXES)
        ]
    if not targets:
        raise ValueError(f"{archive} does not look like PLOSComb.7z")
    dest = Path(dest)
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=dest, targets=targets)
    for child in (dest / "PLOSComb").iterdir():  # drop the top-level folder: dest/NetworkModel, dest/Data, ...
        child.rename(dest / child.name)
    (dest / "PLOSComb").rmdir()
    return dest


def original_weights_path() -> Path:
    """Path of the original miRAW paper weights: ``$PYMIRAW_ORIGINAL_WEIGHTS`` if set, else the copy bundled
    with the package (``pymiraw/resources/miraw_original.pt``, converted bit-exactly from the paper's DL4J
    ``bestModel.bin``). Nothing is downloaded."""
    override = os.environ.get(ORIGINAL_WEIGHTS_ENV)
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"{ORIGINAL_WEIGHTS_ENV}={override} does not exist")
        return p
    from pymiraw import resources

    return resources.original_weights()
