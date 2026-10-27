"""Manifest-driven download/upload of heavy artefacts from/to S3, with sha256 verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifest.yaml"
ORIGINAL_WEIGHTS_KEY = "original-weights"
ORIGINAL_WEIGHTS_ENV = "PYMIRAW_ORIGINAL_WEIGHTS"  # point at a local copy to skip S3 entirely


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
            f"cannot read {uri}: {e}\nOn the dev machine run the command through "
            "scripts/with-s3-creds.sh, or set PYMIRAW_ORIGINAL_WEIGHTS to a local copy."
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


def original_weights_path() -> Path:
    """Path of the original miRAW paper weights (DL4J ``bestModel.bin``), fetched on first use."""
    override = os.environ.get(ORIGINAL_WEIGHTS_ENV)
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"{ORIGINAL_WEIGHTS_ENV}={override} does not exist")
        return p
    return pull(ORIGINAL_WEIGHTS_KEY)
