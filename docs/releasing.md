# Releasing miRAW on PyPI

Releases are built and published by [.github/workflows/publish.yml](../.github/workflows/publish.yml) on free
GitHub-hosted runners. PyPI *trusted publishing* is used, so no API token is stored anywhere.

## One-time setup

1. **PyPI**: sign in at <https://pypi.org>, open *Your account → Publishing* and add a **pending publisher**:
   PyPI project name `miraw`, owner `AlbertPlaPlanas`, repository `miraw`, workflow `publish.yml`,
   environment `pypi`.
2. **TestPyPI**: the same at <https://test.pypi.org> (a separate account), with environment `testpypi`.
3. **GitHub**: in the repository, *Settings → Environments*, create the environments `pypi` and `testpypi`.
   Adding yourself as a required reviewer on `pypi` makes every release wait for one click.

## Every release

1. Set the version in `src/miraw/__init__.py` (`__version__`, the only place it is written), add a log entry in
   `docs/PROGRESS.md`, and merge to `main`.
2. Optional dry run: *Actions → publish → Run workflow*. It builds the wheel and the sdist, runs the smoke test
   (`scripts/smoke_installed.py`: a clean install of the wheel on Linux, macOS and Windows, Python 3.11 and 3.13,
   predictions compared with known values, a short training) and uploads to TestPyPI. Check the page and
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ miraw`.
3. Tag and push: `git tag v0.3.0 && git push origin v0.3.0` (the tag must equal the version, the workflow checks it).
   The workflow builds, smoke-tests and publishes to PyPI.
4. Check <https://pypi.org/project/miraw/> and `pip install miraw` in a fresh environment.

A version can be uploaded to PyPI only once: to fix a release, publish a new version.

## What is in the package

* **Wheel** (about 1.8 MB): the `miraw` package, including `miraw/resources/` (the original weights, the paper's
  reference results and the example data).
* **sdist**: the package, the tests, `README.md`, `LICENSE`, `CITATION.cff`, `pyproject.toml`, and the two repository
  files the tests read (`data/manifest.yaml`, `scripts/check_secrets.py`). Not the notebooks, docs, CI or conda files.
* **PyPI page**: `README.md`, rewritten at build time by `hatch-fancy-pypi-readme` (see `pyproject.toml`): relative
  links and images become absolute GitHub URLs, internal anchors point at the GitHub page, and the Mermaid diagram,
  which PyPI cannot draw, becomes a link. Preview it with `python -m build` and `twine check --strict dist/*`.

To build and check locally: `pip install build twine "readme_renderer[md]"`, then `python -m build` and
`twine check --strict dist/*`.
