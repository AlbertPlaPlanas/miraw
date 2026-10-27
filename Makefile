# One-word entry points. Run inside the conda env:  conda activate pymiraw
ENV ?= pymiraw
PY  ?= python

.PHONY: env env-update install hooks test test-all lint parity train-cv data-pull data-verify

env:            ## create the conda environment
	conda env create -f environment.yml
env-update:
	conda env update -f environment.yml --prune
install:        ## editable install of the package (after `conda activate pymiraw`)
	$(PY) -m pip install -e . --no-deps
hooks:          ## install the pre-commit secret guards (once per clone)
	pre-commit install
test:           ## fast tests (what CI runs); tests needing the original weights skip if unavailable
	$(PY) -m pytest -m "not slow and not acceptance"
test-all:       ## everything (needs data/original, see README)
	$(PY) -m pytest
lint:
	ruff check src tests scripts && ruff format --check src tests scripts
data-pull:      ## unpack the original weights/data from PLOSComb.7z (see README)
	pymiraw data unpack-archive PLOSComb.7z
data-verify:
	pymiraw data verify original-weights
parity:         ## Gate 1: reproduce the paper's benchmark with the original weights (hours)
	pymiraw parity $(data/original/Data/TestData/balanced10) --output-dir runs/parity --cache-dir $$HOME/.cache/pymiraw/foldcache
train-cv:       ## site-level cross-validation on the released folds, both orientations
	for o in old-default natural; do pymiraw train $(data/original/Data/CrossVal/RelaxedPW60_14K_35K/binary) --orientation $$o --seeds 3 --output-dir runs/train_$$o; done
