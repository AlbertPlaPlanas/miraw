# One-word entry points. Run inside the conda env:  conda activate miraw
ENV ?= miraw
PY  ?= python

.PHONY: env env-update install hooks test test-all lint parity train-cv data-pull data-verify gate2 verify-gates

TEST_SETS ?= data/original/Data/TestData/balanced10
CV_FOLDS  ?= data/original/Data/CrossVal/RelaxedPW60_14K_35K/binary

env:            ## create the conda environment
	conda env create -f environment.yml
env-update:
	conda env update -f environment.yml --prune
install:        ## editable install of the package (after `conda activate miraw`)
	$(PY) -m pip install -e . --no-deps
hooks:          ## install the pre-commit secret guards (once per clone)
	pre-commit install
test:           ## fast tests (what CI runs); tests needing data/original skip if it is absent
	$(PY) -m pytest -m "not slow and not acceptance"
test-all:       ## everything (needs data/original, see README)
	$(PY) -m pytest
lint:
	ruff check src tests scripts && ruff format --check src tests scripts
data-pull:      ## unpack the original weights/data from PLOSComb.7z (see README)
	miraw data unpack-archive PLOSComb.7z
data-verify:
	miraw data verify original-weights
gate2:          ## Gate 2 non-inferiority report from the committed results
	python scripts/gate2_report.py
parity:         ## Gate 1: reproduce the paper's benchmark with the original weights (hours)
	miraw parity $(TEST_SETS) --output-dir runs/parity --cache-dir $$HOME/.cache/miraw/foldcache
train-cv:       ## site-level cross-validation on the released folds, both orientations (paper regime)
	for o in old-default natural; do miraw train $(CV_FOLDS) --orientation $$o --seeds 3 --output-dir runs/train_$$o; done
verify-gates:   ## re-run Gates 0-2 from scratch and diff every number with docs/results/ (hours)
	python scripts/verify_gates.py
