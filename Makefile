# One-word entry points. Run inside the conda env:  conda activate pymiraw
ENV ?= pymiraw
PY  ?= python
S3  := scripts/with-s3-creds.sh

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
test-all:       ## everything, fetching the original weights from S3 on the dev box
	$(S) $(PY) -m pytest
lint:
	ruff check src tests scripts && ruff format --check src tests scripts
data-pull:      ## download the original miRAW weights (S3)
	$(S) pymiraw data pull original-weights
data-verify:
	pymiraw data verify original-weights
parity:         ## Gate 1: reproduce the paper's benchmark with the original weights (hours)
	$(S) pymiraw parity $(TEST_DIR) --output-dir runs/parity --cache-dir $$HOME/.cache/pymiraw/foldcache
train-cv:       ## site-level cross-validation on the released folds, both orientations
	for o in old-default natural; do pymiraw train $(FOLDS_DIR) --orientation $$o --seeds 3 --output-dir runs/train_$$o; done
