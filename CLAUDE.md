# Instructions for Claude Code sessions in this repo

pyMIRAW is a Python/PyTorch port of miRAW. Work spans many sessions and machines, so the state of
the work lives in git, not in anyone's memory.

1. **At the start of every session read `docs/PROGRESS.md` and `docs/PLAN.md`** (in that order).
   PROGRESS.md says what is done, what is running, what is next and what is blocked; PLAN.md holds
   the design, the resolved decisions and the acceptance gates.
2. **Keep `docs/PROGRESS.md` current as you work**: update it after every meaningful step (feature
   finished, test/benchmark result, decision, blocker), and always before ending a session. Record
   numbers and commands, not impressions. Add newest entries at the top of the "Log" section.
3. Never put credentials in any file. S3 access: `scripts/with-s3-creds.sh <cmd>` (reads the
   gitignored `.secrets.md`). Run `pre-commit install` once per clone.
4. Use the conda env `pymiraw` (`conda env create -f environment.yml`, then `pip install -e . --no-deps`).
5. The original weights flag is `--original-weights` (default). Do not change a default model
   without passing the gates in PLAN.md section 6.
