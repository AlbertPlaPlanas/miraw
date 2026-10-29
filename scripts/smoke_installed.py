"""Smoke test of an installed pyMIRAW (a wheel in a clean environment, as the publish workflow does).

Checks that the bundled weights load, that predictions equal known values (the same on every platform), that
parallel workers run (spawned on macOS and Windows) and that training works. Run it outside the source tree,
or anywhere with the src/ layout: `python scripts/smoke_installed.py`. `--spawn` uses spawned workers on any
platform (the default on macOS and Windows), to test that path on Linux too.

The code is under `if __name__ == "__main__":` because spawned workers re-run the main script when they start.
"""

import multiprocessing
import sys

import pandas as pd

import pymiraw

TOL = 1e-5
LET7A = "UGAGGUAGUAGGUUGUAUAGUU"
MIR21 = "UAGCUUAUCAGACUGAUGUUGA"
UTR = "UAGCUGAGCGGCGAACCACUAGAAAAGGUUAACUAUACAACCUACUACCUCACAGACCCCGGAGCCCAGCCGUCACGAUUGU"  # has a let-7 site


def check(name: str, got: float, expected: float) -> None:
    ok = abs(got - expected) <= TOL
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got:.6f} (expected {expected:.6f})")
    if not ok:
        sys.exit(1)


def main() -> None:
    if "--spawn" in sys.argv:
        multiprocessing.set_start_method("spawn")
    print("pyMIRAW", pymiraw.__version__, "from", pymiraw.__file__, "| workers:", multiprocessing.get_start_method())
    print(pymiraw.load_model())

    pair = pymiraw.example_pairs().iloc[0]
    result = pymiraw.predict_pair(pair.Mature_mirna_transcript, pair["3UTR_transcript"])
    check("example pair 0", result.prediction, 0.998879)

    table = pd.DataFrame({"mirna_seq": [LET7A, MIR21], "utr": [UTR, UTR]})
    preds = pymiraw.predict(table, workers=2, progress=False)  # two worker processes
    check("let-7a vs a let-7 site", preds.genes.Prediction[0], 0.999649)
    check("miR-21 vs the same 3'UTR", preds.genes.Prediction[1], -0.982088)

    model = pymiraw.train(pymiraw.example_sites("train").head(600), max_epochs=2, accelerator="cpu")
    print("ok   training:", model, model.meta["metrics"])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
