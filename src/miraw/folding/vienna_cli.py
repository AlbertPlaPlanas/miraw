"""ViennaRNA backend driving the command-line tools (any installed version, e.g. 2.2.5)."""

from __future__ import annotations

import re
import subprocess

from miraw.folding.base import Fold

_ENERGY = re.compile(r"\(\s*(-?\d+\.\d+)\s*\)\s*$")


class ViennaCLI:
    name = "vienna-cli"

    def __init__(self, bin_dir: str = "", temperature: float = 37.0):
        self.bin = bin_dir.rstrip("/") + "/" if bin_dir else ""
        self.temperature = temperature
        out = subprocess.run([f"{self.bin}RNAcofold", "--version"], capture_output=True, text=True, check=True).stdout
        self.version = out.split()[-1]

    def _run(self, tool: str, extra: list[str], stdin: str) -> list[str]:
        cmd = [f"{self.bin}{tool}", "--noconv", "--noPS", f"--temp={self.temperature}", *extra]
        res = subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=True)
        return [ln for ln in res.stdout.splitlines() if not ln.startswith("WARNING")]

    @staticmethod
    def _parse(seq: str, line: str) -> Fold:
        m = _ENERGY.search(line)
        if not m:
            raise ValueError(f"cannot parse folding output: {line!r}")
        return Fold(seq, line[: m.start()].strip().split()[0], float(m.group(1)))

    def cofold(self, site: str, mirna: str) -> Fold:
        seq = f"{site}&{mirna}"
        lines = self._run("RNAcofold", [], seq + "\n")
        return self._parse(seq, lines[1])

    def fold(self, sequence: str) -> Fold:
        lines = self._run("RNAcofold", [], sequence + "\n")
        return self._parse(sequence, lines[1])

    def fold_with_open_gap(self, sequence: str, start: int, end: int) -> Fold:
        constraint = "".join("x" if start <= i <= end else "." for i in range(len(sequence)))
        lines = self._run("RNAfold", ["-C"], f"{sequence}\n{constraint}\n")
        return self._parse(sequence, lines[1])


def make_backend(kind: str = "bindings", **kw):
    if kind in ("bindings", "vienna-bindings"):
        from miraw.folding.vienna_bindings import ViennaBindings

        return ViennaBindings(**kw)
    if kind in ("cli", "vienna-cli"):
        return ViennaCLI(**kw)
    raise ValueError(f"unknown folding backend {kind!r}")
