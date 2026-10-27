"""ViennaRNA backend using the in-process ``RNA`` Python bindings (fast, default)."""

from __future__ import annotations

from pymiraw.folding.base import Fold


class ViennaBindings:
    name = "vienna-bindings"

    def __init__(self, temperature: float = 37.0):
        import RNA  # noqa: PLC0415  (optional heavy dependency)

        self._rna = RNA
        self.version = RNA.__version__
        self.temperature = temperature
        md = RNA.md()
        md.temperature = temperature
        self._md = md

    def cofold(self, site: str, mirna: str) -> Fold:
        fc = self._rna.fold_compound(f"{site}&{mirna}", self._md)
        structure, energy = fc.mfe_dimer()
        if "&" not in structure:  # bindings omit the strand separator that RNAcofold prints
            structure = structure[: len(site)] + "&" + structure[len(site) :]
        return Fold(f"{site}&{mirna}", structure, round(energy, 2))

    def fold(self, sequence: str) -> Fold:
        fc = self._rna.fold_compound(sequence, self._md)
        structure, energy = fc.mfe()
        return Fold(sequence, structure, round(energy, 2))

    def fold_with_open_gap(self, sequence: str, start: int, end: int) -> Fold:
        constraint = "".join("x" if start <= i <= end else "." for i in range(len(sequence)))
        fc = self._rna.fold_compound(sequence, self._md)
        fc.constraints_add(constraint, self._rna.CONSTRAINT_DB_DEFAULT | self._rna.CONSTRAINT_DB_ENFORCE_BP)
        structure, energy = fc.mfe()
        return Fold(sequence, structure, round(energy, 2))
