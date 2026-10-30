"""Dot-bracket helpers ported from the original CandidateSiteFinder (bug-for-bug)."""

from __future__ import annotations

_COMPLEMENT = {")": "(", "(": ")", "}": "{", "{": "}", "]": "[", "[": "]"}


def is_watson_crick(n1: str, n2: str) -> bool:
    a, b = n1.upper(), n2.upper()
    if a == "A" and b in "UT":
        return True
    if b == "A" and a in "UT":
        return True
    return {a, b} == {"G", "C"}


def is_wobble(n1: str, n2: str) -> bool:
    a, b = n1.upper(), n2.upper()
    if a == "I" or b == "I":
        return a != "G" and b != "G"
    return (a == "G" and b in "TU") or (a in "TU" and b == "G")


def find_binding_start_in_sequence(structure: str) -> int:
    """Index of the first unmatched '(' scanning left from '&': the paired site nucleotide nearest
    the miRNA 5' end. Used as the anchor when aligning the seed inside the site window."""
    if "&" not in structure:
        return -1
    i, open_par = structure.index("&"), 0
    while i >= 0:
        c = structure[i]
        if c in ")}":
            open_par += 1
        if c in "({":
            if open_par == 0:
                return i
            open_par -= 1
        i -= 1
    return -1


def check_binding_from_structure(structure: str | None) -> bool:
    """True if the miRNA strand has more closing than opening brackets (a real duplex)."""
    if not structure or "&" not in structure:
        return False
    tail = structure[structure.index("&") + 1 :]
    closing = sum(c in ")]}" for c in tail)
    opening = sum(c in "([{" for c in tail)
    return closing > opening


def get_pair_at(pos: int, transcript: str, structure: str) -> str:
    """Two-character string ``partner + base`` for the paired position ``pos`` (else the base)."""
    sp, tp = structure[pos], transcript[pos]
    if sp in ".,":
        return sp
    comp = _COMPLEMENT.get(sp)
    if sp in ")}":
        found, depth, i = False, 0, pos - 1
        while not found and i >= 0:
            if structure[i] == comp:
                if depth == 0:
                    found = True
                else:
                    depth -= 1
            if structure[i] == sp:
                depth += 1
            i -= 1
        if found:
            return transcript[i + 1] + tp
    if sp in "{(":
        found, depth, i = False, 0, pos - 1
        while not found and i < len(structure):
            if structure[i] == comp:
                if depth == 0:
                    found = True
                else:
                    depth -= 1
            if structure[i] == sp:
                depth += 1
            i += 1
        if found and i < len(transcript):
            return transcript[i] + tp
    return tp
