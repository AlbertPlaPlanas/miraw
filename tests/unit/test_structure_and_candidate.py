import pytest

from pymiraw.sites.candidate import CandidateSite
from pymiraw.sites.structure import (
    check_binding_from_structure,
    find_binding_start_in_sequence,
    get_pair_at,
    is_watson_crick,
    is_wobble,
)


@pytest.mark.parametrize(
    ("a", "b", "wc", "wob"),
    [
        ("A", "U", True, False),
        ("a", "T", True, False),
        ("G", "C", True, False),
        ("G", "U", False, True),
        ("T", "g", False, True),
        ("A", "G", False, False),
        ("I", "A", False, True),
    ],
)
def test_pairing_tables(a, b, wc, wob):
    assert is_watson_crick(a, b) is wc and is_wobble(a, b) is wob
    assert is_watson_crick(b, a) is wc and is_wobble(b, a) is wob


# site GGGAAA & miRNA UUUCCC : fully paired duplex
SEQ, STRUCT = "GGGAAA&UUUCCC", "((((((&))))))"


def test_binding_detected():
    assert check_binding_from_structure(STRUCT)
    assert not check_binding_from_structure("......&......")
    assert not check_binding_from_structure(None)
    assert not check_binding_from_structure("((((")  # no strand separator


def test_binding_start_is_first_unmatched_open_bracket_left_of_ampersand():
    assert find_binding_start_in_sequence("..((..&..))..") == 3
    assert find_binding_start_in_sequence("((((((&))))))") == 5
    assert find_binding_start_in_sequence("(.(..&..).).") == 2
    assert find_binding_start_in_sequence("no amp") == -1


def test_get_pair_at_returns_partner_then_own_base():
    # miRNA position 7 (first ')') pairs with the last '(' (index 5)
    assert get_pair_at(7, SEQ, STRUCT) == "A" + "U"
    assert get_pair_at(12, SEQ, STRUCT) == "G" + "C"
    assert get_pair_at(0, "A", ".") == "."


def test_align_and_extend_shifts_and_pads_both_sides():
    utr = "ACGU" * 10  # length 40
    c = CandidateSite(start=2, end=12, transcript="x", seed_start_in_sequence=3)
    c.align_and_extend(utr, site_length=10, seed_start_index=8)  # shift +5 -> start -3
    assert (c.start, c.end) == (0, 7)
    assert c.transcript == "LLL" + utr[0:7] and len(c.transcript) == 10
    assert c.seed_start_in_sequence == 8
    c2 = CandidateSite(start=35, end=45, seed_start_in_sequence=8)
    c2.align_and_extend(utr, 10, 5)  # shift -3 -> start 38
    assert c2.transcript == utr[38:40] + "L" * 8 and (c2.start, c2.end) == (38, 40)


def test_align_inside_utr_is_exact_window():
    utr = "ACGUACGUAC" * 5
    c = CandidateSite(start=20, end=30, seed_start_in_sequence=4)
    c.align_and_extend(utr, 10, 4)
    assert c.transcript == utr[20:30]
