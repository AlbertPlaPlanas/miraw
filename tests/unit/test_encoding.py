import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from miraw.encoding.onehot import OneHotFlat, normalize_orientation, one_hot


def test_nucleotide_codes_are_c_a_g_u():
    assert one_hot("CAGU", 4).tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_t_equals_u_and_case_insensitive():
    assert np.array_equal(one_hot("tT", 2), one_hot("uU", 2))


def test_gaps_and_unknown_are_zero_and_padding_is_zero():
    assert one_hot("LN ", 5).sum() == 0


def test_truncates_to_max_len():
    assert one_hot("ACGUACGU", 3).shape == (3, 4)


def test_old_default_reverses_site_natural_does_not():
    old, nat = OneHotFlat(4, 4, "old-default"), OneHotFlat(4, 4, "natural")
    mirna, site = "ACGU", "CAGU"
    assert np.array_equal(old.encode(mirna, site)[16:], one_hot("UGAC", 4).ravel())
    assert np.array_equal(nat.encode(mirna, site)[16:], one_hot("CAGU", 4).ravel())
    assert np.array_equal(old.encode(mirna, site)[:16], nat.encode(mirna, site)[:16])


def test_short_site_reversed_then_zero_padded():
    enc = OneHotFlat(2, 4, "old-default")
    site_part = enc.encode("AA", "CA")[8:].reshape(4, 4)
    assert site_part[0].tolist() == [0, 1, 0, 0] and site_part[1].tolist() == [1, 0, 0, 0]
    assert site_part[2:].sum() == 0


def test_gap_padded_site_keeps_gaps_after_reversal():
    # 'LLCA' reversed is 'ACLL': the gaps end up at the end
    site_part = OneHotFlat(1, 4).encode("A", "LLCA")[4:].reshape(4, 4)
    assert site_part[0].tolist() == [0, 1, 0, 0] and site_part[2:].sum() == 0


def test_alias_and_invalid_orientation():
    assert normalize_orientation("reversed") == "old-default"
    with pytest.raises(ValueError):
        normalize_orientation("sideways")


def test_batch_matches_single():
    enc = OneHotFlat(5, 6)
    sites = ["ACGUAC", "GGGLLL"]
    batch = enc.encode_batch("ACGUA", sites)
    assert batch.shape == (2, enc.output_dim) == (2, 44)
    for i, s in enumerate(sites):
        assert np.array_equal(batch[i], enc.encode("ACGUA", s))


@given(st.text(alphabet="ACGUTLN ", max_size=60), st.text(alphabet="ACGUTLN ", max_size=60))
def test_length_invariant_and_binary(mirna, site):
    enc = OneHotFlat(30, 40)
    v = enc.encode(mirna, site)
    assert v.shape == (280,) and set(np.unique(v)) <= {0.0, 1.0}
    assert v.reshape(-1, 4).sum(axis=1).max() <= 1
