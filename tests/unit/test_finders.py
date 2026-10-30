"""Finder logic with a scripted fake folding backend (no ViennaRNA needed)."""

from miraw.folding.base import Fold, FoldService
from miraw.sites.finders.default import DefaultFinder, PersonalizedFinder
from miraw.sites.finders.pita import PitaFinder
from miraw.sites.finders.registry import get_site_finder
from miraw.sites.finders.targetscan import TargetScanFinder

MIRNA = "UGAGGUAGUAGGUUGUAUAGUU"  # let-7a, 22 nt


class FakeBackend:
    """Folds every window with a caller-provided function (site, mirna) -> Fold."""

    name, version = "fake", "0"

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def cofold(self, site, mirna):
        self.calls += 1
        return self.fn(site, mirna)

    def fold(self, seq):
        return Fold(seq, "." * len(seq), 0.0)

    def fold_with_open_gap(self, seq, a, b):
        return Fold(seq, "." * len(seq), 0.0)


def duplex_fold(site, mirna, n_pairs=8, energy=-10.0, site_pair_start=None):
    """Co-fold where the miRNA's nt 2.. pair with the site's last nucleotides."""
    s = len(site)
    struct = ["."] * (s + 1 + len(mirna))
    struct[s] = "&"
    start = (s - n_pairs) if site_pair_start is None else site_pair_start
    for k in range(n_pairs):
        struct[start + k] = "("
        struct[s + 1 + 1 + (n_pairs - 1 - k)] = ")"  # miRNA nt index 1.. (0-based), nested pairs
    return Fold(site + "&" + mirna, "".join(struct), energy)


def no_binding(site, mirna):
    return Fold(site + "&" + mirna, "." * (len(site) + 1 + len(mirna)), 0.0)


def test_registry_parses_original_type_strings():
    f = get_site_finder("Personalized;1;9;7")
    assert isinstance(f, PersonalizedFinder) and (f.seed_region_start, f.seed_region_size, f.min_binds_seed) == (
        1,
        9,
        7,
    )
    assert isinstance(get_site_finder("miRAW"), DefaultFinder)
    assert isinstance(get_site_finder("Default"), DefaultFinder)
    assert isinstance(get_site_finder("PITA"), PitaFinder)
    assert isinstance(get_site_finder("TS"), TargetScanFinder)
    assert isinstance(get_site_finder("garbage"), DefaultFinder)  # original falls back to Default
    assert get_site_finder("Personalized;0;9;20").min_binds_seed == 9  # capped to seed size


def test_default_scan_order_skips_position_zero():
    utr = "A" * 60
    f = DefaultFinder(max_site_length=40)
    assert f.scan_starts(MIRNA, utr) == list(range(20, 0, -1))


def test_personalized_scan_includes_position_zero_and_truncated_windows():
    f = PersonalizedFinder(max_site_length=40)
    assert f.scan_starts(MIRNA, "A" * 60) == list(range(38, -1, -1))


def test_no_duplex_gives_no_sites():
    svc = FoldService(FakeBackend(no_binding))
    assert DefaultFinder().find_candidates(MIRNA, "ACGU" * 25, svc) == []


def test_duplex_with_enough_seed_pairs_is_candidate_and_aligned():
    svc = FoldService(FakeBackend(lambda s, m: duplex_fold(s, m, n_pairs=8)))
    sites = DefaultFinder(max_site_length=40, seed_alignment_offset=5).find_candidates(MIRNA, "ACGU" * 25, svc)
    assert sites
    assert all(len(s.transcript) == 40 for s in sites)
    assert all(s.binds_in_seed >= 6 and s.free_energy < 0 for s in sites)
    # consecutive kept sites are at least win_step apart (the original's overlap rule)
    starts = [s.start for s in sites]
    assert all(a - b >= 5 for a, b in zip(starts, starts[1:], strict=False))


def test_overlap_keeps_lowest_energy():
    def fold(site, mirna):
        e = -20.0 if site.startswith("GG") else -5.0
        return duplex_fold(site, mirna, 8, e)

    utr = "A" * 30 + "GG" + "A" * 28 + "C" * 30
    sites = DefaultFinder().find_candidates(MIRNA, utr, FoldService(FakeBackend(fold)))
    starts = [s.start for s in sites]
    assert all(a - b >= 5 for a, b in zip(starts, starts[1:], strict=False))  # win_step spacing
    assert -20.0 in [s.free_energy for s in sites]  # the strongest site of its cluster survives


def test_positive_energy_and_zero_energy_rejected():
    for e in (0.0, 1.5):
        svc = FoldService(FakeBackend(lambda s, m, e=e: duplex_fold(s, m, 8, e)))
        assert DefaultFinder().find_candidates(MIRNA, "ACGU" * 25, svc) == []


def test_min_binds_threshold():
    svc = FoldService(FakeBackend(lambda s, m: duplex_fold(s, m, n_pairs=5)))
    assert DefaultFinder(min_binds_seed=6).find_candidates(MIRNA, "ACGU" * 25, svc) == []
    assert DefaultFinder(min_binds_seed=5).find_candidates(MIRNA, "ACGU" * 25, svc) != []


def test_fold_service_memoizes():
    backend = FakeBackend(no_binding)
    svc = FoldService(backend)
    svc.cofold("ACGU", MIRNA), svc.cofold("ACGU", MIRNA)
    assert backend.calls == 1


def test_disk_cache_roundtrip(tmp_path):
    import random

    rng = random.Random(1)
    utr = "".join(rng.choice("ACGU") for _ in range(100))
    b1 = FakeBackend(no_binding)
    FoldService(b1, tmp_path).prefetch_windows(MIRNA, utr, 40)
    assert b1.calls == len(utr) - 39
    b2 = FakeBackend(no_binding)
    svc2 = FoldService(b2, tmp_path)
    svc2.prefetch_windows(MIRNA, utr, 40)
    assert b2.calls == 0 and svc2.cofold(utr[3:43], MIRNA).energy == 0.0 and b2.calls == 0
