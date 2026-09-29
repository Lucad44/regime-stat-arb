import numpy as np

from regimearb.selection.pairs import benjamini_hochberg, screen_pairs


def test_benjamini_hochberg_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert np.isclose(q[0], 0.01) and np.isclose(q[1], 0.04)
    assert np.all(q >= p)


def test_screen_beats_the_base_rate_and_respects_reuse_cap(small_universe):
    prices, truth = small_universe
    lp = np.log(prices)
    n_assets = lp.shape[1]
    base_rate = len(truth.pairs) / (n_assets * (n_assets - 1) / 2)

    hits = picked = 0
    for start in (0, 300, 600):  # power varies a lot by window, so pool a few
        sel = screen_pairs(lp.iloc[start : start + 500], top_k=30, fdr=0.25, max_pairs=8, max_reuse=1)
        used = list(sel["y"]) + list(sel["x"])
        assert len(used) == len(set(used))
        picked += len(sel)
        hits += sum(frozenset((r.y, r.x)) in truth.pair_set() for r in sel.itertuples())
    assert picked > 0
    assert hits / picked > 5 * base_rate
