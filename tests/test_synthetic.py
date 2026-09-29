import numpy as np

from regimearb.data import SyntheticSpec, make_universe


def test_reproducible_and_shapes():
    spec = SyntheticSpec(n_days=400, n_leaders=8, n_planted=3, seed=5)
    p1, t1 = make_universe(spec)
    p2, _ = make_universe(spec)
    assert p1.shape == (400, 11)
    assert np.allclose(p1.values, p2.values)
    assert (p1 > 0).all().all()
    assert len(t1.pairs) == 3 and len(t1.pair_set()) == 3


def test_planted_pairs_are_cointegrated_but_decoys_are_not():
    from statsmodels.tsa.stattools import coint

    prices, truth = make_universe(SyntheticSpec(n_days=1500, n_leaders=10, n_planted=4, seed=2))
    lp = np.log(prices)
    followers = {f for f, _ in truth.pairs}
    planted_p = [coint(lp[f], lp[l], maxlag=1, autolag=None)[1] for f, l in truth.pairs]
    decoys = [c for c in lp.columns if c not in followers][:6]
    decoy_p = [coint(lp[decoys[i]], lp[decoys[i + 1]], maxlag=1, autolag=None)[1] for i in range(5)]
    assert np.median(planted_p) < np.median(decoy_p)


def test_regime_chain_has_both_states():
    _, truth = make_universe(SyntheticSpec(n_days=2000, n_leaders=6, n_planted=3, seed=8))
    share = truth.regime.mean().values
    assert np.all((share > 0.03) & (share < 0.6))
