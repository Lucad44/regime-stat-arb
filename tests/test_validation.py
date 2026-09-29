import numpy as np

from regimearb.validation import (
    cscv,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_sharpe,
)


def test_psr_is_half_at_benchmark_and_monotone_in_sample_size():
    assert np.isclose(probabilistic_sharpe_ratio(0.05, 0.05, 500), 0.5)
    lo = probabilistic_sharpe_ratio(0.05, 0.0, 250)
    hi = probabilistic_sharpe_ratio(0.05, 0.0, 1000)
    assert 0.5 < lo < hi < 1.0


def test_negative_skew_and_fat_tails_lower_the_psr():
    base = probabilistic_sharpe_ratio(0.08, 0.0, 500, skew=0.0, kurt=3.0)
    ugly = probabilistic_sharpe_ratio(0.08, 0.0, 500, skew=-1.5, kurt=12.0)
    assert ugly < base


def test_expected_max_sharpe_grows_with_number_of_trials():
    vals = [expected_max_sharpe(n, 0.001) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals) and vals[0] > 0


def test_dsr_penalises_the_best_of_many_noise_strategies(rng):
    noise = rng.normal(0, 0.01, size=(1500, 200))
    sr = noise.mean(0) / noise.std(0, ddof=1)
    out = deflated_sharpe_ratio(noise[:, sr.argmax()], sr)
    assert out["psr_vs_zero"] > 0.9  # looks great on its own...
    assert out["dsr"] < 0.6  # ...but not once you count the 200 tries


def test_min_track_record_length():
    assert np.isinf(min_track_record_length(-0.1))
    assert min_track_record_length(0.1) < min_track_record_length(0.05)


def test_cscv_null_pbo_is_near_one_half(rng):
    pbos = [
        cscv(rng.normal(0, 0.01, size=(1600, 40)), max_splits=800, seed=s).pbo for s in range(25)
    ]
    assert 0.4 < np.mean(pbos) < 0.65


def test_cscv_low_pbo_when_one_config_is_genuinely_better(rng):
    R = rng.normal(0, 0.01, size=(1600, 30))
    R[:, 7] += 0.0015
    res = cscv(R)
    assert res.pbo < 0.05
    assert res.prob_oos_loss < 0.05
    assert res.oos_best_sharpe.mean() > 1.0


def test_cscv_matches_brute_force_on_a_split(rng):
    import itertools

    R = rng.normal(0.0002, 0.01, size=(1600, 12))
    res = cscv(R, n_blocks=16)
    blocks = R.reshape(16, 100, 12)
    combo = list(itertools.combinations(range(16), 8))[137]
    oos = [b for b in range(16) if b not in combo]
    sr = lambda a: a.mean(0) / a.std(0, ddof=1)  # noqa: E731
    s_in, s_out = sr(blocks[list(combo)].reshape(-1, 12)), sr(blocks[oos].reshape(-1, 12))
    best = s_in.argmax()
    w = ((s_out < s_out[best]).sum() + 1) / 13
    assert np.isclose(res.logits[137], np.log(w / (1 - w)))


def test_bootstrap_ci_contains_point_estimate(rng):
    r = rng.normal(0.0004, 0.01, size=1000)
    boots = stationary_bootstrap_sharpe(r, n_boot=300, seed=1)
    point = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert np.percentile(boots, 2.5) < point < np.percentile(boots, 97.5)
