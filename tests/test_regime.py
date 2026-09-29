import numpy as np

from regimearb.diagnostics import auc
from regimearb.models.regime import MarkovSwitchingAR1, fit_single_ar1


def simulate_switching(n, seed):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.985 if state[t - 1] == 0 else 0.94
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    phi = np.array([0.9, 0.995])
    eta = np.array([0.008, 0.015])
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi[state[t]] * x[t - 1] + eta[state[t]] * rng.standard_normal()
    return x, state


def test_em_log_likelihood_never_decreases():
    x, _ = simulate_switching(1500, seed=1)
    m = MarkovSwitchingAR1(n_starts=1).fit(x)
    tr = np.array(m.ll_trace_)
    assert len(tr) > 3
    assert np.all(np.diff(tr) > -1e-6)


def test_states_ordered_by_dispersion_and_regimes_are_detected():
    x, state = simulate_switching(2500, seed=2)
    m = MarkovSwitchingAR1().fit(x)
    assert m.params_.sigma_eq[0] < m.params_.sigma_eq[1]
    p = m.filter(x)[:, 0]
    assert auc(p, (state == 0).astype(int)) > 0.8


def test_two_states_beat_one_only_when_they_exist():
    x, _ = simulate_switching(2500, seed=3)
    m = MarkovSwitchingAR1().fit(x)
    assert fit_single_ar1(x)[1] - m.bic(len(x) - 1) > 10

    rng = np.random.default_rng(9)
    y = np.zeros(2500)
    for t in range(1, 2500):
        y[t] = 0.9 * y[t - 1] + 0.01 * rng.standard_normal()
    m2 = MarkovSwitchingAR1().fit(y)
    assert fit_single_ar1(y)[1] - m2.bic(len(y) - 1) < 5


def test_filter_is_causal_and_normalised():
    x, _ = simulate_switching(800, seed=5)
    m = MarkovSwitchingAR1().fit(x[:500])
    full, part = m.filter(x), m.filter(x[:600])
    assert np.allclose(full[:600], part)
    assert np.allclose(full.sum(axis=1), 1.0)
    # the smoother is allowed to look ahead, so it must not equal the filter
    assert not np.allclose(m.smooth(x)[:-1], full[:-1])
