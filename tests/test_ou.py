import numpy as np
from scipy.signal import lfilter

from regimearb.models.ou import bertram_entry, cycle_time, fit_ou, profit_rate


def sim_ou(theta, sig_eq, dt, n, seed):
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta * dt)
    eta = sig_eq * np.sqrt(1 - phi**2)
    return lfilter([1.0], [1.0, -phi], rng.normal(0, eta, n))


def test_fit_ou_recovers_parameters():
    x = sim_ou(theta=0.08, sig_eq=0.02, dt=1.0, n=20000, seed=0)
    f = fit_ou(x)
    assert abs(f.theta - 0.08) < 0.01
    assert abs(f.sigma_eq - 0.02) < 0.002
    assert abs(f.half_life - np.log(2) / 0.08) < 1.5


def test_bertram_cycle_time_matches_monte_carlo():
    theta, dt = 1.0, 0.002
    x = sim_ou(theta, 1.0, dt, n=3_000_000, seed=4)
    a, m = -1.0, 0.0
    below, above = np.flatnonzero(x <= a), np.flatnonzero(x >= m)
    entries, pos = [], 0
    while True:
        i = below[np.searchsorted(below, pos)] if np.searchsorted(below, pos) < len(below) else None
        if i is None:
            break
        j_idx = np.searchsorted(above, i)
        if j_idx >= len(above):
            break
        entries.append(i)
        pos = above[j_idx]
    cycles = np.diff(entries) * dt
    analytic = cycle_time(a, m, theta, sigma_eq=1.0)
    assert len(cycles) > 500
    assert abs(cycles.mean() / analytic - 1.0) < 0.10


def test_optimal_entry_is_a_maximum_and_grows_with_cost():
    ks = []
    for c in (0.1, 0.3, 0.6):
        k = bertram_entry(c)
        ks.append(k)
        rate = {kk: profit_rate(-kk, 0.0, c, theta=1.0) for kk in (k - 0.15, k, k + 0.15)}
        assert rate[k] >= rate[k - 0.15] and rate[k] >= rate[k + 0.15]
    assert ks[0] < ks[1] < ks[2]
