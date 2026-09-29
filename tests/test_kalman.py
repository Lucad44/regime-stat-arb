import numpy as np

from regimearb.models.kalman import OUStateSpaceHedge, SSParams


def dense_reference(y, x, p: SSParams, b0, s0, P0, burn):
    """Textbook matrix Kalman filter for the same model, used to check the unrolled one."""
    F = np.diag([1.0, p.phi])
    c = np.array([0.0, p.mu * (1 - p.phi)])
    Q = np.diag([p.q_b, p.q_s])
    m = np.array([b0, s0])
    P = np.array(P0, dtype=float)
    ll, betas, spreads = 0.0, [], []
    for t in range(len(y)):
        m = F @ m + c
        P = F @ P @ F.T + Q
        H = np.array([x[t], 1.0])
        e = y[t] - H @ m
        S = H @ P @ H + p.r
        K = P @ H / S
        m = m + K * e
        P = P - np.outer(K, H @ P)
        if t >= burn:
            ll -= 0.5 * (np.log(2 * np.pi * S) + e**2 / S)
        betas.append(m[0])
        spreads.append(m[1])
    return np.array(betas), np.array(spreads), ll


def simulate_pair(rng, n=800, phi=0.92, sig_eq=0.02, drift=0.0):
    x = np.cumsum(rng.normal(0, 0.012, n)) + 4.6
    s = np.zeros(n)
    eta = sig_eq * np.sqrt(1 - phi**2)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + eta * rng.standard_normal()
    beta = 1.0 + np.cumsum(rng.normal(0, drift, n))
    return beta * (x - x[0]) + s + 4.5, x, beta, s


def test_unrolled_filter_matches_dense_matrix_version(rng):
    y, x, *_ = simulate_pair(rng, n=300)
    p = SSParams(q_b=1e-7, q_s=1.2e-4, phi=0.93, mu=4.5, r=1e-6)
    m = OUStateSpaceHedge(p, x_ref=float(x.mean()), beta_init=1.0, burn=20)
    out = m.filter(y, x)
    xc = x - x.mean()
    P0 = [[0.05, 0], [0, p.sigma_eq**2]]
    b_ref, s_ref, ll_ref = dense_reference(y, xc, p, 1.0, y[0] - xc[0], P0, 20)
    assert np.allclose(out.beta, b_ref, atol=1e-9)
    assert np.allclose(out.spread, s_ref, atol=1e-9)
    assert np.isclose(out.loglike, ll_ref, atol=1e-6)


def test_filter_is_causal(rng):
    y, x, *_ = simulate_pair(rng, n=400)
    m = OUStateSpaceHedge.fit(y[:300], x[:300])
    full = m.filter(y, x)
    part = m.filter(y[:250], x[:250])
    assert np.allclose(full.beta[:250], part.beta)
    assert np.allclose(full.spread[:250], part.spread)


def test_fit_recovers_ou_persistence(rng):
    y, x, *_ = simulate_pair(rng, n=1200, phi=0.92)
    p = OUStateSpaceHedge.fit(y, x).params
    assert abs(p.phi - 0.92) < 0.04
    assert 0.014 < p.sigma_eq < 0.028


def test_tracks_drifting_hedge_ratio_better_than_static_ols(rng):
    errs_ss, errs_ols = [], []
    for _ in range(6):
        y, x, beta, _ = simulate_pair(rng, n=900, drift=0.004)
        tr, te = slice(0, 500), slice(500, 900)
        m = OUStateSpaceHedge.fit(y[tr], x[tr])
        b_hat = m.filter(y, x).beta[te]
        b_ols = np.polyfit(x[tr], y[tr], 1)[0]
        errs_ss.append(np.sqrt(np.mean((b_hat - beta[te]) ** 2)))
        errs_ols.append(np.sqrt(np.mean((b_ols - beta[te]) ** 2)))
    assert np.median(errs_ss) < np.median(errs_ols)
