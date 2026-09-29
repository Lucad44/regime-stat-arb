"""Simulated equity universe with known cointegrated pairs.

The point of simulating is that the ground truth is known, so every estimator
in the package can be checked against it (hedge ratio path, regime labels, OU
parameters) instead of only being judged on a noisy backtest.

Layout of the universe
  * "leaders": factor-driven log-price random walks (one GARCH market factor,
    a few sector factors, idiosyncratic noise)
  * "followers": planted pairs. log P_f = a + beta_t * (log P_l - log P_l[0]) + s_t
    where beta_t drifts slowly and s_t is a two-regime OU process:
      regime 0 - fast mean reversion (the tradable regime)
      regime 1 - near unit root with inflated noise (the "relationship broke" regime)
  * pairs of leaders in the same sector are highly correlated but NOT
    cointegrated, which is exactly what a naive correlation screen picks up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SyntheticSpec:
    n_days: int = 2520
    n_leaders: int = 24
    n_planted: int = 10
    n_sectors: int = 4
    seed: int = 7
    start: str = "2010-01-04"
    mkt_daily_vol: float = 0.009
    idio_daily_vol: float = 0.011
    obs_noise: float = 0.0005
    # planted spread dynamics
    halflife_range: tuple[float, float] = (5.0, 18.0)
    sigma_eq_range: tuple[float, float] = (0.012, 0.030)
    p_stay_mr: float = 0.985
    p_stay_break: float = 0.94
    break_phi: float = 0.995
    break_vol_mult: float = 1.8
    beta_kappa: float = 0.002
    beta_sigma: float = 0.004


@dataclass
class SyntheticTruth:
    pairs: list[tuple[str, str]]  # (follower, leader)
    beta: pd.DataFrame
    spread: pd.DataFrame
    regime: pd.DataFrame  # 0 = mean-reverting, 1 = breakdown
    params: dict[str, dict] = field(default_factory=dict)

    def pair_set(self) -> set[frozenset[str]]:
        return {frozenset(p) for p in self.pairs}


def _garch_series(rng, n, vol, alpha=0.08, beta=0.90, df=6):
    omega = vol**2 * (1.0 - alpha - beta)
    scale = np.sqrt((df - 2.0) / df)
    z = rng.standard_t(df, size=n) * scale
    r = np.empty(n)
    var = vol**2
    for t in range(n):
        r[t] = np.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    return r


def _markov_chain(rng, n, p00, p11):
    state = np.zeros(n, dtype=int)
    u = rng.random(n)
    for t in range(1, n):
        stay = p00 if state[t - 1] == 0 else p11
        state[t] = state[t - 1] if u[t] < stay else 1 - state[t - 1]
    return state


def make_universe(spec: SyntheticSpec | None = None) -> tuple[pd.DataFrame, SyntheticTruth]:
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n, nl, npl = spec.n_days, spec.n_leaders, spec.n_planted
    if npl > nl:
        raise ValueError("need at least as many leaders as planted pairs")

    mkt = _garch_series(rng, n, spec.mkt_daily_vol)
    sector_f = rng.normal(0.0, 0.006, size=(n, spec.n_sectors))
    sector_of = rng.integers(0, spec.n_sectors, size=nl)
    b_mkt = rng.uniform(0.6, 1.4, size=nl)
    b_sec = rng.uniform(0.5, 1.2, size=nl)
    idio = rng.standard_t(5, size=(n, nl)) * np.sqrt(3 / 5) * spec.idio_daily_vol

    lead_ret = mkt[:, None] * b_mkt + sector_f[:, sector_of] * b_sec + idio
    lead_lp = np.log(rng.uniform(40, 200, size=nl)) + np.cumsum(lead_ret, axis=0)

    chosen = rng.choice(nl, size=npl, replace=False)
    fol_lp = np.empty((n, npl))
    beta_paths = np.empty((n, npl))
    spreads = np.empty((n, npl))
    regimes = np.empty((n, npl), dtype=int)
    params: list[dict] = []

    for k, li in enumerate(chosen):
        hl = rng.uniform(*spec.halflife_range)
        phi0 = 0.5 ** (1.0 / hl)
        sig_eq = rng.uniform(*spec.sigma_eq_range)
        eta0 = sig_eq * np.sqrt(1.0 - phi0**2)
        phi1, eta1 = spec.break_phi, eta0 * spec.break_vol_mult

        reg = _markov_chain(rng, n, spec.p_stay_mr, spec.p_stay_break)
        eps = rng.standard_normal(n)
        s = np.empty(n)
        s[0] = rng.normal(0.0, sig_eq)
        for t in range(1, n):
            if reg[t] == 0:
                s[t] = phi0 * s[t - 1] + eta0 * eps[t]
            else:
                s[t] = phi1 * s[t - 1] + eta1 * eps[t]

        b0 = rng.uniform(0.7, 1.3)
        b = np.empty(n)
        b[0] = b0
        shocks = rng.normal(0.0, spec.beta_sigma, size=n)
        for t in range(1, n):
            b[t] = b[t - 1] + spec.beta_kappa * (b0 - b[t - 1]) + shocks[t]

        x = lead_lp[:, li]
        fol_lp[:, k] = np.log(rng.uniform(40, 200)) + b * (x - x[0]) + s
        beta_paths[:, k], spreads[:, k], regimes[:, k] = b, s, reg
        params.append(
            dict(halflife=hl, phi0=phi0, sigma_eq=sig_eq, eta0=eta0, phi1=phi1, eta1=eta1, beta0=b0)
        )

    log_p = np.hstack([lead_lp, fol_lp])
    log_p = log_p + rng.normal(0.0, spec.obs_noise, size=log_p.shape)

    # shuffle so planted followers are not sitting in the last columns
    perm = rng.permutation(nl + npl)
    names = np.array([f"S{i:02d}" for i in range(nl + npl)])
    col_name = np.empty(nl + npl, dtype=object)
    col_name[perm] = names  # original column j gets name names[perm-inverse]
    idx = pd.bdate_range(spec.start, periods=n)
    prices = pd.DataFrame(np.exp(log_p), index=idx, columns=list(col_name))
    prices = prices[sorted(prices.columns)]

    leader_names = [col_name[i] for i in chosen]
    follower_names = [col_name[nl + k] for k in range(npl)]
    truth = SyntheticTruth(
        pairs=list(zip(follower_names, leader_names, strict=True)),
        beta=pd.DataFrame(beta_paths, index=idx, columns=follower_names),
        spread=pd.DataFrame(spreads, index=idx, columns=follower_names),
        regime=pd.DataFrame(regimes, index=idx, columns=follower_names),
        params=dict(zip(follower_names, params, strict=True)),
    )
    return prices, truth
