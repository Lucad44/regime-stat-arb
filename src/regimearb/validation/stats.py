"""Sharpe-ratio inference following Bailey & Lopez de Prado.

All Sharpe ratios in this module are per-period (not annualised); convert with
sqrt(periods) at the edges. `kurt` is raw kurtosis (3 for a normal), matching
the formulas in the papers.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def _moments(r: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    return float(sr), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), len(r)


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """P(true SR > benchmark) given an estimate from n_obs returns with these moments."""
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(max(var_term, 1e-12))))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max of N iid Sharpe estimates] under the null of zero true skill."""
    if n_trials < 2:
        return 0.0
    return float(
        np.sqrt(sr_variance)
        * ((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
           + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    )  # fmt: skip


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray) -> dict:
    """PSR against the Sharpe you'd expect the best of N no-skill trials to show.

    `trial_sharpes` are the per-period Sharpes of *every* configuration tried
    (including the one being evaluated), not just the survivors.
    """
    sr, skew, kurt, n = _moments(returns)
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    sr0 = expected_max_sharpe(len(trials), float(np.var(trials, ddof=1)))
    return dict(
        sharpe=sr,
        sharpe_ann=sr * np.sqrt(252),
        benchmark_sharpe=sr0,
        benchmark_sharpe_ann=sr0 * np.sqrt(252),
        psr_vs_zero=probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt),
        dsr=probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt),
        n_trials=len(trials),
    )


def min_track_record_length(
    sr: float, sr_benchmark: float = 0.0, skew: float = 0.0, kurt: float = 3.0, alpha: float = 0.05
) -> float:
    """Observations needed before sr is distinguishable from the benchmark at level alpha."""
    if sr <= sr_benchmark:
        return np.inf
    z = norm.ppf(1.0 - alpha)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * (z / (sr - sr_benchmark)) ** 2)


def stationary_bootstrap_sharpe(
    returns: np.ndarray, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Politis-Romano stationary bootstrap of the annualised Sharpe ratio."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
        s = r[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    return out
