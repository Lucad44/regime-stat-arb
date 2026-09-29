"""Ornstein-Uhlenbeck helpers and Bertram (2010) optimal trading thresholds.

For dX = -theta (X - mu) dt + sigma dW, a trader who buys at level a and sells
at level m (a < m), paying a round-trip cost c, completes a full a -> m -> a
cycle in expected time

    E[T] = (pi / theta) * ( erfi(m sqrt(theta)/sigma) - erfi(a sqrt(theta)/sigma) )

(mu = 0 after demeaning). Expected profit per trade is m - a - c, so the profit
rate is (m - a - c) / E[T]. Working in units of the stationary std
sigma_eq = sigma / sqrt(2 theta) removes theta and sigma from the optimisation:
the optimal entry depends only on cost / sigma_eq, which is what makes the rule
cheap to apply pair by pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfi


@dataclass(frozen=True)
class OUFit:
    phi: float
    mu: float
    sigma_eq: float
    theta: float  # per-step mean reversion speed
    half_life: float


def fit_ou(x: np.ndarray) -> OUFit:
    """AR(1) by OLS, mapped to exact-discretisation OU parameters (dt = 1 step)."""
    x = np.asarray(x, dtype=float)
    x0, x1 = x[:-1], x[1:]
    phi, c = np.polyfit(x0, x1, 1)
    resid = x1 - (c + phi * x0)
    phi = float(np.clip(phi, 1e-4, 0.99999))
    mu = c / (1.0 - phi)
    sig_eq = float(resid.std(ddof=2) / np.sqrt(1.0 - phi**2))
    theta = -np.log(phi)
    return OUFit(phi, float(mu), sig_eq, float(theta), float(np.log(2.0) / theta))


def half_life(x: np.ndarray) -> float:
    phi = np.polyfit(x[:-1] - x.mean(), x[1:] - x.mean(), 1)[0]
    if phi <= 0 or phi >= 1:
        return np.inf
    return float(-np.log(2.0) / np.log(phi))


def cycle_time(a: float, m: float, theta: float, sigma_eq: float = 1.0) -> float:
    """Expected a -> m -> a cycle time; a, m in the same units as sigma_eq."""
    s = sigma_eq * np.sqrt(2.0 * theta)  # diffusion coefficient of the OU
    k = np.sqrt(theta) / s
    return float(np.pi / theta * (erfi(m * k) - erfi(a * k)))


def profit_rate(a: float, m: float, cost: float, theta: float, sigma_eq: float = 1.0) -> float:
    """Expected profit per unit time for a single-sided (long) cycle."""
    return (m - a - cost) / cycle_time(a, m, theta, sigma_eq)


def bertram_entry(cost_in_sigma: float, exit_level: float = 0.0, k_max: float = 4.0) -> float:
    """Optimal |entry| in stationary-std units for a symmetric two-sided strategy.

    `cost_in_sigma` is the round-trip cost divided by sigma_eq. Exit is fixed at
    `exit_level` (0 = the long-run mean). Returns k > 0 such that longs open at -k.
    Theta drops out of the argmax, so it is set to 1 here.
    """

    def neg_rate(k):
        a, m = -k, exit_level
        if m - a - cost_in_sigma <= 0:
            return 1e6
        return -profit_rate(a, m, cost_in_sigma, theta=1.0)

    lo = max(exit_level + cost_in_sigma, 0.05) + 1e-3
    res = minimize_scalar(neg_rate, bounds=(lo, k_max), method="bounded", options={"xatol": 1e-4})
    return float(res.x)
