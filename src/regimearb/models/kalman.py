"""State-space cointegration model with an explicit OU spread.

    y_t = beta_t * (x_t - x_ref) + s_t + v_t         v ~ N(0, r)
    beta_t = beta_{t-1} + w1                          w1 ~ N(0, q_b)
    s_t    = mu (1 - phi) + phi s_{t-1} + w2          w2 ~ N(0, q_s)

The usual "Kalman pairs trading" recipe lets beta and the intercept float and
reads the innovation as the spread, which conflates a drifting hedge ratio with
a mean-reverting deviation. Here the spread is its own AR(1) state, so a slow
beta drift and a fast OU deviation are separated by their different dynamics.
Parameters are fitted by maximum likelihood on the formation window only, and
the filter is then run forward with those parameters frozen.

The 2x2 covariance algebra is written out by hand: it is about 20x faster than
np.dot on tiny matrices and lets numba compile the whole loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .._jit import njit


@njit(cache=True)
def _kf(y, x, q_b, q_s, phi, mu, r, b0, s0, p0_b, p0_s, burn):
    n = y.shape[0]
    beta = np.empty(n)
    spread = np.empty(n)
    innov = np.empty(n)
    innov_var = np.empty(n)
    b, s = b0, s0
    p11, p12, p22 = p0_b, 0.0, p0_s
    c = mu * (1.0 - phi)
    ll = 0.0
    for t in range(n):
        b_pred = b
        s_pred = c + phi * s
        a11 = p11 + q_b
        a12 = phi * p12
        a22 = phi * phi * p22 + q_s

        xt = x[t]
        e = y[t] - (xt * b_pred + s_pred)
        S = xt * xt * a11 + 2.0 * xt * a12 + a22 + r
        k1 = (a11 * xt + a12) / S
        k2 = (a12 * xt + a22) / S

        b = b_pred + k1 * e
        s = s_pred + k2 * e
        p11 = a11 - S * k1 * k1
        p12 = a12 - S * k1 * k2
        p22 = a22 - S * k2 * k2
        if p11 < 1e-14:
            p11 = 1e-14
        if p22 < 1e-14:
            p22 = 1e-14

        beta[t] = b
        spread[t] = s
        innov[t] = e
        innov_var[t] = S
        if t >= burn:
            ll -= 0.5 * (np.log(2.0 * np.pi * S) + e * e / S)
    return beta, spread, innov, innov_var, ll


@dataclass(frozen=True)
class SSParams:
    q_b: float
    q_s: float
    phi: float
    mu: float
    r: float

    @property
    def sigma_eq(self) -> float:
        return float(np.sqrt(self.q_s / (1.0 - self.phi**2)))

    @property
    def half_life(self) -> float:
        return float(np.log(2.0) / -np.log(self.phi)) if 0 < self.phi < 1 else np.inf


@dataclass
class FilterOutput:
    beta: np.ndarray
    spread: np.ndarray
    innov: np.ndarray
    innov_var: np.ndarray
    loglike: float


def _unpack(theta: np.ndarray) -> SSParams:
    q_b, u, q_s, mu, r = np.exp(theta[0]), theta[1], np.exp(theta[2]), theta[3], np.exp(theta[4])
    phi = float(np.clip(1.0 / (1.0 + np.exp(-u)), 0.02, 0.9995))
    return SSParams(q_b=float(q_b), q_s=float(q_s), phi=phi, mu=float(mu), r=float(r))


class OUStateSpaceHedge:
    def __init__(self, params: SSParams, x_ref: float, beta_init: float, burn: int = 20):
        self.params = params
        self.x_ref = x_ref
        self.beta_init = beta_init
        self.burn = burn

    def filter(self, y: np.ndarray, x: np.ndarray) -> FilterOutput:
        p = self.params
        y = np.ascontiguousarray(y, dtype=float)
        xc = np.ascontiguousarray(x, dtype=float) - self.x_ref
        s0 = y[0] - self.beta_init * xc[0]
        beta, s, e, S, ll = _kf(
            y, xc, p.q_b, p.q_s, p.phi, p.mu, p.r,
            self.beta_init, s0, 0.05, p.sigma_eq**2, self.burn,
        )  # fmt: skip
        return FilterOutput(beta, s, e, S, ll)

    @classmethod
    def fit(cls, y: np.ndarray, x: np.ndarray, burn: int = 20) -> OUStateSpaceHedge:
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        x_ref = float(x.mean())
        xc = x - x_ref

        # OLS + AR(1) on the residual give a sensible starting point
        beta_ols, alpha_ols = np.polyfit(xc, y, 1)
        res = y - beta_ols * xc - alpha_ols
        phi0 = float(np.clip(np.corrcoef(res[:-1], res[1:])[0, 1], 0.3, 0.98))
        q_s0 = max(float(res.std()) ** 2 * (1.0 - phi0**2), 1e-8)

        def nll(theta):
            p = _unpack(theta)
            s0 = y[0] - beta_ols * xc[0]
            *_, ll = _kf(y, xc, p.q_b, p.q_s, p.phi, p.mu, p.r, beta_ols, s0, 0.05,
                         p.sigma_eq**2, burn)  # fmt: skip
            return -ll if np.isfinite(ll) else 1e12

        bounds = [(-25, -4), (-3, 8), (-20, -3), (alpha_ols - 5 * res.std(), alpha_ols + 5 * res.std()),
                  (-18, -6)]  # fmt: skip
        best = None
        for lqb0 in (-16.0, -12.0):
            theta0 = np.array([lqb0, np.log(phi0 / (1 - phi0)), np.log(q_s0), alpha_ols, -14.0])
            out = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
            if best is None or out.fun < best.fun:
                best = out
        return cls(_unpack(best.x), x_ref=x_ref, beta_init=float(beta_ols), burn=burn)
