"""Two-state Markov-switching AR(1), fitted by EM (Baum-Welch).

    x_t = c_k + phi_k x_{t-1} + sigma_k eps_t,    k = S_t, S_t ~ Markov(P)

The forward pass is done with per-step scaling so that long series do not
underflow. Two things matter for using it in a backtest:

  * `filter` returns P(S_t | x_1..x_t) using only past data. The smoothed
    probabilities from `smooth` look ahead and must never feed a signal.
  * A regime split is only used if BIC prefers it to a single AR(1). On a pair
    that never actually breaks down the two-state fit is just overfitting noise.

After fitting, states are sorted by stationary std sigma / sqrt(1 - phi^2), so
state 0 is always the one where the spread stays tightly confined. Sorting by
phi looks more natural but is fragile: over short breakdown episodes the
persistence is poorly identified while the dispersion is not, and on simulated
data with known labels, ordering by phi gave a median out-of-sample AUC of 0.37 for
the "calm" probability, i.e. worse than chance, because the labels were often swapped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._jit import njit


@njit(cache=True)
def _log_emission(x0, x1, c, phi, sig):
    n, k = x0.shape[0], c.shape[0]
    out = np.empty((n, k))
    for j in range(k):
        for t in range(n):
            z = (x1[t] - c[j] - phi[j] * x0[t]) / sig[j]
            out[t, j] = -0.5 * z * z - np.log(sig[j]) - 0.9189385332046727
    return out


@njit(cache=True)
def _forward(logb, P, pi0):
    n, k = logb.shape
    alpha = np.empty((n, k))
    scale = np.empty(n)
    ll = 0.0
    for t in range(n):
        m = logb[t, 0]
        for j in range(1, k):
            if logb[t, j] > m:
                m = logb[t, j]
        tot = 0.0
        for j in range(k):
            if t == 0:
                pred = pi0[j]
            else:
                pred = 0.0
                for i in range(k):
                    pred += alpha[t - 1, i] * P[i, j]
            alpha[t, j] = pred * np.exp(logb[t, j] - m)
            tot += alpha[t, j]
        for j in range(k):
            alpha[t, j] /= tot
        scale[t] = tot
        ll += np.log(tot) + m
    return alpha, scale, ll


@njit(cache=True)
def _backward(logb, P, scale):
    n, k = logb.shape
    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        m = logb[t + 1, 0]
        for j in range(1, k):
            if logb[t + 1, j] > m:
                m = logb[t + 1, j]
        for i in range(k):
            acc = 0.0
            for j in range(k):
                acc += P[i, j] * np.exp(logb[t + 1, j] - m) * beta[t + 1, j]
            beta[t, i] = acc / scale[t + 1]
    return beta


@njit(cache=True)
def _xi_sum(alpha, beta, logb, P, scale):
    n, k = logb.shape
    xi = np.zeros((k, k))
    for t in range(n - 1):
        m = logb[t + 1, 0]
        for j in range(1, k):
            if logb[t + 1, j] > m:
                m = logb[t + 1, j]
        for i in range(k):
            for j in range(k):
                xi[i, j] += (
                    alpha[t, i] * P[i, j] * np.exp(logb[t + 1, j] - m) * beta[t + 1, j]
                    / scale[t + 1]
                )  # fmt: skip
    return xi


def _stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    pi = np.abs(pi)
    return pi / pi.sum()


@dataclass
class MSAR1Params:
    c: np.ndarray
    phi: np.ndarray
    sigma: np.ndarray
    P: np.ndarray

    @property
    def mu(self) -> np.ndarray:
        return self.c / (1.0 - self.phi)

    @property
    def sigma_eq(self) -> np.ndarray:
        return self.sigma / np.sqrt(np.clip(1.0 - self.phi**2, 1e-8, None))


class MarkovSwitchingAR1:
    def __init__(self, n_states: int = 2, n_iter: int = 150, tol: float = 1e-6,
                 n_starts: int = 4, seed: int = 0):  # fmt: skip
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.n_starts = n_starts
        self.seed = seed
        self.params_: MSAR1Params | None = None
        self.loglike_: float = -np.inf
        self.ll_trace_: list[float] = []

    @property
    def n_params(self) -> int:
        k = self.n_states
        return 3 * k + k * (k - 1)

    def bic(self, n_obs: int) -> float:
        return self.n_params * np.log(n_obs) - 2.0 * self.loglike_

    def _init(self, x0, x1, rng, start):
        k = self.n_states
        phi_ols = np.polyfit(x0, x1, 1)[0]
        res_sd = float((x1 - np.polyval(np.polyfit(x0, x1, 1), x0)).std())
        if start == 0:
            phi = np.linspace(0.85, 0.995, k)
            sig = res_sd * np.linspace(0.8, 1.6, k)
        else:
            phi = np.sort(rng.uniform(0.4, 0.999, size=k))
            sig = res_sd * rng.uniform(0.6, 2.0, size=k)
        c = x0.mean() * (1.0 - phi) + rng.normal(0, 0.05 * res_sd, size=k)
        P = np.full((k, k), 0.05 / max(k - 1, 1))
        np.fill_diagonal(P, 0.95)
        _ = phi_ols
        return c, phi, sig, P

    def _em(self, x0, x1, init):
        k = self.n_states
        c, phi, sig, P = (a.copy() for a in init)
        pi0 = _stationary(P)
        floor = 1e-4 * x1.std()
        prev = -np.inf
        trace = []
        for _ in range(self.n_iter):
            logb = _log_emission(x0, x1, c, phi, sig)
            alpha, scale, ll = _forward(logb, P, pi0)
            beta = _backward(logb, P, scale)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = _xi_sum(alpha, beta, logb, P, scale)

            P = xi / xi.sum(axis=1, keepdims=True)
            pi0 = gamma[0]
            for j in range(k):
                w = gamma[:, j]
                W = w.sum()
                if W < 5:
                    continue
                sx, sy = w @ x0, w @ x1
                sxx, sxy = w @ (x0 * x0), w @ (x0 * x1)
                den = W * sxx - sx * sx
                phi[j] = np.clip((W * sxy - sx * sy) / den, -0.5, 0.9995) if den > 1e-12 else phi[j]
                c[j] = (sy - phi[j] * sx) / W
                r = x1 - c[j] - phi[j] * x0
                sig[j] = max(np.sqrt((w @ (r * r)) / W), floor)
            trace.append(ll)
            if abs(ll - prev) < self.tol * (1.0 + abs(ll)):
                break
            prev = ll
        logb = _log_emission(x0, x1, c, phi, sig)
        _, _, ll = _forward(logb, P, pi0)
        return MSAR1Params(c, phi, sig, P), ll, trace

    def fit(self, x: np.ndarray) -> MarkovSwitchingAR1:
        x = np.asarray(x, dtype=float)
        x0, x1 = x[:-1], x[1:]
        rng = np.random.default_rng(self.seed)
        best = None
        for s in range(self.n_starts):
            try:
                params, ll, trace = self._em(x0, x1, self._init(x0, x1, rng, s))
            except FloatingPointError:
                continue
            if np.isfinite(ll) and (best is None or ll > best[1]):
                best = (params, ll, trace)
        if best is None:
            raise RuntimeError("EM failed for every start")
        params, ll, trace = best
        order = np.argsort(params.sigma_eq)
        params = MSAR1Params(params.c[order], params.phi[order], params.sigma[order],
                             params.P[np.ix_(order, order)])  # fmt: skip
        self.params_, self.loglike_, self.ll_trace_ = params, ll, trace
        return self

    @classmethod
    def from_params(cls, params: MSAR1Params) -> MarkovSwitchingAR1:
        """Wrap fixed parameters (e.g. an informative prior) so `filter` can be used."""
        obj = cls(n_states=len(params.phi))
        obj.params_ = params
        return obj

    def _require(self) -> MSAR1Params:
        if self.params_ is None:
            raise RuntimeError("call fit() first")
        return self.params_

    def filter(self, x: np.ndarray) -> np.ndarray:
        """P(S_t = k | x_{1:t}) for every t; row 0 is the stationary distribution."""
        p = self._require()
        x = np.asarray(x, dtype=float)
        logb = _log_emission(x[:-1], x[1:], p.c, p.phi, p.sigma)
        pi0 = _stationary(p.P)
        alpha, _, _ = _forward(logb, p.P, pi0)
        return np.vstack([pi0, alpha])

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """Two-sided probabilities. Diagnostics only - this uses future data."""
        p = self._require()
        x = np.asarray(x, dtype=float)
        logb = _log_emission(x[:-1], x[1:], p.c, p.phi, p.sigma)
        pi0 = _stationary(p.P)
        alpha, scale, _ = _forward(logb, p.P, pi0)
        beta = _backward(logb, p.P, scale)
        g = alpha * beta
        g /= g.sum(axis=1, keepdims=True)
        return np.vstack([pi0, g])


def fit_single_ar1(x: np.ndarray) -> tuple[float, float, float]:
    """Plain AR(1) log-likelihood and BIC (same conditioning as the switching model)."""
    x = np.asarray(x, dtype=float)
    x0, x1 = x[:-1], x[1:]
    phi, c = np.polyfit(x0, x1, 1)
    r = x1 - c - phi * x0
    s2 = float(r @ r) / len(r)
    n = len(r)
    ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1.0)
    bic = 3 * np.log(n) - 2.0 * ll
    return float(ll), float(bic), float(phi)
