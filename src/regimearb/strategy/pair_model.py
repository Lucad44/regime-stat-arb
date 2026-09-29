"""Everything that is estimated per pair on a formation window, bundled up."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..models.kalman import OUStateSpaceHedge
from ..models.ou import bertram_entry, fit_ou
from ..models.regime import MarkovSwitchingAR1, MSAR1Params, fit_single_ar1

log = logging.getLogger(__name__)

BIC_MARGIN = 2.0  # "positive" evidence (Kass & Raftery) before trusting a regime split
WARMUP = 20
# Generic stress prior, used when the formation window shows no breakdown to learn from.
# (Cointegration screens preferentially pick windows without breakdowns, so this is the
# common case, not the exception.) Deliberately not tuned to any particular data set.
PRIOR_PHI_BREAK = 0.99
PRIOR_VOL_MULT = 1.5
PRIOR_P_STAY = (0.99, 0.93)  # (calm, stressed) daily persistence
MIN_DISPERSION_RATIO = 1.25  # stressed state must be visibly wider than the calm one


@dataclass
class PairSignals:
    beta: np.ndarray
    spread: np.ndarray
    z: np.ndarray
    p_mr: np.ndarray


@dataclass
class PairFit:
    ss: OUStateSpaceHedge
    regime: MarkovSwitchingAR1 | None
    mu: float  # long-run mean / stationary std of the spread *in the mean-reverting regime*
    sigma_eq: float
    half_life: float
    z_entry_analytic: float
    bic_gain: float  # BIC(single) - BIC(switching); > BIC_MARGIN means the split is learned
    regime_source: str = "fitted"  # "fitted" or "prior"

    @property
    def uses_regimes(self) -> bool:
        return self.regime_source == "fitted"

    def signals(self, y: np.ndarray, x: np.ndarray) -> PairSignals:
        """Run the frozen model forward. Causal: output[:k] depends on y[:k], x[:k] only."""
        out = self.ss.filter(y, x)
        z = (out.spread - self.mu) / self.sigma_eq
        p = self.regime.filter(out.spread)[:, 0]
        return PairSignals(out.beta, out.spread, z, p)


def fit_pair(
    y: np.ndarray,
    x: np.ndarray,
    cost_bps: float = 8.0,
    hl_bounds: tuple[float, float] = (2.0, 60.0),
    seed: int = 0,
) -> PairFit | None:
    try:
        ss = OUStateSpaceHedge.fit(y, x)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        log.debug("state-space fit failed: %s", exc)
        return None

    lo, hi = hl_bounds
    if not lo <= ss.params.half_life <= hi:
        return None

    out = ss.filter(y, x)
    s = out.spread[WARMUP:]
    n_obs = len(s) - 1

    _, bic_single, _ = fit_single_ar1(s)
    regime = None
    source = "fitted"
    bic_gain = 0.0
    try:
        ms = MarkovSwitchingAR1(seed=seed).fit(s)
        bic_gain = bic_single - ms.bic(n_obs)
        if bic_gain > BIC_MARGIN and ms.params_.sigma_eq[1] > MIN_DISPERSION_RATIO * ms.params_.sigma_eq[0]:
            regime = ms
    except (RuntimeError, FloatingPointError):
        pass

    if regime is not None:
        prm = regime.params_
        mu, sig, phi = float(prm.mu[0]), float(prm.sigma_eq[0]), float(prm.phi[0])
    else:
        f = fit_ou(s)
        mu, sig, phi = f.mu, f.sigma_eq, f.phi
        source = "prior"
        eta = sig * np.sqrt(1.0 - phi**2)
        c_calm, c_stress = mu * (1.0 - phi), mu * (1.0 - PRIOR_PHI_BREAK)
        P = np.array([[PRIOR_P_STAY[0], 1 - PRIOR_P_STAY[0]], [1 - PRIOR_P_STAY[1], PRIOR_P_STAY[1]]])
        regime = MarkovSwitchingAR1.from_params(MSAR1Params(
            c=np.array([c_calm, c_stress]), phi=np.array([phi, PRIOR_PHI_BREAK]),
            sigma=np.array([eta, PRIOR_VOL_MULT * eta]), P=P))  # fmt: skip

    if not 0.0 < phi < 1.0:
        return None
    hl = float(np.log(2.0) / -np.log(phi))
    if not lo <= hl <= hi:
        return None

    beta_bar = float(np.mean(np.abs(out.beta[WARMUP:])))
    round_trip = 2.0 * cost_bps / 1e4 * (1.0 + beta_bar)  # spread units (log-price of y)
    k_star = bertram_entry(round_trip / sig)
    return PairFit(ss, regime, mu, sig, hl, k_star, bic_gain, source)
