"""Candidate pair selection on a formation window.

Testing all N(N-1)/2 pairs for cointegration and keeping the small p-values is
a textbook multiple-testing trap: with 500 pairs at 5% you expect 25 "discoveries"
from pure noise. Two guards here:

  1. A cheap pre-screen on *factor-residual* correlation. Raw return
     correlation is dominated by the market and sector factors, which says
     nothing about a pair-specific long-run link, so the leading principal
     components are removed first.
  2. Benjamini-Hochberg FDR control over the pairs that reach the
     Engle-Granger test, instead of a raw p < 0.05 cut.
"""

from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from ..models.ou import half_life

log = logging.getLogger(__name__)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _residual_corr(log_prices: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    rets = log_prices.diff().dropna()
    z = (rets - rets.mean()) / rets.std()
    u, s, vt = np.linalg.svd(z.values, full_matrices=False)
    k = min(n_factors, len(s))
    resid = z.values - (u[:, :k] * s[:k]) @ vt[:k]
    return pd.DataFrame(np.corrcoef(resid, rowvar=False), index=z.columns, columns=z.columns)


def _eg_pvalue(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Engle-Granger both ways; take the better direction and pay a factor 2 for it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_ab = coint(a, b, trend="c", maxlag=1, autolag=None)[1]
        p_ba = coint(b, a, trend="c", maxlag=1, autolag=None)[1]
    return (min(2.0 * min(p_ab, p_ba), 1.0), "ab" if p_ab <= p_ba else "ba")


def screen_pairs(
    log_prices: pd.DataFrame,
    n_factors: int = 3,
    top_k: int = 120,
    fdr: float = 0.10,
    hl_bounds: tuple[float, float] = (2.0, 60.0),
    max_pairs: int = 10,
    max_reuse: int = 2,
) -> pd.DataFrame:
    """Return selected pairs as a frame with columns y, x, corr, pvalue, qvalue, hl_ols."""
    rc = _residual_corr(log_prices, n_factors)
    names = list(rc.columns)
    cand = [(a, b, rc.loc[a, b]) for a, b in itertools.combinations(names, 2)]
    cand.sort(key=lambda t: -abs(t[2]))
    cand = cand[:top_k]

    rows = []
    for a, b, c in cand:
        pv, direction = _eg_pvalue(log_prices[a].values, log_prices[b].values)
        y, x = (a, b) if direction == "ab" else (b, a)
        beta, alpha = np.polyfit(log_prices[x].values, log_prices[y].values, 1)
        spread = log_prices[y].values - beta * log_prices[x].values - alpha
        rows.append((y, x, c, pv, half_life(spread)))
    if not rows:
        return pd.DataFrame(columns=["y", "x", "corr", "pvalue", "qvalue", "hl_ols"])

    df = pd.DataFrame(rows, columns=["y", "x", "corr", "pvalue", "hl_ols"])
    df["qvalue"] = benjamini_hochberg(df["pvalue"].values)
    lo, hi = hl_bounds
    df = df[(df["qvalue"] <= fdr) & df["hl_ols"].between(lo, hi)]
    df = df.sort_values(["qvalue", "pvalue"]).reset_index(drop=True)

    # don't let one asset dominate the book
    use: dict[str, int] = {}
    keep = []
    for i, r in df.iterrows():
        if use.get(r.y, 0) >= max_reuse or use.get(r.x, 0) >= max_reuse:
            continue
        use[r.y] = use.get(r.y, 0) + 1
        use[r.x] = use.get(r.x, 0) + 1
        keep.append(i)
        if len(keep) >= max_pairs:
            break
    out = df.loc[keep].reset_index(drop=True)
    log.debug("screen: %d candidates -> %d after FDR/hl -> %d kept", len(cand), len(df), len(out))
    return out[["y", "x", "corr", "pvalue", "qvalue", "hl_ols"]]
