"""Combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado, Zhu).

Given the return series of N strategy configurations over the same T periods:
cut time into S blocks, and for every way of choosing S/2 blocks as "in-sample"
(the rest are "out-of-sample"), pick the configuration with the best IS Sharpe
and look at where it lands in the OOS ranking. The probability of backtest
overfitting (PBO) is the share of splits where the IS winner ends up at or below
the OOS median.

Block sufficient statistics (n, sum, sum of squares) make every split a couple of
small matrix products, so all C(16, 8) = 12,870 splits run in well under a second.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class CSCVResult:
    pbo: float
    logits: np.ndarray
    is_best_sharpe: np.ndarray  # annualised
    oos_best_sharpe: np.ndarray  # annualised
    prob_oos_loss: float
    slope: float  # OOS = intercept + slope * IS, over the IS winners
    intercept: float
    n_splits: int
    n_configs: int


def _sharpe_from_blocks(n_b, s_b, ss_b, mask):
    n = mask @ n_b
    s = mask @ s_b
    ss = mask @ ss_b
    mean = s / n[:, None]
    var = (ss / n[:, None] - mean**2) * n[:, None] / (n[:, None] - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = mean / np.sqrt(var)
    return np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)


def cscv(
    returns: np.ndarray, n_blocks: int = 16, max_splits: int | None = None, seed: int = 0
) -> CSCVResult:
    r = np.asarray(returns, dtype=float)
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    T, N = r.shape
    if N < 2:
        raise ValueError("need at least two configurations")
    per = T // n_blocks
    r = r[T - per * n_blocks :]  # drop the oldest remainder so blocks are equal length
    blocks = r.reshape(n_blocks, per, N)
    n_b = np.full(n_blocks, float(per))
    s_b = blocks.sum(axis=1)
    ss_b = (blocks**2).sum(axis=1)

    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_splits and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    M = len(combos)
    mask = np.zeros((M, n_blocks))
    for i, c in enumerate(combos):
        mask[i, list(c)] = 1.0

    sr_is = _sharpe_from_blocks(n_b, s_b, ss_b, mask)
    sr_oos = _sharpe_from_blocks(n_b, s_b, ss_b, 1.0 - mask)

    best = sr_is.argmax(axis=1)
    rows = np.arange(M)
    oos_of_best = sr_oos[rows, best]
    rank = (sr_oos < oos_of_best[:, None]).sum(axis=1) + 1  # 1 = worst, N = best
    w = rank / (N + 1.0)
    logits = np.log(w / (1.0 - w))

    is_best = sr_is[rows, best]
    slope, intercept = np.polyfit(is_best, oos_of_best, 1) if np.ptp(is_best) > 0 else (0.0, 0.0)
    ann = np.sqrt(252)
    return CSCVResult(
        pbo=float((logits <= 0).mean()),
        logits=logits,
        is_best_sharpe=is_best * ann,
        oos_best_sharpe=oos_of_best * ann,
        prob_oos_loss=float((oos_of_best < 0).mean()),
        slope=float(slope),
        intercept=float(intercept * ann),
        n_splits=M,
        n_configs=N,
    )
