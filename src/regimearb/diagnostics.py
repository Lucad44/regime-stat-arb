"""Model checks against known ground truth (only possible on simulated data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data.synthetic import SyntheticTruth
from .models.kalman import OUStateSpaceHedge
from .strategy.pair_model import fit_pair


def auc(score: np.ndarray, label: np.ndarray) -> float:
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classic_kalman_beta(y, x, delta=1e-4, ve=None, n_train=None):
    """The standard random-walk [beta, alpha] Kalman filter used in most pairs-trading
    write-ups, as a baseline. Innovation variance defaults to the OLS residual variance."""
    y = np.asarray(y)
    x = np.asarray(x)
    if ve is None:
        n_train = n_train or len(y)
        b, a = np.polyfit(x[:n_train], y[:n_train], 1)
        ve = float(np.var(y[:n_train] - b * x[:n_train] - a))
    vw = delta / (1.0 - delta)
    th = np.array([np.polyfit(x[:60], y[:60], 1)[0], 0.0])
    P = np.eye(2) * 1e-2
    out = np.empty(len(y))
    for t in range(len(y)):
        P = P + vw * np.eye(2)
        H = np.array([x[t], 1.0])
        S = H @ P @ H + ve
        K = P @ H / S
        th = th + K * (y[t] - H @ th)
        P = P - np.outer(K, H @ P)
        out[t] = th[0]
    return out


def rolling_ols_beta(y, x, win):
    out = np.full(len(y), np.nan)
    for t in range(win, len(y)):
        out[t] = np.polyfit(x[t - win + 1 : t + 1], y[t - win + 1 : t + 1], 1)[0]
    return out


def _rmse(est: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((est - truth) ** 2)))


def recovery_study(
    prices: pd.DataFrame, truth: SyntheticTruth, formation: int = 504, horizon: int = 126,
    starts: tuple[int, ...] | None = None,
):  # fmt: skip
    """Fit on a formation window, evaluate on the next `horizon` days, for every planted pair."""
    lp = np.log(prices)
    n = len(prices)
    starts = starts or tuple(range(formation, n - horizon, horizon * 3))
    err_rows, auc_list, hl_rows = [], [], []
    for t0 in starts:
        for f, l in truth.pairs:
            y, x = lp[f].values, lp[l].values
            tr, te = slice(t0 - formation, t0), slice(t0, t0 + horizon)
            seg = slice(t0 - formation, t0 + horizon)
            b_true = truth.beta[f].values[te]

            ss = OUStateSpaceHedge.fit(y[tr], x[tr])
            b_ss = ss.filter(y[seg], x[seg]).beta[formation:]
            b_ols = np.full(horizon, np.polyfit(x[tr], y[tr], 1)[0])
            b_kal = classic_kalman_beta(y[seg], x[seg], n_train=formation)[formation:]
            roll = rolling_ols_beta(y[seg], x[seg], 120)[formation:]
            err_rows.append({
                "static OLS": _rmse(b_ols, b_true),
                "rolling OLS 120d": _rmse(roll, b_true),
                "classic RW Kalman": _rmse(b_kal, b_true),
                "OU state-space": _rmse(b_ss, b_true),
            })

            fit = fit_pair(y[tr], x[tr], hl_bounds=(1.0, 200.0))
            if fit is None:
                continue
            p = fit.signals(y[seg], x[seg]).p_mr[formation:]
            a = auc(p, (truth.regime[f].values[te] == 0).astype(int))
            if np.isfinite(a):
                auc_list.append(a)
            hl_rows.append({"true": truth.params[f]["halflife"], "est": fit.half_life})
    return pd.DataFrame(err_rows), np.array(auc_list), pd.DataFrame(hl_rows)


def selection_quality(selections: pd.DataFrame, truth: SyntheticTruth) -> dict:
    if selections.empty:
        return dict(precision=np.nan, recall=np.nan, windows=0)
    tset = truth.pair_set()
    hit = selections.apply(lambda r: frozenset((r.y, r.x)) in tset, axis=1)
    per_w = selections.assign(hit=hit).groupby("window")["hit"].sum()
    n_windows = int(selections["window"].max()) + 1
    return dict(
        precision=float(hit.mean()),
        recall=float(per_w.sum() / (len(truth.pairs) * n_windows)),
        planted_per_window=float(per_w.mean()),
        pairs_traded=int(len(selections)),
    )
