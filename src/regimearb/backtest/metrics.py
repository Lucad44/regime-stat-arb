from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def performance_summary(returns: pd.Series, periods: int = TRADING_DAYS) -> dict:
    r = returns.dropna()
    if len(r) < 3 or r.std() == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0, sortino=0.0, max_dd=0.0, calmar=0.0,
                    skew=0.0, kurtosis=3.0, hit_rate=0.0, n_obs=len(r))  # fmt: skip
    ann_ret = float((1.0 + r).prod() ** (periods / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(periods))
    downside = r[r < 0].std()
    mdd = max_drawdown(r)
    return dict(
        ann_return=ann_ret,
        ann_vol=vol,
        sharpe=float(r.mean() / r.std() * np.sqrt(periods)),
        sortino=float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else np.inf,
        max_dd=mdd,
        calmar=float(ann_ret / abs(mdd)) if mdd < 0 else np.inf,
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
        hit_rate=float((r > 0).mean()),
        n_obs=int(len(r)),
    )
