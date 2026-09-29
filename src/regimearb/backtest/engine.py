"""Walk-forward engine, split into two stages.

Stage 1 (`build_signal_panel`) is the expensive part: at every re-formation
date it screens pairs, fits the state-space and regime models on the trailing
window, and runs the frozen models forward over the next `step` days. Nothing
in stage 1 depends on the trading rule, so it is done once.

Stage 2 (`simulate`) turns the stored signals into positions and P&L for a
given StrategyParams / cost model. That is cheap, which is what makes the
parameter sweep (and the overfitting diagnostics built on top of it) feasible.

Timing convention: a position decided at the close of bar t is filled at the
close of bar t + execution_lag and earns from the following bar. With the
default lag of 1 there is a full extra day between seeing a signal and owning
it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, CostConfig, PortfolioConfig
from ..selection.pairs import screen_pairs
from ..strategy.pair_model import fit_pair
from ..strategy.signals import EXIT_NAMES, StrategyParams, generate_positions
from .metrics import TRADING_DAYS, performance_summary

log = logging.getLogger(__name__)


@dataclass
class PairWindow:
    y: str
    x: str
    window: int
    index: pd.DatetimeIndex
    beta: np.ndarray
    spread: np.ndarray
    z: np.ndarray
    p_mr: np.ndarray
    ret_y: np.ndarray
    ret_x: np.ndarray
    z_entry_analytic: float
    half_life: float
    uses_regimes: bool
    qvalue: float


@dataclass
class SignalPanel:
    windows: list[PairWindow]
    index: pd.DatetimeIndex
    selections: pd.DataFrame  # one row per (window, pair) that was screened in
    max_pairs: int


@dataclass
class BacktestResult:
    returns: pd.Series  # net, after costs and vol-scaling
    gross: pd.Series
    costs: pd.Series
    turnover: pd.Series
    n_open: pd.Series
    trades: pd.DataFrame
    params: StrategyParams
    scale: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> dict:
        s = performance_summary(self.returns)
        s["gross_sharpe"] = performance_summary(self.gross)["sharpe"]
        s["ann_turnover"] = float(self.turnover.mean() * TRADING_DAYS)
        s["ann_cost_drag"] = float(self.costs.mean() * TRADING_DAYS)
        s["n_trades"] = int(len(self.trades))
        if len(self.trades):
            s["avg_hold_days"] = float(self.trades["days"].mean())
            s["trade_win_rate"] = float((self.trades["pnl"] > 0).mean())
        return s


def build_signal_panel(prices: pd.DataFrame, cfg: Config) -> SignalPanel:
    wf = cfg.walk_forward
    lp = np.log(prices)
    rets = prices.pct_change().fillna(0.0)
    n = len(prices)

    windows: list[PairWindow] = []
    sel_rows = []
    for w, t0 in enumerate(range(wf.formation, n - 5, wf.step)):
        t1 = min(t0 + wf.step, n)
        train = lp.iloc[t0 - wf.formation : t0]
        chosen = screen_pairs(
            train, n_factors=wf.n_factors, top_k=wf.top_k, fdr=wf.fdr,
            hl_bounds=(wf.hl_min, wf.hl_max), max_pairs=wf.max_pairs, max_reuse=wf.max_reuse,
        )  # fmt: skip
        fitted = 0
        for r in chosen.itertuples():
            seg = lp.iloc[t0 - wf.formation : t1]
            yv, xv = seg[r.y].values, seg[r.x].values
            fit = fit_pair(yv[: wf.formation], xv[: wf.formation], cost_bps=cfg.costs.cost_bps,
                           hl_bounds=(wf.hl_min, wf.hl_max), seed=w)  # fmt: skip
            if fit is None:
                continue
            sig = fit.signals(yv, xv)
            sl = slice(wf.formation, None)
            windows.append(
                PairWindow(
                    y=r.y, x=r.x, window=w, index=seg.index[sl],
                    beta=sig.beta[sl], spread=sig.spread[sl], z=sig.z[sl], p_mr=sig.p_mr[sl],
                    ret_y=rets[r.y].iloc[t0:t1].values, ret_x=rets[r.x].iloc[t0:t1].values,
                    z_entry_analytic=fit.z_entry_analytic, half_life=fit.half_life,
                    uses_regimes=fit.uses_regimes, qvalue=r.qvalue,
                )
            )  # fmt: skip
            sel_rows.append(dict(window=w, start=prices.index[t0], y=r.y, x=r.x, qvalue=r.qvalue,
                                 half_life=fit.half_life, regimes=fit.uses_regimes,
                                 bic_gain=fit.bic_gain))  # fmt: skip
            fitted += 1
        log.info("window %d [%s]: %d screened, %d fitted", w, prices.index[t0].date(),
                 len(chosen), fitted)  # fmt: skip

    idx = prices.index[wf.formation :]
    return SignalPanel(windows, idx, pd.DataFrame(sel_rows), wf.max_pairs)


def _shift(a: np.ndarray, k: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill, dtype=float)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def _extract_trades(pw: PairWindow, pos, reason, net, lag, slot_w) -> list[dict]:
    rows = []
    n = len(pos)
    i = 0
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        side = np.sign(pos[i])
        j = i
        while j < n and np.sign(pos[j]) == side:
            j += 1
        a, b = i + lag, min(j + lag, n)  # held span after the fill delay
        if a < n:
            rows.append(dict(
                y=pw.y, x=pw.x, window=pw.window, entry=pw.index[a],
                exit=pw.index[b - 1], days=b - a, side=int(side),
                pnl=float(net[a:b].sum() * slot_w),
                exit_reason=EXIT_NAMES[int(reason[j])] if j < n else "window_end",
            ))  # fmt: skip
        i = j
    return rows


def simulate(
    panel: SignalPanel,
    params: StrategyParams,
    costs: CostConfig | None = None,
    portfolio: PortfolioConfig | None = None,
    keep_trades: bool = True,
) -> BacktestResult:
    costs = costs or CostConfig()
    portfolio = portfolio or PortfolioConfig()
    idx = panel.index
    pos_of = {d: i for i, d in enumerate(idx)}
    gross = np.zeros(len(idx))
    cost = np.zeros(len(idx))
    turn = np.zeros(len(idx))
    n_open = np.zeros(len(idx))
    slot_w = portfolio.gross_leverage / panel.max_pairs
    lag = 1 + params.execution_lag
    trades: list[dict] = []

    for pw in panel.windows:
        pos, reason = generate_positions(pw.z, pw.p_mr, pw.z_entry_analytic, pw.half_life, params)
        held = _shift(pos, lag)
        b = _shift(pw.beta, lag, fill=pw.beta[0])
        denom = 1.0 + np.abs(b)
        w_y = held / denom
        w_x = -held * b / denom

        g = w_y * pw.ret_y + w_x * pw.ret_x
        tv = np.abs(np.diff(w_y, prepend=0.0)) + np.abs(np.diff(w_x, prepend=0.0))
        tv[-1] += abs(w_y[-1]) + abs(w_x[-1])  # everything is flat at the next re-formation
        c = tv * costs.cost_bps / 1e4
        borrow = (np.maximum(-w_y, 0) + np.maximum(-w_x, 0)) * costs.borrow_bps / 1e4 / TRADING_DAYS

        ii = np.fromiter((pos_of[d] for d in pw.index), dtype=int, count=len(pw.index))
        gross[ii] += slot_w * g
        cost[ii] += slot_w * (c + borrow)
        turn[ii] += slot_w * tv
        n_open[ii] += (held != 0)
        if keep_trades:
            net_pair = g - c - borrow
            trades += _extract_trades(pw, pos, reason, net_pair, lag, slot_w)

    net = gross - cost
    scale = np.ones(len(idx))
    if portfolio.vol_target:
        lam = 0.5 ** (1.0 / portfolio.vol_halflife)
        var = np.empty(len(idx))
        v = np.nan
        for t in range(len(idx)):
            var[t] = v
            v = net[t] ** 2 if np.isnan(v) else lam * v + (1 - lam) * net[t] ** 2
        vol = np.sqrt(var * TRADING_DAYS)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = portfolio.vol_target / vol
        cap = portfolio.max_leverage / portfolio.gross_leverage
        scale = np.where(np.isfinite(raw) & (np.arange(len(idx)) > 20), np.clip(raw, 0, cap), 1.0)

    as_s = lambda a: pd.Series(a * scale, index=idx)  # noqa: E731
    tdf = pd.DataFrame(trades)
    return BacktestResult(
        returns=as_s(net), gross=as_s(gross), costs=as_s(cost), turnover=as_s(turn),
        n_open=pd.Series(n_open, index=idx), trades=tdf, params=params,
        scale=pd.Series(scale, index=idx),
    )  # fmt: skip
