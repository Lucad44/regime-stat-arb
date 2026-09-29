"""Parameter sweep over the trading rule, reusing one signal panel."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd

from .backtest.engine import SignalPanel, simulate
from .config import Config
from .strategy.signals import StrategyParams

ENTRY = [("bertram", 0.0), ("fixed", 1.25), ("fixed", 1.5), ("fixed", 2.0)]
EXITS = [0.0, 0.25, 0.5]
STOPS = [3.0, 4.0, 5.0]
GATES = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.9, 0.0), (0.9, 0.3)]  # (gate_in, gate_out)


def default_grid(base: StrategyParams) -> list[StrategyParams]:
    out = []
    for (mode, z), ze, zs, (gi, go) in itertools.product(ENTRY, EXITS, STOPS, GATES):
        out.append(dataclasses.replace(
            base, entry_mode=mode, z_entry=z or base.z_entry, z_exit=ze, z_stop=zs,
            gate_in=gi, gate_out=go, prob_sizing=gi > 0,
        ))  # fmt: skip
    return out


def gate_label(gi: float, go: float) -> str:
    return "none" if gi == 0 else f"{gi:g}/{go:g}"


def run_sweep(panel: SignalPanel, cfg: Config, grid: list[StrategyParams] | None = None):
    grid = grid or default_grid(cfg.strategy)
    rets, rows = [], []
    for i, p in enumerate(grid):
        res = simulate(panel, p, cfg.costs, cfg.portfolio, keep_trades=False)
        s = res.summary()
        rows.append(dict(
            id=i, entry=p.entry_mode if p.entry_mode == "bertram" else f"z={p.z_entry:g}",
            z_exit=p.z_exit, z_stop=p.z_stop, gate=gate_label(p.gate_in, p.gate_out),
            sharpe=s["sharpe"], gross_sharpe=s["gross_sharpe"], ann_return=s["ann_return"],
            max_dd=s["max_dd"], ann_turnover=s["ann_turnover"],
        ))  # fmt: skip
        rets.append(res.returns.values)
    table = pd.DataFrame(rows)
    return table, np.column_stack(rets)
