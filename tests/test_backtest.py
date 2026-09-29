import dataclasses

import numpy as np

from regimearb.backtest import build_signal_panel, simulate
from regimearb.config import CostConfig
from regimearb.data import SyntheticSpec, make_universe
from regimearb.strategy import StrategyParams, fit_pair, generate_positions


def test_state_machine_basic_round_trip():
    z = np.array([0, -0.5, -2.5, -1.5, -0.6, 0.1, 0.2, 1.0])
    p = np.ones_like(z)
    pos, reason = generate_positions(z, p, 2.0, 5.0, StrategyParams(entry_mode="fixed", z_entry=2.0, z_exit=0.25))
    assert list(pos[:2]) == [0, 0]
    assert pos[2] == 1 and pos[3] == 1 and pos[4] == 1  # long while below the mean
    assert pos[5] == 0 and reason[5] == 1  # closed at target
    assert pos[6] == 0 and pos[7] == 0


def test_stop_loss_and_regime_exit():
    prm = StrategyParams(entry_mode="fixed", z_entry=2.0, z_stop=4.0, gate_in=0.5, gate_out=0.3)
    z = np.array([0, 2.5, 3.0, 4.5, 3.0])
    pos, reason = generate_positions(z, np.ones(5), 2.0, 5.0, prm)
    assert pos[1] == -1 and reason[3] == 2
    p = np.array([1, 1, 1, 0.1, 1.0])
    pos, reason = generate_positions(np.array([0, 2.5, 3.0, 3.0, 3.0]), p, 2.0, 5.0, prm)
    assert reason[3] == 3 and pos[3] == 0


def test_no_entry_when_regime_gate_closed():
    prm = StrategyParams(entry_mode="fixed", z_entry=2.0, gate_in=0.7)
    pos, _ = generate_positions(np.array([0, -3.0, -3.0]), np.array([1, 0.5, 0.5]), 2.0, 5.0, prm)
    assert np.all(pos == 0)


def test_pair_signals_have_no_lookahead():
    prices, truth = make_universe(SyntheticSpec(n_days=900, n_leaders=6, n_planted=2, seed=3))
    f, l = truth.pairs[0]
    lp = np.log(prices)
    y, x = lp[f].values, lp[l].values
    fit = fit_pair(y[:400], x[:400], hl_bounds=(1.0, 200.0))
    assert fit is not None
    full = fit.signals(y[:600], x[:600])
    cut = fit.signals(y[:500], x[:500])
    for name in ("beta", "spread", "z", "p_mr"):
        assert np.allclose(getattr(full, name)[:500], getattr(cut, name)), name
    # and a shock to the future must not touch the past
    y2 = y[:600].copy()
    y2[520:] += 0.5
    shocked = fit.signals(y2, x[:600])
    assert np.allclose(shocked.z[:520], full.z[:520])


def test_costs_only_reduce_returns_and_zero_cost_equals_gross(small_universe, small_cfg):
    prices, _ = small_universe
    panel = build_signal_panel(prices, small_cfg)
    base = simulate(panel, small_cfg.strategy, CostConfig(0.0, 0.0), small_cfg.portfolio)
    assert np.allclose(base.returns.values, base.gross.values)
    costly = simulate(panel, small_cfg.strategy, CostConfig(20.0, 100.0), small_cfg.portfolio)
    assert costly.returns.sum() <= base.returns.sum()
    assert np.allclose(costly.gross.values, base.gross.values)


def test_execution_lag_shifts_pnl_by_one_bar(small_universe, small_cfg):
    prices, _ = small_universe
    panel = build_signal_panel(prices, small_cfg)
    free = CostConfig(0.0, 0.0)
    r0 = simulate(panel, dataclasses.replace(small_cfg.strategy, execution_lag=0), free, small_cfg.portfolio)
    r1 = simulate(panel, dataclasses.replace(small_cfg.strategy, execution_lag=1), free, small_cfg.portfolio)
    assert not np.allclose(r0.gross.values, r1.gross.values)
    assert (r1.n_open > 0).sum() > 0


def test_flat_book_earns_nothing(small_universe, small_cfg):
    prices, _ = small_universe
    panel = build_signal_panel(prices, small_cfg)
    never = dataclasses.replace(small_cfg.strategy, entry_mode="fixed", z_entry=50.0, z_stop=100.0)
    res = simulate(panel, never, small_cfg.costs, small_cfg.portfolio)
    assert np.allclose(res.returns.values, 0.0)
    assert len(res.trades) == 0


def test_signal_panel_ignores_data_after_each_window(small_universe, small_cfg):
    """Truncating the price history must leave every earlier window's signals unchanged."""
    prices, _ = small_universe
    full = build_signal_panel(prices, small_cfg)
    n_keep = 4  # windows 0-3 sit entirely inside the truncated history
    cut_at = small_cfg.walk_forward.formation + n_keep * small_cfg.walk_forward.step
    part = build_signal_panel(prices.iloc[:cut_at], small_cfg)
    kept_full = [w for w in full.windows if w.window < n_keep]
    kept_part = [w for w in part.windows if w.window < n_keep]
    assert len(kept_full) == len(kept_part) > 0
    for a, b in zip(kept_full, kept_part, strict=True):
        assert (a.y, a.x) == (b.y, b.x)
        assert np.allclose(a.z, b.z)
