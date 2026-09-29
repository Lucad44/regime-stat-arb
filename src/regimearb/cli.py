from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from . import viz
from .backtest import build_signal_panel, simulate
from .config import Config
from .data import SyntheticSpec, clean_prices, load_csv, make_universe
from .diagnostics import recovery_study, selection_quality
from .sweep import run_sweep
from .validation import cscv, deflated_sharpe_ratio, min_track_record_length

log = logging.getLogger("regimearb")


def _load(cfg: Config, csv: str | None):
    d = cfg.data
    if csv or d.source == "csv":
        return clean_prices(load_csv(csv or d.csv_path)), None
    spec = SyntheticSpec(n_days=d.n_days, n_leaders=d.n_leaders, n_planted=d.n_planted, seed=d.seed)
    return make_universe(spec)


def _fmt(s: dict) -> str:
    keys = ["sharpe", "gross_sharpe", "ann_return", "ann_vol", "max_dd", "ann_turnover",
            "n_trades", "trade_win_rate", "avg_hold_days"]  # fmt: skip
    return "  ".join(f"{k}={s[k]:.3f}" if isinstance(s.get(k), float) else f"{k}={s.get(k)}"
                     for k in keys if k in s)  # fmt: skip


def cmd_run(args, cfg: Config) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prices, truth = _load(cfg, args.csv)
    log.info("universe: %d days x %d assets", *prices.shape)
    panel = build_signal_panel(prices, cfg)
    base = cfg.strategy
    ungated = dict(gate_in=0.0, gate_out=0.0, prob_sizing=False)
    variants = {
        "gated, analytic entry (1.0 sigma floor)": base,
        "ungated, analytic entry (1.0 sigma floor)": dataclasses.replace(base, **ungated),
        "ungated, unclipped analytic entry": dataclasses.replace(base, z_entry_floor=0.5, **ungated),
        "ungated, fixed 2.0 sigma entry": dataclasses.replace(
            base, entry_mode="fixed", z_entry=2.0, **ungated),
    }
    results = {k: simulate(panel, v, cfg.costs, cfg.portfolio) for k, v in variants.items()}
    summary = {k: r.summary() for k, r in results.items()}
    for k, s in summary.items():
        print(f"{k:44s} {_fmt(s)}")

    if truth is not None:
        q = selection_quality(panel.selections, truth)
        summary["selection_vs_truth"] = q
        print("selection vs planted pairs:", {k: round(v, 3) for k, v in q.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    first = next(iter(results.values()))
    first.trades.to_csv(out / "trades.csv", index=False)
    panel.selections.to_csv(out / "selections.csv", index=False)
    viz.plot_equity({k: r.returns for k, r in results.items()}, out / "equity.png")

    if panel.windows:
        tset = truth.pair_set() if truth else set()
        pick = next((w for w in panel.windows if frozenset((w.y, w.x)) in tset), panel.windows[0])
        follower = None
        if truth:
            follower = next((f for f, l in truth.pairs if {f, l} == {pick.y, pick.x}), None)
        viz.plot_pair(pick, truth, out / "pair_example.png", follower)
    print(f"wrote results to {out}/")


def cmd_sweep(args, cfg: Config) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prices, _ = _load(cfg, args.csv)
    panel = build_signal_panel(prices, cfg)
    table, R = run_sweep(panel, cfg)
    table.to_csv(out / "sweep_results.csv", index=False)
    n = len(table)

    per_period = R.mean(axis=0) / np.where(R.std(axis=0, ddof=1) > 0, R.std(axis=0, ddof=1), np.nan)
    per_period = np.nan_to_num(per_period)
    best = int(np.argmax(per_period))
    dsr = deflated_sharpe_ratio(R[:, best], per_period)
    cs = cscv(R, n_blocks=16)
    mtrl = min_track_record_length(dsr["sharpe"])

    report = dict(
        n_configs=n,
        best_config=table.loc[best].to_dict(),
        best_sharpe_ann=dsr["sharpe_ann"],
        expected_max_sharpe_under_null_ann=dsr["benchmark_sharpe_ann"],
        psr_vs_zero=dsr["psr_vs_zero"],
        deflated_sharpe_ratio=dsr["dsr"],
        min_track_record_years=mtrl / 252,
        pbo=cs.pbo,
        prob_oos_loss=cs.prob_oos_loss,
        is_oos_slope=cs.slope,
        median_config_sharpe=float(np.median(per_period) * np.sqrt(252)),
        share_configs_sharpe_positive=float((per_period > 0).mean()),
    )
    (out / "overfitting_report.json").write_text(json.dumps(report, indent=2, default=float))
    for k, v in report.items():
        print(f"{k:38s} {v if not isinstance(v, float) else round(v, 3)}")

    viz.plot_pbo(cs, out / "pbo.png")
    heat = table.pivot_table(index="entry", columns="gate", values="sharpe", aggfunc="median")
    heat = heat[[c for c in ["none", "0.6/0", "0.6/0.3", "0.9/0", "0.9/0.3"] if c in heat.columns]]
    viz.plot_sweep_heatmap(heat, out / "sweep_heatmap.png",
                           "Median net Sharpe over exit/stop grid, by entry rule and regime gate")  # fmt: skip
    print(f"wrote results to {out}/")


def cmd_diagnose(args, cfg: Config) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prices, truth = _load(cfg, args.csv)
    if truth is None:
        raise SystemExit("diagnose needs simulated data (ground truth)")
    errs, aucs, hl = recovery_study(prices, truth)
    print("out-of-sample hedge-ratio RMSE vs true beta (median over pairs x windows):")
    print(errs.median().round(4).to_string())
    print(f"\nregime AUC (calm vs breakdown): median {np.median(aucs):.2f}, "
          f"IQR [{np.percentile(aucs, 25):.2f}, {np.percentile(aucs, 75):.2f}], n={len(aucs)}")  # fmt: skip
    print(f"half-life: median abs error {np.median(np.abs(hl.est - hl.true)):.2f} days "
          f"(median true {hl.true.median():.1f})")  # fmt: skip
    errs.describe().to_csv(out / "beta_rmse.csv")
    viz.plot_recovery(errs, aucs, hl, out / "recovery.png")
    print(f"wrote results to {out}/")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="regimearb", description=__doc__)
    ap.add_argument("--log-level", default="WARNING")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("run", cmd_run), ("sweep", cmd_sweep), ("diagnose", cmd_diagnose)]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None, help="TOML config (defaults are built in)")
        sp.add_argument("--csv", default=None, help="wide csv of prices instead of simulated data")
        sp.add_argument("--out", default=f"results/{name}")
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.from_toml(args.config) if args.config else Config()
    args.fn(args, cfg)
