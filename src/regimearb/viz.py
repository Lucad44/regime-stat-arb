from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
})  # fmt: skip


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_equity(curves: dict[str, pd.Series], path, title="Walk-forward equity (net of costs)"):
    fig, (ax, ad) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2]})  # fmt: skip
    for name, r in curves.items():
        w = (1 + r).cumprod()
        ax.plot(w.index, w.values, label=name, lw=1.3)
        ad.plot(w.index, (w / w.cummax() - 1).values * 100, lw=1.0)
    ax.set_title(title)
    ax.set_ylabel("growth of 1")
    ax.legend(frameon=False)
    ad.set_ylabel("drawdown %")
    _save(fig, path)


def plot_pair(pw, truth=None, path="pair.png", follower: str | None = None):
    """Estimated hedge ratio, z-score with trades, and P(mean-reverting) for one pair-window."""
    from .strategy.signals import StrategyParams, generate_positions

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(pw.index, pw.beta, label="state-space beta", lw=1.3)
    if truth is not None and follower is not None:
        ax.plot(pw.index, truth.beta[follower].reindex(pw.index), "k--", lw=1, label="true beta")
    ax.set_ylabel("hedge ratio")
    ax.legend(frameon=False)
    ax.set_title(f"{pw.y} vs {pw.x}  (window {pw.window})")

    ax = axes[1]
    pos, _ = generate_positions(pw.z, pw.p_mr, pw.z_entry_analytic, pw.half_life, StrategyParams())
    ax.plot(pw.index, pw.z, lw=1.0, color="tab:blue")
    ax.axhline(0, color="k", lw=0.6)
    ax.fill_between(pw.index, -6, 6, where=pos > 0, color="tab:green", alpha=0.15, label="long spread")
    ax.fill_between(pw.index, -6, 6, where=pos < 0, color="tab:red", alpha=0.15, label="short spread")
    ax.set_ylim(max(-6, np.nanmin(pw.z) - 0.5), min(6, np.nanmax(pw.z) + 0.5))
    ax.set_ylabel("spread z")
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[2]
    ax.plot(pw.index, pw.p_mr, color="tab:purple", lw=1.2, label="filtered P(calm regime)")
    if truth is not None and follower is not None:
        reg = truth.regime[follower].reindex(pw.index).values
        ax.fill_between(pw.index, 0, 1, where=reg == 1, color="grey", alpha=0.3, label="true breakdown")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    _save(fig, path)


def plot_pbo(res, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    a.hist(res.logits, bins=40, color="tab:blue", alpha=0.8)
    a.axvline(0, color="k", lw=1)
    a.set_title(f"OOS rank logit of IS winner - PBO = {res.pbo:.2f}")
    a.set_xlabel("logit(relative OOS rank)")
    b.scatter(res.is_best_sharpe, res.oos_best_sharpe, s=4, alpha=0.25)
    xs = np.linspace(res.is_best_sharpe.min(), res.is_best_sharpe.max(), 10)
    b.plot(xs, res.intercept + res.slope * xs, "r-", lw=1.2, label=f"slope {res.slope:.2f}")
    b.axhline(0, color="k", lw=0.6)
    b.set_xlabel("IS Sharpe of selected config")
    b.set_ylabel("OOS Sharpe of that config")
    b.set_title(f"Performance degradation, P(OOS<0) = {res.prob_oos_loss:.2f}")
    b.legend(frameon=False)
    _save(fig, path)


def plot_sweep_heatmap(table: pd.DataFrame, path, title):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.8)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=0)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def plot_recovery(errs: pd.DataFrame, aucs: np.ndarray, hl: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    med = errs.median().sort_values()
    axes[0].barh(med.index, med.values, color="tab:blue")
    axes[0].set_title("Hedge-ratio error out of sample\n(median RMSE vs true beta)")
    axes[1].hist(aucs, bins=np.linspace(0, 1, 21), color="tab:purple", alpha=0.85)
    axes[1].axvline(0.5, color="k", lw=1)
    axes[1].set_title(f"AUC of P(calm) vs true regime\nmedian {np.median(aucs):.2f}")
    axes[2].scatter(hl["true"], hl["est"], s=14)
    lim = [0, max(hl.max()) * 1.05]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set_xlabel("true half-life (calm regime)")
    axes[2].set_ylabel("estimated")
    axes[2].set_title("Half-life recovery")
    _save(fig, path)
