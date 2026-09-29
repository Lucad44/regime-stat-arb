from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_csv(path: str | Path, date_col: int | str = 0) -> pd.DataFrame:
    """Wide csv of adjusted closes: one column per ticker, dates in `date_col`."""
    df = pd.read_csv(path, index_col=date_col, parse_dates=True)
    return df.sort_index()


def load_yfinance(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Download adjusted closes with a small on-disk cache. Needs the `data` extra."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = f"{'-'.join(sorted(tickers))[:80]}_{start}_{end or 'now'}.csv"
    fp = cache / key
    if fp.exists():
        return load_csv(fp)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("pip install regimearb[data] to use the yfinance loader") from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = raw["Close"] if "Close" in raw else raw
    px.to_csv(fp)
    return px


def clean_prices(
    prices: pd.DataFrame, max_missing: float = 0.02, ffill_limit: int = 3
) -> pd.DataFrame:
    """Drop illiquid/short-history columns, forward-fill small gaps, drop leftovers."""
    keep = prices.columns[prices.isna().mean() <= max_missing]
    dropped = set(prices.columns) - set(keep)
    if dropped:
        log.info("dropping %d columns with >%.0f%% missing", len(dropped), 100 * max_missing)
    out = prices[keep].ffill(limit=ffill_limit).dropna(how="any")
    return out[(out > 0).all(axis=1)]
