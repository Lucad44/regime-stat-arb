from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from .strategy.signals import StrategyParams


@dataclass
class DataConfig:
    source: str = "synthetic"  # "synthetic" | "csv"
    csv_path: str | None = None
    seed: int = 7
    n_days: int = 2520
    n_leaders: int = 24
    n_planted: int = 10


@dataclass
class WalkForwardConfig:
    formation: int = 504  # trading days used to select pairs / fit models
    step: int = 126  # re-formation cadence
    max_pairs: int = 10
    top_k: int = 60  # candidates passed to the cointegration test
    n_factors: int = 3  # PCs stripped before the correlation screen
    fdr: float = 0.20
    hl_min: float = 2.0
    hl_max: float = 60.0
    max_reuse: int = 2  # max pairs an asset can appear in per window


@dataclass
class CostConfig:
    cost_bps: float = 8.0  # per unit of traded notional (commission + slippage)
    borrow_bps: float = 50.0  # annualised, charged on short notional


@dataclass
class PortfolioConfig:
    gross_leverage: float = 2.0
    vol_target: float | None = None  # annualised; None = off
    vol_halflife: float = 30.0
    max_leverage: float = 4.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyParams = field(default_factory=StrategyParams)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg = cls()
        for f in fields(cfg):
            section = raw.get(f.name)
            if section is None:
                continue
            target = getattr(cfg, f.name)
            unknown = set(section) - {g.name for g in fields(target)}
            if unknown:
                raise ValueError(f"unknown keys in [{f.name}]: {sorted(unknown)}")
            for k, v in section.items():
                setattr(target, k, v)
        return cfg

    def to_dict(self) -> dict:
        def conv(obj):
            if is_dataclass(obj):
                return {f.name: conv(getattr(obj, f.name)) for f in fields(obj)}
            return obj

        return conv(self)
