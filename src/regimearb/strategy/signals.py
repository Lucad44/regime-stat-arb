from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._jit import njit

EXIT_TARGET, EXIT_STOP, EXIT_REGIME, EXIT_TIME = 1, 2, 3, 4
EXIT_NAMES = {0: "open", 1: "target", 2: "stop", 3: "regime", 4: "time"}


@dataclass
class StrategyParams:
    entry_mode: str = "bertram"  # "bertram" | "fixed"
    z_entry: float = 2.0  # used when entry_mode == "fixed"
    # Clip range for the analytic entry level. The continuous-time optimum is usually below
    # 1 sigma (median ~0.7 on the simulated data), so the floor binds most of the time.
    z_entry_floor: float = 1.0
    z_entry_cap: float = 3.0
    z_exit: float = 0.25
    z_stop: float = 4.0
    gate_in: float = 0.6  # min P(mean-reverting regime) to open a trade; 0 = no gate
    gate_out: float = 0.3  # close if P falls below this; 0 = never
    max_hold_hl: float = 4.0  # time stop in half-lives
    prob_sizing: bool = True
    execution_lag: int = 1  # extra bars between the signal and the fill


@njit(cache=True)
def _run_state_machine(z, p, z_in, z_out, z_stop, gate_in, gate_out, max_hold, prob_sizing):
    n = z.shape[0]
    pos = np.zeros(n)
    reason = np.zeros(n, dtype=np.int8)
    side = 0
    size = 0.0
    held = 0
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            continue
        if side != 0:
            held += 1
            u = -side * zt  # > 0 while the spread is still on the far side of the mean
            code = 0
            if u <= z_out:
                code = 1
            elif u >= z_stop:
                code = 2
            elif p[t] < gate_out:
                code = 3
            elif held >= max_hold:
                code = 4
            if code != 0:
                side = 0
                reason[t] = code
            else:
                pos[t] = side * size
        elif abs(zt) >= z_in and abs(zt) < z_stop and p[t] >= gate_in:
            side = -1 if zt > 0 else 1
            size = min(1.0, p[t]) if prob_sizing else 1.0
            held = 0
            pos[t] = side * size
    return pos, reason


def entry_level(z_entry_analytic: float, params: StrategyParams) -> float:
    if params.entry_mode == "fixed":
        return params.z_entry
    return float(np.clip(z_entry_analytic, params.z_entry_floor, params.z_entry_cap))


def generate_positions(
    z: np.ndarray,
    p_mr: np.ndarray,
    z_entry_analytic: float,
    half_life: float,
    params: StrategyParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed position in [-1, 1] decided at each close, plus an exit-reason code.

    Long spread (+) means long y / short beta*x. Whatever is decided on bar t is
    only tradable `execution_lag` bars later; the shift is applied in the
    backtest, not here, so that this function stays a pure function of the
    signals seen up to t.
    """
    max_hold = max(params.max_hold_hl * half_life, 2.0)
    return _run_state_machine(
        np.ascontiguousarray(z, dtype=float),
        np.ascontiguousarray(p_mr, dtype=float),
        entry_level(z_entry_analytic, params),
        params.z_exit,
        params.z_stop,
        params.gate_in,
        params.gate_out,
        max_hold,
        params.prob_sizing,
    )
