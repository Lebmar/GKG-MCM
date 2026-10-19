"""Development-only fixed portfolios and transparent onboarding assessments.

Asset order is equity, government bonds, government bills. This module accepts
arrays rather than loading data, so callers control the development/validation
split. No function reads or selects on the held-out final scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .core import execute_target, metric_arrays


HORIZONS = (5, 6, 15, 25, 30)
LOSS_THRESHOLDS = (0.10, 0.15, 0.25, 0.35, 0.40)


@dataclass(frozen=True)
class Client:
    """Questionnaire inputs; tolerance is not a guaranteed loss ceiling."""

    name: str
    horizon: int
    loss_tolerance: float
    liquidity: str


CLIENTS = (
    Client("A", 5, 0.10, "high"),
    Client("B", 15, 0.25, "medium"),
    Client("C", 30, 0.40, "low"),
    Client("D", 25, 0.15, "medium"),
    Client("E", 6, 0.35, "high"),
)

# Explicit policy assumptions, not statistically optimized parameters or rules
# supplied by the problem. Callers should vary these in sensitivity analysis.
DEFAULT_LIQUIDITY_FLOORS = {"high": 0.30, "medium": 0.10, "low": 0.00}


def weight_grid(step: float = 0.05, *, include_equal: bool = True) -> np.ndarray:
    """Return all long-only simplex weights on a regular grid, plus exact 1/3.

    ``step`` must divide one. At 5 percentage points the regular grid has 231
    targets; including the off-grid exact equal-weight benchmark gives 232.
    """
    if not np.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("step must be finite and in (0, 1]")
    units = int(round(1 / step))
    if not np.isclose(units * step, 1.0, atol=1e-12):
        raise ValueError("step must divide one exactly, e.g. 0.05 or 0.10")
    weights = np.array(
        [(i / units, j / units, (units - i - j) / units)
         for i in range(units + 1) for j in range(units - i + 1)],
        dtype=float,
    )
    equal = np.full(3, 1 / 3)
    if include_equal and not np.any(np.all(np.isclose(weights, equal), axis=1)):
        weights = np.vstack((weights, equal))
    return weights


def _validate_market(returns: np.ndarray, inflation: np.ndarray | None = None):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 3 or returns.shape[2] != 3:
        raise ValueError("returns must have shape (scenarios, years, 3)")
    if returns.shape[0] == 0 or returns.shape[1] == 0:
        raise ValueError("at least one scenario and one year are required")
    if not np.all(np.isfinite(returns)) or np.any(returns <= -1):
        raise ValueError("asset returns must be finite and greater than -1")
    if inflation is None:
        return returns
    inflation = np.asarray(inflation, dtype=float)
    if inflation.shape != returns.shape[:2]:
        raise ValueError("inflation must have shape (scenarios, years)")
    if not np.all(np.isfinite(inflation)) or np.any(inflation <= -1):
        raise ValueError("inflation must be finite and greater than -1")
    return returns, inflation


def simulate_fixed_target(
    returns: np.ndarray,
    target: Sequence[float],
    cost: float = 0.0025,
    *,
    initial_wealth: float = 100.0,
) -> dict[str, np.ndarray]:
    """Rebalance to a fixed target before each year's return, charging costs.

    ``wealth[:, 0]`` is the original pre-trade wealth. Initial allocation is
    charged; there is no terminal liquidation. ``trades`` records nonzero
    transfers, not merely scheduled checks. Execution uses the core fee
    convention consistently with all other strategies.
    """
    returns = _validate_market(returns)
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or np.any(~np.isfinite(target)):
        raise ValueError("target must contain three finite weights")
    if np.any(target < 0) or not np.isclose(target.sum(), 1.0, atol=1e-12):
        raise ValueError("target must be nonnegative and sum to one")
    if not np.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("initial_wealth must be positive and finite")
    if not np.isfinite(cost) or cost < 0 or cost >= 1:
        raise ValueError("cost must be finite and in [0, 1)")
    n, years, _ = returns.shape
    positions = np.zeros((n, 3))
    positions[:, 2] = initial_wealth
    wealth = np.full((n, years + 1), initial_wealth)
    costs = np.zeros((n, years))
    moved = np.zeros((n, years))
    for t in range(years):
        positions, costs[:, t], moved[:, t] = execute_target(positions, target, cost)
        positions = positions * (1.0 + returns[:, t, :])
        wealth[:, t + 1] = positions.sum(axis=1)
    trades = moved > initial_wealth * 1e-12
    return {"wealth": wealth, "costs": costs, "moved": moved, "trades": trades}


def _summarize(metrics: dict[str, np.ndarray], initial_wealth: float = 100.0):
    terminal = np.asarray(metrics["terminal_real"])
    result = {
        "n_scenarios": int(len(terminal)),
        "cagr_median": float(np.median(metrics["cagr"])),
        "cagr_p10": float(np.quantile(metrics["cagr"], 0.10)),
        "mdd_median": float(np.median(metrics["mdd"])),
        "mdd_p90": float(np.quantile(metrics["mdd"], 0.90)),
        "mdd_mean": float(np.mean(metrics["mdd"])),
        "volatility_median": float(np.median(metrics["volatility"])),
        "terminal_real_median": float(np.median(terminal)),
        "mean_log_growth": float(np.mean(np.log(terminal / initial_wealth))),
    }
    for loss in LOSS_THRESHOLDS:
        result[f"breach_{round(100 * loss):02d}"] = float(np.mean(metrics["mdd"] > loss))
    return result


def evaluate_grid(
    returns: np.ndarray,
    inflation: np.ndarray,
    step: float = 0.05,
    cost: float = 0.0025,
    *,
    horizons: Sequence[int] = HORIZONS,
    include_equal: bool = True,
) -> pd.DataFrame:
    """Evaluate annually rebalanced fixed targets at client-specific horizons.

    One output row is one target/horizon pair. Metrics are first calculated
    within each complete scenario, then summarized across scenarios. The
    mean_log_growth column is cumulative (not annualized), suitable for an
    explicitly horizon-dependent utility comparison.
    """
    returns, inflation = _validate_market(returns, inflation)
    horizons = tuple(horizons)
    if not horizons or any(int(h) != h or h < 1 or h > returns.shape[1] for h in horizons):
        raise ValueError("horizons must be positive integers within the supplied history")
    rows = []
    for target_id, target in enumerate(weight_grid(step, include_equal=include_equal)):
        sim = simulate_fixed_target(returns, target, cost)
        for h in horizons:
            h = int(h)
            metrics = metric_arrays(sim["wealth"][:, :h + 1], inflation[:, :h])
            rows.append({
                "target_id": target_id,
                "w_equity": float(target[0]),
                "w_bond": float(target[1]),
                "w_bill": float(target[2]),
                "horizon": h,
                **_summarize(metrics),
                "cost_mean": float(sim["costs"][:, :h].sum(axis=1).mean()),
                "moved_mean": float(sim["moved"][:, :h].sum(axis=1).mean()),
                "trade_fraction": float(sim["trades"][:, :h].mean()),
            })
    return pd.DataFrame(rows)


def rank_profile_candidates(
    grid: pd.DataFrame,
    risk_aversion: float,
    bill_floor: float,
    *,
    horizon: int = 30,
    neighbor_l1: float = 0.10,
) -> pd.DataFrame:
    """Rank one declared risk policy; do not interpret this as a final choice.

    J = E[log(real terminal wealth / initial wealth)] - lambda E[MDD].
    Neighbor summaries use only feasible targets in the supplied evaluated
    grid, within L1 weight distance ``neighbor_l1`` (5-point swaps at 0.10).
    The target itself is excluded. They diagnose local parameter sensitivity;
    they are not independent validation or a statistical confidence bound.
    """
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("risk_aversion must be nonnegative and finite")
    if not np.isfinite(bill_floor) or not 0 <= bill_floor <= 1:
        raise ValueError("bill_floor must be in [0, 1]")
    if not np.isfinite(neighbor_l1) or neighbor_l1 <= 0:
        raise ValueError("neighbor_l1 must be positive and finite")
    candidates = grid.loc[
        (grid.horizon == horizon) & (grid.w_bill >= bill_floor - 1e-12)
    ].copy()
    if candidates.empty:
        raise ValueError("no evaluated candidates satisfy this policy")
    candidates["utility"] = (
        candidates.mean_log_growth - risk_aversion * candidates.mdd_mean
    )
    weights = candidates[["w_equity", "w_bond", "w_bill"]].to_numpy()
    distances = np.abs(weights[:, None, :] - weights[None, :, :]).sum(axis=2)
    utility = candidates.utility.to_numpy()
    n_neighbors, minima, medians = [], [], []
    for row in distances:
        values = utility[(row > 1e-10) & (row <= neighbor_l1 + 1e-12)]
        n_neighbors.append(len(values))
        minima.append(float(values.min()) if len(values) else np.nan)
        medians.append(float(np.median(values)) if len(values) else np.nan)
    candidates["neighbor_count"] = n_neighbors
    candidates["neighbor_utility_min"] = minima
    candidates["neighbor_utility_median"] = medians
    candidates["gap_to_best"] = candidates.utility.max() - candidates.utility
    candidates["distance_to_best"] = distances[np.argmax(utility)]
    return candidates.sort_values(
        ["utility", "w_equity", "w_bond"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _wilson_upper(breaches: int, n: int, z: float = 1.959963984540054) -> float:
    """Upper endpoint of a two-sided 95% Wilson interval (diagnostic only)."""
    p = breaches / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(min(1.0, center + radius))


def assess_onboarding(
    returns: np.ndarray,
    inflation: np.ndarray,
    portfolios: Mapping[str, Sequence[float]],
    *,
    clients: Sequence[Client] = CLIENTS,
    alpha: float = 0.10,
    liquidity_floors: Mapping[str, float] | None = None,
    cost: float = 0.0025,
) -> pd.DataFrame:
    """Assess every client/portfolio pair without forcing an allocation.

    Feasible means empirical P(MDD_H > stated tolerance) <= alpha and bills
    above the declared liquidity floor. Wilson upper endpoints expose finite
    scenario uncertainty but do not silently change this empirical criterion.
    No feasible option means refer for human review. A feasible option is not
    a loss guarantee; withdrawal adequacy is not tested because no withdrawals
    are supplied in the problem.
    """
    returns, inflation = _validate_market(returns, inflation)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    floors = dict(DEFAULT_LIQUIDITY_FLOORS if liquidity_floors is None else liquidity_floors)
    for key, value in floors.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid liquidity floor for {key}")
    rows = []
    for portfolio, target in portfolios.items():
        target = np.asarray(target, dtype=float)
        sim = simulate_fixed_target(returns, target, cost)
        for client in clients:
            h = client.horizon
            if h < 1 or h > returns.shape[1]:
                raise ValueError(f"insufficient history for client {client.name}")
            if client.liquidity not in floors:
                raise ValueError(f"no liquidity floor defined for {client.liquidity}")
            if not 0 <= client.loss_tolerance <= 1:
                raise ValueError("client loss tolerance must be in [0, 1]")
            metrics = metric_arrays(sim["wealth"][:, :h + 1], inflation[:, :h])
            n = len(metrics["mdd"])
            count = int(np.count_nonzero(metrics["mdd"] > client.loss_tolerance))
            breach = count / n
            loss_pass = breach <= alpha + 1e-12
            floor = floors[client.liquidity]
            liquidity_pass = target[2] >= floor - 1e-12
            rows.append({
                "client": client.name, "portfolio": str(portfolio),
                "horizon": h, "loss_tolerance": client.loss_tolerance,
                "liquidity": client.liquidity, "bill_floor": floor,
                "w_equity": target[0], "w_bond": target[1], "w_bill": target[2],
                **_summarize(metrics),
                "breach_count": count, "breach_rate": breach,
                "breach_wilson_upper95": _wilson_upper(count, n),
                "alpha": alpha, "loss_pass": bool(loss_pass),
                "liquidity_pass": bool(liquidity_pass),
                "feasible": bool(loss_pass and liquidity_pass),
            })
    return pd.DataFrame(rows)
