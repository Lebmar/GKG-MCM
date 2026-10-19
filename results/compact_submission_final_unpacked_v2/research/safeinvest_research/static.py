"""Fixed-target annual research simulators; no data loading or model selection.

Callers must enforce development-only scenario access using ``load_paths``.
These array-only routines cannot infer scenario provenance from return values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from safeinvest.core import execute_target

from .metrics import NUMERIC_TOL


def simplex_grid(step: float = 0.05, include_equal: bool = True) -> NDArray[np.float64]:
    """Three-asset simplex lattice, including vertices and optional exact thirds.

    ``step`` must be positive and divide one (for example .05 or .025). The
    integer lattice includes zero weights and 100% single-asset targets. Exact
    one-third weights are inserted if absent, with duplicates removed to
    ``NUMERIC_TOL``. Output is lexicographically sorted with shape ``(m,3)``.
    The .05 and .025 grids contain 232 and 862 rows respectively when thirds
    are included; there are no restrictions on minimum asset weights.
    """
    if not np.isscalar(step) or isinstance(step, (bool, np.bool_)):
        raise ValueError("step must be a positive scalar dividing one")
    step = float(step)
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("step must be finite and lie in (0,1]")
    denominator = int(round(1 / step))
    if not np.isclose(denominator * step, 1.0, rtol=0.0, atol=NUMERIC_TOL):
        raise ValueError("step must divide one")
    if not isinstance(include_equal, (bool, np.bool_)):
        raise ValueError("include_equal must be boolean")
    weights = np.asarray([
        [i / denominator, j / denominator, (denominator - i - j) / denominator]
        for i in range(denominator + 1)
        for j in range(denominator - i + 1)
    ], dtype=np.float64)
    equal = np.full(3, 1 / 3)
    if include_equal and not np.any(np.all(np.abs(weights - equal) <= NUMERIC_TOL, axis=1)):
        weights = np.vstack((weights, equal))
    return weights[np.lexsort((weights[:, 2], weights[:, 1], weights[:, 0]))]


def simulate_annual_targets(
    returns: ArrayLike,
    inflation: ArrayLike,
    targets: ArrayLike,
    cost_rate: float = 0.0025,
) -> dict[str, NDArray]:
    """Execute every fixed target annually with exact self-financing fees.

    Returns have shape ``(n,H,3)`` in equity/bond/bill order and inflation has
    shape ``(n,H)``; all rates are decimal and gross factors must be positive.
    Targets have shape ``(m,3)``; a single ``(3,)`` target is promoted to m=1.
    Each scenario/candidate starts with 100 units entirely in bills. Initial
    allocation and all later annual restorations use the unchanged V1
    ``execute_target`` accounting convention, with fee-after target weights.

    The bounded output dictionary contains ``nominal_wealth`` and
    ``real_wealth`` of shape ``(n,m,H+1)``, ``fees``, ``moved`` and ``trades``
    of shape ``(n,m,H)``, and the normalized ``targets`` of shape ``(m,3)``.
    Time zero is before allocation, so all reported path metrics include the
    initial fee. No per-year asset-holdings histories are retained. Trade flags
    mean transferred amount exceeds the numerical tolerance of 1e-12 units.
    The routine never opens a data file or chooses a candidate.
    """
    market = np.asarray(returns, dtype=np.float64)
    infl = np.asarray(inflation, dtype=np.float64)
    weights = np.asarray(targets, dtype=np.float64)
    if market.ndim != 3 or market.shape[-1] != 3 or market.shape[0] == 0 or market.shape[1] == 0:
        raise ValueError("returns must have shape (n,H,3), with n,H positive")
    n, horizon, _ = market.shape
    if infl.shape != (n, horizon):
        raise ValueError("inflation must have shape (n,H), matching returns")
    if not np.all(np.isfinite(market)) or np.any(market <= -1):
        raise ValueError("asset returns must be finite with positive gross factors")
    if not np.all(np.isfinite(infl)) or np.any(infl <= -1):
        raise ValueError("inflation must be finite with positive gross factors")
    if weights.shape == (3,):
        weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != 3 or weights.shape[0] == 0:
        raise ValueError("targets must have shape (m,3), with m positive")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("target weights must be finite and nonnegative")
    sums = weights.sum(axis=1, keepdims=True)
    if not np.all(np.isclose(sums, 1.0, rtol=0.0, atol=NUMERIC_TOL)):
        raise ValueError("each target must sum to one")
    weights = weights / sums
    if not np.isscalar(cost_rate) or isinstance(cost_rate, (bool, np.bool_)):
        raise ValueError("cost_rate must be a scalar in [0,1)")
    rate = float(cost_rate)
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("cost_rate must be finite and lie in [0,1)")
    candidate_count = len(weights)
    nominal = np.full((n, candidate_count, horizon + 1), 100.0)
    fees = np.zeros((n, candidate_count, horizon))
    moved = np.zeros_like(fees)
    holdings = np.zeros((n, candidate_count, 3))
    holdings[:, :, 2] = 100.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for year in range(horizon):
                post, fee, transfer = execute_target(holdings, weights[None, :, :], rate)
                fees[:, :, year] = fee
                moved[:, :, year] = transfer
                holdings = post * (1 + market[:, None, year, :])
                nominal[:, :, year + 1] = holdings.sum(axis=-1)
            prices = np.column_stack((np.ones(n), np.cumprod(1 + infl, axis=1)))
            real = nominal / prices[:, None, :]
        except FloatingPointError as exc:
            raise ValueError("portfolio or inflation calculation exceeds numerical range") from exc
    if not np.all(np.isfinite(real)) or np.any(real <= 0) or np.any(nominal <= 0):
        raise ValueError("wealth paths must remain finite and strictly positive")
    return {
        "nominal_wealth": nominal,
        "real_wealth": real,
        "fees": fees,
        "moved": moved,
        "trades": moved > NUMERIC_TOL,
        "targets": weights.copy(),
    }
