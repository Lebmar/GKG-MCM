"""Self-financing portfolio accounting and purchasing-power metrics.

The last asset axis is always equity, government bonds, and the bill proxy.
Returns and inflation are decimal rates, not percentage-point values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def execute_target(
    holdings: ArrayLike,
    target: ArrayLike,
    cost_rate: float = 0.0025,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Trade into fee-*after* target weights without double charging transfers.

    Inputs broadcast over leading axes and must end in three assets. For
    pre-trade holdings ``x``, wealth ``V``, and target weights ``w``, solve

        C = cost_rate * sum(max((V - C) * w - x, 0)).

    Thus ``moved`` is the actual money entering destination assets, fees are
    paid from sale proceeds, and post-trade wealth is exactly ``V - C``.
    Initial allocation is treated in exactly the same way as later trades.

    An actual transfer of 20 units costs 0.05 at the default rate. Setting a
    20% post-fee equity target from 100 units of bills instead transfers
    slightly less than 20 because the fee reduces the investable wealth.

    Returns post-trade holdings, fees, and transferred amounts. A zero-wealth
    portfolio stays zero. Cost rates must be finite and lie in [0, 1).
    """
    x = np.asarray(holdings, dtype=np.float64)
    w = np.asarray(target, dtype=np.float64)
    if x.ndim < 1 or w.ndim < 1 or x.shape[-1] != 3 or w.shape[-1] != 3:
        raise ValueError("holdings and target must have a final asset axis of length 3")
    try:
        x, w = np.broadcast_arrays(x, w)
    except ValueError as exc:
        raise ValueError("holdings and target leading axes must broadcast") from exc
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("holdings must be finite and nonnegative")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("target weights must be finite and nonnegative")
    weight_sum = np.sum(w, axis=-1, keepdims=True)
    if not np.all(np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("target weights must sum to one")
    if not np.isscalar(cost_rate):
        raise ValueError("cost_rate must be a scalar")
    rate = float(cost_rate)
    if not np.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("cost_rate must be finite and lie in [0, 1)")
    # Remove only roundoff in a supplied simplex, preserving exact funding.
    w = w / weight_sum
    wealth = np.sum(x, axis=-1)
    if not np.all(np.isfinite(wealth)):
        raise ValueError("total holdings must be finite")
    cost = np.zeros_like(wealth)
    # With three assets there can be at most two strict destination assets.
    # As the fee grows, destinations can only leave the active set. Solving
    # the linear equation and updating that set three times is sufficient.
    active = wealth[..., None] * w > x
    for _ in range(3):
        active_weight = np.sum(np.where(active, w, 0.0), axis=-1)
        active_holdings = np.sum(np.where(active, x, 0.0), axis=-1)
        cost = rate * (wealth * active_weight - active_holdings)
        cost = np.maximum(cost / (1.0 + rate * active_weight), 0.0)
        active = (wealth - cost)[..., None] * w > x
    post = (wealth - cost)[..., None] * w
    moved = np.sum(np.maximum(post - x, 0.0), axis=-1)
    # Ratios of existing holdings can differ by one ULP after normalization.
    # Treat only such floating-point noise as no trade, retaining holdings.
    tolerance = 32 * np.finfo(np.float64).eps * wealth
    no_trade = np.all(np.abs(wealth[..., None] * w - x) <= tolerance[..., None], axis=-1)
    post = np.where(no_trade[..., None], x, post)
    cost = np.where(no_trade, 0.0, cost)
    moved = np.where(no_trade, 0.0, moved)
    return post, cost, moved


def actual_wealth(wealth: ArrayLike, inflation: ArrayLike) -> FloatArray:
    """Deflate net nominal paths, retaining the uninflated initial point.

    ``wealth`` has shape (n, H + 1), ``inflation`` has shape (n, H).
    The opening point precedes initial trading, so its value must be retained
    in the caller's paths to capture initial allocation fees in all metrics.
    """
    nominal = np.asarray(wealth, dtype=np.float64)
    infl = np.asarray(inflation, dtype=np.float64)
    if nominal.ndim != 2 or infl.ndim != 2:
        raise ValueError("wealth and inflation must be two-dimensional path arrays")
    if nominal.shape[0] == 0 or infl.shape[1] == 0:
        raise ValueError("at least one path and one return period are required")
    if nominal.shape != (infl.shape[0], infl.shape[1] + 1):
        raise ValueError("wealth must have one initial point plus one per inflation period")
    if not np.all(np.isfinite(nominal)) or np.any(nominal <= 0):
        raise ValueError("wealth must be finite and strictly positive")
    if not np.all(np.isfinite(infl)) or np.any(infl <= -1):
        raise ValueError("inflation must be finite and greater than -1")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        price_index = np.cumprod(1.0 + infl, axis=1)
        price_index = np.concatenate((np.ones((nominal.shape[0], 1)), price_index), axis=1)
        real = nominal / price_index
    if not np.all(np.isfinite(real)) or np.any(real <= 0):
        raise ValueError("wealth or cumulative inflation exceeds numerical range")
    return real


def metric_arrays(wealth: ArrayLike, inflation: ArrayLike) -> dict[str, FloatArray]:
    """Return per-path real CAGR, maximum drawdown, volatility, and terminal wealth.

    Maximum drawdown includes the initial pre-allocation point. Volatility is
    the sample standard deviation (ddof=1) of annual real net returns, including
    first-year allocation fees; a one-period path has volatility zero by
    convention because sample variability cannot be estimated from one value.
    """
    real = actual_wealth(wealth, inflation)
    horizon = real.shape[1] - 1
    annual = real[:, 1:] / real[:, :-1] - 1.0
    peaks = np.maximum.accumulate(real, axis=1)
    return {
        "cagr": np.power(real[:, -1] / real[:, 0], 1.0 / horizon) - 1.0,
        "mdd": np.max(1.0 - real / peaks, axis=1),
        "volatility": np.std(annual, axis=1, ddof=1) if horizon > 1 else np.zeros(real.shape[0]),
        "terminal_real": real[:, -1],
    }
