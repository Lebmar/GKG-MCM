"""Purchasing-power growth and path-loss summaries for development research.

All inputs are arrays supplied by the caller; this module never loads market
data. Scenarios occupy axis zero, and time occupies the last wealth axis.
``NUMERIC_TOL`` is a floating-point equality convention, not a risk budget.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

NUMERIC_TOL = 1e-12
SUMMARY_METADATA = {
    "scenario_axis": 0,
    "growth_definition": "log(real_terminal/100)/H",
    "median_method": "numpy.median; mean of the two middle values for even sample sizes",
    "p10_p90_method": "inverted_cdf; empirical CDF generalized inverse",
    "cvar90_definition": "average of the worst 10% probability mass, including fractional boundary atoms",
    "numerical_boundary_tolerance": NUMERIC_TOL,
    "risk_event": "MDD - L > 1e-12; smaller differences treated as numerical equality",
}
FloatArray = NDArray[np.float64]


def _probabilities(probabilities: ArrayLike | None, size: int) -> FloatArray:
    if probabilities is None:
        return np.full(size, 1.0 / size)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (size,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be a finite nonnegative vector matching the scenario axis")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=NUMERIC_TOL):
        raise ValueError("probabilities must sum to one within NUMERIC_TOL")
    return values / total


def _axis_index(axis: int, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer")
    if not -ndim <= axis < ndim:
        raise ValueError("axis is outside the loss array")
    return int(axis) % ndim


def discrete_cvar(
    losses: ArrayLike,
    probabilities: ArrayLike | None = None,
    level: float = 0.90,
    axis: int = 0,
) -> FloatArray:
    """Average the worst ``1-level`` probability mass, including partial atoms.

    This equals ``min_eta eta + E[(loss-eta)+]/(1-level)`` for finite
    distributions. Equal-probability samples need not have an integer number
    of observations in the tail. A one-dimensional probability vector applies
    to the nominated axis and is shared across all remaining array axes.
    Finite losses may have either sign. The result has that axis removed.
    """
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim == 0 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a nonempty finite array with a scenario axis")
    axis = _axis_index(axis, values.ndim)
    if not np.isscalar(level) or isinstance(level, (bool, np.bool_)):
        raise ValueError("level must be a scalar strictly between zero and one")
    level = float(level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between zero and one")
    ordered_values = np.moveaxis(values, axis, 0)
    masses = _probabilities(probabilities, ordered_values.shape[0])
    order = np.argsort(-ordered_values, axis=0, kind="stable")
    worst_first = np.take_along_axis(ordered_values, order, axis=0)
    mass_shape = (len(masses),) + (1,) * (ordered_values.ndim - 1)
    broadcast_masses = np.broadcast_to(masses.reshape(mass_shape), ordered_values.shape)
    sorted_masses = np.take_along_axis(broadcast_masses, order, axis=0)
    mass_before = np.cumsum(sorted_masses, axis=0) - sorted_masses
    tail_mass = 1.0 - level
    included_mass = np.minimum(np.maximum(tail_mass - mass_before, 0.0), sorted_masses)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = np.asarray(np.sum(included_mass * worst_first, axis=0) / tail_mass)
        except FloatingPointError as exc:
            raise ValueError("CVaR calculation exceeds numerical range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("CVaR calculation exceeds numerical range")
    return result


def path_metrics(real_wealth: ArrayLike) -> dict[str, FloatArray]:
    """Return per-scenario metrics for real wealth shaped ``(n, ..., H+1)``.

    Every path must start with the original pre-allocation wealth of 100.
    ``log_growth`` is ``log(real_terminal/100)/H``; CAGR is its exponential
    minus one. Maximum relative drawdown includes time zero, and volatility
    is sample standard deviation of annual net real returns (ddof=1).
    For a one-year path volatility is zero by convention, since a sample
    standard deviation cannot be estimated from one observation.
    """
    values = np.asarray(real_wealth, dtype=np.float64)
    if values.ndim < 2 or any(dimension == 0 for dimension in values.shape) or values.shape[-1] < 2:
        raise ValueError("real wealth must have shape (n, ..., H+1), with n,H positive")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("real wealth must be finite and strictly positive")
    if not np.all(np.isclose(values[..., 0], 100.0, rtol=0.0, atol=NUMERIC_TOL)):
        raise ValueError("real wealth must retain the initial pre-allocation value 100")
    horizon = values.shape[-1] - 1
    log_growth = (np.log(values[..., -1]) - np.log(100.0)) / horizon
    peaks = np.maximum.accumulate(values, axis=-1)
    drawdowns = (peaks - values) / peaks
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            annual_returns = values[..., 1:] / values[..., :-1] - 1.0
            volatility = np.std(annual_returns, axis=-1, ddof=1) if horizon > 1 else np.zeros(values.shape[:-1])
            cagr = np.expm1(log_growth)
        except FloatingPointError as exc:
            raise ValueError("real returns exceed numerical range") from exc
    return {
        "log_growth": log_growth,
        "cagr": cagr,
        "mdd": np.max(drawdowns, axis=-1),
        "volatility": volatility,
        "terminal_real": values[..., -1].copy(),
    }


def empirical_risk(
    mdd: ArrayLike,
    loss_tolerance: float,
    probabilities: ArrayLike | None = None,
) -> dict[str, FloatArray]:
    """Full-path exceedance frequency and expected excess, over scenario axis 0.

    A drawdown exceeding L by at most ``1e-12`` is treated as numerical equality
    and contributes zero to both frequency and excess. This avoids classifying
    floating-point representations of equality as a breach. It is not an
    additional economically meaningful loss allowance.
    """
    values = np.asarray(mdd, dtype=np.float64)
    if values.ndim == 0 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("mdd must be a nonempty finite array with scenarios on axis zero")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("relative drawdown must lie in [0,1]")
    if not np.isscalar(loss_tolerance) or isinstance(loss_tolerance, (bool, np.bool_)):
        raise ValueError("loss_tolerance must be a scalar in [0,1]")
    tolerance = float(loss_tolerance)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("loss_tolerance must be a finite scalar in [0,1]")
    masses = _probabilities(probabilities, values.shape[0])
    masses = masses.reshape((len(masses),) + (1,) * (values.ndim - 1))
    excess = values - tolerance
    breached = excess > NUMERIC_TOL
    return {
        "breach_probability": np.sum(masses * breached, axis=0),
        "expected_excess": np.sum(masses * np.where(breached, excess, 0.0), axis=0),
    }


def summarize_paths(
    metrics: Mapping[str, ArrayLike],
    loss_tolerance: float | None = None,
) -> dict[str, FloatArray]:
    """Aggregate equally weighted scenarios, preserving candidate-weight axes.

    ``metrics`` is the dictionary from ``path_metrics``. P10/P90 use the empirical
    CDF generalized inverse (``method='inverted_cdf'``), while medians use the
    standard mean of the two middle values for even samples. Thus P90 MDD <= L
    corresponds to at most 10% strict empirical exceedances; apply the same
    ``NUMERIC_TOL`` on both boundary comparisons. CVaR90 handles fractional tail
    observations without an interpolated quantile cutoff. ``SUMMARY_METADATA``
    exposes these conventions for an experiment manifest. If L is supplied,
    add direct-count breach probability and expected excess.
    """
    required = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
    if any(key not in metrics for key in required):
        raise ValueError("metrics is missing required path-metric fields")
    fields = {key: np.asarray(metrics[key], dtype=np.float64) for key in required}
    shape = fields["log_growth"].shape
    if len(shape) == 0 or any(dimension == 0 for dimension in shape):
        raise ValueError("metric arrays need a nonempty scenario axis")
    if any(value.shape != shape or not np.all(np.isfinite(value)) for value in fields.values()):
        raise ValueError("all metric arrays must be finite and share the same shape")
    if np.any(fields["mdd"] < 0) or np.any(fields["mdd"] > 1):
        raise ValueError("relative drawdown must lie in [0,1]")
    if np.any(fields["volatility"] < 0) or np.any(fields["terminal_real"] <= 0) or np.any(fields["cagr"] < -1):
        raise ValueError("path metrics contain impossible volatility, terminal wealth, or CAGR")
    result = {
        "mean_log_growth": np.mean(fields["log_growth"], axis=0),
        "median_cagr": np.median(fields["cagr"], axis=0),
        "p10_cagr": np.quantile(fields["cagr"], 0.1, axis=0, method="inverted_cdf"),
        "median_mdd": np.median(fields["mdd"], axis=0),
        "p90_mdd": np.quantile(fields["mdd"], 0.9, axis=0, method="inverted_cdf"),
        "mean_mdd": np.mean(fields["mdd"], axis=0),
        "cvar90_mdd": discrete_cvar(fields["mdd"], level=0.9, axis=0),
        "median_volatility": np.median(fields["volatility"], axis=0),
        "median_terminal_real": np.median(fields["terminal_real"], axis=0),
    }
    if loss_tolerance is not None:
        result.update(empirical_risk(fields["mdd"], loss_tolerance))
    return result
