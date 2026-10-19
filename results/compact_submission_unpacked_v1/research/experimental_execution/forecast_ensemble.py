"""Small, fixed one-step forecast ensembles for development research.

The caller supplies the *previous calendar period* state.  This module never
loads a market file, imports a policy engine, or observes next-period returns.
Channels are three real returns, ``(1+r)/(1+inflation)-1``, followed by nominal
inflation.  Weights address the caller's current-calendar training rows in the
same training-ID order.  They are descriptive forecasts, not safe-support
probabilities, confidence intervals, or independent ensemble samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


MIN_SCALE = 1e-8
CONDITIONAL_SHRINKAGE = .5
CHANNELS = ("equity_real_return", "bond_real_return", "bill_real_return", "inflation")


@dataclass(frozen=True)
class ForecastEnsemble:
    """Weights have shape ``(n_actual, n_forecasts, n_training)``."""

    weights: NDArray[np.float64]
    labels: tuple[str, ...]
    metadata: dict[str, Any]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _training_ids(values: Sequence[int]) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if (raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu"
            or np.any(raw < 1) or np.any(raw > 120)
            or np.unique(raw).size != raw.size):
        raise ValueError("training_ids must be unique integer development IDs1-120")
    return raw.astype(np.int64, copy=True)


def _state(values: ArrayLike, rows: int, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric state values")
    state = np.asarray(values, dtype=np.float64)
    if (state.shape != (rows, 4) or not np.all(np.isfinite(state))
            or np.any(state <= -1)):
        raise ValueError(f"{name} must have shape({rows},4), finite real returns/inflation>-1")
    return state.copy()


def build_forecast_ensemble(
    training_ids: Sequence[int], *, period_index: int, n_actual: int,
    training_previous_state: ArrayLike | None = None,
    actual_previous_state: ArrayLike | None = None,
) -> ForecastEnsemble:
    """Build fixed uniform, conditional-shrinkage and deterministic-block weights.

    At period0 both previous states must be absent, and only uniform is
    available.  At period>=1, standardization uses training mean and population
    standard deviation only, floored at1e-8.  The Gaussian kernel is
    ``exp(-.5 * mean(standardized_distance**2))``; if all kernels underflow,
    that actual row falls back to uniform before50% uniform shrinkage.

    Three balanced blocks are consecutive ``array_split`` chunks of sorted
    training IDs, mapped back to the input order.  At least three training
    scenarios are required for a positive-period ensemble.  These blocks are
    fixed sensitivity forecasts, not three independent statistical samples.
    """
    ids = _training_ids(training_ids)
    actual_n = _positive_int(n_actual, "n_actual")
    if (isinstance(period_index, (bool, np.bool_))
            or not isinstance(period_index, (int, np.integer))
            or not 0 <= period_index <= 29):
        raise ValueError("period_index must be an integer0-29")
    n = len(ids)
    uniform = np.full((actual_n, n), 1. / n, dtype=np.float64)
    metadata: dict[str, Any] = {
        "training_ids": ids.tolist(),
        "period_index": int(period_index),
        "n_actual": actual_n,
        "channels": list(CHANNELS),
        "alignment": "previous calendar state predicts the current calendar training rows",
        "conditional_weight": CONDITIONAL_SHRINKAGE,
        "standard_deviation_ddof": 0,
        "minimum_scale": MIN_SCALE,
        "independent_ensemble_samples": False,
        "confidence_interval": False,
        "safe_support_probabilities": False,
    }
    if period_index == 0:
        if training_previous_state is not None or actual_previous_state is not None:
            raise ValueError("period0 must not receive previous-period observations")
        weights = uniform[:, None, :].copy()
        weights.setflags(write=False)
        metadata.update({"conditional_available": False, "block_training_ids": [],
                         "kernel_uniform_fallback_rows": []})
        return ForecastEnsemble(weights, ("uniform",), metadata)
    if n < 3:
        raise ValueError("positive-period ensemble requires at least3 training IDs")
    if training_previous_state is None or actual_previous_state is None:
        raise ValueError("positive periods require both previous-period states")
    train = _state(training_previous_state, n, "training_previous_state")
    actual = _state(actual_previous_state, actual_n, "actual_previous_state")
    with np.errstate(over="ignore", invalid="ignore"):
        center = train.mean(axis=0)
        scale = np.maximum(train.std(axis=0, ddof=0), MIN_SCALE)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("training state moments exceed numerical range")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        # Subtracting the same training center cancels in this difference.
        # Keep that center in metadata to make the train-only information
        # contract auditable, without contaminating it with actual states.
        distance = (actual[:, None, :] - train[None, :, :]) / scale
        squared_distance = np.mean(distance * distance, axis=2)
        kernels = np.exp(-.5 * squared_distance)
    totals = kernels.sum(axis=1)
    fallback = ~np.isfinite(totals) | (totals <= 0)
    conditional = uniform.copy()
    np.divide(kernels, totals[:, None], out=conditional, where=~fallback[:, None])
    shrunk = (1. - CONDITIONAL_SHRINKAGE) * uniform + CONDITIONAL_SHRINKAGE * conditional
    order = np.argsort(ids, kind="stable")
    block_indices = np.array_split(order, 3)
    blocks = []
    for indices in block_indices:
        row = np.zeros(n, dtype=np.float64)
        row[indices] = 1. / len(indices)
        blocks.append(np.broadcast_to(row, (actual_n, n)))
    weights = np.stack((uniform, shrunk, *blocks), axis=1)
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=2), 1., atol=1e-14, rtol=0)):
        raise ValueError("invalid forecast probability weights")
    weights.setflags(write=False)
    metadata.update({
        "conditional_available": True,
        "training_mean": center.tolist(),
        "training_scale": scale.tolist(),
        "block_training_ids": [ids[index].tolist() for index in block_indices],
        "kernel_uniform_fallback_rows": np.flatnonzero(fallback).tolist(),
    })
    return ForecastEnsemble(weights, ("uniform", "shrink_conditional", "id_block_1", "id_block_2", "id_block_3"), metadata)


def build_forecast_weights(
    trainprev: ArrayLike | None, actualprev: ArrayLike | None,
    training_ids: Sequence[int], *, actual_n: int | None = None,
) -> dict[str, Any]:
    """Thin dictionary adapter; the caller supplies the calendar alignment.

    ``None,None`` means inception and requires ``actual_n``.  Otherwise the
    actual row count is inferred from ``actualprev``.  This adapter cannot
    infer the real calendar index, so positive-period metadata states that
    explicitly rather than inventing an observed date.
    """
    inception = trainprev is None and actualprev is None
    if inception:
        if actual_n is None:
            raise ValueError("inception requires actual_n")
        result = build_forecast_ensemble(training_ids, period_index=0, n_actual=actual_n)
    else:
        if trainprev is None or actualprev is None:
            raise ValueError("both previous-period states must be supplied together")
        raw = np.asarray(actualprev)
        if raw.ndim != 2:
            raise ValueError("actualprev must be a two-dimensional state array")
        inferred = raw.shape[0]
        if actual_n is not None and _positive_int(actual_n, "actual_n") != inferred:
            raise ValueError("actual_n disagrees with actualprev")
        result = build_forecast_ensemble(training_ids, period_index=1, n_actual=inferred,
            training_previous_state=trainprev, actual_previous_state=actualprev)
        result.metadata["period_index"] = None
        result.metadata["calendar_alignment_supplied_by_caller"] = True
    return {"weights": result.weights, "labels": list(result.labels), "metadata": result.metadata}
