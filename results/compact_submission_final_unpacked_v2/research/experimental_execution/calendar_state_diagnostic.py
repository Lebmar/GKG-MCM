"""Pure fixed-comparator diagnostic of calendar fan state omission.

This fits no trading rule. One previous-year joint observation is used to
predict market log quantities, and held-out future values are scoring-only.
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("equity_real_log", "bond_real_log", "bill_real_log", "inflation_log")
TARGETS = ("next_year", "remaining_to_original30_mean_annual")


def market_log_variables(returns, inflation):
    r, pi = np.asarray(returns, float), np.asarray(inflation, float)
    if (r.ndim != 3 or r.shape[-1] != 3 or r.shape[1] != 30
            or len(r) == 0 or pi.shape != r.shape[:2]
            or not np.isfinite(r).all() or not np.isfinite(pi).all()
            or np.any(r <= -1) or np.any(pi <= -1)):
        raise ValueError("Require complete finite joint 30-year market paths")
    log_inflation = np.log1p(pi)
    return np.concatenate((np.log1p(r) - log_inflation[..., None],
                           log_inflation[..., None]), axis=-1)


def diagnostic_arrays(log_variables):
    z = np.asarray(log_variables, float)
    if z.ndim != 3 or z.shape[1:] != (30, 4) or not len(z) or not np.isfinite(z).all():
        raise ValueError("Require finite (n,30,4) original-calendar log quantities")
    # A decision for original year t (2..30) sees ONLY original year t-1.
    x = z[:, :-1, :].copy()
    future = z[:, 1:, :]
    remaining = np.cumsum(future[:, ::-1, :], axis=1)[:, ::-1, :]
    remaining = remaining / np.arange(29, 0, -1)[None, :, None]
    return x, {"next_year": future.copy(),
               "remaining_to_original30_mean_annual": remaining}


def _finite_cube(value, name):
    a = np.asarray(value, float)
    if a.ndim != 3 or a.shape[-1] != 4 or not len(a) or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite nonempty (paths,years,4)")
    return a


def fit_calendar_state(features, targets, *, penalty=1.):
    """Train-only calendar intercepts plus one fixed pooled ridge slope.

    Feature and outcome scales are pooled RMS residuals after subtracting
    each training calendar-year mean. All four quantities share a linear
    model, but losses are reported per target; no trading Q is estimated.
    Objective: mean||Y_standardized-X_standardized B||² + penalty||B||².
    """
    x, y = _finite_cube(features, "features"), _finite_cube(targets, "targets")
    if x.shape != y.shape or (isinstance(penalty, (bool, np.bool_))
            or not np.isscalar(penalty) or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("Require matching cubes and one positive finite ridge penalty")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    xs, ys = np.sqrt(np.mean(xc**2, axis=(0, 1))), np.sqrt(np.mean(yc**2, axis=(0, 1)))
    # Deterministic channels remain exactly their calendar mean and get a
    # conventional unit scale; this is recorded, never selected on eval data.
    xs = np.where(xs > 1e-15, xs, 1.)
    ys = np.where(ys > 1e-15, ys, 1.)
    xx, yy = (xc / xs).reshape(-1, 4), (yc / ys).reshape(-1, 4)
    n = len(xx)
    coef = np.linalg.solve(xx.T @ xx / n + float(penalty) * np.eye(4), xx.T @ yy / n)
    return {"x_calendar_mean": xm, "y_calendar_mean": ym,
            "x_scale": xs, "y_scale": ys, "standardized_coefficients": coef,
            "penalty": float(penalty), "training_paths": len(x), "training_years": x.shape[1]}


def predict_calendar_state(model, features):
    x = _finite_cube(features, "features")
    if x.shape[1:] != model["x_calendar_mean"].shape:
        raise ValueError("Feature calendar differs from frozen training model")
    baseline = np.broadcast_to(model["y_calendar_mean"], x.shape).copy()
    state = baseline + ((x - model["x_calendar_mean"]) / model["x_scale"]) @ model["standardized_coefficients"] * model["y_scale"]
    return baseline, state


def validate_folds(folds, allowed_ids):
    expected = set(allowed_ids)
    if expected != set(range(1, 121)) or len(allowed_ids) != 120 or len(folds) != 5:
        raise ValueError("Diagnostic requires the original 120 development IDs and five folds")
    seen = []
    for f in folds:
        train, test = list(f["training_ids"]), list(f["evaluation_ids"])
        if (len(train) != 96 or len(test) != 24 or len(set(train)) != 96
                or len(set(test)) != 24 or set(train) & set(test)
                or set(train) | set(test) != expected):
            raise ValueError("Every fold must be a complete disjoint original 96/24 partition")
        seen.extend(test)
    if len(seen) != 120 or set(seen) != expected or len(set(seen)) != 120:
        raise ValueError("Every original path must be evaluated exactly once")


def per_path_losses(actual, baseline, state, scale):
    y, b, s = [_finite_cube(v, "actual/predictions") for v in (actual, baseline, state)]
    sd = np.asarray(scale, float)
    if y.shape != b.shape or y.shape != s.shape or sd.shape != (4,) or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("Require common prediction shape and positive training-only channel scales")
    # Collapse annual rows BEFORE any path/fold aggregation.
    return {"baseline_mse": np.mean((y-b)**2, axis=1),
            "state_mse": np.mean((y-s)**2, axis=1),
            "baseline_standardized_mse": np.mean(((y-b)/sd)**2, axis=1),
            "state_standardized_mse": np.mean(((y-s)/sd)**2, axis=1),
            "baseline_bias": np.mean(b-y, axis=1),
            "state_bias": np.mean(s-y, axis=1)}
