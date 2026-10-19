"""Conditional finite-family binomial evidence for fixed static policies.

The theorem requires iid complete paths from a common synthetic distribution
and a fixed finite policy/event family. Independence of scenarios alone does
not establish identical distributions. These bounds do not certify dynamic
policies, real markets, growth or tail severity, nor restore a blind test.
"""
from __future__ import annotations

from math import ceil, log

import numpy as np
from scipy.stats import beta


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def simultaneous_upper(counts, n, family_events, delta=.05):
    """Bonferroni one-sided CP upper bounds, one event for each fixed pair.

    With probability >=1-delta over iid datasets, every fixed event risk is
    below its bound. The events themselves need not be independent. Menu
    selection inside this fixed family can then be arbitrary/data dependent.
    family_events must include every event for which joint coverage is claimed.
    """
    n = _positive_integer(n, "n")
    family_events = _positive_integer(family_events, "family_events")
    values = np.asarray(counts)
    if (not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values > n)):
        raise ValueError("counts must be integer exceedances in 0..n")
    if (isinstance(delta, (bool, np.bool_)) or not np.isscalar(delta)
            or not np.isfinite(delta) or not 0 < delta < 1):
        raise ValueError("delta must lie strictly between zero and one")
    per_event = float(delta) / family_events
    upper = np.ones(values.shape, dtype=float)
    mask = values < n
    upper[mask] = beta.isf(per_event, values[mask] + 1, n - values[mask])
    return upper


def supported_count_limit(n, alpha, family_events, delta=.05):
    """Largest supported empirical count; -1 means even zero lacks support."""
    n = _positive_integer(n, "n")
    if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must lie in [0,1]")
    values = simultaneous_upper(np.arange(n + 1), n, family_events, delta)
    accepted = np.flatnonzero(values <= alpha)
    return int(accepted[-1]) if len(accepted) else -1


def minimum_zero_count_sample(alpha, family_events, delta=.05):
    """Minimum iid n for a zero-count simultaneous upper bound <=alpha."""
    _positive_integer(family_events, "family_events")
    if (not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1
            or not np.isscalar(delta) or not np.isfinite(delta) or not 0 < delta < 1):
        raise ValueError("require alpha and delta strictly between zero and one")
    return ceil(log(delta / family_events) / log(1 - alpha))
