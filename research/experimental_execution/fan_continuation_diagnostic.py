"""Pure, fixed-state descriptive diagnostics; no market loading or policy fitting.

Deleting a fan member holds the archived state, target and technical price
fixed.  It is neither a refit nor a leave-one-out evaluation of the learner.
"""
from __future__ import annotations

import math
import numpy as np

TOL = 1e-12


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + " must be an integer")
    return int(value)


def fixed_choice(qh, qr, kh, kr, n, eta, horizon=30):
    """The original finite-price rule, or the separate risk-first label None."""
    n, h = integer(n, "N", 1), integer(horizon, "original H", 1)
    kh, kr = integer(kh, "hold count"), integer(kr, "restore count")
    if max(kh, kr) > n or not np.isfinite([qh, qr]).all():
        raise ValueError("Finite net logs and bounded integer counts required")
    if eta is not None and (isinstance(eta, (bool, np.bool_)) or not np.isfinite(eta) or eta < 0):
        raise ValueError("Nonnegative finite eta or risk-first None required")
    dq, dp = float(qr) - float(qh), (kr - kh) / n
    score = None if eta is None else dq - h * float(eta) * dp
    if score is not None and not math.isfinite(score):
        raise ValueError("Nonfinite finite-price score")
    restore = (kr < kh or kr == kh and dq > TOL) if eta is None else score > TOL
    return dict(restore=bool(restore), score=score, delta_net_log_growth=dq,
                delta_breach_probability=dp)


def fan_deletions(hold_logs, restore_logs, hold_bad, restore_bad, eta, *, horizon=30):
    """One exact N-1 recomputation per original member, preserving its order.

    The single-branch probability change is at most 1/N.  A paired signed
    restore-minus-hold probability difference can change by as much as 2/N.
    Event differences are converted to signed integers before subtraction.
    """
    qh, qr = np.asarray(hold_logs, float), np.asarray(restore_logs, float)
    bh, br = np.asarray(hold_bad), np.asarray(restore_bad)
    n = len(qh) if qh.ndim == 1 else 0
    if (n < 2 or qr.shape != (n,) or bh.shape != (n,) or br.shape != (n,)
            or bh.dtype.kind != "b" or br.dtype.kind != "b"
            or not np.isfinite(qh).all() or not np.isfinite(qr).all()):
        raise ValueError("At least two paired finite log outcomes and boolean events required")
    kh, kr = int(bh.sum()), int(br.sum())
    full = fixed_choice(float(qh.mean()), float(qr.mean()), kh, kr, n, eta, horizon)
    rows = []
    for j in range(n):
        dh, dr = (float(qh.sum()) - float(qh[j])) / (n - 1), (float(qr.sum()) - float(qr[j])) / (n - 1)
        dkh, dkr = kh - int(bh[j]), kr - int(br[j])
        chosen = fixed_choice(dh, dr, dkh, dkr, n - 1, eta, horizon)
        ph_change, pr_change = dkh / (n - 1) - kh / n, dkr / (n - 1) - kr / n
        paired_change = chosen["delta_breach_probability"] - full["delta_breach_probability"]
        if max(abs(ph_change), abs(pr_change)) > 1 / n + TOL or abs(paired_change) > 2 / n + TOL:
            raise AssertionError("Finite-fan deletion probability identity failed")
        rows.append(dict(member_index=j, remaining_fan_n=n - 1,
            drop_qg_hold=dh, drop_qg_restore=dr, drop_bad_hold=dkh, drop_bad_restore=dkr,
            drop_score=chosen["score"], drop_restore=chosen["restore"],
            drop_delta_net_log_growth=chosen["delta_net_log_growth"],
            drop_delta_breach_probability=chosen["delta_breach_probability"],
            hold_probability_change=ph_change, restore_probability_change=pr_change,
            paired_probability_change=paired_change, action_flip=chosen["restore"] != full["restore"]))
    return full, rows


def continuation_gaps(adaptive, chosen_B, *, year, horizon=30):
    """Saved adaptive minus same-current-action B; never an unchosen Q^pi."""
    y, h = integer(year, "year", 1), integer(horizon, "H", 1)
    if y > h:
        raise ValueError("Year exceeds original H")
    for values in (adaptive, chosen_B):
        if not isinstance(values["ever_breached"], (bool, np.bool_)):
            raise ValueError("Observed event boolean required")
        if not np.isfinite([values[k] for k in ("total_log_growth", "history_mdd", "nominal_fees", "terminal_real")]).all():
            raise ValueError("Finite observed continuation metrics required")
    gaps = {"continuation_gap_" + key: float(adaptive[key]) - float(chosen_B[key])
            for key in ("total_log_growth", "history_mdd", "nominal_fees", "terminal_real")}
    gaps["continuation_gap_event"] = int(adaptive["ever_breached"]) - int(chosen_B["ever_breached"])
    if y == h:
        for key, gap in gaps.items():
            scale = max(1., abs(float(adaptive.get(key.removeprefix("continuation_gap_"), 0))))
            if abs(gap) > 2e-12 * scale:
                raise AssertionError("Last-year chosen-current-action continuation must agree")
    return gaps
