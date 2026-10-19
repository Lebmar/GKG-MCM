"""Diagnostic full-simplex max-min LP with exact destination-buy fees.

No strategy is imported or executed. Numerical certificates are reported,
not exact-arithmetic proofs, policies, liquidity promises or future support.
"""
from __future__ import annotations

from itertools import product
import numpy as np
from scipy.optimize import linprog

SOLVER_METHOD = "highs"
SOLVER_OPTIONS = dict(presolve=True, primal_feasibility_tolerance=1e-9,
    dual_feasibility_tolerance=1e-9, ipm_optimality_tolerance=1e-10, time_limit=30.)
CERTIFICATE_TOL = 1e-8
COMPARISON_TOL = 1e-8
BUY_PATTERNS = tuple(product((False, True), repeat=3))


def inputs(holdings, cpi, support_real_gross, cost_rate):
    x = np.asarray(holdings, float); a = np.asarray(support_real_gross, float)
    if (x.shape != (3,) or not np.isfinite(x).all() or np.any(x < 0)
            or not np.isfinite(x.sum()) or x.sum() <= 0):
        raise ValueError("Finite nonnegative three-asset holdings with positive total required")
    if (a.ndim != 2 or a.shape[1] != 3 or len(a) < 1 or not np.isfinite(a).all() or np.any(a <= 0)):
        raise ValueError("Positive finite joint training real-gross support (N,3) required")
    for value, name in ((cpi, "cpi"), (cost_rate, "cost_rate")):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(name + " must be a finite numeric scalar")
    if cpi <= 0 or not 0 <= cost_rate < 1 or not np.isfinite(x.sum()/cpi) or x.sum()/cpi <= 0:
        raise ValueError("Positive finite real wealth and fee in[0,1) required")
    return x, float(cpi), a, float(cost_rate)


def build_pattern_lp(holdings, cpi, support_real_gross, *, cost_rate=.0025, buy_mask):
    """Equivalent normalized variables u=y/V, v=z/(V/CPI).

    On a fixed buy mask, sum[(1+c*Ibuy)*u]=1+c*sum[Ibuy*x/V].
    Buy bounds are u>=x/V; sell bounds are0<=u<=x/V. Equality belongs
    to both masks, so all eight closed regions cover every physical y.
    """
    x, cp, a, rate = inputs(holdings, cpi, support_real_gross, cost_rate)
    mask = np.asarray(buy_mask)
    if mask.shape != (3,) or mask.dtype.kind != "b":
        raise ValueError("buy_mask must contain exactly three booleans")
    wealth = float(x.sum()); real = wealth/cp; previous = x/wealth
    lower = np.r_[np.where(mask, previous, 0.), -np.inf]
    upper = np.r_[np.where(mask, np.inf, previous), np.inf]
    return dict(c=np.array([0., 0., 0., -1.]),
        A_ub=np.column_stack((-a, np.ones(len(a)))), b_ub=np.zeros(len(a)),
        A_eq=np.array([np.r_[1+rate*mask, 0.]]),
        b_eq=np.array([1+rate*previous[mask].sum()]), lower=lower, upper=upper,
        buy_mask=mask.copy(), previous=previous, holdings=x.copy(), support=a.copy(),
        nominal_wealth=wealth, real_wealth=real, cpi=cp, cost_rate=rate)


def check_solution(problem, result):
    """Independently calculate primal/dual/KKT errors from a solver result.

    SciPy marginals are derivatives of the minimization objective with respect
    to RHS/bounds: inequality/upper <=0, lower >=0. Stationarity therefore is
    c-Aub.T*mub-Aeq.T*mueq-mul-muu=0, not the usual opposite-sign convention.
    """
    q = np.asarray(result.x, float); c = problem["c"]
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("A finite four-variable solution is required")
    aub, bub, aeq, beq = (problem[k] for k in ("A_ub", "b_ub", "A_eq", "b_eq"))
    lo, hi = problem["lower"], problem["upper"]
    mi = np.asarray(result.ineqlin.marginals, float); me = np.asarray(result.eqlin.marginals, float)
    ml = np.asarray(result.lower.marginals, float); mu = np.asarray(result.upper.marginals, float)
    if (mi.shape != bub.shape or me.shape != beq.shape or ml.shape != q.shape or mu.shape != q.shape
            or not all(np.isfinite(v).all() for v in (mi, me, ml, mu))):
        raise ValueError("Complete finite dual marginals required")
    finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
    slack = bub-aub@q; eq = aeq@q-beq
    lower_slack = q[finite_lo]-lo[finite_lo]; upper_slack = hi[finite_hi]-q[finite_hi]
    primal = max(0., float((-slack).max()), float(np.abs(eq).max()),
        float((-lower_slack).max(initial=0)), float((-upper_slack).max(initial=0)))
    stationarity = c-aub.T@mi-aeq.T@me-ml-mu
    signs = max(0., float(mi.max(initial=0)), float((-ml).max(initial=0)), float(mu.max(initial=0)),
        float(np.abs(ml[~finite_lo]).max(initial=0)), float(np.abs(mu[~finite_hi]).max(initial=0)))
    complementarity = max(float(np.abs(mi*slack).max(initial=0)),
        float(np.abs(ml[finite_lo]*lower_slack).max(initial=0)),
        float(np.abs(mu[finite_hi]*upper_slack).max(initial=0)))
    objective = float(c@q)
    dual = float(bub@mi+beq@me+lo[finite_lo]@ml[finite_lo]+hi[finite_hi]@mu[finite_hi])
    true_fee_ratio = float(problem["cost_rate"]*np.maximum(q[:3]-problem["previous"], 0).sum())
    linear_fee_ratio = float(problem["cost_rate"]*(q[:3]-problem["previous"])[problem["buy_mask"]].sum())
    true_funding = float(q[:3].sum()+true_fee_ratio-1)
    worst = float((problem["support"]@q[:3]).min())
    residuals = dict(primal_max_violation=primal,
        dual_stationarity_max_abs=float(np.abs(stationarity).max()), dual_sign_max_violation=signs,
        complementary_max_abs=complementarity, primal_dual_objective_gap_abs=abs(objective-dual),
        solver_objective_gap_abs=abs(objective-float(result.fun)),
        true_self_finance_residual_abs=abs(true_funding), true_vs_pattern_fee_gap_abs=abs(true_fee_ratio-linear_fee_ratio),
        worst_support_vs_z_gap_abs=abs(worst-q[3]))
    return dict(certificate_passed=all(np.isfinite(v) and v <= CERTIFICATE_TOL for v in residuals.values()),
        residuals=residuals, normalized_solution=q.copy(), normalized_dual_objective=dual,
        normalized_primal_objective=objective, normalized_worst_support=worst,
        support_slacks=slack, equality_residual=eq, dual_stationarity=stationarity,
        inequality_marginals=mi, equality_marginals=me, lower_marginals=ml, upper_marginals=mu,
        posttrade_holdings=q[:3]*problem["nominal_wealth"],
        nominal_fee=true_fee_ratio*problem["nominal_wealth"],
        worst_next_real=worst*problem["real_wealth"],
        dual_objective_maxmin_real=-dual*problem["real_wealth"],
        true_self_finance_residual=true_funding*problem["nominal_wealth"])


def solve_full_simplex(holdings, cpi, support_real_gross, *, budget_floor, cost_rate=.0025):
    """Return all8 witnesses and an explicitly numerical classification.

    The simplex allows zero assets. This diagnostic benchmark is not an
    additional product/action; the account is never changed. A failed LP or
    certificate leaves uncertainty visible. Equality within the independent
    comparison tolerance remains a boundary case, not a proof of infeasibility.
    """
    x, cp, support, rate = inputs(holdings, cpi, support_real_gross, cost_rate)
    if (isinstance(budget_floor, (bool, np.bool_)) or not np.isscalar(budget_floor)
            or not np.isfinite(budget_floor) or budget_floor < 0):
        raise ValueError("Finite nonnegative original one-step budget floor required")
    records = []
    for index, pattern in enumerate(BUY_PATTERNS):
        lp = build_pattern_lp(x, cp, support, cost_rate=rate, buy_mask=pattern)
        record = dict(pattern_index=index, buy_mask=list(pattern),
            normalized_equality_rhs=float(lp["b_eq"][0]), certificate_passed=False)
        try:
            result = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                b_eq=lp["b_eq"], bounds=list(zip(lp["lower"], lp["upper"])),
                method=SOLVER_METHOD, options=SOLVER_OPTIONS.copy())
            record.update(solver_status=int(result.status), solver_success=bool(result.success),
                solver_message=str(result.message), iterations=int(result.nit),
                crossover_iterations=int(getattr(result, "crossover_nit", 0)))
            if result.status == 0 and result.success:
                record.update(check_solution(lp, result))
        except Exception as exc:
            record.update(solver_status=None, solver_success=False, error_type=type(exc).__name__, error=str(exc))
        records.append(record)
    valid = [r for r in records if r["certificate_passed"]]
    all_resolved = len(valid) == len(BUY_PATTERNS)
    real = float(x.sum()/cp); band = COMPARISON_TOL*real
    best = max(valid, key=lambda r: (r["worst_next_real"], -r["pattern_index"])) if valid else None
    largest_dual = max(r["dual_objective_maxmin_real"] for r in valid) if valid else None
    if best is None:
        classification = "solver_or_certificate_unresolved"
    elif best["worst_next_real"] > budget_floor+band:
        classification = "numerical_physical_feasible_witness_above_budget"
    elif all_resolved and largest_dual < budget_floor-band:
        classification = "all_eight_numerical_optima_below_budget"
    elif all_resolved:
        classification = "numerical_boundary_unresolved"
    else:
        classification = "incomplete_solver_or_certificate_evidence"
    return dict(classification=classification, all_eight_numerically_resolved=all_resolved,
        pattern_count=8, passed_certificate_count=len(valid), budget_floor=float(budget_floor),
        current_real_wealth=real, comparison_band_real=band,
        hold_worst_next_real=float((support@x/cp).min()),
        best_pattern_index=None if best is None else best["pattern_index"],
        best_worst_next_real=None if best is None else best["worst_next_real"],
        best_posttrade_holdings=None if best is None else best["posttrade_holdings"],
        best_nominal_fee=None if best is None else best["nominal_fee"],
        best_budget_margin_real=None if best is None else best["worst_next_real"]-budget_floor,
        maximum_dual_objective_real=largest_dual, patterns=records,
        exact_arithmetic_impossibility_proof=False, account_action_changed=False,
        future_or_recursive_feasibility_guarantee=False)
