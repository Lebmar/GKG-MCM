"""Short exact history-tree reference for a *specified* finite outcome model.

This is deliberately limited to at most three joint asset/inflation outcomes
and at most three periods.  Outcomes are iid draws from the caller's stated
probabilities; this is a modelling assumption, not an inference about data.
The only actions are hold and restore one fixed target, with exact fees.

Each nondominated pair (g,q) represents a deterministic, nonanticipating
history-policy tree.  g is expected total log *real* wealth growth from the
current wealth; q is probability that an annual observed drawdown has ever
exceeded L over the complete horizon, including an already-latched breach.
Children are combined by Cartesian products *before* one root chance budget
is applied.  No child receives a fresh alpha.  This class permits different
actions after different histories even when their financial states coincide;
it is wider than an unaugmented Markov policy in (u, relative wealth, b, h).

An optional convex-mixture calculation bounds this finite deterministic
policy family.  It mixes whole root policies independently of future draws;
it is a mathematical relaxation, not deployable randomized advice.  Nothing
here solves a continuous 30-year risk-control problem or supplies a
population-risk certificate.  No files or market data are read.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from safeinvest.core import execute_target

MDD_TOL = 1e-12
# A declared absolute machine-scale convention, not a proved accumulated
# error bound or a market-calibrated safety margin.  Alpha=0 stays exactly 0.
PROBABILITY_ROUNDOFF_GUARD = 64 * np.finfo(np.float64).eps


def _scalar(value, name, lower=None, upper=None, upper_open=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    number = float(value)
    if (not np.isfinite(number) or (lower is not None and number < lower)
            or (upper is not None and (number >= upper if upper_open else number > upper))):
        raise ValueError(f"Invalid {name}")
    return number


def _target(value):
    weights = np.asarray(value, float)
    if (weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("target must be one finite nonnegative three-asset simplex point")
    return weights / weights.sum()


def _readonly(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RiskState:
    """Actual nominal holdings, CPI, prior real peak and absorbing breach flag."""

    holdings: tuple[float, float, float]
    cpi: float = 1.0
    real_peak: float = 100.0
    breached: bool = False

    @property
    def real_wealth(self) -> float:
        return sum(self.holdings) / self.cpi


def _state(value: RiskState, loss_tolerance: float) -> RiskState:
    if not isinstance(value, RiskState):
        raise ValueError("initial_state must be a RiskState")
    holdings = np.asarray(value.holdings, float)
    if (holdings.shape != (3,) or not np.all(np.isfinite(holdings)) or np.any(holdings < 0)
            or not np.isfinite(holdings.sum()) or holdings.sum() <= 0):
        raise ValueError("Holdings must have finite positive total wealth")
    cpi = _scalar(value.cpi, "CPI", lower=0)
    peak = _scalar(value.real_peak, "real peak", lower=0)
    if cpi == 0 or peak == 0 or not isinstance(value.breached, (bool, np.bool_)):
        raise ValueError("CPI/real peak must be positive and breached must be boolean")
    real = float(holdings.sum() / cpi)
    if not np.isfinite(real) or real <= 0 or real > peak + MDD_TOL * max(1, peak):
        raise ValueError("Current real wealth must be positive and no greater than its recorded peak")
    peak = max(real, peak)
    breached = bool(value.breached or (1 - real / peak) - loss_tolerance > MDD_TOL)
    return RiskState(tuple(float(x) for x in holdings), cpi, peak, breached)


@dataclass(frozen=True)
class PolicyNode:
    """An action chosen before the next outcome, followed by observed branches."""

    action: str
    outcome_ids: tuple[int, ...] = ()
    children: tuple[PolicyNode, ...] = ()


@dataclass(frozen=True)
class PolicyPoint:
    g: float
    q: float
    policy: PolicyNode | None = None


@dataclass(frozen=True)
class MixtureBound:
    feasible: bool
    g: float | None
    q: float | None
    # (index in the supplied finite point sequence, root mixing probability)
    components: tuple[tuple[int, float], ...]
    dual_lambda: float | None
    dual_value: float | None
    duality_gap: float | None
    deterministic_relaxation_gap: float | None
    requested_alpha: float
    effective_alpha: float
    probability_roundoff_guard: float

    @property
    def q_minus_requested_alpha(self) -> float | None:
        return None if self.q is None else self.q - self.requested_alpha


@dataclass(frozen=True)
class RiskTreeModel:
    gross: NDArray[np.float64]
    inflation: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    outcome_ids: tuple[int, ...]
    target: NDArray[np.float64]
    horizon: int
    cost_rate: float
    loss_tolerance: float
    on_breach: str
    initial_state: RiskState


@dataclass(frozen=True)
class RiskTreeResult:
    model: RiskTreeModel
    alpha: float
    points: tuple[PolicyPoint, ...]
    selected: PolicyPoint | None
    randomized_bound: MixtureBound
    nodes_evaluated: int
    root_pairs_before_pruning: int
    pruning_used: bool

    @property
    def requested_alpha(self) -> float:
        return self.alpha

    @property
    def effective_alpha(self) -> float:
        return _effective_alpha(self.alpha)

    @property
    def probability_roundoff_guard(self) -> float:
        return PROBABILITY_ROUNDOFF_GUARD

    @property
    def selected_q_minus_requested_alpha(self) -> float | None:
        return None if self.selected is None else self.selected.q - self.alpha

    @property
    def metadata(self):
        return {
            "scope": "Fully enumerated short history-tree reference under supplied finite iid model",
            "max_horizon": 3, "max_outcomes": 3,
            "g": "Expected total log(real terminal wealth / real current wealth), not annualized",
            "q": "Probability of any annual-endpoint MDD-L>1e-12, including already-latched breach",
            "alpha_used_only_at_root": True,
            "requested_alpha": self.requested_alpha,
            "effective_alpha": self.effective_alpha,
            "probability_roundoff_guard": self.probability_roundoff_guard,
            "applied_probability_guard": self.effective_alpha - self.requested_alpha,
            "selected_q_minus_requested_alpha": self.selected_q_minus_requested_alpha,
            "root_feasibility": "q <= effective_alpha; alpha=0 permits only exactly zero computed q",
            "probability_guard_meaning": "64*float64 epsilon absolute numerical convention, not a proved total rounding-error bound or a statistical risk margin; relative budget expansion may be substantial near zero",
            "mixture_dual_scope": "Primal/dual agreement is for effective_alpha; this only gives a looser upper bound at requested_alpha, not strict original-budget feasibility or original-problem strong duality",
            "on_breach": self.model.on_breach,
            "on_breach_meaning": ("Hold actual assets while nonresponse is pending; no adviser or recovery payoff is invented"
                                   if self.model.on_breach == "hold_only" else
                                   "Continue the same two actions; the breach flag remains absorbing"),
            "policy_class": "Deterministic nonanticipating full-history trees; conditional budgets are implicit in branch choices",
            "pruning": "Exact weak dominance in q and g, with at least one strict inequality; exact duplicates keep a canonical policy",
            "peak_observations": "Initial point and annual endpoints; no extra intraperiod fee-only observation or terminal liquidation",
            "randomized_bound": "Convex relaxation of finite whole-policy root pairs; not deployable randomized advice",
            "population_probability_inferred": False,
            "continuous_or_30year_solver": False,
            "observable_support": "Duplicate positive-probability joint (gross,inflation) rows are rejected, so invisible labels cannot serve as random coins",
        }


def _step(state, action, gross, inflation, target, cost_rate, loss_tolerance):
    if action == "hold":
        post, fee = np.asarray(state.holdings, float), 0.0
    elif action == "restore":
        post, fee, _ = execute_target(state.holdings, target, cost_rate)
        fee = float(fee)
    else:
        raise ValueError("Action must be hold or restore")
    holdings = post * gross
    cpi = state.cpi * (1 + inflation)
    real = float(holdings.sum() / cpi)
    if not np.all(np.isfinite(holdings)) or not np.isfinite(cpi) or cpi <= 0 or not np.isfinite(real) or real <= 0:
        raise ValueError("Outcome exceeds the numerical wealth/CPI range")
    peak = max(state.real_peak, real)
    breached = state.breached or (1 - real / peak) - loss_tolerance > MDD_TOL
    next_state = RiskState(tuple(float(x) for x in holdings), float(cpi), peak, bool(breached))
    return next_state, float(np.log(real) - np.log(state.real_wealth)), fee


def step_state(state: RiskState, action: str, gross: ArrayLike, inflation: float,
               target: ArrayLike, *, cost_rate: float, loss_tolerance: float):
    """Apply one specified action/outcome; return next state, log-real gain, fee.

    This accounting helper does not choose an action or reset a breach.  The
    annual endpoint includes the exact fee once and that year's inflation.
    """
    loss = _scalar(loss_tolerance, "loss_tolerance", 0, 1)
    cost = _scalar(cost_rate, "cost_rate", 0, 1, upper_open=True)
    infl = _scalar(inflation, "inflation", -1)
    draw = np.asarray(gross, float)
    if infl <= -1 or draw.shape != (3,) or not np.all(np.isfinite(draw)) or np.any(draw <= 0):
        raise ValueError("One outcome needs positive asset gross and inflation greater than -1")
    return _step(_state(state, loss), action, draw, infl, _target(target), cost, loss)


def _policy_key(policy):
    if policy is None:
        return ()
    return (policy.action, tuple(_policy_key(child) for child in policy.children))


def pareto_frontier(points: Sequence[PolicyPoint]) -> tuple[PolicyPoint, ...]:
    """Safely remove exact dominated/duplicate (g,q) points, without epsilon dominance."""
    ordered = sorted(_validated_points(points), key=lambda p: (p.q, -p.g, _policy_key(p.policy)))
    best_growth = -np.inf
    kept = []
    for point in ordered:
        if point.g > best_growth:
            kept.append(point)
            best_growth = point.g
    return tuple(kept)


def _validated_points(points):
    points = tuple(points)
    if not points or any(not isinstance(p, PolicyPoint) or not np.isfinite(p.g)
                         or not np.isfinite(p.q) or not 0 <= p.q <= 1 for p in points):
        raise ValueError("Require nonempty finite PolicyPoint(g,q) with q in [0,1]")
    return points


def select_deterministic(points: Sequence[PolicyPoint], alpha: float) -> PolicyPoint | None:
    """Apply one root chance constraint at the declared numerical budget.

    For 0<alpha<1, the effective budget adds 64 float64 eps (capped at one).
    Alpha=0 remains exactly zero.  This convention is not original-budget
    exact feasibility; :class:`RiskTreeResult` exposes both budgets and slack.
    """
    alpha = _scalar(alpha, "alpha", 0, 1)
    feasible = [p for p in _validated_points(points) if p.q <= _effective_alpha(alpha)]
    return min(feasible, key=lambda p: (-p.g, p.q, _policy_key(p.policy))) if feasible else None


def _effective_alpha(alpha):
    return 0.0 if alpha == 0 else min(1.0, alpha + PROBABILITY_ROUNDOFF_GUARD)


def finite_mixture_bound(points: Sequence[PolicyPoint], alpha: float) -> MixtureBound:
    """Solve the finite two-constraint root-mixture LP by its one/two-point vertices.

    Dual = min_{lambda>=0} lambda*effective_alpha + max_i(g_i-lambda*q_i).
    Both primal and dual are reported; the deterministic relaxation gap is
    separate from numerical primal/dual disagreement.  No random recommendation
    is produced, only indices and probabilities witnessing this finite bound.
    """
    points = _validated_points(points)
    alpha = _scalar(alpha, "alpha", 0, 1)
    requested_alpha = alpha
    effective_alpha = _effective_alpha(alpha)
    deterministic = select_deterministic(points, alpha)
    if deterministic is None:
        return MixtureBound(False, None, None, (), None, None, None, None,
                             requested_alpha, effective_alpha, PROBABILITY_ROUNDOFF_GUARD)
    alpha = effective_alpha
    # Dominated pairs cannot help any nonnegative-lambda dual or feasible
    # mixture.  Keep original indices while avoiding quadratic work on all
    # syntactically different but economically identical enumerated trees.
    useful = []
    best = -np.inf
    for index in sorted(range(len(points)), key=lambda i: (points[i].q, -points[i].g, _policy_key(points[i].policy), i)):
        if points[index].g > best:
            useful.append((index, points[index]))
            best = points[index].g
    options = [(p.g, p.q, ((i, 1.0),)) for i, p in useful if p.q <= alpha]
    for i, left in useful:
        if left.q >= alpha:
            continue
        for j, right in useful:
            if right.q <= alpha:
                continue
            probability = (alpha - left.q) / (right.q - left.q)
            growth = (1 - probability) * left.g + probability * right.g
            options.append((growth, alpha, ((i, 1 - probability), (j, probability))))
    growth, risk, components = min(options, key=lambda p: (-p[0], p[1], p[2]))
    lambdas = [0.0]
    for i, _ in components:
        for _, other in useful:
            denominator = other.q - points[i].q
            if denominator:
                slope = (other.g - points[i].g) / denominator
                if np.isfinite(slope) and slope >= 0:
                    lambdas.append(float(slope))
    dual_value, multiplier = min((lam * alpha + max(p.g - lam * p.q for _, p in useful), lam) for lam in lambdas)
    gap = float(dual_value - growth)
    if abs(gap) > 1e-10 * max(1, abs(growth), abs(dual_value)):
        raise RuntimeError("Finite root-mixture primal and dual do not agree")
    return MixtureBound(True, float(growth), float(risk), components, float(multiplier), float(dual_value),
                        gap, float(growth - deterministic.g), requested_alpha, effective_alpha,
                        PROBABILITY_ROUNDOFF_GUARD)


def _model(gross, inflation, probabilities, target, horizon, cost_rate, loss_tolerance, on_breach, initial_state):
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or not 0 <= horizon <= 3:
        raise ValueError("horizon must be an integer in [0,3]")
    draw, infl, probability = np.asarray(gross, float), np.asarray(inflation, float), np.asarray(probabilities, float)
    if draw.ndim != 2 or draw.shape[1] != 3 or not 1 <= len(draw) <= 3 or not np.all(np.isfinite(draw)) or np.any(draw <= 0):
        raise ValueError("gross must contain one to three positive joint three-asset outcomes")
    if infl.shape != (len(draw),) or not np.all(np.isfinite(infl)) or np.any(infl <= -1):
        raise ValueError("inflation must match outcomes and exceed -1")
    if (probability.shape != (len(draw),) or not np.all(np.isfinite(probability)) or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Outcome probabilities must be nonnegative and sum to one")
    if on_breach not in ("hold_only", "continue"):
        raise ValueError("Explicit on_breach must be hold_only or continue")
    loss = _scalar(loss_tolerance, "loss_tolerance", 0, 1)
    cost = _scalar(cost_rate, "cost_rate", 0, 1, upper_open=True)
    if initial_state is None:
        initial_state = RiskState((0.0, 0.0, 100.0))
    state = _state(initial_state, loss)
    support = probability > 0
    if len(np.unique(np.column_stack((draw[support], infl[support])), axis=0)) != int(support.sum()):
        raise ValueError("Duplicate positive-probability joint outcomes must first be combined; invisible IDs cannot randomize history policies")
    probability = probability[support]
    probability /= probability.sum()
    return RiskTreeModel(_readonly(draw[support]), _readonly(infl[support]), _readonly(probability),
                          tuple(int(i) for i in np.flatnonzero(support)), _readonly(_target(target)), int(horizon),
                          cost, loss, on_breach, state)


def solve_risk_tree(gross: ArrayLike, inflation: ArrayLike, probabilities: ArrayLike,
                    target: ArrayLike, *, horizon: int, loss_tolerance: float,
                    alpha: float, on_breach: str, cost_rate: float = .0025,
                    initial_state: RiskState | None = None, prune: bool = True) -> RiskTreeResult:
    """Enumerate short nonanticipating policy trees and apply alpha once at root.

    ``on_breach`` is deliberately mandatory.  ``hold_only`` models a stated
    nonresponse-pending assumption, not an adviser action.  ``continue`` keeps
    both original actions after breach; neither choice clears that event.
    Set prune=False only to cross-check tiny examples by full enumeration.
    """
    if not isinstance(prune, (bool, np.bool_)):
        raise ValueError("prune must be boolean")
    alpha = _scalar(alpha, "alpha", 0, 1)
    model = _model(gross, inflation, probabilities, target, horizon, cost_rate, loss_tolerance, on_breach, initial_state)
    count = 0
    root_pairs_count = 0

    def visit(state, remaining):
        nonlocal count, root_pairs_count
        count += 1
        if remaining == 0:
            return (PolicyPoint(0.0, float(state.breached), PolicyNode("terminal")),)
        actions = ("hold",) if state.breached and model.on_breach == "hold_only" else ("hold", "restore")
        candidates = []
        for action in actions:
            continuations, increments = [], []
            for draw, infl in zip(model.gross, model.inflation):
                next_state, growth, _ = _step(state, action, draw, infl, model.target, model.cost_rate, model.loss_tolerance)
                continuations.append(visit(next_state, remaining - 1))
                increments.append(growth)
            for children in product(*continuations):
                growth = sum(float(p) * (increment + child.g)
                             for p, increment, child in zip(model.probabilities, increments, children))
                risk = sum(float(p) * child.q for p, child in zip(model.probabilities, children))
                if not np.isfinite(growth) or risk < -1e-14 or risk > 1 + 1e-14:
                    raise ValueError("Policy value outside the numerical range")
                policy = PolicyNode(action, model.outcome_ids, tuple(child.policy for child in children))
                candidates.append(PolicyPoint(float(growth), float(np.clip(risk, 0, 1)), policy))
        if remaining == model.horizon:
            root_pairs_count = len(candidates)
        return pareto_frontier(candidates) if prune else tuple(candidates)

    points = visit(model.initial_state, model.horizon)
    if model.horizon == 0:
        root_pairs_count = 1
    return RiskTreeResult(model, alpha, points, select_deterministic(points, alpha), finite_mixture_bound(points, alpha),
                           count, root_pairs_count, bool(prune))


def action_for_history(policy: PolicyNode, history: Sequence[int]) -> str:
    """Read an already-fixed action after observed original outcome identities."""
    node = policy
    if not isinstance(node, PolicyNode):
        raise ValueError("A policy tree is required")
    for outcome in history:
        if (isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, (int, np.integer))
                or outcome not in node.outcome_ids):
            raise ValueError("History contains an unavailable outcome or extends past the terminal node")
        node = node.children[node.outcome_ids.index(outcome)]
    return node.action


def evaluate_policy(result: RiskTreeResult, policy: PolicyNode) -> PolicyPoint:
    """Replay one frozen history-policy tree under its model, never reselecting.

    This shares the reference accounting helper; independent audits should
    separately reconstruct fees and path risks rather than treating replay
    as an independent numerical oracle.
    """
    model = result.model

    def visit(state, remaining, node):
        if not isinstance(node, PolicyNode):
            raise ValueError("Missing policy node")
        if remaining == 0:
            if node.action != "terminal" or node.children or node.outcome_ids:
                raise ValueError("Terminal policy node has an action or further branch")
            return 0.0, float(state.breached)
        allowed = ("hold",) if state.breached and model.on_breach == "hold_only" else ("hold", "restore")
        if node.action not in allowed or node.outcome_ids != model.outcome_ids or len(node.children) != len(model.gross):
            raise ValueError("Policy violates action/response rule or does not cover exactly the declared outcomes")
        growth, risk = 0.0, 0.0
        for draw, infl, probability, child in zip(model.gross, model.inflation, model.probabilities, node.children):
            next_state, increment, _ = _step(state, node.action, draw, infl, model.target, model.cost_rate, model.loss_tolerance)
            child_g, child_q = visit(next_state, remaining - 1, child)
            growth += float(probability) * (increment + child_g)
            risk += float(probability) * child_q
        return growth, risk

    g, q = visit(model.initial_state, model.horizon, policy)
    return PolicyPoint(float(g), float(np.clip(q, 0, 1)), policy)
