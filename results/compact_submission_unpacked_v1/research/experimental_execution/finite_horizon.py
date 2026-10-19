"""Finite-horizon hold/restore prototype with proportional transaction fees.

For each *fixed* target ``w`` the only actions are to hold the current weights
``u`` or restore ``w``.  The state is (remaining periods, current weights);
wealth factors out of the log objective and proportional execution costs.

``joint_gross[n, T, 3]`` is supplied by the caller.  Its n*T joint asset-return
vectors form an equally weighted, stationary empirical mixture.  The solver
does not read files, draw Monte Carlo samples, preserve the historical time
order, or assert that the n*T observations are statistically independent.
The Bellman model uses a new draw from this fixed mixture each period.

Values at future states use nonnegative barycentric interpolation on a
regular triangular simplex mesh.  Current-period returns and execution fees
are evaluated at the actual weights, including off-mesh weights.  Thus a
one-period comparison is exact for the supplied empirical distribution; for
longer horizons this is an approximate dynamic program, not a continuous-
state optimum, a drawdown constraint, or a population-risk guarantee.

The three asset coordinates follow safeinvest.core: equity, bonds, bills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from safeinvest.core import execute_target

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ACTION_TOL = 1e-12
_EPS = np.finfo(np.float64).eps


def _readonly(value: ArrayLike, dtype: Any = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _weights(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3 or result.size == 0:
        raise ValueError(f"{name} must have a nonempty final asset axis of length 3")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = result.sum(axis=-1, keepdims=True)
    if not np.all(np.isclose(total, 1.0, atol=1e-12, rtol=0)):
        raise ValueError(f"{name} must sum to one on the asset axis")
    return result / total


def _cost_rate(value: float) -> float:
    if not np.isscalar(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError("cost_rate must be a scalar in [0,1)")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result < 1:
        raise ValueError("cost_rate must be finite and lie in [0,1)")
    return result


def _gross_samples(joint_gross: ArrayLike) -> FloatArray:
    gross = np.asarray(joint_gross, dtype=np.float64)
    if gross.ndim != 3 or gross.shape[-1] != 3 or min(gross.shape[:2]) < 1:
        raise ValueError("joint_gross must have nonempty shape (scenarios, years, 3)")
    if not np.all(np.isfinite(gross)) or np.any(gross <= 0):
        raise ValueError("Every joint asset gross return must be finite and positive")
    return gross.reshape(-1, 3)


@dataclass(frozen=True)
class SimplexMesh:
    """Regular mesh, ordered by increasing first then second asset weight.

    ``states`` contains exactly (K+1)(K+2)/2 lattice points for K=1/step.
    There is no additional equal-thirds point when thirds are off the mesh.
    Arrays created by :func:`triangular_mesh` are read-only.
    """

    step: float
    divisions: int
    states: FloatArray

    @property
    def size(self) -> int:
        return len(self.states)


def triangular_mesh(step: float = 0.05) -> SimplexMesh:
    """Construct the three-asset triangular lattice including every boundary.

    ``step`` must divide one, up to floating-point representation error.  The
    default has 231 states, not the 232-target research candidate catalogue.
    """
    if not np.isscalar(step) or isinstance(step, (bool, np.bool_)):
        raise ValueError("step must be a positive scalar reciprocal of an integer")
    step = float(step)
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("step must be finite and lie in (0,1]")
    inverse = 1.0 / step
    if not np.isfinite(inverse) or inverse > np.iinfo(np.int32).max:
        raise ValueError("step is outside the supported integer lattice range")
    divisions = int(round(inverse))
    if not np.isclose(inverse, divisions, atol=1e-12, rtol=0):
        raise ValueError("step must be the reciprocal of an integer")
    states = [(i / divisions, j / divisions, (divisions - i - j) / divisions)
              for i in range(divisions + 1) for j in range(divisions - i + 1)]
    return SimplexMesh(1.0 / divisions, divisions, _readonly(states))


def _lattice_index(i: IntArray, j: IntArray, divisions: int) -> IntArray:
    return i * (divisions + 1) - i * (i - 1) // 2 + j


def barycentric_coordinates(points: ArrayLike, mesh: SimplexMesh) -> tuple[IntArray, FloatArray]:
    """Return three vertex IDs and nonnegative weights for each simplex point.

    Input and output leading axes agree: (...,3) -> (...,3), (...,3).
    A shared-edge point may be represented by either adjacent triangle; a
    zero coefficient makes the choice immaterial.  Only roundoff-sized
    lattice/edge errors are snapped or clipped, followed by normalization.
    """
    if not isinstance(mesh, SimplexMesh):
        raise ValueError("mesh must be constructed by triangular_mesh")
    u = _weights(points, "points")
    leading = u.shape[:-1]
    scaled = u.reshape(-1, 3)[:, :2] * mesh.divisions
    nearest = np.rint(scaled)
    snap = 64 * _EPS * max(1, mesh.divisions)
    scaled = np.where(np.abs(scaled - nearest) <= snap, nearest, scaled)
    base = np.floor(scaled).astype(np.int64)
    i, j = base[:, 0], base[:, 1]
    a, b = (scaled - base).T
    total = i + j
    if np.any(total > mesh.divisions):
        raise ValueError("Point lies outside the numerical simplex")
    corners = total == mesh.divisions
    lower = ~corners & ((a + b <= 1) | (total == mesh.divisions - 1))
    upper = ~(corners | lower)
    vertices = np.empty((len(base), 3), dtype=np.int64)
    coefficients = np.empty((len(base), 3), dtype=np.float64)
    if np.any(corners):
        vertex = _lattice_index(i[corners], j[corners], mesh.divisions)
        vertices[corners] = vertex[:, None]
        coefficients[corners] = (1.0, 0.0, 0.0)
    if np.any(lower):
        il, jl = i[lower], j[lower]
        vertices[lower] = np.stack((_lattice_index(il, jl, mesh.divisions),
                                    _lattice_index(il + 1, jl, mesh.divisions),
                                    _lattice_index(il, jl + 1, mesh.divisions)), axis=-1)
        coefficients[lower] = np.stack((1 - a[lower] - b[lower], a[lower], b[lower]), axis=-1)
    if np.any(upper):
        iu, ju = i[upper], j[upper]
        vertices[upper] = np.stack((_lattice_index(iu + 1, ju, mesh.divisions),
                                    _lattice_index(iu, ju + 1, mesh.divisions),
                                    _lattice_index(iu + 1, ju + 1, mesh.divisions)), axis=-1)
        coefficients[upper] = np.stack((1 - b[upper], 1 - a[upper], a[upper] + b[upper] - 1), axis=-1)
    if np.any(coefficients < -snap) or np.any(vertices < 0) or np.any(vertices >= mesh.size):
        raise RuntimeError("Invalid triangle or materially negative interpolation coefficient")
    coefficients = np.maximum(coefficients, 0.0)
    coefficients /= coefficients.sum(axis=-1, keepdims=True)
    return vertices.reshape(leading + (3,)), coefficients.reshape(leading + (3,))


def interpolate_on_mesh(values: ArrayLike, points: ArrayLike, mesh: SimplexMesh) -> FloatArray:
    """Interpolate values with shape (mesh.size, ...) at points (...,3)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != mesh.size or not np.all(np.isfinite(values)):
        raise ValueError("values must be finite with a first axis matching mesh.size")
    indices, coefficients = barycentric_coordinates(points, mesh)
    tail_axes = values.ndim - 1
    weighted = values[indices] * coefficients.reshape(coefficients.shape + (1,) * tail_axes)
    return weighted.sum(axis=indices.ndim - 1)


def _transition_terms_flat(gross: FloatArray, states: FloatArray, mesh: SimplexMesh,
                           chunk_size: int) -> tuple[FloatArray, FloatArray]:
    """Build a dense Markov interpolation operator without a sample-by-mesh matrix."""
    states = states.reshape(-1, 3)
    transition = np.zeros((len(states), mesh.size), dtype=np.float64)
    log_return = np.empty(len(states), dtype=np.float64)
    for begin in range(0, len(states), chunk_size):
        end = min(begin + chunk_size, len(states))
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            money = states[begin:end, None, :] * gross[None, :, :]
            portfolio_gross = money.sum(axis=-1)
            drifted = money / portfolio_gross[..., None]
        if not np.all(np.isfinite(portfolio_gross)) or np.any(portfolio_gross <= 0):
            raise ValueError("Portfolio gross return exceeds the numerical range")
        vertex, coefficient = barycentric_coordinates(drifted, mesh)
        row_offset = np.arange(end - begin, dtype=np.int64)[:, None, None] * mesh.size
        accumulated = np.bincount((vertex + row_offset).ravel(),
                                  weights=(coefficient / len(gross)).ravel(),
                                  minlength=(end - begin) * mesh.size)
        transition[begin:end] = accumulated.reshape(end - begin, mesh.size)
        log_return[begin:end] = np.log(portfolio_gross).mean(axis=1)
    sums = transition.sum(axis=1, keepdims=True)
    if (not np.all(np.isfinite(transition)) or np.any(transition < 0)
            or not np.all(np.isclose(sums, 1, atol=1e-10, rtol=0))):
        raise RuntimeError("Transition rows must be nonnegative and sum to one")
    transition /= sums
    if not np.all(np.isfinite(log_return)):
        raise ValueError("Expected log return exceeds the numerical range")
    return transition, log_return


def _expected_log_flat(gross: FloatArray, states: FloatArray, chunk_size: int) -> FloatArray:
    """Current-period expectation, without allocating a future-state operator."""
    result = np.empty(len(states), dtype=np.float64)
    for begin in range(0, len(states), chunk_size):
        end = min(begin + chunk_size, len(states))
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            portfolio = (states[begin:end, None, :] * gross[None, :, :]).sum(axis=-1)
        if not np.all(np.isfinite(portfolio)) or np.any(portfolio <= 0):
            raise ValueError("Portfolio gross return exceeds the numerical range")
        result[begin:end] = np.log(portfolio).mean(axis=1)
    return result


def transition_terms(joint_gross: ArrayLike, states: ArrayLike, mesh: SimplexMesh,
                     *, chunk_size: int = 32) -> tuple[FloatArray, FloatArray]:
    """Return P and L for supplied states under the equal joint empirical mixture.

    P has shape (number of states, mesh.size), is nonnegative and row-stochastic.
    L is the mean of log(state dot gross), not log of the mean gross return.
    Each drifted point contributes its three interpolation coefficients.  This
    preserves joint asset observations and needs no Monte Carlo approximation.
    """
    gross = _gross_samples(joint_gross)
    u = _weights(states, "states")
    if u.ndim == 1:
        u = u[None, :]
    if u.ndim != 2:
        raise ValueError("states must be one weight vector or a matrix of weight vectors")
    chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
    return _transition_terms_flat(gross, u, mesh, chunk_size)


def exact_fee_fraction(current_weights: ArrayLike, target: ArrayLike,
                       cost_rate: float = 0.0025) -> FloatArray:
    """Return C/V using core.execute_target at unit pretrade wealth.

    C solves C=c*sum positive((1-C)*target-current_weights).  In particular,
    restoring all bills to all equity costs c/(1+c), rather than exactly c.
    Broadcasting follows execute_target.  A hold action pays no fee.
    """
    current = _weights(current_weights, "current_weights")
    desired = _weights(target, "target")
    _, fraction, _ = execute_target(current, desired, _cost_rate(cost_rate))
    return fraction


@dataclass(frozen=True)
class FiniteHorizonSolution:
    """Read-only tables for independent fixed-target hold/restore problems.

    V and Qhold_grid have shape (H+1, mesh states, targets); tradebase has
    shape (H+1, targets).  The h=0 tables are terminal zero placeholders.
    ``V[h]=max(Qhold,Qtrade)``; execution uses a separate strict 1e-12 gain
    guard.  Forgoing a positive gain no larger than this guard is at most
    H*1e-12 in the finite interpolated model, apart from mesh approximation.
    No claim of a continuous-state error bound is made.
    """

    mesh: SimplexMesh
    targets: FloatArray
    gross_samples: FloatArray
    horizon: int
    cost_rate: float
    V: FloatArray
    Qhold_grid: FloatArray
    tradebase: FloatArray
    P_grid: FloatArray
    P_target: FloatArray
    L_grid: FloatArray
    L_target: FloatArray
    log_fee_grid: FloatArray

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "solver": "finite-horizon fixed-target hold/restore approximate dynamic program",
            "return_model": "equal-weight stationary mixture of supplied joint scenario-year gross vectors",
            "sample_count": len(self.gross_samples),
            "sample_independence_claimed": False,
            "mesh_step": self.mesh.step,
            "mesh_states": self.mesh.size,
            "target_count": len(self.targets),
            "online_hold_extension": "direct empirical current log return; barycentric future value only",
            "interpolation": "nonnegative linear triangular barycentric; transition rows sum to one",
            "objective": "expected terminal log wealth factor, without drawdown or chance constraints",
            "action_gain_tolerance": ACTION_TOL,
            "value_table_uses_unthresholded_max": True,
            "continuous_state_optimum_claimed": False,
            "continuous_state_error_bound": None,
            "population_risk_guarantee": False,
        }


def solve_finite_horizon(joint_gross: ArrayLike, targets: ArrayLike, *, horizon: int = 30,
                         step: float = 0.05, cost_rate: float = 0.0025,
                         chunk_size: int = 32) -> FiniteHorizonSolution:
    """Solve all target columns in one batch; never choose between targets.

    At remaining horizon h, Qhold=Ls+Ps@V[h-1].  Target k's tradebase is
    Lt[k]+sum_m Pt[k,m]*V[h-1,m,k], so future execution costs are retained.
    Qtrade=log(1-rho(state,target))+tradebase.  Targets need not lie on the
    mesh: their current return and drift operator are computed separately.
    """
    horizon = _integer(horizon, "horizon")
    chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
    rate = _cost_rate(cost_rate)
    gross = _gross_samples(joint_gross)
    desired = _weights(targets, "targets")
    if desired.ndim == 1:
        desired = desired[None, :]
    if desired.ndim != 2:
        raise ValueError("targets must be one weight vector or a matrix of weight vectors")
    mesh = triangular_mesh(step)
    pg, lg = _transition_terms_flat(gross, mesh.states, mesh, chunk_size)
    pt, lt = _transition_terms_flat(gross, desired, mesh, chunk_size)
    rho = exact_fee_fraction(mesh.states[:, None, :], desired[None, :, :], rate)
    log_fee = np.log1p(-rho)
    shape = (horizon + 1, mesh.size, len(desired))
    values, hold = np.zeros(shape), np.zeros(shape)
    base = np.zeros((horizon + 1, len(desired)))
    identity = np.all(np.abs(mesh.states[:, None, :] - desired[None, :, :]) <= 32 * _EPS, axis=-1)
    for remaining in range(1, horizon + 1):
        previous = values[remaining - 1]
        base[remaining] = lt + np.einsum("km,mk->k", pt, previous, optimize=True)
        hold[remaining] = lg[:, None] + pg @ previous
        trade = log_fee + base[remaining][None, :]
        values[remaining] = np.maximum(hold[remaining], trade)
        # Identical weights are the same physical action; suppress only ULP
        # differences between separately accumulated grid/target operators.
        values[remaining][identity] = hold[remaining][identity]
    return FiniteHorizonSolution(mesh, _readonly(desired), _readonly(gross), horizon, rate,
                                 _readonly(values), _readonly(hold), _readonly(base),
                                 _readonly(pg), _readonly(pt), _readonly(lg), _readonly(lt),
                                 _readonly(log_fee))


def _broadcast_integer(value: ArrayLike, name: str, shape: tuple[int, ...], upper: int) -> IntArray:
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.integer) or np.issubdtype(result.dtype, np.bool_):
        raise ValueError(f"{name} must contain integers")
    try:
        result = np.broadcast_to(result, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast with current_weights") from exc
    if np.any(result < 0) or np.any(result > upper):
        raise ValueError(f"{name} must lie in [0,{upper}]")
    return result.astype(np.int64, copy=False)


def evaluate_action(solution: FiniteHorizonSolution, current_weights: ArrayLike,
                    remaining: ArrayLike, target_id: ArrayLike,
                    *, chunk_size: int = 32) -> dict[str, NDArray]:
    """Compare hold and restore at actual weights without any future-return input.

    ``remaining`` and ``target_id`` are integer scalars or arrays broadcasting
    with the leading weight axes.  Hold Q is L(u)+P(u)@V[h-1,:,target], using
    the frozen training mixture.  It is *not* interpolation of grid hold Q.
    Trade Q uses the exact current fee plus the stored target tradebase.

    Returns arrays ``hold_q``, ``trade_q``, ``gain``, ``trade``, ``identity``,
    ``fee_fraction``, ``current_log_return`` and ``one_period_gain`` with the
    common leading shape.  The last two expose the direct current expectation
    and log(1-rho)+L(target)-L(current), without a second prediction query.
    The fee fraction is
    hypothetical when trade=False; no fee is incurred by a hold action.
    For h=0, both Qs, gain and fee are zero and trading is prohibited.  When
    current weights equal the target (within core's ULP identity guard), the
    two Qs are set equal and hold is forced.  Otherwise trade iff gain>1e-12.
    """
    if not isinstance(solution, FiniteHorizonSolution):
        raise ValueError("solution must be produced by solve_finite_horizon")
    chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
    u = _weights(current_weights, "current_weights")
    try:
        shape = np.broadcast_shapes(u.shape[:-1], np.shape(remaining), np.shape(target_id))
    except ValueError as exc:
        raise ValueError("current_weights, remaining and target_id must broadcast") from exc
    r = _broadcast_integer(remaining, "remaining", shape, solution.horizon)
    k = _broadcast_integer(target_id, "target_id", shape, len(solution.targets) - 1)
    flat_u = np.broadcast_to(u, shape + (3,)).reshape(-1, 3)
    flat_r, flat_k = r.ravel(), k.ravel()
    desired = solution.targets[flat_k]
    identity = np.all(np.abs(flat_u - desired) <= 32 * _EPS, axis=-1)
    active = flat_r > 0
    qhold = np.zeros(len(flat_u))
    qtrade = np.zeros(len(flat_u))
    fees = np.zeros(len(flat_u))
    current_log = np.zeros(len(flat_u))
    one_period_gain = np.zeros(len(flat_u))
    if np.any(active):
        one_period = flat_r == 1
        longer = flat_r > 1
        # At h=1 V0=0 exactly; avoid building a needless transition matrix.
        if np.any(one_period):
            current_log[one_period] = _expected_log_flat(solution.gross_samples, flat_u[one_period], chunk_size)
            qhold[one_period] = current_log[one_period]
        if np.any(longer):
            operator, current_log[longer] = _transition_terms_flat(solution.gross_samples, flat_u[longer],
                                                                  solution.mesh, chunk_size)
            continuation = solution.V[flat_r[longer] - 1, :, flat_k[longer]]
            qhold[longer] = current_log[longer] + np.einsum("sm,sm->s", operator, continuation, optimize=True)
        fees[active] = exact_fee_fraction(flat_u[active], desired[active], solution.cost_rate)
        qtrade[active] = np.log1p(-fees[active]) + solution.tradebase[flat_r[active], flat_k[active]]
        one_period_gain[active] = (np.log1p(-fees[active]) + solution.L_target[flat_k[active]]
                                   - current_log[active])
    qtrade[identity] = qhold[identity]
    fees[identity] = 0.0
    one_period_gain[identity] = 0.0
    gain = qtrade - qhold
    trade = active & ~identity & (gain > ACTION_TOL)
    return {"hold_q": qhold.reshape(shape), "trade_q": qtrade.reshape(shape),
            "gain": gain.reshape(shape), "trade": trade.reshape(shape),
            "identity": identity.reshape(shape), "fee_fraction": fees.reshape(shape),
            "current_log_return": current_log.reshape(shape),
            "one_period_gain": one_period_gain.reshape(shape)}
