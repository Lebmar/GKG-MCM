"""Exhaustive synthetic-tree audit of the fixed-target execution prototype.

The continuous reference uses independent fee bisection and explicit outcome
trees. Production transition, interpolation and fee routines never define its
values. The prototype is called only to solve and choose its tested policy.
No market file or cached market outcome is loaded by this script.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/finite_horizon_tree_audit_protocol.json"
OUTPUT = ROOT / "results/model_review/finite_horizon_execution_v1/synthetic_tree_audit.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Synthetic weights must be finite, nonnegative three-vectors")
    if not np.isclose(result.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("Synthetic weights must sum to one")
    return result / result.sum()


def independent_fee(current, target, rate):
    """Unit-wealth fee root, without importing either production fee solver."""
    current, target = normalize(current), normalize(target)
    if rate == 0 or np.array_equal(current, target):
        return 0.
    lower, upper = 0., float(rate)
    for _ in range(90):
        middle = lower + (upper - lower) / 2
        purchases = sum(max((1 - middle) * w - x, 0.) for w, x in zip(target, current))
        if middle > rate * purchases:
            upper = middle
        else:
            lower = middle
    return lower + (upper - lower) / 2


class ContinuousTree:
    """Exact finite support recursion, up to floating-point arithmetic."""
    def __init__(self, gross, target, rate):
        self.gross = np.asarray(gross, float)
        self.target, self.rate = normalize(target), rate
        if self.gross.ndim != 2 or self.gross.shape[1] != 3 or np.any(self.gross <= 0):
            raise ValueError("Positive joint gross support required")
        self.counters = Counter()

    def outcomes(self, current, restore):
        u = normalize(current)
        fee = independent_fee(u, self.target, self.rate) if restore else 0.
        base = self.target if restore else u
        fee_reward = math.log1p(-fee)
        for gross in self.gross:
            money = base * gross
            factor = float(money.sum())
            state = normalize(money / factor)
            yield fee_reward + math.log(factor), tuple(float(v) for v in state)

    @lru_cache(maxsize=None)
    def optimal(self, state, horizon):
        self.counters["optimal_unique_state_horizon_nodes"] += 1
        if horizon == 0:
            return {"value": 0., "hold_q": 0., "restore_q": 0., "gain": 0., "action": "hold"}
        values = []
        for restore in (False, True):
            branch_values = []
            for reward, next_state in self.outcomes(state, restore):
                self.counters["optimal_outcome_branches"] += 1
                branch_values.append(reward + self.optimal(next_state, horizon - 1)["value"])
            values.append(math.fsum(branch_values) / len(branch_values))
        # Exactly equal portfolio weights are one physical action, including
        # tiny floating-point differences accumulated by state normalization.
        if np.max(np.abs(np.asarray(state) - self.target)) <= 4 * np.finfo(float).eps:
            values[1] = values[0]
        qhold, qtrade = values
        return {"value": max(values), "hold_q": qhold, "restore_q": qtrade,
                "gain": qtrade - qhold, "action": "restore" if qtrade > qhold else "hold"}

    @lru_cache(maxsize=None)
    def baseline(self, state, horizon, restore):
        self.counters["baseline_unique_state_horizon_nodes"] += 1
        if horizon == 0:
            return 0.
        branches = [reward + self.baseline(next_state, horizon - 1, restore)
                    for reward, next_state in self.outcomes(state, restore)]
        self.counters["baseline_outcome_branches"] += len(branches)
        return math.fsum(branches) / len(branches)


def independent_grid_value(solution, horizon, current):
    """Generic area-coordinate interpolation, used only to display grid V.

    This is not used by the optimal reference or implemented-policy replay.
    Triangle coordinates are constructed independently of prototype helpers.
    """
    u = normalize(current)
    divisions = int(round(1 / solution.mesh.step))
    states = np.asarray(solution.mesh.states)
    lookup = {tuple(np.rint(s[:2] * divisions).astype(int)): i for i, s in enumerate(states)}
    triangles = []
    for i in range(divisions):
        for j in range(divisions - i):
            triangles.append([lookup[i, j], lookup[i + 1, j], lookup[i, j + 1]])
            if i + j < divisions - 1:
                triangles.append([lookup[i + 1, j], lookup[i, j + 1], lookup[i + 1, j + 1]])
    triangles = np.asarray(triangles, int)
    points = states[triangles, :2]
    a, b, c = points[:, 0], points[:, 1], points[:, 2]
    ab, ac, au = b - a, c - a, u[:2] - a
    area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    coefficient_b = (au[:, 0] * ac[:, 1] - au[:, 1] * ac[:, 0]) / area
    coefficient_c = (ab[:, 0] * au[:, 1] - ab[:, 1] * au[:, 0]) / area
    coefficients = np.column_stack((1 - coefficient_b - coefficient_c, coefficient_b, coefficient_c))
    valid = np.flatnonzero(np.all(coefficients >= -1e-12, axis=1))
    if not len(valid):
        raise RuntimeError("Independent display interpolation found no containing triangle")
    chosen = int(valid[0])
    weights = np.maximum(coefficients[chosen], 0)
    weights /= weights.sum()
    value = float(np.dot(weights, solution.V[horizon, triangles[chosen], 0]))
    exact_node = np.flatnonzero(np.max(np.abs(states - u), axis=1) <= 1e-13)
    return {"value": value, "initial_state_on_mesh": bool(len(exact_node)),
            "kind": "projected-chain grid-table value, independently interpolated only for display",
            "triangle_vertex_ids": triangles[chosen].tolist(), "barycentric_weights": weights.tolist()}


def implemented_policy_value(tree, solution, evaluate_action, initial, horizon):
    """Enumerate the policy's continuous-state payoffs with independent fees."""
    queries = {}
    maximum_fee_error = 0.
    counter = Counter()

    @lru_cache(maxsize=None)
    def replay(state, remaining):
        nonlocal maximum_fee_error
        counter["unique_state_horizon_nodes"] += 1
        query = evaluate_action(solution, np.asarray(state), remaining, 0)
        restore = bool(query["trade"])
        data = {"hold_q": float(query["hold_q"]), "restore_q": float(query["trade_q"]),
                "gain": float(query["gain"]), "trade": restore,
                "fee_fraction": float(query["fee_fraction"]), "identity": bool(query["identity"])}
        queries[state, remaining] = data
        if remaining == 0:
            if restore or data["gain"] != 0 or data["fee_fraction"] != 0:
                raise RuntimeError("Implementation permits a terminal trade or terminal fee")
            return 0.
        independent = independent_fee(state, tree.target, tree.rate)
        maximum_fee_error = max(maximum_fee_error, abs(data["fee_fraction"] - independent))
        branches = []
        for reward, next_state in tree.outcomes(state, restore):
            counter["outcome_branches"] += 1
            branches.append(reward + replay(next_state, remaining - 1))
        return math.fsum(branches) / len(branches)

    value = replay(tuple(initial), horizon)
    return value, queries[tuple(initial), horizon], maximum_fee_error, dict(counter)


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.failures = []

    def require(self, condition, name, context=None):
        self.counts[name] += 1
        if not condition:
            self.failures.append({"check": name, "context": context})


def run(protocol, checks):
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
    from experimental_execution.finite_horizon import solve_finite_horizon, evaluate_action

    rows, reference_counts = [], []
    solve_count = 0
    maximum_horizon = max(protocol["horizons"])
    for case in protocol["cases"]:
        initial = tuple(float(v) for v in normalize(case["initial_state"]))
        gross = np.asarray(case["gross"], float)
        if len(gross) not in (2, 3):
            raise ValueError("Audit supports must have exactly two or three outcomes")
        for rate in protocol["cost_rates"]:
            tree = ContinuousTree(gross, case["target"], rate)
            exact = {h: tree.optimal(initial, h) for h in protocol["horizons"]}
            baselines = {h: {"never_restore": tree.baseline(initial, h, False),
                             "restore_every_period": tree.baseline(initial, h, True)} for h in protocol["horizons"]}
            for horizon in protocol["horizons"]:
                for name, value in baselines[horizon].items():
                    checks.require(exact[horizon]["value"] - value >= -1e-10,
                                   "exact_optimum_dominates_continuous_baseline", (case["name"], rate, horizon, name))
            for step in protocol["mesh_steps"]:
                solution = solve_finite_horizon(gross[None, :, :], np.asarray(case["target"]),
                                                 horizon=maximum_horizon, step=step, cost_rate=rate)
                solve_count += 1
                checks.require(not solution.metadata["continuous_state_optimum_claimed"]
                               and not solution.metadata["population_risk_guarantee"], "honest_prototype_metadata", case["name"])
                checks.require(np.all(solution.P_grid >= 0) and np.allclose(solution.P_grid.sum(axis=1), 1, atol=1e-12, rtol=0),
                               "prototype_transition_stochastic", (case["name"], rate, step))
                for horizon in protocol["horizons"]:
                    implemented, query, fee_error, counts = implemented_policy_value(tree, solution, evaluate_action, initial, horizon)
                    optimum = exact[horizon]
                    gap = optimum["value"] - implemented
                    context = (case["name"], rate, horizon, step)
                    checks.require(gap >= -1e-10, "implemented_policy_cannot_exceed_exact_optimum", (context, gap))
                    checks.require(fee_error <= 1e-11, "all_policy_queries_match_independent_fee", (context, fee_error))
                    if horizon == 1:
                        checks.require(abs(query["hold_q"] - optimum["hold_q"]) <= 1e-10,
                                       "h1_direct_hold_q_exact", context)
                        checks.require(abs(query["restore_q"] - optimum["restore_q"]) <= 1e-10,
                                       "h1_direct_restore_q_exact", context)
                        checks.require(abs(gap) <= 1e-10, "h1_actual_policy_value_exact", context)
                    if case["name"] in ("offmesh_identity_thirds", "vertex_identity"):
                        checks.require(not query["trade"] and query["fee_fraction"] == 0,
                                       "identity_first_action_holds_without_fee", context)
                    if case["name"] == "dominated_bill_target":
                        checks.require(optimum["action"] == "hold" and not query["trade"],
                                       "dominated_target_not_restored", context)
                    if case["name"] == "initial_fee_amortization_h2" and rate == .0025:
                        expected_restore = horizon >= 2
                        checks.require((optimum["action"] == "restore") == expected_restore
                                       and query["trade"] == expected_restore, "initial_fee_amortized_with_horizon", context)
                    if case["name"] == "future_fee_closed_form" and rate == .0025 and horizon == 2:
                        closed = 2 * math.log(1.1) + math.log1p(-.00025 / 1.00125)
                        checks.require(abs(optimum["value"] - closed) <= 1e-10 and abs(implemented - closed) <= 1e-10,
                                       "future_fee_closed_form", (context, optimum["value"], implemented, closed))
                    if case["name"] == "offmesh_last_period_counterexample" and rate == .0025 and horizon == 1:
                        checks.require(not query["trade"], "direct_query_avoids_rejected_grid_q_trade", context)
                    display = independent_grid_value(solution, horizon, initial)
                    policy_action = "restore" if query["trade"] else "hold"
                    mismatch = policy_action != optimum["action"]
                    best_baseline = max(baselines[horizon].values())
                    row = {"case": case["name"], "cost_rate": rate, "horizon": horizon,
                           "mesh_step": step, "mesh_states": solution.mesh.size,
                           "initial_state": list(initial), "target": normalize(case["target"]).tolist(),
                           "equal_mass_joint_gross": case["gross"], "continuous_exact": optimum,
                           "continuous_baseline_values": baselines[horizon],
                           "projected_chain_grid_table": display,
                           "initial_direct_query": query,
                           "initial_unthresholded_greedy_prediction": max(query["hold_q"], query["restore_q"]),
                           "initial_executed_action_prediction": query["restore_q"] if query["trade"] else query["hold_q"],
                           "implemented_policy_continuous_expected_value": implemented,
                           "optimal_minus_implemented_signed": gap,
                           "grid_table_minus_continuous_optimum": display["value"] - optimum["value"],
                           "grid_table_minus_implemented_policy": display["value"] - implemented,
                           "greedy_prediction_minus_implemented_policy": max(query["hold_q"], query["restore_q"]) - implemented,
                           "implemented_minus_best_continuous_baseline": implemented - best_baseline,
                           "first_action_mismatch": mismatch,
                           "substantive_first_action_mismatch": mismatch and abs(optimum["gain"]) > 1e-10,
                           "maximum_queried_fee_fraction_error": fee_error,
                           "implemented_policy_tree_counts": counts}
                    if case["name"] == "offmesh_last_period_counterexample" and horizon == 1:
                        # Show the rejected shortcut independently; it is not
                        # used by the tested direct-transition policy.
                        rejected_hold = display["value"]
                        # V1 includes max over actions, so display of Qhold1
                        # requires the same independent geometry with that table.
                        class HoldTable:
                            mesh = solution.mesh
                            V = solution.Qhold_grid
                        rejected_hold = independent_grid_value(HoldTable(), horizon, initial)["value"]
                        row["rejected_grid_q_shortcut"] = {"interpolated_hold_q": rejected_hold,
                                                           "restore_q": query["restore_q"],
                                                           "would_restore": query["restore_q"] - rejected_hold > 1e-12,
                                                           "used_for_implemented_policy": False}
                    rows.append(row)
            reference_counts.append({"case": case["name"], "cost_rate": rate, **dict(tree.counters)})
    pairs = []
    for case in protocol["cases"]:
        for rate in protocol["cost_rates"]:
            for horizon in protocol["horizons"]:
                group = [r for r in rows if r["case"] == case["name"] and r["cost_rate"] == rate and r["horizon"] == horizon]
                group.sort(key=lambda r: -r["mesh_step"])
                coarse, fine = group
                pairs.append({"case": case["name"], "cost_rate": rate, "horizon": horizon,
                              "coarse_step": coarse["mesh_step"], "fine_step": fine["mesh_step"],
                              "coarse_actual_policy_gap": coarse["optimal_minus_implemented_signed"],
                              "fine_actual_policy_gap": fine["optimal_minus_implemented_signed"],
                              "fine_minus_coarse_policy_value": fine["implemented_policy_continuous_expected_value"] - coarse["implemented_policy_continuous_expected_value"],
                              "first_action_changed": coarse["initial_direct_query"]["trade"] != fine["initial_direct_query"]["trade"],
                              "no_monotone_improvement_asserted": True})
    aggregates = {}
    for step in protocol["mesh_steps"]:
        selected = [r for r in rows if r["mesh_step"] == step]
        aggregates[str(step)] = {"cases_costs_horizons": len(selected),
                                "maximum_actual_policy_gap": max(r["optimal_minus_implemented_signed"] for r in selected),
                                "minimum_actual_policy_gap_signed": min(r["optimal_minus_implemented_signed"] for r in selected),
                                "maximum_abs_grid_table_error": max(abs(r["grid_table_minus_continuous_optimum"]) for r in selected),
                                "maximum_abs_prediction_minus_actual_policy": max(abs(r["greedy_prediction_minus_implemented_policy"]) for r in selected),
                                "first_action_mismatches": sum(r["first_action_mismatch"] for r in selected),
                                "substantive_first_action_mismatches": sum(r["substantive_first_action_mismatch"] for r in selected),
                                "actual_policy_below_best_baseline_count_beyond_tolerance": sum(r["implemented_minus_best_continuous_baseline"] < -1e-10 for r in selected),
                                "maximum_queried_fee_fraction_error": max(r["maximum_queried_fee_fraction_error"] for r in selected)}
    return {"comparison_rows": rows, "mesh_resolution_pairs": pairs, "summary_by_mesh": aggregates,
            "prototype_solve_count": solve_count, "continuous_reference_counters": reference_counts,
            "comparisons": len(rows), "independent_cases": len(protocol["cases"])}


def main():
    checks = Checks()
    sources = [PROTOCOL, Path(__file__), ROOT / "research/experimental_execution/finite_horizon.py",
               ROOT / "src/safeinvest/core.py", ROOT / "tests/test_finite_horizon_execution.py"]
    sources_start = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    protocol = json.loads(PROTOCOL.read_text())
    report = {"audit_version": protocol["version"], "completed_utc": None,
              "scope": protocol["scope"], "source_sha256_start": sources_start,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "market_data_loaded": False, "policy_selected_or_changed": False,
              "reference_independence": "No production fee, transition or interpolation routine defines the continuous reference or realized tree payoffs. The prototype chooses its own tested policy actions only.",
              "limitations": ["Finite predeclared synthetic cases only; no universal approximation bound",
                              "Exact control is restricted to hold/restore the specified fixed target",
                              "Expected-log comparisons under the defined iid support, without MDD constraints",
                              "Projected grid-table values differ conceptually from independently replayed continuous-state policy values"]}
    try:
        report.update(run(protocol, checks))
    except Exception as exc:
        checks.failures.append({"check": "audit_execution_exception", "type": type(exc).__name__,
                                "message": str(exc), "traceback": traceback.format_exc()})
    sources_end = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    checks.require(sources_start == sources_end, "sources_unchanged_during_tree_audit")
    report.update({"passed": not checks.failures, "check_invocations": dict(checks.counts),
                   "failures": checks.failures, "source_sha256_end": sources_end,
                   "completed_utc": datetime.now(timezone.utc).isoformat()})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    OUTPUT.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if sha(OUTPUT) != digest:
        raise RuntimeError("Tree audit output hash mismatch")
    print(json.dumps({"passed": report["passed"], "comparisons": report.get("comparisons"),
                      "summary_by_mesh": report.get("summary_by_mesh"), "failures": report["failures"],
                      "output": str(OUTPUT), "output_sha256": digest}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
