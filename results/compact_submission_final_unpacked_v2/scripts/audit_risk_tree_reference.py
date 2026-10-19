"""Independent, market-free enumeration audit of the short risk-tree reference.

The reference below uses its own scalar fee bisection and breadth-first action
bitmasks.  It never calls the reviewed module's accounting, fee, transition,
Pareto, policy-replay or mixture helpers.  The module under review supplies only
solve_risk_tree and its result structures.  All cases are fixed synthetic toys.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/model_review/risk_tree_reference_v1/independent_audit.json"
AUDIT_TOL = 2e-14
MDD_EVENT_TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__).resolve(), ROOT / "research/experimental_execution/risk_tree_reference.py",
             ROOT / "src/safeinvest/core.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def independent_restore(holdings, target, cost_rate):
    """Solve fee = c * total positive purchases independently by bisection."""
    wealth = float(sum(holdings))
    if not cost_rate:
        return [wealth * float(x) for x in target]
    low, high = 0.0, wealth * cost_rate
    for _ in range(90):
        fee = (low + high) / 2
        purchases = sum(max((wealth - fee) * float(target[j]) - holdings[j], 0.0) for j in range(3))
        if fee > cost_rate * purchases:
            high = fee
        else:
            low = fee
    fee = (low + high) / 2
    return [(wealth - fee) * float(x) for x in target]


def cases():
    """Fixed before inspecting the reviewed solver's output for these cases."""
    return [
        dict(name="H1_finite_duality_gap", gross=[[.8, 1, 1], [math.exp((.1 - .2 * math.log(.8)) / .8), 1, 1]],
             inflation=[0, 0], probabilities=[.2, .8], horizon=1, target=[1, 0, 0], loss_tolerance=.1,
             cost_rate=0, on_breach="continue", alpha=.1),
        dict(name="H2_joint_three_outcomes", gross=[[.7, 1.08, 1.01], [1.3, .95, 1], [1, 1.02, .99]],
             inflation=[.02, -.01, .01], probabilities=[.2, .3, .5], horizon=2, target=[.5, .25, .25],
             loss_tolerance=.12, cost_rate=.0025, on_breach="continue", alpha=.1),
        dict(name="H3_absorbing_hold_only", gross=[[.85, 1.04, 1.01], [1.25, .99, 1]],
             inflation=[.04, -.01], probabilities=[.25, .75], horizon=3, target=[.7, .2, .1],
             loss_tolerance=.2, cost_rate=.0025, on_breach="hold_only", alpha=.1, initial=[40, 20, 40]),
        dict(name="H2_initial_latched_review", gross=[[.8, 1, 1], [1.3, 1, 1]], inflation=[0, 0],
             probabilities=[.2, .8], horizon=2, target=[1, 0, 0], loss_tolerance=.1, cost_rate=.0025,
             on_breach="hold_only", alpha=.1, initial=[20, 20, 40], peak=100, breached=True),
        dict(name="H1_fee_only_endpoint", gross=[[1, 1, 1]], inflation=[0], probabilities=[1.], horizon=1,
             target=[1, 0, 0], loss_tolerance=.001, cost_rate=.0025, on_breach="continue", alpha=.1),
    ]


def independent_value(case, action_at_history, *, reject_pending_restore=False):
    """Direct terminal path enumeration, with an absorbing annual-endpoint event."""
    m, horizon = len(case["probabilities"]), case["horizon"]
    expected_growth = risk = 0.0
    for sequence in itertools.product(range(m), repeat=horizon):
        holdings = list(case.get("initial", [0., 0., 100.]))
        cpi = case.get("cpi", 1.)
        peak = case.get("peak", sum(holdings) / cpi)
        initial_real = sum(holdings) / cpi
        breached = case.get("breached", False) or 1 - initial_real / peak > case["loss_tolerance"] + MDD_EVENT_TOL
        probability, history = 1., ()
        for outcome in sequence:
            action = action_at_history(history)
            if breached and case["on_breach"] == "hold_only":
                if reject_pending_restore and action != "hold":
                    raise AssertionError("Frozen policy restores after a latched hold-only breach")
                action = "hold"
            if action == "restore":
                holdings = independent_restore(holdings, case["target"], case["cost_rate"])
            elif action != "hold":
                raise AssertionError("Unknown action in independent replay")
            holdings = [holdings[j] * case["gross"][outcome][j] for j in range(3)]
            cpi *= 1 + case["inflation"][outcome]
            real = sum(holdings) / cpi
            peak = max(peak, real)
            breached = breached or 1 - real / peak > case["loss_tolerance"] + MDD_EVENT_TOL
            probability *= case["probabilities"][outcome]
            history += (outcome,)
        expected_growth += probability * math.log((sum(holdings) / cpi) / initial_real)
        risk += probability * breached
    return float(expected_growth), float(risk)


def independent_points(case):
    m, horizon = len(case["probabilities"]), case["horizon"]
    number_of_action_nodes = sum(m ** h for h in range(horizon))
    points = []
    for mask in range(1 << number_of_action_nodes):
        def action(history):
            index = 0
            for outcome in history:
                index = m * index + 1 + outcome
            return "restore" if (mask >> index) & 1 else "hold"
        growth, risk = independent_value(case, action)
        points.append((growth, risk, mask))
    return points


def independent_mixture(points, alpha):
    """One/two-policy vertices of the finite root-mixture relaxation."""
    feasible = [(g, q) for g, q, _ in points if q <= alpha]
    if not feasible:
        return None
    best = max(g for g, _ in feasible)
    for g1, q1, _ in points:
        if q1 >= alpha:
            continue
        for g2, q2, _ in points:
            if q2 > alpha:
                fraction = (alpha - q1) / (q2 - q1)
                best = max(best, (1 - fraction) * g1 + fraction * g2)
    return float(best)


def reviewed_policy_action(policy, history):
    """Read frozen tree structure; no reviewed policy-replay helper is used."""
    node = policy
    for outcome in history:
        node = node.children[node.outcome_ids.index(outcome)]
    return node.action


def run_audit():
    # Import only after the independent implementation and fixed cases exist.
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
    from experimental_execution.risk_tree_reference import RiskState, solve_risk_tree

    before = sources()
    rows, checks = [], []
    max_optimum_error = max_pair_error = max_replay_error = max_mixture_error = 0.

    def check(condition, name):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    for case in cases():
        independent = independent_points(case)
        runs = []
        arguments = {key: case[key] for key in ["horizon", "loss_tolerance", "alpha", "on_breach", "cost_rate"]}
        if "initial" in case:
            arguments["initial_state"] = RiskState(tuple(case["initial"]), case.get("cpi", 1.),
                                                    case.get("peak", sum(case["initial"])), case.get("breached", False))
        for prune in [True, False]:
            result = solve_risk_tree(case["gross"], case["inflation"], case["probabilities"], case["target"],
                                     **arguments, prune=prune)
            metadata = result.metadata
            requested = float(getattr(result, "requested_alpha", metadata.get("requested_alpha", case["alpha"])))
            effective = float(getattr(result, "effective_alpha", metadata.get("effective_alpha", result.alpha)))
            feasible = [(g, q, mask) for g, q, mask in independent if q <= effective]
            optimum = max((g for g, _, _ in feasible), default=None)
            actual = None if result.selected is None else result.selected.g
            prefix = f"{case['name']}/prune={prune}"
            check((actual is None) == (optimum is None), prefix + "/feasibility")
            optimum_error = 0. if actual is None else abs(actual - optimum)
            check(optimum_error <= AUDIT_TOL, prefix + "/optimal_expected_growth")
            pair_error = max(min(max(abs(point.g - g), abs(point.q - q)) for g, q, _ in independent)
                             for point in result.points)
            check(pair_error <= AUDIT_TOL, prefix + "/all_returned_pairs_have_independent_counterpart")
            replay_error = 0.
            if result.selected is not None:
                replay_g, replay_q = independent_value(case, lambda history: reviewed_policy_action(result.selected.policy, history),
                                                       reject_pending_restore=True)
                replay_error = max(abs(replay_g - result.selected.g), abs(replay_q - result.selected.q))
                check(replay_error <= AUDIT_TOL, prefix + "/independent_selected_policy_replay")
                check(replay_q <= effective + AUDIT_TOL, prefix + "/effective_chance_budget")
            mixture_g = independent_mixture(independent, effective)
            check((mixture_g is None) == (not result.randomized_bound.feasible), prefix + "/mixture_feasibility")
            mixture_error = 0. if mixture_g is None else abs(mixture_g - result.randomized_bound.g)
            check(mixture_error <= AUDIT_TOL, prefix + "/mixture_expected_growth")
            if result.randomized_bound.feasible:
                check(abs(result.randomized_bound.duality_gap) <= AUDIT_TOL, prefix + "/mixture_primal_dual_agreement")
                check(result.randomized_bound.g + AUDIT_TOL >= result.selected.g, prefix + "/mixture_is_upper_bound")
            max_optimum_error = max(max_optimum_error, optimum_error)
            max_pair_error = max(max_pair_error, pair_error)
            max_replay_error = max(max_replay_error, replay_error)
            max_mixture_error = max(max_mixture_error, mixture_error)
            runs.append({"prune": prune, "requested_alpha": requested, "effective_alpha": effective,
                         "numerical_budget_expansion": effective - requested, "selected_g": actual,
                         "selected_q": None if result.selected is None else result.selected.q,
                         "selected_q_minus_requested_alpha": None if result.selected is None else result.selected.q - requested,
                         "independent_optimal_g": optimum, "independent_mixture_g": mixture_g,
                         "reviewed_mixture_g": result.randomized_bound.g,
                         "max_optimum_error": optimum_error, "max_root_pair_error": pair_error,
                         "max_selected_policy_replay_error": replay_error, "mixture_growth_error": mixture_error,
                         "returned_root_pairs": len(result.points), "solver_metadata": metadata})
        rows.append({"case": case, "independently_enumerated_masks": len(independent), "runs": runs})
    after = sources()
    check(before == after, "sources_stable_during_audit")
    return {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
            "runtime": {"python": sys.version, "numpy": np.__version__},
            "case_count": len(rows), "solver_comparisons": sum(len(row["runs"]) for row in rows),
            "checks_passed": len(checks), "checks": checks,
            "maximum_errors": {"optimal_expected_growth": max_optimum_error, "returned_root_pair": max_pair_error,
                               "selected_policy_independent_replay": max_replay_error, "mixture_growth": max_mixture_error},
            "source_sha256_at_start": before, "source_sha256_at_end": after,
            "market_data_read": False, "final_scenarios_read": False,
            "independence": "Own 90-step scalar fee bisection, breadth-first action bitmasks and terminal path enumeration; no reviewed step/fee/evaluate/Pareto/mixture helper used",
            "scope": "Finite synthetic iid support, deterministic nonanticipating full-history policy trees; this class can exceed financial-state-only deterministic Markov policies",
            "observations": "Initial real peak plus annual endpoints only; fees are included in endpoint wealth; no fee-only observation or terminal liquidation",
            "numerical_semantics": {"audit_absolute_tolerance": AUDIT_TOL, "mdd_event_tolerance": MDD_EVENT_TOL,
                                    "budget": "Requested/effective numerical chance budgets are separately recorded; comparisons use the same effective budget, without claiming exact requested-alpha feasibility"},
            "non_guarantees": ["No market model calibration or policy selection", "No population or real-market risk certificate",
                               "No continuous-state/long-horizon solver validation", "No general floating-point, approximation or policy-error bound",
                               "Root randomization is a mathematical relaxation, not a deployed advice policy"],
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()
    result = run_audit()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "cases": result["case_count"], "comparisons": result["solver_comparisons"],
                      "checks": result["checks_passed"], "maximum_errors": result["maximum_errors"],
                      "output": str(arguments.out), "output_sha256": sha(arguments.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
