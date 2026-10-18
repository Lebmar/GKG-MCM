"""Pure contest-service governance; no market I/O, fitting or investment actions.

Training candidates support conditional catalogue proposals only. Admission needs
a separately verified, version-matched whole-service validation receipt. A target
reserve convention is deliberately distinct from a live cash-balance promise.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real


PRODUCTS = ("Conservative", "Balanced", "Growth")
DEFAULT_FLOORS = {"High": .30, "Medium": .10, "Low": 0.0}
LARGER_FLOORS = {"High": .50, "Medium": .20, "Low": 0.0}
POLICY_ID = "contest_ordinal_target_reserve_v1"
ALPHA = .10
COST_RATE = .0025
TOL = 1e-12
RISK_METRIC = "annual_endpoint_net_real_MDD_v1"
GROWTH_METRIC = "mean_annual_log_net_real_terminal_growth_v1"
SEVERITY_METRIC = "fractional_empirical_CVaR90_MDD_v1"
EXCESS_METRIC = "mean_positive_MDD_excess_above_individual_L_v1"
TRAINING_RULE = "complete_empirical_count_screen_v1"
PROPOSAL_SORT_RULE = "global_G_excess_CVaR_canonical_bands_v1"
JOINT_RECEIPT_SCHEMA = "whole_service_validation_with_fit_lineage_v1"
COVERAGE_UNITS = ("illustrative_profile_cases", "training_refit_runs", "market_scenario_paths")


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} outside declared range")
    return value


def _count(value, name, low=0):
    value = _number(value, name, low=low)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _weights(values, *, positive):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("weights require exactly three assets")
    result = [_number(v, "asset weight", 0, 1) for v in values]
    if abs(sum(result) - 1) > TOL or (positive and min(result) <= 0):
        raise ValueError("fully invested weights and positive product assets required")
    return result


def _profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("profile must be a mapping")
    name = profile.get("id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile id is required")
    h = _count(profile.get("horizon"), "horizon", 1)
    loss = _number(profile.get("loss_tolerance"), "loss tolerance", 0, 1)
    label = profile.get("liquidity")
    if h > 30 or not 0 < loss < 1 or label not in DEFAULT_FLOORS:
        raise ValueError("profile outside the declared short-input domain")
    return {"id": name, "horizon": h, "loss_tolerance": loss, "liquidity": label}


def _catalogue(portfolios):
    if not isinstance(portfolios, (list, tuple)) or len(portfolios) != 3:
        raise ValueError("exactly three common products required")
    if not all(isinstance(p, dict) for p in portfolios) or {p.get("name") for p in portfolios} != set(PRODUCTS):
        raise ValueError("three distinct named products required")
    by_name = {p["name"]: _weights(p.get("weights"), positive=True) for p in portfolios}
    return [{"name": name, "weights": by_name[name]} for name in PRODUCTS]


def make_policy(floors=None):
    """Return a self-identifying normative policy, never an estimated preference."""
    floors = dict(DEFAULT_FLOORS if floors is None else floors)
    if set(floors) != set(DEFAULT_FLOORS):
        raise ValueError("all three liquidity labels are required")
    floors = {k: _number(floors[k], f"{k} floor", 0, 1) for k in DEFAULT_FLOORS}
    if not floors["High"] >= floors["Medium"] >= floors["Low"]:
        raise ValueError("ordinal floors must be monotone")
    policy = {"policy_id": POLICY_ID, "service_alpha": ALPHA, "cost_rate": COST_RATE,
              "target_bill_floors": floors, "risk_metric_id": RISK_METRIC,
              "growth_metric_id": GROWTH_METRIC, "severity_metric_id": SEVERITY_METRIC,
              "excess_metric_id": EXCESS_METRIC, "proposal_sort_rule_id": PROPOSAL_SORT_RULE,
              "training_rule_id": TRAINING_RULE, "comparison_tolerance": TOL,
              "runtime_drift_rule": "notice_only_no_investment_action"}
    return {**policy, "fingerprint": _digest(policy)}


def catalogue_fingerprint(portfolios):
    return _digest(_catalogue(portfolios))


def target_compatibility(profile, target_weights, *, floors=None):
    """Screen a proposed target without altering it or approving its financial merits."""
    profile, target = _profile(profile), _weights(target_weights, positive=True)
    floor = make_policy(floors)["target_bill_floors"][profile["liquidity"]]
    compatible = target[2] >= floor-TOL
    return {"target_reserve_compatible": compatible, "target_bill_floor": floor,
            "target_weights": target, "reason_code": ("ordinal_target_reserve_compatible" if compatible
                else "reject_target_below_ordinal_reserve_convention"),
            "risk_or_execution_approved": False, "decision_action": None}


def _lineage(value, n):
    if not isinstance(value, dict) or not _sha_like(value.get("source_artifact_sha256")):
        raise ValueError("source fit artifact hash required")
    record, ids = value.get("source_record"), value.get("training_ids")
    if not isinstance(record, str) or not record or not isinstance(ids, list):
        raise ValueError("fit record and original training ids required")
    ids = [_count(i, "training scenario id", 1) for i in ids]
    if len(ids) != n or ids != sorted(set(ids)) or any(i > 120 for i in ids):
        raise ValueError("original development training ids must match training n")
    return {"source_artifact_sha256": value["source_artifact_sha256"],
            "source_record": record, "training_ids": ids}


def _training_payload(profile, by_name):
    candidates = []
    for name in PRODUCTS:
        c = by_name[name]
        fields = {"portfolio": name, "weights": _weights(c["weights"], positive=True),
                  "horizon": int(c["horizon"]), "loss_tolerance": float(c["loss_tolerance"]),
                  "n": int(c["n"]), "breach_count": int(c["breach_count"]), "complete": True}
        fields.update({key: float(c[key]) for key in ("training_gamma", "cost_rate",
            "mean_annual_log_growth", "expected_excess", "cvar90_mdd")})
        fields.update({key: c[key] for key in ("risk_metric_id", "growth_metric_id", "severity_metric_id",
            "excess_metric_id", "training_rule_id", "source_policy_id", "source_context")})
        fields["fit_lineage"] = _lineage(c["fit_lineage"], fields["n"])
        candidates.append(fields)
    return {"profile": _profile(profile), "training_candidates": candidates}


def _candidate_issues(candidates, profile, catalogue):
    """An incomplete or mismatched evidence set is not a financial risk failure."""
    if not isinstance(candidates, (list, tuple)) or len(candidates) != 3:
        return ["complete_three_product_training_evidence_required"], {}
    if not all(isinstance(c, dict) for c in candidates):
        return ["training_candidates_must_be_mappings"], {}
    if {c.get("portfolio") for c in candidates} != set(PRODUCTS):
        return ["training_product_set_mismatch"], {}
    issues, by_name = [], {c["portfolio"]: c for c in candidates}
    catalogue = {p["name"]: p["weights"] for p in catalogue}
    contexts = set()
    for name in PRODUCTS:
        c = by_name[name]
        try:
            n = _count(c.get("n"), "training n", 1)
            k = _count(c.get("breach_count"), "training breaches")
            gamma = _number(c.get("training_gamma"), "training gamma", 0, 1)
            cost = _number(c.get("cost_rate"), "training cost rate", 0, 1)
            _count(c.get("horizon"), "training horizon", 1)
            _number(c.get("loss_tolerance"), "training loss tolerance", 0, 1)
            _number(c.get("mean_annual_log_growth"), "training growth")
            _number(c.get("cvar90_mdd"), "training CVaR", 0, 1)
            _number(c.get("expected_excess"), "training expected excess", 0, 1)
            weights = _weights(c.get("weights"), positive=True)
            lineage = _lineage(c.get("fit_lineage"), n)
            if k > n:
                raise ValueError("breach count exceeds training n")
            if any(abs(a-b) > TOL for a, b in zip(weights, catalogue[name])):
                issues.append(f"{name}:training_target_mismatch")
            if (c.get("horizon") != profile["horizon"] or
                    c.get("loss_tolerance") != profile["loss_tolerance"]):
                issues.append(f"{name}:training_H_L_mismatch")
            if c.get("complete") is not True:
                issues.append(f"{name}:incomplete_training_evidence")
            if cost != COST_RATE:
                issues.append(f"{name}:training_cost_rate_mismatch")
            if gamma > ALPHA:
                issues.append(f"{name}:training_gamma_above_service_alpha")
            for key, expected in [("risk_metric_id", RISK_METRIC),
                                  ("growth_metric_id", GROWTH_METRIC),
                                  ("severity_metric_id", SEVERITY_METRIC),
                                  ("excess_metric_id", EXCESS_METRIC),
                                  ("training_rule_id", TRAINING_RULE)]:
                if c.get(key) != expected:
                    issues.append(f"{name}:{key}_mismatch")
            if not c.get("source_policy_id") or not c.get("source_context"):
                issues.append(f"{name}:missing_training_provenance")
            contexts.add((n, gamma, c.get("source_policy_id"), c.get("source_context"), _digest(lineage)))
        except (ValueError, TypeError):
            issues.append(f"{name}:invalid_training_evidence")
    if len(contexts) > 1:
        issues.append("mixed_training_context_or_screen")
    return issues, by_name


def propose_assignment(profile, portfolios, candidates, *, floors=None):
    """Select a conditional product; never convert training evidence into service admission."""
    profile, catalogue, policy = _profile(profile), _catalogue(portfolios), make_policy(floors)
    floor = policy["target_bill_floors"][profile["liquidity"]]
    result = {"profile": profile, "service_policy": policy,
              "catalogue_fingerprint": _digest(catalogue), "catalogue": catalogue,
              "target_bill_floor": floor, "assignment": "Review", "target_weights": None,
              "proposal_status": "review", "review_type": None, "reason_codes": [],
              "pair_assessments": [], "financial_outcome": None,
              "training_evidence_payload": None, "training_evidence_fingerprint": None,
              "training_evidence_is_joint_service_validation": False}
    issues, by_name = _candidate_issues(candidates, profile, catalogue)
    if issues:
        return {**result, "review_type": "model_review", "reason_codes": issues}
    payload = _training_payload(profile, by_name)
    result.update(training_evidence_payload=payload, training_evidence_fingerprint=_digest(payload))
    eligible = []
    for p in catalogue:
        name, c = p["name"], by_name[p["name"]]
        kmax = math.floor(float(c["training_gamma"])*int(c["n"]) + TOL)
        risk = int(c["breach_count"]) <= kmax
        liquidity = p["weights"][2] >= floor-TOL
        pair = {"portfolio": name, "risk_training_candidate": risk,
                "liquidity_target_compatible": liquidity,
                "training_breach_count": int(c["breach_count"]), "training_n": int(c["n"]),
                "training_gamma": float(c["training_gamma"]), "cost_rate": float(c["cost_rate"]),
                "training_count_limit": kmax,
                "mean_annual_log_growth": float(c["mean_annual_log_growth"]),
                "cvar90_mdd": float(c["cvar90_mdd"]),
                "expected_excess": float(c["expected_excess"]),
                "source_policy_id": c["source_policy_id"], "source_context": c["source_context"]}
        result["pair_assessments"].append(pair)
        if risk and liquidity:
            eligible.append(pair)
    if not eligible:
        risk_any = any(p["risk_training_candidate"] for p in result["pair_assessments"])
        return {**result, "review_type": "liquidity_review" if risk_any else "risk_review",
                "reason_codes": ["no_risk_candidate_satisfies_target_reserve" if risk_any
                                 else "no_training_risk_candidate"]}
    gmax = max(p["mean_annual_log_growth"] for p in eligible)
    band = [p for p in eligible if p["mean_annual_log_growth"] >= gmax-TOL]
    emin = min(p["expected_excess"] for p in band)
    band = [p for p in band if p["expected_excess"] <= emin+TOL]
    smin = min(p["cvar90_mdd"] for p in band)
    band = [p for p in band if p["cvar90_mdd"] <= smin+TOL]
    selected = min(band, key=lambda p: PRODUCTS.index(p["portfolio"]))
    name = selected["portfolio"]
    return {**result, "assignment": name,
            "target_weights": next(p["weights"] for p in catalogue if p["name"] == name),
            "proposal_status": "conditional_proposal", "review_type": None,
            "reason_codes": ["H_L_training_candidate_and_ordinal_target_reserve_compatible"],
            "selected_training_scores": {k: selected[k] for k in
                ("mean_annual_log_growth", "expected_excess", "cvar90_mdd", "training_breach_count", "training_n")}}


def _sha_like(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _proposal_integrity_issues(proposal):
    """Recompute identities instead of trusting mutable copies of their hash labels."""
    issues = []
    try:
        profile = _profile(proposal["profile"])
        spec = proposal["service_policy"]
        canonical_policy = make_policy(spec["target_bill_floors"])
        if spec != canonical_policy:
            issues.append("proposal_policy_fingerprint_or_content_inconsistent")
        catalogue = _catalogue(proposal["catalogue"])
        if proposal["catalogue_fingerprint"] != _digest(catalogue):
            issues.append("proposal_catalogue_fingerprint_inconsistent")
        payload = proposal["training_evidence_payload"]
        training_issues, by_name = _candidate_issues(payload["training_candidates"], profile, catalogue)
        if training_issues:
            issues.append("proposal_training_evidence_invalid")
        else:
            canonical_payload = _training_payload(profile, by_name)
            if payload != canonical_payload or proposal["training_evidence_fingerprint"] != _digest(canonical_payload):
                issues.append("proposal_training_evidence_fingerprint_inconsistent")
        selected = next(p for p in catalogue if p["name"] == proposal["assignment"])
        if _weights(proposal["target_weights"], positive=True) != selected["weights"]:
            issues.append("proposal_target_not_the_assigned_catalogue_product")
        floor = canonical_policy["target_bill_floors"][profile["liquidity"]]
        if proposal["target_bill_floor"] != floor or selected["weights"][2] < floor-TOL:
            issues.append("proposal_target_reserve_inconsistent")
        if proposal["proposal_status"] != "conditional_proposal" or proposal["review_type"] is not None:
            issues.append("proposal_status_inconsistent")
    except (ValueError, TypeError, KeyError, StopIteration):
        issues.append("proposal_internal_structure_invalid")
    return issues


def assess_admission(proposal, evidence=None, *, expected_execution_policy_id):
    """Match a caller-verified whole-policy receipt, without claiming a probability guarantee.

    This pure function cannot authenticate files or audit truth. The caller must
    verify the cited manifest/audit hashes and the substantive validation scope.
    """
    if not isinstance(expected_execution_policy_id, str) or not expected_execution_policy_id:
        raise ValueError("expected execution policy version is required")
    base = {"conditional_assignment": proposal["assignment"], "effective_assignment": "Review",
            "service_admitted": False, "financial_outcome": None,
            "not_a_risk_guarantee": True, "receipt_authentication_is_callers_responsibility": True}
    if proposal["assignment"] == "Review":
        return {**base, "service_status": proposal["review_type"],
                "review_type": proposal["review_type"], "reason_codes": proposal["reason_codes"]}
    reasons = _proposal_integrity_issues(proposal)
    if reasons:
        return {**base, "service_status": "pending_joint_validation", "review_type": "model_review",
                "reason_codes": reasons}
    if evidence is not None and not isinstance(evidence, dict):
        reasons.append("joint_evidence_not_a_mapping")
    evidence = evidence if isinstance(evidence, dict) else {}
    p, spec = proposal["profile"], proposal["service_policy"]
    expected = {"service_policy_id": spec["policy_id"],
                "receipt_schema": JOINT_RECEIPT_SCHEMA,
                "service_policy_fingerprint": spec["fingerprint"],
                "catalogue_fingerprint": proposal["catalogue_fingerprint"],
                "training_evidence_fingerprint": proposal["training_evidence_fingerprint"],
                "profile_id": p["id"], "horizon": p["horizon"],
                "loss_tolerance": p["loss_tolerance"], "liquidity": p["liquidity"],
                "product": proposal["assignment"], "target_weights": proposal["target_weights"],
                "risk_metric_id": RISK_METRIC, "growth_metric_id": GROWTH_METRIC,
                "severity_metric_id": SEVERITY_METRIC,
                "excess_metric_id": EXCESS_METRIC, "proposal_sort_rule_id": PROPOSAL_SORT_RULE,
                "cost_rate": COST_RATE, "service_alpha": ALPHA,
                "execution_policy_id": expected_execution_policy_id,
                "validation_scope": "joint_service_and_execution_pipeline",
                "validation_unit": "whole_scenario_pipeline", "status": "passed",
                "training_candidate_only": False, "receipts_verified_by_caller": True}
    for key, value in expected.items():
        actual = evidence.get(key)
        if isinstance(value, bool):
            matched = actual is value
        elif isinstance(value, Real):
            matched = isinstance(actual, Real) and not isinstance(actual, bool) and actual == value
        else:
            matched = key in evidence and actual == value
        if not matched:
            reasons.append(f"joint_evidence_{key}_mismatch_or_missing")
    for key in ("source_manifest_sha256", "independent_audit_sha256"):
        if not _sha_like(evidence.get(key)):
            reasons.append(f"joint_evidence_{key}_missing_or_invalid")
    if reasons:
        return {**base, "service_status": "pending_joint_validation", "review_type": "model_review",
                "reason_codes": reasons}
    return {**base, "effective_assignment": proposal["assignment"], "service_admitted": True,
            "service_status": "joint_evidence_matched_conditional_service", "review_type": None,
            "reason_codes": ["version_matched_joint_receipt_not_a_risk_guarantee"]}


def classify_balanced_case(proposal, admission):
    """Do not confuse a non-Balanced product, a conditional case and effective admission."""
    label = proposal["assignment"]
    conditional = ("q1_review" if label == "Review" else
                   "conditional_balanced_case" if label == "Balanced" else "not_balanced_case")
    if not admission["service_admitted"]:
        effective = admission["review_type"]
    else:
        effective = "balanced_case" if admission["effective_assignment"] == "Balanced" else "not_balanced_case"
    return {"conditional_case": conditional, "effective_case": effective,
            "original_horizon": proposal["profile"]["horizon"], "financial_outcome": None}


def runtime_notices(profile, target_weights, actual_weights, *, floors=None,
                    known_need_changed=False, questionnaire_changed=False,
                    model_applicable=True, review_pending=False, history=None):
    """Notice drift, escalate changed facts; issue no investment/pending-hold action."""
    profile, target, actual = _profile(profile), _weights(target_weights, positive=True), _weights(actual_weights, positive=False)
    policy = make_policy(floors)
    for value in (known_need_changed, questionnaire_changed, model_applicable, review_pending):
        if not isinstance(value, bool):
            raise ValueError("runtime status flags must be boolean")
    if history is not None and not isinstance(history, dict):
        raise ValueError("history must be a mapping or None")
    floor = policy["target_bill_floors"][profile["liquidity"]]
    notices, reasons = [], []
    if actual[2] < target[2]-TOL:
        notices.append("bill_share_below_target")
    if actual[2] < floor-TOL:
        notices.append("bill_share_below_policy_target_floor")
    if target[2] < floor-TOL:
        reasons.append("current_target_incompatible_with_declared_policy")
    if known_need_changed:
        reasons.append("known_liquidity_need_changed")
    if questionnaire_changed:
        reasons.append("questionnaire_changed")
    if not model_applicable:
        reasons.append("model_not_applicable")
    if review_pending:
        reasons.append("review_already_pending_no_automatic_reentry")
    review = bool(reasons)
    return {"request_review": review, "review_reason_codes": reasons, "notices": notices,
            "target_bill_floor": floor, "target_bill_weight": target[2], "actual_bill_weight": actual[2],
            "observed_policy_floor_gap": max(floor-actual[2], 0.0),
            "freeze_unapproved_new_targets": review, "decision_action": None,
            "existing_controller_disposition": ("caller_manages_authorized_pending_policy" if review
                                                else "unchanged_by_drift_notice"),
            "emergency_liability_action": None, "history": deepcopy(history),
            "history_reset": False, "live_cash_floor_guaranteed": False}


def coverage_summary(*, original_n, case_n, automated_n, unit):
    """Descriptive counts with an explicit unit; never infer customer-population coverage."""
    original_n = _count(original_n, "original n")
    case_n = _count(case_n, "case n")
    automated_n = _count(automated_n, "automated n")
    if unit not in COVERAGE_UNITS or not original_n >= case_n >= automated_n:
        raise ValueError("invalid coverage unit or nested counts")
    return {"unit": unit, "original_n": original_n, "case_n": case_n,
            "automated_n": automated_n, "outside_case_n": original_n-case_n,
            "case_review_n": case_n-automated_n,
            "auto_over_original": automated_n/original_n if original_n else None,
            "auto_given_case": automated_n/case_n if case_n else None,
            "customer_population_coverage_estimated": False,
            "financial_outcome": None}
