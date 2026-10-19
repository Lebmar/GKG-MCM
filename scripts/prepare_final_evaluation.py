"""Locked evaluation adapter with separate development and final releases.

Default description performs no market parsing. Development reproduction
requires the frozen preparation protocol and loads only IDs1--120 with
allow_final=False. Final parsing requires a NEW, explicit engineering release
binding this unchanged source, object specification and a passed development
gate/audit. Preparation never creates or enables that release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
PROTOCOL = Path("docs/final_evaluation_protocol.json")
RELEASE = Path("docs/final_evaluation_release.json")
DEV_OUT = Path("results/model_review/final_evaluation_development_gate_v1")
FINAL_OUT = Path("results/final_locked_evaluation_v1")
JOINT = Path("results/model_review/joint_liquidity_refit_v1")
ASSETS = ("equity", "bond", "bill")
METRICS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
ACCOUNTS = ("nominal_fees", "nominal_moved", "trade_fraction", "target_execution_fraction", "terminal_equity_weight", "terminal_bond_weight", "terminal_bill_weight")
FINANCIAL = (*METRICS, *ACCOUNTS, "breach", "excess", "first_crossing")
SUMMARY_FINANCIAL = ("breach_count", "breach_fraction", "mean_log_growth", "mean_cagr", "median_cagr", "p10_cagr",
    "mean_mdd", "median_mdd", "p90_mdd", "cvar90_mdd", "mean_volatility", "mean_nominal_fees", "mean_trade_fraction")
GROUP_KEYS = ("object_id", "question", "profile", "horizon", "loss_tolerance", "financial_scope")
TOL = 1e-12
REQUIRED_SOURCES = {
    "scripts/prepare_final_evaluation.py", "tests/test_prepare_final_evaluation.py",
    "src/safeinvest/core.py", "src/safeinvest/data.py",
    "research/safeinvest_research/metrics.py", "research/experimental_execution/review_pending.py",
    "research/experimental_execution/fan_rollout.py", "research/experimental_execution/risk_priced_fan_rollout.py",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_fingerprint(ids, returns, inflation):
    digest = hashlib.sha256()
    for value in (ids, returns, inflation):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def object_specification(registry, profiles):
    """Pure locked objects; suitability and finance availability are separate."""
    current = registry["current_C_execution"]
    if (current["final_status"] != "execution_review" or current["financial_outputs"] is not None
            or current["H"] != 30 or current["L"] != .4 or current["original_target"] != [.45, .05, .5]
            or current["original_selected_eta"] != .03 or registry["safeguards"]["automatic_target_updates"] is not False):
        raise ValueError("Unchanged rejected current C contract required")
    static = registry["current_static_full120"]
    if static["profile_Q1_assignments"] != dict(A="Review", B="Conservative", C="Balanced", D="Review", E="Growth"):
        raise ValueError("Frozen Q1 assignments changed")
    expected_profiles = [("A", 5, .1, "high"), ("B", 15, .25, "medium"), ("C", 30, .4, "low"), ("D", 25, .15, "medium"), ("E", 6, .35, "high")]
    if [(p["name"], p["horizon"], p["loss_tolerance"], p["liquidity"]) for p in profiles] != expected_profiles:
        raise ValueError("All original profile own H/L/liquidity values required")
    common = dict(question="Q2", profile="C", horizon=30, loss_tolerance=.4, liquidity="low", target_updates=False,
                  q1_assignment="Balanced", case_status="balanced_case")
    q2 = [dict(common, object_id="guarded_C_service", financial_scope="none_preinvestment_review",
               target=[.45, .05, .5], execution_rule="no_account_started", service_admitted=False,
               financial_available=False, execution_status="execution_review", annual_eta=None),
          dict(common, object_id="unapproved_fixed_shadow", financial_scope="rejected_candidate_shadow",
               target=[.45, .05, .5], execution_rule="fixed_risk_priced_fan_then_absorbing_pending", service_admitted=False,
               financial_available=True, execution_status="research_shadow_not_admitted", annual_eta=.03),
          dict(common, object_id="same_target_B", financial_scope="hypothetical_reference",
               target=[.45, .05, .5], execution_rule="annual_then_absorbing_pending", service_admitted=None,
               financial_available=True, execution_status="hypothetical_reference", annual_eta=None),
          dict(common, object_id="exact_thirds_annual", financial_scope="hypothetical_reference",
               target=[1/3, 1/3, 1/3], execution_rule="annual_including_after_crossing", service_admitted=None,
               financial_available=True, execution_status="hypothetical_reference", annual_eta=None)]
    q1 = []
    for profile in profiles:
        name = profile["name"]; assignment = static["profile_Q1_assignments"][name]
        available = assignment != "Review"
        q1.append(dict(object_id="Q1_" + name + "_static_product", question="Q1", profile=name,
            horizon=profile["horizon"], loss_tolerance=profile["loss_tolerance"], liquidity=profile["liquidity"],
            assignment=assignment, target=static["portfolios"][assignment] if available else None,
            q1_suitable=available, financial_available=available, service_admitted=None,
            execution_status="q1_product_reference" if available else "q1_review",
            case_status="suitable_product_reference" if available else "q1_review",
            financial_scope="suitability_product_annual_reference" if available else "none_Q1_review",
            execution_rule="annual_including_after_crossing" if available else "no_account_started",
            current_Q2_C_execution="execution_review" if name == "C" else "not_applicable_to_C_only_execution_case",
            target_updates=False, annual_eta=None))
    return dict(asset_order=list(ASSETS), cost_rate=.0025, initial_nominal_wealth=100.,
        initial_asset="bill", Q2_objects=q2, Q1_objects=q1,
        forecast_training_ids=list(range(1, 121)), forecast="own chronological joint empirical suffix, current action then each own B/pending branch",
        no_final_information_for_selection=True, no_reapproval_after_evaluation=True,
        severe_example="development outer_r5_112, c502_refit, original112; fixed before final evaluation",
        Q3="Reuse independently verified development known-cost bridge; no new fee search or final stress selection")


def validate_definition(p, registry, profiles):
    if p.get("version") != "locked-final-evaluation-preparation-v1" or p.get("final_execution_ready") is not False:
        raise ValueError("The immutable preparation protocol never enables final execution")
    if (p.get("development_ids") != list(range(1, 121)) or p.get("final_ids") != list(range(121, 181))
            or p.get("development_allow_final") is not False or p.get("final_release_path") != str(RELEASE)
            or p.get("development_output_directory") != str(DEV_OUT) or p.get("final_output_directory") != str(FINAL_OUT)):
        raise ValueError("Fixed development/final identity or output contract differs")
    spec = object_specification(registry, profiles)
    if p.get("object_specification") != spec or p.get("object_specification_sha256") != canonical_sha(spec):
        raise ValueError("Locked object specification differs from unchanged registry")
    fields = p["exact_reproduction_fields"]
    if ([len(fields[name]) for name in ("unapproved_fixed_shadow", "same_target_B", "exact_thirds_annual")] != [48, 36, 36]
            or any(len(set(values)) != len(values) for values in fields.values())):
        raise ValueError("Complete distinct48/36/36field inventories required")
    return spec


def freeze_snapshot(p, root=ROOT):
    """Read bytes only; no CSV values are parsed by this hash closure."""
    root = Path(root); answer = {}
    for key in ("execution_sources_sha256", "frozen_input_sha256"):
        mapping = p[key]
        if not isinstance(mapping, dict) or not mapping:
            raise RuntimeError("Preparation source/input hashes are not frozen")
        actual = {}
        for name, expected in mapping.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Repository-relative SHA256 closure required")
            actual[name] = sha(root / name)
        if actual != mapping:
            raise RuntimeError("Frozen evaluation source or input changed")
        answer[key] = actual
    if not REQUIRED_SOURCES <= set(answer["execution_sources_sha256"]):
        raise RuntimeError("The complete adapter/test/production-kernel source closure is required")
    return answer


def authorize_development(p):
    if p.get("development_execution_ready") is not True or p.get("final_execution_ready") is not False:
        raise RuntimeError("Freeze preparation before the development-only reproduction gate; final remains disabled")


def authorize_final(p, release, development_manifest, development_audit, *, preparation_sha256, development_manifest_sha256):
    """Pure release check; a CLI flag and a passed dev gate cannot enable it."""
    authorize_development(p)
    if (not isinstance(release, dict) or release.get("version") != "locked-final-evaluation-release-v1"
            or release.get("final_execution_ready") is not True):
        raise RuntimeError("A separate explicit frozen engineering final release is required")
    if (release.get("preparation_protocol_sha256") != preparation_sha256
            or release.get("execution_sources_sha256") != p["execution_sources_sha256"]
            or release.get("object_specification_sha256") != p["object_specification_sha256"]
            or release.get("development_gate_manifest_sha256") != development_manifest_sha256
            or release.get("training_ids") != list(range(1, 121)) or release.get("evaluation_ids") != list(range(121, 181))
            or release.get("output_directory") != str(FINAL_OUT)):
        raise RuntimeError("Final release must bind the exact preparation, source, objects, dev gate and fixed IDs")
    if (development_manifest.get("status") != "passed" or development_manifest.get("final_scenarios_loaded") is not False
            or development_manifest.get("preparation_protocol_sha256") != preparation_sha256
            or development_manifest.get("object_specification_sha256") != p["object_specification_sha256"]
            or development_manifest.get("execution_sources_sha256") != p["execution_sources_sha256"]
            or development_audit.get("status") != "passed"
            or development_audit.get("parent_manifest_sha256") != development_manifest_sha256):
        raise RuntimeError("Passed, independently bound unchanged development gate required")


def exact_array(observed, expected, field):
    a, b = np.asarray(observed), np.asarray(expected)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("Exact array dtype/shape changed: " + field)
    same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
    if not same:
        raise ValueError("Exact array values/NULLmask changed: " + field)
    return dict(field=field, dtype=str(a.dtype), shape=list(a.shape), scalar_values=int(a.size), exact=True)


def verify_development_evidence(root, manifest, audit):
    """Current bytes of the already passed gate, not merely its status text."""
    root = Path(root)
    for start, end in (("sources", "sources_at_end"), ("input_sha256", "input_sha256_at_end")):
        if manifest.get(start) != manifest.get(end) or not manifest.get(start):
            raise RuntimeError("Development gate source/input snapshots differ")
        for name, expected in manifest[start].items():
            if sha(root / name) != expected:
                raise RuntimeError("Current development-gate dependency differs")
    for name, expected in manifest["outputs_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(root / DEV_OUT / name) != expected:
            raise RuntimeError("Current development-gate output differs")
    before, after = audit["sources_and_artifacts_at_start"], audit["sources_and_artifacts_at_end"]
    if not before or before != after:
        raise RuntimeError("Bound development audit snapshots differ")
    for name, expected in before.items():
        if sha(root / name) != expected:
            raise RuntimeError("Current independently audited development artifact differs")


def extract_price(simulation):
    n, prices = np.asarray(simulation["real_wealth"]).shape[:2]
    if prices != 1:
        raise ValueError("Only the already fixed one-price shadow may be evaluated")
    common = {"cpi", "targets", "loss_tolerances", "annual_prices"}; out = {}
    for name, value in simulation.items():
        if not isinstance(value, np.ndarray):
            continue
        if name in common:
            out[name] = value.copy()
        elif value.ndim >= 2 and value.shape[:2] == (n, 1):
            out[name] = value[:, 0].copy()
        else:
            raise ValueError("Unexpected frozen price-axis shape: " + name)
    return out


def squeeze_reference(arm, target):
    out = {}
    for name, value in arm.items():
        if not isinstance(value, np.ndarray):
            continue
        if name == "cpi":
            out[name] = value.copy()
        elif name in ("targets", "loss_tolerances"):
            continue
        elif value.ndim >= 2 and value.shape[1] == 1:
            out[name] = value[:, 0].copy()
        else:
            raise ValueError("Unexpected singleton reference target axis: " + name)
    out["targets"] = np.asarray(target, float).copy()
    return out


def simulate_locked(actual_returns, actual_inflation, training_returns, training_inflation, spec):
    """All accounts are fixed before evaluation; no fitting/selection imported."""
    from experimental_execution.risk_priced_fan_rollout import simulate_prices
    from experimental_execution.review_pending import simulate_paired
    arms = {}
    target = spec["Q2_objects"][1]["target"]
    arms["unapproved_fixed_shadow"] = extract_price(simulate_prices(actual_returns, actual_inflation, training_returns,
        training_inflation, target, np.array([.03]), L=.4, cost_rate=.0025, horizon=30))
    arms["same_target_B"] = squeeze_reference(simulate_paired(actual_returns, actual_inflation, target, .4,
        horizon=30, cost_rate=.0025)["pending_hold"], target)
    thirds = [1/3, 1/3, 1/3]
    arms["exact_thirds_annual"] = squeeze_reference(simulate_paired(actual_returns, actual_inflation, thirds, .4,
        horizon=30, cost_rate=.0025)["annual"], thirds)
    for obj in spec["Q1_objects"]:
        if obj["financial_available"]:
            arms[obj["object_id"]] = squeeze_reference(simulate_paired(actual_returns, actual_inflation, obj["target"],
                obj["loss_tolerance"], horizon=obj["horizon"], cost_rate=.0025)["annual"], obj["target"])
    return arms


def path_rows(arm, ids, obj, scope):
    base = {key: obj[key] for key in GROUP_KEYS}
    base.update(execution_status=obj["execution_status"], service_admitted=obj["service_admitted"],
                financial_available=obj["financial_available"], scope=scope,
                q1_suitable=obj.get("q1_suitable"), q1_assignment=obj.get("q1_assignment", obj.get("assignment")),
                case_status=obj["case_status"], liquidity=obj["liquidity"])
    if not obj["financial_available"]:
        if arm is not None:
            raise ValueError("No physical account or substituted financials for a rejected service")
        return [{**base, "scenario_id": int(sid), **dict.fromkeys(FINANCIAL)} for sid in ids]
    if arm is None:
        raise ValueError("A research/product-reference account is missing")
    from safeinvest_research.metrics import path_metrics
    values = path_metrics(arm["real_wealth"])
    if np.asarray(arm["real_wealth"]).shape != (len(ids), obj["horizon"] + 1):
        raise ValueError("Each object must retain its own original horizon")
    values.update(nominal_fees=arm["fees"].sum(axis=1), nominal_moved=arm["moved"].sum(axis=1),
        trade_fraction=arm["trades"].mean(axis=1), target_execution_fraction=arm["target_execution"].mean(axis=1))
    values.update({"terminal_" + asset + "_weight": arm["weights"][:, -1, j] for j, asset in enumerate(ASSETS)})
    rows = []
    for i, sid in enumerate(ids):
        crossing = int(arm["first_crossing"][i]); breach = bool(arm["ever_breached"][i, -1])
        if breach != (float(values["mdd"][i]) - obj["loss_tolerance"] > TOL):
            raise ValueError("Metrics and original strict breach history differ")
        rows.append({**base, "scenario_id": int(sid), **{name: float(values[name][i]) for name in (*METRICS, *ACCOUNTS)},
                     "breach": breach, "excess": float(values["mdd"][i]) - obj["loss_tolerance"] if breach else 0.,
                     "first_crossing": None if crossing < 0 else crossing})
    return rows


def summarize(rows):
    """One object population only; no pooling of quantiles or NULL fill."""
    if not rows:
        raise ValueError("Original population must be preserved even when every financial is NULL")
    if len({r["scenario_id"] for r in rows}) != len(rows) or any(any(r[k] != rows[0][k] for k in GROUP_KEYS) for r in rows):
        raise ValueError("Unique original IDs in one object population required")
    available = [r for r in rows if r["financial_available"]]
    for r in rows:
        if not r["financial_available"] and any(r[name] is not None for name in FINANCIAL):
            raise ValueError("Unserved finance must remain NULL")
    first = rows[0]; admitted = first["service_admitted"]
    result = {key: first[key] for key in GROUP_KEYS}
    result.update(scope=first["scope"], execution_status=first["execution_status"], original_n=len(rows),
        case_status=first["case_status"], q1_assignment=first["q1_assignment"], liquidity=first["liquidity"],
        case_n=len(rows) if first["case_status"] == "balanced_case" else None,
        financial_n=len(available), service_automatic_n=None if admitted is None else sum(r["service_admitted"] is True for r in rows),
        preinvestment_review_n=len(rows) if first["execution_status"] == "execution_review" else None,
        execution_review_n=len(rows) if first["execution_status"] == "execution_review" else None,
        q1_suitable_n=sum(r["q1_suitable"] is True for r in rows) if first["question"] == "Q1" else None,
        q1_review_n=sum(r["q1_suitable"] is False for r in rows) if first["question"] == "Q1" else None,
        **dict.fromkeys(SUMMARY_FINANCIAL))
    if not available:
        return result
    def array(name):
        values = np.asarray([r[name] for r in available], float)
        if not np.isfinite(values).all():
            raise ValueError("Finite observed research financials required")
        return values
    cagr, mdd = array("cagr"), array("mdd")
    tail = np.sort(mdd)[::-1]; mass = .1 * len(tail); whole = int(np.floor(mass))
    cvar = (float(tail[:whole].sum()) + (0 if whole == len(tail) else (mass - whole) * float(tail[whole]))) / mass
    k = sum(bool(r["breach"]) for r in available)
    result.update(breach_count=k, breach_fraction=k / len(available), mean_log_growth=float(array("log_growth").mean()),
        mean_cagr=float(cagr.mean()), median_cagr=float(np.median(cagr)), p10_cagr=float(np.quantile(cagr, .1, method="inverted_cdf")),
        mean_mdd=float(mdd.mean()), median_mdd=float(np.median(mdd)), p90_mdd=float(np.quantile(mdd, .9, method="inverted_cdf")),
        cvar90_mdd=cvar, mean_volatility=float(array("volatility").mean()), mean_nominal_fees=float(array("nominal_fees").mean()),
        mean_trade_fraction=float(array("trade_fraction").mean()))
    return result


def evaluation_rows(arms, ids, spec, scope):
    q2, q1, summaries = [], [], []
    for question, objects, output in (("Q2", spec["Q2_objects"], q2), ("Q1", spec["Q1_objects"], q1)):
        for obj in objects:
            rows = path_rows(arms.get(obj["object_id"]), ids, obj, scope)
            output.extend(rows); summaries.append(summarize(rows))
    if len(q2) != 4 * len(ids) or len(q1) != 5 * len(ids) or len(summaries) != 9:
        raise ValueError("All fixed service, shadow, reference and product objects required")
    return q2, q1, summaries


def verify_development_arrays(arms, archive, p):
    groups = dict(unapproved_fixed_shadow="c0__joint_tie_set_refit_learner", same_target_B="c0__annual_then_pending", exact_thirds_annual="c0__equal_thirds_annual")
    results = {}
    for obj, prefix in groups.items():
        fields = p["exact_reproduction_fields"][obj]
        if sorted(arms[obj]) != sorted(fields):
            raise ValueError("Every generated physical/Q/meta field must match the exact frozen inventory")
        expected_keys = {key[len(prefix + "__"):]: key for key in archive.files if key.startswith(prefix + "__")}
        if sorted(expected_keys) != sorted(fields):
            raise ValueError("Frozen original trace inventory differs")
        results[obj] = [exact_array(arms[obj][field], archive[expected_keys[field]], obj + "/" + field) for field in fields]
    return results


def verify_Q1_reference(rows, original):
    frame = pd.DataFrame(rows)
    saved = original.loc[original.context.eq("full_development") & original.view.eq("effective")]
    if len(frame) != 600 or len(saved) != 600 or frame.duplicated(["profile", "scenario_id"]).any():
        raise ValueError("All5profiles times120developmentIDs are required")
    errors = {}; matched = nulls = 0
    for obj in frame.object_id.unique():
        block = frame.loc[frame.object_id.eq(obj)].sort_values("scenario_id")
        ref = saved.loc[saved.client.eq(block.profile.iloc[0])].sort_values("scenario_id")
        if block.scenario_id.tolist() != list(range(1, 121)) or ref.scenario_id.tolist() != list(range(1, 121)):
            raise ValueError("Q1 original development identities differ")
        if not np.array_equal(block.horizon, ref.horizon) or not np.array_equal(block.loss_tolerance, ref.loss_tolerance):
            raise ValueError("Q1 own H/L differs")
        if not np.array_equal(block.financial_available, ref.assigned) or not np.array_equal(block.q1_assignment, ref.assignment):
            raise ValueError("Q1 suitability/reference availability differs")
        if not block.financial_available.any():
            if not ref[[*METRICS, "breach", "excess"]].isna().all().all():
                raise ValueError("Q1 original Review financials must be NULL")
            nulls += len(block); continue
        for name in (*METRICS, "excess"):
            a, b = block[name].to_numpy(float), ref[name].to_numpy(float)
            if not np.allclose(a, b, atol=2e-10, rtol=5e-13):
                raise ValueError("Q1 annual product reference differs: " + name)
            errors[name] = max(errors.get(name, 0.), float(np.max(np.abs(a - b))))
        if not np.array_equal(block.breach, ref.breach):
            raise ValueError("Q1 annual breach identities differ")
        matched += len(block)
    return dict(available_product_paths=matched, preserved_Review_paths=nulls, decimal_CSV_maximum_absolute_errors=errors,
                tolerance=dict(atol=2e-10, rtol=5e-13), interpretation="Q1 static annual product references, not Q2 execution admission")


def preparation(root=ROOT):
    root = Path(root); p = readj(root / PROTOCOL)
    registry = readj(root / "docs/main_model_registry.json")
    profiles = readj(root / "docs/positive_asset_specification_protocol.json")["clients"]
    return p, validate_definition(p, registry, profiles)


def update_attempt(out, state, phase, **changes):
    state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **changes)
    writej(Path(out) / "manifest.json", state)


def run_archived_attempt(out, initial_state, work):
    """Create before parsing; retain failed/incomplete evidence, never retry."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(initial_state, status="incomplete", started_utc=datetime.now(timezone.utc).isoformat(),
                 final_value_access_started=False, final_values_may_have_been_parsed=False,
                 final_scenarios_loaded=False, completed_final_array_load=False)
    update_attempt(out, state, "all_guards_passed_no_market_values_read")
    try:
        result = work(state)
        if readj(out / "manifest.json").get("status") not in ("passed", "complete"):
            raise RuntimeError("Attempt returned without a complete result manifest")
        return result
    except BaseException as exc:
        errorlog = out / "error.log"
        errorlog.write_text(traceback.format_exc())
        state.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(),
                     exception_type=type(exc).__name__, exception_message=str(exc),
                     error_log_sha256=sha(errorlog), rerun_allowed=False,
                     interpretation="Preserved failed single attempt. No automatic retry, deletion, reselection or admission change.")
        writej(out / "failure_receipt.json", state)
        state["failure_receipt_sha256"] = sha(out / "failure_receipt.json")
        writej(out / "manifest.json", state)
        raise


def development_gate(root=ROOT):
    root = Path(root); p, spec = preparation(root); authorize_development(p)
    out = root / DEV_OUT
    if out.exists():
        raise RuntimeError("Do not overwrite a development reproduction archive")
    before = freeze_snapshot(p, root); protocol_digest = sha(root / PROTOCOL); started = time.perf_counter()
    initial = dict(kind="development_reproduction", preparation_protocol_sha256=protocol_digest,
        object_specification_sha256=p["object_specification_sha256"], execution_sources_sha256=before["execution_sources_sha256"],
        requested_ids=list(range(1, 121)), allow_final=False, final_release_created=False, final_execution_enabled=False)
    return run_archived_attempt(out, initial, lambda state: _development_attempt(root, p, spec, before, protocol_digest, started, out, state))


def _development_attempt(root, p, spec, before, protocol_digest, started, out, state):
    from safeinvest.data import load_paths
    update_attempt(out, state, "loading_development_only_1_to_120")
    returns, inflation, ids = load_paths(root / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if ids.tolist() != list(range(1, 121)):
        raise ValueError("Guarded complete development order required")
    fingerprint = array_fingerprint(ids, returns, inflation)
    parent_fingerprint = readj(root / JOINT / "manifest.json")["development_arrays_sha256"]
    if fingerprint != parent_fingerprint:
        raise ValueError("Development arrays differ from the original frozen parent lineage")
    groups = readj(root / JOINT / "trace_groups.json")["actual"]
    for name in ("c0__joint_tie_set_refit_learner", "c0__annual_then_pending", "c0__equal_thirds_annual"):
        if groups[name]["scenario_ids"] != ids.tolist():
            raise ValueError("Original c0 account identity order differs")
    update_attempt(out, state, "reproducing_fixed_accounts_on_development")
    arms = simulate_locked(returns, inflation, returns, inflation, spec)
    with np.load(root / JOINT / "actual_traces.npz", allow_pickle=False) as original:
        arrays = verify_development_arrays(arms, original, p)
    q2, q1, summaries = evaluation_rows(arms, ids, spec, "development_resubstitution_reproduction_only")
    q1_check = verify_Q1_reference(q1, pd.read_csv(root / JOINT / "q1_effective_paths.csv"))
    after = freeze_snapshot(p, root)
    if before != after or sha(root / PROTOCOL) != protocol_digest:
        raise RuntimeError("Frozen preparation changed during development gate")
    update_attempt(out, state, "writing_checked_development_outputs")
    writej(out / "exact_array_reproduction.json", arrays); writej(out / "Q1_reference_reproduction.json", q1_check)
    for name, rows in (("Q2_paths", q2), ("Q1_paths", q1), ("summary", summaries)):
        pd.DataFrame(rows).to_csv(out / (name + ".csv.gz"), index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    writej(out / "object_specification.json", spec)
    save_trace_archive(out / "development_traces.npz", arms, ids)
    outputs = {path.name: sha(path) for path in sorted(out.iterdir()) if path.name != "manifest.json"}
    writej(out / "manifest.json", dict(status="passed", completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started, preparation_protocol_sha256=protocol_digest,
        object_specification_sha256=p["object_specification_sha256"], execution_sources_sha256=before["execution_sources_sha256"],
        sources=before["execution_sources_sha256"], sources_at_end=after["execution_sources_sha256"],
        input_sha256=before["frozen_input_sha256"], input_sha256_at_end=after["frozen_input_sha256"],
        outputs_sha256=outputs, scenario_ids_loaded=ids.tolist(), final_scenarios_loaded=False,
        exact_array_fields=sum(map(len, arrays.values())), Q1_available_reference_paths=q1_check["available_product_paths"],
        development_arrays_sha256=fingerprint, frozen_parent_development_arrays_sha256=parent_fingerprint,
        Q1_Review_paths=q1_check["preserved_Review_paths"], final_release_created=False,
        final_execution_enabled=False, new_fits=0, selection_changed=False, admission_changed=False,
        final_value_access_started=False, final_values_may_have_been_parsed=False,
        interpretation="Exact development adapter reproduction, not a final evaluation or independent risk validation"))
    print(json.dumps(dict(status="passed", manifest_sha256=sha(out / "manifest.json"), exact_array_fields=sum(map(len, arrays.values())), final_scenarios_loaded=False)))


def save_trace_archive(path, arms, ids):
    """Persist only actually simulated reference/shadow accounts, never Review."""
    if "guarded_C_service" in arms or "Q1_A_static_product" in arms or "Q1_D_static_product" in arms:
        raise ValueError("Review accounts cannot have simulated physical trees")
    archive = {}
    for name, arm in arms.items():
        for field, value in arm.items():
            if isinstance(value, np.ndarray):
                archive[name + "__" + field] = value
        archive[name + "__scenario_ids"] = np.asarray(ids).copy()
    np.savez_compressed(path, **archive)


def final_evaluation(root=ROOT):
    """Dormant until a separate release plus bound dev audit explicitly exist."""
    root = Path(root); p, spec = preparation(root)
    release_path = root / RELEASE
    if not release_path.is_file():
        raise RuntimeError("No separate final release exists; final market parsing is disabled")
    release = readj(release_path); release_digest = sha(release_path)
    devpath = root / DEV_OUT / "manifest.json"; auditpath = root / DEV_OUT / "independent_audit.json"
    if not devpath.is_file() or not auditpath.is_file():
        raise RuntimeError("Development reproduction and its bound independent audit are required first")
    dev, audit = readj(devpath), readj(auditpath)
    authorize_final(p, release, dev, audit, preparation_sha256=sha(root / PROTOCOL), development_manifest_sha256=sha(devpath))
    if release.get("development_gate_audit_sha256") != sha(auditpath):
        raise RuntimeError("Final release must bind the exact independent development audit")
    verify_development_evidence(root, dev, audit)
    before = freeze_snapshot(p, root); out = root / FINAL_OUT
    if out.exists():
        raise RuntimeError("Only one non-overwriting fixed final evaluation is allowed")
    initial = dict(kind="locked_final_evaluation", release_sha256=release_digest,
        preparation_protocol_sha256=release["preparation_protocol_sha256"], object_specification_sha256=p["object_specification_sha256"],
        execution_sources_sha256=before["execution_sources_sha256"], development_gate_manifest_sha256=sha(devpath),
        development_gate_audit_sha256=sha(auditpath), requested_training_ids=list(range(1, 121)), requested_evaluation_ids=list(range(121, 181)))
    return run_archived_attempt(out, initial, lambda state: _final_attempt(root, p, spec, release, release_path,
        release_digest, devpath, auditpath, before, out, state))


def _final_attempt(root, p, spec, release, release_path, release_digest, devpath, auditpath, before, out, state):
    # Final values can be parsed ONLY below every independent release gate.
    from safeinvest.data import load_paths
    update_attempt(out, state, "loading_frozen_development_forecast_fan")
    train, train_inflation, train_ids = load_paths(root / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    train_fingerprint = array_fingerprint(train_ids, train, train_inflation)
    if train_fingerprint != readj(root / JOINT / "manifest.json")["development_arrays_sha256"]:
        raise ValueError("The frozen original development forecast fan changed")
    update_attempt(out, state, "parsing_final_121_to_180", final_value_access_started=True,
                   final_values_may_have_been_parsed=True)
    actual, actual_inflation, ids = load_paths(root / "data/SafeInvest_Synthetic_Data.csv", range(121, 181), allow_final=True)
    update_attempt(out, state, "complete_final_arrays_loaded", final_scenarios_loaded=True, completed_final_array_load=True)
    if train_ids.tolist() != p["development_ids"] or ids.tolist() != p["final_ids"]:
        raise ValueError("Frozen original training and final identities differ")
    update_attempt(out, state, "evaluating_only_frozen_objects")
    arms = simulate_locked(actual, actual_inflation, train, train_inflation, spec)
    q2, q1, summaries = evaluation_rows(arms, ids, spec, "locked_final_evaluation_previously_exposed_not_blind")
    after = freeze_snapshot(p, root)
    if (before != after or sha(release_path) != release_digest or sha(root / PROTOCOL) != release["preparation_protocol_sha256"]
            or sha(devpath) != release["development_gate_manifest_sha256"] or sha(auditpath) != release["development_gate_audit_sha256"]):
        raise RuntimeError("Release/preparation/source/input changed during final evaluation")
    update_attempt(out, state, "writing_locked_final_outputs")
    for name, rows in (("Q2_paths", q2), ("Q1_paths", q1), ("summary", summaries)):
        pd.DataFrame(rows).to_csv(out / (name + ".csv.gz"), index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    save_trace_archive(out / "evaluation_traces.npz", arms, ids)
    writej(out / "object_specification.json", spec)
    writej(out / "existing_development_context.json", p["existing_development_context"])
    outputs = {path.name: sha(path) for path in sorted(out.iterdir()) if path.name != "manifest.json"}
    writej(out / "manifest.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        release_sha256=release_digest, preparation_protocol_sha256=release["preparation_protocol_sha256"],
        object_specification_sha256=p["object_specification_sha256"], development_gate_manifest_sha256=sha(devpath),
        development_gate_audit_sha256=sha(auditpath), sources=before["execution_sources_sha256"], sources_at_end=after["execution_sources_sha256"],
        input_sha256=before["frozen_input_sha256"], input_sha256_at_end=after["frozen_input_sha256"], outputs_sha256=outputs,
        training_ids_loaded=train_ids.tolist(), evaluation_ids_loaded=ids.tolist(), final_scenarios_loaded=True,
        training_development_arrays_sha256=train_fingerprint, evaluation_arrays_sha256=array_fingerprint(ids, actual, actual_inflation),
        final_value_access_started=True, final_values_may_have_been_parsed=True, completed_final_array_load=True,
        new_fits=0, selection_changed=False, admission_changed=False, current_C_service="execution_review", current_C_service_financials=None,
        Q2_path_rows=len(q2), Q1_path_rows=len(q1), summary_rows=len(summaries),
        interpretation="One locked evaluation of unchanged decisions and separately labelled shadow/references; oldV1 exposure disclosed, not newly blind; no paper or submission output"))
    print(json.dumps(dict(status="complete", manifest_sha256=sha(out / "manifest.json"), Q2_path_rows=len(q2), Q1_path_rows=len(q1), summary_rows=len(summaries))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--describe", action="store_true")
    group.add_argument("--validate-development", action="store_true")
    group.add_argument("--execute-final", action="store_true")
    args = parser.parse_args()
    if args.validate_development:
        development_gate()
    elif args.execute_final:
        final_evaluation()
    else:
        protocol, spec = preparation()
        print(json.dumps(dict(development_execution_ready=protocol["development_execution_ready"],
            preparation_final_execution_ready=protocol["final_execution_ready"], separate_final_release_exists=(ROOT / RELEASE).is_file(),
            no_market_parsing=True, object_specification=spec), ensure_ascii=False, indent=2))
