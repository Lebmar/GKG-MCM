"""Secondary finite-family static-risk evidence; development paths 1--120 only.

The main empirical research remains unchanged. This diagnostic screens the
same fixed targets using docs/simultaneous_static_risk_protocol.json, then
reuses the declared common-menu selection. Run only after source changes to
the main research procedure have settled. Importing this script loads no data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
import run_risk_menu_research as primary
from safeinvest_research.evidence import simultaneous_upper, supported_count_limit
from safeinvest_research.metrics import NUMERIC_TOL, SUMMARY_METADATA
from safeinvest_research.selection import fit_menu

PROTOCOL = ROOT / "docs/simultaneous_static_risk_protocol.json"
OUT = ROOT / "results/model_review/simultaneous_static_risk_v1"


def source_manifest():
    paths = sorted((ROOT / "research/safeinvest_research").glob("*.py"))
    paths += [Path(__file__).resolve(), Path(primary.__file__).resolve(),
              ROOT / "src/safeinvest/core.py", ROOT / "src/safeinvest/data.py",
              primary.PROTOCOL, PROTOCOL]
    return {str(p.relative_to(ROOT)): primary.sha(p) for p in paths}


def validate_scope(protocol, evidence_protocol, arrays):
    """Do not extend the 1160-event assertion to a different policy family."""
    family = protocol["candidate_family"]
    expected = primary.simplex_grid(.05)
    checks = [
        evidence_protocol["family_events"] == 1160,
        evidence_protocol["candidate_count"] == 232,
        evidence_protocol["client_events_per_candidate"] == 5,
        evidence_protocol["delta"] == .05,
        evidence_protocol["n_full"] == 120,
        evidence_protocol["n_fold_train"] == 96,
        family["weight_grid_step"] == .05,
        family["cost_rate"] == .0025,
        family["include_exact_equal_thirds"] is True,
        family["minimum_each_asset"] == 0,
        protocol["allowed_scenarios"] == list(range(1, 121)),
        len(protocol["clients"]) == 5,
        protocol["alpha_cases"] == evidence_protocol["alpha_cases"],
        list(protocol["liquidity_cases"]) == evidence_protocol["liquidity_cases"],
        np.array_equal(arrays["scenario_ids"], np.arange(1, 121)),
        np.shape(arrays["weights"]) == (232, 3),
    ]
    if not all(checks):
        raise ValueError("The declared 232-target/5-event/.0025 development family changed")
    if not np.allclose(arrays["weights"], expected, rtol=0, atol=1e-14):
        raise ValueError("Cache weights do not match the declared fixed candidate family")
    expected_profiles = [("A", 5, .10), ("B", 15, .25), ("C", 30, .40),
                         ("D", 25, .15), ("E", 6, .35)]
    actual = [(p["name"], p["horizon"], p["loss_tolerance"]) for p in protocol["clients"]]
    if actual != expected_profiles:
        raise ValueError("The fixed client horizon/loss events changed")
    primary.validate_folds(protocol)


def family_evidence(metrics, weights, profiles, indices, *, family_events=1160, delta=.05):
    """Compute fixed-pair counts and U once for one unique original-path sample."""
    indices = np.asarray(indices)
    weights = np.asarray(weights, float)
    if (indices.ndim != 1 or len(indices) == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Evidence requires distinct original scenario indices, not bootstrap duplicates")
    if (weights.ndim != 2 or weights.shape[1] != 3 or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=NUMERIC_TOL)):
        raise ValueError("Require fully invested long-only target weights")
    if len(weights) * len(profiles) > family_events:
        raise ValueError("Confidence allocation does not cover all candidate/client events")
    counts = []
    for profile in profiles:
        losses = np.asarray(metrics[profile["horizon"]]["mdd"])
        if (losses.ndim != 2 or losses.shape[1] != len(weights)
                or np.any(indices < 0) or np.any(indices >= len(losses))):
            raise ValueError("Invalid training indices or candidate loss dimensions")
        selected = losses[indices]
        if not np.all(np.isfinite(selected)) or np.any(selected < 0) or np.any(selected > 1):
            raise ValueError("Training MDD must be finite and in [0,1]")
        counts.append(np.count_nonzero(selected > profile["loss_tolerance"] + NUMERIC_TOL, axis=0))
    counts = np.asarray(counts, dtype=int)
    return {"indices": indices.copy(), "n_training": len(indices), "counts": counts,
            "upper": simultaneous_upper(counts, len(indices), family_events, delta),
            "family_events": int(family_events), "delta": float(delta)}


def fit_evidence_case(metrics, weights, profiles, evidence, *, requested_alpha,
                      liquidity_floors, count_limit=None):
    """Pass a mathematical count cutoff to the unchanged empirical solver.

    The cutoff is computational, not a new investor risk preference. If even
    zero cannot meet the requested budget, do not call the solver or create a
    demand-identified catalogue. Pair screening still reports every U.
    """
    n = evidence["n_training"]
    if any(p["liquidity"] not in liquidity_floors for p in profiles) or any(
            not np.isfinite(value) or not 0 <= value <= 1 for value in liquidity_floors.values()):
        raise ValueError("Require a finite [0,1] liquidity floor for each profile label")
    if count_limit is None:
        count_limit = supported_count_limit(n, requested_alpha,
                                            evidence["family_events"], evidence["delta"])
    else:
        # Defend the internal precomputed cutoff against an inconsistent caller.
        values = simultaneous_upper(np.arange(n + 1), n, evidence["family_events"], evidence["delta"])
        accepted = np.flatnonzero(values <= requested_alpha)
        expected_limit = int(accepted[-1]) if len(accepted) else -1
        if count_limit != expected_limit:
            raise ValueError("Supplied count cutoff differs from the specified family bound")
    if count_limit < 0:
        fitted = {"n_training": n, "menu": [],
                  "assignments": {p["name"]: "Review" for p in profiles},
                  "client_product_details": [], "selection": None, "completion": None,
                  "oracle_candidate_count": {p["name"]: 0 for p in profiles}}
    else:
        fitted = fit_menu(metrics, weights, profiles, evidence["indices"],
                          alpha=count_limit / n, liquidity_floors=liquidity_floors)
        solver_alpha = fitted.pop("alpha")
        if solver_alpha != count_limit / n:
            raise RuntimeError("Solver reported an unexpected empirical threshold")
    fitted.update({
        "requested_alpha": float(requested_alpha),
        "solver_empirical_alpha": count_limit / n if count_limit >= 0 else None,
        "supported_max_breach_count": int(count_limit),
        "liquidity_floors": dict(liquidity_floors),
        "evidence_status": "evidence_insufficient_even_at_zero" if count_limit < 0 else "finite_family_screen_applied",
        "family_bound": {"family_events": evidence["family_events"], "delta": evidence["delta"],
                         "per_event_delta": evidence["delta"] / evidence["family_events"],
                         "zero_count_upper": float(simultaneous_upper(np.array(0), n,
                                                           evidence["family_events"], evidence["delta"])),
                         "method": "one-sided Clopper-Pearson with Bonferroni fixed-family allocation",
                         "scope": "Individual fixed product-client exceedance probabilities, conditional on iid path and fixed-family assumptions; not growth/severity or a joint assertion across folds"},
        "threshold_interpretation": "requested_alpha is the declared service-policy case. solver_empirical_alpha is only the equivalent count threshold implementing U<=requested_alpha; it is not a newly fitted client preference.",
    })
    chosen = {(d["client"], d["candidate_index"]) for d in fitted["client_product_details"] if d["chosen"]}
    pair_rows = []
    for j, profile in enumerate(profiles):
        for candidate, target in enumerate(weights):
            count = int(evidence["counts"][j, candidate])
            upper = float(evidence["upper"][j, candidate])
            supported = count_limit >= 0 and count <= count_limit
            if supported != (upper <= requested_alpha):
                raise RuntimeError("Empirical count and simultaneous-upper-bound screens disagree")
            liquidity_pass = bool(target[2] >= liquidity_floors[profile["liquidity"]] - NUMERIC_TOL)
            row = {"client": profile["name"], "candidate_index": candidate,
                   "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                   "n_training": n, "breach_count": count, "breach_probability": count / n,
                   "simultaneous_risk_upper": upper, "requested_alpha": float(requested_alpha),
                   "solver_empirical_alpha": fitted["solver_empirical_alpha"],
                   "family_events": evidence["family_events"], "delta": evidence["delta"],
                   "evidence_status": ("evidence_insufficient" if count_limit < 0 else
                                       "risk_budget_supported" if supported else "risk_budget_not_supported"),
                   "risk_evidence_supported": bool(supported), "liquidity_pass": liquidity_pass,
                   "evidence_eligible": bool(supported and liquidity_pass),
                   "chosen": (profile["name"], candidate) in chosen}
            if row["chosen"] and not row["evidence_eligible"]:
                raise RuntimeError("Selected assignment lacks requested risk evidence or liquidity eligibility")
            pair_rows.append(row)
    lookup = {(r["client"], r["candidate_index"]): r for r in pair_rows}
    for detail in fitted["client_product_details"]:
        row = lookup[(detail["client"], detail["candidate_index"])]
        detail["solver_empirical_feasible"] = detail.pop("sample_feasible")
        detail.pop("one_path_boundary", None)
        detail.update({key: row[key] for key in ["requested_alpha", "simultaneous_risk_upper",
                                               "risk_evidence_supported", "evidence_eligible", "evidence_status"]})
        detail["supported_count_boundary"] = detail["breach_count"] == count_limit
        if detail["solver_empirical_feasible"] != row["evidence_eligible"]:
            raise RuntimeError("Menu solver eligibility differs from the family evidence screen")
    fitted["family_pair_details"] = pair_rows
    return fitted


def fit_context(protocol, evidence_protocol, arrays, metrics, indices, *, context,
                pair_file="family_pair_screening.csv"):
    """Reuse the same fixed-pair bound across all alpha/reserve cases."""
    evidence = family_evidence(metrics, arrays["weights"], protocol["clients"], indices,
                               family_events=evidence_protocol["family_events"], delta=evidence_protocol["delta"])
    limits = {float(alpha): supported_count_limit(len(indices), alpha, evidence["family_events"], evidence["delta"])
              for alpha in protocol["alpha_cases"]}
    results, pairs = {}, []
    for name, alpha, liquidity, floors in primary.policy_cases(protocol):
        fitted = fit_evidence_case(metrics, arrays["weights"], protocol["clients"], evidence,
                                   requested_alpha=alpha, liquidity_floors=floors, count_limit=limits[float(alpha)])
        fitted["liquidity_case"] = liquidity
        fitted["context"] = context
        fitted["training_scenario_ids"] = arrays["scenario_ids"][indices].astype(int).tolist()
        for row in fitted.pop("family_pair_details"):
            pairs.append({"context": context, "case": name, "liquidity_case": liquidity, **row})
        fitted["family_pair_screening"] = {"file": pair_file, "context": context,
                                          "case": name, "row_count": len(arrays["weights"]) * len(protocol["clients"]),
                                          "all_pairs_evidence_insufficient": limits[float(alpha)] < 0}
        results[name] = fitted
    return results, pairs


def oof_records(protocol, arrays, metrics, fitted_cases, fold):
    """Retain referred paths with null outcomes; no artificial bill fallback."""
    records = []
    for case, fitted in fitted_cases.items():
        menu = {p["name"]: p for p in fitted["menu"]}
        for profile in protocol["clients"]:
            assigned = fitted["assignments"][profile["name"]]
            item = menu.get(assigned)
            for scenario_id in fold["test_ids"]:
                index = scenario_id - 1
                if arrays["scenario_ids"][index] != scenario_id:
                    raise ValueError("Scenario indexing does not match the development cache")
                row = {"case": case, "requested_alpha": fitted["requested_alpha"],
                       "solver_empirical_alpha": fitted["solver_empirical_alpha"],
                       "liquidity_case": fitted["liquidity_case"], "fold": fold["fold"],
                       "scenario_id": int(scenario_id), "client": profile["name"],
                       "horizon": profile["horizon"], "loss_tolerance": profile["loss_tolerance"],
                       "assignment": assigned, "assigned": item is not None, "candidate_index": None,
                       "training_simultaneous_risk_upper": None, "cagr": None, "log_growth": None,
                       "mdd": None, "breach": None, "excess": None}
                if item is not None:
                    candidate = item["candidate_index"]
                    detail = next(d for d in fitted["client_product_details"]
                                  if d["client"] == profile["name"] and d["chosen"])
                    fields = metrics[profile["horizon"]]
                    loss = float(fields["mdd"][index, candidate])
                    breached = loss > profile["loss_tolerance"] + NUMERIC_TOL
                    row.update({"candidate_index": candidate,
                                "training_simultaneous_risk_upper": detail["simultaneous_risk_upper"],
                                "cagr": float(fields["cagr"][index, candidate]),
                                "log_growth": float(fields["log_growth"][index, candidate]),
                                "mdd": loss, "breach": bool(breached),
                                "excess": loss - profile["loss_tolerance"] if breached else 0.0})
                records.append(row)
    return records


def summarize_oof(records):
    frame = pd.DataFrame(records)
    summaries = []
    for (case, client), group in frame.groupby(["case", "client"], sort=False):
        assigned = group.loc[group.assigned]
        summary = {"case": case, "client": client, "requested_alpha": float(group.requested_alpha.iloc[0]),
                   "total_test_paths": len(group), "assigned_test_paths": len(assigned),
                   "review_test_paths": int((~group.assigned).sum()),
                   "automatic_fraction": len(assigned) / len(group),
                   "auto_folds": int(group.groupby("fold").assigned.first().sum()),
                   "review_folds": int((~group.groupby("fold").assigned.first()).sum()),
                   "breach_count": None, "breach_probability": None, "mean_log_growth": None,
                   "median_cagr": None, "p10_cagr": None, "median_mdd": None, "p90_mdd": None,
                   "expected_excess": None, "internal_calibration_warning": None,
                   "interpretation": "Mixed-policy OOF internal diagnostic with overlapping training sets; no pooled binomial certificate or joint-fold assertion"}
        if len(assigned):
            count = int(assigned.breach.astype(bool).sum())
            probability = count / len(assigned)
            summary.update({"breach_count": count, "breach_probability": probability,
                            "mean_log_growth": float(assigned.log_growth.mean()),
                            "median_cagr": float(assigned.cagr.median()),
                            "p10_cagr": float(np.quantile(assigned.cagr, .10, method="inverted_cdf")),
                            "median_mdd": float(assigned.mdd.median()),
                            "p90_mdd": float(np.quantile(assigned.mdd, .90, method="inverted_cdf")),
                            "expected_excess": float(assigned.excess.mean()),
                            "internal_calibration_warning": probability > summary["requested_alpha"] + NUMERIC_TOL})
        summaries.append(summary)
    return summaries


def run_validation(protocol, evidence_protocol, arrays, metrics, *, pair_file="family_pair_screening.csv"):
    primary.validate_folds(protocol)
    results = {name: [] for name, *_ in primary.policy_cases(protocol)}
    pairs, records = [], []
    for fold in protocol["validation"]["folds"]:
        indices = np.asarray(fold["train_ids"], dtype=int) - 1
        fitted, rows = fit_context(protocol, evidence_protocol, arrays, metrics, indices,
                                   context=f"fold_{fold['fold']}_train", pair_file=pair_file)
        for case, fit in fitted.items():
            fit.update({"fold": fold["fold"], "test_scenario_ids": fold["test_ids"]})
            results[case].append(fit)
        pairs.extend(rows)
        records.extend(oof_records(protocol, arrays, metrics, fitted, fold))
        print(json.dumps({"stage": "simultaneous_validation", "fold": fold["fold"],
                          "assignments": {case: fit["assignments"] for case, fit in fitted.items()}}), flush=True)
    return results, pairs, records, summarize_oof(records)


def development_input_manifest(arrays, metrics):
    digest = hashlib.sha256()
    for value in [arrays["scenario_ids"], arrays["weights"],
                  *[metrics[h][key] for h in sorted(metrics) for key in sorted(metrics[h])]]:
        digest.update(np.ascontiguousarray(value).tobytes())
    cache = primary.OUT / "cache" / f"{primary.cache_key(.05, .0025)}.npz"
    cache_meta = cache.with_suffix(".json")
    manifest = json.loads(cache_meta.read_text())
    if manifest["npz_sha256"] != primary.sha(cache):
        raise RuntimeError("Primary cache changed after its validation")
    return {"scope": "Development paths 1–120 only; original loader allow_final=False",
            "scenario_ids": arrays["scenario_ids"].astype(int).tolist(),
            "candidate_count": len(arrays["weights"]), "cost_rate": .0025,
            "cache": str(cache.relative_to(ROOT)), "cache_sha256": primary.sha(cache),
            "cache_metadata": str(cache_meta.relative_to(ROOT)), "cache_metadata_sha256": primary.sha(cache_meta),
            "development_arrays_sha256": manifest["development_arrays_sha256"],
            "diagnostic_input_arrays_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "validate", "all"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    sources = source_manifest()
    protocol = json.loads(primary.PROTOCOL.read_text())
    evidence_protocol = json.loads(PROTOCOL.read_text())
    arrays, metrics = primary.get_cache(protocol, step=.05, cost=.0025)
    validate_scope(protocol, evidence_protocol, arrays)
    inputs = development_input_manifest(arrays, metrics)
    outputs, pairs = {}, []
    pair_file = f"{args.stage}_family_pair_screening.csv"
    if args.stage in {"fit", "all"}:
        full, full_pairs = fit_context(protocol, evidence_protocol, arrays, metrics, np.arange(120),
                                       context="full_development", pair_file=pair_file)
        outputs["full_development_menus.json"] = full
        pairs.extend(full_pairs)
    if args.stage in {"validate", "all"}:
        folds, fold_pairs, records, summaries = run_validation(protocol, evidence_protocol, arrays, metrics,
                                                               pair_file=pair_file)
        outputs["validation_fold_menus.json"] = folds
        outputs["validation_oof_paths.csv"] = pd.DataFrame(records)
        outputs["validation_oof_summary.csv"] = pd.DataFrame(summaries)
        pairs.extend(fold_pairs)
    outputs[pair_file] = pd.DataFrame(pairs)
    if source_manifest() != sources:
        raise RuntimeError("Sources changed while fitting; no diagnostic outputs were written")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(args.out / name, index=False)
        else:
            primary.write_json(args.out / name, value)
    primary.write_json(args.out / f"{args.stage}_manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "stage": args.stage,
        "sources": sources, "inputs": inputs, "summary_metadata": SUMMARY_METADATA,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
        "outputs_sha256": {name: primary.sha(args.out / name) for name in outputs},
        "protocol": evidence_protocol,
        "scope": "Secondary conservative static diagnostic; original empirical research remains primary; retrospectively proposed after exploratory exposure",
        "bound_reuse": "1160 fixed product-client event bounds reused for nine alpha/reserve cases, without an extra case/menu multiplicity factor",
        "fold_scope": "Separate training-sample bounds only; no confidence assertion jointly covering all folds and no pooled OOF certificate",
        "referrals": "Retained in all coverage denominators; investment outcomes null, without invented adviser or fallback performance",
        "bootstrap": "Not used for certification or as additional independent evidence",
        "final_scenarios_loaded": False,
    })
    print(f"Wrote {args.stage} secondary diagnostic to {args.out}", flush=True)


if __name__ == "__main__":
    main()
