"""Posthoc OOB diagnostics of frozen development bootstrap assignments.

OOB observations overlap, and policies differ by replicate. All statistics
describe this empirical resampling procedure, not independent market evidence.
Importing the module opens no market or result files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/model_review/risk_menu_v1"
BOOT = BASE / "bootstrap/alpha_10__risk_only"
OUT = BOOT / "oob"
PROTOCOL = ROOT / "docs/bootstrap_oob_diagnostic_protocol.json"
SAMPLES = ("inbag", "oob", "original120_including_inbag")
FIELDS = ("mean_log_growth", "breach_count", "breach_probability", "expected_excess", "cvar90_mdd")
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def reconstruct_samples(seed, replicates, n_scenarios):
    for name, value in (("seed", seed), ("replicates", replicates), ("n_scenarios", n_scenarios)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    if seed < 0 or replicates < 1 or n_scenarios < 1:
        raise ValueError("Require nonnegative seed and positive replicate/scenario counts")
    return np.random.default_rng(seed).integers(0, n_scenarios, size=(replicates, n_scenarios))


def sample_metrics(log_growth_vector, mdd_vector, indices, loss_tolerance):
    growth, mdd = np.asarray(log_growth_vector, float), np.asarray(mdd_vector, float)
    indices = np.asarray(indices)
    if growth.ndim != 1 or mdd.shape != growth.shape or len(growth) == 0:
        raise ValueError("Require equally sized nonempty one-dimensional growth/MDD arrays")
    if indices.ndim != 1 or (len(indices) and not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("Require one-dimensional integer sample indices")
    if np.any(indices < 0) or np.any(indices >= len(growth)):
        raise ValueError("Sample index outside supplied scenarios")
    if not np.isfinite(loss_tolerance) or not 0 <= loss_tolerance <= 1:
        raise ValueError("Loss tolerance must lie in [0,1]")
    if not len(indices):
        return dict.fromkeys(FIELDS)
    g, d = growth[indices], mdd[indices]
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(d)) or np.any(d < 0) or np.any(d > 1):
        raise ValueError("Selected growth/MDD must be finite, with MDD in [0,1]")
    breaches = d - loss_tolerance > TOL
    ordered = np.sort(d)[::-1]
    tail_count = .1 * len(ordered)
    whole = int(np.floor(tail_count))
    tail_sum = float(ordered[:whole].sum())
    if tail_count > whole:
        tail_sum += (tail_count - whole) * float(ordered[whole])
    return {"mean_log_growth": float(g.mean()), "breach_count": int(breaches.sum()),
            "breach_probability": float(breaches.mean()),
            "expected_excess": float(np.where(breaches, d - loss_tolerance, 0).mean()),
            "cvar90_mdd": tail_sum / tail_count}


def candidate_id(value):
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Candidate IDs cannot be boolean")
    number = float(value)
    if not np.isfinite(number) or number != int(number) or number < 0:
        raise ValueError("Candidate ID must be a nonnegative integer or null")
    return int(number)


def indexed_records(records, n):
    result = {}
    for record in records:
        index = int(record["replicate"])
        if index in result or index < 0 or index >= n:
            raise ValueError("Duplicate or invalid stored replicate identity")
        result[index] = record
    if set(result) != set(range(n)):
        raise ValueError("Stored replicate identities must cover the sample matrix exactly")
    return result


def build_rows(samples, assignments, menus, profiles, metrics, n_scenarios=120):
    samples = np.asarray(samples)
    if (samples.ndim != 2 or samples.shape[1] != n_scenarios
            or not np.issubdtype(samples.dtype, np.integer)
            or np.any(samples < 0) or np.any(samples >= n_scenarios)):
        raise ValueError("Require a valid replicate-by-scenario bootstrap sample matrix")
    stored = indexed_records(assignments, len(samples))
    catalogues = indexed_records(menus, len(samples))
    rows, memberships = [], []
    for replicate, inbag in enumerate(samples):
        unique = np.unique(inbag)
        oob = np.setdiff1d(np.arange(n_scenarios), unique)
        if len(unique) + len(oob) != n_scenarios or np.intersect1d(unique, oob).size:
            raise RuntimeError("Inbag/OOB membership is not a disjoint complete partition")
        membership = {"replicate": replicate, "inbag_indices_with_multiplicity": inbag.tolist(),
                      "inbag_unique_indices": unique.tolist(), "oob_indices": oob.tolist(),
                      "inbag_scenario_ids_with_multiplicity": (inbag + 1).tolist(),
                      "inbag_unique_scenario_ids": (unique + 1).tolist(), "oob_scenario_ids": (oob + 1).tolist()}
        memberships.append(membership)
        menu = catalogues[replicate]["candidate_indices_by_risk_label"]
        if len(set(menu)) != len(menu) or len(menu) > 3:
            raise ValueError("Stored catalogue must contain at most three distinct targets")
        roles = dict(zip(("Conservative", "Balanced", "Growth"), menu))
        for profile in profiles:
            client, horizon, loss = profile["name"], profile["horizon"], profile["loss_tolerance"]
            assignment = stored[replicate][client]
            candidate = candidate_id(stored[replicate][client + "_candidate_index"])
            assigned = assignment != "Review"
            if (assigned and (assignment not in roles or candidate != roles[assignment])) or (not assigned and candidate is not None):
                raise ValueError("Frozen assignment/candidate differs from its frozen catalogue")
            row = {"replicate": replicate, "client": client, "horizon": horizon, "loss_tolerance": loss,
                   "assignment": assignment, "assigned": assigned, "candidate_index": candidate,
                   "inbag_n": len(inbag), "inbag_unique_n": len(unique), "oob_n": len(oob),
                   "oob_assigned_path_count": len(oob) if assigned else 0,
                   "original120_including_inbag_n": n_scenarios,
                   "original120_including_inbag_is_validation": False,
                   **{key: membership[key] for key in ("inbag_scenario_ids_with_multiplicity",
                                                       "inbag_unique_scenario_ids", "oob_scenario_ids")}}
            for sample, indices in (("inbag", inbag), ("oob", oob),
                                    ("original120_including_inbag", np.arange(n_scenarios))):
                values = dict.fromkeys(FIELDS)
                if assigned:
                    fields = metrics[horizon]
                    if fields["log_growth"].shape != fields["mdd"].shape or fields["mdd"].shape[0] != n_scenarios:
                        raise ValueError("Cached metric dimensions differ from scenario scope")
                    if not 0 <= candidate < fields["mdd"].shape[1]:
                        raise ValueError("Frozen candidate outside supplied cache")
                    values = sample_metrics(fields["log_growth"][:, candidate], fields["mdd"][:, candidate], indices, loss)
                row.update({sample + "_" + key: value for key, value in values.items()})
            rows.append(row)
    return rows, memberships


def describe(values):
    numbers = np.asarray([v for v in values if v is not None], float)
    if not len(numbers):
        return {"count": 0, "mean": None, "median": None, "p025": None, "p975": None,
                "minimum": None, "maximum": None}
    if not np.all(np.isfinite(numbers)):
        raise ValueError("Summary contains nonfinite values")
    return {"count": len(numbers), "mean": float(numbers.mean()), "median": float(np.median(numbers)),
            "p025": float(np.quantile(numbers, .025, method="inverted_cdf")),
            "p975": float(np.quantile(numbers, .975, method="inverted_cdf")),
            "minimum": float(numbers.min()), "maximum": float(numbers.max())}


def recurrence_diagnostics(rows, menus, full_fitted):
    original_tuple = tuple(p["candidate_index"] for p in full_fitted["menu"])
    stored_tuples = [tuple(p["candidate_indices_by_risk_label"]) for p in menus]
    n = len(menus)
    a_auto = {row["replicate"]: row["assigned"] for row in rows if row["client"] == "A"}
    clients = sorted({row["client"] for row in rows})
    result = {"replicates": n, "original_risk_label_menu_tuple": list(original_tuple),
              "exact_ordered_menu_count": sum(t == original_tuple for t in stored_tuples),
              "exact_ordered_menu_frequency": sum(t == original_tuple for t in stored_tuples) / n,
              "same_unordered_target_set_count": sum(set(t) == set(original_tuple) for t in stored_tuples),
              "same_unordered_target_set_frequency": sum(set(t) == set(original_tuple) for t in stored_tuples) / n,
              "distinct_ordered_menus": len(set(stored_tuples)), "by_client": {}}
    for client in clients:
        subset = [r for r in rows if r["client"] == client]
        original_role = full_fitted["assignments"][client]
        original_target = next((p["candidate_index"] for p in full_fitted["menu"] if p["name"] == original_role), None)
        auto = [r for r in subset if r["assigned"]]
        same_target = sum(r["candidate_index"] == original_target for r in auto) if original_target is not None else None
        joint = Counter((r["assignment"], bool(a_auto.get(r["replicate"], False))) for r in subset)
        changed = Counter((r["assignment"] != original_role, bool(a_auto.get(r["replicate"], False))) for r in subset)
        result["by_client"][client] = {
            "original_assignment": original_role, "original_target": original_target,
            "original_target_recurrence_count": same_target,
            "original_target_frequency_all_replicates": same_target / n if same_target is not None else None,
            "original_target_frequency_conditional_automatic": same_target / len(auto) if same_target is not None and auto else None,
            "original_referral_recurrence_count": sum(not r["assigned"] for r in subset) if original_target is None else None,
            "role_counts": dict(Counter(r["assignment"] for r in subset)),
            "role_joint_A_automatic": [{"assignment": role, "A_automatic": flag, "count": count}
                                        for (role, flag), count in sorted(joint.items())],
            "role_changed_joint_A_automatic": [{"role_changed": change, "A_automatic": flag, "count": count}
                                                for (change, flag), count in sorted(changed.items())],
            "interpretation": "Risk labels describe each replicate's relative catalogue positions; role movements do not identify changing investor preferences."}
    return result


def summarize_rows(rows, menus, full_fitted, alpha):
    summary = {"case": "alpha_10__risk_only", "alpha": alpha,
               "aggregation": "Equal weight per automatically assigned replicate-client, not pooled OOB path rows.",
               "interpretation": "Posthoc functional diagnostics conditional on the empirical120-path distribution and stored bootstrap selection procedure; all quantiles are descriptive, without population confidence or pooled binomial assertions.",
               "row_count": len(rows), "automatic_rows": sum(r["assigned"] for r in rows),
               "review_rows": sum(not r["assigned"] for r in rows), "by_client": {},
               "recurrence_and_role_movement": recurrence_diagnostics(rows, menus, full_fitted)}
    for client in sorted({r["client"] for r in rows}):
        subset = [r for r in rows if r["client"] == client]
        auto = [r for r in subset if r["assigned"]]
        valid_oob = [r for r in auto if r["oob_n"] > 0]
        stats = {"replicates": len(subset), "automatic_replicates": len(auto),
                 "review_replicates": len(subset) - len(auto), "automatic_frequency": len(auto) / len(subset),
                 "automatic_with_nonempty_oob": len(valid_oob),
                 "oob_n_all_replicates": describe([r["oob_n"] for r in subset]),
                 "oob_n_conditional_automatic": describe([r["oob_n"] for r in auto]),
                 "inbag_unique_n_all_replicates": describe([r["inbag_unique_n"] for r in subset]),
                 "sample_metrics": {}, "oob_minus_inbag": {},
                 "oob_empirical_breach_proportion_above_alpha_count": sum(r["oob_breach_probability"] > alpha + TOL for r in valid_oob),
                 "oob_empirical_breach_proportion_above_alpha_fraction":
                     sum(r["oob_breach_probability"] > alpha + TOL for r in valid_oob) / len(valid_oob) if valid_oob else None,
                 "above_alpha_interpretation": "A discrete small-OOB-sample diagnostic, not a hypothesis test or population-risk probability."}
        for sample in SAMPLES:
            stats["sample_metrics"][sample] = {field: describe([r[sample + "_" + field] for r in auto]) for field in FIELDS}
        for field in ("mean_log_growth", "breach_probability", "expected_excess", "cvar90_mdd"):
            stats["oob_minus_inbag"][field] = describe([r["oob_" + field] - r["inbag_" + field] for r in valid_oob])
        summary["by_client"][client] = stats
    return summary


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    return stream.getvalue().encode()


def verify_reconstruction(samples, assignments, menus, metrics, weights, profiles, full_fitted, indices):
    # Exactly the three predeclared seed checks; never use OOB to choose a target.
    sys.path.insert(0, str(ROOT / "research"))
    from safeinvest_research.selection import fit_menu
    stored, catalogues = indexed_records(assignments, len(samples)), indexed_records(menus, len(samples))
    checks = []
    for replicate in indices:
        fitted = fit_menu(metrics, weights, profiles, samples[replicate], alpha=full_fitted["alpha"],
                          liquidity_floors=full_fitted["liquidity_floors"])
        actual_menu = [p["candidate_index"] for p in fitted["menu"]]
        expected_menu = catalogues[replicate]["candidate_indices_by_risk_label"]
        expected_assignments = {p["name"]: stored[replicate][p["name"]] for p in profiles}
        menu_ok = actual_menu == expected_menu
        weight_ok = np.allclose([p["weights"] for p in fitted["menu"]],
                                catalogues[replicate]["weights_by_risk_label"], atol=1e-14, rtol=0)
        assignments_ok = fitted["assignments"] == expected_assignments
        chosen_ok = True
        chosen = {}
        for p in profiles:
            detail = next((d for d in fitted["client_product_details"] if d["client"] == p["name"] and d["chosen"]), None)
            chosen[p["name"]] = detail["candidate_index"] if detail else None
            chosen_ok &= chosen[p["name"]] == candidate_id(stored[replicate][p["name"] + "_candidate_index"])
        record = {"replicate": replicate, "menu_ids_match": menu_ok, "menu_weights_match": bool(weight_ok),
                  "assignments_match": assignments_ok, "chosen_target_ids_match": bool(chosen_ok),
                  "candidate_indices_by_risk_label": actual_menu, "assignments": fitted["assignments"],
                  "chosen_candidate_ids": chosen, "inbag_unique_n": len(np.unique(samples[replicate])),
                  "oob_n": samples.shape[1] - len(np.unique(samples[replicate]))}
        checks.append(record)
        if not (menu_ok and weight_ok and assignments_ok and chosen_ok):
            raise RuntimeError("Seed reconstruction differs from frozen fit: " + json.dumps(record))
    return {"passed": True, "replicates": indices, "checks": checks,
            "scope": "Unchanged inbag fitting for fixed seed reconstruction only; not an OOB target search."}


def main():
    protocol_hash_before_inputs = sha(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text())
    if not (protocol["case"] == "alpha_10__risk_only" and protocol["replicates"] == 1000
            and protocol["seed"] == 87020261018 and protocol["forbidden_final_scenario_access"] is True
            and protocol["allowed_scenario_ids"] == [1, 120]):
        raise ValueError("Diagnostic protocol scope differs from the declared case")
    cache = BASE / "cache/step_0500__cost_002500.npz"
    metadata = cache.with_suffix(".json")
    input_paths = [PROTOCOL, ROOT / "docs/risk_menu_development_protocol.json", cache, metadata,
                   BOOT / "summary.json", BOOT / "assignments.csv", BOOT / "menus.json",
                   BOOT / "selection_diagnostics.csv", BASE / "full_development_menus.json", BASE / "fit_manifest.json"]
    inputs_start = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    if inputs_start[str(PROTOCOL.relative_to(ROOT))] != protocol_hash_before_inputs:
        raise RuntimeError("Protocol changed while loading inputs")
    bootstrap = json.loads((BOOT / "summary.json").read_text())
    primary = json.loads((ROOT / "docs/risk_menu_development_protocol.json").read_text())
    meta = json.loads(metadata.read_text())
    if (bootstrap["replicates"] != protocol["replicates"] or bootstrap["seed"] != protocol["seed"]
            or bootstrap["case"] != protocol["case"] or primary["stability"]["bootstrap_seed"] != protocol["seed"]
            or meta["scenario_ids"] != list(range(1, 121)) or meta["candidate_count"] != 232
            or meta["cost_rate"] != .0025 or meta["step"] != .05 or meta["npz_sha256"] != sha(cache)):
        raise RuntimeError("Frozen bootstrap/cache identity or scope mismatch")
    sources_start = dict(bootstrap["sources"])
    for relative, digest in meta["source_hashes"].items():
        if relative in sources_start and sources_start[relative] != digest:
            raise RuntimeError("Bootstrap and numerical-cache source snapshots differ")
        sources_start[relative] = digest
    sources_start[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    test_path = ROOT / "tests/test_bootstrap_oob.py"
    if test_path.exists():
        sources_start[str(test_path.relative_to(ROOT))] = sha(test_path)
    if {name: sha(ROOT / name) for name in sources_start} != sources_start:
        raise RuntimeError("Frozen source snapshot changed before OOB diagnostic")
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if not np.array_equal(arrays["scenario_ids"], np.arange(1, 121)) or arrays["weights"].shape != (232, 3):
        raise RuntimeError("Only the fixed development grid is permitted")
    sys.path.insert(0, str(ROOT / "src"))
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    development_hash = hashlib.sha256()
    for value in (ids, returns, inflation):
        development_hash.update(np.ascontiguousarray(value).tobytes())
    if development_hash.hexdigest() != meta["development_arrays_sha256"]:
        raise RuntimeError("Guarded development arrays differ from cached input provenance")
    metrics = {h: {field: arrays[f"H{h}__{field}"] for field in ("log_growth", "mdd")}
               for h in primary["horizons"]}
    full_menus = json.loads((BASE / "full_development_menus.json").read_text())
    full = full_menus[protocol["case"]]
    fitted_manifest = json.loads((BASE / "fit_manifest.json").read_text())
    if (fitted_manifest["menus_sha256"] != sha(BASE / "full_development_menus.json")
            or fitted_manifest["sources"] != bootstrap["sources"]
            or fitted_manifest["scenario_ids"] != ids.tolist()
            or full["alpha"] != .10 or full["liquidity_floors"] != {"high": 0., "medium": 0., "low": 0.}):
        raise RuntimeError("Full menu artifact hash differs from fit manifest")
    assignments, menus = read_csv(BOOT / "assignments.csv"), json.loads((BOOT / "menus.json").read_text())
    samples = reconstruct_samples(protocol["seed"], protocol["replicates"], 120)
    reconstruction = verify_reconstruction(samples, assignments, menus, metrics, arrays["weights"], primary["clients"],
                                             full, protocol["reconstruction_checks"]["fixed_replicates"])
    rows, memberships = build_rows(samples, assignments, menus, primary["clients"], metrics)
    if len(rows) != 5000 or any(r["assigned"] and r["inbag_breach_probability"] > full["alpha"] + TOL for r in rows):
        raise RuntimeError("Frozen automatic decisions fail their reconstructed inbag screen")
    summary = summarize_rows(rows, menus, full, full["alpha"])
    for client, stats in summary["by_client"].items():
        if (stats["automatic_frequency"] != bootstrap["automatic_frequency"][client]
                or summary["recurrence_and_role_movement"]["by_client"][client]["role_counts"]
                != bootstrap["client_assignment_counts"][client]):
            raise RuntimeError("Reconstructed frozen assignments differ from bootstrap summary")
    stored_diagnostics = indexed_records(read_csv(BOOT / "selection_diagnostics.csv"), len(samples))
    for replicate in range(len(samples)):
        selected_rows = [row for row in rows if row["replicate"] == replicate]
        if sum(row["assigned"] for row in selected_rows) != int(stored_diagnostics[replicate]["covered_clients"]):
            raise RuntimeError("Reconstructed coverage differs from stored bootstrap selection diagnostics")
    for catalogue in menus:
        target_ids = catalogue["candidate_indices_by_risk_label"]
        if not np.allclose(catalogue["weights_by_risk_label"], arrays["weights"][target_ids], atol=1e-14, rtol=0):
            raise RuntimeError("Stored bootstrap target weights differ from fixed-grid candidate identities")
    if summary["row_count"] != 5000:
        raise RuntimeError("Missing replicate/client coverage records")
    inputs_end = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    if inputs_end != inputs_start:
        raise RuntimeError("Input artifacts changed during OOB diagnostic; no outputs written")
    sources_end = {name: sha(ROOT / name) for name in sources_start}
    if sources_end != sources_start:
        raise RuntimeError("Sources changed during OOB diagnostic; no outputs written")
    payloads = {"memberships.json": json_bytes({"index_convention": "indices are zero-based; scenario_ids are one-based",
                                               "memberships": memberships}),
                "replicate_client_metrics.csv": csv_bytes(rows), "summary.json": json_bytes(summary),
                "reconstruction_checks.json": json_bytes(reconstruction)}
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256_before_input_loading": protocol_hash_before_inputs,
                "case": protocol["case"], "seed": protocol["seed"], "replicates": protocol["replicates"],
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "input_artifact_sha256_start": inputs_start, "input_artifact_sha256_end": inputs_end,
                "source_sha256_start": sources_start, "source_sha256_end": sources_end,
                "bootstrap_indices_sha256": hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest(),
                "development_arrays_sha256": development_hash.hexdigest(), "scenario_ids": ids.tolist(),
                "outputs_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
                "scope": protocol["scope"], "interpretation": summary["interpretation"],
                "membership_and_row_coverage": {"replicate_memberships": len(memberships), "replicate_client_rows": len(rows),
                                                  "automatic_rows": summary["automatic_rows"], "review_rows": summary["review_rows"]},
                "final_scenarios_loaded": False, "oob_used_for_selection": False,
                "reconstruction_only_refits": reconstruction["replicates"],
                "outputs_verified_after_write": True}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        (OUT / name).write_bytes(value)
    written_hashes = {name: sha(OUT / name) for name in payloads}
    if written_hashes != manifest["outputs_sha256"]:
        raise RuntimeError("Written output hash mismatch")
    manifest["output_file_sha256_after_write"] = written_hashes
    if ({name: sha(ROOT / name) for name in sources_start} != sources_start
            or {str(p.relative_to(ROOT)): sha(p) for p in input_paths} != inputs_start):
        raise RuntimeError("Provenance changed during output writing; no completion manifest written")
    (OUT / "manifest.json").write_bytes(json_bytes(manifest))
    print(json.dumps({"output": str(OUT), "reconstruction_passed": reconstruction["passed"],
                      "rows": len(rows), "automatic_rows": summary["automatic_rows"],
                      "review_rows": summary["review_rows"]}))


if __name__ == "__main__":
    main()
