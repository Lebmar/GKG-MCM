"""Account for eleven prospectively frozen pairs of previously seen paths.

This is a post-hoc mechanism diagnosis. It does not fit, rescore a price,
change an admission decision, or supply financials for today's refused C.
The only new simulation is four missing same-target inner B references,
using the production trade function and an explicit original-history loop.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from diagnostics.failure_path_mechanisms import (Checks, TOL, prepare_account,
    endpoint_records, annual_records, paired_records, cross_stage_summary)

PROTOCOL = ROOT / "docs/failure_path_mechanism_protocol.json"
OUT = ROOT / "results/model_review/failure_path_mechanisms_v1"
PARENTS = ("joint_liquidity_refit_v1", "refit_stability_v1")
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
EXPECTED_PAIRS = {"full_35", "full_112", "full_52", "full_98", "inner_35", "inner_112", "inner_52", "inner_98", "outer_r2_52", "outer_r4_112", "outer_r5_112"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def readj(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_protocol(p):
    if p.get("execution_ready") is not True:
        raise RuntimeError("Protocol and analysis sources must be frozen first")
    if (p.get("data", {}).get("scenario_ids") != list(range(1, 121))
            or p["data"].get("allow_final") is not False
            or p.get("original_H") != 30 or p.get("loss_tolerance") != .4 or p.get("cost_rate") != .0025
            or p.get("output_directory") != str(OUT.relative_to(ROOT))):
        raise ValueError("Original guarded development/accounting contract differs")
    pairs = p["inventory"]
    if len(pairs) != 11 or {a["pair_id"] for a in pairs} != EXPECTED_PAIRS:
        raise ValueError("Exactly the eleven fixed original-ID pairs required")
    for x in pairs:
        tr, sid = x["training_ids"], x["scenario_id"]
        if (not isinstance(sid, int) or isinstance(sid, bool) or sid not in (35, 52, 98, 112)
                or len(tr) != x["training_fan_n"] or len(set(tr)) != len(tr)
                or any(not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= 120 for i in tr)
                or x["folder"] not in PARENTS or x["trace_scope"] not in ("actual", "validation")
                or x["selected_candidate_order"] not in (4, 5)):
            raise ValueError("Fixed identity/own training source differs")
        full, inner = x["stage"] == "full_fixed_training", x["stage"] == "selected_beta_inner"
        if (x["stage"] not in ("full_fixed_training", "selected_beta_inner", "outer_oof")
                or (sid in tr) != full or len(tr) != (120 if full else 90 if inner else 96)
                or (x["B_group"] is None) != inner or (x["trace_scope"] == "validation") != inner
                or x["folder"] != PARENTS[0 if full or inner else 1]):
            raise ValueError("Full/inner/outer distinction differs")
    return pairs


def snapshot(p, protocol_digest):
    if sha(PROTOCOL) != protocol_digest:
        raise RuntimeError("Frozen protocol changed")
    results = {}
    for key in ("execution_sources_sha256", "frozen_input_sha256"):
        expected = p[key]
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Nonempty frozen hash maps required")
        observed = {}
        for name, digest in expected.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Repository-relative SHA256 entries required")
            observed[name] = sha(ROOT / name)
        if observed != expected:
            raise RuntimeError("Frozen source/input differs: " + key)
        results[key] = observed
    required = {"scripts/diagnose_failure_path_mechanisms.py", "research/diagnostics/failure_path_mechanisms.py", "tests/test_failure_path_mechanisms.py", "src/safeinvest/core.py", "src/safeinvest/data.py"}
    if not required <= set(results["execution_sources_sha256"]):
        raise ValueError("Missing execution sources in frozen protocol")
    return results


def reconstruct_B(returns, inflation, target, *, loss=.4, cost=.0025):
    """Production-fee reference, annual restore only until first real breach."""
    from safeinvest.core import execute_target
    r, z, target = np.asarray(returns, float), np.asarray(inflation, float), np.asarray(target, float)
    h = len(r)
    x, cp, pk, worst, bad, tau = np.array([0., 0., 100.]), 1., 100., 0., False, -1
    names = ("holdings", "weights", "nominal_wealth", "real_wealth", "real_peak", "current_drawdown", "running_mdd", "ever_breached", "cpi",
             "posttrade_holdings", "fees", "moved", "pending_at_decision", "target_execution", "actions", "trades")
    out = {name: [] for name in names}
    def endpoint():
        V = float(x.sum())
        for name, val in (("holdings", x.copy()), ("weights", x / V), ("nominal_wealth", V), ("real_wealth", V / cp),
                          ("real_peak", pk), ("current_drawdown", max(1 - V / cp / pk, 0.)), ("running_mdd", worst), ("ever_breached", bad), ("cpi", cp)):
            out[name].append(val)
    endpoint()
    for t in range(h):
        restore = not bad
        post, fee, moved = execute_target(x, target, cost) if restore else (x.copy(), 0., 0.)
        for name, val in (("posttrade_holdings", post.copy()), ("fees", float(fee)), ("moved", float(moved)),
                          ("pending_at_decision", bad), ("target_execution", restore), ("actions", "restore" if restore else "hold"), ("trades", bool(moved > TOL))):
            out[name].append(val)
        x = post * (1 + r[t]); cp *= 1 + z[t]
        real = float(x.sum()) / cp
        pk = max(pk, real); worst = max(worst, max(1 - real / pk, 0.))
        bad = bad or worst - loss > TOL
        if tau == -1 and bad:
            tau = t + 1
        endpoint()
    out = {name: np.asarray(value) for name, value in out.items()}
    out["first_crossing"] = np.asarray(tau)
    return out


def extract(archive, group, ids, sid):
    if sid not in ids or len(ids) != len(set(ids)):
        raise ValueError("Original scenario missing or duplicated")
    index = ids.index(sid)
    metadata_fields = {"targets", "loss_tolerances", "annual_prices"}
    prefix = group + "__"
    out = {}
    for key in archive.files:
        if key.startswith(prefix):
            field = key[len(prefix):]
            value = archive[key]
            out[field] = value.copy() if field in metadata_fields else value[index].copy()
    if not out:
        raise ValueError("Frozen saved account missing")
    return out


def metadata_columns(item):
    return {k: item[k] for k in ("pair_id", "stage", "scenario_id", "folder", "job_id", "context", "repeat_id", "selected_candidate_order", "selected_beta", "training_fan_n")}


def verify_inner_B_saved_metrics(item, b, references, checks):
    rows = references.loc[references.job_id.eq(item["job_id"]) & references.scenario_id.eq(item["scenario_id"])]
    checks.require(len(rows) == 1, "inner_B/unique_saved_reference")
    row = rows.iloc[0]
    checks.require(isinstance(row.assigned, (bool, np.bool_)) and bool(row.assigned), "inner_B/saved_available")
    checks.require(int(row.candidate_local_index) == item["target_local_id"] and int(row.candidate_original232_index) == item["target_original232_id"]
                   and int(row.selection_training_n) == item["training_fan_n"], "inner_B/saved_own_target")
    h = len(b["fees"]); growth = math.log(float(b["real_wealth"][-1] / b["real_wealth"][0])) / h
    metrics = {"mdd": float(b["running_mdd"][-1]), "terminal_real": float(b["real_wealth"][-1]),
               "log_growth": growth, "cagr": float(np.expm1(growth)), "nominal_fees": float(sum(b["fees"])),
               "nominal_moved": float(sum(b["moved"])), "trade_fraction": float(np.mean(b["trades"])),
               "target_execution_fraction": float(np.mean(b["target_execution"]))}
    metrics.update({"terminal_" + asset: float(b["weights"][-1, j]) for j, asset in enumerate(("equity", "bond", "bill"))})
    for field, value in metrics.items():
        checks.close(float(row[field]), value, "inner_B/saved_" + field)
    checks.require(isinstance(row.breach, (bool, np.bool_)) and bool(row.breach) == bool(b["ever_breached"][-1]), "inner_B/saved_breach")
    checks.require(pd.isna(row.first_crossing) if b["first_crossing_value"] is None else float(row.first_crossing) == b["first_crossing_value"], "inner_B/saved_crossing")


def check_group(item, groups, key, *, model):
    if key not in groups:
        raise ValueError("Frozen trace group unavailable")
    group = groups[key]; md = group["metadata"]
    if (item["scenario_id"] not in group["scenario_ids"] or md["job_id"] != item["job_id"]
            or md["context"] != item["context"] or md["selection_training_n"] != item["training_fan_n"]
            or int(md["candidate_local_index"]) != item["target_local_id"]
            or int(md["candidate_original232_index"]) != item["target_original232_id"]
            or group["selected_candidate_order"] != (item["selected_candidate_order"] if model else None)):
        raise ValueError("Own job/target/order trace metadata differs")
    return group


def verify_static_metadata(item, bundle):
    product = bundle["q2_case"]["product"]
    if (bundle["refit"]["training_ids"] != item["training_ids"] or product is None
            or product["weights"] != item["target"] or product["candidate_index"] != item["target_local_id"]
            or product["candidate_original232_index"] != item["target_original232_id"]
            or bundle["refit"]["selection"]["selected_gamma"] != item["static_gamma"]
            or bundle["training_price_selections"][str(item["selected_beta"])]["selected_candidate_order"] != item["selected_candidate_order"]):
        raise ValueError("Frozen own-job target/training/price metadata differs")


def run():
    if OUT.exists():
        raise RuntimeError("Refuse to overwrite a previous forensic archive")
    p = readj(PROTOCOL); items = validate_protocol(p); protocol_digest = sha(PROTOCOL)
    started = time.perf_counter(); before = snapshot(p, protocol_digest)
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)):
        raise ValueError("Original development identity order changed")
    digest = hashlib.sha256()
    for array in (ids, returns, inflation):
        digest.update(np.ascontiguousarray(array).tobytes())
    groups = {folder: readj(ROOT / "results/model_review" / folder / "trace_groups.json") for folder in PARENTS}
    static = {PARENTS[0]: readj(ROOT / "results/model_review" / PARENTS[0] / "static_refits.json"),
              PARENTS[1]: readj(ROOT / "results/model_review" / PARENTS[1] / "static_refits.json.gz")}
    archives = {(folder, scope): np.load(ROOT / "results/model_review" / folder / (scope + "_traces.npz"), allow_pickle=False)
                for folder, scope in sorted({(item["folder"], item["trace_scope"]) for item in items})}
    references = pd.read_csv(ROOT / "results/model_review/benefit_cost_execution_v1/inner_reference_paths.csv.gz")
    endpoints, annual, paired, summaries, witnesses, availability = [], [], [], [], [], []
    model_accounts = {}
    checks = Checks()
    try:
        for item in items:
            verify_static_metadata(item, static[item["folder"]][item["job_id"]])
            md = metadata_columns(item); sid = item["scenario_id"]
            r, z = returns[sid - 1], inflation[sid - 1]
            gs = groups[item["folder"]][item["trace_scope"]]
            mg = check_group(item, gs, item["model_group"], model=True)
            M = extract(archives[item["folder"], item["trace_scope"]], item["model_group"], mg["scenario_ids"], sid)
            checks.close(M["targets"], item["target"], "source/own_target", atol=0, rtol=0)
            if item["B_group"] is None:
                B = reconstruct_B(r, z, item["target"], loss=p["loss_tolerance"], cost=p["cost_rate"])
                B_source = "new_same_own_target_annual_then_pending_production_fee_reference"
            else:
                bg = check_group(item, gs, item["B_group"], model=False)
                B = extract(archives[item["folder"], item["trace_scope"]], item["B_group"], bg["scenario_ids"], sid)
                checks.close(B["targets"], item["target"], "source/B_own_target", atol=0, rtol=0)
                B_source = "frozen_saved_trace"
            ma = prepare_account(M, r, z, item["target"], loss=p["loss_tolerance"], cost=p["cost_rate"], is_model=True,
                                 annual_price=PRICES[item["selected_candidate_order"]], fan_n=item["training_fan_n"], checks=checks)
            ba = prepare_account(B, r, z, item["target"], loss=p["loss_tolerance"], cost=p["cost_rate"], checks=checks)
            if item["B_group"] is None:
                verify_inner_B_saved_metrics(item, ba, references, checks)
            model_accounts[item["pair_id"]] = ma
            for name, account, source in (("model", ma, "frozen_saved_trace"), ("B", ba, B_source)):
                arm_md = {**md, "arm": name, "account_source": source}
                endpoints.extend({**arm_md, **row} for row in endpoint_records(account, loss=p["loss_tolerance"]))
                annual.extend({**arm_md, **row} for row in annual_records(account))
            pr, ps, ws = paired_records(ma, ba, r, z, loss=p["loss_tolerance"], checks=checks)
            paired.extend({**md, **row} for row in pr)
            summaries.append({**md, **ps})
            witnesses.extend({**md, **row} for row in ws)
            availability.append({**md, "model_group": item["model_group"], "B_group": item["B_group"],
                                 "model_source": "frozen_saved_trace", "B_source": B_source,
                                 "model_first_crossing": ma["first_crossing_value"], "B_first_crossing": ba["first_crossing_value"]})
    finally:
        for archive in archives.values():
            archive.close()
    if (len(endpoints), len(annual), len(paired), len(summaries)) != (682, 660, 330, 11):
        raise ValueError("Complete paired path inventory differs")
    by_pair = {item["pair_id"]: item for item in items}
    cross_stage = []
    for sid in (35, 112, 52, 98):
        fm, im = by_pair[f"full_{sid}"], by_pair[f"inner_{sid}"]
        cross_stage.append(dict(scenario_id=sid, full_pair_id=fm["pair_id"], inner_pair_id=im["pair_id"],
            full_job_id=fm["job_id"], inner_job_id=im["job_id"],
            full_target=json.dumps(fm["target"], separators=(",", ":")), inner_target=json.dumps(im["target"], separators=(",", ":")),
            full_training_ids=json.dumps(fm["training_ids"], separators=(",", ":")), inner_training_ids=json.dumps(im["training_ids"], separators=(",", ":")),
            full_training_fan_n=fm["training_fan_n"], inner_training_fan_n=im["training_fan_n"],
            full_candidate_order=fm["selected_candidate_order"], inner_candidate_order=im["selected_candidate_order"],
            full_annual_price=PRICES[fm["selected_candidate_order"]], inner_annual_price=PRICES[im["selected_candidate_order"]],
            target_changed=fm["target"] != im["target"], fan_changed=fm["training_ids"] != im["training_ids"],
            price_changed=fm["selected_candidate_order"] != im["selected_candidate_order"],
            full_evaluation_in_training=True, inner_evaluation_in_training=False,
            **cross_stage_summary(model_accounts[fm["pair_id"]], model_accounts[im["pair_id"]])))
    after = snapshot(p, protocol_digest)
    if before != after:
        raise RuntimeError("Frozen input or source changed during diagnosis")
    OUT.mkdir(parents=True, exist_ok=False)
    tables = {"account_endpoints": endpoints, "annual_states": annual, "paired_annual": paired,
              "paired_summary": summaries, "crossing_witnesses": witnesses, "cross_stage_summary": cross_stage}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUT / (name + ".csv.gz"), index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
    writej(OUT / "paired_summary.json", summaries)
    writej(OUT / "cross_stage_summary.json", cross_stage)
    writej(OUT / "source_trace_presence.json", availability)
    writej(OUT / "consistency_checks.json", dict(status="passed", checks_passed=checks.count,
           maximum_absolute_errors=checks.maximum_absolute_errors, scope="Fixed saved-account arithmetic and copied Q score algebra; not new fan calibration"))
    outputs = {path.name: sha(path) for path in sorted(OUT.iterdir())}
    writej(OUT / "manifest.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started, protocol_sha256=protocol_digest,
        sources=before["execution_sources_sha256"], sources_at_end=after["execution_sources_sha256"],
        input_sha256=before["frozen_input_sha256"], input_sha256_at_end=after["frozen_input_sha256"],
        outputs_sha256=outputs, row_counts={name: len(rows) for name, rows in tables.items()},
        development_array_sha256=digest.hexdigest(), original_market_scenario_ids_loaded=list(map(int, ids)),
        final_scenarios_loaded=False, new_model_simulations=0, new_fits=0, new_forecast_evaluations=0,
        new_reference_simulations=4, saved_model_accounts=11, saved_B_accounts=7,
        selection_changed=False, admission_changed=False, current_C_service_financials_added=False,
        interpretation="Post-hoc paired accounting, not a causal identification, probability calibration or fresh held-out trial"))
    print(json.dumps(dict(status="complete", manifest_sha256=sha(OUT / "manifest.json"), row_counts={name: len(rows) for name, rows in tables.items()}, checks_passed=checks.count)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run only after protocol, source and synthetic tests are frozen")
    run()
