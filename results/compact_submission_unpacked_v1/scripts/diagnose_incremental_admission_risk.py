"""Run the predeclared read-only incremental-admission forensic ledger.

Reads only saved audited development artifacts. It never loads returns,
inflation, raw scenario tables, simulators, model fitters, or selectors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from diagnostics.incremental_admission_risk import (
    ConsistencyChecks, distribution_summary, select_event_witnesses, summarize_saved_path,
)

BASE = ROOT / "results/model_review/joint_liquidity_refit_v1"
OUT = ROOT / "results/model_review/incremental_admission_risk_v1"
PROTOCOL = ROOT / "docs/incremental_admission_risk_diagnostic_protocol.json"
PRIMARY = "joint_tie_set_refit_learner"
CANONICAL = "joint_canonical_learner"
REFERENCES = ("annual_then_pending", "equal_thirds_annual")
POLICIES = (PRIMARY, *REFERENCES)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path): return json.loads(Path(path).read_text())
def writej(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_snapshot(protocol):
    paths = [ROOT / name for name in protocol["source_sha256"]]
    paths.extend([PROTOCOL, Path(__file__).resolve(), ROOT / "research/diagnostics/incremental_admission_risk.py",
                  ROOT / "tests/test_incremental_admission_risk.py"])
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def admission_inventory(rows):
    """Preserve every original OOF ID; no financial imputation."""
    out = []
    for sid in range(1, 121):
        primary = rows.loc[rows.policy.eq(PRIMARY) & rows.scenario_id.eq(sid)]
        canonical = rows.loc[rows.policy.eq(CANONICAL) & rows.scenario_id.eq(sid)]
        if len(primary) != 1 or len(canonical) != 1:
            raise ValueError("One original-ID row per learner required")
        p, c = primary.iloc[0], canonical.iloc[0]
        if p.context_id != c.context_id:
            raise ValueError("Same-ID learners came from different contexts")
        auto_p, auto_c = p.execution_status == "automated", c.execution_status == "automated"
        if auto_c and not auto_p:
            raise ValueError("Unexpected lost-canonical cohort; protocol requires revision before calculation")
        partition = "incremental_admission" if auto_p and not auto_c else "common_admission" if auto_p and auto_c else "neither_admitted"
        out.append(dict(scenario_id=sid, context=p.context, context_id=int(p.context_id), cohort=partition,
                        primary_execution_status=p.execution_status, canonical_execution_status=c.execution_status,
                        primary_automated=auto_p, canonical_automated=auto_c))
    frame = pd.DataFrame(out)
    counts = frame.cohort.value_counts().to_dict()
    if counts != {"common_admission": 48, "neither_admitted": 48, "incremental_admission": 24}:
        raise ValueError("Predeclared cohort sizes changed")
    if frame.loc[frame.cohort.eq("incremental_admission"), "context"].unique().tolist() != ["outer_fold_2"]:
        raise ValueError("Predeclared newly admitted outer-fold-2 identity changed")
    return frame


def pair_records(path_records, annual_records):
    paths = pd.DataFrame(path_records); annual = pd.DataFrame(annual_records)
    out = []
    metrics = ("log_growth", "cagr", "terminal_real", "mdd", "excess", "nominal_fees", "nominal_moved",
               "actual_trade_count", "target_execution_count", "pending_decision_count", "max_end_year_tv_drift", "max_end_year_equity", "min_end_year_bill")
    for sid in sorted(paths.scenario_id.unique()):
        p = paths.loc[paths.scenario_id.eq(sid) & paths.policy.eq(PRIMARY)].iloc[0]
        pa = annual.loc[annual.scenario_id.eq(sid) & annual.policy.eq(PRIMARY)].sort_values("year")
        for reference in REFERENCES:
            r = paths.loc[paths.scenario_id.eq(sid) & paths.policy.eq(reference)].iloc[0]
            ra = annual.loc[annual.scenario_id.eq(sid) & annual.policy.eq(reference)].sort_values("year")
            if pa.year.tolist() != ra.year.tolist() or len(pa) != 30:
                raise ValueError("Paired original-year identity differs")
            different = pa.action.to_numpy() != ra.action.to_numpy()
            diverged = np.flatnonzero(different)
            pre_equity_delta = pa.pretrade_equity.to_numpy() - ra.pretrade_equity.to_numpy()
            pre_bill_delta = pa.pretrade_bill.to_numpy() - ra.pretrade_bill.to_numpy()
            out.append(dict(scenario_id=int(sid), context=str(p.context), primary_policy=PRIMARY, reference_policy=reference,
                            primary_breach=bool(p.breach), reference_breach=bool(r.breach),
                            primary_first_crossing=None if pd.isna(p.first_crossing) else int(p.first_crossing),
                            reference_first_crossing=None if pd.isna(r.first_crossing) else int(r.first_crossing),
                            first_action_difference_year=None if not len(diverged) else int(diverged[0]) + 1,
                            action_difference_n=int(np.sum(different)),
                            mean_pretrade_equity_delta=float(np.mean(pre_equity_delta)),
                            max_pretrade_equity_delta=float(np.max(pre_equity_delta)),
                            min_pretrade_bill_delta=float(np.min(pre_bill_delta)),
                            **{"delta_" + field: float(p[field] - r[field]) for field in metrics}))
    return out


def paired_summary(records):
    rows = []
    for reference in REFERENCES:
        d = [r for r in records if r["reference_policy"] == reference]
        rows.append(dict(primary_policy=PRIMARY, reference_policy=reference, n=len(d),
                         both_breach_n=sum(x["primary_breach"] and x["reference_breach"] for x in d),
                         primary_only_breach_n=sum(x["primary_breach"] and not x["reference_breach"] for x in d),
                         reference_only_breach_n=sum(not x["primary_breach"] and x["reference_breach"] for x in d),
                         neither_breach_n=sum(not x["primary_breach"] and not x["reference_breach"] for x in d),
                         **{"mean_" + field: float(np.mean([x[field] for x in d]))
                            for field in d[0] if field.startswith("delta_")},
                         mdd_increase_n=sum(x["delta_mdd"] > 1e-12 for x in d),
                         log_growth_increase_n=sum(x["delta_log_growth"] > 1e-12 for x in d),
                         lower_fees_n=sum(x["delta_nominal_fees"] < -1e-12 for x in d),
                         any_action_difference_path_n=sum(x["action_difference_n"] > 0 for x in d)))
    return rows


def run():
    protocol = readj(PROTOCOL)
    for name, expected in protocol["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    start = source_snapshot(protocol)
    if readj(BASE / "independent_audit.json")["status"] != "passed":
        raise ValueError("Prior numerical audit must pass before forensic extraction")
    manifest = readj(BASE / "manifest.json")
    if manifest["final_scenarios_loaded"] or manifest["scenario_ids_loaded"] != list(range(1, 121)):
        raise ValueError("Only development IDs 1-120 are permitted")
    for name, expected in manifest["outputs_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Saved stage output no longer matches manifest: " + name)
    rows = pd.read_csv(BASE / "client_paths.csv")
    rows = rows.loc[rows.scope.eq("outer_oof")].copy()
    # pandas cannot infer the mixed observed/NULL boolean breach field; convert
    # only literal saved True/False values and leave missing records unavailable.
    rows["breach"] = rows.breach.map(lambda x: x if isinstance(x, (bool, np.bool_)) or pd.isna(x) else {"True": True, "False": False}[x])
    inv = admission_inventory(rows)
    rejected = rows.loc[rows.policy.isin([PRIMARY, CANONICAL]) & ~rows.execution_status.eq("automated")]
    financial = ["log_growth", "cagr", "mdd", "terminal_real", "nominal_fees", "nominal_moved", "trade_fraction", "excess", "breach", "first_crossing"]
    if not rejected[financial].isna().all().all():
        raise ValueError("Unavailable learner financial outcomes were imputed")
    groups = readj(BASE / "trace_groups.json")["actual"]
    checks = ConsistencyChecks(); incremental = []; common = []; annual = []
    with np.load(BASE / "actual_traces.npz", allow_pickle=False) as archive:
        for record in inv.to_dict("records"):
            if record["cohort"] == "neither_admitted":
                continue
            sid, cid = record["scenario_id"], record["context_id"]
            cpi_by_policy = []
            for policy in POLICIES:
                key = f"c{cid}__{policy}"; meta = groups[key]
                i = meta["scenario_ids"].index(sid)
                arm = {name[len(key) + 2:]: archive[name] for name in archive.files if name.startswith(key + "__")}
                table = rows.loc[rows.policy.eq(policy) & rows.scenario_id.eq(sid)]
                if len(table) != 1 or not bool(table.iloc[0].assigned):
                    raise ValueError("Missing observed same-ID saved arm")
                if int(table.iloc[0].horizon) != 30 or table.iloc[0].loss_tolerance != .4:
                    raise ValueError("Original C H30/L.4 scope changed")
                path, years = summarize_saved_path(arm, i, table.iloc[0], policy=policy, checks=checks)
                cpi_by_policy.append(arm["cpi"][i])
                identity = dict(scenario_id=int(sid), context=record["context"], context_id=int(cid),
                                client="C", horizon=30, loss_tolerance=.4, policy=policy,
                                hypothetical_reference=policy in REFERENCES, cohort=record["cohort"])
                if record["cohort"] == "incremental_admission":
                    incremental.append({**identity, **path})
                    annual.extend({**identity, **year} for year in years)
                else:
                    common.append({**identity, **path})
            checks.close(cpi_by_policy[0], cpi_by_policy[1], "same_original_ID/CPI_primary_B", atol=0, rtol=0)
            checks.close(cpi_by_policy[0], cpi_by_policy[2], "same_original_ID/CPI_primary_EQ", atol=0, rtol=0)
    if len(incremental) != 72 or len(annual) != 2160 or len(common) != 144:
        raise ValueError("Complete three-arm cohort inventory differs")
    witnesses = select_event_witnesses(incremental)
    witness_annual = [row for row in annual if row["scenario_id"] in witnesses["selected_scenario_ids"]]
    pairs = pair_records(incremental, annual)
    summaries = [{"cohort": "incremental_admission", "policy": p, **distribution_summary([r for r in incremental if r["policy"] == p])} for p in POLICIES]
    common_summaries = [{"cohort": "common_admission", "context": context, "policy": p,
                         **distribution_summary([r for r in common if r["policy"] == p and (context == "pooled_common_48" or r["context"] == context)])}
                        for context in ("pooled_common_48", "outer_fold_1", "outer_fold_5") for p in POLICIES]
    ps = paired_summary(pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"admission_partition": inv, "incremental_paths": pd.DataFrame(incremental),
              "incremental_paired_paths": pd.DataFrame(pairs), "incremental_summary": pd.DataFrame(summaries),
              "incremental_paired_summary": pd.DataFrame(ps), "incremental_annual_states": pd.DataFrame(annual),
              "event_witness_annual_states": pd.DataFrame(witness_annual), "common_context_summary": pd.DataFrame(common_summaries)}
    for name, frame in tables.items(): frame.to_csv(OUT / (name + ".csv"), index=False, float_format="%.17g")
    writej(OUT / "consistency_checks.json", dict(status="passed", checks_passed=checks.count,
           saved_accounts_checked=216, annual_decisions_checked=6480,
           maximum_absolute_errors=checks.maximum_absolute_errors,
           independent_scope="Saved-trace accounting identities only; prior separate scalar replay audit supplies raw-return and gate checks",
           unavailable_learner_rows_checked_for_NULL=int(len(rejected))))
    findings = dict(status="completed_read_only_diagnostic", client="C", horizon=30, loss_tolerance=.4,
                    admission_counts=inv.cohort.value_counts().to_dict(), incremental_scenario_ids=inv.loc[inv.cohort.eq("incremental_admission"), "scenario_id"].tolist(),
                    incremental_summary=summaries, paired_summary=ps, witness_selection=witnesses,
                    first_crossing_records=[{k: r[k] for k in ("scenario_id", "policy", "first_crossing", "mdd", "mdd_peak_year", "mdd_trough_year", "added_mdd_after_crossing", "recovery_endpoint", "right_censored_recovery")}
                                            for r in incremental if r["breach"]],
                    interpretation_limits=protocol["interpretation_limits"], selection_changed=False,
                    new_fits=0, new_simulations=0, raw_market_paths_loaded=False, final_scenarios_loaded=False)
    writej(OUT / "findings.json", findings)
    end = source_snapshot(protocol)
    if start != end: raise ValueError("Source changed during diagnostic calculation")
    outputs = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}
    writej(OUT / "manifest.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
           protocol_sha256=sha(PROTOCOL), sources_and_artifacts_at_start=start, sources_and_artifacts_at_end=end,
           outputs_sha256=outputs, row_counts={name: len(frame) for name, frame in tables.items()},
           no_fitting_or_policy_selection=True, raw_market_paths_loaded=False, final_scenarios_loaded=False,
           descriptive_market_path_units_only=True))
    print(json.dumps(dict(status="passed", manifest_sha256=sha(OUT / "manifest.json"), row_counts={name: len(frame) for name, frame in tables.items()},
                         incremental_summary=summaries, paired_summary=ps, witnesses=witnesses), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run: parser.error("Use --run after the predeclared protocol and synthetic tests are fixed")
    run()
