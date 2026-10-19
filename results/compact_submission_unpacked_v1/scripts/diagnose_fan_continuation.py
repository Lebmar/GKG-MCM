"""Read-only frozen-state fan deletion and continuation diagnostic.

No controller, fit, target or admission is changed.  Main refuses an unfrozen
protocol and an existing archive before any market values are loaded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from experimental_execution.fan_continuation_diagnostic import (TOL, fixed_choice,
    fan_deletions, continuation_gaps, integer)

PROTOCOL = ROOT / "docs/fan_continuation_diagnostic_protocol.json"
OUT = ROOT / "results/model_review/fan_continuation_diagnostic_v1"
OLD = "joint_tie_set_refit_learner"
PRICES = (0., .001, .003, .01, .03, .1, .3, 1., None)
PARENTS = ("joint_liquidity_refit_v1", "refit_stability_v1")


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


def writecsv(path, rows, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False,
        compression={"method": "gzip", "mtime": 0})


def safe_map(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty frozen SHA map is required")
    for rel, digest in mapping.items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Repository-relative SHA256 entries required")
    return {rel: sha(ROOT / rel) for rel in mapping}


def validate_protocol(p):
    """Metadata-only validation, before loading any market return values."""
    if p.get("execution_ready") is not True:
        raise RuntimeError("Diagnostic protocol and sources are not frozen")
    if (p.get("data", {}).get("scenario_ids") != list(range(1, 121))
            or p["data"].get("allow_final") is not False
            or p.get("original_H") != 30 or p.get("loss_tolerance") != .4 or p.get("cost_rate") != .0025
            or p.get("output_directory") != "results/model_review/fan_continuation_diagnostic_v1"):
        raise ValueError("Fixed original development data and accounting contract changed")
    slots = p.get("inventory", [])
    if len(slots) != 39 or p.get("expected_slots") != 39 or len({s["slot_id"] for s in slots}) != 39:
        raise ValueError("Exactly39 distinct prospective slots are required")
    groups = {}
    for s in slots:
        sid = integer(s["scenario_id"], "original scenario", 1)
        cid = integer(s["context_id"], "context")
        year = integer(s["year"], "decision year", 1)
        tr = s["training_ids"]
        if (sid > 120 or year not in (2, 16, 30) or s["folder"] not in PARENTS
                or s["group"] != f"c{cid}__{OLD}" or not isinstance(s["final_guard_admitted"], bool)
                or any(integer(x, "training ID", 1) > 120 for x in tr) or len(set(tr)) != len(tr)
                or len(tr) != (120 if cid == 0 else 96)
                or (sid in tr) != (cid == 0)
                or s["folder"] != PARENTS[0 if cid == 0 else 1]
                or s["repeat_id"] != (0 if cid == 0 else cid // 100)):
            raise ValueError("Frozen whole-path fan/actual identity changed")
        order = integer(s["selected_candidate_order"], "candidate order")
        if order >= len(PRICES) or not np.isfinite(s["selected_beta"]):
            raise ValueError("Fixed archived policy index and beta required")
        groups.setdefault((s["folder"], s["group"]), []).append(s)
    if len(groups) != 13:
        raise ValueError("Exactly13 archived old-auto fits required")
    for group in groups.values():
        if sorted(s["year"] for s in group) != [2, 16, 30]:
            raise ValueError("No replacement or extra witness dates allowed")
        if any({k: v for k, v in s.items() if k not in ("slot_id", "year")} !=
               {k: v for k, v in group[0].items() if k not in ("slot_id", "year")} for s in group):
            raise ValueError("One fixed scenario, state lineage and policy per archived group required")
    return slots


def snapshot(p, protocol_digest):
    if sha(PROTOCOL) != protocol_digest:
        raise RuntimeError("Protocol changed during diagnostic")
    sources = safe_map(p["execution_sources_sha256"])
    inputs = safe_map(p["frozen_input_sha256"])
    if sources != p["execution_sources_sha256"] or inputs != p["frozen_input_sha256"]:
        raise RuntimeError("Frozen source or input changed")
    required = {str(Path(__file__).resolve().relative_to(ROOT)),
        "research/experimental_execution/fan_continuation_diagnostic.py",
        "research/experimental_execution/fan_rollout.py",
        "research/experimental_execution/risk_priced_fan_rollout.py",
        "research/experimental_execution/fixed_state_counterfactual.py",
        "src/safeinvest/core.py", "src/safeinvest/data.py"}
    if not required.issubset(sources):
        raise RuntimeError("Execution source closure is incomplete")
    inputs[str(PROTOCOL.relative_to(ROOT))] = protocol_digest
    return sources, inputs


def near(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-12, atol=2e-12, equal_nan=True):
        raise AssertionError("Frozen-state identity mismatch: " + label)


def validate_archives(p, slots):
    """Archive lineage/audit metadata only; no new diagnostic numerical data."""
    archives = {}
    admissions = readj(ROOT / "results/model_review/global_policy_admission_v1/admission.json")
    amap = {r["context"]: r for r in admissions}
    oldauto = {r["context"] for r in admissions if r["original_execution_status"] == "automated"}
    if len(oldauto) != 13 or {s["context"] for s in slots} != oldauto:
        raise ValueError("Inventory must retain every and only the13 original old-auto fits")
    contexts = {c["name"]: c for c in readj(ROOT / "docs/refit_stability_protocol.json")["contexts"]}
    c0 = readj(ROOT / "results/model_review/joint_liquidity_refit_v1/static_refits.json")["c0_refit"]["context"]
    contexts[c0["name"]] = c0
    for folder in PARENTS:
        path = ROOT / "results/model_review" / folder
        m, audit = readj(path / "manifest.json"), readj(path / "independent_audit.json")
        if audit.get("status") != "passed" or m.get("final_scenarios_loaded") is not False:
            raise ValueError("Complete independently audited development parent required")
        binding = audit.get("parent_manifest_sha256", audit.get("sources_and_artifacts_at_start", {}).get(str((path / "manifest.json").relative_to(ROOT))))
        if binding != sha(path / "manifest.json"):
            raise ValueError("Parent audit/manifest binding changed")
        for name in ("actual_traces.npz", "trace_groups.json", "context_selections.json"):
            if m["outputs_sha256"][name] != sha(path / name):
                raise ValueError("Archived selected trace identity changed")
        archives[folder] = dict(groups=readj(path / "trace_groups.json")["actual"],
            choices=readj(path / "context_selections.json"),
            traces=np.load(path / "actual_traces.npz", allow_pickle=False), manifest=m)
    for s in slots:
        archive, adm, context = archives[s["folder"]], amap[s["context"]], contexts[s["context"]]
        g = archive["groups"][s["group"]]; meta = g["metadata"]
        old = archive["choices"][s["context"]]; choice = old["methods"][OLD]
        if (context["context_id"] != s["context_id"] or context["training_ids"] != s["training_ids"]
                or old["training_ids"] != s["training_ids"] or g["q_subset_scenario_id"] != s["scenario_id"]
                or g["q_subset_years"] != [2, 16, 30] or meta["context"] != s["context"]
                or meta["context_id"] != s["context_id"] or meta["horizon"] != 30 or meta["loss_tolerance"] != .4
                or meta["policy"] != OLD or meta["training_beta"] != s["selected_beta"]
                or g["selected_candidate_order"] != s["selected_candidate_order"]
                or choice["selected_candidate_order"] != s["selected_candidate_order"]
                or choice["selected_beta"] != s["selected_beta"] or choice["execution_status"] != "automated"
                or adm["final_automated"] != s["final_guard_admitted"]
                or adm["selected_beta"] != s["selected_beta"] or adm["selected_candidate_order"] != s["selected_candidate_order"]
                or g["scenario_ids"] != old["evaluation_ids"]):
            raise ValueError("Prospective slot and frozen selected-policy lineage differ")
    return archives


def diagnose_slot(s, archive, returns, inflation):
    from experimental_execution.fan_rollout import evaluate_actions
    from experimental_execution.risk_priced_fan_rollout import priced_choices
    from experimental_execution.fixed_state_counterfactual import continue_from_state
    group = s["group"]; g = archive["groups"][group]; z = archive["traces"]
    i, t = g["scenario_ids"].index(s["scenario_id"]), s["year"] - 1
    def array(key): return z[group + "__" + key]
    def state(key): return array(key)[i, t]
    target = array("targets"); eta = PRICES[s["selected_candidate_order"]]
    near(array("annual_prices"), [np.inf if eta is None else eta], "eta")
    near(array("loss_tolerances"), [.4], "original L")
    x, cp, pk, dd = state("holdings"), float(state("cpi")), float(state("real_peak")), float(state("running_mdd"))
    pending, action = bool(state("pending_at_decision")), str(state("actions"))
    if pending != bool(state("ever_breached")) or pending != (dd - .4 > TOL):
        raise ValueError("Saved absorbing pending state inconsistent")
    row = {k: v for k, v in s.items() if k != "training_ids"}
    row.update(scope="full_training_in_sample" if s["context_id"] == 0 else "outer_oof",
        state_status="pending_skip" if pending else "active", actual_action=action,
        actual_path_in_fan=s["scenario_id"] in s["training_ids"], fan_n=len(s["training_ids"]),
        eta_technical=eta, eta_kind="risk_first" if eta is None else "finite_price",
        holdings_equity=float(x[0]), holdings_bond=float(x[1]), holdings_bill=float(x[2]),
        cpi=cp, real_peak=pk, running_mdd=dd, current_real=float(x.sum()) / cp,
        target_equity=float(target[0]), target_bond=float(target[1]), target_bill=float(target[2]),
        saved_qg_hold=None, saved_qg_restore=None, saved_bad_hold=None, saved_bad_restore=None,
        saved_score=None, recomputed_qg_hold=None, recomputed_qg_restore=None,
        recomputed_bad_hold=None, recomputed_bad_restore=None, recomputed_score=None,
        recomputed_restore=None, delta_net_log_growth=None, delta_breach_probability=None,
        deletion_n=None, deletion_action_flips=None, self_deleted_action_flip=None)
    if pending:
        if action != "hold" or not np.isnan(state("qg_hold")) or int(state("forecast_n")) != -1:
            raise ValueError("Pending skip must preserve inactive diagnostics")
        return row, [], []
    ids = np.asarray(s["training_ids"], dtype=int)
    q = evaluate_actions(x, cp, pk, dd, False, returns[ids - 1], inflation[ids - 1],
        year_index=t, target=target, L=.4, cost_rate=.0025, horizon=30)
    logs = {a: q[a]["total_log_growth"][0] for a in ("hold", "restore")}
    bad = {a: q[a]["ever_breached"][0] for a in ("hold", "restore")}
    full, drops = fan_deletions(logs["hold"], logs["restore"], bad["hold"], bad["restore"], eta)
    qh, qr = float(logs["hold"].mean()), float(logs["restore"].mean())
    kh, kr = int(bad["hold"].sum()), int(bad["restore"].sum())
    frozen_action, frozen_score, _ = priced_choices(qh, qr, kh, kr, len(ids), np.inf if eta is None else eta, 30)
    if bool(frozen_action) != full["restore"] or action != ("restore" if full["restore"] else "hold"):
        raise AssertionError("Recomputed fullfan action differs from frozen selected action")
    near([state("qg_hold"), state("qg_restore")], [qh, qr], "saved net Qg")
    if (int(state("qbad_hold_count")), int(state("qbad_restore_count")), int(state("forecast_n"))) != (kh, kr, len(ids)):
        raise AssertionError("Saved fullfan event counts differ")
    near(state("predicted_score"), frozen_score, "saved score")
    row.update(saved_qg_hold=float(state("qg_hold")), saved_qg_restore=float(state("qg_restore")),
        saved_bad_hold=kh, saved_bad_restore=kr, saved_score=float(state("predicted_score")) if eta is not None else None,
        recomputed_qg_hold=qh, recomputed_qg_restore=qr, recomputed_bad_hold=kh,
        recomputed_bad_restore=kr, recomputed_score=full["score"], recomputed_restore=full["restore"],
        delta_net_log_growth=full["delta_net_log_growth"], delta_breach_probability=full["delta_breach_probability"],
        deletion_n=len(drops), deletion_action_flips=sum(d["action_flip"] for d in drops))
    members, deletion_rows = [], []
    for j, sid in enumerate(ids.tolist()):
        identity = dict(slot_id=s["slot_id"], training_scenario_id=sid, self_deleted=sid == s["scenario_id"])
        deletion_rows.append({**identity, **drops[j]})
        if identity["self_deleted"]: row["self_deleted_action_flip"] = drops[j]["action_flip"]
        members.append(dict(slot_id=s["slot_id"], member_index=j, training_scenario_id=sid,
            is_actual_source_path=sid == s["scenario_id"], hold_total_log_growth=float(logs["hold"][j]),
            restore_total_log_growth=float(logs["restore"][j]), hold_ever_breached=bool(bad["hold"][j]),
            restore_ever_breached=bool(bad["restore"][j]),
            delta_net_log_growth=float(logs["restore"][j] - logs["hold"][j]),
            delta_breach_event=int(bad["restore"][j]) - int(bad["hold"][j])))
    branches = {}
    for a in ("hold", "restore"):
        branches[a] = continue_from_state(x, cp, pk, dd, target, .4,
            returns[s["scenario_id"] - 1], inflation[s["scenario_id"] - 1],
            year_index=t, original_horizon=30, current_action=a, cost_rate=.0025)
        b = branches[a]
        for key, value in dict(total_log_growth=b["total_log_growth"], ever_breached=b["breach"],
            history_mdd=b["history_mdd"], nominal_fees=b["nominal_fees"], terminal_real=b["terminal_real"]).items():
            row["B_" + a + "_" + key] = value
        row["forecast_realized_" + a + "_log_residual"] = b["total_log_growth"] - (qh if a == "hold" else qr)
        row["forecast_realized_" + a + "_event_residual"] = int(b["breach"]) - (kh if a == "hold" else kr) / len(ids)
    chosen = branches[action]
    near(chosen["posttrade_holdings"][0], array("posttrade_holdings")[i, t], "same chosen current trade")
    near(chosen["fees"][0], float(state("fees")), "same chosen current fee")
    adaptive = dict(total_log_growth=float(np.log(array("real_wealth")[i, -1] / row["current_real"])),
        ever_breached=bool(array("ever_breached")[i, -1]), history_mdd=float(array("running_mdd")[i, -1]),
        nominal_fees=float(array("fees")[i, t:].sum()), terminal_real=float(array("real_wealth")[i, -1]))
    for key, value in adaptive.items(): row["adaptive_" + key] = value
    chosen_metrics = dict(total_log_growth=chosen["total_log_growth"], ever_breached=chosen["breach"],
        history_mdd=chosen["history_mdd"], nominal_fees=chosen["nominal_fees"], terminal_real=chosen["terminal_real"])
    row.update(continuation_gaps(adaptive, chosen_metrics, year=s["year"]))
    future = array("actions")[i, t + 1:]
    row.update(future_decisions=29-t, future_action_mismatches=int(np.sum(future != chosen["actions"][1:])),
        adaptive_future_restores=int(np.sum(future == "restore")),
        B_future_restores=int(np.sum(chosen["actions"][1:] == "restore")))
    return row, members, deletion_rows


def describe(values):
    values = [float(v) for v in values if v is not None]
    if not values: return dict(n=0, minimum=None, median=None, maximum=None, mean=None)
    return dict(n=len(values), minimum=min(values), median=float(np.median(values)),
                maximum=max(values), mean=float(np.mean(values)))


def summarize(states, members, deletions):
    scopes = []
    for scope in ("full_training_in_sample", "outer_oof"):
        for guard in (None, False, True):
            selected = [r for r in states if r["scope"] == scope and (guard is None or r["final_guard_admitted"] == guard)]
            active = [r for r in selected if r["state_status"] == "active"]
            slot_ids = {r["slot_id"] for r in selected}; ds = [d for d in deletions if d["slot_id"] in slot_ids]
            self_ds = [d for d in ds if d["self_deleted"]]
            scopes.append(dict(scope=scope, final_guard_admitted=guard,
                slots=len(selected), active_slots=len(active), pending_slots=len(selected)-len(active),
                fits=len({r["group"] for r in selected}), distinct_original_market_ids=sorted({r["scenario_id"] for r in selected}),
                active_distinct_original_market_ids=sorted({r["scenario_id"] for r in active}),
                active_distinct_original_market_n=len({r["scenario_id"] for r in active}),
                year30_active_control_count=sum(r["year"] == 30 for r in active),
                deletion_records=len(ds), deletion_action_flips=sum(d["action_flip"] for d in ds),
                slots_with_any_deletion_flip=sum(r["deletion_action_flips"] > 0 for r in active),
                self_deletion_records=len(self_ds), self_deletion_action_flips=sum(d["action_flip"] for d in self_ds),
                continuation_log_gap=describe([r["continuation_gap_total_log_growth"] for r in active]),
                continuation_mdd_gap=describe([r["continuation_gap_history_mdd"] for r in active]),
                continuation_fee_gap=describe([r["continuation_gap_nominal_fees"] for r in active]),
                continuation_event_gap=describe([r["continuation_gap_event"] for r in active]),
                future_action_mismatches=sum(r["future_action_mismatches"] for r in active)))
    return dict(slots=len(states), fits=len({r["group"] for r in states}),
        distinct_original_market_ids=sorted({r["scenario_id"] for r in states}),
        distinct_original_market_n=len({r["scenario_id"] for r in states}),
        guard_admitted_slots=sum(r["final_guard_admitted"] for r in states),
        active_slots=sum(r["state_status"] == "active" for r in states),
        active_distinct_original_market_ids=sorted({r["scenario_id"] for r in states if r["state_status"] == "active"}),
        active_distinct_original_market_n=len({r["scenario_id"] for r in states if r["state_status"] == "active"}),
        year30_active_control_count=sum(r["year"] == 30 and r["state_status"] == "active" for r in states),
        pending_slots=sum(r["state_status"] == "pending_skip" for r in states),
        branch_member_records=len(members), deletion_records=len(deletions), groups=scopes,
        interpretation=["39 predeclared archived witness slots from13 fits and8 distinct original markets, not39 independent trials.",
            "Only3 full-training slots can contain their actual market suffix. Their fixed-state self deletions are not refits or learner LOOCV.",
            "OOF actual market IDs are absent from their own training fan. Full-training self inclusion remains resubstitution, not independent validation.",
            "Deletion records share states and original paths; no confidence interval or probability calibration is inferred.",
            "Continuation gaps compare identical current action then B to saved future adaptive decisions; unchosen-action adaptive Q is unavailable.",
            "Fees already enter net Q. No action, price, beta, target or admission is altered; current full-client Review remains."])


def main():
    started = time.perf_counter(); p = readj(PROTOCOL)
    slots = validate_protocol(p)
    if OUT.exists(): raise RuntimeError("Never overwrite an archived diagnostic")
    protocol_digest = sha(PROTOCOL); sources, inputs = snapshot(p, protocol_digest)
    archives = validate_archives(p, slots)
    from safeinvest.data import load_paths
    returns, inflation, ids = load_paths(ROOT / p["data"]["path"], range(1, 121), allow_final=False)
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    for archive in archives.values():
        if archive["manifest"]["development_arrays_sha256"] != digest.hexdigest():
            raise ValueError("Guarded development arrays differ from frozen parent")
    OUT.mkdir(parents=True)
    states, members, deletions = [], [], []
    writej(OUT / "manifest.json", dict(status="incomplete", protocol_sha256=protocol_digest))
    try:
        for s in slots:
            if time.perf_counter() - started > p["budget_minutes"] * 60:
                raise RuntimeError("Frozen diagnostic budget exceeded; preserve incomplete archive")
            if snapshot(p, protocol_digest) != (sources, inputs): raise RuntimeError("Start snapshot changed")
            state, branch_rows, drop_rows = diagnose_slot(s, archives[s["folder"]], returns, inflation)
            states.append(state); members.extend(branch_rows); deletions.extend(drop_rows)
            if snapshot(p, protocol_digest) != (sources, inputs): raise RuntimeError("End snapshot changed")
        inventory = [{**s, "training_ids": json.dumps(s["training_ids"], separators=(",", ":"))} for s in slots]
        writecsv(OUT / "state_inventory.csv.gz", inventory)
        writecsv(OUT / "state_diagnostics.csv.gz", states)
        writecsv(OUT / "branch_members.csv.gz", members)
        writecsv(OUT / "deletions.csv.gz", deletions)
        writej(OUT / "summary.json", summarize(states, members, deletions))
        sources_end, inputs_end = snapshot(p, protocol_digest)
        outputs = {f.name: sha(f) for f in OUT.iterdir() if f.is_file() and f.name != "manifest.json"}
        writej(OUT / "manifest.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter()-started, protocol_sha256=protocol_digest,
            sources=sources, sources_at_end=sources_end, input_sha256=inputs, input_sha256_at_end=inputs_end,
            outputs_sha256=outputs, scenario_ids_loaded=ids.tolist(), final_scenarios_loaded=False,
            development_arrays_sha256=digest.hexdigest(), new_fits=0, policy_actions_changed=0,
            row_counts=dict(state_inventory=len(inventory), state_diagnostics=len(states),
                branch_members=len(members), deletions=len(deletions)),
            diagnostic_scope="fixed archived states; no calibration, policy improvement or independent validation claim"))
    finally:
        for archive in archives.values(): archive["traces"].close()
    print(json.dumps(dict(status="complete", output=str(OUT), slots=len(states), active=sum(r["state_status"] == "active" for r in states))))


if __name__ == "__main__": main()
