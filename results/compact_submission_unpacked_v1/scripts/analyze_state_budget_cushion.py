"""Saved-only development analysis: matched cohorts and one-step forecasts.

No market parser or policy/forecast engine is imported.  This program describes
one frozen saved experiment; it never chooses a new policy or alters admission.
Its explicit analysis freeze must pass before saved numerical arrays are read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/state_budget_cushion_analysis_protocol.json"
FREEZE = ROOT / "docs/state_budget_cushion_analysis_source_freeze.json"
REFERENCES = ("same_target_B", "same_target_annual", "exact_thirds_annual")
LABELS = ("uniform", "shrink_conditional", "id_block_1", "id_block_2", "id_block_3")
PAIRED_METRICS = ("log_growth", "cagr", "mdd", "nominal_fees", "trade_fraction", "automatic_fraction")
POLICY_FIELDS = ("job_id", "scenario_id", "period", "phase", "forecast", "predicted_log_growth",
    "realized_log_growth", "growth_error", "predicted_paired_benefit", "realized_paired_benefit",
    "benefit_error", "predicted_next_breach", "actual_next_breach", "brier", "decision_paired_score",
    "predicted_positive_realized_nonpositive", "directional_support_miss", "supported_safe_floor_failure", "decision_source", "score_ranks_action", "cushion_scale")
INTERVENTION_METRICS = (*PAIRED_METRICS, "support_review")
FINANCIAL = ("log_growth", "cagr", "mdd", "volatility", "terminal_real", "nominal_fees",
             "trade_fraction", "breach", "first_crossing", "automatic_fraction",
             "protection_trade_fraction", "support_review", "mitigation_trade_fraction", "mitigation_nominal_fees")
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite output must be explicitly NULL, not silently cleaned")
    return value


def writej(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def quantile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(p * len(values)) - 1)] if values else None


def median(values):
    values = sorted(values)
    n = len(values)
    return ((values[(n - 1) // 2] + values[n // 2]) / 2) if n else None


def cvar90(values):
    values = sorted(values, reverse=True)
    if not values:
        return None
    mass = .1 * len(values)
    return math.fsum(min(max(mass - i, 0.), 1.) * value for i, value in enumerate(values)) / mass


def ids_map(rows):
    result = {}
    for row in rows:
        sid = row["scenario_id"]
        if isinstance(sid, bool) or not isinstance(sid, (int, np.integer)) or not 1 <= sid <= 120 or sid in result:
            raise ValueError("Require unique integer original development IDs1-120")
        result[int(sid)] = row
        if not isinstance(row["assigned"], (bool, np.bool_)):
            raise ValueError("Assigned must be a genuine boolean")
        if not row["assigned"]:
            if any(row.get(k) is not None for k in FINANCIAL):
                raise ValueError("Initial noninvestment financial fields must all be NULL")
        else:
            for key in (*PAIRED_METRICS, "volatility"):
                if not math.isfinite(row[key]):
                    raise ValueError("Invested rows require finite complete-H metrics")
    return result


def cohort_summary(rows):
    rows = list(rows)
    ids_map(rows)
    funded = [r for r in rows if r["assigned"]]
    n = len(funded)
    result = dict(original_n=len(rows), financial_n=n, initial_noninvested_n=len(rows)-n,
        not_balanced_n=sum(not r["assigned"] and r.get("review_reason") == "not_balanced_case" for r in rows),
        q1_review_n=sum(not r["assigned"] and r.get("review_reason") == "q1_review" for r in rows),
        execution_review_n=sum(not r["assigned"] and r.get("review_reason") not in ("not_balanced_case","q1_review") for r in rows),
        initial_review_n=sum(not r["assigned"] and r.get("review_reason") != "not_balanced_case" for r in rows),
        initial_coverage=n/len(rows) if rows else None,
        funded_support_referral_n=sum(bool(r.get("support_review")) for r in funded))
    result.update({key: None for key in ("breach_count", "breach_fraction", "mean_log_growth", "median_cagr",
        "p10_cagr", "mean_mdd", "median_mdd", "p90_mdd", "cvar90_mdd", "mean_nominal_fees",
        "mean_trade_fraction", "mean_automatic_fraction")})
    if n:
        mdd = [r["mdd"] for r in funded]
        result.update(breach_count=sum(r["breach"] for r in funded),
            breach_fraction=sum(r["breach"] for r in funded)/n,
            mean_log_growth=mean(r["log_growth"] for r in funded),
            median_cagr=median(r["cagr"] for r in funded), p10_cagr=quantile([r["cagr"] for r in funded], .1),
            mean_mdd=mean(mdd), median_mdd=median(mdd), p90_mdd=quantile(mdd, .9), cvar90_mdd=cvar90(mdd),
            mean_nominal_fees=mean(r["nominal_fees"] for r in funded),
            mean_trade_fraction=mean(r["trade_fraction"] for r in funded),
            mean_automatic_fraction=mean(r["automatic_fraction"] for r in funded))
    return result


def compare_cohorts(model_rows, reference_rows):
    """All-original reference plus exact same-invested-ID pairing, including NULL."""
    model, ref = ids_map(model_rows), ids_map(reference_rows)
    if set(model) != set(ref):
        raise ValueError("Comparison original cohorts differ")
    funded = sorted(sid for sid, row in model.items() if row["assigned"])
    if any(not ref[sid]["assigned"] for sid in funded):
        raise ValueError("An invested policy lacks its same-ID reference")
    pairs = [dict(scenario_id=sid, **{key: model[sid][key]-ref[sid][key] for key in PAIRED_METRICS}) for sid in funded]
    partition = {key: [] for key in ("neither", "model_only", "reference_only", "both")}
    for sid in funded:
        a, b = bool(model[sid]["breach"]), bool(ref[sid]["breach"])
        partition["both" if a and b else "model_only" if a else "reference_only" if b else "neither"].append(sid)
    differences = {}
    for key in PAIRED_METRICS:
        values = [r[key] for r in pairs]
        differences[key] = dict(mean=mean(values), median=median(values),
            positive_n=sum(v > TOL for v in values), negative_n=sum(v < -TOL for v in values),
            numerical_tie_n=sum(abs(v) <= TOL for v in values))
    return dict(model_original=cohort_summary(model.values()), reference_all_original=cohort_summary(ref.values()),
        matched_model=cohort_summary([model[sid] for sid in funded]),
        matched_reference=cohort_summary([ref[sid] for sid in funded]),
        matched_ids=funded, initial_noninvested_ids=sorted(set(model)-set(funded)),
        paired_differences=differences, event_id_partition=partition), pairs


def reconstruct_joint_market(arm):
    """Invert the positive three-asset annual reference, not a raw market file."""
    post = np.asarray(arm["posttrade_holdings"], float)
    after = np.asarray(arm["holdings"], float)[:, 1:]
    cpi = np.asarray(arm["cpi"], float)
    if (post.shape != after.shape or post.ndim != 3 or post.shape[-1] != 3
            or cpi.shape != (post.shape[0], post.shape[1]+1)
            or np.any(post <= 0) or not np.isfinite(post).all()
            or np.any(cpi <= 0) or not np.isfinite(cpi).all()):
        raise ValueError("Need positive three-asset post holdings and consistent CPI")
    inflation_gross = cpi[:, 1:]/cpi[:, :-1]
    gross = (after/post)/inflation_gross[:, :, None]
    if not np.isfinite(gross).all() or np.any(gross <= 0):
        raise ValueError("Cannot reconstruct positive real gross")
    return gross, inflation_gross-1


def forecast_market_rows(training_ids, actual_ids, market, inflation, *, job_id):
    """Own simple Gaussian calculation; one row per actual market/year, no policy copies."""
    if len(set(training_ids)) != len(training_ids) or len(set(actual_ids)) != len(actual_ids) or set(training_ids) & set(actual_ids):
        raise ValueError("OOF forecast requires unique disjoint actual/training IDs")
    if any(type(sid) is not int or not 1 <= sid <= 120 for sid in [*training_ids, *actual_ids]):
        raise ValueError("OOF forecast is development-only")
    train = np.asarray([market[sid] for sid in training_ids], float)
    train_pi = np.asarray([inflation[sid] for sid in training_ids], float)
    n, h, _ = train.shape
    rows = []
    for sid in actual_ids:
        for t in range(h):
            current = train[:, t]-1
            center = current.mean(axis=0)
            scale = np.maximum(current.std(axis=0, ddof=0), 1e-8)
            actual = np.asarray(market[sid][t])-1
            forecasts = {"uniform": center}
            fallback = False
            if t:
                prev = np.column_stack((train[:, t-1]-1, train_pi[:, t-1]))
                a = np.r_[np.asarray(market[sid][t-1])-1, inflation[sid][t-1]]
                prev_scale = np.maximum(prev.std(axis=0, ddof=0), 1e-8)
                with np.errstate(over="ignore", under="ignore"):
                    kernels = np.exp(-.5*np.mean(((prev-a)/prev_scale)**2, axis=1))
                total = float(kernels.sum())
                fallback = not math.isfinite(total) or total <= 0
                weights = np.full(n, 1/n) if fallback else .5/n + .5*kernels/total
                forecasts["shrink_conditional"] = weights @ current
            row = dict(job_id=job_id, scenario_id=sid, period=t+1,
                phase="inception" if t == 0 else "positive_period", conditional_uniform_fallback=fallback)
            for c, asset in enumerate(("equity", "bond", "bill")):
                row["actual_"+asset] = float(actual[c]); row["training_scale_"+asset] = float(scale[c])
            for label in ("uniform", "shrink_conditional"):
                pred = forecasts.get(label)
                row[label+"_standardized_mse"] = None if pred is None else float(np.mean(((pred-actual)/scale)**2))
                row[label+"_raw_mse"] = None if pred is None else float(np.mean((pred-actual)**2))
                for c, asset in enumerate(("equity", "bond", "bill")):
                    row[label+"_pred_"+asset] = None if pred is None else float(pred[c])
            rows.append(row)
    return rows


def policy_forecast_rows(arm, scenario_ids, *, job_id, L=.4):
    """One-step prediction checks on outer selected invested accounts only."""
    rows = []
    n, h = arm["gate_evaluated"].shape
    if len(scenario_ids) != n:
        raise ValueError("Policy trace IDs do not align")
    for i, sid in enumerate(scenario_ids):
        for t in range(h):
            if not arm["gate_evaluated"][i, t]:
                continue
            selected = int(arm["selected_index"][i, t]); k = max(selected, 0)
            source = str(arm['actual_decision_source'][i,t])
            cushion = float(arm["cushion_scale"][i,t])
            if not math.isfinite(cushion) or not 0 <= cushion <= 1:
                raise ValueError("Active cushion scale must be a finite diagnostic in[0,1]")
            mitigation = source == 'mitigation'
            if mitigation:
                k=int(arm['mitigation_selected_index'][i,t])
                if selected != -1 or not 0<=k<9 or arm['support_feasible'][i,t]:
                    raise ValueError('Failed primary and actual mitigation branches must stay separate')
            elif source != 'primary':
                raise ValueError('Only primary/mitigation active decisions may produce forecasts')
            members = int(arm["forecast_member_count"][i, t])
            if members != (1 if t == 0 else 5):
                raise ValueError("Unexpected forecast members/calendar alignment")
            cp = float(arm["cpi"][i, t]); x = arm["holdings"][i, t]
            post = arm["posttrade_holdings"][i, t]
            gross = arm["realized_real_gross"][i, t]
            before = float(x.sum()/cp)
            actual_next = float(gross @ post/cp); hold_next = float(gross @ x/cp)
            if not math.isclose(actual_next, float(arm["real_wealth"][i,t+1]), abs_tol=1e-9, rel_tol=1e-11):
                raise ValueError("Saved actual real gross/account mismatch")
            realized_growth = math.log(actual_next/before); realized_benefit = math.log(actual_next/hold_next)
            reason = str(arm["gate_reason"][i,t])
            phase = "inception" if t == 0 else "referral" if selected < 0 else "protection" if reason == "risk_budget_protection" else "growth" if arm["trades"][i,t] else "hold"
            if mitigation:
                phase='mitigation_trade' if arm['mitigation_trade'][i,t] else 'mitigation_hold'
            minimum = float(arm['mitigation_worst_next_real_selected'][i,t]) if mitigation else float(arm["budget_floor"][i,t]+arm["candidate_min_support_slack"][i,t,k])
            tolerance = TOL*before
            support_miss = actual_next < minimum-tolerance
            floor_failure = bool(arm["support_feasible"][i,t]) and actual_next < float(arm["budget_floor"][i,t])-tolerance
            actual_event = float(arm["running_mdd"][i,t+1])-L > TOL
            for m in range(members):
                prefix='mitigation_candidate_member_' if mitigation else 'candidate_member_'
                pred_growth = float(arm[prefix+'log_growth'][i,t,k,m])
                pred_benefit = float(arm[prefix+'advantage'][i,t,k,m])
                probability = float(arm['mitigation_candidate_member_predicted_next_breach' if mitigation else 'candidate_predicted_next_breach'][i,t,k,m])
                if not all(math.isfinite(v) for v in (pred_growth,pred_benefit,probability)) or not -TOL <= probability <= 1+TOL:
                    raise ValueError("Active prediction fields must be finite")
                rows.append(dict(job_id=job_id,scenario_id=int(sid),period=t+1,phase=phase,
                    forecast=LABELS[m],predicted_log_growth=pred_growth,realized_log_growth=realized_growth,
                    growth_error=pred_growth-realized_growth, predicted_paired_benefit=pred_benefit,
                    realized_paired_benefit=realized_benefit,benefit_error=pred_benefit-realized_benefit,
                    predicted_next_breach=probability,actual_next_breach=actual_event,
                    brier=(probability-float(actual_event))**2,
                    decision_paired_score=float(arm["selected_score"][i,t]),
                    predicted_positive_realized_nonpositive=pred_benefit>TOL and realized_benefit<=TOL,
                    directional_support_miss=support_miss,supported_safe_floor_failure=floor_failure,
                    decision_source=source,score_ranks_action=not mitigation,cushion_scale=cushion))
    return rows


def scenario_equal(rows, field):
    grouped = {}
    for row in rows:
        if row.get(field) is not None:
            grouped.setdefault(row["scenario_id"], []).append(float(row[field]))
    return mean(mean(values) for values in grouped.values())


def prediction_summaries(market_rows, policy_rows):
    """No pooled inference from annual rows or the five ensemble members."""
    market_summary = []
    for phase in ("inception", "positive_period"):
        subset = [r for r in market_rows if r["phase"] == phase]
        for label in ("uniform", "shrink_conditional"):
            usable = [r for r in subset if r[label+"_standardized_mse"] is not None]
            market_summary.append(dict(phase=phase,forecast=label,rows=len(usable),
                original_scenarios=len({r["scenario_id"] for r in usable}),
                scenario_equal_standardized_mse=scenario_equal(usable,label+"_standardized_mse"),
                scenario_equal_raw_mse=scenario_equal(usable,label+"_raw_mse")))
    policy_summary, bins = [], []
    for label in LABELS:
        for phase in ("inception", "growth", "protection", "hold", "referral", "mitigation_trade", "mitigation_hold"):
            subset = [r for r in policy_rows if r["forecast"] == label and r["phase"] == phase]
            group = dict(forecast=label,phase=phase,rows=len(subset),original_scenarios=len({r["scenario_id"] for r in subset}))
            group.update(growth_bias=scenario_equal(subset,"growth_error"),benefit_bias=scenario_equal(subset,"benefit_error"),
                brier=scenario_equal(subset,"brier"),predicted_positive_realized_nonpositive_fraction=scenario_equal(subset,"predicted_positive_realized_nonpositive"),
                directional_support_miss_fraction=scenario_equal(subset,"directional_support_miss"),
                supported_safe_floor_failure_fraction=scenario_equal(subset,"supported_safe_floor_failure"))
            for key in ("growth", "benefit"):
                squared = [dict(r,squared=r[key+"_error"]**2) for r in subset]
                value = scenario_equal(squared,"squared")
                group[key+"_rmse"] = None if value is None else math.sqrt(value)
            policy_summary.append(group)
            for b in range(3):
                selected = [r for r in subset if min(2,int(max(0.,r["predicted_next_breach"])*3)) == b]
                bins.append(dict(forecast=label,phase=phase,bin=b,rows=len(selected),
                    original_scenarios=len({r["scenario_id"] for r in selected}),
                    scenario_equal_predicted=scenario_equal(selected,"predicted_next_breach"),
                    scenario_equal_actual=scenario_equal(selected,"actual_next_breach")))
    # One decision per market clock, not five copies of a policy's ensemble.
    decisions = {}
    for row in policy_rows:
        key = (row["job_id"],row["scenario_id"],row["period"])
        if key not in decisions:
            decisions[key] = dict(row,decision_error=row["decision_paired_score"]-row["realized_paired_benefit"])
    decision_summary = []
    for phase in ("inception", "growth", "protection", "hold", "referral", "mitigation_trade", "mitigation_hold"):
        subset = [r for r in decisions.values() if r["phase"] == phase]
        squared = [dict(r,squared=r["decision_error"]**2) for r in subset]
        mse = scenario_equal(squared,"squared")
        decision_summary.append(dict(phase=phase,rows=len(subset),original_scenarios=len({r["scenario_id"] for r in subset}),
            paired_score_bias=scenario_equal(subset,"decision_error"),paired_score_rmse=None if mse is None else math.sqrt(mse),
            mean_cushion_scale=scenario_equal(subset,"cushion_scale"),
            minimum_cushion_scale=min((r["cushion_scale"] for r in subset),default=None),
            maximum_cushion_scale=max((r["cushion_scale"] for r in subset),default=None),
            directional_support_miss_fraction=scenario_equal(subset,"directional_support_miss"),
            supported_safe_floor_failure_fraction=scenario_equal(subset,"supported_safe_floor_failure")))
    return dict(market=market_summary,policy=policy_summary,decision=decision_summary,reliability_bins=bins,
        interpretation="Scenario-equal descriptive OOF diagnostics; no conditional calibration, probability guarantee or independent ensemble samples. Mitigation pairedscore is an annotation, not its maxmin ranking objective. Pending has no fresh prediction. Cushion m is a proposal-state diagnostic, never an event probability.")


def read_path_rows(path):
    bools = {"assigned", "breach", "support_review"}
    ints = {"scenario_id", "context_id", "first_crossing"}
    numeric = set(FINANCIAL)-bools-ints
    with gzip.open(path,"rt",newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "": row[key] = None
            elif key in bools:
                if value not in ("True","False"): raise ValueError("Malformed boolean")
                row[key] = value == "True"
            elif key in ints: row[key] = int(float(value)) if key == "first_crossing" else int(value)
            elif key in numeric: row[key] = float(value)
    return rows


def arm_arrays(archive, group):
    return {name:archive[group["group_id"]+"__"+name] for name in group["array_names"]}


def write_csv(path, rows, fieldnames=None):
    keys = list(fieldnames) if fieldnames is not None else sorted({key for row in rows for key in row})
    with gzip.GzipFile(filename=str(path),mode="wb",mtime=0) as zipped:
        import io
        with io.TextIOWrapper(zipped,newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=keys)
            writer.writeheader();writer.writerows(rows)


def exact_initial_identity(current_directory,parent_directory):
    """All inner identities and first-year accounting exact; later rules may diverge."""
    current_directory,parent_directory=Path(current_directory),Path(parent_directory)
    current_groups=json.loads((current_directory/'trace_groups.json').read_text())
    parent_groups=json.loads((parent_directory/'trace_groups.json').read_text())
    def indexed(groups):
        selected=[g for g in groups if g['kind']=='inner_candidate']
        result={g['group_id']:g for g in selected}
        if len(result)!=len(selected):raise ValueError('Duplicate initial identity group')
        return result
    current,parent=indexed(current_groups),indexed(parent_groups)
    if set(current)!=set(parent):raise ValueError('Inner initial group inventory differs')
    counts=dict(groups=len(current),funded_paths=0,initial_null_paths=0,exact_array_slices=0)
    with np.load(current_directory/'traces.npz',allow_pickle=False) as ca,np.load(parent_directory/'traces.npz',allow_pickle=False) as pa:
        for key,group in sorted(current.items()):
            old=parent[key]
            for field in ('job_id','policy','scenario_ids','training_ids','target','candidate','status'):
                if group.get(field)!=old.get(field):raise ValueError('Fixed-candidate initial metadata differs: '+field)
            funded=bool(group['array_names'])
            if funded!=bool(old['array_names']):raise ValueError('Initial financial identity differs')
            if not funded:
                counts['initial_null_paths']+=len(group['scenario_ids']);continue
            a,b=arm_arrays(ca,group),arm_arrays(pa,old)
            n=len(group['scenario_ids']);counts['funded_paths']+=n
            for field in ('holdings','real_wealth','real_peak','running_mdd','cpi','posttrade_holdings','fees','moved','trades'):
                left,right=np.asarray(a[field]),np.asarray(b[field])
                if left.shape!=right.shape or left.dtype!=right.dtype or left.shape[0]!=n:
                    raise ValueError('Initial account dtype/shape differs: '+field)
                upto=2 if field in ('holdings','real_wealth','real_peak','running_mdd','cpi') else 1
                if left.ndim<2 or left.shape[1]<upto:raise ValueError('OriginalH missing first-year accounting')
                x,y=left[:,:upto],right[:,:upto]
                equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
                if not equal:raise ValueError('Initial/first-year exact account differs: '+field)
                counts['exact_array_slices']+=1
    return dict(status='passed',**counts,caution='Exact deterministic initial-year identity only. Later guards/actions/states may change. No pre-firstguard equality or population causal claim.')


def inner_intervention_contrasts(rows,parent_rows):
    """All context/tuple groups; fixed initial funding, full-H effects including guard changes."""
    def indexed(data):
        selected=[r for r in data if r['scope']=='inner_candidate']
        result={(r['job_id'],r['policy'],r['scenario_id']):r for r in selected}
        if len(result)!=len(selected):raise ValueError('Duplicated inner complete-policy contrast')
        return result
    current,parent=indexed(rows),indexed(parent_rows)
    if set(current)!=set(parent):raise ValueError('Candidate/job/ID sets differ from single-factor parent')
    groups={}
    for key,r in current.items():
        b=parent[key]
        if r['context_id']!=b['context_id']:raise ValueError('Context changed in fixed candidate')
        if r['assigned']!=b['assigned']:raise ValueError('The intervention cannot alter initial funding')
        groups.setdefault((r['context_id'],r['policy']),[]).append((r,b))
    output=[]
    for (context,candidate),all_pairs in sorted(groups.items()):
        all_pairs.sort(key=lambda pair:pair[0]['scenario_id'])
        if len({a['scenario_id'] for a,b in all_pairs})!=len(all_pairs):
            raise ValueError('Repeated market within a context/tuple')
        pairs=[(a,b) for a,b in all_pairs if a['assigned']]
        differences={}
        for metric in INTERVENTION_METRICS:
            values=[float(a[metric])-float(b[metric]) for a,b in pairs]
            differences[metric]=dict(mean=mean(values),median=median(values),positive_n=sum(v>TOL for v in values),negative_n=sum(v < -TOL for v in values),numerical_tie_n=sum(abs(v)<=TOL for v in values))
        event_partition={label:[] for label in ('neither','current_only','parent_only','both')}
        funding_partition={label:[] for label in ('both_funded','current_only','parent_only','both_review')}
        for a,b in all_pairs:
            funding_partition['both_funded' if a['assigned'] and b['assigned'] else 'current_only' if a['assigned'] else 'parent_only' if b['assigned'] else 'both_review'].append(a['scenario_id'])
        for a,b in pairs:
            event_partition['both' if a['breach'] and b['breach'] else 'current_only' if a['breach'] else 'parent_only' if b['breach'] else 'neither'].append(a['scenario_id'])
        output.append(dict(context_id=context,candidate_id=candidate,original_n=len(all_pairs),paired_financial_n=len(pairs),
            original_ids=[a['scenario_id'] for a,b in all_pairs],matched_funded_ids=[a['scenario_id'] for a,b in pairs],
            funding_partition=funding_partition,current=cohort_summary(a for a,b in all_pairs),parent=cohort_summary(b for a,b in all_pairs),differences=differences,event_id_partition=event_partition,
            caution='Within-context same initial-funded IDs and same fixed candidate. Guard timing and automatic continuity may change. Repeated contexts/candidates reuse markets and are never pooled as independent observations; full-H consequences include all pending exposures.'))
    return output


def selected_coverage_contrasts(rows,parent_rows):
    output=[]
    for scope in ('outer_selected','full_selected'):
        a=ids_map([r for r in rows if r['scope']==scope]);b=ids_map([r for r in parent_rows if r['scope']==scope])
        if set(a)!=set(b):raise ValueError('Selected original cohorts differ')
        partition={key:[] for key in ('both_funded','current_only','parent_only','both_review')}
        for sid in sorted(a):partition['both_funded' if a[sid]['assigned'] and b[sid]['assigned'] else 'current_only' if a[sid]['assigned'] else 'parent_only' if b[sid]['assigned'] else 'both_review'].append(sid)
        both=partition['both_funded']
        output.append(dict(scope=scope,coverage_partition=partition,current_all=cohort_summary(a.values()),parent_all=cohort_summary(b.values()),current_both=cohort_summary(a[s] for s in both),parent_both=cohort_summary(b[s] for s in both),
            caution='Selected candidates can differ after training-only re-selection; same-funded comparisons are descriptive policy comparisons, not isolated cushion effects. Review has no financial counterfactual.'))
    return output


def run(input_directory, output_directory):
    plan, freeze = json.loads(PLAN.read_text()), json.loads(FREEZE.read_text())
    if not freeze["execution_ready"] or freeze["protocol_sha256"] != sha(PLAN):
        raise RuntimeError("Analysis source/protocol must be frozen before numeric access")
    if plan["development_ids"] != list(range(1,121)) or plan["allow_final"] is not False:
        raise RuntimeError("Development-only analysis")
    if "scripts/analyze_state_budget_cushion.py" not in freeze["source_sha256"]:
        raise RuntimeError("Analysis executable must bind its own frozen source")
    for name,digest in freeze['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Analysis frozen input changed: '+name)
    for name, digest in freeze["source_sha256"].items():
        if sha(ROOT/name) != digest: raise RuntimeError("Analysis source changed: "+name)
    source, output = Path(input_directory).resolve(), Path(output_directory).resolve()
    if source!=(ROOT/plan["producer_directory"]).resolve():
        raise RuntimeError("Only the declared frozen producer directory may be analyzed")
    manifest = json.loads((source/"manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["new_final_reads"] != 0
            or manifest["final_scenarios_loaded"] is not False
            or manifest["protocol_sha256"] != plan["producer_protocol_sha256"]
            or manifest["source_freeze_sha256"] != plan["producer_source_freeze_sha256"]
            or manifest["sources_and_inputs_at_start"] != manifest["sources_and_inputs_at_end"]):
        raise RuntimeError("Producer completion/development/source identity failed")
    hashes = {name:sha(source/name) for name in manifest["outputs_sha256"]}
    if hashes != manifest["outputs_sha256"]: raise RuntimeError("Producer output bytes changed")
    if output.exists(): raise RuntimeError("Refuse overwriting any analysis attempt")
    output.mkdir(parents=True)
    snapshot = dict(source_manifest_sha256=sha(source/"manifest.json"),
        protocol_sha256=sha(PLAN),source_freeze_sha256=sha(FREEZE),
        analysis_source_sha256={name:sha(ROOT/name) for name in freeze["source_sha256"]},
        producer_outputs_at_start=hashes,analysis_input_sha256=freeze["input_sha256"],
        parent_manifest_sha256=sha(ROOT/'results/state_budget_mitigation_v1/manifest.json'),
        parent_path_rows_sha256=sha(ROOT/'results/state_budget_mitigation_v1/path_metrics.csv.gz'))
    writej(output/"manifest.json",dict(status="running",**snapshot))
    try:
        _analyze_numeric(source,output,manifest,hashes,snapshot)
    except BaseException as error:
        import traceback
        (output/"failure.txt").write_text(traceback.format_exc())
        writej(output/"manifest.json",dict(status="failed",error_type=type(error).__name__,
            error=str(error),failed_utc=datetime.now(timezone.utc).isoformat(),**snapshot))
        raise


def _analyze_numeric(source, output, manifest, hashes, snapshot):
    rows = read_path_rows(source/"path_metrics.csv.gz")
    groups = json.loads((source/"trace_groups.json").read_text())
    comparisons, pairs = [], []
    for scope, reference_scope in (("outer_selected","outer_reference"),("full_selected","full_training_reference")):
        model = [r for r in rows if r["scope"] == scope]
        if sorted(ids_map(model)) != list(range(1,121)): raise ValueError("Every original ID must remain once")
        for ref in REFERENCES:
            result, differences = compare_cohorts(model,[r for r in rows if r["scope"] == reference_scope and r["policy"] == ref])
            comparisons.append(dict(scope=scope,reference=ref,**result))
            pairs.extend(dict(scope=scope,reference=ref,**r) for r in differences)
    market, inflation, outer_groups, policy_rows = {}, {}, [], []
    with np.load(source/"traces.npz",allow_pickle=False) as archive:
        for group in groups:
            if (group["kind"] == "reference" and group["policy"] == "exact_thirds_annual"
                    and group["job_id"] != "c0_refit" and group["job_id"].endswith("_refit")):
                arm = arm_arrays(archive,group)
                gross, pi = reconstruct_joint_market(arm)
                for i,sid in enumerate(group["scenario_ids"]):
                    if sid in market or type(sid) is not int or not 1 <= sid <= 120: raise ValueError("Duplicate/nondevelopment market")
                    market[sid], inflation[sid] = gross[i], pi[i]
                outer_groups.append(group)
        if sorted(market) != list(range(1,121)) or len(outer_groups) != 5:
            raise ValueError("Require exactly five OOF reference groups covering original120")
        market_rows = []
        for group in outer_groups:
            market_rows += forecast_market_rows(group["training_ids"],group["scenario_ids"],market,inflation,job_id=group["job_id"])
        for group in groups:
            if group["kind"] == "outer_selected":
                if set(group["training_ids"]) & set(group["scenario_ids"]): raise ValueError("Selected policy is not OOF")
                if not group["array_names"]: continue
                arm = arm_arrays(archive,group)
                predicted_train = np.transpose(np.asarray([market[sid] for sid in group["training_ids"]]),(1,0,2))
                realized = np.asarray([market[sid] for sid in group["scenario_ids"]])
                if not np.allclose(arm["training_real_gross"],predicted_train,atol=1e-12,rtol=1e-11) or not np.allclose(arm["realized_real_gross"],realized,atol=1e-12,rtol=1e-11):
                    raise ValueError("Saved market trace and physical reference reconstruction differ")
                policy_rows += policy_forecast_rows(arm,group["scenario_ids"],job_id=group["job_id"])
    parent=ROOT/'results/state_budget_mitigation_v1'
    parent_manifest=json.loads((parent/'manifest.json').read_text())
    if parent_manifest['status']!='complete' or any(sha(parent/name)!=digest for name,digest in parent_manifest['outputs_sha256'].items()):
        raise ValueError('Immutable V2b parent changed')
    parent_rows=read_path_rows(parent/'path_metrics.csv.gz')
    initial_check=exact_initial_identity(source,parent)
    if initial_check['groups']!=144:raise ValueError('Require all144 inner initial-identity groups')
    writej(output/'intervention_initial_identity.json',initial_check)
    interventions=inner_intervention_contrasts(rows,parent_rows)
    if len(interventions)!=36 or {g["context_id"] for g in interventions}!=set(range(6)):
        raise ValueError("Require all36 context/tuple contrasts including all-NULL groups")
    coverage=selected_coverage_contrasts(rows,parent_rows)
    writej(output/'intervention_contrasts.json',interventions)
    writej(output/'selected_coverage_contrasts.json',coverage)
    writej(output/"cohort_comparisons.json",comparisons)
    write_csv(output/"paired_paths.csv.gz",pairs,("scope","reference","scenario_id",*PAIRED_METRICS))
    write_csv(output/"market_forecasts.csv.gz",market_rows)
    write_csv(output/"policy_forecasts.csv.gz",policy_rows,POLICY_FIELDS)
    writej(output/"prediction_summary.json",prediction_summaries(market_rows,policy_rows))
    if snapshot['parent_manifest_sha256']!=sha(ROOT/'results/state_budget_mitigation_v1/manifest.json') or snapshot['parent_path_rows_sha256']!=sha(ROOT/'results/state_budget_mitigation_v1/path_metrics.csv.gz'):
        raise RuntimeError('Immutable parent changed during analysis')
    if hashes != {name:sha(source/name) for name in hashes}: raise RuntimeError("Producer outputs changed during analysis")
    if snapshot["source_manifest_sha256"] != sha(source/"manifest.json"):
        raise RuntimeError("Producer manifest changed during analysis")
    if snapshot["protocol_sha256"] != sha(PLAN) or snapshot["source_freeze_sha256"] != sha(FREEZE):
        raise RuntimeError("Analysis protocol/freeze changed during execution")
    if snapshot['analysis_input_sha256']!={name:sha(ROOT/name) for name in snapshot['analysis_input_sha256']}:
        raise RuntimeError('Analysis input changed during execution')
    if snapshot["analysis_source_sha256"] != {name:sha(ROOT/name) for name in snapshot["analysis_source_sha256"]}:
        raise RuntimeError("Analysis source changed during execution")
    writej(output/"manifest.json",dict(status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
        **snapshot,producer_outputs_at_end=hashes,new_model_runs=0,new_raw_market_reads=0,
        final_scenarios_loaded=False,comparisons=len(comparisons),paired_rows=len(pairs),market_rows=len(market_rows),
        policy_forecast_rows=len(policy_rows),outputs_sha256={p.name:sha(p) for p in output.iterdir() if p.name != "manifest.json"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input",type=Path,default=ROOT/"results/state_budget_cushion_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"results/state_budget_cushion_analysis_v1")
    args = parser.parse_args()
    if args.execute: run(args.input,args.output)
    else: print(json.dumps(dict(development_only=True,saved_arrays_only=True,raw_market_reads=0,
        paired_metrics=PAIRED_METRICS,required_freeze=str(FREEZE)),indent=2))
