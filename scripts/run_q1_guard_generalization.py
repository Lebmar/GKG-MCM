"""Q1 guard generalization: locked V2b rule applied to profiles B and E.

Development-only resubstitution diagnostic. No final scenario is ever parsed
(the loader gate enforces allow_final=False). No refit, selection or parameter
change: the guard uses the locked release_0.5_uniform candidate identically to
the profile-C locked object, only with each profile's own target, horizon and
loss limit. The frozen protocol and source freeze must match before any market
value is loaded.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/q1_guard_generalization_protocol.json'
FREEZE = ROOT / 'docs/q1_guard_generalization_source_freeze.json'
DEFAULT_OUT = ROOT / 'results/q1_guard_generalization_v1'
DEV_IDS = list(range(1, 121))
TOL = 1e-12
COST = 0.0025

FINANCIAL = ('log_growth', 'cagr', 'mdd', 'volatility', 'terminal_real',
             'nominal_fees', 'trade_fraction', 'breach', 'first_crossing',
             'automatic_fraction', 'protection_trade_fraction', 'support_review',
             'mitigation_trade_fraction', 'mitigation_nominal_fees')
PAIR = ('log_growth', 'cagr', 'mdd', 'nominal_fees', 'trade_fraction', 'automatic_fraction')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def writej(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        with io.TextIOWrapper(stream, newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def freeze_gate():
    """Verify protocol/freeze consistency and all frozen hashes before market parsing."""
    if not FREEZE.is_file():
        raise RuntimeError('Source freeze does not exist; freeze sources before first numeric run')
    p = json.loads(PROTOCOL.read_text())
    f = json.loads(FREEZE.read_text())
    if not f.get('execution_ready') or f.get('protocol_sha256') != sha(PROTOCOL):
        raise RuntimeError('Freeze the protocol before any market parsing')
    if p['development_ids'] != DEV_IDS or p['allow_final'] is not False:
        raise RuntimeError('Development-only study contract broken')
    if p['locked_guard_spec']['candidate_id'] != 'release_0.5_uniform':
        raise RuntimeError('Only the locked V2b candidate may be reused')
    for key in ('source_sha256', 'input_sha256'):
        for name, digest in f[key].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError('Frozen dependency changed: ' + name)
    return p, f


def squeeze(arm):
    """Drop the single-target axis of a paired reference arm: (n,1,...)->(n,...)."""
    return {key: (value if key == 'cpi' else value[:, 0])
            for key, value in arm.items()
            if isinstance(value, np.ndarray) and key not in ('targets', 'loss_tolerances')
            and (key == 'cpi' or value.ndim >= 2)}


def rows_from_guard(obj, result, ids, metrics):
    """Per-path rows for a V2b guard arm, same conventions as the locked final evaluation."""
    common = dict(object_id=obj['object_id'], profile=obj['profile'], assignment=obj['assignment'],
                  horizon=obj['horizon'], L=obj['L'], rule=obj['rule'],
                  execution_status=result['status'])
    if not result['financial_available']:
        if result['arm'] is not None:
            raise ValueError('Review cannot invent a physical arm')
        return [dict(**common, scenario_id=s, assigned=False, review_reason=result['status'],
                     **dict.fromkeys(FINANCIAL)) for s in ids]
    arm = result['arm']
    h = obj['horizon']
    n = len(ids)
    if arm['real_wealth'].shape != (n, h + 1) or arm['fees'].shape != (n, h):
        raise ValueError('Own-horizon physical account shape required')
    values = metrics(arm['real_wealth'])
    rows = []
    for i, sid in enumerate(ids):
        pending = arm['pending_at_decision'][i]
        guard = arm.get('guard_triggered', np.zeros((n, h), bool))[i]
        reason = arm.get('gate_reason', np.full((n, h), '', dtype='U64'))[i]
        mit = reason == 'risk_mitigation_then_review'
        trade = arm['trades'][i]
        first = int(arm['first_crossing'][i])
        breach = bool(arm['ever_breached'][i, -1])
        if breach != (float(values['mdd'][i]) - obj['L'] > TOL):
            raise ValueError('Original L-event convention differs')
        rows.append(dict(**common, scenario_id=sid, assigned=True,
                         **{k: float(v[i]) for k, v in values.items()},
                         nominal_fees=float(arm['fees'][i].sum()),
                         trade_fraction=float(trade.mean()), breach=breach,
                         first_crossing=None if first < 0 else first,
                         automatic_fraction=float(1 - np.mean(pending | guard)),
                         protection_trade_fraction=float(np.mean((reason == 'risk_budget_protection') | (mit & trade))),
                         support_review=bool(np.any(arm.get('guard_pending', np.zeros((n, h + 1), bool))[i])),
                         mitigation_trade_fraction=float(np.mean(mit & trade)),
                         mitigation_nominal_fees=float(arm['fees'][i][mit].sum())))
    return rows


def rows_from_reference(obj, arm, ids, metrics):
    """Per-path rows for a static reference arm (annual / pending_hold)."""
    h = obj['horizon']
    n = len(ids)
    if arm['real_wealth'].shape != (n, h + 1):
        raise ValueError('Reference arm shape mismatch')
    values = metrics(arm['real_wealth'])
    rows = []
    for i, sid in enumerate(ids):
        pending = arm['pending_at_decision'][i]
        first = int(arm['first_crossing'][i])
        breach = bool(arm['ever_breached'][i, -1])
        if breach != (float(values['mdd'][i]) - obj['L'] > TOL):
            raise ValueError('Original L-event convention differs')
        rows.append(dict(object_id=obj['object_id'], profile=obj['profile'], assignment=obj['assignment'],
                         horizon=h, L=obj['L'], rule=obj['rule'], execution_status='funded',
                         scenario_id=sid, assigned=True,
                         **{k: float(v[i]) for k, v in values.items()},
                         nominal_fees=float(arm['fees'][i].sum()),
                         trade_fraction=float(arm['trades'][i].mean()), breach=breach,
                         first_crossing=None if first < 0 else first,
                         automatic_fraction=float(1 - np.mean(pending)),
                         protection_trade_fraction=0.0, support_review=False,
                         mitigation_trade_fraction=0.0, mitigation_nominal_fees=0.0))
    return rows


def summary(rows):
    """Identical definitions to the locked final evaluation summary."""
    if len({r['scenario_id'] for r in rows}) != len(rows):
        raise ValueError('Preserve original unique population')
    if len({r['object_id'] for r in rows}) != 1:
        raise ValueError('Summaries cannot pool objects/horizons')
    funded = [r for r in rows if r['assigned']]
    n = len(funded)
    result = {k: rows[0][k] for k in ('object_id', 'profile', 'assignment', 'horizon', 'L', 'rule')}
    result.update(original_n=len(rows), financial_n=n, initial_review_n=len(rows) - n,
                  initial_coverage=n / len(rows))
    fields = ('breach_count', 'breach_fraction', 'mean_log_growth', 'median_cagr', 'p10_cagr',
              'mean_mdd', 'median_mdd', 'p90_mdd', 'cvar90_mdd', 'mean_volatility',
              'mean_nominal_fees', 'mean_trade_fraction', 'mean_automatic_fraction',
              'funded_support_review_n', 'mean_protection_trade_fraction',
              'mean_mitigation_trade_fraction', 'mean_mitigation_nominal_fees')
    result.update(dict.fromkeys(fields))
    for r in rows:
        if not r['assigned'] and any(r[k] is not None for k in FINANCIAL):
            raise ValueError('Noninvestment finance must be NULL')
    if not n:
        return result
    avg = lambda k: math.fsum(float(r[k]) for r in funded) / n
    quant = lambda v, p: sorted(v)[max(0, math.ceil(p * len(v)) - 1)]
    losses = [r['mdd'] for r in funded]
    cagrs = [r['cagr'] for r in funded]
    mass = 0.1 * n
    tail = math.fsum(x * min(max(mass - i, 0.), 1.) for i, x in enumerate(sorted(losses, reverse=True))) / mass
    result.update(breach_count=sum(r['breach'] for r in funded),
                  breach_fraction=sum(r['breach'] for r in funded) / n,
                  mean_log_growth=avg('log_growth'), median_cagr=float(np.median(cagrs)),
                  p10_cagr=quant(cagrs, 0.1), mean_mdd=avg('mdd'), median_mdd=float(np.median(losses)),
                  p90_mdd=quant(losses, 0.9), cvar90_mdd=tail, mean_volatility=avg('volatility'),
                  mean_nominal_fees=avg('nominal_fees'), mean_trade_fraction=avg('trade_fraction'),
                  mean_automatic_fraction=avg('automatic_fraction'),
                  funded_support_review_n=sum(r['support_review'] for r in funded),
                  mean_protection_trade_fraction=avg('protection_trade_fraction'),
                  mean_mitigation_trade_fraction=avg('mitigation_trade_fraction'),
                  mean_mitigation_nominal_fees=avg('mitigation_nominal_fees'))
    return result


def compare(model_rows, reference_rows):
    """Paired same-ID comparison, same shape as the locked final evaluation."""
    a = {r['scenario_id']: r for r in model_rows}
    b = {r['scenario_id']: r for r in reference_rows}
    if set(a) != set(b) or len(a) != len(model_rows) or len(b) != len(reference_rows):
        raise ValueError('Complete same original-ID comparison required')
    ids = sorted(s for s in a if a[s]['assigned'])
    if any(not b[s]['assigned'] for s in ids):
        raise ValueError('Same-ID reference unavailable')
    paired = [dict(scenario_id=s, **{k: a[s][k] - b[s][k] for k in PAIR}) for s in ids]
    differences = {k: dict(mean=math.fsum(r[k] for r in paired) / len(ids) if ids else None,
                           median=float(np.median([r[k] for r in paired])) if ids else None,
                           positive_n=sum(r[k] > TOL for r in paired),
                           negative_n=sum(r[k] < -TOL for r in paired),
                           numerical_tie_n=sum(abs(r[k]) <= TOL for r in paired)) for k in PAIR}
    parts = {k: [] for k in ('neither', 'model_only', 'reference_only', 'both')}
    for s in ids:
        x, y = a[s]['breach'], b[s]['breach']
        parts['both' if x and y else 'model_only' if x else 'reference_only' if y else 'neither'].append(s)
    return dict(model=summary(model_rows), reference=summary(reference_rows),
                same_funded_ids=ids, paired_differences=differences, event_id_partition=parts), paired


def run(output=DEFAULT_OUT):
    p, f = freeze_gate()
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError('Refuse overwriting any existing attempt')
    output.mkdir(parents=True)
    manifest = dict(status='running', phase='before_market_parser',
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    protocol_sha256=sha(PROTOCOL), source_freeze_sha256=sha(FREEZE),
                    sources_and_inputs_at_start={**f['source_sha256'], **f['input_sha256']},
                    final_scenarios_loaded=False, market_value_access_started=False,
                    evaluation_scope='development_resubstitution_diagnostic',
                    historical_exposure=p['historical_exposure'])
    writej(output / 'manifest.json', manifest)
    try:
        sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research')]
        from safeinvest.data import load_paths
        from safeinvest_research.metrics import path_metrics
        from experimental_execution.review_pending import simulate_paired
        from experimental_execution.state_budget_mitigation import simulate_state_budget

        manifest.update(phase='load_development', market_value_access_started=True)
        writej(output / 'manifest.json', manifest)
        r, pi, ids = load_paths(ROOT / 'data/SafeInvest_Synthetic_Data.csv', DEV_IDS, allow_final=False)
        if ids.tolist() != DEV_IDS:
            raise RuntimeError('Development identity mismatch')
        manifest.update(development_ids_loaded=ids.tolist(), phase='simulate')
        writej(output / 'manifest.json', manifest)

        all_rows, summaries, comparisons, paired_rows = [], [], [], []
        arrays, groups, witnesses = {}, [], {}
        for obj in p['objects']:
            oid = obj['object_id']
            target = np.asarray(obj['target'], float)
            if target.shape != (3,) or np.any(target <= 0) or not np.isclose(target.sum(), 1, atol=TOL, rtol=0):
                raise ValueError('Strictly positive three-asset target required: ' + oid)
            H, L = obj['horizon'], obj['L']
            if obj['rule'] == 'mitigation':
                result = simulate_state_budget(r, pi, r, pi, target, L=L, horizon=H, cost_rate=COST,
                                               release_fraction=0.5, forecast_rule='uniform',
                                               training_ids=DEV_IDS, actual_ids=DEV_IDS,
                                               support_mode='calendar')
                rows = rows_from_guard(obj, result, DEV_IDS, path_metrics)
                witnesses[oid] = dict(status=result['status'], witness=result['initial_witness'])
                arm = result['arm']
            else:
                both = simulate_paired(r, pi, target, L, horizon=H, cost_rate=COST)
                arm = squeeze(both[obj['rule']])
                rows = rows_from_reference(obj, arm, DEV_IDS, path_metrics)
            all_rows.extend(rows)
            summaries.append(summary(rows))
            names = []
            if arm is not None:
                for key, value in arm.items():
                    if isinstance(value, np.ndarray):
                        arrays[oid + '__' + key] = value
                        names.append(key)
            groups.append(dict(object_id=oid, profile=obj['profile'], rule=obj['rule'],
                               horizon=H, L=L, target=target.tolist(), status=rows[0]['execution_status'],
                               scenario_ids=DEV_IDS, array_names=names))
            print(json.dumps(dict(completed=oid, status=rows[0]['execution_status'],
                                  breaches=sum(x['breach'] for x in rows if x['assigned']))), flush=True)

        for profile in ('B', 'E'):
            guard = [x for x in all_rows if x['object_id'] == profile + '_guard']
            for ref_name in (profile + '_same_target_annual', profile + '_exact_thirds_annual'):
                ref = [x for x in all_rows if x['object_id'] == ref_name]
                comp, paired = compare(guard, ref)
                comparisons.append(dict(profile=profile, model_object=profile + '_guard',
                                        reference_object=ref_name, **comp))
                paired_rows.extend(dict(profile=profile, reference_object=ref_name, **x) for x in paired)

        if len(all_rows) != 960 or len(summaries) != 8:
            raise RuntimeError('Complete object inventory required (8 arms x 120 paths)')
        write_csv(output / 'path_metrics.csv.gz', all_rows)
        write_csv(output / 'summary.csv.gz', summaries)
        write_csv(output / 'paired_paths.csv.gz', paired_rows)
        writej(output / 'comparisons.json', comparisons)
        writej(output / 'trace_groups.json', groups)
        writej(output / 'initial_witnesses.json', witnesses)
        np.savez_compressed(output / 'traces.npz', **arrays)
        manifest['sources_and_inputs_at_end'] = {name: sha(ROOT / name)
                                                 for name in manifest['sources_and_inputs_at_start']}
        if manifest['sources_and_inputs_at_end'] != manifest['sources_and_inputs_at_start']:
            raise RuntimeError('Frozen source/input changed during run')
        manifest.update(status='complete', phase='complete',
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        path_rows=len(all_rows), summary_rows=len(summaries),
                        trace_groups=len(groups), array_fields=len(arrays), new_final_reads=0,
                        model_history='Locked V2b reused unchanged for B/E; all locked C artifacts untouched.',
                        outputs_sha256={x.name: sha(x) for x in output.iterdir() if x.name != 'manifest.json'})
        writej(output / 'manifest.json', manifest)
    except BaseException as exc:
        (output / 'failure.txt').write_text(traceback.format_exc())
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                        failed_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False)
        writej(output / 'manifest.json', manifest)
        raise
    print(json.dumps(dict(status='complete', output=str(output),
                          manifest_sha256=sha(output / 'manifest.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.execute:
        run(args.output)
    else:
        print(json.dumps(dict(source_only=True, development_only=True, market_values_parsed=False,
                              protocol=str(PROTOCOL.relative_to(ROOT))), indent=2))
