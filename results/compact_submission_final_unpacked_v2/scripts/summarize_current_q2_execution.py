"""Extract current Q2 development evidence from frozen artifacts only.

No market loader, fit, forecast or execution simulator is imported. Run this
script before plot_current_q2_execution.py. An expanded learner is opt-in and
requires an independently audited completed artifact supplied by the root.
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
sys.path.insert(0, str(ROOT / 'research'))
from safeinvest_research.metrics import SUMMARY_METADATA, summarize_paths

PROTOCOL = ROOT / 'docs/q2_execution_development_protocol.json'
OUT = ROOT / 'results/model_review/q2_execution_development_v2'
PENDING = ROOT / 'results/model_review/review_pending_execution_v1'
PRICE = ROOT / 'results/model_review/risk_priced_fan_execution_v1'
EQUAL = ROOT / 'results/model_review/q2_oof_benchmark_v1'
FINITE = ROOT / 'results/model_review/finite_horizon_execution_v1'
STRICT = ROOT / 'results/model_review/positive_asset_specification_v1'
BASE_POLICIES = ['strict_annual', 'annual_then_pending', 'risk_priced_learner', 'equal_weight_annual']
METRICS = ['log_growth', 'cagr', 'mdd', 'volatility', 'terminal_real', 'nominal_fees', 'nominal_moved', 'trade_fraction']
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def verify_manifest(folder):
    """Verify saved outputs without opening data outside the named artifact."""
    manifest = read_json(folder / 'manifest.json')
    for name, expected in manifest['outputs_sha256'].items():
        if sha(folder / name) != expected:
            raise RuntimeError(f'Frozen artifact output hash mismatch: {folder.name}/{name}')
    return manifest


def require_audit(folder):
    audit = read_json(folder / 'independent_audit.json')
    if audit['status'] not in ['passed', 'pass', 'all_checks_passed']:
        raise RuntimeError(f'Independent audit is not passed: {folder}')
    return audit


def normalize(frame, policy, *, pending=False, equal=False, expanded=False):
    frame = frame.copy()
    frame['policy'] = policy
    if pending:
        frame = frame.rename(columns={'actual_trade_fraction': 'trade_fraction'})
    if equal:
        frame = frame.rename(columns={'fold': 'outer_fold'})
        frame['source_context'] = frame.context
        frame['context'] = 'outer_fold_' + frame.outer_fold.astype(int).astype(str)
        frame['assigned'] = True
        frame['assignment'] = 'EqualWeightReference'
        frame['risk_assignment'] = 'EqualWeightReference'
        frame['candidate_local_index'] = np.nan
        frame['candidate_original232_index'] = np.nan
    else:
        frame['source_context'] = frame.context
        if 'risk_assignment' not in frame:
            frame['risk_assignment'] = frame.assignment
    frame['scope'] = 'outer_oof'
    frame['reference_only'] = equal
    frame['same_initial_target_as_strict_annual'] = not equal
    if expanded:
        frame['parent_target_update_fraction_per30'] = frame.target_update_fraction
        frame['target_update_fraction'] = frame.target_update_count / 29
        if not np.allclose(frame.loc[frame.assigned,'target_update_count']/30, frame.loc[frame.assigned,'parent_target_update_fraction_per30'],atol=1e-13,rtol=0):
            raise RuntimeError('Expanded saved update count/fraction disagree')
    else:
        frame['target_update_count'] = np.where(frame.assigned, 0., np.nan)
        frame['target_update_fraction'] = np.where(frame.assigned, 0., np.nan)
    frame['target_update_opportunities'] = np.where(frame.assigned, 29., np.nan)
    return frame


def check_cohort(frame, keys):
    if len(frame) != 120 or frame.scenario_id.nunique() != 120 or sorted(frame.scenario_id) != list(range(1, 121)):
        raise RuntimeError('A policy must retain all original 120 scenario IDs once')
    mapping = frame[['scenario_id', 'outer_fold']].astype(int).sort_values('scenario_id').reset_index(drop=True)
    if not mapping.equals(keys):
        raise RuntimeError('The original held-out fold mapping changed')
    if not frame.groupby('outer_fold').size().eq(24).all():
        raise RuntimeError('Each original fold must contain 24 observations')
    auto = frame.loc[frame.assigned]
    if not np.isfinite(auto[METRICS]).all().all():
        raise RuntimeError('Automatic financial metrics must be finite')
    if not frame.loc[~frame.assigned, METRICS].isna().all().all():
        raise RuntimeError('Review financial metrics must remain NULL')
    if not np.allclose(auto.cagr, np.expm1(auto.log_growth), atol=1e-13, rtol=1e-12):
        raise RuntimeError('Saved CAGR/log growth identity failed')
    if not np.allclose(auto.trade_fraction * 30, np.rint(auto.trade_fraction * 30), atol=1e-11, rtol=0):
        raise RuntimeError('Trade fraction is not an integer actual trade count / H')


def summarize(frame):
    auto = frame.loc[frame.assigned]
    result = {'policy': str(frame.policy.iloc[0]), 'total_paths': len(frame), 'auto_n': len(auto),
              'review_n': len(frame) - len(auto), 'horizon': 30, 'loss_tolerance': .4,
              'reference_only': bool(frame.reference_only.iloc[0])}
    if not len(auto):
        return {**result, **{k: None for k in ['mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd','breach_count','breach_frequency_conditional_auto','breaches_per_original120','mean_nominal_fees','mean_nominal_moved','mean_trade_fraction','mean_target_update_fraction']}}
    stats = summarize_paths({k: auto[k].to_numpy() for k in ['log_growth','cagr','mdd','volatility','terminal_real']}, .4)
    breach = int((auto.mdd - .4 > TOL).sum())
    result.update({k: float(stats[k]) for k in ['mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd']})
    result.update({'breach_count': breach, 'breach_frequency_conditional_auto': breach / len(auto),
                   'breaches_per_original120': breach / 120, 'mean_nominal_fees': float(auto.nominal_fees.mean()),
                   'mean_nominal_moved': float(auto.nominal_moved.mean()), 'mean_trade_fraction': float(auto.trade_fraction.mean()),
                   'mean_target_update_fraction': float(auto.target_update_fraction.mean()),
                   'target_update_opportunities_per_auto_path': 29,
                   'actual_trade_count': int(np.rint((auto.trade_fraction * 30).sum())),
                   'target_update_count': int(auto.target_update_count.sum())})
    ordered_c = np.sort(auto.cagr.to_numpy()); ordered_d = np.sort(auto.mdd.to_numpy()); n = len(auto)
    for name, value in [('median_cagr', np.median(ordered_c)), ('p10_cagr', ordered_c[int(np.ceil(.1*n))-1]), ('median_mdd', np.median(ordered_d)), ('p90_mdd', ordered_d[int(np.ceil(.9*n))-1])]:
        if not np.isclose(result[name], value, atol=1e-13, rtol=0):
            raise RuntimeError(f'Independent pooled order statistic failed: {name}')
    if n == 120 and not np.isclose(result['cvar90_mdd'], ordered_d[-12:].mean(), atol=1e-13, rtol=0):
        raise RuntimeError('CVaR90 must equal mean of worst 12 for 120 equally weighted paths')
    return result


def compare_summary(result, saved, trade_field):
    error = 0.
    for k in ['mean_log_growth','median_cagr','p10_cagr','median_mdd','p90_mdd','cvar90_mdd','breach_count','mean_nominal_fees','mean_nominal_moved']:
        if not np.isclose(result[k], saved[k], atol=1e-12, rtol=1e-12):
            raise RuntimeError(f'Recomputed/source summary differs: {result["policy"]}/{k}')
        error = max(error, abs(float(result[k]) - float(saved[k])))
    if not np.isclose(result['mean_trade_fraction'], saved[trade_field], atol=1e-12, rtol=1e-12):
        raise RuntimeError('Recomputed/source mean trade fraction differs')
    return error


def episode_diagnostics(real):
    real = np.asarray(real)
    peak = np.maximum.accumulate(real)
    dd = 1 - real / peak
    running = np.maximum.accumulate(dd)
    crossings = np.flatnonzero(running - .4 > TOL)
    tau = int(crossings[0]) if len(crossings) else None
    trough = int(np.argmax(dd)); peak_t = int(np.argmax(real[:trough + 1]))
    recovery = None
    if tau is not None:
        found = np.flatnonzero(real[tau+1:] >= peak[tau])
        if len(found): recovery = int(tau + 1 + found[0])
    mdd_recovery_candidates = np.flatnonzero(real[trough+1:] >= real[peak_t])
    return {'real_peak': peak, 'current_drawdown': dd, 'running_mdd': running,
            'ever_breached': running - .4 > TOL,
            'summary': {'mdd': float(running[-1]), 'mdd_peak_year': peak_t, 'mdd_trough_year': trough,
                        'mdd_peak_real_wealth': float(real[peak_t]), 'mdd_trough_real_wealth': float(real[trough]),
                        'recovered_after_mdd_trough': bool(len(mdd_recovery_candidates)),
                        'mdd_peak_recovery_year': int(trough + 1 + mdd_recovery_candidates[0]) if len(mdd_recovery_candidates) else None,
                        'first_crossing': tau, 'ever_breached': tau is not None,
                        'trigger_peak_real_wealth': float(peak[tau]) if tau is not None else None,
                        'recovery_endpoint': recovery, 'recovery_time': recovery - tau if recovery is not None else None,
                        'right_censored': recovery is None if tau is not None else None,
                        'available_followup': 30 - tau if tau is not None else None,
                        'observed_followup': (recovery - tau if recovery is not None else 30 - tau) if tau is not None else None}}


def load_pending_episode(z, group, arm, sid):
    index = group['scenario_ids'].index(sid); prefix = f'{group["prefix"]}__{arm}__'
    fields = ['real_wealth','nominal_wealth','holdings','weights','posttrade_holdings','fees','moved','trades','actions','pending_at_decision','real_peak','current_drawdown','running_mdd','ever_breached']
    out = {k: z[prefix+k][index,0].copy() for k in fields}
    out['cpi'] = z[prefix+'cpi'][index].copy(); out['target_weights'] = np.broadcast_to(group['target'], (30,3)).copy()
    out['derived_fields'] = []; return out


def load_price_episode(z, context, sid, target):
    prefix = f'{context}__C__risk_priced_learner__'; ids = z[prefix+'scenario_ids'].tolist(); index = ids.index(sid)
    fields = ['real_wealth','nominal_wealth','holdings','weights','posttrade_holdings','fees','moved','trades','actions','pending_at_decision','real_peak','current_drawdown','running_mdd','ever_breached','cpi']
    out = {k: z[prefix+k][index].copy() for k in fields}
    out['target_weights'] = np.broadcast_to(target,(30,3)).copy(); out['derived_fields'] = []
    return out


def load_equal_episode(z, equal_rows, sid):
    row = equal_rows.loc[equal_rows.scenario_id == sid].iloc[0]
    local = equal_rows.loc[equal_rows.outer_fold == row.outer_fold].sort_values('scenario_id').scenario_id.tolist().index(sid)
    prefix = f'{row.source_context}__equal_weight_annual__'
    out = {k: z[prefix+k][local,0].copy() for k in ['real_wealth','nominal_wealth','fees','moved','trades']}
    endweights = z[prefix+'end_weights'][local,0]
    out['weights'] = np.vstack([[0.,0.,1.],endweights]); out['holdings'] = out['nominal_wealth'][:,None] * out['weights']
    out['cpi'] = out['nominal_wealth'] / out['real_wealth']; out['target_weights'] = np.full((30,3),1/3)
    out['posttrade_holdings'] = (out['nominal_wealth'][:-1]-out['fees'])[:,None] * out['target_weights']
    out['actions'] = np.full(30,'restore'); out['pending_at_decision'] = np.zeros(30,dtype=bool)
    out['derived_fields'] = ['cpi from saved nominal/real','holdings from saved nominal/endweights','posttrade holdings from saved wealth/fee and fixed exactthird target','restore intent from frozen annual protocol, actual trades remain saved flags']
    return out



def load_expanded_episode(z, context, sid, expected_matrix=None):
    prefix = f'{context}__C__expanded_target_learner__'
    ids = z[prefix+'scenario_ids'].tolist(); index = ids.index(sid)
    fields = ['real_wealth','nominal_wealth','holdings','weights','posttrade_holdings','fees','moved','trades','actions','pending_at_decision','real_peak','current_drawdown','running_mdd','ever_breached','cpi','current_target_index','current_target_before','current_target_after','target_updated']
    out = {k:z[prefix+k][index].copy() for k in fields}
    matrix = z[prefix+'target_matrix']; out['target_matrix'] = matrix.copy()
    if expected_matrix is not None and not np.allclose(matrix,expected_matrix,atol=1e-16,rtol=0): raise RuntimeError('Expanded saved target matrix differs from frozen fold menu')
    if not np.array_equal(out['current_target_before'],out['current_target_index'][:-1]) or not np.array_equal(out['current_target_after'],out['current_target_index'][1:]): raise RuntimeError('Expanded target before/after endpoint identity failed')
    out['target_weights'] = matrix[out['current_target_after']].copy()
    out['derived_fields'] = ['period declared target weights indexed from saved target_matrix and saved current_target_after']
    if out['target_updated'][0] or not np.array_equal(out['target_updated'][1:],np.diff(out['current_target_index'])[1:]!=0):
        raise RuntimeError('Expanded target update flags fail saved index identity')
    return out

def validate_episode(out, saved):
    real = out['real_wealth']; diagnostic = episode_diagnostics(real)
    if real.shape != (31,) or not np.isclose(real[0],100) or not np.all(real > 0):
        raise RuntimeError('Episode wealth has wrong horizon or inception')
    for k in ['real_peak','current_drawdown','running_mdd','ever_breached']:
        if k in out and not np.allclose(out[k],diagnostic[k],atol=1e-12,rtol=1e-12):
            raise RuntimeError(f'Saved episode state disagrees: {k}')
        out[k] = diagnostic[k]
    calculated = {'log_growth': np.log(real[-1]/100)/30, 'cagr': np.expm1(np.log(real[-1]/100)/30),
                  'mdd': diagnostic['summary']['mdd'], 'terminal_real': real[-1], 'nominal_fees': out['fees'].sum(),
                  'nominal_moved': out['moved'].sum(), 'trade_fraction': out['trades'].mean()}
    for k,v in calculated.items():
        if not np.isclose(v, saved[k], atol=1e-11,rtol=1e-12): raise RuntimeError(f'Episode/path CSV disagreement: {k}')
    if not np.allclose(out['holdings'].sum(axis=-1),out['nominal_wealth'],atol=1e-11,rtol=1e-12): raise RuntimeError('Holdings wealth identity failed')
    if not np.allclose(out['real_wealth']*out['cpi'],out['nominal_wealth'],atol=1e-11,rtol=1e-12): raise RuntimeError('CPI wealth identity failed')
    if not np.allclose(out['posttrade_holdings'].sum(axis=-1),out['nominal_wealth'][:-1]-out['fees'],atol=1e-11,rtol=1e-12): raise RuntimeError('Posttrade single fee identity failed')
    if 'target_updated' in out and int(out['target_updated'].sum()) != int(saved['target_update_count']): raise RuntimeError('Expanded NPZ/CSV target update count differs')
    return {**diagnostic['summary'], **{k:float(v) for k,v in calculated.items()},
            'target_update_count':int(out['target_updated'].sum()) if 'target_updated' in out else 0,
            'target_update_years_start':np.flatnonzero(out['target_updated']).astype(int).tolist() if 'target_updated' in out else [],
            'actual_trade_years_start': np.flatnonzero(out['trades']).astype(int).tolist(),
            'restore_decision_years_start': np.flatnonzero(out['actions']=='restore').astype(int).tolist(),
            'derived_fields': out['derived_fields']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-expanded',action='store_true',help='Include only the root-notified stable and independently audited expanded artifact')
    parser.add_argument('--expanded-dir',type=Path,default=None,help='Opt-in only after root stable notice and passed independent audit; no unselected OOF evaluation')
    args = parser.parse_args()
    if args.include_expanded and args.expanded_dir is None: args.expanded_dir = ROOT/'results/model_review/bounded_target_updates_v1'
    protocol = read_json(PROTOCOL)
    base_input_start = {p:sha(ROOT/p) for p in protocol['provenance']}
    if base_input_start != protocol['provenance']: raise RuntimeError('Protocol frozen dependency changed')
    input_start = dict(base_input_start)
    expanded_folder = args.expanded_dir.resolve() if args.expanded_dir is not None else None
    if expanded_folder is not None:
        if expanded_folder != ROOT/'results/model_review/bounded_target_updates_v1': raise RuntimeError('Only the root-declared expanded artifact is supported')
        require_audit(expanded_folder); verify_manifest(expanded_folder)
        input_start.update({str((expanded_folder/name).relative_to(ROOT)):sha(expanded_folder/name) for name in ['manifest.json','independent_audit.json','client_paths.csv','summary.csv','actual_traces.npz','context_selections.json']})
    sources = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),PROTOCOL,ROOT/'research/safeinvest_research/metrics.py']}
    for folder in [PENDING,PRICE,EQUAL,FINITE,STRICT]: verify_manifest(folder)
    require_audit(PENDING); require_audit(PRICE)
    pending = pd.read_csv(PENDING/'client_path_metrics.csv'); price = pd.read_csv(PRICE/'client_paths.csv'); equal = pd.read_csv(EQUAL/'oof_paths.csv')
    original = pending.loc[(pending.scope=='oof') & (pending.client=='C') & (pending.arm=='annual') & (pending.horizon==30)].copy()
    base = normalize(original,'strict_annual',pending=True)
    pend = normalize(pending.loc[(pending.scope=='oof') & (pending.client=='C') & (pending.arm=='pending_hold') & (pending.horizon==30)],'annual_then_pending',pending=True)
    priced = normalize(price.loc[(price.scope=='outer_oof') & (price.client=='C') & (price.policy=='risk_priced_learner') & (price.horizon==30)],'risk_priced_learner')
    eq = normalize(equal.loc[(equal.policy=='equal_weight_annual') & (equal.horizon==30)],'equal_weight_annual',equal=True)
    keys = base[['scenario_id','outer_fold']].astype(int).sort_values('scenario_id').reset_index(drop=True)
    groups = read_json(PENDING/'execution_groups.json'); refits = read_json(STRICT/'nested_context_refits.json'); mapping = read_json(STRICT/'candidate_mapping.json')
    group_by_context = {g['context']:g for g in groups if g['scope']=='oof' and g['client']=='C'}
    target_map = {}; target_matrices = {}
    for ctx,g in group_by_context.items():
        refit = refits[ctx]
        if refit['effective_assignments']['C'] != 'Balanced' or set(g['scenario_ids']) != set(refit['outer_test_ids']): raise RuntimeError('Frozen strict assignment/fold mismatch')
        p = next(p for p in refit['raw_fit']['menu'] if p['name']=='Balanced'); m = mapping[p['candidate_local_index']]
        if p['candidate_original232_index'] != m['candidate_original232_index'] or not np.allclose(p['weights'],g['target'],atol=1e-16,rtol=0): raise RuntimeError('Frozen local/global target mapping mismatch')
        target_map[ctx] = p
        target_matrices[ctx] = np.array([next(item['weights'] for item in refit['raw_fit']['menu'] if item['name']==role) for role in ['Conservative','Balanced','Growth']])
    frames = [base,pend,priced,eq]
    expanded = None; expanded_controller_choices = []
    if expanded_folder is not None:
        expanded_source = pd.read_csv(expanded_folder/'client_paths.csv')
        fixed_copy = expanded_source.loc[(expanded_source.scope=='outer_oof') & (expanded_source.policy=='fixed_target_learner')].sort_values('scenario_id')
        old = priced.sort_values('scenario_id')
        if len(fixed_copy)!=120 or not np.array_equal(fixed_copy.scenario_id,old.scenario_id) or not np.allclose(fixed_copy[METRICS],old[METRICS],atol=1e-11,rtol=1e-12,equal_nan=True): raise RuntimeError('Expanded study parent fixed learner differs')
        expanded = normalize(expanded_source.loc[(expanded_source.scope=='outer_oof') & (expanded_source.policy=='expanded_target_learner') & (expanded_source.client=='C') & (expanded_source.horizon==30)],'expanded_target_learner',expanded=True)
        frames.append(expanded)
        selections = read_json(expanded_folder/'context_selections.json')
        expanded_controller_choices = [{'context':ctx,'selected_controller_order':selection['selected_controller_order'],'selected_controller':selection['selected_controller']} for ctx,selection in selections.items()]
    for frame in frames:
        check_cohort(frame,keys)
        if not frame.reference_only.iloc[0]:
            for row in frame.itertuples():
                product = target_map[row.context]
                if row.risk_assignment != 'Balanced' or int(row.candidate_local_index) != product['candidate_local_index'] or int(row.candidate_original232_index) != product['candidate_original232_index']:
                    raise RuntimeError('Current execution policy changed frozen strict target')
            weights = np.array([target_map[ctx]['weights'] for ctx in frame.context])
            frame[['w_equity','w_bond','w_bill']] = weights
        elif not np.allclose(frame[['w_equity','w_bond','w_bill']],1/3,atol=1e-16,rtol=0): raise RuntimeError('Reference target is not exactthirds')
    # The duplicated priced reference must match the independent pending source on every common numeric path metric.
    prior = normalize(price.loc[(price.scope=='outer_oof') & (price.client=='C') & (price.policy=='annual_then_pending')],'annual_then_pending')
    a = pend.sort_values('scenario_id'); b = prior.sort_values('scenario_id')
    if not np.allclose(a[METRICS],b[METRICS],atol=1e-11,rtol=1e-12): raise RuntimeError('Duplicate pending reference changed')
    summaries = [summarize(f) for f in frames]
    pending_saved = pd.read_csv(PENDING/'profile_summary.csv'); price_saved = pd.read_csv(PRICE/'summary.csv'); equal_saved = pd.read_csv(EQUAL/'summary.csv')
    summary_errors = {}
    for result,arm in zip(summaries[:2],['annual','pending_hold']):
        saved = pending_saved.loc[(pending_saved.scope=='oof') & (pending_saved.client=='C') & (pending_saved.arm==arm) & (pending_saved.subset=='all_assigned')].iloc[0]
        summary_errors[result['policy']] = compare_summary(result,saved,'actual_trade_fraction')
    summary_errors['risk_priced_learner'] = compare_summary(summaries[2],price_saved.loc[(price_saved.scope=='outer_oof') & (price_saved.client=='C') & (price_saved.policy=='risk_priced_learner')].iloc[0],'mean_trade_fraction')
    summary_errors['equal_weight_annual'] = compare_summary(summaries[3],equal_saved.loc[equal_saved.policy=='equal_weight_annual'].iloc[0],'actual_trade_fraction')
    if expanded_folder is not None:
        expanded_saved = pd.read_csv(expanded_folder/'summary.csv')
        saved = expanded_saved.loc[(expanded_saved.scope=='outer_oof') & (expanded_saved.client=='C') & (expanded_saved.policy=='expanded_target_learner')].iloc[0]
        summary_errors['expanded_target_learner'] = compare_summary(summaries[4],saved,'mean_trade_fraction')
        auto = expanded.loc[expanded.assigned]
        if len(auto) and not np.isclose(auto.target_update_count.mean()/30,saved.mean_target_update_fraction,atol=1e-12,rtol=0): raise RuntimeError('Expanded saved update mean count/30 differs')
    fields = ['scope','context','source_context','outer_fold','scenario_id','policy','assigned','assignment','risk_assignment','reference_only','same_initial_target_as_strict_annual','candidate_local_index','candidate_original232_index','w_equity','w_bond','w_bill','horizon','loss_tolerance'] + METRICS + ['target_update_count','target_update_opportunities','target_update_fraction']
    paths = pd.concat([f[fields] for f in frames],ignore_index=True)
    paths['breach'] = (paths.mdd - .4 > TOL).astype(object)
    paths.loc[~paths.assigned,'breach'] = None
    pairs = []
    base_index = base.set_index('scenario_id')
    for frame in frames[1:]:
        for row in frame.itertuples():
            origin = base_index.loc[row.scenario_id]; common = bool(row.assigned and origin.assigned)
            pairs.append({'policy':row.policy,'scenario_id':int(row.scenario_id),'outer_fold':int(row.outer_fold),'common_auto':common,'reference_only':bool(row.reference_only),'same_initial_target':not bool(row.reference_only),
                          **{f'delta_{k}':float(getattr(row,k)-origin[k]) if common else None for k in METRICS+['target_update_count','target_update_fraction']},
                          'delta_breach':int(row.mdd-.4>TOL)-int(origin.mdd-.4>TOL) if common else None})
    # Prespecified annual-only selection, never recomputed using learner ranks.
    chosen = base.sort_values(['mdd','scenario_id'],ascending=[False,True]).iloc[0]
    sid,ctx,fold = int(chosen.scenario_id),str(chosen.context),int(chosen.outer_fold)
    group = group_by_context[ctx]; target = group['target']
    # Load only saved C OOF arrays. Validate all 480 path records, so the old
    # exactthird NPZ's sorted-within-fold row convention is checked, not assumed.
    with np.load(PENDING/'executed_paths.npz',allow_pickle=False) as z:
        pending_arrays = {k:z[k] for k in z.files if k.startswith('outer_fold_') and '__client_C__' in k}
    with np.load(PRICE/'actual_traces.npz',allow_pickle=False) as z:
        price_arrays = {k:z[k] for k in z.files if k.startswith('outer_fold_') and '__C__risk_priced_learner__' in k}
    with np.load(FINITE/'executed_wealth.npz',allow_pickle=False) as z:
        equal_arrays = {k:z[k] for k in z.files if k.startswith('fold_') and '__equal_weight_annual__' in k}
    expanded_arrays = {}
    if expanded_folder is not None:
        with np.load(expanded_folder/'actual_traces.npz',allow_pickle=False) as z:
            expanded_arrays = {k:z[k] for k in z.files if k.startswith('outer_fold_') and '__C__expanded_target_learner__' in k}
    all_trace_checks = 0
    for frame in frames:
        for _,row in frame.loc[frame.assigned].iterrows():
            path_group = group_by_context[row.context]; path_sid = int(row.scenario_id)
            if row.policy == 'strict_annual': arm = load_pending_episode(pending_arrays,path_group,'annual',path_sid)
            elif row.policy == 'annual_then_pending': arm = load_pending_episode(pending_arrays,path_group,'pending_hold',path_sid)
            elif row.policy == 'risk_priced_learner': arm = load_price_episode(price_arrays,row.context,path_sid,path_group['target'])
            elif row.policy == 'expanded_target_learner': arm = load_expanded_episode(expanded_arrays,row.context,path_sid,target_matrices[row.context])
            else: arm = load_equal_episode(equal_arrays,eq,path_sid)
            validate_episode(arm,row); all_trace_checks += 1
    episodes = {'strict_annual':load_pending_episode(pending_arrays,group,'annual',sid),
                'annual_then_pending':load_pending_episode(pending_arrays,group,'pending_hold',sid),
                'risk_priced_learner':load_price_episode(price_arrays,ctx,sid,target),
                'equal_weight_annual':load_equal_episode(equal_arrays,eq,sid)}
    if expanded is not None and bool(expanded.loc[expanded.scenario_id==sid,'assigned'].iloc[0]): episodes['expanded_target_learner'] = load_expanded_episode(expanded_arrays,ctx,sid,target_matrices[ctx])
    episode_summary, yearly, arrays = {},[],{}
    for policy,arm in episodes.items():
        row = paths.loc[(paths.policy==policy) & (paths.scenario_id==sid)].iloc[0]
        episode_summary[policy] = validate_episode(arm,row)
        for key,value in arm.items():
            if isinstance(value,np.ndarray): arrays[f'{policy}__{key}'] = value
        for t in range(31):
            current = {'policy':policy,'scenario_id':sid,'outer_fold':fold,'year_endpoint':t,'real_wealth':float(arm['real_wealth'][t]),'nominal_wealth':float(arm['nominal_wealth'][t]),'cpi':float(arm['cpi'][t]),'real_peak':float(arm['real_peak'][t]),'current_drawdown':float(arm['current_drawdown'][t]),'running_mdd':float(arm['running_mdd'][t]),'ever_breached':bool(arm['ever_breached'][t])}
            for j,name in enumerate(['equity','bond','bill']): current[f'actual_{name}_weight'] = float(arm['weights'][t,j]); current[f'nominal_{name}_holding'] = float(arm['holdings'][t,j])
            current.update({'current_target_index_at_endpoint':int(arm['current_target_index'][t]) if 'current_target_index' in arm else None,'target_index_before_next_period_decision':int(arm['current_target_before'][t]) if t<30 and 'current_target_before' in arm else None,'target_index_after_next_period_decision':int(arm['current_target_after'][t]) if t<30 and 'current_target_after' in arm else None})
            for j,name in enumerate(['equity','bond','bill']): current[f'target_{name}_weight_for_next_period'] = float(arm['target_weights'][t,j]) if t<30 else None
            current.update({'action_at_start_of_period_ending_here':str(arm['actions'][t-1]) if t else None,'actual_trade_in_period':bool(arm['trades'][t-1]) if t else None,'nominal_fee_in_period':float(arm['fees'][t-1]) if t else None,'nominal_moved_in_period':float(arm['moved'][t-1]) if t else None,'pending_at_period_start':bool(arm['pending_at_decision'][t-1]) if t else None,'target_updated_in_period':bool(arm['target_updated'][t-1]) if t and 'target_updated' in arm else (False if t else None),'year_end_TV_drift_from_declared_target':float(.5*np.abs(arm['weights'][t]-arm['target_weights'][t-1]).sum()) if t else None})
            yearly.append(current)
    explanation = 'Your drawdown has exceeded the agreed limit. Automatic rebalancing is paused, and your current assets are held while we request your review. Holding does not undo the loss or guarantee recovery; we will reassess your needs before changing the investment plan.'
    if len(explanation.split())>50: raise RuntimeError('Customer explanation exceeds 50 words')
    OUT.mkdir(parents=True,exist_ok=True)
    pairs_frame = pd.DataFrame(pairs)
    pair_summary = []
    for policy,group in pairs_frame.groupby('policy',sort=False):
        common = group.loc[group.common_auto]
        pair_summary.append({'policy':policy,'total_original_pairs':len(group),'common_auto_n':len(common),'reference_only':bool(group.reference_only.iloc[0]),'same_initial_target':bool(group.same_initial_target.iloc[0]),**{'mean_'+field:float(common[field].mean()) if len(common) else None for field in common if field.startswith('delta_')}})
    pd.DataFrame(pair_summary).to_csv(OUT/'paired_summary.csv',index=False)
    paths.to_csv(OUT/'oof_paths.csv',index=False); pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False); pairs_frame.to_csv(OUT/'paired_vs_strict_annual.csv',index=False); pd.DataFrame(yearly).to_csv(OUT/'episode_yearly.csv',index=False);np.savez_compressed(OUT/'episode_arrays.npz',**arrays)
    write_json(OUT/'episode.json',{'selection':protocol['episode_preselection'],'scenario_id':sid,'context':ctx,'outer_fold':fold,'horizon':30,'loss_tolerance':.4,'strict_target':target,'equal_target':[1/3]*3,'policies':episode_summary,'customer_review_explanation_candidate':explanation,'customer_explanation_words':len(explanation.split()),'caution':'Annual worst-case selection favors an annual severe episode; this single path cannot determine typical policy performance or choose a winner.'})
    write_json(OUT/'summary.json',{'status':'complete_five_audited_policies' if expanded_folder is not None else 'complete_current_four_audited_policies','expanded_target_learner':'included_audit_passed' if expanded_folder is not None else 'not_read; awaiting stable audited artifact','cohort':protocol['cohort'],'summaries':summaries,'expanded_training_choices':expanded_controller_choices,'expanded_interpretation':'No general claim that target updates are ineffective; these are the frozen family selections on reused development training contexts.','summary_metadata':SUMMARY_METADATA,'units':protocol['metrics'],'limitations':protocol['limitations'],'policy_choice_made':False,'raw_market_loaded':False,'final_scenarios_loaded':False})
    input_end={p:sha(ROOT/p) for p in input_start}; sources_end={p:sha(ROOT/p) for p in sources}
    if input_start!=input_end or sources!=sources_end: raise RuntimeError('Input/source changed during extraction')
    names=['oof_paths.csv','summary.csv','paired_vs_strict_annual.csv','paired_summary.csv','episode_yearly.csv','episode_arrays.npz','episode.json','summary.json']
    write_json(OUT/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_five_audited_policies' if expanded_folder is not None else 'complete_current_four_audited_policies','sources':sources,'sources_at_end':sources_end,'inputs_sha256':input_start,'inputs_sha256_at_end':input_end,'outputs_sha256':{n:sha(OUT/n) for n in names},'checks':{'source_saved_summary_max_abs_errors':summary_errors,'all_policies_same_original120_fold_mapping':True,'all_fixed_targets_strict_local_original232_verified':True,'exactthird_reference_different_target':True,'quantiles_pooled_before_order_statistics':True,'episode_recomputed_from_actual_frozen_arrays':True,'all_oof_auto_path_trace_checks':all_trace_checks,'exactthird_npz_row_order_confirmed_by_every_path_metrics':True,'episode_scenario_id':sid,'episode_fold':fold,'review_null_preserved':True},'market_read':False,'final_read':False,'final_scenarios_loaded':False,'refit':False,'policy_adoption_or_selection':False,'expanded_artifact_read':expanded_folder is not None})
    print(pd.DataFrame(summaries).to_string(index=False));print(f'Episode annual-only preselection: scenario {sid}, {ctx}; customer explanation {len(explanation.split())} words')


if __name__=='__main__': main()
