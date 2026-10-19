"""Independent released-final accounting, field, identity, NULL and fixed-Q audit.

No preparation producer or production simulator/policy/metrics is imported.
Old independent fee/replay/scalar-fan and independent export arithmetic are
explicitly reused. Final values are opened only after complete release closure.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, math, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_review_pending_execution import independent_path as replay_reference
from audit_fan_rollout_execution import reference_actual, scalar_fan
from audit_final_evaluation_development_gate import (Checks, FIN, SUMMARY_FIN,
    independent_path as export_metrics, summary_expected, require_financial_NULL,
    source_gate_review)

PROTOCOL = ROOT / 'docs/final_evaluation_protocol.json'
PLAN = ROOT / 'docs/final_locked_evaluation_audit_plan.json'
FREEZE = ROOT / 'docs/final_locked_evaluation_audit_source_freeze.json'
RELEASE = ROOT / 'docs/final_evaluation_release.json'
DEV = ROOT / 'results/model_review/final_evaluation_development_gate_v1'
OUT = ROOT / 'results/final_locked_evaluation_v1'
ATTEMPT = ROOT / 'results/final_locked_audit_attempt_v1'
SCOPE = 'locked_final_evaluation_previously_exposed_not_blind'
TOL = 1e-12
SHADOW = 'unapproved_fixed_shadow'
COMMON = {'targets', 'loss_tolerances', 'annual_prices'}
Q_FLOAT = ('qg_hold', 'qg_restore', 'predicted_score', 'predicted_growth_advantage_restore')
Q_INT = ('qbad_hold_count', 'qbad_restore_count', 'forecast_n', 'predicted_breach_count_delta_restore')
Q_OTHER = ('gate_reason', 'restoration_predicted_growth_sacrifice')
BASE = ('object_id', 'question', 'profile', 'horizon', 'loss_tolerance', 'financial_scope')
PATH_META = ('execution_status', 'service_admitted', 'financial_available', 'scope', 'q1_suitable',
             'q1_assignment', 'case_status', 'liquidity', 'scenario_id')
OUTPUTS = {'Q2_paths.csv.gz', 'Q1_paths.csv.gz', 'summary.csv.gz', 'evaluation_traces.npz',
           'object_specification.json', 'existing_development_context.json'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): digest.update(chunk)
    return digest.hexdigest()


def readj(path): return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fingerprint(ids, returns, inflation):
    h = hashlib.sha256()
    for value in (ids, returns, inflation): h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def verify_map(mapping, prefix=ROOT):
    if not isinstance(mapping, dict) or not mapping: raise ValueError('Nonempty SHA256 closure required')
    for name, expected in mapping.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Relative immutable SHA256 dependency required')
        if sha(prefix / path) != expected: raise ValueError('Immutable dependency changed: ' + name)


def authorization_metadata(p, release, manifest, dev, dev_audit, *, preparation_sha, release_sha,
                           dev_sha, dev_audit_sha, plan_sha):
    """Pure prerequisite check. No file opening, parser or NPZ access."""
    if (p.get('final_execution_ready') is not False or p.get('development_execution_ready') is not True
        or p.get('development_ids') != list(range(1, 121)) or p.get('final_ids') != list(range(121, 181))):
        raise ValueError('Original immutable preparation identities required')
    if release.get('version') != 'locked-final-evaluation-release-v1' or release.get('final_execution_ready') is not True:
        raise ValueError('Separate explicit final release required')
    bound = dict(preparation_protocol_sha256=preparation_sha,
        execution_sources_sha256=p['execution_sources_sha256'], object_specification_sha256=p['object_specification_sha256'],
        development_gate_manifest_sha256=dev_sha, development_gate_audit_sha256=dev_audit_sha,
        final_audit_plan_sha256=plan_sha, training_ids=list(range(1, 121)), evaluation_ids=list(range(121, 181)),
        output_directory='results/final_locked_evaluation_v1')
    if any(release.get(k) != v for k, v in bound.items()): raise ValueError('Release hash/identity binding differs')
    if (dev.get('status') != 'passed' or dev.get('final_scenarios_loaded') is not False
        or dev.get('preparation_protocol_sha256') != preparation_sha
        or dev.get('object_specification_sha256') != p['object_specification_sha256']
        or dev_audit.get('status') != 'passed' or dev_audit.get('parent_manifest_sha256') != dev_sha):
        raise ValueError('Passed independently bound development gate required')
    final_bound = dict(release_sha256=release_sha, preparation_protocol_sha256=preparation_sha,
        object_specification_sha256=p['object_specification_sha256'], development_gate_manifest_sha256=dev_sha,
        development_gate_audit_sha256=dev_audit_sha, training_ids_loaded=list(range(1, 121)),
        evaluation_ids_loaded=list(range(121, 181)), final_scenarios_loaded=True, final_value_access_started=True,
        final_values_may_have_been_parsed=True, completed_final_array_load=True, new_fits=0,
        selection_changed=False, admission_changed=False, current_C_service='execution_review',
        current_C_service_financials=None, Q2_path_rows=240, Q1_path_rows=300, summary_rows=9)
    if (not set(final_bound) <= set(manifest) or manifest.get('status') != 'complete'
        or any(manifest.get(k) != v for k, v in final_bound.items())):
        raise ValueError('Complete final producer bound to unchanged release required')
    for m in (manifest, dev):
        if (m.get('sources') != m.get('sources_at_end') or m.get('input_sha256') != m.get('input_sha256_at_end')
            or m.get('sources') != p['execution_sources_sha256'] or m.get('input_sha256') != p['frozen_input_sha256']):
            raise ValueError('Complete unchanged source/input snapshots required')
    if set(manifest.get('outputs_sha256', {})) != OUTPUTS: raise ValueError('Exact six producer outputs required')


def preflight():
    # Only JSON control metadata and byte hashes are read here. No value parser.
    if (OUT / 'independent_audit.json').exists() or ATTEMPT.exists():
        raise RuntimeError('Never overwrite or automatically retry an independent final audit')
    if not RELEASE.is_file() or not (OUT / 'manifest.json').is_file() or not FREEZE.is_file():
        raise RuntimeError('Root release, complete final producer and independent audit freeze required first')
    p, release, m = readj(PROTOCOL), readj(RELEASE), readj(OUT / 'manifest.json')
    dev, dev_audit, freeze = readj(DEV / 'manifest.json'), readj(DEV / 'independent_audit.json'), readj(FREEZE)
    if freeze.get('execution_ready') is not True: raise RuntimeError('Independent audit source not released')
    must = {'scripts/audit_final_locked_evaluation.py', 'tests/test_audit_final_locked_evaluation.py',
        'docs/final_locked_evaluation_audit_plan.json', 'scripts/audit_review_pending_execution.py',
        'scripts/audit_fan_rollout_execution.py', 'scripts/audit_final_evaluation_development_gate.py',
        'src/safeinvest/data.py'}
    if not must <= set(freeze.get('source_sha256', {})): raise ValueError('Complete independent helper/source closure required')
    verify_map(freeze['source_sha256'])
    authorization_metadata(p, release, m, dev, dev_audit, preparation_sha=sha(PROTOCOL), release_sha=sha(RELEASE),
        dev_sha=sha(DEV / 'manifest.json'), dev_audit_sha=sha(DEV / 'independent_audit.json'), plan_sha=sha(PLAN))
    for mapping in (p['execution_sources_sha256'], p['frozen_input_sha256']): verify_map(mapping)
    verify_map(m['outputs_sha256'], OUT); verify_map(dev['outputs_sha256'], DEV)
    before, after = dev_audit['sources_and_artifacts_at_start'], dev_audit['sources_and_artifacts_at_end']
    if not before or before != after: raise ValueError('Current independently audited development closure required')
    verify_map(before)
    files = set(before) | set(p['execution_sources_sha256']) | set(p['frozen_input_sha256']) | set(freeze['source_sha256'])
    files |= {str(x.relative_to(ROOT)) for x in (PROTOCOL, RELEASE, PLAN, FREEZE, DEV / 'manifest.json',
        DEV / 'independent_audit.json', OUT / 'manifest.json')}
    files |= {str((OUT / x).relative_to(ROOT)) for x in m['outputs_sha256']}
    files |= {str((DEV / x).relative_to(ROOT)) for x in dev['outputs_sha256']}
    return p, m, {name: sha(ROOT / name) for name in sorted(files)}


def compare_array(a, b, checks, label, *, atol=3e-11, rtol=3e-13):
    a, b = np.asarray(a), np.asarray(b)
    checks.check(a.shape == b.shape, label + '/shape')
    if a.dtype.kind in 'bUiS' or b.dtype.kind in 'bUiS':
        checks.check(np.array_equal(a, b), label + '/exact'); return
    checks.check(np.array_equal(np.isnan(a), np.isnan(b)), label + '/NULLmask')
    checks.check(np.array_equal(np.isfinite(a), np.isfinite(b)), label + '/finite mask')
    finite = np.isfinite(a) & np.isfinite(b)
    error = float(np.max(np.abs(a[finite] - b[finite]), initial=0.))
    checks.maximum_error = max(checks.maximum_error, error)
    checks.check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + '/value')


def validate_layout(observed, template, fields, *, n=60, old_n=120):
    if set(observed) != set(fields) or set(template) != set(fields) or len(set(fields)) != len(fields):
        raise ValueError('Complete distinct saved field inventory required')
    for name in fields:
        a, b = np.asarray(observed[name]), np.asarray(template[name])
        shape = b.shape if name in COMMON else (n, *b.shape[1:])
        if name not in COMMON and (b.ndim < 1 or b.shape[0] != old_n): raise ValueError('Template original row dimension differs')
        if a.dtype != b.dtype or a.shape != shape: raise ValueError('Full dtype/shape changed: ' + name)


def gate_algebra(qh, qr, kh, kr, *, h=30, eta=.03, n=120):
    if (not np.isfinite([qh, qr]).all() or not all(isinstance(x, (int, np.integer)) for x in (kh, kr))
        or not 0 <= kh <= n or not 0 <= kr <= n): raise ValueError('Finite original fan mean/count required')
    score = (qr - qh) - h * eta * ((kr - kh) / n)
    return score, 'restore' if score > TOL else 'hold'


def check_shadow_decision(arm, i, t, bad, checks, label):
    action = str(arm['actions'][i, t])
    if t == 0 or bad:
        expected = 'restore' if t == 0 else 'hold'
        checks.check(action == expected, label + '/mandatory or absorbing pending action')
        for key in Q_FLOAT: checks.check(np.isnan(arm[key][i, t]), label + '/' + key + ' inactive NULL')
        for key in Q_INT: checks.check(arm[key][i, t] == -1, label + '/' + key + ' inactive sentinel')
        reason = 'mandatory_initial_restore' if t == 0 else 'already_breached_pending_hold'
        checks.check(arm['gate_reason'][i, t] == reason, label + '/inactive reason')
        checks.check(not arm['restoration_predicted_growth_sacrifice'][i, t], label + '/inactive sacrifice false')
        return action
    kh, kr = arm['qbad_hold_count'][i, t], arm['qbad_restore_count'][i, t]
    qh, qr = float(arm['qg_hold'][i, t]), float(arm['qg_restore'][i, t])
    score, expected = gate_algebra(qh, qr, kh, kr)
    checks.check(arm['forecast_n'][i, t] == 120, label + '/original120training only')
    checks.check(action == expected, label + '/finite gate action')
    checks.value(arm['predicted_score'][i, t], score, label + '/originalH finite score', atol=3e-13)
    checks.value(arm['predicted_growth_advantage_restore'][i, t], qr - qh, label + '/growth difference', atol=3e-13)
    checks.check(arm['predicted_breach_count_delta_restore'][i, t] == kr - kh, label + '/paired count difference')
    checks.check(arm['gate_reason'][i, t] == ('finite_score_positive' if expected == 'restore' else 'finite_score_nonpositive'), label + '/finite gate reason')
    checks.check(bool(arm['restoration_predicted_growth_sacrifice'][i, t]) == (expected == 'restore' and qr - qh < -TOL), label + '/risk-cost sacrifice flag')
    if expected == 'restore' and qr - qh <= TOL: checks.check(kr < kh, label + '/nonpositive growth requires risk reduction')
    return action


def metrics_from_reference(reference, loss):
    return export_metrics({key: np.expand_dims(np.asarray(value), 0) for key, value in reference.items()}, 0, loss)


def metadata_row(obj, sid):
    base = {key: obj[key] for key in BASE}
    return dict(base, execution_status=obj['execution_status'], service_admitted=obj['service_admitted'],
        financial_available=obj['financial_available'], scope=SCOPE, q1_suitable=obj.get('q1_suitable'),
        q1_assignment=obj.get('q1_assignment', obj.get('assignment')), case_status=obj['case_status'],
        liquidity=obj['liquidity'], scenario_id=int(sid))


def fixed_witnesses(arm, refs, ids, train, train_inflation, checks):
    results = []
    id_index = {int(sid): i for i, sid in enumerate(ids)}
    for sid in (121, 150, 180):
      i = id_index[sid]; ref = refs[sid]
      for year in (2, 16, 30):
        t = year - 1
        row = dict(scenario_id=sid, original_year=year, original_horizon=30, training_n=120,
                   training_ids=list(range(1, 121)), action=str(arm['actions'][i, t]))
        if bool(ref['ever_breached'][t]):
            row.update(status='skipped_original_slot_pending', replacement=None); results.append(row); continue
        q = scalar_fan(ref['holdings'][t], ref['cpi'][t], ref['real_peak'][t], ref['running_mdd'][t],
            [.45, .05, .50], .4, train, train_inflation, t, cost_rate=.0025)
        for action in ('hold', 'restore'):
            checks.check(q[action]['logs'].shape == q[action]['bad'].shape == (120,), f'witness/{sid}/{year}/{action}/complete fan')
            checks.check(np.isfinite(q[action]['logs']).all(), f'witness/{sid}/{year}/{action}/finite')
            checks.value(arm['qg_' + action][i, t], q[action]['mean'], f'witness/{sid}/{year}/{action}/net remaining logQ', atol=3e-11, rtol=3e-13)
            checks.check(arm['qbad_' + action + '_count'][i, t] == q[action]['count'], f'witness/{sid}/{year}/{action}/whole original MDD count')
            row['qg_' + action] = q[action]['mean']; row['qbad_' + action + '_count'] = q[action]['count']
            row[action + '_branch_logs_sha256'] = hashlib.sha256(np.ascontiguousarray(q[action]['logs']).tobytes()).hexdigest()
            row[action + '_branch_bad_sha256'] = hashlib.sha256(np.ascontiguousarray(q[action]['bad']).tobytes()).hexdigest()
        score, action = gate_algebra(q['hold']['mean'], q['restore']['mean'], q['hold']['count'], q['restore']['count'])
        checks.check(action == row['action'], f'witness/{sid}/{year}/independent Q action')
        row.update(status='passed_independent_scalar_fan', predicted_score=score); results.append(row)
    checks.check(len(results) == 9, 'all9fixed witness slots retained')
    return results


def values_audit(p, manifest, before, state):
    # This callback is reachable only below preflight and durable audit archive.
    sys.path.insert(0, str(ROOT / 'src'))
    from safeinvest.data import load_paths
    c = Checks()
    train, train_inflation, train_ids = load_paths(ROOT / 'data/SafeInvest_Synthetic_Data.csv', range(1, 121), allow_final=False)
    c.check(train_ids.tolist() == list(range(1, 121)), 'training original1..120 chronology')
    c.check(fingerprint(train_ids, train, train_inflation) == manifest['training_development_arrays_sha256'], 'same frozen original training arrays')
    state.update(phase='opening_authorized_final_values', final_value_access_started=True)
    writej(ATTEMPT / 'attempt.json', state)
    actual, inflation, ids = load_paths(ROOT / 'data/SafeInvest_Synthetic_Data.csv', range(121, 181), allow_final=True)
    c.check(ids.tolist() == list(range(121, 181)), 'final original121..180 chronology')
    c.check(fingerprint(ids, actual, inflation) == manifest['evaluation_arrays_sha256'], 'same complete original final arrays')
    spec = p['object_specification']
    c.check(spec == readj(OUT / 'object_specification.json') and canonical(spec) == p['object_specification_sha256'], 'exact unchanged9object specification')
    c.check(readj(OUT / 'existing_development_context.json') == p['existing_development_context'], 'existing development context unchanged')
    objects = spec['Q2_objects'] + spec['Q1_objects']; objmap = {x['object_id']: x for x in objects}
    available = {x['object_id'] for x in objects if x['financial_available']}
    c.check(len(objmap) == 9 and len(available) == 6, '9objects6physical groups')
    frames = {name: pd.read_csv(OUT / (name + '.csv.gz'), float_precision='round_trip') for name in ('Q2_paths', 'Q1_paths', 'summary')}
    c.check([len(frames[x]) for x in ('Q2_paths', 'Q1_paths', 'summary')] == [240, 300, 9], 'complete240/300/9denominators')
    for question in ('Q2', 'Q1'):
        f = frames[question + '_paths']
        c.check(set(f.columns) == set((*BASE, *PATH_META, *FIN)), question + '/complete path field inventory')
        c.check(not f.duplicated(['object_id', 'scenario_id']).any(), question + '/unique object and originalID')
        c.check(set(f.object_id) == {x['object_id'] for x in spec[question + '_objects']}, question + '/all object cohorts')
    summary = frames['summary']; c.check(not summary.object_id.duplicated().any() and set(summary.object_id) == set(objmap), 'all9unique summaries')
    allrefs = {}; count = dict(accounts=0, periods=0, endpoints=0, review_rows=0, active_shadow_gates=0, pending_shadow_periods=0, array_fields=0)
    with np.load(OUT / 'evaluation_traces.npz', allow_pickle=False) as saved, np.load(DEV / 'development_traces.npz', allow_pickle=False) as old:
      c.check({x.split('__', 1)[0] for x in saved.files} == available, 'physical groups exactly6 no C/A/DReview')
      for obj in objects:
        name = obj['object_id']; rows = []; refs = {}; arm = None
        if obj['financial_available']:
            arm = {key.split('__', 1)[1]: saved[key] for key in saved.files if key.startswith(name + '__')}
            compare_array(arm.pop('scenario_ids'), ids, c, name + '/metadata original IDs')
            template = {key.split('__', 1)[1]: old[key] for key in old.files if key.startswith(name + '__') and not key.endswith('__scenario_ids')}
            fields = p['exact_reproduction_fields'][name] if name in p['exact_reproduction_fields'] else p['exact_reproduction_fields']['same_target_B']
            validate_layout(arm, template, fields); count['array_fields'] += len(fields)
            compare_array(arm['targets'], obj['target'], c, name + '/target metadata')
            if name == SHADOW:
                compare_array(arm['annual_prices'], [.03], c, name + '/fixed eta metadata')
                compare_array(arm['loss_tolerances'], [.4], c, name + '/original L metadata')
        for i, sid0 in enumerate(ids):
            sid = int(sid0); row = metadata_row(obj, sid)
            if arm is None:
                row.update(dict.fromkeys(FIN)); count['review_rows'] += 1
            else:
                h, loss = obj['horizon'], obj['loss_tolerance']; target = np.asarray(obj['target'])
                if name == SHADOW:
                    def action_at_year(t, x, cp, pk, worst, bad):
                        action = check_shadow_decision(arm, i, t, bad, c, f'{name}/{sid}/{t+1}')
                        if t and not bad: count['active_shadow_gates'] += 1
                        if bad: count['pending_shadow_periods'] += 1
                        return action
                    ref = reference_actual(actual[i, :h], inflation[i, :h], target, loss, action_at_year, cost_rate=.0025)
                else:
                    rule = 'pending_hold' if name == 'same_target_B' else 'annual'
                    ref = replay_reference(actual[i, :h], inflation[i, :h], target, loss, arm=rule, cost_rate=.0025)
                    ref = {k: v for k, v in ref.items() if k not in ('targets', 'loss_tolerances')}
                expected_physical = set(arm) - COMMON - set(Q_FLOAT) - set(Q_INT) - set(Q_OTHER)
                c.check(set(ref) == expected_physical and len(ref) == 35, name + '/all35physical fields independently replayed')
                for field in sorted(ref): compare_array(arm[field][i], ref[field], c, f'{name}/{sid}/{field}')
                row.update(metrics_from_reference(ref, loss)); refs[sid] = ref
                count['accounts'] += 1; count['periods'] += h; count['endpoints'] += h + 1
            rows.append(row)
            frame = frames[obj['question'] + '_paths']
            block = frame.loc[frame.object_id.eq(name) & frame.scenario_id.eq(sid)]
            c.check(len(block) == 1, name + f'/{sid}/one original path row')
            observed = block.iloc[0]
            if not obj['financial_available']: require_financial_NULL(observed)
            for key, value in row.items(): c.value(observed[key], value, name + f'/{sid}/export/' + key)
        allrefs[name] = refs
        expected = summary_expected(rows, obj); expected['scope'] = SCOPE
        c.check(set(summary.columns) == set(expected), name + '/complete summary field inventory')
        block = summary.loc[summary.object_id.eq(name)]; c.check(len(block) == 1, name + '/one original summary')
        for key, value in expected.items(): c.value(block.iloc[0][key], value, name + '/summary/' + key)
        if name == SHADOW: witnesses = fixed_witnesses(arm, refs, ids, train, train_inflation, c)
    for key, expected in [('accounts', 360), ('periods', 8460), ('endpoints', 8820), ('review_rows', 180), ('array_fields', 228)]:
        c.check(count[key] == expected, 'complete independent denominator/' + key)
    after = {name: sha(ROOT / name) for name in before}
    c.check(before == after, 'all release/protocol/source/input/output/development evidence unchanged during audit')
    return dict(version='independent-locked-final-evaluation-audit-v1', status='passed',
        completed_utc=datetime.now(timezone.utc).isoformat(), parent_manifest_sha256=sha(OUT / 'manifest.json'),
        release_sha256=sha(RELEASE), preparation_protocol_sha256=sha(PROTOCOL), audit_plan_sha256=sha(PLAN),
        audit_source_freeze_sha256=sha(FREEZE), checks=c.n, maximum_absolute_numeric_error=c.maximum_error,
        counts=count, fixed_Q_witnesses=witnesses,
        independently_recomputed_active_Q_slots=sum(x['status'] == 'passed_independent_scalar_fan' for x in witnesses),
        skipped_pending_Q_slots=sum(x['status'] == 'skipped_original_slot_pending' for x in witnesses),
        sources_and_artifacts_at_start=before, sources_and_artifacts_at_end=after,
        reused_independent_engines=['100-step self-financing fee bisection', 'whole originalH state and reference policy replay',
            'chronological scalar empiricalfan', 'independent saved-array financial and summary arithmetic'],
        source_guard_review=source_gate_review((ROOT / 'scripts/prepare_final_evaluation.py').read_text()),
        limitations=['Shadow complete physical replay uses saved actions and all active saved-Q algebra; independent two-action scalar Q is restricted to predeclared9slots, not every adaptiveQ.',
            'The guarded market parser is reused and byte/array identities are bound; it is not a separately written market parser.',
            'Previously exposed oldV1 final IDs are not newly blind. No tuning, reapproval, severe-example selection or forecast calibration claim.',
            'Current C remains executionReview with financialNULL; product suitability and rejected shadow/reference financials do not admit the current service.'])


def run():
    p, manifest, before = preflight()
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    state = dict(status='incomplete', phase='all_release_and_source_guards_passed_no_final_values_read',
        started_utc=datetime.now(timezone.utc).isoformat(), final_value_access_started=False,
        parent_manifest_sha256=sha(OUT / 'manifest.json'), auditor_source_sha256=sha(Path(__file__)))
    writej(ATTEMPT / 'attempt.json', state)
    try:
        result = values_audit(p, manifest, before, state)
        writej(OUT / 'independent_audit.json', result)
        state.update(status='passed', phase='completed_independent_audit', receipt_sha256=sha(OUT / 'independent_audit.json'))
        writej(ATTEMPT / 'attempt.json', state)
        print(json.dumps(dict(status='passed', checks=result['checks'], counts=result['counts'],
            independent_Q_slots=result['independently_recomputed_active_Q_slots'], receipt_sha256=state['receipt_sha256'])))
    except BaseException as exc:
        (ATTEMPT / 'error.log').write_text(traceback.format_exc())
        state.update(status='failed', phase=state['phase'], exception_type=type(exc).__name__, exception_message=str(exc),
                     error_log_sha256=sha(ATTEMPT / 'error.log'), rerun_allowed=False)
        writej(ATTEMPT / 'attempt.json', state)
        raise


if __name__ == '__main__': run()
