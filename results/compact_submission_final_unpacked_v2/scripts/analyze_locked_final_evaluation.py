"""Read-only, predeclared paired interpretation of the locked final results."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT/'results/final_locked_evaluation_v1'
OUT = ROOT/'results/final_locked_analysis_v1'
PLAN = ROOT/'docs/final_evaluation_analysis_plan.json'
METRICS = ('log_growth', 'cagr', 'mdd', 'nominal_fees', 'trade_fraction')
REFS = ('same_target_B', 'exact_thirds_annual')
TOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value == 'True':
        return True
    if value == 'False':
        return False
    raise ValueError('Explicit finite event boolean required')


def median(values):
    v = sorted(values); n = len(v)
    if n == 0:
        raise ValueError('Nonempty paired sample required')
    return v[n//2] if n % 2 else (v[n//2-1]+v[n//2])/2


def paired(shadow, reference, reference_name):
    """Exact same original IDs; every adverse result is retained."""
    if shadow.empty or shadow.scenario_id.duplicated().any() or reference.scenario_id.duplicated().any():
        raise ValueError('Nonempty unique original IDs required')
    if set(shadow.scenario_id) != set(reference.scenario_id):
        raise ValueError('Paired populations must be identical')
    a = shadow.set_index('scenario_id').sort_index()
    b = reference.set_index('scenario_id').sort_index()
    rows = []
    events = {name: [] for name in ('neither', 'shadow_only', 'reference_only', 'both')}
    for sid in a.index:
        left, right = a.loc[sid], b.loc[sid]
        event_m, event_b = flag(left.breach), flag(right.breach)
        key = 'both' if event_m and event_b else 'shadow_only' if event_m else 'reference_only' if event_b else 'neither'
        events[key].append(int(sid))
        row = dict(scenario_id=int(sid), reference=reference_name, shadow_breach=event_m,
                   reference_breach=event_b, event_pair=key)
        for metric in METRICS:
            l, r = float(left[metric]), float(right[metric])
            if not math.isfinite(l) or not math.isfinite(r):
                raise ValueError('Available shadow/reference finance must be finite')
            row['delta_'+metric] = l-r
        rows.append(row)
    summary = dict(reference=reference_name, n=len(rows), difference_direction='unapproved shadow minus reference',
                   original_ids=[r['scenario_id'] for r in rows], event_original_ids=events,
                   event_counts={key: len(value) for key, value in events.items()}, metric_differences={})
    for metric in METRICS:
        values = [r['delta_'+metric] for r in rows]
        summary['metric_differences'][metric] = dict(mean=math.fsum(values)/len(values), median=median(values),
            positive=sum(v>TOL for v in values), negative=sum(v < -TOL for v in values), equality=sum(abs(v)<=TOL for v in values))
    return rows, summary


def main():
    if OUT.exists():
        raise RuntimeError('Do not overwrite a completed interpretation archive')
    m, audit = readj(FINAL/'manifest.json'), readj(FINAL/'independent_audit.json')
    release = readj(ROOT/'docs/final_evaluation_release.json')
    if (m['status'] != 'complete' or audit['status'] != 'passed'
            or audit['parent_manifest_sha256'] != sha(FINAL/'manifest.json')
            or m['release_sha256'] != sha(ROOT/'docs/final_evaluation_release.json')
            or release['analysis_plan_sha256'] != sha(PLAN)):
        raise RuntimeError('Exact released, independently audited final outputs and prior analysis plan required')
    plan = readj(PLAN)
    if not plan['declared_before_current_final_read']:
        raise RuntimeError('Interpretation must follow the prior declared plan')
    inputs = [Path(__file__), PLAN, ROOT/'docs/final_evaluation_release.json',
              FINAL/'manifest.json', FINAL/'independent_audit.json']
    for name, expected in m['outputs_sha256'].items():
        path = FINAL/name
        if path.parent != FINAL or sha(path) != expected:
            raise RuntimeError('Final output hash differs')
        inputs.append(path)
    snapshot = lambda: {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    before = snapshot()
    data = pd.read_csv(FINAL/'Q2_paths.csv.gz', float_precision='round_trip')
    summary = pd.read_csv(FINAL/'summary.csv.gz', float_precision='round_trip')
    if len(data) != 240 or len(summary) != 9:
        raise ValueError('Complete predeclared objects required')
    main_service = data[data.object_id.eq('guarded_C_service')]
    finance_fields = list(METRICS)+['volatility','terminal_real','breach','excess','first_crossing']
    if len(main_service) != 60 or not main_service[finance_fields].isna().all().all():
        raise ValueError('Current service must retain all60 NULL financial records')
    shadow = data[data.object_id.eq('unapproved_fixed_shadow')]
    all_rows, comparisons = [], []
    for reference in REFS:
        ref = data[data.object_id.eq(reference)]
        if sorted(shadow.scenario_id.tolist()) != list(range(121,181)) or sorted(ref.scenario_id.tolist()) != list(range(121,181)):
            raise ValueError('All60 prescribed original evaluation IDs required')
        rows, comp = paired(shadow, ref, reference)
        all_rows.extend(rows); comparisons.append(comp)
    if before != snapshot():
        raise RuntimeError('Source/evidence changed during interpretation')
    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(all_rows).to_csv(OUT/'paired_differences.csv.gz', index=False, float_format='%.17g', compression={'method':'gzip','mtime':0})
    writej(OUT/'paired_summary.json', comparisons)
    (OUT/'official_summary.csv.gz').write_bytes((FINAL/'summary.csv.gz').read_bytes())
    outputs = {p.name: sha(p) for p in OUT.iterdir()}
    writej(OUT/'manifest.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        sources_and_artifacts_at_start=before, sources_and_artifacts_at_end=snapshot(), outputs_sha256=outputs,
        parent_manifest_sha256=sha(FINAL/'manifest.json'), analysis_plan_sha256=sha(PLAN),
        all_prescribed_final_IDs_retained=True, paired_comparisons=2, paired_rows=120, no_new_fit_or_simulation=True,
        model_or_admission_changed=False, primary_C_service='execution_review', primary_finance=None,
        inference='Descriptive paired outcomes of unapproved shadow and both references; oldV1 exposure precludes newlyblind claims; no reapproval or future-risk certificate.'))
    print(json.dumps(dict(status='complete', comparisons=2, rows=len(all_rows))))


if __name__ == '__main__':
    main()
