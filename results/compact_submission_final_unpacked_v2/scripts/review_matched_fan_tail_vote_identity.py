"""Post-hoc identity regrouping of already audited fixed-state fan members.

No market parser, simulator, policy selector or altered fan is executed.
Every removed member and every nonzero paired event vote is retained.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/model_review/matched_parameter_fan_contrast_v1'
OUT = ROOT / 'docs/matched_fan_tail_vote_identity_review.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError('Do not overwrite a completed derivative review')
    paths = [Path(__file__), FOLDER/'manifest.json', FOLDER/'independent_audit.json',
             FOLDER/'fan_members.csv', FOLDER/'same_state_Q.csv']
    snapshot = lambda: {str(p.relative_to(ROOT)): sha(p) for p in paths}
    before = snapshot()
    audit = json.loads((FOLDER/'independent_audit.json').read_text())
    assert audit['status'] == 'passed'
    assert audit['producer_manifest_sha256'] == sha(FOLDER/'manifest.json')
    members = pd.read_csv(FOLDER/'fan_members.csv', float_precision='round_trip')
    fixed_q = pd.read_csv(FOLDER/'same_state_Q.csv', float_precision='round_trip')
    assert len(members) == 335 and len(fixed_q) == 3
    groups = {label: frame.set_index('fan_scenario_id').sort_index()
              for label, frame in members.groupby('fan_label')}
    full, outer = groups['full120'], groups['outer96']
    assert len(full) == 120 and len(outer) == 96
    assert set(outer.index) < set(full.index)
    fields = ['hold_total_log_growth', 'restore_total_log_growth',
              'hold_ever_breached', 'restore_ever_breached']
    assert full.loc[outer.index, fields].equals(outer[fields])
    removed = full.drop(outer.index)
    event_votes = full[full.hold_ever_breached != full.restore_ever_breached]
    records = []
    errors = []
    for label, frame in [*groups.items(), ('removed24', removed)]:
        hold, restore = (int(frame[k].sum()) for k in fields[2:])
        delta = math.fsum(frame.restore_total_log_growth-frame.hold_total_log_growth)/len(frame)
        records.append(dict(label=label, n=len(frame), bad_hold=hold, bad_restore=restore,
                            mean_paired_total_log_advantage=delta))
        if label != 'removed24':
            row = fixed_q[fixed_q.fan_label == label].iloc[0]
            assert (hold, restore) == (row.qbad_hold_count, row.qbad_restore_count)
            error = abs(delta-row.predicted_growth_advantage_restore)
            errors.append(error)
            assert error <= 1e-12
    full_delta = math.fsum(full.restore_total_log_growth-full.hold_total_log_growth)
    split_delta = math.fsum(outer.restore_total_log_growth-outer.hold_total_log_growth)
    split_delta += math.fsum(removed.restore_total_log_growth-removed.hold_total_log_growth)
    assert abs(full_delta-split_delta) <= 1e-12
    assert before == snapshot()
    encode = lambda f: json.loads(f.reset_index().to_json(orient='records', double_precision=15))
    result = dict(
        version='posthoc-audited-fan-tail-vote-identity-review-v1', status='passed',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        prior_exposure='All three fan scores/actions and full member outcomes were already known. This is an explicitly post-hoc identity regrouping, not a new forecast or independently chosen experiment.',
        contrast='Only original full112/outer_r5_112 common predecision year5 state; fixed45/5/50, eta.03, H30/L.4.',
        sources_and_artifacts_at_start=before, sources_and_artifacts_at_end=snapshot(),
        group_counts_and_growth=records,
        all_full120_nonzero_paired_event_votes=encode(event_votes),
        all24_members_excluded_by_original_outer_fan=encode(removed),
        maximum_aggregate_Q_roundoff=max(errors),
        interpretation='The only full120 member whose hold/restore event labels differ is original95: hold breaches, restore does not. The original outer96 omits95, as well as23 other members including actual112. Common96 member branch outcomes are exactly identical. The full-to-outer change therefore loses that one modeled tail-reduction vote while also changing the paired growth mean and denominator.',
        limits=[
            'This identifies the member supporting one local model-internal discrete event difference; it is not a unique causal attribution of the score or eventual risk.',
            'No95-only deletion is simulated, no alternative fan is selected, and no suggestion to drop95 or choose a preferred subset follows.',
            'The other23 removed members and all120 votes remain visible, including actual112 whose removal alone does not flip the original action.',
            'Derivative exact regrouping of independently audited member arrays is not an additional independent market replay, calibrated tail law, full learner LOOCV or population claim.',
            'The existing currentC executionReview, financialNULL and all risk/admission rules are unchanged.'
        ], new_market_loads=0, new_simulations=0, new_fits=0, final_scenarios_read=False)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], nonzero_vote_ids=event_votes.index.tolist(),
                         removed_ids=removed.index.tolist(), maximum_roundoff=max(errors))))


if __name__ == '__main__':
    main()
