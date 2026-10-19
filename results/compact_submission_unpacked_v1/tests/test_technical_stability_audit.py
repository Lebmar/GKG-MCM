from pathlib import Path
import ast,copy,json,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_technical_stability as a


def jobs_and_fits():
    jobs=[];fits={}
    for repeat in range(1,6):
        order=list(range(1,121))
        for fold in range(1,6):
            ev=order[24*(fold-1):24*fold];train=sorted(set(order)-set(ev))
            for inner in range(1,6):
                test=train[24*(inner-1):24*inner] if inner<5 else ev
                training=sorted(set(train)-set(test)) if inner<5 else train
                name=f'{repeat}_{fold}_{inner}'
                job=dict(job_id=name,repeat_id=repeat,outer_fold=fold,stage='inner' if inner<5 else 'refit',training_ids=training,evaluation_ids=test)
                jobs.append(job);fits[name]=dict(refit=dict(training_ids=training.copy()))
    return jobs,fits


class StabilityAuditTests(unittest.TestCase):
    def test_each_repeat_covers_original_ids_once_with_own_fits(self):
        j,f=jobs_and_fits();a.validate_jobs(j,f)

    def test_missing_duplicate_or_final_jobs_rejected(self):
        j,f=jobs_and_fits()
        with self.assertRaises(ValueError):a.validate_jobs(j[:-1],f)
        bad=copy.deepcopy(j);bad[0]['evaluation_ids'][0]=121
        with self.assertRaises(ValueError):a.validate_jobs(bad,f)
        bad=copy.deepcopy(j);bad[1]['evaluation_ids']=bad[0]['evaluation_ids']
        with self.assertRaises(ValueError):a.validate_jobs(bad,f)

    def test_own_training_fit_cannot_be_replaced(self):
        j,f=jobs_and_fits();f[j[0]['job_id']]['refit']['training_ids']=j[1]['training_ids']
        with self.assertRaises(ValueError):a.validate_jobs(j,f)

    def test_identifier_type_range_and_duplicates(self):
        for values in ([True],[1.5],[121],[1,1],[]):
            with self.assertRaises(ValueError):a.unique_development(values)

    def test_reference_unavailable_status_keeps_actual_Q1_review_reason(self):
        g=dict(kind='reference',target=None,status='unavailable')
        self.assertEqual(a.null_reason(g,dict(status='risk_review')),'q1_review')
        self.assertEqual(a.null_reason(g,dict(status='not_balanced_case')),'not_balanced_case')
        g['kind']='outer_selected';g['status']='no_inner_feasible_candidate'
        self.assertEqual(a.null_reason(g,dict(status='risk_review')),'no_inner_feasible_candidate')

    def test_missing_release_stops_before_market_or_npz(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'plan';f=Path(temp)/'freeze'
            p.write_text('{}');f.write_text(json.dumps(dict(execution_ready=False,plan_sha256=a.sha(p))))
            with patch.object(a,'PLAN',p),patch.object(a,'FREEZE',f),patch.object(a.np,'load') as loader:
                with self.assertRaises(RuntimeError):a.authorize()
                loader.assert_not_called()

    def test_empty_funded_cohort_stays_NULL_in_both_versions(self):
        rows=[dict(scenario_id=1,assigned=False,review_reason='no_inner_feasible_candidate')]
        self.assertIsNone(a.original.summary(rows)['mean_log_growth'])
        self.assertIsNone(a.mitigation.summary(rows)['mean_mitigation_fees'])

    def test_reused_independent_math_contains_no_production_policy_import(self):
        source=Path(a.__file__).read_text();tree=ast.parse(source)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('experimental_execution' in name or 'safeinvest.core' in name for name in imports))
        self.assertIn('range(1,121),allow_final=False',source)


if __name__=='__main__':unittest.main()
