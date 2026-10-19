from pathlib import Path
import ast,copy,json,sys,tempfile,unittest
import numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_cushion_stability as a


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

    def test_scalar_cushion_algebra_and_boundaries(self):
        for loss in (.1,.4,.9):
            self.assertEqual(a.scalar_cushion(100.,100.,loss),1.)
            self.assertEqual(a.scalar_cushion((1-loss)*100,100.,loss),0.)
            self.assertAlmostEqual(a.scalar_cushion(80,100,.4),.625)
        self.assertEqual(a.scalar_cushion(99.999,100,0),0.)
        self.assertEqual(a.scalar_cushion(100,100,0),1.)
        self.assertEqual(a.scalar_cushion(40,100,1),1.)
        for values in ((0,100,.4),(100,90,.4),(100,100,1.1),(float('nan'),100,.4)):
            with self.assertRaises(ValueError):a.scalar_cushion(*values)

    def witness(self):
        active=np.array([[True,True,False]])
        arm=dict(gate_evaluated=active,cushion_scale=np.array([[1.,.625,np.nan]]),
            proposal_target_weights=np.full((1,3,4,3),np.nan),cpi=np.ones((1,4)),
            holdings=np.array([[[0.,0.,100.],[40.,0.,40.],[40.,0.,40.],[40.,0.,40.]]]),real_peak=np.ones((1,4))*100)
        target=[.45,.05,.5]
        for t,m in enumerate((1.,.625)):
            for p,aa in enumerate((1,.75,.5,.25)):
                for j in range(3):arm['proposal_target_weights'][0,t,p,j]=aa*m*target[j]+((1-aa*m) if j==2 else 0)
        return arm,target,dict(group_id='handmade',kind='inner_candidate')

    def test_allactive_formula_and_pending_null_without_forecast_inputs(self):
        arm,target,g=self.witness();checks=a.original.Checks()
        result=a.cushion_witnesses(arm,target,g,checks)
        self.assertEqual(result,dict(active=2,pending=1))
        self.assertGreater(checks.n,10)
        arm['cushion_scale'][0,2]=0.
        with self.assertRaises(AssertionError):a.cushion_witnesses(arm,target,g,a.original.Checks())

    def test_incorrect_cushion_or_destination_rejected(self):
        arm,target,g=self.witness();arm['cushion_scale'][0,1]=.7
        with self.assertRaises(AssertionError):a.cushion_witnesses(arm,target,g,a.original.Checks())
        arm,target,g=self.witness();arm['proposal_target_weights'][0,1,2,0]+=.01
        with self.assertRaises(AssertionError):a.cushion_witnesses(arm,target,g,a.original.Checks())

    def test_reference_never_receives_cushion_pseudo_account(self):
        c=a.original.Checks()
        self.assertEqual(a.cushion_witnesses({},[1/3]*3,dict(kind='reference',group_id='ref'),c),dict(active=0,pending=0))
        with self.assertRaises(AssertionError):a.cushion_witnesses({'cushion_scale':[]},[1/3]*3,dict(kind='reference',group_id='ref'),c)

    def test_reused_independent_math_contains_no_production_policy_import(self):
        source=Path(a.__file__).read_text();tree=ast.parse(source)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('experimental_execution' in name or 'safeinvest.core' in name for name in imports))
        self.assertIn('range(1,121),allow_final=False',source)


if __name__=='__main__':unittest.main()
