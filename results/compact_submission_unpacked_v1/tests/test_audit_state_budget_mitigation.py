"""Independent fallback oracle checks; only constructed numerical states."""
from pathlib import Path
import ast,math,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_mitigation as a


def failure_state():
    x=np.array([100.,0.,0.]);support=np.tile([.2,.75,.75],(3,1));growth=np.tile([2.,1.,1.],(3,1));weights=np.full((1,3),1/3)
    return x,support,growth,weights


def combined():
    x,s,g,w=failure_state()
    return a.scalar_decision(x,1,100,0,s,g,w,target=[.45,.05,.5],release=.5,rule='uniform')


class MitigationOracleTests(unittest.TestCase):
    def test_nine_candidates_exact_fee_and_funding(self):
        x,s,g,w=failure_state();m=a.scalar_mitigation(x,1,s,target=[.45,.05,.5],budget=80,growth=g,weights=w,peak=100,worst=0)
        self.assertEqual(m['candidate_posttrade'].shape,(9,3))
        np.testing.assert_array_equal(m['candidate_q'],[0,.5,1,.5,1,.5,1,.5,1])
        for y,fee,moved in zip(m['candidate_posttrade'],m['candidate_fees'],m['candidate_moved']):
            self.assertAlmostEqual(sum(y)+fee,100,places=12)
            self.assertAlmostEqual(fee,.0025*sum(np.maximum(y-x,0)),places=12)
            self.assertAlmostEqual(fee,.0025*moved,places=12)
    def test_ranking_ignores_adverse_growth_annotation(self):
        z=combined();self.assertTrue(z['mitigation_trade'])
        self.assertGreater(z['mitigation_worst_next_real_improvement'],0)
        self.assertLess(z['selected_score'],0);self.assertTrue(z['growth_sacrifice'])
    def test_original13points_and_failed_index_remain_unchanged(self):
        x,s,g,w=failure_state();old=a.base.scalar_decision(x,1,100,0,s,g,w,target=[.45,.05,.5],release=.5,rule='uniform')
        z=combined();self.assertEqual(z['selected_index'],-1)
        for key in old:
            if key.startswith('candidate_') or key.startswith('interval_'):
                np.testing.assert_array_equal(z[key],old[key])
        self.assertFalse(z['support_feasible']);self.assertTrue(z['guard_triggered'])
    def test_actual_executed_branch_is_not_primary_candidate0(self):
        z=combined();self.assertFalse(np.array_equal(z['posttrade_holdings'],z['candidate_posttrade'][0]))
        self.assertGreater(z['fee'],0);self.assertEqual(z['candidate_fees'][0],0)
        np.testing.assert_array_equal(z['primary_failed_hold_posttrade'],[100,0,0])
        self.assertEqual(z['primary_failed_hold_score'],0)
        self.assertEqual(z['actual_decision_source'],'mitigation')
    def test_residual_shortfall_not_recognized_as_floor_certificate(self):
        z=combined();self.assertGreater(z['worst_real_shortfall_selected'],0)
        self.assertGreater(z['worst_real_shortfall_improvement'],0)
        self.assertFalse(z['support_feasible'])
    def test_fee_aware_no_improvement_holds(self):
        x=[100.,0.,0.];s=np.ones((3,3));m=a.scalar_mitigation(x,1,s,target=[.45,.05,.5],budget=110,growth=s,weights=np.full((1,3),1/3),peak=100,worst=0)
        self.assertEqual(m['selected_index'],0);self.assertFalse(m['trade']);self.assertEqual(m['fee'],0)
    def test_global_band_not_pairwise_tolerance_chain(self):
        R=100;t=a.TOL*R
        worst=np.array([20.,30.,30-.8*t,30-1.6*t,21,21,21,21,21])
        choice=a.choose_worst(worst,[0,3,2,1,4,4,4,4,4],[-1,0,0,1,1,2,2,3,3],R)
        self.assertEqual(choice,2)
    def test_strict_threshold_scales_with_current_real_wealth(self):
        for scale in (1.,1000.):
            R=100*scale;w=[R,R+.9*a.TOL*R]+[R]*7
            self.assertEqual(a.choose_worst(w,[0,1,1,1,1,1,1,1,1],[-1,0,0,1,1,2,2,3,3],R),0)
    def test_inception_Review_never_has_fallback(self):
        s=np.full((3,3),.3)
        z=a.scalar_decision([0,0,100],1,100,0,s,s,np.full((1,3),1/3),target=[.45,.05,.5],release=.5,rule='uniform',initial=True)
        self.assertEqual(z['status'],'initial_review');self.assertFalse(z['mitigation_evaluated'])
        self.assertEqual(z['mitigation_selected_index'],-1)
    def test_safe_primary_remains_primary_with_blank_fallback(self):
        s=np.ones((3,3));z=a.scalar_decision([0,0,100],1,100,0,s,s,np.full((1,3),1/3),target=[.45,.05,.5],release=.5,rule='uniform')
        self.assertEqual(z['selected_index'],0);self.assertFalse(z['mitigation_evaluated'])
        self.assertEqual(z['actual_decision_source'],'primary');self.assertTrue(np.isnan(z['mitigation_candidate_fees']).all())
    def test_new_member_forecasts_have_fee_once(self):
        z=combined();v=100.
        for k in range(9):
            expected=math.log1p(-z['mitigation_candidate_fees'][k]/v)+z['mitigation_candidate_member_gross_advantage'][k,0]
            self.assertAlmostEqual(z['mitigation_candidate_member_advantage'][k,0],expected,places=13)
        s=z['mitigation_selected_index']
        self.assertAlmostEqual(z['selected_score'],z['mitigation_candidate_member_advantage'][s,0],places=14)
    def test_primary_diagnostic_view_does_not_overwrite_actual_account(self):
        z=combined();arm={}
        for k,v in z.items():
            name={'fee':'fees','action':'actions'}.get(k,k)
            if k=='status':continue
            arm[name]=np.asarray(v)[None,None,...]
        arm['target_execution']=np.array([[True]])
        original=arm['posttrade_holdings'].copy();view=a.primary_view(arm)
        np.testing.assert_array_equal(view['posttrade_holdings'][0,0],[100,0,0])
        self.assertEqual(view['fees'][0,0],0)
        np.testing.assert_array_equal(arm['posttrade_holdings'],original)
        self.assertGreater(arm['fees'][0,0],0)
    def test_mitigation_metrics_use_actual_trade_and_fee(self):
        arm=dict(real_wealth=np.array([[100.,90.,85.]]),running_mdd=np.array([[0.,.1,.15]]),fees=np.array([[.2,0.]]),
            moved=np.array([[80.,0.]]),trades=np.array([[True,False]]),pending_at_decision=np.array([[True,True]]),
            guard_triggered=np.array([[True,False]]),guard_pending=np.array([[False,True,True]]),
            actions=np.array([['mitigate','hold']]),gate_reason=np.array([['risk_mitigation_then_review','guard_pending_hold']]))
        m=a.path_metrics(arm,0)
        self.assertEqual(m['automatic_fraction'],0);self.assertEqual(m['mitigation_trade_fraction'],.5)
        self.assertEqual(m['protection_trade_fraction'],.5);self.assertEqual(m['mitigation_nominal_fees'],.2)
    def test_empty_funded_population_additional_summary_is_NULL(self):
        s=a.summary([dict(scenario_id=1,assigned=False,review_reason='no_inner_feasible_candidate')])
        self.assertIsNone(s['mean_mitigation_fees']);self.assertIsNone(s['mean_mitigation_trade_fraction'])
    def test_schema_inventories_full120_inner92(self):
        self.assertEqual(len(a.PRIMARY_FIELDS|a.MIT_FIELDS),120)
        omitted={k for k in a.PRIMARY_FIELDS|a.MIT_FIELDS if k.startswith(('candidate_','mitigation_candidate_')) or k in ('training_previous_state','training_real_gross')}
        self.assertEqual(len((a.PRIMARY_FIELDS|a.MIT_FIELDS)-omitted),92)
    def test_no_production_import_or_trigger_implies_hold_regression(self):
        text=Path(a.__file__).read_text();tree=ast.parse(text)
        modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('experimental_execution' in m or 'safeinvest.core' in m for m in modules))
        physical=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='physical')
        body=ast.get_source_segment(text,physical)
        self.assertNotIn('post[before|new]',body);self.assertIn('post[before]',body)


if __name__=='__main__':unittest.main()
