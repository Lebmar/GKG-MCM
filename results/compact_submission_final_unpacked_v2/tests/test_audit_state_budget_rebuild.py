"""Independent rebuild oracle checks on invented states only; no data files."""
from pathlib import Path
import math,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_rebuild as a


def row(sid,*,assigned=True,breach=False,mdd=.1,growth=.03,fee=.5,support=False):
    return dict(scenario_id=sid,assigned=assigned,log_growth=growth,cagr=math.expm1(growth),mdd=mdd,
        volatility=.1,terminal_real=200.,nominal_fees=fee,trade_fraction=.5,breach=breach,
        first_crossing=1 if breach else None,automatic_fraction=.8,protection_trade_fraction=.1,support_review=support)


def synthetic_reference():
    n,h=1,30;x=np.zeros((n,h+1,3));x[:,:,2]=100.;w=x/100
    arm=dict(holdings=x,weights=w,posttrade_holdings=x[:,:-1].copy(),nominal_wealth=np.full((n,h+1),100.),
        real_wealth=np.full((n,h+1),100.),real_peak=np.full((n,h+1),100.),current_drawdown=np.zeros((n,h+1)),
        running_mdd=np.zeros((n,h+1)),ever_breached=np.zeros((n,h+1),bool),cpi=np.ones((n,h+1)),
        end_weights=w[:,1:].copy(),first_crossing=np.array([-1]),triggered=np.array([False]),
        recovered=np.array([False]),recovery_endpoint=np.array([-1]),recovery_time=np.array([-1]),
        right_censored=np.array([False]),available_followup=np.array([-1]),observed_followup=np.array([-1]),
        added_severity=np.array([np.nan]),terminal_excess=np.array([0.]),actions=np.full((n,h),'restore',dtype='U7'),
        pretrade_wealth=np.full((n,h),100.),target_execution=np.ones((n,h),bool),pending_at_decision=np.zeros((n,h),bool),trades=np.zeros((n,h),bool))
    for key in ('fees','moved','fee_fraction','moved_fraction','pretrade_tv_drift','end_year_tv_drift'):arm[key]=np.zeros((n,h))
    for key in ('drawdown_at_trigger','peak_at_trigger','real_wealth_at_trigger'):arm[key]=np.array([np.nan])
    group=dict(group_id='synthetic',kind='reference',policy='same_target_annual',array_names=list(arm),omitted_diagnostic_arrays=[])
    return arm,group


class IndependentStateBudgetTests(unittest.TestCase):
    def decide(self,x=(0.,0.,100.),support=None,growth=None,initial=False,**kw):
        support=np.ones((3,3)) if support is None else np.asarray(support)
        growth=support if growth is None else np.asarray(growth)
        return a.scalar_decision(x,1.,100.,0.,support,growth,np.full((1,len(growth)),1/len(growth)),
            target=[.45,.05,.5],release=.5,rule='uniform',initial=initial,**kw)
    def test_full_switch_closed_form_and_unique_physical_fee(self):
        y,f,m=a.fee_endpoint([0,0,100],[1,0,0])
        self.assertAlmostEqual(f,100*.0025/1.0025,places=13)
        self.assertAlmostEqual(f,.0025*m,places=13);self.assertAlmostEqual(sum(y)+f,100,places=13)
    def test_partial_line_exact_fee_and_not_linear_prefee_weights(self):
        x=np.array([40.,10.,50.]);w=np.array([.2,.6,.2]);end,f,m=a.fee_endpoint(x,w)
        for lam in (0.,.3,1.):
            post=x+lam*(end-x)
            self.assertAlmostEqual(sum(post)+lam*f,100,places=12)
            self.assertAlmostEqual(.0025*sum(np.maximum(post-x,0)),lam*f,places=13)
        self.assertFalse(np.allclose((x+.3*(end-x))/(100-.3*f),.7*x/100+.3*w,atol=1e-8,rtol=0))
    def test_positive_negative_and_zero_slope_intersection(self):
        z=a.scalar_interval([-.2,.8,0],[1,-1,0])
        self.assertTrue(z['feasible']);self.assertAlmostEqual(z['lo'],.2);self.assertAlmostEqual(z['hi'],.8)
        self.assertEqual(z['lower_member'],0);self.assertEqual(z['upper_member'],1)
    def test_zero_slope_impossible_not_clipped_into_feasible(self):
        z=a.scalar_interval([-.01],[0]);self.assertFalse(z['feasible']);self.assertTrue(np.isnan(z['lo']))
    def test_contradictory_interval_remains_empty(self):
        self.assertFalse(a.scalar_interval([-.8,.2],[1,-1])['feasible'])
    def test_initial_only_full_original_target_and13slots(self):
        z=self.decide(initial=True)
        self.assertEqual(z['selected_index'],1);self.assertEqual(z['candidate_posttrade'].shape,(13,3))
        self.assertEqual(list(np.flatnonzero(np.isfinite(z['candidate_lambda']))),[0,1])
        self.assertFalse(z['candidate_feasible'][0]);self.assertEqual(z['candidate_lambda'][1],1)
    def test_initial_Review_does_not_invent_bill_account(self):
        z=self.decide(support=np.full((3,3),.5),initial=True)
        self.assertEqual(z['status'],'initial_review');self.assertEqual(z['selected_index'],-1)
    def test_fee_net_safe_hold_rejects_costly_no_growth_trade(self):
        z=self.decide();self.assertEqual(z['selected_index'],0)
        self.assertEqual(z['selected_score'],0);self.assertEqual(z['fee'],0)
    def test_hold_infeasible_protection_quantifies_shortfall(self):
        z=self.decide(x=[100,0,0],support=np.tile([.5,1,1],(3,1)))
        self.assertFalse(z['hold_feasible']);self.assertGreaterEqual(z['selected_index'],0)
        self.assertEqual(z['gate_reason'],'risk_budget_protection')
        self.assertGreater(z['worst_real_shortfall_hold'],0)
        self.assertLessEqual(z['worst_real_shortfall_selected'],1e-10)
    def test_no_proposal_feasible_explicit_guard_hold(self):
        z=self.decide(support=np.full((3,3),.5))
        self.assertEqual(z['status'],'no_feasible_action');self.assertEqual(z['action'],'hold')
        self.assertEqual(z['selected_index'],-1);self.assertTrue(z['guard_triggered'])
    def test_roundoff_negative_headroom_clamped_to_client_floor(self):
        r=90.-8e-11
        z=a.scalar_decision([0,0,r],1,100,1-r/100,np.ones((3,3)),np.ones((3,3)),np.full((1,3),1/3),target=[.45,.05,.5],release=.5,rule='uniform',L=.1)
        self.assertEqual(z['budget_floor'],90.)
    def test_global_tie_band_not_pairwise_tolerance_chain(self):
        scores=np.array([0.,3e-12,2.1e-12,1.2e-12]);eligible=np.ones(4,bool)
        # The cheap last point is within tolerance of point2, but not max3e-12.
        i=a.choose_band(scores,eligible,[0.,3.,2.,1.],[-1,0,1,2],initial=False,hold_feasible=True)
        self.assertEqual(i,2)
    def test_safe_growth_tiny_positive_is_hold(self):
        self.assertEqual(a.choose_band([0.,.9e-12],[True,True],[0.,1.],[-1,0],initial=False,hold_feasible=True),0)
    def test_risk_forced_trade_can_have_negative_paired_score(self):
        self.assertEqual(a.choose_band([0.,-.02],[False,True],[0.,.1],[-1,0],initial=False,hold_feasible=False),1)
    def test_pair_min_advantage_is_not_difference_of_min_values(self):
        hold=np.array([0.,2.]);trade=np.array([1.,1.5])
        self.assertGreater(min(trade)-min(hold),0);self.assertLess(min(trade-hold),0)
    def test_forecasts_inception_uniform_and_blocks_original_order(self):
        ids=[9,1,7,4,2];self.assertEqual(a.scalar_forecasts(ids).shape,(1,5))
        z=np.tile([.1,.02,.005,.03],(5,1));p=a.scalar_forecasts(ids,z,z[0])
        np.testing.assert_array_equal(p[:2],np.full((2,5),.2))
        np.testing.assert_array_equal(p[2:],[[0,.5,0,0,.5],[0,0,.5,.5,0],[1,0,0,0,0]])
    def test_forecast_permutation_preserves_ID_alignment(self):
        ids=np.array([9,1,7,4,2]);z=np.array([[.1,.02,0,.03],[-.1,.03,.01,.05],[.02,.04,.002,.01],[.3,0,.006,.08],[-.2,.1,.02,.03]])
        order=[2,0,4,1,3];base=a.scalar_forecasts(ids,z,z[0]);other=a.scalar_forecasts(ids[order],z[order],z[0])
        np.testing.assert_allclose(other,base[:,order],atol=2e-16,rtol=0)
    def test_kernel_underflow_falls_back_per_row(self):
        z=np.tile([.02,.01,.005,.03],(5,1));p=a.scalar_forecasts([1,2,3,4,5],z,[1e100]*4)
        np.testing.assert_array_equal(p[:2],np.full((2,5),.2))
    def test_invalid_development_training_ID_rejected(self):
        for ids in ([1,121,2],[1,1,2],[]):
            with self.assertRaises(ValueError):a.scalar_forecasts(ids)
    def test_CVaR_fractional_mass_and_empirical_quantiles(self):
        self.assertAlmostEqual(a.cvar(range(1,25)),(24+23+.4*22)/2.4)
        self.assertEqual(a.quantile(range(1,25),.1),3);self.assertEqual(a.median([1,2,4,8]),3)
    def test_empty_financial_summary_is_NULL_not_zero_risk(self):
        z=a.summary([row(1,assigned=False),row(2,assigned=False)])
        self.assertEqual(z['original_n'],2);self.assertEqual(z['financial_n'],0)
        self.assertIsNone(z['breach_fraction']);self.assertIsNone(z['mean_log_growth'])
    def test_distinguishes_Q1_Review_nonbalanced_and_execution_Review(self):
        rows=[row(i,assigned=False) for i in range(1,4)]
        rows[0]['review_reason']='not_balanced_case';rows[1]['review_reason']='q1_review'
        z=a.summary(rows)
        self.assertEqual(z['initial_noninvested_n'],3)
        self.assertEqual(z['not_balanced_n'],1);self.assertEqual(z['q1_review_n'],1)
        self.assertEqual(z['initial_execution_review_n'],1)
        self.assertIsNone(z['breach_count']);self.assertIsNone(z['mean_nominal_fees'])
    def test_same_U_B_prevents_easy_rejection_bias(self):
        candidates={s['candidate_id']:[row(1,mdd=.1),row(2,assigned=False)] for s in a.SPECS}
        B=[row(1,mdd=.05),row(2,mdd=.9)]
        z=a.select(candidates,B)
        self.assertIsNone(z['selected_candidate_id'])
        self.assertTrue(all('inner_mean_mdd_worse_than_same_cohort_B' in e['reasons'] for e in z['evidence']))
    def test_selector_retains_original_population_and_deterministic_tie(self):
        B=[row(i,mdd=.2) for i in range(1,11)]
        candidates={s['candidate_id']:[row(i) for i in range(1,11)] for s in a.SPECS}
        z=a.select(candidates,B);self.assertEqual(z['selected_candidate_id'],a.SPECS[0]['candidate_id'])
        candidates[a.SPECS[0]['candidate_id']]=candidates[a.SPECS[0]['candidate_id']][1:]
        with self.assertRaises(ValueError):a.select(candidates,B)
    def test_NULLmask_corruption_and_missing_records_fail(self):
        with self.assertRaisesRegex(AssertionError,'NULLmask'):a.Checks().arr([0.],[np.nan],'x')
        with self.assertRaisesRegex(AssertionError,'complete keys'):a.record_check({},dict(x=1),a.Checks(),'x')
    def test_complete_synthetic_physical_reference_and_schema(self):
        arm,group=synthetic_reference();self.assertEqual(set(arm),a.PHYSICAL)
        a.schema(arm,group,1,3,a.Checks());a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),[0,0,1],group,a.Checks())
        metric=a.path_metrics(arm,0)
        self.assertEqual(metric['trade_fraction'],0);self.assertEqual(metric['automatic_fraction'],1)
        self.assertEqual(metric['mdd'],0);self.assertIsNone(metric['first_crossing'])
    def test_fee_physical_corruption_detected(self):
        arm,group=synthetic_reference();arm['fees'][0,0]=.1
        with self.assertRaisesRegex(AssertionError,'fee equation'):
            a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),[0,0,1],group,a.Checks())
    def test_CPI_or_peak_reset_detected(self):
        for key in ('cpi','real_peak'):
            arm,group=synthetic_reference();arm[key][0,-1]=50.
            with self.subTest(key=key),self.assertRaises(AssertionError):
                a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),[0,0,1],group,a.Checks())
    def test_missing_physical_field_detected(self):
        arm,group=synthetic_reference();del arm['first_crossing']
        with self.assertRaisesRegex(AssertionError,'inventory'):a.schema(arm,group,1,3,a.Checks())
    def test_independent_source_never_imports_producer(self):
        import ast
        tree=ast.parse(Path(a.__file__).read_text())
        modules=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any(name and ('experimental_execution' in name or 'safeinvest.core' in name or 'metrics' in name) for name in modules))
        funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        run=ast.get_source_segment(Path(a.__file__).read_text(),funcs['run'])
        self.assertLess(run.index('preflight()'),run.index('ATTEMPT.mkdir'))
        self.assertLess(run.index('ATTEMPT.mkdir'),run.index('audit_values('))


if __name__=='__main__':unittest.main()
