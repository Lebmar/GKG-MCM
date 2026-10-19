"""Constructed-state/file checks only; no real final values or producer imports."""
import ast,csv,gzip,copy
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import audit_mitigation_locked_final_evaluation as a


def row(sid=121,assigned=True,oid='mitigation_main',mdd=.2):
    obj=next(x for x in a.expected_objects() if x['object_id']==oid)
    r=dict(**obj,scope='previously_exposed_nonblind',execution_status='funded' if assigned else 'q1_review',assigned=assigned,scenario_id=sid)
    if not assigned:r.update(dict.fromkeys(a.FIN));return r
    r.update(log_growth=.02,cagr=.02,mdd=mdd,volatility=.1,terminal_real=180.,nominal_fees=.3,trade_fraction=.2,breach=mdd>obj['L']+a.TOL,first_crossing=None,
        automatic_fraction=.4,protection_trade_fraction=.1,support_review=True,mitigation_trade_fraction=1/obj['horizon'],mitigation_nominal_fees=.1)
    return r


def flat_reference(h,n=2):
    x=np.zeros((n,h+1,3));x[:,:,2]=100.;weights=x/100.
    scalar={'first_crossing':np.full(n,-1,int),'triggered':np.zeros(n,bool),'drawdown_at_trigger':np.full(n,np.nan),'peak_at_trigger':np.full(n,np.nan),'real_wealth_at_trigger':np.full(n,np.nan),
        'recovery_endpoint':np.full(n,-1,int),'recovery_time':np.full(n,-1,int),'recovered':np.zeros(n,bool),'right_censored':np.zeros(n,bool),'available_followup':np.full(n,-1,int),'observed_followup':np.full(n,-1,int),'added_severity':np.full(n,np.nan),'terminal_excess':np.zeros(n)}
    return dict(holdings=x,weights=weights,nominal_wealth=np.full((n,h+1),100.),real_wealth=np.full((n,h+1),100.),real_peak=np.full((n,h+1),100.),current_drawdown=np.zeros((n,h+1)),running_mdd=np.zeros((n,h+1)),ever_breached=np.zeros((n,h+1),bool),cpi=np.ones((n,h+1)),
        posttrade_holdings=x[:,:-1].copy(),fees=np.zeros((n,h)),moved=np.zeros((n,h)),pretrade_wealth=np.full((n,h),100.),fee_fraction=np.zeros((n,h)),moved_fraction=np.zeros((n,h)),pretrade_tv_drift=np.zeros((n,h)),end_year_tv_drift=np.zeros((n,h)),
        target_execution=np.ones((n,h),bool),pending_at_decision=np.zeros((n,h),bool),actions=np.full((n,h),'restore'),trades=np.zeros((n,h),bool),end_weights=weights[:,1:].copy(),**scalar)


class FinalAuditTests(unittest.TestCase):
    def test_exact14_objects_horizons_and_review(self):
        objects=a.expected_objects();self.assertEqual(len(objects),14);self.assertEqual(len({o['object_id'] for o in objects}),14)
        self.assertEqual([o['object_id'] for o in objects if o['rule']=='Review'],['Q1_A','Q1_D'])
        for name,h,L,_ in a.PROFILES:
            for suffix in ('','_thirds'):
                obj=next(o for o in objects if o['object_id']=='Q1_'+name+suffix);self.assertEqual((obj['horizon'],obj['L']),(h,L))
    def test_main_fixed_separate_reference_rules(self):
        x={o['object_id']:o for o in a.expected_objects()}
        self.assertEqual(x['mitigation_main']['target'],[.45,.05,.5]);self.assertEqual(x['same_target_B']['rule'],'pending_hold')
        self.assertEqual(x['same_target_annual']['rule'],'annual');self.assertIsNone(x['Q1_A']['target']);self.assertIsNone(x['Q1_A_thirds']['q1_suitable'])
    def test_reference_physical_original5_and6_years(self):
        for h,L in ((5,.1),(6,.35),(15,.25)):
            arm=flat_reference(h);meta=dict(group_id='test',kind='reference',policy='annual',horizon=h,L=L,array_names=list(arm),omitted_diagnostic_arrays=[])
            c=a.Checks();a.reference_schema(arm,meta,2,120,c);a.physical(arm,np.zeros((2,h,3)),np.zeros((2,h)),np.array([0,0,1]),meta,c)
            self.assertEqual(a.path_metrics(arm,0,L)['automatic_fraction'],1.)
    def test_own_loss_tolerance_changes_event(self):
        arm=flat_reference(5,1);arm['real_wealth'][0]=[100,100,80,90,90,90];arm['running_mdd'][0]=[0,0,.2,.2,.2,.2]
        x=a.path_metrics(arm,0,.1);y=a.path_metrics(arm,0,.4)
        self.assertTrue(x['breach']);self.assertFalse(y['breach']);self.assertEqual(x['first_crossing'],2);self.assertIsNone(y['first_crossing'])
    def test_ordinary_service_guard_period_excluded(self):
        arm=flat_reference(5,1);arm['pending_at_decision'][0,1:]=True
        self.assertAlmostEqual(a.path_metrics(arm,0,.4)['automatic_fraction'],.2)
    def test_summary_inverse_cdf_and_fractional_tail(self):
        rows=[row(121+i,mdd=.1*(i+1)) for i in range(6)]
        s=a.summary(rows);self.assertAlmostEqual(s['cvar90_mdd'],.6);self.assertAlmostEqual(s['median_mdd'],.35);self.assertAlmostEqual(s['p90_mdd'],.6)
        self.assertEqual(s['original_n'],6);self.assertEqual(s['financial_n'],6)
    def test_review_summary_remains_NULL(self):
        rows=[row(121+i,False,'Q1_A') for i in range(60)];s=a.summary(rows)
        self.assertEqual(s['initial_review_n'],60);self.assertIsNone(s['breach_count']);self.assertIsNone(s['mean_nominal_fees']);self.assertEqual(s['initial_coverage'],0.)
    def test_empty_matched_reference_is_not_all60_summary(self):
        model=[row(121+i,False,'Q1_A') for i in range(60)];ref=[row(121+i,True,'Q1_A_thirds') for i in range(60)]
        result,pairs=a.compare(model,ref);self.assertFalse(pairs);self.assertEqual(result['reference_all_original']['financial_n'],60)
        self.assertEqual(result['same_funded_reference']['original_n'],0);self.assertIsNone(result['same_funded_reference']['mean_mdd']);self.assertEqual(len(result['initial_review_ids']),60)
    def test_exact_same_id_pairing_and_event_partition(self):
        model=[row(121,mdd=.5),row(122,mdd=.1)];ref=[row(121,oid='same_target_B',mdd=.1),row(122,oid='same_target_B',mdd=.5)]
        result,pairs=a.compare(model,ref);self.assertEqual(result['event_id_partition']['model_only'],[121]);self.assertEqual(result['event_id_partition']['reference_only'],[122])
        self.assertAlmostEqual(result['paired_differences']['mdd']['mean'],0)
        with self.assertRaises(ValueError):a.compare(model,ref[:1])
    def test_no_pooling_objects_or_duplicate_ids(self):
        with self.assertRaises(ValueError):a.summary([row(),row()])
        with self.assertRaises(ValueError):a.summary([row(),row(122,oid='Q1_C')])
    def test_csv_Review_cannot_be_zero(self):
        expected=[row(121,False,'Q1_A')]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x.csv.gz';columns=sorted(k for k in expected[0] if k!='target')
            def save(bad=False):
                with gzip.open(p,'wt',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();r=dict(expected[0]);r['mdd']=0 if bad else None;w.writerow(r)
            save();a.csv_compare(p,expected,a.Checks(),'review')
            save(True)
            with self.assertRaises(AssertionError):a.csv_compare(p,expected,a.Checks(),'badreview')
    def test_no_final_release_blocks_before_parser(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('safeinvest.data.load_paths',side_effect=AssertionError('must not parse')) as loader:
                with self.assertRaises(RuntimeError):a.preflight(Path(folder))
                loader.assert_not_called()
    def test_failed_attempt_retained_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(RuntimeError):a.run(root)
            state=a.readj(root/a.ATTEMPT/'attempt.json');self.assertEqual(state['status'],'failed');self.assertFalse(state['final_value_access_started'])
            self.assertTrue((root/a.ATTEMPT/'failure.txt').exists())
            with self.assertRaises(RuntimeError):a.run(root)
    def test_initial_independent_oracle_stays_V2b(self):
        s=np.ones((3,3));z=a.scalar_decision([0,0,100],1,100,0,s,s,np.full((1,3),1/3),target=[.45,.05,.5],release=.5,rule='uniform',initial=True)
        self.assertEqual(z['selected_index'],1);self.assertFalse(z['mitigation_evaluated']);self.assertNotIn('cushion_scale',z)
    def test_no_production_import_and_all_active_scope(self):
        text=Path(a.__file__).read_text();tree=ast.parse(text)
        modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('experimental_execution' in name or 'safeinvest.core' in name or 'safeinvest_research' in name for name in modules))
        self.assertIn('slots=np.argwhere(active)',text);self.assertNotIn('54fixed',text)


if __name__=='__main__':unittest.main()
