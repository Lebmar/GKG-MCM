"""Constructed physical paths only; no supplied market or saved result arrays."""
from pathlib import Path
import sys,tempfile,unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
from diagnostics import review_operation as op
import diagnose_review_operation as driver
from experimental_execution.review_pending import simulate_paired


def constructed_main(tau=2):
    gross=np.array([[[1.1,1.02,1.01],[.7,.94,1.01],[1.3,1.05,1.01],[1.04,1.01,1.01]]])
    cp=np.array([[1.,1.02,1.04,1.06,1.08]])
    a=op.static_reference(gross,cp,[.45,.15,.4],rule='hold')
    n,h=a['fees'].shape
    a.update(first_guard_trigger=np.array([tau]),guard_triggered=np.zeros((n,h),bool),guard_pending=np.zeros((n,h+1),bool),
        was_pending_input=np.zeros((n,h),bool),was_guard_pending_input=np.zeros((n,h),bool),
        automatic_service=np.ones((n,h),bool),gate_reason=np.full((n,h),'support_safe_hold',dtype='U64'),
        realized_real_gross=gross/(cp[:,1:]/cp[:,:-1])[:,:,None])
    if tau>=0:
        a['guard_triggered'][:,tau-1]=True;a['guard_pending'][:,tau:]=True
        a['was_pending_input'][:,tau:]=True;a['was_guard_pending_input'][:,tau:]=True;a['automatic_service'][:,tau-1:]=False
        a['gate_reason'][:,tau-1]='support_family_no_feasible_hold';a['gate_reason'][:,tau:]='guard_pending_hold'
    return a


class ReviewOperationTests(unittest.TestCase):
    def test_phase_partition_and_both_exact_growth_identities(self):
        a=constructed_main();rows,parts,accounts=op.phase_rows(a,[9],job_id='synthetic')
        self.assertEqual([r['phase'] for r in rows],['before_guard','guard_year','after_guard','after_guard'])
        self.assertEqual(sum(r['periods'] for r in parts),4)
        self.assertAlmostEqual(sum(r['net_log_growth'] for r in parts),np.log(a['real_wealth'][0,-1]/100),places=13)
        self.assertEqual(accounts[0]['first_guard'],2)
        for row in rows:
            self.assertAlmostEqual(row['log_fee']+row['real_portfolio_log_growth'],row['net_log_growth'],places=13)
            self.assertAlmostEqual(sum(row[x+'_real_gain'] for x in op.ASSETS)-row['real_fee_at_charge'],row['real_wealth_end']-row['real_wealth_start'],places=11)

    def test_no_guard_is_not_discarded_or_given_fictitious_trigger(self):
        rows,parts,accounts=op.phase_rows(constructed_main(-1),[120],job_id='synthetic')
        self.assertTrue(all(r['phase']=='no_guard' for r in rows));self.assertIsNone(accounts[0]['first_guard'])
        self.assertEqual(len(parts),4);self.assertFalse(accounts[0]['review_debit_within_H'])
        with self.assertRaises(ValueError):op.phase_rows(constructed_main(-1),[121],job_id='synthetic')

    def test_pending_fee_or_physical_trade_is_not_assumed_away(self):
        a=constructed_main();a['fees'][0,2]=.1
        with self.assertRaises(ValueError):op.audit_saved_arm(a)
        a=constructed_main();a['posttrade_holdings'][0,2,0]+=.1
        with self.assertRaises(ValueError):op.audit_saved_arm(a)

    def test_zero_cost_reproduces_all_original_financial_state_exactly(self):
        a=constructed_main();r=op.debit_review_cost(a['real_wealth'][0],a['real_peak'][0],a['running_mdd'][0],a['holdings'][0],a['cpi'][0],2,0.)
        for key in ('real_wealth','real_peak','running_mdd'):np.testing.assert_array_equal(r[key],a[key][0])
        self.assertTrue(r['applicable']);self.assertEqual(r['review_fee_nominal'],0.)

    def test_last_year_or_missing_guard_has_no_within_horizon_review_debit(self):
        a=constructed_main()
        for tau in (-1,4):
            r=op.debit_review_cost(a['real_wealth'][0],a['real_peak'][0],a['running_mdd'][0],a['holdings'][0],a['cpi'][0],tau,.01)
            self.assertFalse(r['applicable']);self.assertIsNone(r['debit_endpoint']);self.assertEqual(r['review_fee_nominal'],0.)
            np.testing.assert_array_equal(r['real_wealth'],a['real_wealth'][0])

    def test_debit_retains_trigger_year_predebit_peak_and_can_create_event(self):
        real=np.array([100.,150.,200.,210.]);peak=real.copy();mdd=np.zeros(4);holdings=np.repeat((real/3)[:,None],3,axis=1)
        r=op.debit_review_cost(real,peak,mdd,holdings,np.ones(4),2,.1,L=.05)
        np.testing.assert_array_equal(r['real_wealth'],[100.,150.,180.,189.])
        self.assertEqual(r['real_peak'][2],200.);self.assertAlmostEqual(r['running_mdd'][-1],.1)
        self.assertTrue(r['breach']);self.assertEqual(r['first_crossing'],2)
        self.assertAlmostEqual(r['review_fee_nominal'],20.)

    def test_break_even_reports_negative_capacity_and_inapplicable_as_null(self):
        good=op.break_even(120,110,80,applicable=True);self.assertAlmostEqual(good['max_nominal_fee'],80/12)
        bad=op.break_even(90,100,80,applicable=True);self.assertTrue(bad['zero_fee_already_worse']);self.assertIsNone(bad['max_nominal_fee'])
        self.assertLess(bad['signed_fraction_threshold'],0)
        self.assertTrue(all(v is None for v in op.break_even(90,100,0,applicable=False).values()))

    def test_independent_annual_reconstructs_every_old_field_on_constructed_market(self):
        rng=np.random.default_rng(311);market=rng.uniform(-.2,.2,(4,8,3));pi=rng.uniform(-.01,.05,(4,8));w=np.array([.45,.15,.4])
        wrapped=simulate_paired(market,pi,w,.4,horizon=8,cost_rate=.0025)['annual']
        old={k:v if k=='cpi' else v[:,0] for k,v in wrapped.items() if isinstance(v,np.ndarray) and k not in ('targets','loss_tolerances') and (k=='cpi' or v.ndim>=2)}
        own=op.static_reference(1+market,old['cpi'],w,rule='annual')
        check=op.compare_all_reference_fields(own,old);self.assertEqual(check['fields'],len(old))
        recovered,_=op.reconstruct_market(old);again=op.static_reference(recovered,old['cpi'],w,rule='annual')
        op.compare_all_reference_fields(again,old)

    def test_static_hold_only_initial_fee_and_bill_real_loss_is_possible(self):
        g=np.full((2,5,3),1.01);cp=np.tile(1.1**np.arange(6),(2,1))
        r=op.static_reference(g,cp,[0,0,1],rule='hold',L=.1)
        self.assertTrue(np.all(r['fees']==0));self.assertTrue(np.all(r['ever_breached'][:,-1]))
        mixed=op.static_reference(g,cp,[.45,.15,.4],rule='hold')
        self.assertTrue(np.all(mixed['fees'][:,0]>0));self.assertTrue(np.all(mixed['fees'][:,1:]==0))
        self.assertTrue(np.all(mixed['target_execution'][:,0]));self.assertFalse(np.any(mixed['target_execution'][:,1:]))

    def test_old_reference_discrete_fields_and_financial_values_both_checked(self):
        a=op.static_reference(np.ones((1,3,3)),np.ones((1,4)),[.45,.15,.4],rule='annual');b={k:v.copy() for k,v in a.items()}
        b['actions'][0,1]='hold'
        with self.assertRaises(ValueError):op.compare_all_reference_fields(a,b)
        b={k:v.copy() for k,v in a.items()};b['real_peak'][0,2]+=.01
        with self.assertRaises(ValueError):op.compare_all_reference_fields(a,b)

    def test_missing_activation_blocks_npz_before_attempt_creation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'docs').mkdir()
            driver.writej(root/driver.PLAN,dict(version='review-operation-development-diagnostic-v1',execution_ready=False))
            driver.writej(root/driver.FREEZE,dict(source_sha256={},input_sha256={}))
            with mock.patch.object(driver.np,'load') as loader:
                with self.assertRaises(FileNotFoundError):driver.run(root)
            loader.assert_not_called();self.assertFalse((root/driver.OUT).exists())


if __name__=='__main__':unittest.main()
