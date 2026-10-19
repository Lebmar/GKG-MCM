"""Constructed independent ledger tests; no supplied saved arrays or raw data."""
from pathlib import Path
import ast,csv,gzip,json,math,sys,tempfile,unittest
from unittest import mock
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_review_operation as a


class IndependentReviewAuditTests(unittest.TestCase):
    def test_exact20_transfer_cost_is005_once(self):
        total=99.95;w=[20/total,0,79.95/total]
        post,fee,moved=a.fee_endpoint([0,0,100],w)
        np.testing.assert_allclose(post,[20,0,79.95],atol=1e-12,rtol=0)
        self.assertAlmostEqual(fee,.05,places=13);self.assertAlmostEqual(moved,20,places=12)
        self.assertAlmostEqual(sum(post)+fee,100,places=12)

    def test_full_switch_and_no_fee_have_analytic_values(self):
        post,fee,moved=a.fee_endpoint([100,0,0],[0,0,1])
        self.assertAlmostEqual(post[2],100/1.0025,places=12)
        self.assertAlmostEqual(fee,.0025*100/1.0025,places=13)
        self.assertAlmostEqual(moved,post[2],places=13)
        self.assertEqual(a.fee_endpoint([20,30,50],[.2,.3,.5],0),([20.,30.,50.],0.,0.))

    def test_static_hold_initial_fee_and_no_later_fee(self):
        r=a.static_reference(np.ones((1,3,3)),np.ones((1,4)),[.2,.3,.5],'hold')
        expected=100/(1+.0025*.5)
        np.testing.assert_allclose(r['real_wealth'][0],[100,expected,expected,expected],atol=1e-12,rtol=0)
        self.assertAlmostEqual(r['fees'][0,0],100-expected,places=12)
        np.testing.assert_array_equal(r['fees'][0,1:],[0.,0.])
        self.assertEqual(a.financial(r['real_wealth'][0],r['fees'][0],r['trades'][0])['trade_fraction'],1/3)

    def test_two_year_annual_fee_analytic_and_no_double_charge(self):
        gross=np.array([[[2.,.5,1.],[.5,2.,1.]]]);cp=np.ones((1,3))
        r=a.static_reference(gross,cp,[.5,.5,0.],'annual')
        initial=100/1.0025;second=(1.25*initial+.0025*.25*initial)/(1+.0025*.5)
        self.assertAlmostEqual(r['nominal_wealth'][0,1],initial*1.25,places=11)
        self.assertAlmostEqual(r['nominal_wealth'][0,2],second*1.25,places=11)
        self.assertAlmostEqual(r['fees'][0,1],1.25*initial-second,places=12)
        self.assertEqual(r['actions'][0].tolist(),['restore','restore'])

    def test_bill_real_return_can_breach_and_recovery_censor_is_retained(self):
        r=a.static_reference(np.full((1,4,3),1.01),np.array([[1,1.2,1.44,1.728,2.0736]]),[0,0,1],'hold')
        self.assertTrue(r['ever_breached'][0,-1]);self.assertTrue(r['right_censored'][0]);self.assertFalse(r['recovered'][0])
        self.assertTrue(np.all(r['fees']==0))

    def test_two_observations_retain_predebit_peak_and_immediate_loss(self):
        real=np.array([100.,150.,200.,210.]);peak=real.copy();m=np.zeros(4);x=np.repeat((real/3)[:,None],3,axis=1)
        r=a.cost_path(real,peak,m,x,np.ones(4),2,.1)
        np.testing.assert_array_equal(r['real_wealth'],[100,150,180,189])
        self.assertEqual(r['real_peak'][2],200.);self.assertAlmostEqual(r['running_mdd'][-1],.1)
        naive=a.financial(r['real_wealth'],np.zeros(3),np.zeros(3,bool))
        self.assertEqual(naive['mdd'],0.)  # A single post-debit annual sequence loses the old200peak.
        self.assertAlmostEqual(a.financial(r['real_wealth'],np.zeros(3),np.zeros(3,bool),r['running_mdd'][-1])['mdd'],.1)
        self.assertAlmostEqual(r['review_fee_nominal'],20.)

    def test_fee_zero_exact_and_last_year_or_no_guard_not_charged(self):
        r=np.array([100.,130.,100.,160.]);p=np.maximum.accumulate(r);m=np.maximum.accumulate(1-r/p);x=np.repeat((r/3)[:,None],3,axis=1)
        for tau,q in [(2,0.),(-1,.01),(3,.01)]:
            out=a.cost_path(r,p,m,x,np.ones(4),tau,q)
            for key,original in [('real_wealth',r),('real_peak',p),('running_mdd',m)]:np.testing.assert_array_equal(out[key],original)
            self.assertEqual(out['review_fee_nominal'],0.)
        self.assertIsNone(a.cost_path(r,p,m,x,np.ones(4),-1,.01)['debit_endpoint'])

    def test_existing_loss_history_cannot_reset_after_review_expense(self):
        r=np.array([100.,200.,100.,190.]);p=np.maximum.accumulate(r);m=np.maximum.accumulate(1-r/p);x=np.repeat((r/3)[:,None],3,axis=1)
        out=a.cost_path(r,p,m,x,np.ones(4),2,.01)
        self.assertEqual(out['first_crossing'],2);self.assertTrue(out['breach']);self.assertGreaterEqual(out['running_mdd'][-1],.5)

    def test_phase_identities_partition_and_pending_without_strategy_call(self):
        gross=np.array([[[1.1,1.02,1.01],[.7,.94,1.01],[1.3,1.05,1.01],[1.04,1.01,1.01]]]);cp=np.array([[1,1.02,1.04,1.06,1.08]])
        r=a.static_reference(gross,cp,[.45,.15,.4],'hold');rg=gross/(cp[:,1:]/cp[:,:-1])[:,:,None]
        r.update(first_guard_trigger=np.array([2]),guard_triggered=np.array([[False,True,False,False]]),guard_pending=np.array([[False,False,True,True,True]]),
            was_pending_input=np.array([[False,False,True,True]]),was_guard_pending_input=np.array([[False,False,True,True]]),automatic_service=np.array([[True,False,False,False]]),
            gate_reason=np.full((1,4),'constructed',dtype='U64'),realized_real_gross=rg)
        c=a.Checks();annual,parts,accounts=a.phases(r,[9],'synthetic',rg,c)
        self.assertEqual([x['phase'] for x in annual],['before_guard','guard_year','after_guard','after_guard'])
        self.assertEqual(sum(x['periods'] for x in parts),4)
        self.assertAlmostEqual(math.fsum(x['net_log_growth'] for x in parts),math.log(r['real_wealth'][0,-1]/100),places=13)
        self.assertEqual(accounts[0]['post_guard_periods'],2)
        r['fees'][0,2]=1.
        with self.assertRaises(AssertionError):a.phases(r,[9],'synthetic',rg,a.Checks())

    def test_break_even_negative_capacity_and_no_followup_null(self):
        x=a.break_even(120,110,80,True);self.assertAlmostEqual(x['max_nominal_fee'],80/12)
        x=a.break_even(90,100,80,True);self.assertLess(x['signed_fraction_threshold'],0);self.assertIsNone(x['max_nominal_fee']);self.assertTrue(x['zero_fee_already_worse'])
        self.assertTrue(all(v is None for v in a.break_even(90,100,80,False).values()))

    def test_fractional_cvar_median_and_ordinal_quantile_are_distinct(self):
        self.assertAlmostEqual(a.cvar90(list(range(15))),(14+.5*13)/1.5)
        self.assertEqual(a.median([1,3,5,7]),4);self.assertEqual(a.quantile([1,3,5,7],.9),7)

    def test_sameID_pairs_keep_all_event_categories_and_reject_missing_ID(self):
        static=[];base=[]
        for sid in range(1,121):
            metrics=dict(log_growth=.02,cagr=.03,mdd=.2,nominal_fees=.1,trade_fraction=.1,breach=sid<=2)
            static.append(dict(policy='a0.5_hold',scenario_id=sid,**metrics))
            for policy in ('main',*a.REFS):base.append(dict(policy=policy,scenario_id=sid,**{**metrics,'breach':sid==2 or sid==3}))
        rows,summaries=a.paired(static,base);self.assertEqual(len(rows),480);self.assertEqual(len(summaries),4)
        self.assertEqual(summaries[0]['event_partition'],dict(neither=117,static_only=1,reference_only=1,both=1))
        with self.assertRaises(AssertionError):a.paired(static[:-1],base)

    def test_table_catches_double_fee_wrong_ID_and_NULL(self):
        c=a.Checks();expected=[dict(id=1,fee=.05,missing=None,flag=True)];good=[dict(id='1',fee='0.05',missing='',flag='True')]
        c.table(good,expected,('id',),'constructed')
        for wrong in [dict(good[0],fee='0.1'),dict(good[0],id='2'),dict(good[0],missing='0')]:
            with self.assertRaises(AssertionError):c.table([wrong],expected,('id',),'constructed')

    def test_missing_audit_freeze_blocks_npz_and_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with mock.patch.object(a.np,'load') as load:
                with self.assertRaises(FileNotFoundError):a.run(root)
            load.assert_not_called();self.assertEqual(a.readj(root/a.ATTEMPT/'manifest.json')['status'],'failed')
            with self.assertRaises(RuntimeError):a.run(root)

    def test_no_import_of_new_producer_main_policy_or_fee_solver(self):
        tree=ast.parse(Path(a.__file__).read_text());imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
        self.assertFalse(any(x.startswith(('safeinvest','experimental_execution','diagnostics','diagnose_review_operation','pandas')) for x in imports))


if __name__=='__main__':unittest.main()
