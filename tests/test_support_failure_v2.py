"""Constructed accounting/support counterexamples; no supplied numerical data."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('support_failure_runner_test',ROOT/'scripts/diagnose_support_failure_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def constructed_reconstruction():
    ids=np.arange(1,121);chunks=np.array_split(ids,5);g=[];z={}
    gross=np.ones((120,30,3));gross[:,:,0]=1.05;gross[:,:,1]=1.03;gross[:,:,2]=1.01
    gross[:,6]=[.6,.7,.9];cpig=np.full((120,30),1.02)
    for j,chunk in enumerate(chunks,1):
        name=f'c{j}_refit';ref=name+'_thirds';g.append(dict(group_id=ref,kind='reference',policy='exact_thirds_annual',job_id=name,scenario_ids=chunk.tolist()))
        z[ref+'__posttrade_holdings']=np.ones((24,30,3));a=np.ones((24,31,3));a[:,1:]=gross[chunk-1]
        z[ref+'__holdings']=a;z[ref+'__cpi']=np.concatenate([np.ones((24,1)),np.cumprod(cpig[chunk-1],axis=1)],axis=1)
    for j in range(6):
        actual=ids if j==0 else chunks[j-1];train=ids if j==0 else np.setdiff1d(ids,actual);name=f'c{j}_refit';gid=name+'_main'
        g.append(dict(group_id=gid,kind='full_selected' if j==0 else 'outer_selected',job_id=name,training_ids=train.tolist(),scenario_ids=actual.tolist(),target=[.45,.05,.5],candidate=dict(candidate_id='release_0.5_uniform',release_fraction=.5,forecast_rule='uniform',order=0)))
        z[gid+'__training_real_gross']=np.transpose(gross[train-1]/cpig[train-1,:,None],(1,0,2))
        z[gid+'__realized_real_gross']=gross[actual-1]/cpig[actual-1,:,None]
    return z,g,gross,cpig


class SupportFailureTests(unittest.TestCase):
    def test_original_peak_is_recomputed_not_reused(self):
        v=np.array([[100.,120.,110.]]);cb=np.ones_like(v);cs=np.array([[1.,1.3,1.5]])
        q=m.four_corners(v,cb,v,cs)
        self.assertEqual(q['corners']['BB']['peak'][0,-1],120.)
        self.assertEqual(q['corners']['BS']['peak'][0,-1],100.)
        self.assertGreater(q['corners']['BS']['mdd'][0],q['corners']['BB']['mdd'][0])
        self.assertTrue(np.all(q['log_nominal_path_component']==0))

    def test_four_corner_two_telescopes_and_interaction(self):
        vb=np.array([[100.,130.,90.,110.]]);vs=np.array([[100.,105.,100.,120.]])
        cb=np.array([[1.,1.01,1.02,1.03]]);cs=np.array([[1.,1.05,1.15,1.2]])
        q=m.four_corners(vb,cb,vs,cs)
        np.testing.assert_allclose(q['MDD_total'],q['MDD_CPI_first']+q['MDD_nominal_after_CPI'])
        np.testing.assert_allclose(q['MDD_total'],q['MDD_nominal_first']+q['MDD_CPI_after_nominal'])
        np.testing.assert_allclose(q['MDD_interaction'],q['MDD_CPI_after_nominal']-q['MDD_CPI_first'])
        np.testing.assert_allclose(q['log_nominal_path_component']+q['log_CPI_component'],np.log((vs/cs)/(vb/cb)))

    def test_cpi_shift_is_added_to_annual_inflation_not_real_returns(self):
        cb=np.cumprod(np.array([[1.,1.03,.99,1.04]]),axis=1)
        cs=np.concatenate([np.ones((1,1)),np.cumprod(np.array([[1.05,1.01,1.06]]),axis=1)],axis=1)
        self.assertFalse(np.allclose(cs,cb*1.02))
        np.testing.assert_allclose(cs[:,1:]/cs[:,:-1]-cb[:,1:]/cb[:,:-1],.02)

    def test_joint_rows_are_not_cartesian_asset_minima(self):
        support=np.array([[.4,1.2,1.],[1.2,.4,1.]])
        actual,which=m.directional_minimum(support,np.array([[50.,50.,0.]]),np.array([1.]))
        self.assertAlmostEqual(actual[0],80.)
        self.assertAlmostEqual((support.min(axis=0)*[50,50,0]).sum(),40.)

    def test_pooled_set_inclusion_monotone_for_same_action(self):
        y=np.array([[45.,5.,50.]]);calendar=np.array([[1.1,1.1,1.1],[1.,1.,1.]])
        pool=np.vstack([calendar,[.6,.7,.9]])
        cm,_=m.directional_minimum(calendar,y,np.ones(1));pm,_=m.directional_minimum(pool,y,np.ones(1))
        self.assertLess(pm[0],cm[0]);self.assertGreaterEqual(cm[0],80.);self.assertLess(pm[0],80.)

    def test_initial_fee_exact_once_and_zero_cost(self):
        y,fee=m.initial_full_target([.45,.05,.5],.0025)
        self.assertAlmostEqual(y.sum()+fee,100.);self.assertAlmostEqual(fee,.0025*(y[0]+y[1]))
        np.testing.assert_allclose(y/y.sum(),[.45,.05,.5])
        y0,f0=m.initial_full_target([.45,.05,.5],0.)
        np.testing.assert_array_equal(y0,[45.,5.,50.]);self.assertEqual(f0,0.)

    def test_budget_floor_clamp_at_tolerant_below_floor_state(self):
        f,b=m.budget(59.99999999999,100.,.5)
        self.assertEqual(f,60.);self.assertEqual(b,60.)

    def test_negative_real_bill_with_positive_nominal_bill(self):
        x=np.array([[0.,0.,100.]]);g=np.array([[1.1,1.02,1.01]])
        q=m.one_step_accounting(x,x,np.zeros(1),np.ones(1),np.array([1.04]),g)
        self.assertLess(q['real_gross'][0,2],1.);self.assertLess(q['asset_real_contribution'][0,2],0.)
        self.assertGreater(q['nominal_asset_gain'][0],0.);self.assertLess(q['inflation_erosion'][0],0.)

    def test_both_one_step_accounting_identities_with_fee(self):
        x=np.array([[20.,30.,50.]]);y=np.array([[25.,25.,49.9]]);g=np.array([[.8,1.02,1.01]])
        q=m.one_step_accounting(x,y,np.array([.1]),np.array([1.2]),np.array([1.236]),g)
        self.assertAlmostEqual(q['change'][0],q['asset_real_contribution'].sum()+q['fee_current_real'][0])
        self.assertAlmostEqual(q['change'][0],q['nominal_asset_gain'][0]+q['nominal_fee_at_next_CPI'][0]+q['inflation_erosion'][0])
        with self.assertRaises(AssertionError):m.one_step_accounting(x,y,np.array([.2]),np.array([1.2]),np.array([1.236]),g)

    def test_original_crossing_strict_tolerance(self):
        q=m.original_metrics(np.array([[100.,60.,61.],[100.,59.9,61.]]))
        np.testing.assert_array_equal(q['first_crossing'],[-1,1]);np.testing.assert_array_equal(q['breach'],[False,True])

    def test_reconstruction_all120_and_own96_training_alignment(self):
        z,g,gross,cpig=constructed_reconstruction();c=m.Checks();got,pi,sel,tr=m.reconstruct(z,g,c)
        np.testing.assert_allclose(got,gross);np.testing.assert_allclose(pi,cpig)
        self.assertEqual(len(sel),6);self.assertEqual(tr['c1_refit'].shape,(96,30,3))
        bad=copy.deepcopy(g);bad[0]['scenario_ids'][0]=bad[0]['scenario_ids'][1]
        with self.assertRaises(ValueError):m.reconstruct(z,bad,m.Checks())

    def test_inner_and_full_reference_metadata_are_not_outer_evidence(self):
        z,g,gross,cpig=constructed_reconstruction()
        # Match the canonical metadata shape: c0 inner30/90, outer-context
        # inner24/72, and a full120 reference. No arrays for these unrelated
        # groups are provided, so touching them is itself a test failure.
        distractors=[]
        for job,ev,tr in [('c0_inner1',list(range(1,31)),list(range(31,121))),
                          ('c1_inner1',list(range(25,49)),list(range(49,121))),
                          ('c0_refit',list(range(1,121)),list(range(1,121)))]:
            distractors.append(dict(group_id=job+'__exact_thirds_annual',scenario_ids=ev,kind='reference',policy='exact_thirds_annual',
                job_id=job,training_ids=tr,target=[1/3]*3,array_names=['holdings','posttrade_holdings','cpi'],omitted_diagnostic_arrays=[]))
        got,pi,sel,training=m.reconstruct(z,distractors+g,m.Checks())
        np.testing.assert_allclose(got,gross);np.testing.assert_allclose(pi,cpig)
        self.assertEqual(len(sel),6);self.assertEqual(set(training),{f'c{i}_refit' for i in range(6)})

    def test_exact_five_outer_roles_and_unique_references_are_still_required(self):
        z,g,_,_=constructed_reconstruction()
        bad=[v for v in g if not (v['kind']=='outer_selected' and v['job_id']=='c5_refit')]
        with self.assertRaises(ValueError):m.reconstruct(z,bad,m.Checks())
        dup=copy.deepcopy(next(v for v in g if v['kind']=='reference'));dup['group_id']='duplicate_unallowed'
        with self.assertRaises(ValueError):m.reconstruct(z,g+[dup],m.Checks())

    def test_initial_tables_all_contexts_years_members_and_review(self):
        z,cg,gross,pi=constructed_reconstruction();_,_,sel,training=m.reconstruct(z,cg,m.Checks());groups=[];witnesses=[]
        for job,parent in sel.items():
            for setting in m.SETTINGS:
                c={'fee_0000':0.,'fee_00125':.00125,'fee_005':.005,'fee_01':.01}.get(setting,.0025);mode='pooled' if setting=='pooled_support' else 'calendar'
                y,_=m.initial_full_target(parent['target'],c);val=np.einsum('nha,a->nh',training[job],y);minimum=val.min() if mode=='pooled' else val[:,0].min();ok=bool(minimum>=80.-1e-10)
                g=dict(policy='state_budget_learner',job_id=job,context_id=int(job[1]),cohort='full_resubstitution' if job=='c0_refit' else 'outer_OOF',
                    setting_id=setting,setting=dict(cost_rate=c,support_mode=mode),scenario_ids=parent['scenario_ids'],training_ids=parent['training_ids'],own_fit_target=parent['target'],selected_candidate=parent['candidate'],status='funded' if ok else 'preinvestment_review',financial_available=ok)
                groups.append(g);witnesses.append(dict(setting_id=setting,context_id=int(job[1]),witness=dict(budget_floor=80.,candidate_posttrade=[None,y.tolist()],candidate_min_support_slack=[None,float(minimum-80)])))
        rows,years,members=m.initial_tables(groups,witnesses,training,m.Checks())
        self.assertEqual((len(rows),len(years),len(members)),(42,180,18000))
        self.assertTrue(all(not r['financial_available'] for r in rows if r['setting_id']=='pooled_support'))
        self.assertTrue(all(r['financial_available'] for r in rows if r['setting_id']=='base'))
        self.assertEqual({r['training_year'] for r in years},set(range(1,31)))

    def test_missing_activation_stops_before_saved_npz_or_csv(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with mock.patch.object(m.np,'load') as arrays,mock.patch.object(m.pd,'read_csv') as csv:
                with self.assertRaises(FileNotFoundError):m.run(root)
            arrays.assert_not_called();csv.assert_not_called();self.assertFalse((root/m.OUT).exists())

    def test_pending_directional_miss_is_offline_not_active_certificate_failure(self):
        n,h=2,3;gross=np.ones((120,h,3));gross[:,:,2]=[1.01,.9,.8];cpig=np.ones((120,h))
        x=np.zeros((n,h+1,3));x[:,0,2]=100.
        for t in range(h):x[:,t+1]=x[:,t]*gross[:n,t]
        real=x.sum(axis=2);peak=np.maximum.accumulate(real,axis=1)
        train=np.ones((3,h,3));train[:,:,2]=[.9,.5,1.02]
        fields=dict(holdings=x,posttrade_holdings=x[:,:-1].copy(),fees=np.zeros((n,h)),cpi=np.ones((n,h+1)),real_wealth=real,real_peak=peak,
            running_mdd=np.maximum.accumulate(1-real/peak,axis=1),realized_real_gross=gross[:n],gate_evaluated=np.tile([True,True,False],(n,1)),
            automatic_service=np.tile([True,False,False],(n,1)),guard_triggered=np.tile([False,True,False],(n,1)),
            was_pending_input=np.tile([False,False,True],(n,1)),support_feasible=np.tile([True,False,False],(n,1)),
            budget_floor=np.full((n,h),np.nan),worst_real_shortfall_selected=np.full((n,h),np.nan),first_guard_trigger=np.full(n,2),first_crossing=np.full(n,-1),
            actual_decision_source=np.tile(['primary','mitigation','pending'],(n,1)))
        _,b=m.budget(real[:,:-1],peak[:,:-1],.5)
        fields['budget_floor'][:,:2]=b[:,:2]
        for t in range(2):
            minimum,_=m.directional_minimum(train[:,t],x[:,t],np.ones(n));fields['worst_real_shortfall_selected'][:,t]=np.maximum(b[:,t]-minimum,0.)
        g=dict(group_id='fake',financial_available=True,policy='state_budget_learner',array_names=list(fields),setting_id='base',cohort='outer_OOF',job_id='j',scenario_ids=[1,2],training_ids=[3,4,5],setting=dict(actual_inflation_shift=0.,support_mode='calendar'),selected_candidate=dict(release_fraction=.5))
        rows,summary=m.support_periods({'fake__'+k:v for k,v in fields.items()},[g],{'j':train},gross,cpig,m.Checks())
        self.assertEqual(len(rows),6);part=[r for r in rows if r['prior_pending']]
        self.assertTrue(all(r['directional_miss'] and not r['gate_evaluated'] for r in part))
        sm=next(s for s in summary if s['setting_id']=='base' and s['cohort']=='outer_OOF')
        self.assertEqual(sum(p['periods'] for p in sm['phases']),6)
        self.assertEqual(sm['phases'][2]['supported_safe_actual_budget_failures'],0)

    def test_full_fixed_pairing_corner_output_schema_and_annual_control(self):
        z={};groups=[];chunks=np.array_split(np.arange(1,121),5)
        for cohort in m.COHORTS:
            contexts=[(0,np.arange(1,121))] if cohort=='full_resubstitution' else list(enumerate(chunks,1))
            for context,ids in contexts:
                for policy in m.POLICIES:
                    for setting,shift in [('base',False),('actual_inflation_plus_02',True)]:
                        gid=f'{cohort}_{context}_{policy}_{setting}';n=len(ids);t=np.arange(31)
                        rate=1.025 if shift and policy=='state_budget_learner' else 1.03
                        nominal=np.tile(100*rate**t,(n,1));cpi=np.tile((1.03 if shift else 1.01)**t,(n,1));metric=m.original_metrics(nominal/cpi)
                        values=dict(nominal_wealth=nominal,cpi=cpi,real_wealth=metric['real'],holdings=nominal[:,:,None]*np.array([.3,.2,.5]),
                            posttrade_holdings=nominal[:,:-1,None]*np.array([.3,.2,.5]),fees=np.zeros((n,30)),running_mdd=metric['running_mdd'],first_crossing=metric['first_crossing'])
                        z.update({gid+'__'+k:v for k,v in values.items()});groups.append(dict(group_id=gid,setting_id=setting,cohort=cohort,policy=policy,job_id=f'c{context}_refit',scenario_ids=ids.tolist(),financial_available=True,array_names=list(values)))
        paths,endpoints,summary=m.inflation_tables(z,groups,m.Checks())
        self.assertEqual((len(paths),len(endpoints),len(summary)),(960,29760,8))
        self.assertTrue(all(s['original_n']==s['paired_financial_n']==120 for s in summary))
        self.assertEqual(len(summary[0]['event_quadrants']),16)
        self.assertTrue(all(abs(r['MDD_total']-r['MDD_CPI_first']-r['MDD_nominal_after_CPI'])<1e-12 for r in paths))
        g=next(g for g in groups if g['policy']=='same_target_annual' and g['setting_id']=='actual_inflation_plus_02')
        z[g['group_id']+'__fees'][0,0]=.1
        with self.assertRaises(AssertionError):m.inflation_tables(z,groups,m.Checks())

    def test_existing_attempt_never_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/m.OUT).mkdir(parents=True)
            with mock.patch.object(m,'preflight') as gate:
                with self.assertRaises(RuntimeError):m.run(root)
            gate.assert_not_called()

    def test_sources_have_no_production_policy_or_raw_loader_import(self):
        import ast
        text=Path(m.__file__).read_text();tree=ast.parse(text)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn('load_paths',calls);self.assertNotIn('simulate_state_budget',text);self.assertNotIn('experimental_execution',text)
        self.assertEqual(text.count('np.load('),2)


if __name__=='__main__':unittest.main()
