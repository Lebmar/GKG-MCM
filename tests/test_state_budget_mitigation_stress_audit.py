"""Constructed account/support tests, without raw files or producer imports."""
from pathlib import Path
import ast,copy,math,sys,tempfile,json,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_state_budget_mitigation_stress as a


def reference(cost,shift=0.,target=(.45,.05,.5)):
    n,h=1,30;w=np.array(target);post0,fee,moved=a.base.fee_endpoint([0,0,100],w,cost)
    x=np.tile(post0,(n,h+1,1));x[:,0]=[0,0,100];post=x[:,1:].copy()
    nominal=x.sum(axis=2);cp=np.array([[(1+shift)**t for t in range(h+1)]])
    real=nominal/cp;peak=np.maximum.accumulate(real,axis=1);dd=np.maximum(1-real/peak,0);bad=np.maximum.accumulate(dd-.4>1e-12,axis=1)
    first=int(np.flatnonzero(bad[0])[0]) if bad.any() else -1
    arm=dict(holdings=x,weights=x/nominal[:,:,None],posttrade_holdings=post,nominal_wealth=nominal,real_wealth=real,real_peak=peak,
        current_drawdown=dd,running_mdd=np.maximum.accumulate(dd,axis=1),ever_breached=bad,cpi=cp,pretrade_wealth=nominal[:,:-1],
        fees=np.zeros((n,h)),moved=np.zeros((n,h)),fee_fraction=np.zeros((n,h)),moved_fraction=np.zeros((n,h)),
        pretrade_tv_drift=.5*np.abs(x[:,:-1]/nominal[:,:-1,None]-w).sum(axis=2),end_year_tv_drift=.5*np.abs(x[:,1:]/nominal[:,1:,None]-w).sum(axis=2),
        target_execution=np.ones((n,h),bool),pending_at_decision=np.zeros((n,h),bool),actions=np.full((n,h),'restore'),trades=np.zeros((n,h),bool),
        end_weights=x[:,1:]/nominal[:,1:,None],first_crossing=np.array([first]),triggered=np.array([first>=0]),
        recovered=np.array([False]),recovery_endpoint=np.array([-1]),recovery_time=np.array([-1]),right_censored=np.array([first>=0]),
        available_followup=np.array([h-first if first>=0 else -1]),observed_followup=np.array([h-first if first>=0 else -1]),
        drawdown_at_trigger=np.array([dd[0,first] if first>=0 else np.nan]),peak_at_trigger=np.array([peak[0,first] if first>=0 else np.nan]),
        real_wealth_at_trigger=np.array([real[0,first] if first>=0 else np.nan]),added_severity=np.array([dd[0,-1]-dd[0,first] if first>=0 else np.nan]),
        terminal_excess=np.maximum(dd[:,-1]-.4,0))
    arm['fees'][0,0]=fee;arm['moved'][0,0]=moved;arm['trades'][0,0]=moved>1e-12
    arm['fee_fraction']=arm['fees']/nominal[:,:-1];arm['moved_fraction']=arm['moved']/nominal[:,:-1]
    return arm,w,dict(group_id='constructed',kind='reference',policy='same_target_annual')


class StressAuditTests(unittest.TestCase):
    def test_five_known_fee_rates_independent_implicit_equation(self):
        for rate in (0.,.00125,.0025,.005,.01):
            with self.subTest(rate=rate):
                arm,w,g=reference(rate);a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),w,g,a.base.Checks(),rate)
                f=100*rate*.5/(1+rate*.5)
                self.assertAlmostEqual(arm['fees'][0,0],f,places=13)
                self.assertEqual(arm['fees'][0,1:].sum(),0.)

    def test_double_fee_or_wrong_configured_fee_is_detected(self):
        arm,w,g=reference(.005);arm['fees'][0,0]*=2
        with self.assertRaises(AssertionError):a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),w,g,a.base.Checks(),.005)
        arm,w,g=reference(.005)
        with self.assertRaises(AssertionError):a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),w,g,a.base.Checks(),.0025)

    def test_zero_fee_still_records_trade_and_no_terminal_liquidation(self):
        arm,w,g=reference(0);self.assertTrue(arm['trades'][0,0]);self.assertEqual(arm['fees'].sum(),0.)
        self.assertAlmostEqual(arm['real_wealth'][0,-1],100.)

    def test_actual_inflation_shift_changes_cpi_and_real_wealth_only(self):
        arm,w,g=reference(.0025,.02)
        a.physical(arm,np.zeros((1,30,3)),np.full((1,30),.02),w,g,a.base.Checks(),.0025)
        self.assertAlmostEqual(arm['cpi'][0,-1],1.02**30)
        self.assertAlmostEqual(arm['real_wealth'][0,-1],arm['nominal_wealth'][0,-1]/1.02**30)
        with self.assertRaises(AssertionError):a.physical(arm,np.zeros((1,30,3)),np.zeros((1,30)),w,g,a.base.Checks(),.0025)

    def test_initial_fulltarget_support_and_pooled_refusal(self):
        r=np.zeros((3,30,3));pi=np.zeros((3,30));r[:,1:]=-.7
        spec=dict(release_fraction=.5)
        calendar=a.initial_admission([.45,.05,.5],spec,r,pi,a.SETTINGS[0])
        pooled=a.initial_admission([.45,.05,.5],spec,r,pi,a.SETTINGS[5])
        self.assertTrue(calendar['allowed']);self.assertFalse(pooled['allowed'])
        self.assertEqual(calendar['budget_floor'],80.);self.assertEqual(calendar['support_members'],3)
        self.assertEqual(pooled['support_members'],90)
        self.assertIsNone(a.initial_admission(None,spec,r,pi,a.SETTINGS[0]))
        self.assertIsNone(a.initial_admission([.45,.05,.5],None,r,pi,a.SETTINGS[0]))

    def test_initial_rule_cannot_use_shocked_actual_future(self):
        train=np.zeros((3,30,3));pi=np.zeros((3,30));spec=dict(release_fraction=.75)
        x=a.initial_admission([.45,.05,.5],spec,train,pi,a.SETTINGS[0])
        y=a.initial_admission([.45,.05,.5],spec,train,pi,a.SETTINGS[6])
        self.assertEqual(x['allowed'],y['allowed']);np.testing.assert_array_equal(x['posttrade'],y['posttrade'])
        self.assertEqual(x['budget_floor'],70.)

    def test_base_retained_exact_identity_including_nan_dtype_shape(self):
        canonical=dict(holdings=np.ones((1,2,3)),score=np.array([np.nan]),candidate_score=np.ones((1,13)),training_real_gross=np.ones((30,3,3)))
        thin={k:v for k,v in canonical.items() if k in ('holdings','score')}
        self.assertEqual(a.baseline_exact(thin,canonical,a.base.Checks(),'x')['fields'],2)
        for corrupt in (dict(thin,score=np.array([0.])),dict(thin,holdings=np.ones((1,2,3),dtype='float32')),dict(thin,score=np.array([[np.nan]]))):
            with self.assertRaises(AssertionError):a.baseline_exact(corrupt,canonical,a.base.Checks(),'x')

    def test_empty_financial_comparisons_all42_keep_null_and_alloriginal_refs(self):
        rows=[]
        for setting in a.SETTINGS:
            for cohort in ('outer_OOF','full_resubstitution'):
                for policy in a.OBJECTS:
                    row=dict(setting_id=setting['setting_id'],cohort=cohort,policy=policy,scenario_id=1,assigned=False,review_reason='preinvestment_review')
                    rows.append(row)
        result=a.independent_comparisons(rows)
        self.assertEqual(len(result),42)
        for group in result:
            self.assertEqual(group['same_funded_ids'],[]);self.assertEqual(group['initial_noninvested_ids'],[1])
            self.assertIsNone(group['same_funded_model']['mean_log_growth'])
            self.assertIsNone(group['paired_differences']['mdd']['mean'])
            self.assertEqual(group['model_all_original']['original_n'],1)

    def test_complete168group6720row56summary42comparison_null_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'stress';parent=root/'parent';source.mkdir();parent.mkdir()
            (root/'jobs.json').write_text('{"jobs":[]}');(root/'fits.json').write_text('{}')
            contexts={};canonical_groups=[];canonical_arrays={};groups=[];arrays={};rows=[];witnesses=[]
            for context in range(6):
                ev=list(range(1,121)) if context==0 else list(range(1+24*(context-1),1+24*context))
                tr=list(range(1,121)) if context==0 else sorted(set(range(1,121))-set(ev))
                job=dict(job_id=f'c{context}_refit',training_ids=tr,evaluation_ids=ev)
                selected=dict(group_id=f'base{context}model',array_names=[]);refs={}
                for policy in a.REFS:
                    gid=f'base{context}{policy}';target=[1/3]*3 if policy=='exact_thirds_annual' else [.45,.05,.5]
                    arm,_,_=reference(.0025,target=target);arm={k:np.repeat(v,len(ev),axis=0) for k,v in arm.items()}
                    refs[policy]=dict(group_id=gid,array_names=list(arm))
                    canonical_arrays.update({gid+'__'+k:v for k,v in arm.items()})
                contexts[context]=dict(job=job,target=[.45,.05,.5],spec=None,null_reason='no_inner_feasible_candidate',selected=selected,references=refs)
            for setting in a.SETTINGS:
                for context,ctx in contexts.items():
                    ev=ctx['job']['evaluation_ids'];cohort='full_resubstitution' if context==0 else 'outer_OOF'
                    witnesses.append(dict(context_id=context,setting_id=setting['setting_id'],selected_candidate=None,status='no_inner_feasible_candidate',witness=None))
                    for policy in a.OBJECTS:
                        gid=f"{setting['setting_id']}_{context}_{policy}"
                        funded=policy!='state_budget_learner';common=dict(setting_id=setting['setting_id'],cohort=cohort,context_id=context,job_id=ctx['job']['job_id'],policy=policy)
                        arm={}
                        if funded:
                            target=[1/3]*3 if policy=='exact_thirds_annual' else ctx['target']
                            arm,_,_=reference(setting['cost_rate'],setting['actual_inflation_shift'],target)
                            if policy=='same_target_B':
                                arm['pending_at_decision']=arm['ever_breached'][:,:-1].copy()
                                arm['target_execution']=~arm['pending_at_decision']
                                arm['actions']=np.where(arm['pending_at_decision'],'hold','restore')
                            arm={k:np.repeat(v,len(ev),axis=0) for k,v in arm.items()}
                            arrays.update({gid+'__'+k:v for k,v in arm.items()})
                        groups.append(dict(group_id=gid,**common,training_ids=ctx['job']['training_ids'],scenario_ids=ev,selected_candidate=None,own_fit_target=ctx['target'],status='hypothetical_reference' if funded else ctx['null_reason'],financial_available=funded,setting=setting,array_names=list(arm),omitted_diagnostic_arrays=[]))
                        for i,sid in enumerate(ev):
                            row=dict(**common,scenario_id=sid,assigned=funded)
                            if funded:row.update(a.mitigation.path_metrics(arm,i))
                            else:row.update(review_reason=ctx['null_reason'],**dict.fromkeys(a.mitigation.FIN))
                            rows.append(row)
            summaries=[]
            for setting in a.SETTINGS:
                for cohort in ('outer_OOF','full_resubstitution'):
                    for policy in a.OBJECTS:
                        subset=[r for r in rows if r['setting_id']==setting['setting_id'] and r['cohort']==cohort and r['policy']==policy]
                        summaries.append(dict(setting_id=setting['setting_id'],cohort=cohort,policy=policy,**a.mitigation.summary(subset)))
            pd.DataFrame(rows).to_csv(source/'path_metrics.csv.gz',index=False)
            pd.DataFrame(summaries).to_csv(source/'summary.csv.gz',index=False)
            a.writej(source/'trace_groups.json',groups);a.writej(source/'initial_witnesses.json',witnesses)
            a.writej(source/'cohort_comparisons.json',a.independent_comparisons(rows))
            a.writej(parent/'selections.json',{});a.writej(parent/'trace_groups.json',[])
            np.savez_compressed(source/'traces.npz',**arrays);np.savez_compressed(parent/'traces.npz',**canonical_arrays)
            with patch.object(a,'ROOT',root),patch.object(a,'SOURCE',source),patch.object(a,'independent_contexts',return_value=contexts):
                result=a.audit_numeric(dict(inputs=dict(job_protocol='jobs.json',static_refits='fits.json')),parent,np.zeros((120,30,3)),np.zeros((120,30)),a.base.Checks())
            self.assertEqual((result['groups'],result['path_rows'],result['summary_rows'],result['cohort_comparisons']),(168,6720,56,42))
            self.assertEqual(result['financial_path_rows'],5040);self.assertEqual(result['null_path_rows'],1680)
            self.assertEqual(result['initial_support_checks'],[])

    def test_frozen_ledger_only_changes_cost_parameter(self):
        original=Path(a.mitigation.__file__).read_text();actual=Path(a.__file__).read_text()
        def body(source):
            tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='physical')
            return ast.get_source_segment(source,node)
        expected=body(original).replace('def physical(arm,r,pi,target,group,c):','def physical(arm,r,pi,target,group,c,cost):').replace(".0025*moved,label+'/implicit fee equation once'","cost*moved,label+'/implicit fee equation once'")
        self.assertEqual(body(actual),expected)
        modules=[x.module or '' for x in ast.walk(ast.parse(actual)) if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any('experimental_execution' in m or 'safeinvest.core' in m or m.startswith('run_state') for m in modules))
        self.assertIn('range(1,121),allow_final=False',actual)

    def test_missing_freeze_stops_before_any_npz_or_market(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);p=folder/'plan';f=folder/'freeze';pr=folder/'producer';pf=folder/'producerfreeze'
            for path in (p,f,pr,pf):path.write_text('{}')
            with patch.object(a,'PLAN',p),patch.object(a,'FREEZE',f),patch.object(a,'PROTOCOL',pr),patch.object(a,'PRODUCER_FREEZE',pf),patch.object(a.np,'load') as loader:
                with self.assertRaises(RuntimeError):a.authorize()
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
