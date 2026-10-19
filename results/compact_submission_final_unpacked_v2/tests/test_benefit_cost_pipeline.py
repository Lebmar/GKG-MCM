"""Independent synthetic orchestration/admission tests; no market I/O.

The fake controller quotes predetermined economic evidence and uses the old
pure physical drift kernel only to make valid account-shaped mock results.
It tests the new runner, not the new estimator's economic correctness.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research'),str(ROOT/'scripts')]
import run_benefit_cost_execution as m
from experimental_execution.benefit_cost_execution import InitialEconomicRejection
from experimental_execution.simple_drift_execution import simulate_drift as physical_mock


class BenefitCostPipelineTests(unittest.TestCase):
    @staticmethod
    def fixture(assignment='Balanced'):
        # Only previously frozen split/design JSON is read. All returns below
        # are fabricated, and no saved market/cache/result file is loaded.
        parent=json.loads((ROOT/'docs/refit_stability_protocol.json').read_text())
        full=json.loads((ROOT/'docs/joint_liquidity_refit_protocol.json').read_text())
        p=deepcopy(parent)
        fulljobs=[{**deepcopy(j),'repeat_id':0,'outer_fold':0} for j in full['jobs'][:5]]
        fullcontext=dict(context_id=0,name='full_development',repeat_id=0,outer_fold=0,
                         training_ids=list(range(1,121)),outer_test_ids=[],
                         inner_seed=fulljobs[-1]['static_inner_seed'],inner_folds=[
                             dict(inner_fold=j['dynamic_inner_fold'],train_ids=j['training_ids'],test_ids=j['evaluation_ids'])
                             for j in fulljobs[:4]])
        p.update(methods=deepcopy(m.METHODS),primary_method=m.PRIMARY,client='C',
                 contexts=parent['contexts']+[fullcontext],jobs=parent['jobs']+fulljobs,
                 static_contexts={**parent['static_contexts'],**{k:v for k,v in full['static_contexts'].items() if k.startswith('c0_')}},
                 repeat_count=5,outer_fits=25,training_tasks=130,q2_training_tasks=130,
                 initial_nominal_wealth=100,initial_weights=[0,0,1],references=list(m.REFERENCES),
                 execution_ready=False,budget_numeric_minutes=10,service_alpha=.1)
        p['expected_rows']=dict(training_paths=20160,training_summary=260,
                                inner_validation_paths=5040,inner_reference_paths=2520,
                                inner_gate_summaries=52,client_paths=2880,fit_admission=52)
        bundles={}
        menu=[dict(name='Conservative',candidate_index=8,candidate_original232_index=13,weights=[.2,.3,.5]),
              dict(name='Balanced',candidate_index=17,candidate_original232_index=23,weights=[.4,.3,.3]),
              dict(name='Growth',candidate_index=31,candidate_original232_index=57,weights=[.7,.2,.1])]
        product=next((q for q in menu if q['name']==assignment),None)
        for j in p['jobs']:
            case=m.joint.case_of(assignment)
            bundles[j['job_id']]=dict(context=deepcopy(p['static_contexts'][j['job_id']]),
                q2_case=dict(effective_assignment=assignment,public_status=case,status=case,
                             product=deepcopy(product) if assignment=='Balanced' else None,
                             q1_product=None if assignment=='Review' else deepcopy(product),
                             q1_review_kind='synthetic_referral' if assignment=='Review' else None,
                             q1_review_reasons=['synthetic_referral'] if assignment=='Review' else []),
                refit=dict(training_ids=j['training_ids'],outer_test_ids=[] if j['job_id']=='c0_refit' else j['evaluation_ids'],
                           effective_assignments={'C':assignment},raw_assignments={'C':assignment},
                           raw_fit=None if assignment=='Review' else dict(menu=deepcopy(menu)),
                           selection=dict(selected_gamma=.075)),
                training_price_selections={'poison':'Never reuse old price selection'},
                reference_training_breach_count=987654321)
        r=np.zeros((120,30,3));r[:,0,0]=np.arange(1,121)*1e-6
        return p,bundles,r,np.zeros((120,30)),np.arange(1,121)

    @staticmethod
    def engine(*,reject_primary=False,fail_training=False,fail_inner=False,reject_training_keys=(),reference_bad_ids=()):
        cache={};calls=[];bcalls=[];reject_training_keys=set(reject_training_keys);bad_ids=set(reference_bad_ids)
        def actual_ids(r):return tuple(np.rint(r[:,0,0]/1e-6).astype(int))
        def arm(n,target,bad):
            key=(n,tuple(target),tuple(bad))
            if key not in cache:
                fabricated=np.zeros((n,30,3));fabricated[np.asarray(bad,bool),-1,:]=-.5
                a=dict(physical_mock(fabricated,np.zeros((n,30)),target,0.,L=.4,horizon=30,cost_rate=.0025))
                a['mean_net_log_gain']=np.full((n,30),.01)
                a['mean_gross_log_gain']=np.full((n,30),.012)
                a['fee_log_penalty']=np.full((n,30),.002)
                a['risk_direction_pass']=np.ones((n,30),bool)
                cache[key]=a
            return cache[key]
        def controller(r,pi,tr,target,policy):
            own_ids=actual_ids(tr[0]);training=np.array_equal(r,tr[0]);calls.append((policy,actual_ids(r),own_ids,tuple(target),training))
            if policy==m.PRIMARY and (reject_primary or own_ids in reject_training_keys):
                raise InitialEconomicRejection(dict(economic_pass=np.array([False]),mean_net_log_gain=np.array([-.001]),mean_gross_log_gain=np.array([.001]),fee_log_penalty=np.array([.002])))
            bad=np.zeros(len(r),bool)
            if fail_training and training:bad[0]=True
            if fail_inner and not training:bad[:]=True
            return arm(len(r),target,bad)
        def baseline(r,pi,target,threshold,**kwargs):
            ids=actual_ids(r);bcalls.append((ids,tuple(target),threshold))
            return arm(len(r),target,[i in bad_ids for i in ids])
        return controller,baseline,calls,bcalls

    def execute(self,p,b,r,pi,ids,**options):
        controller,baseline,calls,bcalls=self.engine(**options)
        with patch.object(m,'simulate',side_effect=controller),patch.object(m,'simulate_drift',side_effect=baseline):
            result=m.run_pipeline(p,b,r,pi,ids)
        return result,calls,bcalls

    def assert_inventory(self,result,p):
        self.assertEqual({k:len(v) for k,v in result['tables'].items()},p['expected_rows'])
        self.assertEqual(len(result['choices']),130);self.assertEqual(len(result['completed_jobs']),130)
        for bucket in result['groups'].values():
            for group in bucket.values():
                self.assertIn('training_ids',group);self.assertNotIn('annual_prices_metadata',group)
        client=pd.DataFrame(result['tables']['client_paths'])
        self.assertEqual(len(client.loc[client.scope.eq('outer_oof')]),2400)
        self.assertEqual(len(client.loc[client.scope.eq('in_sample_full120')]),480)
        self.assertFalse(client[['scope','repeat_id','outer_fold','context','scenario_id','policy']].duplicated().any())
        self.assertTrue(client.loc[client.context_id.eq(0),'scope'].eq('in_sample_full120').all())
        self.assertTrue(client.loc[~client.context_id.eq(0),'scope'].eq('outer_oof').all())
        self.assertEqual(set(client.loc[client.context_id.eq(0),'scenario_id']),set(range(1,121)))
        for _,group in client.loc[client.policy.isin(m.METHODS)].groupby(['context','policy']):
            self.assertEqual(group.execution_status.nunique(),1)
            self.assertEqual(group.assigned.nunique(),1)
            self.assertEqual(len(group),120 if group.context_id.iloc[0]==0 else 24)

    def test_all130_jobs_and_full120_is_separate_in_sample_with_own_training(self):
        p,b,r,pi,ids=self.fixture();controller,baseline,calls,bcalls=self.engine();guards=[];done=[]
        with patch.object(m,'simulate',side_effect=controller),patch.object(m,'simulate_drift',side_effect=baseline):
            out=m.run_pipeline(p,b,r,pi,ids,guard=lambda:guards.append(True),on_job=done.append)
        self.assert_inventory(out,p);self.assertEqual(len(guards),260);self.assertEqual(len(done),130)
        self.assertEqual(len(calls),520);self.assertEqual(len(bcalls),260)
        self.assertEqual(len(out['groups']['actual']),104);self.assertEqual(len(out['groups']['validation']),312)
        self.assertTrue(all(g['eligible'] for g in out['tables']['inner_gate_summaries']))
        self.assertTrue(all(a['initial_admitted'] for a in out['tables']['fit_admission']))
        self.assertTrue(all(set(q['methods'])==set(m.METHODS) for q in out['choices'].values()))
        for jid,group in out['groups']['validation'].items():
            self.assertFalse(set(group['scenario_ids'])&set(group['training_ids']))
        for jid,group in out['groups']['actual'].items():
            if group['metadata']['context_id']==0:self.assertEqual(group['scenario_ids'],group['training_ids'])
            else:self.assertFalse(set(group['scenario_ids'])&set(group['training_ids']))

    def test_nonpositive_initial_principal_is_review_and_sensitivity_never_fills_it(self):
        p,b,r,pi,ids=self.fixture();out,calls,_=self.execute(p,b,r,pi,ids,reject_primary=True)
        self.assert_inventory(out,p)
        f=pd.DataFrame(out['tables']['client_paths']);primary=f.loc[f.policy.eq(m.PRIMARY)]
        self.assertFalse(primary.assigned.any());self.assertTrue(primary[list(m.helpers.FINANCIAL)].isna().all().all())
        self.assertTrue(f.loc[f.policy.eq('benefit_cost_remaining_H'),'assigned'].all())
        self.assertTrue(f.loc[f.policy.isin(m.REFERENCES),'assigned'].all())
        for q in out['choices'].values():
            self.assertFalse(q['methods'][m.PRIMARY]['training_eligible'])
            self.assertLess(q['methods'][m.PRIMARY]['initial_mean_net_log_gain'],0)
        gates=pd.DataFrame(out['tables']['inner_gate_summaries']);zero=gates.loc[gates.policy.eq(m.PRIMARY)]
        self.assertTrue(zero.auto_inner_n.eq(0).all());self.assertTrue(zero.conditional_breach_probability.isna().all())
        self.assertTrue(zero.same_U_B_breach_count.isna().all());self.assertTrue(zero.reason.eq('no_inner_auto_evidence').all())
        self.assertFalse(any(not training for policy,_,_,_,training in calls if policy==m.PRIMARY))

    def test_complete_training_risk_failure_never_gets_a_B_fallback(self):
        p,b,r,pi,ids=self.fixture();out,calls,_=self.execute(p,b,r,pi,ids,fail_training=True)
        self.assert_inventory(out,p)
        f=pd.DataFrame(out['tables']['client_paths']);learn=f.loc[f.policy.isin(m.METHODS)]
        self.assertFalse(learn.assigned.any());self.assertTrue(learn[list(m.helpers.FINANCIAL)].isna().all().all())
        self.assertTrue(f.loc[f.policy.isin(m.REFERENCES),'assigned'].all())
        self.assertTrue(all(not d['training_eligible'] for q in out['choices'].values() for d in q['methods'].values()))
        self.assertTrue(all(d['training_breach_cap']==0 for q in out['choices'].values() for d in q['methods'].values()))
        self.assertTrue(all(training for *_,training in calls))

    def test_inner_gate_blocks_all_future24_and120_before_their_simulation(self):
        p,b,r,pi,ids=self.fixture();out,calls,_=self.execute(p,b,r,pi,ids,fail_inner=True)
        self.assert_inventory(out,p)
        self.assertTrue(all(not g['eligible'] for g in out['tables']['inner_gate_summaries']))
        self.assertTrue(all(g['reason']=='inner_absolute_risk_exceeds_reference' for g in out['tables']['inner_gate_summaries']))
        f=pd.DataFrame(out['tables']['client_paths']);learn=f.loc[f.policy.isin(m.METHODS)]
        self.assertFalse(learn.assigned.any());self.assertTrue(learn[list(m.helpers.FINANCIAL)].isna().all().all())
        # Only inner validation (24/72 or30/90) calls may be held-out; there
        # must be no outer24/96 or full120/120 evaluation after its gate fails.
        self.assertTrue(all(len(actual) in (24,30) and len(train) in (72,90) for _,actual,train,_,training in calls if not training))
        self.assertEqual(len(out['groups']['actual']),52)

    def test_same_U_reference_excludes_refused_inner_block(self):
        p,b,r,pi,ids=self.fixture();first=p['jobs'][0]
        key=tuple(first['training_ids']);refused=set(first['evaluation_ids'])
        out,_,_=self.execute(p,b,r,pi,ids,reject_training_keys=(key,),reference_bad_ids=refused)
        rows=pd.DataFrame(out['tables']['inner_gate_summaries']);g=rows.loc[rows.context.eq(first['parent_context'])]
        principal=g.loc[g.policy.eq(m.PRIMARY)].iloc[0];sensitivity=g.loc[g.policy.eq('benefit_cost_remaining_H')].iloc[0]
        self.assertEqual(principal.auto_inner_n,72);self.assertEqual(principal.same_U_B_breach_count,0)
        self.assertEqual(principal.selected_breach_count,0);self.assertTrue(principal.eligible)
        self.assertEqual(sensitivity.auto_inner_n,96);self.assertEqual(sensitivity.same_U_B_breach_count,24)
        self.assertEqual(principal.auto_set_scenario_ids,sorted(set(p['contexts'][0]['training_ids'])-refused))

    def test_q1_review_outside_case_priority_and_null_complete_denominators(self):
        for label in ('Review','Growth'):
            with self.subTest(label=label):
                p,b,r,pi,ids=self.fixture(label);out,calls,bcalls=self.execute(p,b,r,pi,ids)
                self.assert_inventory(out,p);self.assertEqual(calls,[]);self.assertEqual(bcalls,[])
                f=pd.DataFrame(out['tables']['client_paths']);learn=f.loc[f.policy.isin(m.METHODS)]
                self.assertTrue(learn.execution_status.eq(m.joint.case_of(label)).all())
                self.assertTrue(learn[list(m.helpers.FINANCIAL)].isna().all().all())
                self.assertFalse(f.loc[f.policy.eq(m.REFERENCES[0]),'assigned'].any())
                self.assertTrue(f.loc[f.policy.eq(m.REFERENCES[1]),'assigned'].all())
                self.assertEqual(len(out['groups']['validation']),0)

    def test_full120_static_cache_has_no_holdout_and_remains_immutable(self):
        p,b,r,pi,ids=self.fixture();original=deepcopy(b['c0_refit'])
        self.assertEqual(b['c0_refit']['refit']['outer_test_ids'],[])
        self.assertEqual(p['jobs'][-1]['evaluation_ids'],list(range(1,121)))
        restricted=m.restrict_q1_bundles(p,b)
        self.assertEqual(b['c0_refit'],original)
        self.assertEqual(restricted['c0_refit']['refit']['outer_test_ids'],[])
        self.assertEqual(p['jobs'][-1]['evaluation_ids'],list(range(1,121)))
        tampered=deepcopy(b);tampered['c0_refit']['refit']['outer_test_ids']=list(range(1,121))
        with self.assertRaises(ValueError):m.restrict_q1_bundles(p,tampered)

    def test_training_integer_cap_initial_priority_and_absolute_relative_boundaries(self):
        row=dict(initial_economic_eligible=True,breach_count=7)
        self.assertTrue(m.training_decision(row,8,72)['training_eligible'])
        self.assertFalse(m.training_decision({**row,'breach_count':8},8,72)['training_eligible'])
        self.assertFalse(m.training_decision(row,6,72)['training_eligible'])
        self.assertEqual(m.training_decision(row,6,72)['training_breach_cap'],6)
        self.assertTrue(m.training_decision({**row,'breach_count':12},12,120)['training_eligible'])
        init=m.training_decision(dict(initial_economic_eligible=False,breach_count=None),12,120)
        self.assertEqual(init['execution_reason'],'mandatory_initial_restore_economic_nonpositive')
        for k in (True,-1,73,7.0,None):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):m.training_decision({**row,'breach_count':k},8,72)

    def test_own_q1_identity_and_original_split_tampering_rejected_before_any_simulation(self):
        for field,value in (('weights',[.45,.25,.3]),('candidate_index',18),('candidate_original232_index',24)):
            with self.subTest(field=field):
                p,b,r,pi,ids=self.fixture();b[p['jobs'][0]['job_id']]['q2_case']['product'][field]=value
                with patch.object(m,'simulate',side_effect=AssertionError('Must reject before simulation')):
                    with self.assertRaises(ValueError):m.run_pipeline(p,b,r,pi,ids)
        p,b,r,pi,ids=self.fixture();p['jobs'][-1]['evaluation_ids']=[1]
        with self.assertRaises(ValueError):m.validate_protocol(p)

    def test_synthetic_main_no_market_final_parse_no_overwrite_and_deterministic_gzip(self):
        p,b,r,pi,ids=self.fixture();controller,baseline,_,_=self.engine();digest=hashlib.sha256()
        for a in (ids,r,pi):digest.update(np.ascontiguousarray(a).tobytes())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=root/'docs/benefit_cost_execution_protocol.json';out=root/'results/model_review/benefit_cost_execution_v1'
            protocol.parent.mkdir(parents=True)
            for name in ('refit_stability_protocol.json','joint_liquidity_refit_protocol.json'):
                (root/'docs'/name).write_text((ROOT/'docs'/name).read_text())
            for folder,filename,selected in [('refit_stability_v1','static_refits.json.gz',{k:v for k,v in b.items() if not k.startswith('c0_')}),
                                              ('joint_liquidity_refit_v1','static_refits.json',{k:v for k,v in b.items() if k.startswith('c0_')})]:
                base=root/'results/model_review'/folder;base.mkdir(parents=True)
                if filename.endswith('.gz'):(base/filename).write_bytes(gzip.compress(json.dumps(selected).encode(),mtime=0))
                else:(base/filename).write_text(json.dumps(selected))
                (base/'manifest.json').write_text(json.dumps(dict(status='complete',development_arrays_sha256=digest.hexdigest())))
            p.update(execution_ready=True,execution_sources_sha256={'synthetic.py':'a'*64},frozen_input_sha256={})
            protocol.write_text(json.dumps(p));loads=[]
            def load(path,requested,allow_final):
                self.assertEqual(list(requested),list(range(1,121)));self.assertIs(allow_final,False);loads.append(1)
                return r.copy(),pi.copy(),ids.copy()
            with patch.multiple(m,ROOT=root,PROTOCOL=protocol,OUT=out),patch.object(m,'source_manifest',return_value={'synthetic.py':'a'*64}),patch.object(m.helpers,'frozen_inputs',return_value={}),patch.object(m,'load_paths',side_effect=load),patch.object(m,'simulate',side_effect=controller),patch.object(m,'simulate_drift',side_effect=baseline):
                with redirect_stdout(io.StringIO()):m.main()
                with self.assertRaises(RuntimeError):m.main()
            self.assertEqual(loads,[1]);manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['training_tasks'],130)
            self.assertFalse(manifest['final_scenarios_loaded']);self.assertEqual(manifest['sources'],manifest['sources_at_end'])
            self.assertEqual(manifest['row_counts'],p['expected_rows'])
            for name,h in manifest['outputs_sha256'].items():self.assertEqual(m.sha(out/name),h)
            for f in out.glob('*.gz'):self.assertEqual(f.read_bytes()[4:8],bytes(4))
            choices=m.read_json(out/'job_choices.json.gz');self.assertEqual(len(choices),130)
            self.assertEqual(choices['c0_refit']['evaluation_ids'],list(range(1,121)))


if __name__=='__main__':unittest.main()
