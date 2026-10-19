"""130-job synthetic constrained-service pipeline checks; no market/cache I/O.

Mock economic diagnostics are predetermined and deliberately include a
negative inception Q advantage. Physical mock accounts come from a pure
old drift helper; these tests validate orchestration, not forecast truth.
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
import run_constrained_fan_execution as m
import tests.test_benefit_cost_pipeline as old_fixture
from experimental_execution.simple_drift_execution import simulate_drift as physical_mock


class ConstrainedFanPipelineTests(unittest.TestCase):
    @staticmethod
    def fixture(assignment='Balanced'):
        p,b,r,pi,ids=old_fixture.BenefitCostPipelineTests.fixture(assignment)
        p.update(methods=deepcopy(m.METHODS),primary_method=m.PRIMARY,local_alpha=.1)
        p['expected_rows']=dict(training_paths=10080,training_summary=130,
            inner_validation_paths=2520,inner_reference_paths=2520,
            inner_gate_summaries=26,client_paths=2160,fit_admission=26)
        return p,b,r,pi,ids

    @staticmethod
    def engine(*,fail_training=False,fail_inner=False,fail_training_keys=(),reference_bad_ids=()):
        cache={};calls=[];bcalls=[];bad_ids=set(reference_bad_ids);failed=set(fail_training_keys)
        def actual_ids(r):return tuple(np.rint(r[:,0,0]/1e-6).astype(int))
        def arm(n,target,bad):
            key=(n,tuple(target),tuple(bad))
            if key not in cache:
                fabricated=np.zeros((n,30,3));fabricated[np.asarray(bad,bool),-1,:]=-.5
                a=dict(physical_mock(fabricated,np.zeros((n,30)),target,0.,L=.4,horizon=30,cost_rate=.0025))
                a['qg_hold']=np.zeros((n,30));a['qg_restore']=np.full((n,30),.03)
                a['qg_restore'][:,0]=-5. # Not a deployment rejection condition.
                a['qg_restore'][:,1:3]=-.02
                a['gate_reason']=np.full((n,30),'growth_improvement',dtype='U40')
                a['gate_reason'][:,0]='mandatory_initial_restore'
                a['gate_reason'][:,1:3]='least_violation_restore'
                a['gate_reason'][:,3]='risk_constraint_repair' # Positive net.
                a['economic_improvement']=np.ones((n,30),bool);a['economic_improvement'][:,:3]=False
                a['risk_cost_justification']=np.zeros((n,30),bool);a['risk_cost_justification'][:,1:3]=True
                a['local_model_infeasible']=np.zeros((n,30),bool);a['local_model_infeasible'][:,:3]=True
                a['local_constraint_applied']=np.ones((n,30),bool);a['local_constraint_applied'][:,0]=False
                a['repair_meets_cap']=np.ones((n,30),bool);a['repair_meets_cap'][:,:3]=False
                # Poison old fields to ensure no inherited economic gate is
                # consulted by the new complete-risk admission procedure.
                a['mean_net_log_gain']=np.full((n,30),-5.)
                a['initial_economic_eligible']=False
                cache[key]=a
            return cache[key]
        def controller(r,pi,tr,target,policy):
            self_ids=actual_ids(tr[0]);training=np.array_equal(r,tr[0]);calls.append((policy,actual_ids(r),self_ids,tuple(target),training))
            bad=np.zeros(len(r),bool)
            if training and (fail_training or self_ids in failed):bad[0]=True
            if fail_inner and not training:bad[:]=True
            return arm(len(r),target,bad)
        def baseline(r,pi,target,threshold,**kwargs):
            ids=actual_ids(r);bcalls.append((ids,tuple(target),threshold))
            return arm(len(r),target,[sid in bad_ids for sid in ids])
        return controller,baseline,calls,bcalls

    def execute(self,p,b,r,pi,ids,**options):
        controller,baseline,calls,bcalls=self.engine(**options)
        with patch.object(m,'simulate',side_effect=controller),patch.object(m,'simulate_drift',side_effect=baseline):
            out=m.run_pipeline(p,b,r,pi,ids)
        return out,calls,bcalls

    def assert_inventory(self,out,p):
        self.assertEqual({k:len(v) for k,v in out['tables'].items()},p['expected_rows'])
        self.assertEqual(len(out['choices']),130);self.assertEqual(len(out['completed_jobs']),130)
        f=pd.DataFrame(out['tables']['client_paths'])
        self.assertEqual(len(f.loc[f.scope.eq('outer_oof')]),1800)
        self.assertEqual(len(f.loc[f.scope.eq('in_sample_full120')]),360)
        self.assertFalse(f[['scope','repeat_id','outer_fold','context','scenario_id','policy']].duplicated().any())
        self.assertTrue(f.loc[f.context_id.eq(0),'scope'].eq('in_sample_full120').all())
        self.assertTrue(f.loc[~f.context_id.eq(0),'scope'].eq('outer_oof').all())
        for _,g in f.loc[f.policy.eq(m.PRIMARY)].groupby('context'):
            self.assertEqual(g.assigned.nunique(),1);self.assertEqual(g.execution_status.nunique(),1)
            self.assertEqual(len(g),120 if g.context_id.iloc[0]==0 else 24)

    def test_all130_auto_with_negative_initial_Q_no_short_growth_gate_and_no_other_method(self):
        p,b,r,pi,ids=self.fixture();engine,baseline,calls,bcalls=self.engine();guards=[];done=[]
        with patch.object(m,'simulate',side_effect=engine),patch.object(m,'simulate_drift',side_effect=baseline):
            out=m.run_pipeline(p,b,r,pi,ids,guard=lambda:guards.append(1),on_job=done.append)
        self.assert_inventory(out,p);self.assertEqual(len(guards),260);self.assertEqual(len(done),130)
        self.assertEqual(len(calls),260);self.assertEqual(len(bcalls),260)
        self.assertEqual(len(out['groups']['actual']),78);self.assertEqual(len(out['groups']['validation']),208)
        self.assertTrue(all(a['initial_admitted'] for a in out['tables']['fit_admission']))
        self.assertTrue(all(g['eligible'] for g in out['tables']['inner_gate_summaries']))
        self.assertTrue(all(set(q['methods'])=={m.PRIMARY} for q in out['choices'].values()))
        self.assertTrue(all(g['metadata']['policy'] in {m.PRIMARY,*m.REFERENCES} for groups in out['groups'].values() for g in groups.values()))
        for g in out['groups']['validation'].values():self.assertFalse(set(g['scenario_ids'])&set(g['training_ids']))
        for g in out['groups']['actual'].values():
            if g['metadata']['context_id']==0:self.assertEqual(g['scenario_ids'],g['training_ids'])
            else:self.assertFalse(set(g['scenario_ids'])&set(g['training_ids']))

    def test_repair_cost_semantics_are_separate_from_gate_reason_and_initial_is_not_optional(self):
        engine,_,_,_=self.engine();r=np.zeros((2,30,3));r[:,0,0]=[1e-6,2e-6]
        a=engine(r,np.zeros((2,30)),(r,np.zeros((2,30))),[.4,.3,.3],m.PRIMARY)
        s=m.summary_rows(a,2,m.PRIMARY,{'context':'synthetic'})
        self.assertEqual(s['optional_restores'],58)
        self.assertEqual(s['risk_repair_restores'],4) # Cost justification, not2 mechanism labels.
        self.assertEqual(s['economic_improvement_restores'],54)
        self.assertEqual(s['growth_sacrificing_restores'],4) # Negative inception excluded.
        self.assertEqual(s['locally_infeasible_decisions'],4)
        self.assertEqual(s['economic_improvement_restores']+s['risk_repair_restores'],s['optional_restores'])

    def test_complete_training_failure_has_no_B_or_growth_only_fallback(self):
        p,b,r,pi,ids=self.fixture();out,calls,_=self.execute(p,b,r,pi,ids,fail_training=True)
        self.assert_inventory(out,p);f=pd.DataFrame(out['tables']['client_paths']);q=f.loc[f.policy.eq(m.PRIMARY)]
        self.assertFalse(q.assigned.any());self.assertTrue(q[list(m.helpers.FINANCIAL)].isna().all().all())
        self.assertTrue(f.loc[f.policy.isin(m.REFERENCES),'assigned'].all())
        self.assertTrue(all(not d['training_eligible'] and d['training_breach_cap']==0 for q in out['choices'].values() for d in q['methods'].values()))
        self.assertTrue(all(training for *_,training in calls))
        self.assertTrue(all(g['auto_inner_n']==0 and g['conditional_breach_probability'] is None for g in out['tables']['inner_gate_summaries']))

    def test_inner_absolute_failure_blocks_outer24_and_full120_before_evaluation(self):
        p,b,r,pi,ids=self.fixture();out,calls,_=self.execute(p,b,r,pi,ids,fail_inner=True)
        self.assert_inventory(out,p);f=pd.DataFrame(out['tables']['client_paths']);q=f.loc[f.policy.eq(m.PRIMARY)]
        self.assertFalse(q.assigned.any());self.assertTrue(q[list(m.helpers.FINANCIAL)].isna().all().all())
        self.assertTrue(all(g['reason']=='inner_absolute_risk_exceeds_reference' for g in out['tables']['inner_gate_summaries']))
        self.assertTrue(all(len(actual) in (24,30) and len(train) in (72,90) for _,actual,train,_,training in calls if not training))
        self.assertEqual(len(out['groups']['actual']),52)

    def test_own_U_excludes_training_rejected_inner_block_from_B_denominator(self):
        p,b,r,pi,ids=self.fixture();first=p['jobs'][0];key=tuple(first['training_ids']);refused=set(first['evaluation_ids'])
        out,_,_=self.execute(p,b,r,pi,ids,fail_training_keys=(key,),reference_bad_ids=refused)
        g=next(g for g in out['tables']['inner_gate_summaries'] if g['context']==first['parent_context'])
        self.assertEqual(g['auto_inner_n'],72);self.assertEqual(g['selected_breach_count'],0)
        self.assertEqual(g['same_U_B_breach_count'],0);self.assertTrue(g['eligible'])
        self.assertEqual(g['auto_set_scenario_ids'],sorted(set(p['contexts'][0]['training_ids'])-refused))

    def test_Q1_priority_and_all_expected_NULL_rows_without_controller_invocation(self):
        for label in ('Review','Growth'):
            with self.subTest(label=label):
                p,b,r,pi,ids=self.fixture(label);out,calls,bcalls=self.execute(p,b,r,pi,ids)
                self.assert_inventory(out,p);self.assertEqual(calls,[]);self.assertEqual(bcalls,[])
                f=pd.DataFrame(out['tables']['client_paths']);q=f.loc[f.policy.eq(m.PRIMARY)]
                self.assertTrue(q.execution_status.eq(m.joint.case_of(label)).all())
                self.assertTrue(q[list(m.helpers.FINANCIAL)].isna().all().all())
                self.assertFalse(f.loc[f.policy.eq(m.REFERENCES[0]),'assigned'].any())
                self.assertTrue(f.loc[f.policy.eq(m.REFERENCES[1]),'assigned'].all())
                self.assertTrue(all(s['breach_count'] is None for s in out['tables']['training_summary']))

    def test_full120_static_no_holdout_cache_is_not_rewritten_as_dynamic120_evaluation(self):
        p,b,r,pi,ids=self.fixture();original=deepcopy(b['c0_refit'])
        clean=m.restrict_q1_bundles(p,b)
        self.assertEqual(clean['c0_refit']['refit']['outer_test_ids'],[]);self.assertEqual(b['c0_refit'],original)
        self.assertEqual(p['jobs'][-1]['evaluation_ids'],list(range(1,121)))
        b['c0_refit']['refit']['outer_test_ids']=list(range(1,121))
        with self.assertRaises(ValueError):m.restrict_q1_bundles(p,b)

    def test_one_method_local_alpha_and_cache_product_identity_are_fixed(self):
        p,b,r,pi,ids=self.fixture()
        for key,value in [('methods',{**m.METHODS,'another':'another'}),('primary_method','another'),('local_alpha',.2)]:
            with self.subTest(key=key):
                bad=deepcopy(p);bad[key]=value
                with self.assertRaises(ValueError):m.validate_protocol(bad)
        for field,value in [('weights',[.45,.25,.3]),('candidate_index',18),('candidate_original232_index',24)]:
            with self.subTest(field=field):
                bad=deepcopy(b);bad[p['jobs'][0]['job_id']]['q2_case']['product'][field]=value
                with patch.object(m,'simulate',side_effect=AssertionError('Must reject first')):
                    with self.assertRaises(ValueError):m.run_pipeline(p,bad,r,pi,ids)

    def test_fixed_complete_count_cap_no_economic_eligibility_field_required(self):
        self.assertTrue(m.training_decision({'breach_count':7},8,72)['training_eligible'])
        self.assertFalse(m.training_decision({'breach_count':8},8,72)['training_eligible'])
        self.assertFalse(m.training_decision({'breach_count':7},6,72)['training_eligible'])
        self.assertTrue(m.training_decision({'breach_count':12},12,120)['training_eligible'])
        self.assertEqual(m.training_decision({'breach_count':9,'initial_economic_eligible':False},9,120)['training_breach_cap'],9)
        for k in (True,-1,73,7.0,None):
            with self.subTest(k=k),self.assertRaises(ValueError):m.training_decision({'breach_count':k},8,72)

    def test_synthetic_main_export_hashes_no_final_loading_no_overwrite(self):
        p,b,r,pi,ids=self.fixture();engine,baseline,_,_=self.engine();digest=hashlib.sha256()
        for a in (ids,r,pi):digest.update(np.ascontiguousarray(a).tobytes())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=root/'docs/constrained_fan_execution_protocol.json';out=root/'results/model_review/constrained_fan_execution_v1';protocol.parent.mkdir(parents=True)
            for name in ('refit_stability_protocol.json','joint_liquidity_refit_protocol.json'):(root/'docs'/name).write_text((ROOT/'docs'/name).read_text())
            for folder,filename,selected in [('refit_stability_v1','static_refits.json.gz',{k:v for k,v in b.items() if not k.startswith('c0_')}),('joint_liquidity_refit_v1','static_refits.json',{k:v for k,v in b.items() if k.startswith('c0_')})]:
                base=root/'results/model_review'/folder;base.mkdir(parents=True)
                if filename.endswith('.gz'):(base/filename).write_bytes(gzip.compress(json.dumps(selected).encode(),mtime=0))
                else:(base/filename).write_text(json.dumps(selected))
                (base/'manifest.json').write_text(json.dumps({'status':'complete','development_arrays_sha256':digest.hexdigest()}))
            p.update(execution_ready=True,execution_sources_sha256={'synthetic.py':'a'*64},frozen_input_sha256={});protocol.write_text(json.dumps(p));loads=[]
            def load(path,requested,allow_final):
                self.assertEqual(list(requested),list(range(1,121)));self.assertIs(allow_final,False);loads.append(1)
                return r.copy(),pi.copy(),ids.copy()
            with patch.multiple(m,ROOT=root,PROTOCOL=protocol,OUT=out),patch.object(m.identity_helpers,'ROOT',root),patch.object(m,'source_manifest',return_value={'synthetic.py':'a'*64}),patch.object(m.helpers,'frozen_inputs',return_value={}),patch.object(m,'load_paths',side_effect=load),patch.object(m,'simulate',side_effect=engine),patch.object(m,'simulate_drift',side_effect=baseline):
                with redirect_stdout(io.StringIO()):m.main()
                with self.assertRaises(RuntimeError):m.main()
            self.assertEqual(loads,[1]);manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['training_tasks'],130)
            self.assertFalse(manifest['final_scenarios_loaded']);self.assertEqual(manifest['row_counts'],p['expected_rows'])
            self.assertEqual(manifest['sources'],manifest['sources_at_end'])
            for name,h in manifest['outputs_sha256'].items():self.assertEqual(m.sha(out/name),h)
            for f in out.glob('*.gz'):self.assertEqual(f.read_bytes()[4:8],bytes(4))
            self.assertEqual(m.read_json(out/'job_choices.json.gz')['c0_refit']['evaluation_ids'],list(range(1,121)))


if __name__=='__main__':unittest.main()
