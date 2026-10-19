"""125-job synthetic orchestration and selection/gate boundaries, no market I/O."""
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
import run_simple_execution_comparison as m
from experimental_execution.simple_drift_execution import simulate_drift


class SimpleExecutionPipelineTests(unittest.TestCase):
    @staticmethod
    def fixture(assignment='Balanced'):
        # Protocol contains identities/design only, no financial observations.
        p=json.loads((ROOT/'docs/simple_execution_comparison_protocol.json').read_text())
        bundles={};menu=[dict(name='Conservative',candidate_index=8,candidate_original232_index=13,weights=[.2,.3,.5]),
            dict(name='Balanced',candidate_index=17,candidate_original232_index=23,weights=[.4,.3,.3]),
            dict(name='Growth',candidate_index=31,candidate_original232_index=57,weights=[.7,.2,.1])]
        product=next((item for item in menu if item['name']==assignment),None)
        for job in p['jobs']:
            case=m.joint.case_of(assignment);static=p['static_contexts'][job['job_id']]
            bundles[job['job_id']]={
                'context':deepcopy(static),
                'q2_case':dict(effective_assignment=assignment,public_status=case,status=case,
                    product=deepcopy(product) if assignment=='Balanced' else None,
                    q1_product=None if assignment=='Review' else deepcopy(product),
                    q1_review_kind='synthetic_referral' if assignment=='Review' else None,
                    q1_review_reasons=['synthetic_referral'] if assignment=='Review' else []),
                'refit':dict(training_ids=job['training_ids'],outer_test_ids=job['evaluation_ids'],
                    effective_assignments={'C':assignment},raw_assignments={'C':assignment},
                    raw_fit=None if assignment=='Review' else {'menu':deepcopy(menu)},selection={'selected_gamma':.075}),
                'training_price_selections':{'poison':'Old nine-price evidence must never be inspected'},
                'reference_training_breach_count':987654321}
        return p,bundles,np.zeros((120,30,3)),np.zeros((120,30)),np.arange(1,121)

    @staticmethod
    def engine(force_inner_failure=False):
        cache={};calls=[]
        def run(r,pi,target,threshold,**kw):
            calls.append((len(r),threshold))
            key=(len(r),threshold,tuple(target),force_inner_failure)
            if key not in cache:
                synthetic=r.copy()
                # Override only mock evaluation paths to force absolute-gate
                # rejection. Physical-kernel correctness is tested elsewhere;
                # this orchestration mock does not claim economic evidence.
                if force_inner_failure and len(r)==24:synthetic[:,-1,:]=-.5
                cache[key]=simulate_drift(synthetic,pi,target,threshold,**kw)
            return cache[key]
        return run,calls

    def assert_inventory(self,result,p):
        for name,n in p['expected_rows'].items():self.assertEqual(len(result['tables'][name]),n,name)
        self.assertEqual(len(result['completed_jobs']),125)
        self.assertEqual(len(result['training_choices']),125)
        self.assertEqual(len(result['context_selections']),25)
        for scope in result['trace_groups'].values():
            for group in scope.values():
                self.assertNotIn('q_subset_scenario_id',group)
                self.assertNotIn('annual_prices_metadata',group)
                self.assertIn('job_id',group['metadata'])
        for archive in (result['actual_traces'],result['validation_traces']):
            self.assertFalse(any('__qg_' in key or '__annual_prices' in key for key in archive))

    def test_complete125_all_automatic_deduplicates_selected_physical_evaluation(self):
        p,b,r,pi,ids=self.fixture();engine,calls=self.engine();guards=[];jobs=[]
        result=m.run_pipeline(p,b,r,pi,ids,simulator=engine,guard=lambda:guards.append(1),on_job=jobs.append)
        self.assert_inventory(result,p);self.assertEqual(len(guards),250);self.assertEqual(len(jobs),125)
        self.assertEqual(len(calls),125*3+125) #3 training arms, one shared δ0 eval/B.
        self.assertTrue(all(row['assigned'] for row in result['tables']['inner_validation_paths']))
        self.assertTrue(all(row['assigned'] for row in result['tables']['client_paths']))
        self.assertTrue(all(g['eligible'] for g in result['tables']['inner_gate_summaries']))
        for c in result['context_selections'].values():
            self.assertTrue(all(x['selected_candidate_order']==0 for x in c['methods'].values()))
        self.assertEqual(len(result['trace_groups']['actual']),100)
        self.assertEqual(len(result['trace_groups']['validation']),300)
        refs=pd.DataFrame(result['tables']['summary'])
        self.assertTrue(refs.loc[refs.policy.isin(m.REFERENCES),'case_auto_n'].isna().all())
        self.assertTrue(refs.loc[refs.policy.isin(m.LEARNERS),'case_auto_n'].eq(120).all())

    def test_inner_failure_refers_all_outer_paths_without_ranking_final_family(self):
        p,b,r,pi,ids=self.fixture();engine,_=self.engine(True)
        result=m.run_pipeline(p,b,r,pi,ids,simulator=engine)
        self.assert_inventory(result,p)
        self.assertTrue(all(row['assigned'] for row in result['tables']['inner_validation_paths']))
        self.assertTrue(all(not g['eligible'] for g in result['tables']['inner_gate_summaries']))
        frame=pd.DataFrame(result['tables']['client_paths']);learn=frame.loc[frame.policy.isin(m.LEARNERS)]
        self.assertFalse(learn.assigned.any());self.assertTrue(learn[list(m.FINANCIAL)].isna().all().all())
        self.assertTrue(learn.execution_reason.eq('inner_absolute_risk_exceeds_reference').all())
        for c in result['context_selections'].values():
            self.assertTrue(all(not choice['selection_performed'] and choice['selected_candidate_order'] is None for choice in c['methods'].values()))
        # B/EQ are available only as labelled hypothetical references.
        self.assertTrue(frame.loc[frame.policy.isin(m.REFERENCES),'assigned'].all())
        self.assertEqual(len(result['trace_groups']['actual']),50)
        for row in result['tables']['summary']:
            if row['policy'] in m.LEARNERS:self.assertIsNone(row['conditional_breach_probability'])

    def test_q1_review_and_outside_priority_keep_null_and_do_not_invoke_drift_engine(self):
        for label in ('Review','Growth'):
            with self.subTest(label=label):
                p,b,r,pi,ids=self.fixture(label);engine,calls=self.engine()
                result=m.run_pipeline(p,b,r,pi,ids,simulator=engine)
                self.assert_inventory(result,p);self.assertEqual(calls,[])
                frame=pd.DataFrame(result['tables']['client_paths']);learn=frame.loc[frame.policy.isin(m.LEARNERS)]
                self.assertTrue(learn.execution_status.eq(m.joint.case_of(label)).all())
                self.assertTrue(learn[list(m.FINANCIAL)].isna().all().all())
                self.assertFalse(frame.loc[frame.policy.eq(m.REFERENCES[0]),'assigned'].any())
                self.assertTrue(frame.loc[frame.policy.eq(m.REFERENCES[1]),'assigned'].all())
                self.assertEqual(len(result['trace_groups']['validation']),0)
                self.assertTrue(all(g['selected_breach_count'] is None and g['same_U_B_breach_count'] is None for g in result['tables']['inner_gate_summaries']))

    def test_global_growth_band_then_exact_excess_fee_order_without_tolerance_chaining(self):
        rows=[dict(execution_candidate_order=i,tv_threshold=m.THRESHOLDS[i],n_training=72,breach_count=4,
                   mean_log_growth=g,mean_excess=e,mean_nominal_fees=f) for i,(g,e,f) in enumerate(((1.,.02,.3),(1.-.75e-12,.01,.4),(1.-1.5e-12,0.,0.)))]
        result=m.choose_family(rows,4,72,(0,1,2))
        self.assertEqual(result['global_growth_tied_candidate_orders'],[0,1]);self.assertEqual(result['selected_candidate_order'],1)
        rows[0]['mean_excess']=rows[1]['mean_excess'];rows[0]['mean_nominal_fees']=.4+1e-15
        self.assertEqual(m.choose_family(rows,4,72,(0,1,2))['selected_candidate_order'],1)
        rows[0]['mean_nominal_fees']=.4
        self.assertEqual(m.choose_family(rows,4,72,(0,1,2))['selected_candidate_order'],0)
        self.assertEqual(m.choose_family(rows,4,72,(0,))['selected_candidate_order'],0)
        rows[0]['breach_count']=8
        self.assertIsNone(m.choose_family(rows,8,72,(0,))['selected_candidate_order'])

    def test_inner_gate_uses_same_U_reference_and_original_denominator(self):
        rows=[];base=[]
        for sid in range(1,97):
            auto=sid<=48
            rows.append(dict(scenario_id=sid,job_id=f'j{(sid-1)//24}',assigned=auto,
                breach=bool(sid<=4) if auto else None,log_growth=.03 if auto else None,
                case_status='balanced_case',execution_status='automated' if auto else 'execution_review'))
            base.append(dict(scenario_id=sid,job_id=f'j{(sid-1)//24}',assigned=True,breach=bool(5<=sid<=7)))
        gate=m.gate_inner(rows,base,range(1,97))
        self.assertEqual(gate['auto_inner_n'],48);self.assertEqual(gate['absolute_alpha_count_cap'],4)
        self.assertEqual(gate['selected_breach_count'],4);self.assertEqual(gate['same_U_B_breach_count'],3)
        self.assertFalse(gate['eligible']);self.assertEqual(gate['reason'],'inner_risk_exceeds_same_U_B')
        base[7]['breach']=True
        self.assertTrue(m.gate_inner(rows,base,range(1,97))['eligible'])
        for row in rows:row.update(assigned=False,breach=None,log_growth=None,execution_status='execution_review')
        zero=m.gate_inner(rows,base,range(1,97))
        self.assertIsNone(zero['conditional_breach_probability']);self.assertFalse(zero['eligible'])

    def test_frozen_q1_job_identity_mismatch_is_rejected_before_simulation(self):
        p,b,r,pi,ids=self.fixture();b[p['jobs'][0]['job_id']]['refit']['training_ids']=[1]
        with self.assertRaises(ValueError):m.run_pipeline(p,b,r,pi,ids,simulator=lambda *a,**k:None)

    def test_cached_product_target_or_local_global_identity_tampering_is_rejected(self):
        for which,key,value in (('product','weights',[.45,.25,.3]),('product','candidate_index',18),
                                ('product','candidate_original232_index',24),('q1_product','name','Growth')):
            with self.subTest(which=which,key=key):
                p,b,r,pi,ids=self.fixture()
                b[p['jobs'][0]['job_id']]['q2_case'][which][key]=value
                with self.assertRaises(ValueError):m.run_pipeline(p,b,r,pi,ids,simulator=lambda *a,**k:None)

    def test_main_synthetic_guarded_export_no_final_parse_and_no_overwrite(self):
        p,b,r,pi,ids=self.fixture();engine,_=self.engine();digest=hashlib.sha256()
        for a in (ids,r,pi):digest.update(np.ascontiguousarray(a).tobytes())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=root/'docs/simple_execution_comparison_protocol.json';out=root/'results/model_review/simple_execution_comparison_v1'
            protocol.parent.mkdir(parents=True)
            (root/'docs/refit_stability_protocol.json').write_text(json.dumps({k:p[k] for k in ('contexts','jobs','static_contexts')}))
            parent=root/'results/model_review/refit_stability_v1';parent.mkdir(parents=True)
            (parent/'manifest.json').write_text(json.dumps({'status':'complete','development_arrays_sha256':digest.hexdigest()}))
            (parent/'independent_audit.json').write_text(json.dumps({'status':'passed'}))
            (parent/'static_refits.json.gz').write_bytes(gzip.compress(json.dumps(b).encode(),mtime=0))
            p['execution_ready']=True;p['execution_sources_sha256']={'synthetic.py':'a'*64};p['frozen_input_sha256']={}
            protocol.write_text(json.dumps(p));loads=[]
            def load(path,requested,allow_final):
                self.assertEqual(list(requested),list(range(1,121)));self.assertIs(allow_final,False);loads.append(1)
                return r.copy(),pi.copy(),ids.copy()
            with patch.multiple(m,ROOT=root,PROTOCOL=protocol,OUT=out),patch.object(m,'source_manifest',return_value={'synthetic.py':'a'*64}),patch.object(m,'frozen_inputs',return_value={}),patch.object(m,'load_paths',side_effect=load),patch.object(m,'simulate_drift',side_effect=engine):
                with redirect_stdout(io.StringIO()):m.main()
                with self.assertRaises(RuntimeError):m.main()
            self.assertEqual(loads,[1]);manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['training_tasks'],125)
            self.assertFalse(manifest['final_scenarios_loaded']);self.assertEqual(manifest['sources'],manifest['sources_at_end'])
            for name,value in manifest['outputs_sha256'].items():self.assertEqual(m.sha(out/name),value)
            for path in out.glob('*.gz'):self.assertEqual(path.read_bytes()[4:8],bytes(4))
            choice=m.read_json(out/'training_choices.json.gz');self.assertEqual(len(choice),125)
            self.assertEqual(manifest['row_counts']['inner_reference_paths'],2400)
