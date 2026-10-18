"""Complete30-task orchestration smoke with temporary synthetic inputs only.

The stub engine uses annual accounting merely to supply coherent array shapes.
This validates orchestration/serialization/NULLs, not execution-gate economics.
No market CSV, market cache or saved market outcome is loaded.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
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

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'research'),str(PROJECT/'scripts')]
import run_nested_dynamic_calibration as runner
from safeinvest_research.static import simplex_grid
from experimental_execution.review_pending import simulate_paired
import experimental_execution.risk_priced_fan_rollout as kernel


class CompletePipelineSmoke(unittest.TestCase):
    def run_case(self, force_all_prices_ineligible):
        # Read identity-only declared protocols before patching ROOT.
        legacy=json.loads((PROJECT/'docs/positive_asset_specification_protocol.json').read_text())
        protocol=json.loads((PROJECT/'docs/nested_dynamic_calibration_protocol.json').read_text())
        returns=np.zeros((120,30,3)); inflation=np.zeros((120,30)); ids=np.arange(1,121)
        weights=simplex_grid(.05)
        allowed=np.flatnonzero(np.all(weights>=.05-1e-12,axis=1))
        global_index=int(np.flatnonzero(np.all(np.isclose(weights,[.4,.3,.3],atol=1e-14,rtol=0),axis=1))[0])
        local_index=int(np.flatnonzero(allowed==global_index)[0])
        product=dict(name='Balanced',candidate_index=local_index,candidate_local_index=local_index,
                     candidate_original232_index=global_index,weights=weights[global_index].tolist())
        def refit(context):
            return dict(context=context['name'],training_ids=context['training_ids'],
                        selection=dict(selected_gamma=0.),raw_fit=dict(menu=[deepcopy(product)]),
                        effective_assignments={p['name']:'Balanced' for p in legacy['clients']})
        static_calls=[]; simulator_prices=[]
        def static_stub(metrics,w,profiles,context,*,candidate_original_indices):
            static_calls.append(context['name'])
            return dict(refit=refit(context),inner_oof_paths=[],q2_case=dict(client='C',status='balanced_case',
                effective_assignment='Balanced',in_balanced_case=True,product=deepcopy(product),q1_product=deepcopy(product)))
        def engine(actual,actual_infl,training,training_infl,target,prices,**kw):
            self.assertEqual(kw,dict(L=.4,cost_rate=.0025,horizon=30))
            self.assertEqual(training.shape[1:],(30,3)); self.assertEqual(actual.shape[1:],(30,3))
            simulator_prices.append(np.asarray(prices).copy())
            annual=runner.reference.squeeze_pending(simulate_paired(actual,actual_infl,target,.4,horizon=30,cost_rate=.0025)['annual'])
            p=len(prices); result={}
            for name,value in annual.items():
                if name=='cpi': result[name]=value.copy()
                else: result[name]=np.repeat(value[:,None,...],p,axis=1)
            result['annual_prices']=np.asarray(prices).copy(); result['target']=np.asarray(target).copy()
            if force_all_prices_ineligible:
                # Deliberately fake bad complete-path outcomes for every price.
                # This is a selection/NULL smoke, not a financial-kernel test.
                result['real_wealth'][:,:,-1]=50.
                result['nominal_wealth'][:,:,-1]=50.
                result['ever_breached'][:,:,-1]=True
                result['first_crossing'][:]=30
            return result
        def guarded_load(path,scenario_ids,allow_final=False):
            self.assertFalse(allow_final); self.assertEqual(list(scenario_ids),list(range(1,121)))
            return returns.copy(),inflation.copy(),ids.copy()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); protocol_path=root/'docs/nested_dynamic_calibration_protocol.json'
            cache=root/'cache/synthetic.npz'; out=root/'results/model_review/nested_dynamic_calibration_v1'
            protocol_path.parent.mkdir(parents=True); cache.parent.mkdir(parents=True)
            (root/'docs/positive_asset_specification_protocol.json').write_text(json.dumps(legacy))
            old=root/'results/model_review/positive_asset_specification_v1/nested_context_refits.json'
            old.parent.mkdir(parents=True)
            old.write_text(json.dumps({c['name']:refit(c) for c in legacy['contexts']}))
            arrays={'weights':weights,'scenario_ids':ids}
            for h in (5,6,15,25,30):
                for field in runner.reference.METRICS:
                    arrays[f'H{h}__{field}']=np.full((120,232),100. if field=='terminal_real' else 0.)
            np.savez_compressed(cache,**arrays)
            digest=hashlib.sha256()
            for a in (ids,returns,inflation): digest.update(np.ascontiguousarray(a).tobytes())
            cache.with_suffix('.json').write_text(json.dumps(dict(development_arrays_sha256=digest.hexdigest(),npz_sha256=runner.sha(cache))))
            sources={'synthetic_source.py':'a'*64,'docs/nested_dynamic_calibration_protocol.json':'b'*64}
            protocol['execution_ready']=True; protocol['execution_sources_sha256']={'synthetic_source.py':'a'*64}
            protocol_path.write_text(json.dumps(protocol))
            with ExitStack() as stack:
                for name,value in dict(ROOT=root,PROTOCOL=protocol_path,CACHE=cache,OUT=out).items():
                    stack.enter_context(patch.object(runner,name,value))
                stack.enter_context(patch.object(runner,'source_manifest',return_value=sources))
                stack.enter_context(patch.object(runner,'frozen_inputs',return_value={}))
                stack.enter_context(patch.object(runner,'load_paths',side_effect=guarded_load))
                stack.enter_context(patch.object(runner,'calibrate_static',side_effect=static_stub))
                stack.enter_context(patch.object(kernel,'simulate_prices',side_effect=engine))
                with redirect_stdout(io.StringIO()): runner.main()
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['training_tasks'],30); self.assertEqual(len(static_calls),30)
            self.assertEqual(manifest['row_counts']['training_candidate_paths.csv'],21600)
            self.assertEqual(manifest['row_counts']['inner_validation_paths.csv'],3000)
            self.assertEqual(manifest['row_counts']['client_paths.csv'],720)
            self.assertEqual(manifest['row_counts']['summary.csv'],6)
            self.assertEqual(manifest['sources'],manifest['sources_at_end'])
            self.assertEqual(manifest['input_sha256'],manifest['input_sha256_at_end'])
            self.assertFalse(manifest['final_scenarios_loaded'])
            self.assertEqual(manifest['scenario_ids_loaded'],list(range(1,121)))
            self.assertTrue(all(x['refit_all_fields_exact'] for x in json.loads((out/'legacy_static_reproduction.json').read_text()).values()))
            for name,expected_sha in manifest['outputs_sha256'].items():
                self.assertEqual(runner.sha(out/name),expected_sha)
            rows=pd.read_csv(out/'client_paths.csv'); learned=rows.loc[rows.policy==runner.LEARNER]
            self.assertEqual(len(learned),240)
            inner=pd.read_csv(out/'inner_validation_paths.csv')
            if force_all_prices_ineligible:
                self.assertFalse(learned.assigned.any()); self.assertFalse(inner.assigned.any())
                self.assertTrue(learned[list(runner.FINANCIAL)].isna().all().all())
                self.assertTrue(inner[list(runner.FINANCIAL)].isna().all().all())
                self.assertTrue(learned.execution_status.eq('execution_review').all())
                self.assertTrue(all(b is None for b in manifest['selected_beta_by_context'].values()))
                self.assertTrue(all(len(x)==9 for x in simulator_prices))
            else:
                self.assertTrue(learned.assigned.all()); self.assertTrue(inner.assigned.all())
                self.assertTrue(all(b==0. for b in manifest['selected_beta_by_context'].values()))
                self.assertEqual(sum(len(x)==9 for x in simulator_prices),30)
                self.assertEqual(sum(len(x)==1 for x in simulator_prices),29)
            for archive_name in ('actual_traces.npz','validation_traces.npz'):
                with np.load(out/archive_name,allow_pickle=False) as archive:
                    for name in archive.files:
                        if name.endswith('__annual_prices'):
                            self.assertEqual(archive[name].shape,(1,))
                    if force_all_prices_ineligible and archive_name=='validation_traces.npz': self.assertEqual(len(archive.files),0)
            return manifest

    def test_complete30_tasks_all_automatic_zero_return_stub_serializes_every_stage(self):
        self.run_case(False)

    def test_every_price_ineligible_keeps_original_rows_and_financial_nulls_without_fallback(self):
        self.run_case(True)


if __name__=='__main__': unittest.main()
