"""Complete125-job shape/serialization smoke with synthetic-only inputs.

The fake engine supplies coherent annual accounting arrays, not economic gate
evidence. No market data, market cache or saved financial outcomes are read.
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
import run_refit_stability as runner
from safeinvest_research.static import simplex_grid
from experimental_execution.review_pending import simulate_paired
import experimental_execution.risk_priced_fan_rollout as kernel


class CompleteStabilityPipelineSmoke(unittest.TestCase):
    def run_case(self, force_all_prices_ineligible, q1_c_assignment='Balanced'):
        legacy=json.loads((PROJECT/'docs/positive_asset_specification_protocol.json').read_text())
        protocol=json.loads((PROJECT/'docs/refit_stability_protocol.json').read_text())
        returns=np.zeros((120,30,3));inflation=np.zeros((120,30));ids=np.arange(1,121)
        weights=simplex_grid(.05)
        allowed=np.flatnonzero(np.all(weights>=.05-1e-12,axis=1))
        def product(name,target):
            g=int(np.flatnonzero(np.all(np.isclose(weights,target,atol=1e-14,rtol=0),axis=1)).item())
            i=int(np.flatnonzero(allowed==g).item())
            return dict(name=name,candidate_index=i,candidate_local_index=i,
                        candidate_original232_index=g,weights=weights[g].tolist())
        catalogue=[product('Conservative',[.1,.2,.7]),product('Balanced',[.4,.3,.3]),
                   product('Growth',[.8,.15,.05])]
        assignment={p['name']:'Conservative' if p['name']=='A' else 'Balanced' for p in legacy['clients']}
        if q1_c_assignment=='Review':assignment={name:'Review' for name in assignment}
        else:assignment['C']=q1_c_assignment
        def refit(context):
            n=len(context['training_ids']);details=[]
            for p in legacy['clients']:
                for item in catalogue:
                    liquid=item['weights'][2]>=runner.LIQUIDITY_FLOORS[p['liquidity']]-1e-12
                    details.append(dict(client=p['name'],portfolio=item['name'],candidate_index=item['candidate_index'],
                        horizon=p['horizon'],loss_tolerance=p['loss_tolerance'],n=n,breach_count=0,
                        breach_probability=0.,expected_excess=0.,cvar90_mdd=0.,mean_log_growth=0.,
                        liquidity_pass=bool(liquid),training_screen_feasible=bool(liquid),
                        chosen=assignment[p['name']]==item['name']))
            reviews={p['name']:dict(q1_review_kind='calibration_unsupported' if assignment[p['name']]=='Review' else None,
                q1_review_reasons=['calibration_unsupported'] if assignment[p['name']]=='Review' else [],
                effective_assignment=assignment[p['name']],
                feasibility_diagnostics=None if q1_c_assignment=='Review' else dict(
                    risk_feasible_n=172,
                    liquidity_feasible_n=int(np.count_nonzero(weights[allowed,2]>=runner.LIQUIDITY_FLOORS[p['liquidity']]-1e-12)),
                    joint_feasible_n=int(np.count_nonzero(weights[allowed,2]>=runner.LIQUIDITY_FLOORS[p['liquidity']]-1e-12)),
                    menu_joint_feasible_n=sum(item['weights'][2]>=runner.LIQUIDITY_FLOORS[p['liquidity']]-1e-12 for item in catalogue),
                    floor=runner.LIQUIDITY_FLOORS[p['liquidity']],training_n=n,
                    training_screen_gamma=0.)) for p in legacy['clients']}
            return dict(context=context['name'],context_id=context['context_id'],training_ids=context['training_ids'],
                outer_test_ids=context['outer_test_ids'],selection=dict(selected_gamma=0.),
                raw_fit=None if q1_c_assignment=='Review' else dict(menu=deepcopy(catalogue),assignments=deepcopy(assignment),
                    client_product_details=details,training_screen_gamma=0.,liquidity_floors=runner.LIQUIDITY_FLOORS),
                raw_assignments=deepcopy(assignment),effective_assignments=deepcopy(assignment),
                admission_details=[dict(client=p['name'],raw_assignment=assignment[p['name']],
                    effective_assignment=assignment[p['name']],reason='inner_auto_evidence_present',
                    inner_auto_n=n,inner_auto_folds=4,inner_breach_count=0,
                    inner_conditional_breach_probability=0.) for p in legacy['clients']],q1_review_details=reviews)
        static_calls=[];prices_seen=[]
        def static_stub(metrics,w,profiles,context,*,candidate_original_indices):
            self.assertEqual(profiles,legacy['clients'])
            np.testing.assert_array_equal(candidate_original_indices,allowed)
            self.assertEqual(w.shape,(172,3));static_calls.append(deepcopy(context))
            b=deepcopy(catalogue[1]) if q1_c_assignment=='Balanced' else None
            q1_product=deepcopy(catalogue[2]) if q1_c_assignment=='Growth' else deepcopy(b)
            status=runner.case_of(q1_c_assignment)
            return dict(refit=refit(context),inner_oof_paths=[dict(context=context['name'],outer_fold=None,scenario_id=context['training_ids'][0])],inner_summaries=[],inner_fold_fits={},
                q2_case=dict(client='C',status='risk_review' if status=='q1_review' else status,public_status=status,
                    q1_review_kind='calibration_unsupported' if status=='q1_review' else None,
                    q1_review_reasons=['calibration_unsupported'] if status=='q1_review' else [],
                    effective_assignment=q1_c_assignment,
                    in_balanced_case=status=='balanced_case',product=b,q1_product=q1_product),context=deepcopy(context),
                metadata=dict(liquidity_floors=runner.LIQUIDITY_FLOORS,training_only=True))
        def engine(actual,ai,training,ti,target,prices,**kwargs):
            self.assertEqual(kwargs,dict(L=.4,cost_rate=.0025,horizon=30))
            self.assertEqual(training.shape[1:],(30,3));self.assertEqual(actual.shape[1:],(30,3))
            prices_seen.append(np.asarray(prices).copy())
            annual=runner.reference.squeeze_pending(simulate_paired(actual,ai,target,.4,horizon=30,cost_rate=.0025)['annual'])
            result={}
            for name,value in annual.items():
                result[name]=value.copy() if name=='cpi' else np.repeat(value[:,None,...],len(prices),axis=1)
            result['annual_prices']=np.asarray(prices).copy();result['target']=np.asarray(target).copy()
            if force_all_prices_ineligible:
                # Selection/NULL stress only, not financial-kernel validation.
                result['real_wealth'][:,:,-1]=50.;result['nominal_wealth'][:,:,-1]=50.
                result['ever_breached'][:,:,-1]=True;result['first_crossing'][:]=30
            return result
        def guarded_load(path,scenario_ids,allow_final=False):
            self.assertFalse(allow_final);self.assertEqual(list(scenario_ids),list(range(1,121)))
            return returns.copy(),inflation.copy(),ids.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pp=root/'docs/refit_stability_protocol.json'
            cache=root/'cache/synthetic.npz';out=root/'results/model_review/refit_stability_v1'
            pp.parent.mkdir(parents=True);cache.parent.mkdir(parents=True)
            (root/'docs/positive_asset_specification_protocol.json').write_text(json.dumps(legacy))
            old=root/'results/model_review/positive_asset_specification_v1/nested_context_refits.json'
            old.parent.mkdir(parents=True);old.write_text(json.dumps({c['name']:refit(c) for c in legacy['contexts']}))
            arrays={'weights':weights,'scenario_ids':ids}
            for h in (5,6,15,25,30):
                for field in runner.reference.METRICS:
                    arrays[f'H{h}__{field}']=np.full((120,232),100. if field=='terminal_real' else 0.)
            np.savez_compressed(cache,**arrays)
            digest=hashlib.sha256()
            for a in (ids,returns,inflation):digest.update(np.ascontiguousarray(a).tobytes())
            cache.with_suffix('.json').write_text(json.dumps(dict(development_arrays_sha256=digest.hexdigest(),npz_sha256=runner.sha(cache))))
            sources={'synthetic_source.py':'a'*64,'docs/refit_stability_protocol.json':'b'*64}
            protocol['execution_ready']=True;protocol['execution_sources_sha256']={'synthetic_source.py':'a'*64}
            pp.write_text(json.dumps(protocol))
            with ExitStack() as stack:
                for key,value in dict(ROOT=root,PROTOCOL=pp,CACHE=cache,OUT=out).items():
                    stack.enter_context(patch.object(runner,key,value))
                stack.enter_context(patch.object(runner,'source_manifest',return_value=sources))
                stack.enter_context(patch.object(runner,'frozen_inputs',return_value={}))
                stack.enter_context(patch.object(runner,'load_paths',side_effect=guarded_load))
                stack.enter_context(patch.object(runner,'calibrate_liquidity_static',side_effect=static_stub))
                stack.enter_context(patch.object(kernel,'simulate_prices',side_effect=engine))
                with redirect_stdout(io.StringIO()):runner.main()
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['training_tasks'],125);self.assertEqual(len(static_calls),125)
            self.assertEqual(manifest['row_counts']['training_candidate_paths'],86400)
            self.assertEqual(manifest['row_counts']['inner_validation_paths'],12000)
            self.assertEqual(manifest['row_counts']['client_paths'],2400)
            self.assertEqual(manifest['row_counts']['summary'],20)
            self.assertEqual(manifest['row_counts']['q1_context_assignments'],125)
            self.assertEqual(manifest['row_counts']['q1_effective_paths'],3000)
            self.assertEqual(manifest['row_counts']['q1_summary'],25)
            self.assertEqual(manifest['row_counts']['fit_admission'],50)
            self.assertEqual(manifest['sources'],manifest['sources_at_end'])
            self.assertEqual(manifest['input_sha256'],manifest['input_sha256_at_end'])
            self.assertFalse(manifest['final_scenarios_loaded'])
            self.assertEqual(manifest['scenario_ids_loaded'],list(range(1,121)))
            for name,value in manifest['outputs_sha256'].items():self.assertEqual(runner.sha(out/name),value)
            rows=pd.read_csv(out/'client_paths.csv.gz');learned=rows.loc[rows.policy.isin(runner.LEARNERS)]
            self.assertEqual(len(learned),1200)
            inner=pd.read_csv(out/'inner_validation_paths.csv.gz')
            if q1_c_assignment!='Balanced':
                self.assertFalse(learned.assigned.any());self.assertFalse(inner.assigned.any())
                self.assertTrue(learned[list(runner.FINANCIAL)].isna().all().all())
                self.assertTrue(inner[list(runner.FINANCIAL)].isna().all().all())
                self.assertTrue(learned.execution_status.eq(runner.case_of(q1_c_assignment)).all())
                self.assertEqual(prices_seen,[])
                refs=rows.loc[rows.policy=='annual_then_pending']
                self.assertTrue(refs.execution_status.eq('reference_unavailable').all())
                self.assertFalse(refs.assigned.any())
                self.assertTrue(rows.loc[rows.policy=='equal_thirds_annual'].assigned.all())
            elif force_all_prices_ineligible:
                self.assertFalse(learned.assigned.any());self.assertFalse(inner.assigned.any())
                self.assertTrue(learned[list(runner.FINANCIAL)].isna().all().all())
                self.assertTrue(inner[list(runner.FINANCIAL)].isna().all().all())
                self.assertTrue(learned.execution_status.eq('execution_review').all())
                self.assertTrue(all(len(x)==9 for x in prices_seen))
                self.assertTrue(learned.training_beta.isna().all())
            else:
                self.assertTrue(learned.assigned.all());self.assertTrue(inner.assigned.all())
                self.assertEqual(sum(len(x)==9 for x in prices_seen),125)
                # Both final learners choose the same price; deduplicate replay.
                self.assertEqual(sum(len(x)==1 for x in prices_seen),125)
                self.assertTrue(learned.training_beta.eq(0.).all())
            for archive_name in ('actual_traces.npz','validation_traces.npz'):
                with np.load(out/archive_name,allow_pickle=False) as archive:
                    for key in archive.files:
                        if key.endswith('__annual_prices'):self.assertEqual(archive[key].shape,(1,))
                    if (force_all_prices_ineligible or q1_c_assignment!='Balanced') and archive_name=='validation_traces.npz':
                        self.assertEqual(len(archive.files),0)
            return manifest

    def test_complete125_jobs_share_evidence_and_deduplicate_identical_final_orders(self):
        self.run_case(False)

    def test_every_price_ineligible_preserves_original_rows_and_financial_nulls(self):
        self.run_case(True)



if __name__=='__main__':unittest.main()
