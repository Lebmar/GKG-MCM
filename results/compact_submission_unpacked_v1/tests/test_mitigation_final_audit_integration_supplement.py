"""Supplemental adapter integration, outside the already frozen audit sources.

The fixture uses production kernels on constructed zero markets. The frozen
auditor recomputes it independently through its own all-object audit_values.
No real raw/final file or final-output directory is read.
"""
import importlib.util
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'research')]
import audit_mitigation_locked_final_evaluation as audit
from experimental_execution.state_budget_mitigation import simulate_state_budget
from experimental_execution.review_pending import simulate_paired
from safeinvest_research.metrics import path_metrics


class SupplementalIntegration(unittest.TestCase):
    def test_complete_zero_market_layout_with_real_frozen_kernels(self):
        location=ROOT/'scripts/run_mitigation_locked_final_evaluation.py'
        spec_loader=importlib.util.spec_from_file_location('synthetic_final_fixture_producer',location)
        producer=importlib.util.module_from_spec(spec_loader);spec_loader.loader.exec_module(producer)
        registry=dict(development_ids=audit.TRAIN,
            Q1=dict(profile_effective_assignments=dict(A='Review',B='Conservative',C='Balanced',D='Review',E='Growth'),
                portfolios=dict(Conservative=[.25,.3,.45],Balanced=[.45,.05,.5],Growth=[.65,.05,.3]),
                horizons={p[0]:p[1] for p in audit.PROFILES},loss_tolerances={p[0]:p[2] for p in audit.PROFILES}),
            Q2=dict(profile='C',strategic_product='Balanced',target=[.45,.05,.5],horizon=30,L=.4,cost_rate=.0025,selected_candidate_id='release_0.5_uniform',release_fraction=.5,forecast_rule='uniform',support_mode='calendar',service_alpha=.1,persistent_target_relearning=False),
            final_evaluation_contract=dict(training_ids=audit.TRAIN,actual_ids=audit.ACTUAL,adjust_after_final=False,old_final_overwrite=False,new_final_label='previously_exposed_nonblind'))
        spec=producer.object_specification(registry);self.assertEqual(spec['objects'],audit.expected_objects())
        training=np.zeros((120,30,3));training_pi=np.zeros((120,30));actual=np.zeros((60,30,3));actual_pi=np.zeros((60,30))
        results,witness=producer.evaluate_objects(spec,actual,actual_pi,training,training_pi,engine=simulate_state_budget,paired=simulate_paired)
        with tempfile.TemporaryDirectory(prefix='gkg-synthetic-final-audit-') as folder:
            root=Path(folder);out=root/audit.OUT;out.mkdir(parents=True);rows=[];groups=[];arrays={}
            for obj in spec['objects']:
                result=results[obj['object_id']];rows+=producer.path_rows(obj,result,audit.ACTUAL,metrics=path_metrics)
                arm=result['arm'];names=[]
                if arm is not None:
                    for key,value in arm.items():
                        if isinstance(value,np.ndarray):arrays[obj['object_id']+'__'+key]=value;names.append(key)
                groups.append(dict(**obj,status=result['status'],financial_available=result['financial_available'],scenario_ids=audit.ACTUAL,
                    training_ids=audit.TRAIN if obj['rule']=='mitigation' else None,array_names=names,omitted_diagnostic_arrays=[]))
            tables={'Q2_paths.csv.gz':[r for r in rows if r['question']=='Q2'],'Q1_paths.csv.gz':[r for r in rows if r['question']=='Q1'],
                'Q1_thirds_references.csv.gz':[r for r in rows if r['question']=='Q1_reference'],
                'summary.csv.gz':[producer.summary([r for r in rows if r['object_id']==obj['object_id']]) for obj in spec['objects']]}
            comparisons=[];pairs=[]
            compare_objects=[('mitigation_main',name) for name in ('same_target_B','same_target_annual','exact_thirds_annual')]+[('Q1_'+name,'Q1_'+name+'_thirds') for name,_,_,_ in audit.PROFILES]
            for model,reference in compare_objects:
                value,paired=producer.compare([r for r in rows if r['object_id']==model],[r for r in rows if r['object_id']==reference])
                comparisons.append(dict(model_object=model,reference_object=reference,**value));pairs.extend(dict(model_object=model,reference_object=reference,**r) for r in paired)
            tables['paired_paths.csv.gz']=pairs
            for name,data in tables.items():producer.write_csv(out/name,data)
            for name,data in [('object_specification.json',spec),('trace_groups.json',groups),('initial_witness.json',witness),('comparisons.json',comparisons),
                ('severe_example.json',producer.severe_example([r for r in rows if r['object_id']=='mitigation_main'],results))]:producer.writej(out/name,data)
            np.savez_compressed(out/'traces.npz',**arrays)
            manifest=dict(status='synthetic_fixture_only',outputs_sha256={p.name:producer.sha(p) for p in out.iterdir()})
            producer.writej(out/'manifest.json',manifest);self.assertEqual(len(manifest['outputs_sha256']),11)
            calls=[]
            def fake_loader(path,ids,*,allow_final):
                requested=list(ids);calls.append((requested,allow_final))
                if requested==audit.TRAIN and allow_final is False:return training.copy(),training_pi.copy(),np.array(audit.TRAIN)
                if requested==audit.ACTUAL and allow_final is True:return actual.copy(),actual_pi.copy(),np.array(audit.ACTUAL)
                raise AssertionError('Unexpected synthetic parser scope')
            c=audit.Checks()
            with patch('safeinvest.data.load_paths',side_effect=fake_loader):
                result=audit.audit_values(root,dict(object_specification=spec),manifest,c)
            self.assertEqual(calls,[(audit.TRAIN,False),(audit.ACTUAL,True)])
            self.assertEqual(result['counts']['objects'],14);self.assertEqual(result['counts']['path_rows'],840)
            self.assertEqual(result['counts']['physical_accounts'],720);self.assertEqual(result['counts']['periods'],15120)
            self.assertEqual(result['counts']['Review_rows'],120);self.assertEqual(result['counts']['all_main_active_decisions'],1800)
            self.assertEqual(result['counts']['comparison_groups'],8);self.assertEqual(result['counts']['paired_rows'],360)
            self.assertEqual(result['severe_original_id'],121)


if __name__=='__main__':unittest.main()
