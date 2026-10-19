"""Analytical saved certificates and temp artifacts; no solver/data supplied."""
import copy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_guard_witness_audit',ROOT/'scripts/audit_guard_physical_feasibility.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def case(g=.5,floor=60.,currency=1.):
    x=[45.*currency,5.*currency,50.*currency];V=math.fsum(x);cp=currency;R=V/cp;p=[v/V for v in x]
    state=dict(version='V2b',scope='outer_selected',job_id='job1',group_id='c1__selected',scenario_id=1,
        account_row=0,period=2,holdings=x,cpi=cp,real_peak=100.,running_mdd=0.,current_real_wealth=100.,
        budget_floor=floor,release_fraction=1.,cushion_scale=None,training_ids=list(range(25,121)),
        support_real_gross=np.full((96,3),g).tolist(),primary_feasible_point_count=0,
        primary_feasible_interval_count=0,actual_future_used_in_LP=False)
    patterns=[]
    for k in range(8):
        mask=[bool(k&b) for b in (4,2,1)]
        residuals=dict.fromkeys(('primal_max_violation','dual_stationarity_max_abs','dual_sign_max_violation',
            'complementary_max_abs','primal_dual_objective_gap_abs','solver_objective_gap_abs',
            'true_self_finance_residual_abs','true_vs_pattern_fee_gap_abs','worst_support_vs_z_gap_abs'),0.)
        # For equal support g, hold is optimal. These duals certify it in
        # every region: mi0=-1, me=-g, lower-buy=g*c, all other bound duals0.
        patterns.append(dict(pattern_index=k,buy_mask=mask,normalized_equality_rhs=1+.0025*math.fsum(p[j] for j in range(3) if mask[j]),
            certificate_passed=True,solver_status=0,solver_success=True,solver_message='synthetic analytical certificate',iterations=0,crossover_iterations=0,
            residuals=residuals,normalized_solution=p+[g],normalized_dual_objective=-g,normalized_primal_objective=-g,
            normalized_worst_support=g,support_slacks=[0.]*96,equality_residual=[0.],dual_stationarity=[0.]*4,
            inequality_marginals=[-1.]+[0.]*95,equality_marginals=[-g],lower_marginals=[g*.0025 if b else 0. for b in mask]+[0.],
            upper_marginals=[0.]*4,posttrade_holdings=x,nominal_fee=0.,worst_next_real=g*R,
            dual_objective_maxmin_real=g*R,true_self_finance_residual=0.))
    classification=m.LABELS[0] if g*R>floor+1e-8*R else m.LABELS[1] if g*R<floor-1e-8*R else m.LABELS[2]
    result=dict(classification=classification,all_eight_numerically_resolved=True,pattern_count=8,passed_certificate_count=8,
        budget_floor=floor,current_real_wealth=R,comparison_band_real=1e-8*R,hold_worst_next_real=g*R,
        best_pattern_index=0,best_worst_next_real=g*R,best_posttrade_holdings=x,best_nominal_fee=0.,
        best_budget_margin_real=g*R-floor,maximum_dual_objective_real=g*R,patterns=patterns,
        exact_arithmetic_impossibility_proof=False,account_action_changed=False,future_or_recursive_feasibility_guarantee=False)
    return dict(**state,result=result)


def put(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,bytes):path.write_bytes(value)
    else:m.writej(path,value)
    return m.sha(path)


def fixture(root,one_guard=True):
    snapshot={};inventory=[];cases=[case()] if one_guard else []
    for version,folder in m.PARENTS:
        groups=[];arrays={}
        for context in range(6):
            ids=list(range(1,121)) if context==0 else list(range(24*(context-1)+1,24*context+1))
            training=list(range(1,121)) if context==0 else sorted(set(range(1,121))-set(ids))
            funded=one_guard and version=='V2b' and context==1;gid=f'c{context}__selected';n=len(ids);a={}
            if funded:
                a=dict(first_guard_trigger=np.full(n,-1,dtype=np.int64),guard_triggered=np.zeros((n,30),bool),
                    holdings=np.broadcast_to(np.array([45.,5.,50.]),(n,31,3)).copy(),cpi=np.ones((n,31)),
                    real_peak=np.full((n,31),100.),running_mdd=np.zeros((n,31)),budget_floor=np.full((n,30),60.),
                    was_pending_input=np.zeros((n,30),bool),selected_index=np.full((n,30),-1,dtype=np.int64),
                    support_feasible=np.zeros((n,30),bool),candidate_feasible=np.zeros((n,30,13),bool),
                    interval_feasible=np.zeros((n,30,4),bool),training_real_gross=np.ones((30,96,3)))
                a['first_guard_trigger'][0]=2;a['guard_triggered'][0,1]=True;a['training_real_gross'][1]=.5
                arrays.update({gid+'__'+key:value for key,value in a.items()})
            status='funded' if funded else 'no_inner_feasible_candidate';scope='full_selected' if context==0 else 'outer_selected'
            groups.append(dict(group_id=gid,kind=scope,job_id=f'job{context}',scenario_ids=ids,training_ids=training,
                array_names=list(a),status=status,candidate=dict(release_fraction=1.,forecast_rule='uniform')))
            inventory.append(dict(version=version,scope=scope,group_id=gid,job_id=f'job{context}',scenario_ids=ids,
                status=status,physical_arm_present=funded,first_guard_count=int(funded)))
        snapshot[str(folder/'trace_groups.json')]=put(root,folder/'trace_groups.json',groups)
        np.savez_compressed(root/folder/'traces.npz',**arrays);snapshot[str(folder/'traces.npz')]=m.sha(root/folder/'traces.npz')
    lpplan_sha=put(root,m.LP_PLAN,{'execution_ready':True,'synthetic':True})
    source={'synthetic/lp_source.py':put(root,Path('synthetic/lp_source.py'),b'# frozen synthetic dependency\n')}
    lpf=dict(execution_ready=True,protocol_sha256=lpplan_sha,source_sha256=source,input_sha256=snapshot)
    put(root,m.LP_FREEZE,lpf);allbefore={**source,**snapshot}
    summaries=[]
    for version,_ in m.PARENTS:
        for scope in ('outer_selected','full_selected'):
            active=bool(one_guard and version=='V2b' and scope=='outer_selected')
            summaries.append(dict(version=version,scope=scope,original_n=120,group_count=5 if scope=='outer_selected' else 1,
                physical_path_n=24 if active else 0,first_guard_n=int(active),LP_count=8 if active else 0,
                classification_counts={k:int(active and k==m.LABELS[1]) for k in m.LABELS},all_eight_resolved_n=int(active)))
    put(root,m.OUT/'selected_group_inventory.json',inventory);put(root,m.OUT/'summary.json',summaries)
    put(root,m.OUT/'case_inputs_and_lp_witnesses.json.gz',gzip.compress((json.dumps(cases)+'\n').encode(),mtime=0))
    flat=[{k:v.get(k,v['result'].get(k)) for k in m.CSV_FIELDS} for v in cases]
    pd.DataFrame(flat,columns=m.CSV_FIELDS).to_csv(root/m.OUT/'guard_cases.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    outputs={name:m.sha(root/m.OUT/name) for name in m.OUTPUTS}
    parent_sha=put(root,m.OUT/'manifest.json',dict(status='complete',sources_and_inputs_at_start=allbefore,sources_and_inputs_at_end=allbefore,
        protocol_sha256=lpplan_sha,source_freeze_sha256=m.sha(root/m.LP_FREEZE),final_scenarios_loaded=False,
        raw_market_values_parsed=False,account_action_changed=False,new_final_reads=0,outputs_sha256=outputs,
        selected_group_count=12,first_guard_count=len(cases),LP_count=8*len(cases),exact_arithmetic_impossibility_proof=False,
        all_first_guards_preserved=True,new_policy_runs=0))
    audit_sources={name:put(root,Path(name),b'# synthetic independent audit source\n') for name in m.SOURCES}
    p=dict(version='guard-physical-feasibility-independent-witness-audit-v1',execution_ready=True,parent_manifest_sha256=parent_sha,
        certificate_tolerance=m.CERT_TOL,comparison_tolerance=m.COMPARE_TOL,audit_atol=m.ATOL,audit_rtol=m.RTOL,allow_final=False)
    plan_sha=put(root,m.PLAN,p)
    put(root,m.FREEZE,dict(execution_ready=True,plan_sha256=plan_sha,source_sha256=audit_sources,input_sha256=m.evidence_closure(root)))


class IndependentGuardWitnessTests(unittest.TestCase):
    def test_analytical_eight_region_certificate_recomputed_without_solver(self):
        c=m.Checks();r=m.verify_case(case(),c)
        self.assertEqual(r['classification'],m.LABELS[1]);self.assertEqual(c.reported_rawfun_checks,8)
        self.assertGreater(c.n,150);self.assertFalse(r['exact_arithmetic_impossibility_proof'])

    def test_feasible_and_boundary_classifications_use_original_floor(self):
        self.assertEqual(m.verify_case(case(g=1,floor=90),m.Checks())['classification'],m.LABELS[0])
        self.assertEqual(m.verify_case(case(g=1,floor=100),m.Checks())['classification'],m.LABELS[2])

    def test_currency_cpi_conversion_and_exact_fee_are_independent_checks(self):
        m.verify_case(case(currency=17),m.Checks())
        bad=case();bad['result']['patterns'][0]['nominal_fee']=.01
        with self.assertRaises(AssertionError):m.verify_case(bad,m.Checks())

    def test_wrong_dual_or_primal_is_not_accepted_from_success_flag(self):
        for field in ('inequality_marginals','normalized_solution'):
            bad=case();bad['result']['patterns'][0][field][0]+=.01
            with self.assertRaises(AssertionError):m.verify_case(bad,m.Checks())

    def test_mask_missing_region_and_overclaimed_proof_rejected(self):
        for change in ('mask','missing','proof'):
            bad=case()
            if change=='mask':bad['result']['patterns'][0]['buy_mask'][0]=True
            elif change=='missing':bad['result']['patterns'].pop()
            else:bad['result']['exact_arithmetic_impossibility_proof']=True
            with self.assertRaises(AssertionError):m.verify_case(bad,m.Checks())

    def test_rawfun_gap_is_only_reported_range_checked_and_explicitly_counted(self):
        value=case();value['result']['patterns'][0]['residuals']['solver_objective_gap_abs']=1e-10
        m.verify_case(value,m.Checks())  # Not fabricated as an independent res.fun reconstruction.
        value['result']['patterns'][0]['residuals']['solver_objective_gap_abs']=-1
        with self.assertRaises(AssertionError):m.verify_case(value,m.Checks())

    def test_source_case_and_inventory_and_summary_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root);m.run(root)
            receipt=m.readj(root/m.RECEIPT)
            self.assertEqual(receipt['status'],'passed');self.assertEqual(receipt['case_count'],1)
            self.assertEqual(receipt['pattern_count'],8);self.assertEqual(receipt['selected_group_count'],12)
            self.assertEqual(receipt['summary_rows'],4);self.assertEqual(receipt['new_LP_solves'],0)
            self.assertFalse(receipt['raw_solver_objective_reconstructed'])

    def test_allNULL_pipeline_retains_twelve_inventory_and_four_summaries(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root,one_guard=False);m.run(root)
            receipt=m.readj(root/m.RECEIPT);self.assertEqual(receipt['case_count'],0);self.assertEqual(receipt['pattern_count'],0)
            self.assertEqual(receipt['selected_group_count'],12);self.assertEqual(receipt['summary_rows'],4)

    def test_frozen_parent_tamper_or_missing_freeze_blocks_numeric_parsers(self):
        for mode in ('tamper','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);fixture(root)
                if mode=='tamper':(root/m.OUT/'case_inputs_and_lp_witnesses.json.gz').write_bytes(b'changed')
                else:(root/m.FREEZE).unlink()
                with mock.patch.object(m.gzip,'open') as gz,mock.patch.object(m.np,'load') as npz:
                    with self.assertRaises((RuntimeError,FileNotFoundError)):m.run(root)
                gz.assert_not_called();npz.assert_not_called();self.assertFalse((root/m.ATTEMPT).exists())

    def test_failure_preserved_and_retry_refused_without_touching_original(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root);manifest_sha=m.sha(root/m.OUT/'manifest.json')
            with mock.patch.object(m,'audit_values',side_effect=AssertionError('synthetic witness mismatch')):
                with self.assertRaises(AssertionError):m.run(root)
            self.assertEqual(m.readj(root/m.ATTEMPT/'attempt.json')['status'],'failed')
            self.assertTrue((root/m.ATTEMPT/'failure.txt').exists());self.assertFalse((root/m.RECEIPT).exists())
            with mock.patch.object(m.gzip,'open') as gz:
                with self.assertRaises(RuntimeError):m.run(root)
            gz.assert_not_called();self.assertEqual(m.sha(root/m.OUT/'manifest.json'),manifest_sha)

    def test_wrong_classification_and_csv_state_binding_are_detected(self):
        wrong=case();wrong['result']['classification']=m.LABELS[0]
        with self.assertRaises(AssertionError):m.verify_case(wrong,m.Checks())
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root)
            with gzip.open(root/m.OUT/'case_inputs_and_lp_witnesses.json.gz','rt') as stream:values=json.load(stream)
            values[0]['holdings'][0]+=1
            (root/m.OUT/'case_inputs_and_lp_witnesses.json.gz').write_bytes(gzip.compress(json.dumps(values).encode(),mtime=0))
            with self.assertRaises(AssertionError):m.audit_values(root,m.Checks())


if __name__=='__main__':unittest.main()
