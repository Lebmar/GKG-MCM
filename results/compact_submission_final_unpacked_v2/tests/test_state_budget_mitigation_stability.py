"""Synthetic full-repeat identities, own-fit lineage and before-parser gates."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("state_budget_mitigation_stability_runner", ROOT / "scripts/run_state_budget_mitigation_stability.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def jobs_and_fits():
    jobs = []; fits = {}
    for repeat in range(1, 6):
        order = np.roll(np.arange(1, 121), (repeat-1)*7).tolist()
        for fold in range(1, 6):
            test = sorted(order[24*(fold-1):24*fold]); train = sorted(set(order)-set(test))
            context = 100*repeat+fold
            for inner in range(1, 6):
                evaluation = train[24*(inner-1):24*inner] if inner < 5 else test
                training = sorted(set(train)-set(evaluation)) if inner < 5 else train
                jid = f"c{context}_inner{inner}" if inner < 5 else f"c{context}_refit"
                job = dict(job_id=jid, parent_context=f"stability_r{repeat}_outer{fold}", parent_context_id=context,
                    repeat_id=repeat, outer_fold=fold, dynamic_inner_fold=inner if inner < 5 else None,
                    stage="inner" if inner < 5 else "refit", training_ids=training, evaluation_ids=evaluation)
                jobs.append(job)
                fits[jid] = dict(refit=dict(training_ids=training.copy(), outer_test_ids=evaluation.copy()),
                    q2_case=dict(status="balanced_case", in_balanced_case=True, product=dict(weights=[.45,.05,.5])))
    return jobs, fits


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes): path.write_bytes(value)
    elif isinstance(value, str): path.write_text(value)
    else: m.writej(path, value)
    return m.sha(path)


def freeze_fixture(root):
    jobs, fits = jobs_and_fits()
    ids=np.arange(1,121); r=np.zeros((120,30,3)); pi=np.zeros((120,30))
    digest=hashlib.sha256()
    for value in (ids,r,pi): digest.update(np.ascontiguousarray(value).tobytes())
    source={name:put(root,name,"# synthetic frozen source\n") for name in m.REQUIRED_SOURCES}
    put(root,"data/SafeInvest_Synthetic_Data.csv","synthetic byte hash only; never parsed\n")
    jobsha=put(root,m.JOBS,dict(jobs=jobs))
    fitsha=put(root,m.FITS,gzip.compress(json.dumps(fits).encode(),mtime=0))
    parent=put(root,m.PARENT/"manifest.json",dict(status="complete",scenario_ids_loaded=ids.tolist(),
        final_scenarios_loaded=False,development_arrays_sha256=digest.hexdigest(),outputs_sha256={m.FITS.name:fitsha}))
    mapping={str(m.JOBS):jobsha,str(m.FITS):fitsha,str(m.PARENT/"manifest.json"):parent}
    put(root,m.PARENT/"independent_audit.json",dict(status="passed",final_scenarios_loaded=False,
        sources_and_artifacts_at_start=mapping,sources_and_artifacts_at_end=mapping))
    primary=put(root,m.PRIMARY_PROTOCOL,dict(candidates=list(m.CANDIDATES),service_alpha=.1))
    put(root,m.PRIMARY_FREEZE,dict(execution_ready=True,protocol_sha256=primary,source_sha256=source,input_sha256={}))
    protocol=dict(version="state-budget-mitigation-five-repeat-stability-v1",execution_ready=True,development_ids=ids.tolist(),allow_final=False,
        repeat_ids=list(range(1,6)),job_protocol=str(m.JOBS),static_refits=str(m.FITS),candidates=list(m.CANDIDATES),service_alpha=.1)
    protocol_sha=put(root,m.PROTOCOL,protocol)
    inputs={name:m.sha(root/name) for name in m.REQUIRED_INPUTS}
    put(root,m.FREEZE,dict(execution_ready=True,protocol_sha256=protocol_sha,source_sha256=source,input_sha256=inputs))
    return jobs,fits,r,pi,ids


def paired_constant(returns,inflation,target,loss,**kwargs):
    n=len(returns); h=kwargs["horizon"]
    arm=dict(real_wealth=np.full((n,1,h+1),100.),fees=np.zeros((n,1,h)),trades=np.zeros((n,1,h),bool),
        ever_breached=np.zeros((n,1,h+1),bool),first_crossing=np.full((n,1),-1),
        pending_at_decision=np.zeros((n,1,h),bool),target_execution=np.ones((n,1,h),bool),
        weights=np.broadcast_to(np.asarray(target),(n,1,h+1,3)).copy(),cpi=np.ones((n,h+1)))
    return dict(annual=arm,pending_hold=arm)


def initial_review(*args,**kwargs):
    return dict(status="preinvestment_review",financial_available=False,arm=None,initial_witness=dict(status="initial_review"))


def funded_policy(actual,inflation,training,training_inflation,target,**kwargs):
    arm=m.squeeze(paired_constant(actual,inflation,target,.4,horizon=kwargs["horizon"])["annual"])
    return dict(status="funded",financial_available=True,arm=arm,initial_witness=dict(status="selected"))


class StateBudgetMitigationStabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_all125_synthetic_jobs_and_own_training_fits_validate(self):
        jobs,fits=jobs_and_fits();m.validate_stability_jobs(jobs);m.validate_fits(jobs,fits)
        self.assertEqual(len(jobs),125);self.assertEqual(len(fits),125)

    def test_explicit_new_engine_canonical_runner_and_isolated_paths(self):
        self.assertEqual(m.simulate_state_budget.__module__,"experimental_execution.state_budget_mitigation")
        self.assertEqual(m.rows_from_arm.__module__,"run_state_budget_mitigation")
        self.assertEqual(m.summarize_records.__module__,"run_state_budget_mitigation")
        self.assertEqual(m.select_inner.__module__,"experimental_execution.state_budget_selection")
        self.assertEqual(m.DEFAULT_OUT.name,"state_budget_mitigation_stability_v1")
        self.assertIn("mitigation",str(m.PROTOCOL));self.assertIn("mitigation",str(m.PRIMARY_FREEZE))

    def test_mitigation_fees_are_actual_nominal_trigger_fees_and_NULL_is_not_zero(self):
        n,h=2,30
        arm=dict(real_wealth=np.full((n,h+1),100.),fees=np.zeros((n,h)),trades=np.zeros((n,h),bool),
            ever_breached=np.zeros((n,h+1),bool),first_crossing=np.full(n,-1),
            pending_at_decision=np.zeros((n,h),bool),guard_triggered=np.zeros((n,h),bool),
            guard_pending=np.zeros((n,h+1),bool),gate_reason=np.full((n,h),"support_safe_hold",dtype="U64"))
        arm["gate_reason"][0,1]="risk_mitigation_then_review";arm["trades"][0,1]=True
        arm["fees"][0,0]=.2;arm["fees"][0,1]=.07;arm["guard_triggered"][0,1]=True
        arm["pending_at_decision"][0,1:]=True;arm["guard_pending"][0,2:]=True
        rows=m.rows_from_arm(arm,[1,2],{})
        self.assertAlmostEqual(rows[0]["mitigation_nominal_fees"],.07)
        self.assertAlmostEqual(rows[0]["mitigation_trade_fraction"],1/30)
        self.assertAlmostEqual(rows[0]["nominal_fees"],.27)
        self.assertAlmostEqual(rows[0]["automatic_fraction"],1/30)
        summary=m.summarize_records(rows)
        self.assertAlmostEqual(summary["mean_mitigation_fees"],.035)
        self.assertAlmostEqual(summary["mean_mitigation_trade_fraction"],1/60)
        null=m.null_rows([3],{},"preinvestment_review")
        self.assertIsNone(null[0]["mitigation_nominal_fees"])
        self.assertIsNone(null[0]["mitigation_trade_fraction"])
        self.assertIsNone(m.summarize_records(null)["mean_mitigation_fees"])

    def test_inner_archive_keeps_selected_mitigation_evidence_omits_alternative_menus(self):
        arm=dict(real_wealth=np.full((2,31),100.),mitigation_selected_index=np.zeros((2,30),int),
            candidate_posttrade=np.zeros((2,30,13,3)),mitigation_candidate_posttrade=np.zeros((2,30,9,3)))
        retained,omitted=m.archive_arrays(arm,"inner_candidate")
        self.assertIn("mitigation_selected_index",retained)
        self.assertEqual(set(omitted),{"candidate_posttrade","mitigation_candidate_posttrade"})
        selected,dropped=m.archive_arrays(arm,"outer_selected")
        self.assertEqual(dropped,[]);self.assertIn("mitigation_candidate_posttrade",selected)

    def test_missing_job_repeat_or_duplicate_inner_cohort_is_rejected(self):
        jobs,_=jobs_and_fits()
        with self.assertRaises(ValueError):m.validate_stability_jobs(jobs[:-1])
        for key,value in (("repeat_id",6),("parent_context_id",999),("evaluation_ids",[121]*24)):
            bad=copy.deepcopy(jobs);bad[0][key]=value
            with self.assertRaises(ValueError):m.validate_stability_jobs(bad)
        bad=copy.deepcopy(jobs);bad[1]["evaluation_ids"]=bad[0]["evaluation_ids"].copy()
        with self.assertRaises(ValueError):m.validate_stability_jobs(bad)

    def test_no_full120_target_or_wrong_static_evaluation_identity_substitution(self):
        jobs,fits=jobs_and_fits()
        for field,value in (("training_ids",list(range(1,121))),("outer_test_ids",[1])):
            bad=copy.deepcopy(fits);bad[jobs[0]["job_id"]]["refit"][field]=value
            with self.assertRaises(ValueError):m.validate_fits(jobs,bad)
        bad=copy.deepcopy(fits);bad["full120"]=copy.deepcopy(next(iter(fits.values())))
        with self.assertRaises(ValueError):m.validate_fits(jobs,bad)

    def test_metadata_only_freeze_gate_passes_and_never_calls_market_loader(self):
        freeze_fixture(self.root)
        with mock.patch.object(m,"load_paths") as loader:
            p,_,jobs,fits,_=m.freeze_gate(self.root)
        loader.assert_not_called();self.assertFalse(p["allow_final"]);self.assertEqual((len(jobs),len(fits)),(125,125))

    def test_freeze_must_include_entire_canonical_transitive_input_map(self):
        freeze_fixture(self.root)
        extra="docs/synthetic_canonical_parent.json";digest=put(self.root,extra,{"frozen":True})
        primary=m.readj(self.root/m.PRIMARY_FREEZE);primary["input_sha256"][extra]=digest
        m.writej(self.root/m.PRIMARY_FREEZE,primary)
        stability=m.readj(self.root/m.FREEZE)
        stability["input_sha256"][str(m.PRIMARY_FREEZE)]=m.sha(self.root/m.PRIMARY_FREEZE)
        m.writej(self.root/m.FREEZE,stability)
        with mock.patch.object(m,"load_paths") as loader,self.assertRaises(RuntimeError):m.freeze_gate(self.root)
        loader.assert_not_called()
        stability["input_sha256"][extra]=digest;m.writej(self.root/m.FREEZE,stability)
        m.freeze_gate(self.root)

    def test_draft_readiness_finalIDs_family_and_changed_hash_block_before_loader(self):
        for mode in ("draft","final","family","source"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);freeze_fixture(root)
                p=m.readj(root/m.PROTOCOL)
                if mode=="draft":p["execution_ready"]=False
                elif mode=="final":p["development_ids"][-1]=121
                elif mode=="family":p["candidates"]=p["candidates"][:-1]
                if mode=="source":(root/"src/safeinvest/core.py").write_text("changed")
                else:m.writej(root/m.PROTOCOL,p)
                with mock.patch.object(m,"load_paths") as loader,self.assertRaises(RuntimeError):m.run(root/"out",root)
                loader.assert_not_called();self.assertFalse((root/"out").exists())

    def test_incomplete_parent_and_unbound_static_cache_are_rejected(self):
        freeze_fixture(self.root)
        parent_path=self.root/m.PARENT/"manifest.json";parent=m.readj(parent_path);parent["status"]="incomplete"
        m.writej(parent_path,parent)
        frozen=m.readj(self.root/m.FREEZE);frozen["input_sha256"][str(m.PARENT/"manifest.json")]=m.sha(parent_path)
        m.writej(self.root/m.FREEZE,frozen)
        with self.assertRaises(RuntimeError):m.freeze_gate(self.root)

    def test_complete_repeat_keeps_allNULL_learners_and_every_reference_originalID(self):
        jobs,fits=jobs_and_fits();market=np.zeros((120,30,3));pi=np.zeros((120,30));state=m.empty_repeat(1)
        with mock.patch.object(m,"simulate_state_budget",side_effect=initial_review) as candidate, \
             mock.patch.object(m,"simulate_paired",side_effect=paired_constant):
            m.evaluate_repeat(state,jobs,fits,market,pi,service_alpha=.1)
        self.assertEqual(candidate.call_count,120);self.assertEqual(len(state["groups"]),200);self.assertEqual(len(state["rows"]),4800)
        self.assertEqual(len(state["completed_jobs"]),25);self.assertEqual(len(state["summaries"]),4)
        learner=state["summaries"][0];self.assertEqual(learner["original_n"],120);self.assertEqual(learner["financial_n"],0)
        self.assertIsNone(learner["median_cagr"]);self.assertIsNone(learner["breach_count"])
        self.assertTrue(all(s["selected_candidate_id"] is None for s in state["selections"].values()))
        self.assertEqual(sum(g["array_names"]==[] for g in state["groups"]),125)
        self.assertEqual(state["summaries"][1]["financial_n"],120)

    def test_funded_inner_selection_refits_only_that_same_tuple_on_own96(self):
        jobs,fits=jobs_and_fits();state=m.empty_repeat(1)
        with mock.patch.object(m,"simulate_state_budget",side_effect=funded_policy) as candidate, \
             mock.patch.object(m,"simulate_paired",side_effect=paired_constant):
            m.evaluate_repeat(state,jobs,fits,np.zeros((120,30,3)),np.zeros((120,30)),service_alpha=.1)
        self.assertEqual(candidate.call_count,125)
        refits=[call for call in candidate.call_args_list if len(call.kwargs["training_ids"])==96]
        self.assertEqual(len(refits),5)
        for call in refits:
            self.assertEqual(call.kwargs["release_fraction"],m.CANDIDATES[0]["release_fraction"])
            self.assertEqual(call.kwargs["forecast_rule"],m.CANDIDATES[0]["forecast_rule"])
            self.assertFalse(set(call.kwargs["training_ids"])&set(call.kwargs["actual_ids"]))
        self.assertEqual(state["summaries"][0]["financial_n"],120)
        self.assertEqual(state["summaries"][0]["mean_mitigation_fees"],0.)

    def test_selected_tuple_refit_Review_never_rescues_by_second_candidate(self):
        jobs,fits=jobs_and_fits();state=m.empty_repeat(1)
        def fitted(*args,**kwargs):
            return initial_review() if len(kwargs["training_ids"])==96 else funded_policy(*args,**kwargs)
        with mock.patch.object(m,"simulate_state_budget",side_effect=fitted) as candidate, \
             mock.patch.object(m,"simulate_paired",side_effect=paired_constant):
            m.evaluate_repeat(state,jobs,fits,np.zeros((120,30,3)),np.zeros((120,30)),service_alpha=.1)
        self.assertEqual(candidate.call_count,125)
        self.assertTrue(all(choice["selected_candidate_id"]==m.CANDIDATES[0]["candidate_id"] for choice in state["selections"].values()))
        self.assertTrue(all(choice["final_status"]=="execution_review" for choice in state["selections"].values()))
        self.assertEqual(state["summaries"][0]["financial_n"],0)
        self.assertIsNone(state["summaries"][0]["mean_mitigation_fees"])

    def test_q1_review_keeps_exact_thirds_reference_without_cash_fallback(self):
        jobs,fits=jobs_and_fits()
        for job in jobs:
            if job["repeat_id"]==1:
                fits[job["job_id"]]["q2_case"].update(status="risk_review",in_balanced_case=False)
        state=m.empty_repeat(1)
        with mock.patch.object(m,"simulate_state_budget") as candidate, mock.patch.object(m,"simulate_paired",side_effect=paired_constant):
            m.evaluate_repeat(state,jobs,fits,np.zeros((120,30,3)),np.zeros((120,30)),service_alpha=.1)
        candidate.assert_not_called()
        self.assertEqual(state["summaries"][1]["financial_n"],0);self.assertEqual(state["summaries"][3]["financial_n"],120)
        self.assertTrue(all(r["mdd"] is None for r in state["rows"] if r["scope"]=="inner_candidate"))

    def test_per_repeat_archive_retains_NULL_metadata_and_no_pooled600_summary(self):
        state=m.empty_repeat(2);state["groups"]=[dict(group_id="null",array_names=[],scenario_ids=[1])]
        state["rows"]=m.null_rows([1],dict(policy="state_budget_learner"),"initial_review")
        folder=self.root/"repeat_02";m.write_repeat(folder,state,"failed_partial")
        saved=m.readj(folder/"manifest.json")
        self.assertEqual(saved["original_market_scenarios"],120);self.assertEqual(saved["independent_additional_scenarios"],0)
        self.assertEqual(saved["status"],"failed_partial")
        self.assertEqual(m.readj(folder/"trace_groups.json")[0]["array_names"],[])

    def test_parser_failure_preserves_attempt_and_refuses_retry(self):
        freeze_fixture(self.root);out=self.root/"out"
        with mock.patch.object(m,"load_paths",side_effect=ValueError("synthetic parser failure")) as loader,self.assertRaises(ValueError):m.run(out,self.root)
        self.assertEqual(loader.call_args.kwargs,{"allow_final":False})
        self.assertEqual(list(loader.call_args.args[1]),list(range(1,121)))
        saved=m.readj(out/"manifest.json");self.assertEqual(saved["status"],"failed")
        self.assertTrue(saved["market_value_access_started"]);self.assertFalse(saved["final_scenarios_loaded"])
        self.assertTrue((out/"failure.txt").is_file())
        with mock.patch.object(m,"freeze_gate") as gate,self.assertRaises(RuntimeError):m.run(out,self.root)
        gate.assert_not_called()

    def test_failed_repeat_saves_its_partial_outputs_and_original_traceback(self):
        _,_,r,pi,ids=freeze_fixture(self.root);out=self.root/"out"
        def failed(state,*args,**kwargs):
            state["rows"]=m.null_rows([1],dict(policy="state_budget_learner"),"initial_review")
            state["arrays"]["synthetic__real_wealth"]=np.full((1,31),100.)
            state["completed_jobs"]=["synthetic_completed_job"]
            raise ArithmeticError("synthetic policy failure")
        with mock.patch.object(m,"load_paths",return_value=(r,pi,ids)),mock.patch.object(m,"evaluate_repeat",side_effect=failed),self.assertRaises(ArithmeticError):m.run(out,self.root)
        partial=out/"repeat_01_partial"
        self.assertEqual(m.readj(partial/"manifest.json")["status"],"failed_partial")
        with np.load(partial/"traces.npz",allow_pickle=False) as archive:
            self.assertIn("synthetic__real_wealth",archive.files)
        self.assertIn("synthetic policy failure",(out/"failure.txt").read_text())


if __name__=="__main__":unittest.main()
