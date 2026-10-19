"""Pure control-flow, identity, NULL and synthetic stress-boundary checks."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_state_budget_mitigation_stress as stress


def metadata_fixture():
    jobs=[];fits={};selections={};groups=[];w=[.4,.3,.3]
    for c in range(6):
        test=list(range(1,121)) if c==0 else list(range(1+24*(c-1),1+24*c))
        train=list(range(1,121)) if c==0 else sorted(set(range(1,121))-set(test))
        for fold in range(5):
            refit=fold==4;actual=test if refit else [int(i) for i in np.array_split(train,4)[fold]]
            fit=train if refit else sorted(set(train)-set(actual));job_id=f"c{c}_refit" if refit else f"c{c}_inner{fold+1}"
            jobs.append(dict(job_id=job_id,parent_context_id=c,stage="refit" if refit else "inner",training_ids=fit,evaluation_ids=actual))
            fits[job_id]=dict(refit=dict(training_ids=fit),q2_case=dict(in_balanced_case=True,product=dict(weights=w),status="balanced_case"))
            if refit:
                spec=dict(stress.canonical.CANDIDATES[0]);selections[str(c)]=dict(training_ids=train,evaluation_ids=actual,selected_candidate_id=spec["candidate_id"])
                common=dict(job_id=job_id,training_ids=train,scenario_ids=actual,array_names=[])
                groups.append(dict(common,group_id=job_id+"_model",kind="full_selected" if c==0 else "outer_selected",policy=spec["candidate_id"],target=w,candidate=spec))
                for name in stress.canonical.REFS:
                    groups.append(dict(common,group_id=job_id+"_"+name,kind="reference",policy=name,target=[1/3]*3 if name=="exact_thirds_annual" else w))
    return jobs,fits,selections,groups


class FixedSelectedMitigationStressTests(unittest.TestCase):
    def test_seven_fixed_settings_no_parameter_search(self):
        self.assertEqual(len(stress.SETTINGS),7)
        self.assertEqual(stress.SETTINGS[0]["setting_id"],"base")
        self.assertEqual([s["cost_rate"] for s in stress.SETTINGS[:5]],[.0025,0.,.00125,.005,.01])
        self.assertEqual(stress.SETTINGS[5]["support_mode"],"pooled")
        self.assertEqual(stress.SETTINGS[6]["actual_inflation_shift"],.02)

    def test_exact_identity_includes_nan_dtype_shape_values_and_keys(self):
        expected=dict(a=np.array([1.,np.nan]),b=np.array([True,False]),c=np.array(["hold"]))
        self.assertTrue(stress.exact_array_identity(expected,copy.deepcopy(expected))["all_exact"])
        for changed in (dict(expected,a=np.array([1.,np.nan],dtype=np.float32)),
                        dict(expected,a=np.array([[1.,np.nan]])),dict(expected,a=np.array([1.0000000000001,np.nan])),
                        dict(expected,a=np.array([1.,0.])),dict(a=expected["a"])):
            with self.assertRaises(AssertionError):stress.exact_array_identity(changed,expected)

    def test_all_context_bases_must_pass_before_any_stress(self):
        events=[]
        def evaluate(c,s):events.append(("evaluate",c,s["setting_id"]));return c
        def check(c,r):events.append(("check",c))
        def consume(c,s,r):events.append(("consume",c,s["setting_id"]))
        stress.evaluate_two_phase([0,1],evaluate,check,consume,lambda:events.append(("gate",)))
        gate=events.index(("gate",))
        self.assertEqual(sum(e[0]=="evaluate" and e[2]=="base" for e in events[:gate]),2)
        self.assertFalse(any(e[0]=="evaluate" and e[2]!="base" for e in events[:gate]))
        self.assertEqual(sum(e[0]=="evaluate" for e in events),14)

    def test_failed_base_never_reads_or_runs_nonbase_setting(self):
        calls=[]
        def evaluate(c,s):calls.append((c,s["setting_id"]));return None
        def check(c,r):
            if c==1:raise AssertionError("constructed identity failure")
        with self.assertRaises(AssertionError):
            stress.evaluate_two_phase([0,1,2],evaluate,check,lambda *x:None,lambda:self.fail("must not pass"))
        self.assertEqual(calls,[(0,"base"),(1,"base")])

    def test_uniform_thinning_keeps_all_selected_and_physical_arrays(self):
        a=np.array([1.])
        arm=dict(holdings=a,selected_score=a,mitigation_selected_q=a,primary_failed_hold_posttrade=a,
            candidate_score=a,mitigation_candidate_member_advantage=a,training_real_gross=a,
            training_previous_state=a,realized_real_gross=a,actual_previous_state=a)
        kept,omitted=stress.thin_arm(arm)
        self.assertEqual(set(omitted),{"candidate_score","mitigation_candidate_member_advantage","training_real_gross","training_previous_state"})
        self.assertEqual(set(kept),set(arm)-set(omitted))
        self.assertEqual(stress.thin_arm(None),({},[]))

    def test_contexts_bind_own_selected_tuple_target_and_oof_ids(self):
        jobs,fits,selections,groups=metadata_fixture()
        contexts=stress.context_definitions(jobs,fits,selections,groups)
        self.assertEqual(len(contexts),6);self.assertEqual(contexts[0]["cohort"],"full_resubstitution")
        self.assertTrue(all(c["candidate"]==stress.canonical.CANDIDATES[0] for c in contexts))
        bad=copy.deepcopy(selections);bad["1"]["training_ids"][0]=121
        with self.assertRaises(ValueError):stress.context_definitions(jobs,fits,bad,groups)
        bad=copy.deepcopy(groups);bad[1]["target"]=[.3,.3,.4]
        with self.assertRaises(ValueError):stress.context_definitions(jobs,fits,selections,bad)

    def test_future_job_identity_rejected_in_metadata_before_any_loader(self):
        jobs,fits,selections,groups=metadata_fixture();jobs[0]["training_ids"][0]=121
        with self.assertRaises(RuntimeError):stress.context_definitions(jobs,fits,selections,groups)

    def test_actual_inflation_stress_uses_unchanged_training_and_known_fee(self):
        r=np.zeros((120,30,3));pi=np.full((120,30),.03);saved=pi.copy();calls=[]
        context=dict(job=dict(training_ids=[1,2,3],evaluation_ids=[4,5]),target=np.array([.4,.3,.3]),
            candidate=stress.canonical.CANDIDATES[0],null_reason="unused")
        def model(actual,actual_pi,train,train_pi,target,**kw):
            np.testing.assert_array_equal(actual_pi,np.full((2,30),.05))
            np.testing.assert_array_equal(train_pi,np.full((3,30),.03))
            calls.append(kw);return dict(arm=None,status="preinvestment_review",financial_available=False)
        def paired(actual,actual_pi,target,loss,**kw):
            self.assertEqual(kw["cost_rate"],.0025)
            arm=dict(cpi=np.ones((2,31)),real_wealth=np.full((2,1,31),100.))
            return dict(annual=arm,pending_hold=arm)
        result=stress.evaluate_context(context,stress.SETTINGS[-1],r,pi,simulate_model=model,paired_simulator=paired)
        self.assertFalse(result["state_budget_learner"]["financial_available"])
        self.assertEqual(calls[0]["training_ids"],[1,2,3]);self.assertEqual(calls[0]["actual_ids"],[4,5])
        self.assertEqual(calls[0]["support_mode"],"calendar")
        np.testing.assert_array_equal(pi,saved)

    def test_zero_coverage_comparisons_keep_null_and_all_original_references(self):
        rows=[]
        for setting in stress.SETTINGS:
            for cohort in ("outer_OOF","full_resubstitution"):
                common=dict(setting_id=setting["setting_id"],cohort=cohort)
                rows+=stress.canonical.null_rows([1,2],dict(common,policy="state_budget_learner"),"preinvestment_review")
                for ref in stress.canonical.REFS:
                    for sid in (1,2):
                        rows.append(dict(common,policy=ref,scenario_id=sid,assigned=True,
                            log_growth=.03,cagr=.03,mdd=.1,volatility=.1,nominal_fees=.2,trade_fraction=1.,
                            breach=False,support_review=False,automatic_fraction=1.,protection_trade_fraction=0.,
                            mitigation_trade_fraction=0.,mitigation_nominal_fees=0.))
        result=stress.same_cohort_comparisons(rows)
        self.assertEqual(len(result),42)
        self.assertTrue(all(r["same_funded_model"]["breach_fraction"] is None for r in result))
        self.assertTrue(all(r["reference_all_original"]["financial_n"]==2 for r in result))
        self.assertTrue(all(r["paired_differences"]["log_growth"]["mean"] is None for r in result))

    def test_unfrozen_stress_does_not_read_canonical_arrays_or_raw_market(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"protocol";f=Path(temp)/"freeze";p.write_text('{}')
            f.write_text(json.dumps(dict(execution_ready=False,protocol_sha256=stress.sha(p))))
            with patch.object(stress,"PROTOCOL",p),patch.object(stress,"FREEZE",f),patch.object(stress.np,"load") as loader:
                with self.assertRaises(RuntimeError):stress.freeze_gate()
                loader.assert_not_called()

    def test_real_entry_loader_is_dev_only_and_failure_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);p=base/"protocol";f=base/"freeze";p.write_text('{}');f.write_text('{}')
            (base/"manifest.json").write_text('{}')
            def stop_loader(path,ids,*,allow_final):
                self.assertEqual(list(ids),list(range(1,121)));self.assertFalse(allow_final)
                raise RuntimeError("constructed stop immediately after dev-boundary")
            gate=({},dict(source_sha256={},input_sha256={}),base,{},[])
            with patch.object(stress,"PROTOCOL",p),patch.object(stress,"FREEZE",f),patch.object(stress,"freeze_gate",return_value=gate),patch.object(stress.canonical,"load_paths",side_effect=stop_loader),patch.object(stress.np,"load") as arrays:
                with self.assertRaises(RuntimeError):stress.run(base/"out")
                arrays.assert_not_called()
            self.assertEqual(json.loads((base/"out/manifest.json").read_text())["status"],"failed")
            self.assertTrue((base/"out/failure.txt").exists())

    def test_synthetic_complete_pipeline_56summaries_and_pooled_nulls(self):
        jobs,fits,selections,groups=metadata_fixture()
        by_group={};arrays={}
        def arm(n,model):
            h=30;wealth=np.tile(100.*1.01**np.arange(h+1),(n,1))
            value=dict(real_wealth=wealth,pending_at_decision=np.zeros((n,h),bool),
                gate_reason=np.full((n,h),"support_safe_hold"),guard_pending=np.zeros((n,h+1),bool),
                guard_triggered=np.zeros((n,h),bool),first_crossing=np.full(n,-1,int),
                fees=np.zeros((n,h)),trades=np.zeros((n,h),bool),ever_breached=np.zeros((n,h+1),bool),
                holdings=np.repeat((wealth/3)[:,:,None],3,axis=2),cpi=np.ones((n,h+1)),
                selected_score=np.zeros((n,h)))
            if model:value["candidate_score"]=np.full((n,h,13),np.nan)
            return value
        for group in groups:
            model=group["kind"]!="reference"
            value=arm(len(group["scenario_ids"]),model);by_group[group["group_id"]]=value
            group["array_names"]=list(value)
            for name,array in value.items():arrays[group["group_id"]+"__"+name]=array
        contexts=stress.context_definitions(jobs,fits,selections,groups)
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);p=base/"protocol";f=base/"freeze";p.write_text('{}');f.write_text('{}')
            (base/"manifest.json").write_text('{}');np.savez_compressed(base/"traces.npz",**arrays)
            parent=dict(outputs_sha256={"traces.npz":stress.sha(base/"traces.npz")})
            def evaluate(context,setting,r,pi):
                results={}
                for name in stress.OBJECTS:
                    group=context["selected_group"] if name=="state_budget_learner" else context["reference_groups"][name]
                    refused=name=="state_budget_learner" and setting["support_mode"]=="pooled"
                    results[name]=dict(arm=None if refused else by_group[group["group_id"]],
                        financial_available=not refused,status="preinvestment_review" if refused else "funded",initial_witness=None)
                return results
            gate=({},dict(source_sha256={},input_sha256={}),base,parent,contexts)
            returned=(np.zeros((120,30,3)),np.zeros((120,30)),np.arange(1,121))
            with patch.object(stress,"PROTOCOL",p),patch.object(stress,"FREEZE",f),patch.object(stress,"freeze_gate",return_value=gate),patch.object(stress,"evaluate_context",side_effect=evaluate),patch.object(stress.canonical,"load_paths",return_value=returned),patch("builtins.print"):
                stress.run(base/"out")
            receipt=json.loads((base/"out/manifest.json").read_text())
            self.assertEqual(receipt["status"],"complete");self.assertTrue(receipt["base_reproduction_passed"])
            self.assertEqual(receipt["summary_rows"],56);self.assertEqual(receipt["path_rows"],6720)
            comparisons=json.loads((base/"out/cohort_comparisons.json").read_text())
            pooled=[x for x in comparisons if x["setting_id"]=="pooled_support"]
            self.assertTrue(all(x["model_all_original"]["financial_n"]==0 for x in pooled))
            self.assertTrue(all(x["model_all_original"]["breach_fraction"] is None for x in pooled))
            with np.load(base/"out/traces.npz",allow_pickle=False) as archive:
                self.assertFalse(any("__candidate_score" in name for name in archive.files))


if __name__=="__main__":unittest.main()
