"""Synthetic tests of outcome-free matching and the single common-state gate."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/diagnose_matched_parameter_fan_contrast.py"
SPEC = importlib.util.spec_from_file_location("matched_parameter_fan_contrast", SOURCE)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def item(name, fan):
    return {"pair_id": name, "scenario_id": 112, "target": [.45,.05,.5], "annual_eta": .03,
            "original_H": 30, "loss_tolerance": .4, "cost_rate": .0025,
            "kernel_sha256": {"kernel": "a"*64}, "development_arrays_sha256": "b"*64,
            "training_ids": list(fan)}


def account():
    result = {}
    for field in diag.ENDPOINT_FIELDS:
        result[field] = np.zeros((4,3)) if field in ("holdings","weights") else np.zeros(4, dtype=bool if field=="ever_breached" else float)
    for field in diag.PREVIOUS_PERIOD_FIELDS:
        if field=="posttrade_holdings": result[field]=np.zeros((3,3))
        else: result[field]=np.zeros(3, dtype=bool if field in ("trades","pending_at_decision","target_execution") else float)
    return result


class MatchedFanTests(unittest.TestCase):
    def test_matching_is_order_invariant_and_keeps_every_eligible_pair(self):
        cases=[item("a",[1,2,3]),item("b",[1,2]),item("c",[1])]
        expected=diag.eligible_pairs(cases)
        self.assertEqual(len(expected),3)
        self.assertEqual(diag.eligible_pairs(list(reversed(cases))),expected)

    def test_known_outcome_fields_do_not_affect_selection(self):
        a,b=item("a",[1,2]),item("b",[1])
        expected=diag.eligible_pairs([a,b]);a.update(mdd=.9,breach=True,terminal_wealth=0.)
        b.update(mdd=.01,breach=False,terminal_wealth=999.)
        self.assertEqual(diag.eligible_pairs([a,b]),expected)

    def test_each_fixed_parameter_or_market_difference_prevents_matching(self):
        for key,value in [("scenario_id",35),("target",[.4,.1,.5]),("annual_eta",.1),("original_H",29),
                          ("loss_tolerance",.3),("cost_rate",0.),("kernel_sha256",{"kernel":"z"*64}),
                          ("development_arrays_sha256","z"*64)]:
            a,b=item("a",[1,2]),item("b",[1]);b[key]=value
            self.assertEqual(diag.eligible_pairs([a,b]),[],key)

    def test_identical_fan_is_not_a_fan_contrast(self):
        self.assertEqual(diag.eligible_pairs([item("a",[1,2]),item("b",[1,2])]),[])

    def test_first_disagreement_is_fixed_and_no_disagreement_stays_null(self):
        a=["restore","hold","hold"]
        self.assertEqual(diag.first_action_disagreement(a,["restore","restore","restore"]),1)
        self.assertIsNone(diag.first_action_disagreement(a,a))

    def test_inception_or_pending_disagreement_is_not_an_optional_gate(self):
        a,b=account(),account()
        with self.assertRaises(ValueError):diag.verify_prefix(a,b,0)
        b["ever_breached"][1]=True
        with self.assertRaises((ValueError,AssertionError)):diag.verify_prefix(a,b,1)

    def test_predecision_endpoint_is_included_but_later_state_is_not(self):
        a,b=account(),account();b["holdings"][2,0]=999.
        self.assertEqual(diag.verify_prefix(a,b,1)["status"],"passed")
        b["holdings"][1,0]=1.
        with self.assertRaises((ValueError,AssertionError)):diag.verify_prefix(a,b,1)

    def test_discrete_state_must_match_exactly(self):
        a,b=account(),account();b["trades"][0]=True
        with self.assertRaises(ValueError):diag.verify_prefix(a,b,1)

    def test_self_removal_is_one_member_only_and_preserves_order(self):
        a=item("a",range(1,121));b=item("b",[i for i in range(1,121) if i not in range(97,121)])
        fans=diag.fan_specifications(a,b)
        self.assertEqual([len(x[1]) for x in fans],[120,119,96])
        self.assertEqual(fans[1][1],[i for i in range(1,121) if i!=112])
        self.assertEqual(fans[2][1],b["training_ids"])

    def test_original_fan_must_exclude_actual_id_in_outer(self):
        a=item("a",range(1,121));b=item("b",range(25,121))
        with self.assertRaises(ValueError):diag.fan_specifications(a,b)

    def test_saved_Q_reproduction_rejects_action_or_count_change(self):
        row={"fan_n":2,"qg_hold":.1,"qg_restore":.2,"predicted_score":.1,
             "predicted_growth_advantage_restore":.1,"qbad_hold_count":0,"qbad_restore_count":0,
             "predicted_breach_count_delta_restore":0,"chosen_action":"restore","gate_reason":"finite_score_positive"}
        saved={key:np.array([value]) for key,value in row.items() if key not in ("chosen_action","fan_n")}
        saved.update(forecast_n=np.array([2]),actions=np.array(["restore"]))
        self.assertEqual(diag.validate_saved_Q(row,saved,0)["status"],"passed")
        for key,value in [("qbad_restore_count",1),("chosen_action","hold")]:
            changed=copy.deepcopy(row);changed[key]=value
            with self.assertRaises(ValueError):diag.validate_saved_Q(changed,saved,0)


if __name__ == "__main__":
    unittest.main()
