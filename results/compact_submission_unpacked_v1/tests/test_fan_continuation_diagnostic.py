"""Synthetic-only arithmetic, lineage and inactive-state tests; no raw data."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experimental_execution.fan_continuation_diagnostic import (fixed_choice,
    fan_deletions, continuation_gaps)
from experimental_execution.risk_priced_fan_rollout import simulate_prices

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fan_diag_producer", ROOT / "scripts/diagnose_fan_continuation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def protocol():
    slots = []
    for j, cid in enumerate([0] + [100*r + f for r in range(1, 4) for f in range(1, 5)]):
        tr = list(range(1, 121)) if cid == 0 else list(range(2, 98))
        for year in (2, 16, 30):
            slots.append(dict(slot_id=f"s{len(slots)+1:03d}", folder=runner.PARENTS[0 if cid == 0 else 1],
                group=f"c{cid}__{runner.OLD}", context=f"context{cid}", context_id=cid, repeat_id=cid//100,
                final_guard_admitted=False, scenario_id=1, year=year, training_ids=tr,
                selected_candidate_order=4, selected_beta=.075))
    return dict(execution_ready=True, data=dict(scenario_ids=list(range(1,121)),allow_final=False),
        original_H=30,loss_tolerance=.4,cost_rate=.0025,inventory=slots,expected_slots=39,
        output_directory="results/model_review/fan_continuation_diagnostic_v1")


class FanMathTests(unittest.TestCase):
    def test_N_minus_one_matches_explicit_slice_finite_and_risk_first(self):
        h=np.array([.2,-.3,.8,.1,.7]);r=np.array([.1,.4,.6,.2,.9])
        bh=np.array([1,0,1,0,0],bool);br=np.array([0,1,1,0,0],bool)
        for eta in (0.,.03,None):
            full,rows=fan_deletions(h,r,bh,br,eta)
            for j,row in enumerate(rows):
                keep=np.arange(len(h))!=j
                expected=fixed_choice(h[keep].mean(),r[keep].mean(),int(bh[keep].sum()),int(br[keep].sum()),4,eta)
                self.assertAlmostEqual(row['drop_qg_hold'],h[keep].mean(),places=15)
                self.assertAlmostEqual(row['drop_qg_restore'],r[keep].mean(),places=15)
                self.assertEqual(row['drop_bad_hold'],bh[keep].sum())
                self.assertEqual(row['drop_restore'],expected['restore'])
                self.assertEqual(row['action_flip'],expected['restore']!=full['restore'])

    def test_signed_event_difference_and_paired_two_over_N_bound(self):
        h=np.zeros(5);r=np.zeros(5)
        bh=np.array([1,0,0,0,0],bool);br=~bh
        full,rows=fan_deletions(h,r,bh,br,.1)
        self.assertAlmostEqual(rows[0]['paired_probability_change'],2/5)
        self.assertAlmostEqual(rows[0]['hold_probability_change'],-1/5)
        self.assertAlmostEqual(rows[0]['restore_probability_change'],1/5)
        self.assertEqual(rows[1]['drop_bad_restore']-rows[1]['drop_bad_hold'],2)
        reduced=fixed_choice(.1,0.,4,1,5,.03)
        self.assertLess(reduced['delta_breach_probability'],0)
        self.assertTrue(reduced['restore'])

    def test_risk_first_lexicographic_and_finite_original_H(self):
        self.assertTrue(fixed_choice(20.,-20.,2,1,5,None)['restore'])
        self.assertFalse(fixed_choice(-20.,20.,1,2,5,None)['restore'])
        self.assertTrue(fixed_choice(0.,.1,2,2,5,None)['restore'])
        self.assertFalse(fixed_choice(0.,1e-12,2,2,5,None)['restore'])
        self.assertIsNone(fixed_choice(0.,.1,2,2,5,None)['score'])
        self.assertFalse(fixed_choice(0.,.2,0,1,5,.1,30)['restore'])
        self.assertTrue(fixed_choice(0.,.2,0,1,5,.1,1)['restore'])

    def test_nonboolean_events_and_single_member_rejected(self):
        with self.assertRaises(ValueError):fan_deletions([0,0],[0,0],[0,1],[1,0],.03)
        with self.assertRaises(ValueError):fan_deletions([0],[0],[False],[False],.03)

    def test_last_year_chosen_continuation_control(self):
        value=dict(total_log_growth=.03,ever_breached=False,history_mdd=.2,nominal_fees=.1,terminal_real=120.)
        self.assertTrue(all(v==0 for v in continuation_gaps(value,value,year=30).values()))
        other={**value,'total_log_growth':.031}
        with self.assertRaises(AssertionError):continuation_gaps(other,value,year=30)
        self.assertAlmostEqual(continuation_gaps(other,value,year=16)['continuation_gap_total_log_growth'],.001)


class ProducerSyntheticTests(unittest.TestCase):
    def test_protocol_exact_inventory_and_actual_fan_disjointness(self):
        p=protocol();self.assertEqual(len(runner.validate_protocol(p)),39)
        for change in ('duplicate','self_in_OOF','replace_year','changed_target_lineage','final'):
            q=copy.deepcopy(p)
            if change=='duplicate':q['inventory'][1]['slot_id']=q['inventory'][0]['slot_id']
            if change=='self_in_OOF':q['inventory'][3]['training_ids'][0]=1
            if change=='replace_year':q['inventory'][3]['year']=3
            if change=='changed_target_lineage':q['inventory'][4]['selected_beta']=.1
            if change=='final':q['data']['scenario_ids'][-1]=121
            with self.assertRaises(ValueError):runner.validate_protocol(q)

    def test_unfrozen_main_stops_before_artifact_or_market_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'protocol.json';path.write_text(json.dumps(dict(execution_ready=False)))
            with patch.object(runner,'PROTOCOL',path),patch.object(runner,'validate_archives',side_effect=AssertionError('must not read')):
                with self.assertRaises(RuntimeError):runner.main()

    def test_existing_OUT_stops_before_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'protocol.json';path.write_text(json.dumps(protocol()))
            with patch.object(runner,'PROTOCOL',path),patch.object(runner,'OUT',Path(directory)),patch.object(runner,'snapshot',side_effect=AssertionError('must not read')):
                with self.assertRaises(RuntimeError):runner.main()

    def test_frozen_hash_change_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'p.json').write_text('{}');(base/'source.py').write_text('one')
            digest=runner.sha(base/'p.json')
            p=dict(execution_sources_sha256={'source.py':'0'*64},frozen_input_sha256={'p.json':digest})
            with patch.object(runner,'ROOT',base),patch.object(runner,'PROTOCOL',base/'p.json'):
                with self.assertRaises(RuntimeError):runner.snapshot(p,digest)

    def fixture(self,crash=False):
        returns=np.zeros((120,30,3));inflation=np.zeros((120,30))
        returns[:,:,0]=.05;returns[:,:,1]=.02;returns[:,:,2]=.01
        if crash:returns[:,0,:]=-.6
        target=np.array([.4,.3,.3]);arm=simulate_prices(returns[:1],inflation[:1],returns[1:5],inflation[1:5],target,[.03],L=.4,horizon=30)
        # Frozen selected-arm archives remove the singleton price axis.
        squeezed={k:(v[:,0] if v.ndim>=2 and v.shape[:2]==(1,1) else v) for k,v in arm.items()}
        s=dict(slot_id='synthetic',folder=runner.PARENTS[1],group='c101__'+runner.OLD,context='synthetic101',
            context_id=101,repeat_id=1,final_guard_admitted=False,scenario_id=1,year=30,
            training_ids=[2,3,4,5],selected_candidate_order=4,selected_beta=.075)
        z={s['group']+'__'+k:v for k,v in squeezed.items()}
        archive=dict(groups={s['group']:dict(scenario_ids=[1])},traces=z)
        return s,archive,returns,inflation

    def test_active_recomputed_Q_and_last_year_same_action(self):
        s,a,r,inf=self.fixture();row,members,drops=runner.diagnose_slot(s,a,r,inf)
        self.assertEqual(row['state_status'],'active');self.assertEqual(len(members),4);self.assertEqual(len(drops),4)
        self.assertEqual(row['future_decisions'],0)
        self.assertAlmostEqual(row['continuation_gap_total_log_growth'],0.,places=13)
        self.assertEqual(row['continuation_gap_event'],0)

    def test_pending_no_replacement_new_diagnostics_NULL(self):
        s,a,r,inf=self.fixture(crash=True);s['year']=16
        row,members,drops=runner.diagnose_slot(s,a,r,inf)
        self.assertEqual(row['state_status'],'pending_skip');self.assertEqual(members,[]);self.assertEqual(drops,[])
        for key in ('recomputed_qg_hold','recomputed_score','deletion_n','self_deleted_action_flip'):
            self.assertIsNone(row[key])
        self.assertNotIn('continuation_gap_event',row)


if __name__=='__main__':unittest.main()
