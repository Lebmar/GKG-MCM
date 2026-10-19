"""Synthetic intervention semantics; never load markets or saved outcomes."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_failure_action_intervention as a

class InterventionAuditPureTests(unittest.TestCase):
    def test_first_disagreement_is_not_chosen_later_for_good_outcome(self):
        m=['restore']+['hold']*29;b=['restore']*30
        self.assertEqual(a.original_tau(m,b),2)
    def test_no_disagreement_stays_null_and_keeps_six_nominal_slots(self):
        m=['restore']*30
        self.assertIsNone(a.original_tau(m,m))
        slots=a.nominal_slots(None)
        self.assertEqual(len(slots),6)
        self.assertTrue(all(x['status']=='no_intervention' for x in slots))
    def test_inception_is_not_an_optional_action_intervention(self):
        with self.assertRaisesRegex(ValueError,'initial restore'):
            a.original_tau(['hold']*30,['restore']*30)
    def test_original_H_alias_is_retained_at_tau29(self):
        slots=a.nominal_slots(29)
        self.assertEqual(len(slots),6)
        aliases=[r for r in slots if r['status']=='alias']
        self.assertEqual(len(aliases),2)
        self.assertTrue(all(r['role']=='original_H' and r['alias_of']=='tau_plus_one' for r in aliases))
    def test_tau30_preserves_beyond_H_slot_and_alias(self):
        slots=a.nominal_slots(30)
        self.assertEqual(sum(x['status']=='beyond_original_H' for x in slots),2)
        self.assertEqual(sum(x['status']=='alias' for x in slots),2)
    def test_score_is_original_H_and_net_total_log(self):
        z=a.gate(.1,.11,5,6,100,.03)
        self.assertAlmostEqual(z['score'],.001)
        self.assertEqual(z['action'],'restore')
    def test_pending_cannot_resume_despite_real_wealth_recovery(self):
        returns=np.zeros((30,3));returns[0,0]=-.6;returns[1,0]=2
        actions=['restore']+['hold']*28+['restore']
        with self.assertRaisesRegex(AssertionError,'pending'):a.replay(returns,np.zeros(30),[1,0,0],actions)
    def test_original_whole_history_survives_recovery(self):
        returns=np.zeros((30,3));returns[0,0]=-.6;returns[1,0]=2
        r=a.replay(returns,np.zeros(30),[1,0,0],['restore']+['hold']*29)
        self.assertEqual(r['first_crossing'],1)
        self.assertTrue(r['pending_at_decision'][-1])
        self.assertGreater(r['real_wealth'][-1],100)
        self.assertGreater(r['running_mdd'][-1],.6)
    def test_unavailable_endpoint_is_not_zero_filled(self):
        with self.assertRaisesRegex(AssertionError,'NULL'):a.Checks().scalar(0,None,'no crossing')
    def test_negative_or_boolean_event_counts_cannot_be_scores(self):
        with self.assertRaises(ValueError):a.gate(.1,.11,False,3,100,.03)
        with self.assertRaises(ValueError):a.gate(.1,.11,-1,3,100,.03)

    def test_fixed_Q_slot_reuses_identical_state_but_retains_slot_record(self):
        market=np.zeros((30,3));infl=np.zeros(30);target=[0,0,1]
        ref=a.replay(market,infl,target,['restore']+['hold']*29)
        source={field:np.zeros(30) for field in a.QFLOAT}
        source.update({field:np.zeros(30,dtype=int) for field in a.QCOUNT})
        source['forecast_n'][:]=2;source['policy_actions']=np.array(['restore']+['hold']*29)
        cache={};checks=a.Checks();train=np.zeros((2,30,3));pi=np.zeros((2,30))
        first=a.independent_Q_slot(ref,source,2,train,pi,target,.03,checks,'control',cache)
        second=a.independent_Q_slot(ref,source,2,train,pi,target,.03,checks,'treatment',cache)
        self.assertEqual(len(cache),1)
        self.assertEqual(first['status'],'recomputed');self.assertFalse(first['reused_identical_Q_state'])
        self.assertEqual(second['status'],'recomputed');self.assertTrue(second['reused_identical_Q_state'])
    def test_pending_fixed_Q_slot_is_not_replaced_by_an_active_year(self):
        market=np.zeros((30,3));market[0,0]=-.6
        ref=a.replay(market,np.zeros(30),[1,0,0],['restore']+['hold']*29)
        source={field:np.full(30,np.nan) for field in a.QFLOAT}
        source.update({field:np.full(30,-1,dtype=int) for field in a.QCOUNT})
        checks=a.Checks();cache={}
        row=a.independent_Q_slot(ref,source,30,np.zeros((2,30,3)),np.zeros((2,30)),[1,0,0],.03,checks,'fixedterminal',cache)
        self.assertEqual(row['status'],'pending_unavailable');self.assertEqual(row['year'],30)
        self.assertEqual(cache,{})

if __name__=='__main__':unittest.main()
