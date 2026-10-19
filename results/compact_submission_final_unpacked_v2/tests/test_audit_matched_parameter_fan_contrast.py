"""Independent scalar score/metadata/identity rules; no artifact I/O."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_matched_parameter_fan_contrast as a


def case(name,ids):
    return dict(pair_id=name,scenario_id=112,target=[.45,.05,.5],annual_eta=.03,original_H=30,loss_tolerance=.4,cost_rate=.0025,
        kernel_sha256={'same':'kernel'},development_arrays_sha256='same_arrays',training_ids=ids)

class IndependentMatchedFanPureTests(unittest.TestCase):
    def test_metadata_matching_keeps_every_eligible_pair(self):
        x=case('a',[1,2]);y=case('b',[1,3]);z=case('c',[1,4])
        self.assertEqual(len(a.metadata_pairs([z,x,y])),3)
    def test_no_match_when_target_or_actual_market_differs(self):
        x=case('a',[1,2]);y=case('b',[1,3]);y['target']=[.4,.2,.4]
        self.assertEqual(a.metadata_pairs([x,y]),[])
        y=case('b',[1,3]);y['scenario_id']=98
        self.assertEqual(a.metadata_pairs([x,y]),[])
    def test_same_fan_is_not_a_fan_contrast(self):
        self.assertEqual(a.metadata_pairs([case('a',[1,2]),case('b',[1,2])]),[])
    def test_one_self_deletion_removes_only_original_actual_ID(self):
        left=case('full',list(range(1,121)));right=case('outer',list(range(1,97)))
        fans=a.fixed_fans(left,right)
        self.assertEqual([len(x[1]) for x in fans],[120,119,96])
        self.assertEqual(set(fans[0][1])-set(fans[1][1]),{112})
    def test_outerfan_containing_actual_ID_rejected(self):
        left=case('full',list(range(1,121)));right=case('outer',[112]+list(range(2,97)))
        with self.assertRaisesRegex(ValueError,'bridge'):a.fixed_fans(left,right)
    def test_net_score_original_H_and_tie_hold(self):
        result=a.fixed_gate(.1,.11,5,6,100,.03)
        self.assertAlmostEqual(result['predicted_score'],.001)
        self.assertEqual(result['chosen_action'],'restore')
        self.assertEqual(a.fixed_gate(.1,.1,5,5,100,.03)['chosen_action'],'hold')
    def test_fractional_or_boolean_badcounts_rejected(self):
        with self.assertRaises(ValueError):a.fixed_gate(.1,.2,5.5,4,100,.03)
        with self.assertRaises(ValueError):a.fixed_gate(.1,.2,False,4,100,.03)
    def test_NULL_not_zero(self):
        with self.assertRaisesRegex(AssertionError,'NULL'):a.Checks().value(0,None,'missing')

if __name__=='__main__':unittest.main()
