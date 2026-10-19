"""Pure scalar algebra and metadata distractions only; no saved numerics."""
from pathlib import Path
import sys,tempfile,unittest
from unittest import mock
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_support_failure as a


class SupportAuditTests(unittest.TestCase):
    def test_outer_job_positive_identity_excludes_inner_and_full_distractors(self):
        groups=[]
        for j in range(1,6):
            groups.append(dict(kind='outer_selected',job_id='own'+str(j),scenario_ids=[j]))
            groups.append(dict(kind='reference',policy='exact_thirds_annual',job_id='own'+str(j),scenario_ids=[j]))
        for job in ('c0_refit','inner_1','inner_2'):
            groups.append(dict(kind='reference',policy='exact_thirds_annual',job_id=job,scenario_ids=[99]))
        groups.append(dict(kind='full_selected',job_id='c0_refit',scenario_ids=list(range(1,121))))
        self.assertEqual({g['job_id'] for g in a.outer_reference_groups(groups)},{'own'+str(j) for j in range(1,6)})
        groups[1]['scenario_ids']=[8]
        with self.assertRaises(AssertionError):a.outer_reference_groups(groups)

    def test_full_history_peak_and_strict_event_tolerance(self):
        value=a.metrics_one([100,150,100,120]);self.assertEqual(value['peak'],[100,150,150,150])
        self.assertAlmostEqual(value['mdd'],1/3)
        self.assertFalse(a.metrics_one([100,100*(.6-5e-13)])['breach'])
        self.assertTrue(a.metrics_one([100,100*(.6-2e-12)])['breach'])
        self.assertEqual(a.metrics_one([100,80,50])['first_crossing'],2)

    def test_four_corners_revalue_original_peaks_and_both_ordered_decompositions(self):
        data,nominal,cpi,parts=a.corner_values([100,150,90],[1,1,1],[100,120,100],[1,1.1,1.3])
        self.assertAlmostEqual(parts['MDD_CPI_first']+parts['MDD_nominal_after_CPI'],parts['MDD_total'])
        self.assertAlmostEqual(parts['MDD_nominal_first']+parts['MDD_CPI_after_nominal'],parts['MDD_total'])
        for t in range(3):self.assertAlmostEqual(np.log(data['SS']['real'][t]/data['BB']['real'][t]),nominal[t]+cpi[t])
        self.assertNotEqual(data['BS']['peak'],data['BB']['peak'])
        with self.assertRaises(ValueError):a.corner_values([100,150],[1,1],[100,120],[2,2])

    def test_initial_joint_support_fee_and_pooled_inclusion(self):
        train=np.array([[[.9,1.1,1.],[.4,.4,.9]],[[1.1,.9,1.],[1.4,1.4,1.1]]])
        y,fee,b,values=a.initial_values([.45,.15,.4],.0025,.5,train)
        self.assertAlmostEqual(y.sum()+fee,100.);self.assertEqual(b,80.)
        self.assertGreater(values[:,0].min(),b);self.assertLess(values.min(),b)
        self.assertEqual(int(np.sum(train[:,:,2]<1)),1)
        independent_min=np.min(train.reshape(-1,3),axis=0)@y
        self.assertLessEqual(independent_min,values.min())

    def test_directional_minimum_uses_joint_rows_and_cpi(self):
        support=np.array([[.5,1.5,1.],[1.5,.5,1.]])
        minimum,values=a.minima(support,[30,30,40],2.)
        np.testing.assert_array_equal(values,[50,50]);self.assertEqual(minimum,50.)
        self.assertGreater(minimum,np.min(support,axis=0)@np.array([30,30,40])/2)

    def test_canonical_metadata_can_omit_stress_financial_flag_without_inventing_arm(self):
        g=dict(group_id='synthetic',array_names=['real_wealth'])
        archive={'synthetic__real_wealth':np.array([[100.,101.]])}
        np.testing.assert_array_equal(a.load_arm(archive,g,['real_wealth'])['real_wealth'],[[100.,101.]])
        g['financial_available']=False
        with self.assertRaises(AssertionError):a.load_arm(archive,g,['real_wealth'])

    def test_null_zero_boolean_and_extra_population_rows_fail(self):
        check=a.Checks();check.same(None,None);check.same(True,True)
        for actual,expected in [(0,None),(None,0),(1,True)]:
            with self.assertRaises(AssertionError):check.same(actual,expected)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'synthetic.csv';pd.DataFrame([dict(id=1,value=2),dict(id=2,value=3)]).to_csv(path,index=False)
            table=a.Table(path,('id',));table.check(check,dict(id=1,value=2))
            with self.assertRaises(AssertionError):table.complete()
            table.check(check,dict(id=2,value=3));table.complete()

    def test_missing_activation_cannot_parse_any_npz(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'docs').mkdir()
            a.writej(root/a.PLAN,dict(version='independent-support-failure-audit-v1',execution_ready=False))
            a.writej(root/a.FREEZE,dict(source_sha256={},input_sha256={}))
            with mock.patch.object(a.np,'load') as npz:
                with self.assertRaises(FileNotFoundError):a.run(root)
            npz.assert_not_called();self.assertFalse((root/a.ATTEMPT).exists())

    def test_auditor_never_imports_producer_policy_or_raw_loader(self):
        import ast
        tree=ast.parse(Path(a.__file__).read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any('support_failure' in name or 'safeinvest' in name or 'experimental_execution' in name for name in names))


if __name__=='__main__':unittest.main()
