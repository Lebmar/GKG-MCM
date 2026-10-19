"""Independent liquidity scalar checks on constructed states/files only."""
import ast
import csv
import gzip
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import audit_liquidity_capacity as a


class LiquidityAuditTests(unittest.TestCase):
    def test_scalar_share_and_implicit_fee_formula(self):
        z=a.scalar_annual([.4,.2,.4],[.5,0.,-.1],.02,floor=.3)
        bill=.36/(.6+.2+.36);drag=.0025/(1.0025)*(1-bill)
        self.assertAlmostEqual(z['end_bill_share'],bill,places=15);self.assertAlmostEqual(z['hypothetical_liquidation_drag'],drag,places=15)
        self.assertAlmostEqual(z['hypothetical_available_bill_ratio']+drag,1,places=15)
        transferred=(1-bill)/1.0025;self.assertAlmostEqual(drag,.0025*transferred,places=15)
    def test_all_bill_zero_drag(self):
        z=a.scalar_annual([0,0,1],[1,-.5,-.1],0,floor=.3)
        self.assertEqual(z['end_bill_share'],1);self.assertEqual(z['hypothetical_liquidation_drag'],0);self.assertEqual(z['hypothetical_available_bill_ratio'],1)
    def test_common_postfee_scale_cancels(self):
        w=[.4,.2,.4];r=[.5,0.,-.1];z=a.scalar_annual(w,r,0,floor=.3)
        for scale in (.01,100.,1e6):
            x=[scale*w[i]*(1+r[i]) for i in range(3)]
            self.assertAlmostEqual(x[2]/math.fsum(x),z['end_bill_share'],places=14)
    def test_bill_real_loss_separate_from_liquidation(self):
        z=a.scalar_annual([0,0,1],[0,0,.01],.02,floor=.3)
        self.assertLess(z['bill_real_return'],0);self.assertEqual(z['hypothetical_liquidation_drag'],0)
        self.assertEqual(a.scalar_path([z])['mean_bill_real_loss_fraction'],1)
    def test_target_floor_drift_not_target_violation(self):
        z=a.scalar_annual([.7,0,.3],[1,0,0],0,floor=.3)
        self.assertTrue(z['end_below_target_floor']);self.assertEqual(z['target_bill_share'],.3)
        with self.assertRaises(ValueError):a.scalar_annual([.8,0,.2],[0,0,0],0,floor=.3)
    def test_invalid_gross_weights_and_cost(self):
        for kwargs in (dict(weights=[.4,.4,.4],returns=[0,0,0],inflation=0,floor=0),dict(weights=[0,0,1],returns=[0,0,-1],inflation=0,floor=0),dict(weights=[0,0,1],returns=[0,0,0],inflation=-1,floor=0),dict(weights=[0,0,1],returns=[0,0,0],inflation=0,floor=0,cost=1)):
            with self.assertRaises(ValueError):a.scalar_annual(**kwargs)
    def test_own_horizon_path_aggregation(self):
        z=[a.scalar_annual([.5,0,.5],[1,0,0],0,floor=.3),a.scalar_annual([.5,0,.5],[0,0,0],.1,floor=.3)]
        p=a.scalar_path(z);self.assertAlmostEqual(p['mean_end_bill_share'],(1/3+.5)/2,places=15)
        self.assertEqual(p['mean_bill_real_loss_fraction'],.5)
        with self.assertRaises(ValueError):a.scalar_path([])
    def test_empty_funded_summary_all_NULL(self):
        z=a.scalar_summary(a.PROFILES[0],[dict(assigned=False) for _ in range(120)])
        self.assertEqual(z['financial_n'],0);self.assertEqual(z['review_n'],120)
        self.assertTrue(all(z[k] is None for k in a.SUMMARY_FIELDS))
    def test_csv_NULL_and_booleans_are_strict(self):
        self.assertIsNone(a.decoded('',None));self.assertFalse(a.decoded('False',True))
        for value,expected in (('0',None),('',True),('false',True),('2.5',1)):
            with self.assertRaises(AssertionError):a.decoded(value,expected)
    def test_row_identity_and_duplicate_rejected(self):
        c=a.Checks();expected=[dict(job_id='j',scenario_id=1,profile='C',value=0.)]
        actual=[dict(job_id='j',scenario_id='1',profile='C',value='0')]
        a.compare_rows(expected,actual,('job_id','scenario_id','profile'),tuple(expected[0]),c,'test')
        with self.assertRaises(AssertionError):a.compare_rows(expected,actual*2,('job_id','scenario_id','profile'),tuple(expected[0]),c,'duplicate')
    def test_no_production_diagnostic_import(self):
        tree=ast.parse(Path(a.__file__).read_text());modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('diagnostics' in m or 'safeinvest.core' in m for m in modules))
    def test_failed_gate_receipt_and_refuse_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(a,'authorize',side_effect=RuntimeError('synthetic unready freeze')):
                with self.assertRaises(RuntimeError):a.run(root)
            state=a.readj(root/a.ATTEMPT/'attempt.json');self.assertEqual(state['status'],'failed');self.assertFalse(state['market_value_access_started'])
            self.assertTrue((root/a.ATTEMPT/'failure.txt').is_file())
            with self.assertRaises(RuntimeError):a.run(root)

    def synthetic_integration(self,fund_C):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/a.OUT).mkdir(parents=True);jobs=[];fits={};prior=[];paths=[];annual=[]
            for k in range(5):
                ev=list(range(k*24+1,k*24+25));tr=[i for i in range(1,121) if i not in ev];jid=f'outer{k+1}'
                jobs.append(dict(job_id=jid,stage='refit',parent_context_id=k+1,training_ids=tr,evaluation_ids=ev))
                assignments={p[0]:('P' if fund_C and p[0]=='C' else 'Review') for p in a.PROFILES}
                fits[jid]=dict(refit=dict(training_ids=tr,outer_test_ids=ev,effective_assignments=assignments,raw_fit=dict(menu=[dict(name='P',weights=[0,0,1])])))
                for name,h,liq,floor in a.PROFILES:
                    funded=assignments[name]=='P'
                    for sid in ev:
                        prior.append(dict(job_id=jid,scenario_id=sid,profile=name,effective_assignment=assignments[name],assigned=funded,horizon=h))
                        common=dict(job_id=jid,scenario_id=sid,profile=name,horizon=h,liquidity=liq,target_floor=floor,effective_assignment=assignments[name],assigned=funded)
                        values=dict(target_bill_share=1.,mean_end_bill_share=1.,minimum_end_bill_share=1.,fraction_end_below_target_floor=0.,max_hypothetical_liquidation_drag=0.,mean_bill_real_loss_fraction=0.) if funded else dict.fromkeys(a.FIELDS)
                        paths.append(dict(common,**values))
                        if funded:
                            for t in range(1,h+1):annual.append(dict(common,period=t,target_bill_share=1.,end_bill_share=1.,end_below_target_floor=False,hypothetical_liquidation_drag=0.,hypothetical_available_bill_ratio=1.,bill_real_return=0.))
            a.writej(root/'jobs.json',dict(jobs=jobs));a.writej(root/'fits.json',fits)
            def write_csv(path,rows,columns):
                with gzip.open(path,'wt',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
            write_csv(root/'prior.csv.gz',prior,tuple(prior[0]));write_csv(root/a.OUT/'path_capacity.csv.gz',paths,a.PATH_COLUMNS);write_csv(root/a.OUT/'annual_capacity.csv.gz',annual,a.ANNUAL_COLUMNS)
            summaries=[]
            for p in a.PROFILES:
                n=120 if fund_C and p[0]=='C' else 0
                s=dict(profile=p[0],horizon=p[1],liquidity=p[2],target_floor=p[3],original_n=120,financial_n=n,review_n=120-n,**dict.fromkeys(a.SUMMARY_FIELDS))
                if n:s.update(minimum_target_bill_share=1.,maximum_target_bill_share=1.,mean_end_bill_share=1.,minimum_end_bill_share=1.,mean_fraction_end_below_target_floor=0.,paths_with_any_end_below_target_floor=0,maximum_hypothetical_liquidation_drag=0.,mean_bill_real_loss_fraction=0.)
                summaries.append(s)
            a.writej(root/a.OUT/'summary.json',summaries)
            c=a.Checks()
            with patch('safeinvest.data.load_paths',return_value=(np.zeros((120,30,3)),np.zeros((120,30)),np.arange(1,121))):
                counts=a.audit_values(root,dict(jobs='jobs.json',static_fits='fits.json',q1_oof_paths='prior.csv.gz'),dict(annual_rows=len(annual)),c)
            self.assertEqual(counts['path_rows'],600);self.assertEqual(counts['annual_rows'],3600 if fund_C else 0)
            self.assertEqual(counts['assigned_paths'],120 if fund_C else 0)
    def test_all_Review_600_NULL_empty_annual_header(self):self.synthetic_integration(False)
    def test_C_only_120_funded_3600_own_H_endpoints(self):self.synthetic_integration(True)


if __name__=='__main__':unittest.main()
