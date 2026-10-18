"""Synthetic identity, denominator and deterministic-compression checks."""
import gzip,json,sys,tempfile,unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'research')]
import run_refit_stability as r

class DesignTests(unittest.TestCase):
    def test_every_repeat_has_complete_outer_partition_and_nested_train_only_static_folds(self):
        contexts,jobs,static=r.build_design();self.assertEqual(len(jobs),125);self.assertEqual(len(contexts),25)
        for repeat in range(1,6):
            cs=[c for c in contexts if c['repeat_id']==repeat]
            self.assertEqual(sorted(i for c in cs for i in c['outer_test_ids']),list(range(1,121)))
            for c in cs:
                group=[j for j in jobs if j['parent_context_id']==c['context_id']]
                self.assertEqual([j['stage'] for j in group],['inner']*4+['refit'])
                final=static[group[-1]['job_id']]
                self.assertEqual(final['inner_seed'],c['inner_seed']);self.assertEqual(final['inner_folds'],c['inner_folds'])
                for j in group:
                    for f in static[j['job_id']]['inner_folds']:
                        self.assertTrue(set(f['train_ids'])<=set(j['training_ids']));self.assertFalse(set(f['train_ids'])&set(f['test_ids']))
    def test_protocol_rejects_changed_folds_or_economic_rules(self):
        p=json.loads((ROOT/'docs/refit_stability_protocol.json').read_text());r.validate_protocol(p)
        for key,value in [('loss_tolerance',.5),('target_updates_enabled',True),('liquidity_floors',dict(high=.5,medium=.2,low=0))]:
            wrong=deepcopy(p);wrong[key]=value
            with self.assertRaises(ValueError):r.validate_protocol(wrong)
        wrong=deepcopy(p);wrong['jobs'][0]['evaluation_ids'][0]=121
        with self.assertRaises(ValueError):r.validate_protocol(wrong)
    def test_gzip_repeats_exact_bytes_and_preserves_null(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.gz';b=Path(t)/'b.gz';rows=[dict(x=1,y=None),dict(x=2,y=.3)]
            r.write_csv_gz(a,rows);r.write_csv_gz(b,rows);self.assertEqual(a.read_bytes(),b.read_bytes())
            data=pd.read_csv(a,compression='gzip');self.assertTrue(pd.isna(data.y.iloc[0]))
            r.write_json_gz(a,rows);self.assertEqual(json.loads(gzip.decompress(a.read_bytes())),rows)
    def test_per_scenario_repeated_fits_are_not_extra_samples_and_zeroauto_is_null(self):
        rows=[]
        for repeat in range(1,6):
            for policy in r.LEARNERS:
                rows.append(dict(policy=policy,scenario_id=1,repeat_id=repeat,assigned=repeat<3,breach=False if repeat<3 else None,log_growth=.03 if repeat<3 else None,mdd=.2 if repeat<3 else None))
                rows.append(dict(policy=policy,scenario_id=2,repeat_id=repeat,assigned=False,breach=None,log_growth=None,mdd=None))
        stats=r.scenario_stability(rows);self.assertEqual(len(stats),4)
        for s in stats:
            self.assertEqual(s['repeat_records_n'],5)
            if s['scenario_id']==2:self.assertIsNone(s['breached_among_admitted_refits_n']);self.assertIsNone(s['min_mdd'])
            else:self.assertEqual(s['admitted_refits_n'],2)
        with self.assertRaises(ValueError):r.scenario_stability(rows[:-1])

if __name__=='__main__':unittest.main()
